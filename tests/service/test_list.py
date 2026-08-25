"""GET /v1/conversations — the history panel's surface: the principal's own
conversations only (ownership is the filter; foreign ids never appear),
newest first, titled by the first user message."""

import httpx
import pytest

from turnstile.capabilities.persistence.memory_store import MemorySessionStore
from turnstile.kernel.dtos import Done, TextDelta
from turnstile.kernel.engine import Agent
from turnstile.kernel.testkit import ScriptedProvider
from turnstile.products.middleware.references import ReferenceCollector
from turnstile.root import AssembledAgent
from turnstile.service.app import create_app
from turnstile.service.auth import mint_token
from turnstile.service.registry import ConversationRegistry

pytestmark = pytest.mark.service

SECRET = "test-secret-0123456789abcdef-0123456789"


def _app(jwt_secret: str | None = SECRET):
    from turnstile.config import Config

    cfg = Config(
        _env_file=None,  # type: ignore[call-arg]
        jwt_secret=jwt_secret,
        llm={"base_url": "https://ds4.example/v1", "model": "model-fast"},
        kb={
            "embedding_url": "https://e/x",
            "milvus_url": "https://m/x",
            "collection": "c",
            "expr": "e",
        },
    )
    store = MemorySessionStore()

    def scripted_assemble(cfg, session_id, store):
        agent = Agent(
            provider=ScriptedProvider(rounds=[[TextDelta("answer"), Done()]] * 9),
            hooks=[store.hook(session_id)],
            session_id=session_id,
            resume=store.load(session_id),
        )
        return AssembledAgent(agent=agent, references=ReferenceCollector(), store=store)

    app = create_app(cfg)
    app.state.store = store
    app.state.registry = ConversationRegistry(cfg, store, assemble=scripted_assemble)
    return app


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def _bearer(user: str) -> dict:
    return {"Authorization": f"Bearer {mint_token(SECRET, user)}"}


async def _ask(client, cid: str, text: str, headers: dict | None = None) -> None:
    async with client.stream(
        "POST", f"/v1/conversations/{cid}/messages", json={"text": text}, headers=headers or {}
    ) as response:
        await response.aread()
        assert response.status_code == 200


async def test_empty_for_a_principal_with_no_conversations():
    app = _app()
    async with _client(app) as client:
        response = await client.get("/v1/conversations", headers=_bearer("alice"))
    assert response.status_code == 200
    assert response.json() == {"conversations": []}


async def test_lists_own_conversations_newest_first_with_titles():
    app = _app()
    async with _client(app) as client:
        await _ask(client, "c1", "what is my leave benefits", _bearer("alice"))
        await _ask(client, "c1", "and holidays?", _bearer("alice"))  # second turn, same title
        await _ask(client, "c2", "x" * 200, _bearer("alice"))  # long first message
        listing = (await client.get("/v1/conversations", headers=_bearer("alice"))).json()
    assert listing == {
        "conversations": [
            {"conversation_id": "c2", "title": "x" * 60, "turn_counter": 1, "in_flight": False},
            {
                "conversation_id": "c1",
                "title": "what is my leave benefits",
                "turn_counter": 2,
                "in_flight": False,
            },
        ]
    }
    await app.state.registry.shutdown_all()


async def test_foreign_conversations_are_simply_absent():
    app = _app()
    async with _client(app) as client:
        await _ask(client, "c1", "alice's question", _bearer("alice"))
        await _ask(client, "c2", "bob's question", _bearer("bob"))
        alice = (await client.get("/v1/conversations", headers=_bearer("alice"))).json()
        bob = (await client.get("/v1/conversations", headers=_bearer("bob"))).json()
    assert [c["conversation_id"] for c in alice["conversations"]] == ["c1"]
    assert [c["conversation_id"] for c in bob["conversations"]] == ["c2"]
    await app.state.registry.shutdown_all()


async def test_listing_survives_eviction_via_the_store():
    # the live agent is gone but ownership + snapshot persist -> still listed
    app = _app()
    async with _client(app) as client:
        await _ask(client, "c1", "q", _bearer("alice"))
        await app.state.registry.shutdown_all()
        listing = (await client.get("/v1/conversations", headers=_bearer("alice"))).json()
    assert [c["conversation_id"] for c in listing["conversations"]] == ["c1"]


async def test_auth_applies_and_dev_mode_lists_anonymous():
    async with _client(_app()) as client:
        assert (await client.get("/v1/conversations")).status_code == 401
    app = _app(jwt_secret=None)
    async with _client(app) as client:
        await _ask(client, "c1", "dev question")
        listing = (await client.get("/v1/conversations")).json()
    assert listing["conversations"][0]["title"] == "dev question"
    await app.state.registry.shutdown_all()
