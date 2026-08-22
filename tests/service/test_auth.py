"""AuthN + conversation ownership — 401 paths, the anonymous dev mode, and
foreign conversations answering 404 (never 403: existence must not leak)."""

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

SECRET = "test-secret-0123456789abcdef-0123456789"  # >=32 bytes: HS256 floor


def _app(jwt_secret: str | None):
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


def _bearer(user: str, secret: str = SECRET, ttl: int = 3600) -> dict:
    return {"Authorization": f"Bearer {mint_token(secret, user, ttl)}"}


async def _post(client, cid: str, headers: dict | None = None) -> int:
    async with client.stream(
        "POST", f"/v1/conversations/{cid}/messages", json={"text": "q"}, headers=headers or {}
    ) as response:
        await response.aread()
        return response.status_code


# ── authn ──────────────────────────────────────────────────────────────


async def test_auth_off_everything_is_anonymous():
    app = _app(jwt_secret=None)
    async with _client(app) as client:
        assert await _post(client, "c1") == 200  # no header needed in dev mode
        assert (await client.get("/v1/conversations/c1")).status_code == 200
    await app.state.registry.shutdown_all()


async def test_auth_on_rejects_missing_bad_and_expired_tokens():
    app = _app(jwt_secret=SECRET)
    async with _client(app) as client:
        assert await _post(client, "c1") == 401  # no header
        assert await _post(client, "c1", {"Authorization": "Bearer garbage"}) == 401
        wrong = "wrong-secret-0123456789abcdef-01234567"
        assert await _post(client, "c1", _bearer("alice", secret=wrong)) == 401
        assert await _post(client, "c1", _bearer("alice", ttl=-10)) == 401  # expired
        assert (await client.get("/v1/conversations/c1")).status_code == 401
        assert (await client.post("/v1/conversations/c1/cancel")).status_code == 401
    assert app.state.registry.active_ids() == []  # nothing spawned by rejects


async def test_valid_token_resolves_the_principal_and_works():
    app = _app(jwt_secret=SECRET)
    async with _client(app) as client:
        assert await _post(client, "c1", _bearer("alice")) == 200
        assert app.state.store.owner("c1") == "alice"  # first toucher claimed it
    await app.state.registry.shutdown_all()


async def test_health_stays_open():
    app = _app(jwt_secret=SECRET)
    async with _client(app) as client:
        assert (await client.get("/health")).status_code == 200  # container probes


# ── ownership ──────────────────────────────────────────────────────────


async def test_foreign_conversation_is_404_on_every_route():
    app = _app(jwt_secret=SECRET)
    async with _client(app) as client:
        assert await _post(client, "c1", _bearer("alice")) == 200

        # bob sees alice's conversation as nonexistent — 404, never 403
        assert await _post(client, "c1", _bearer("bob")) == 404
        assert (
            await client.get("/v1/conversations/c1", headers=_bearer("bob"))
        ).status_code == 404
        cancel = await client.post("/v1/conversations/c1/cancel", headers=_bearer("bob"))
        assert cancel.status_code == 404

        # alice still owns it; bob gets his own id untouched
        assert (
            await client.get("/v1/conversations/c1", headers=_bearer("alice"))
        ).status_code == 200
        assert await _post(client, "c2", _bearer("bob")) == 200
    await app.state.registry.shutdown_all()


async def test_ownership_survives_eviction_via_the_store():
    # eviction drops the live agent but NOT the snapshot/owner — the resumed
    # conversation still belongs to its principal
    app = _app(jwt_secret=SECRET)
    async with _client(app) as client:
        assert await _post(client, "c1", _bearer("alice")) == 200
        await app.state.registry.shutdown_all()  # stand-in for idle eviction
        assert app.state.registry.active_ids() == []
        assert await _post(client, "c1", _bearer("bob")) == 404  # still alice's
        assert await _post(client, "c1", _bearer("alice")) == 200  # resumes fine
    await app.state.registry.shutdown_all()
