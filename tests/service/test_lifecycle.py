"""Turn lifecycle over HTTP — disconnect detaches (never cancels), explicit
cancel via the endpoint, and the GET refetch surface."""

import asyncio
import json

import httpx
import pytest

from turnstile.capabilities.persistence.memory_store import MemorySessionStore
from turnstile.kernel.dtos import Done, TextDelta
from turnstile.kernel.engine import Agent
from turnstile.products.middleware.references import ReferenceCollector
from turnstile.root import AssembledAgent
from turnstile.service.app import create_app
from turnstile.service.registry import ConversationRegistry

pytestmark = pytest.mark.service


class _GatedProvider:
    """Each round blocks until released — a running turn the test controls."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.release = asyncio.Event()
        self.calls = 0

    def model_name(self) -> str:
        return "gated"

    def context_window(self) -> int:
        return 0

    def bind_session_id(self, session_id: str) -> None:
        pass

    async def chat_stream(self, messages, tools, options):
        call = self.calls = self.calls + 1
        if call == 1:
            await self.release.wait()
        yield TextDelta(self.texts[min(call, len(self.texts)) - 1])
        yield Done()


def _app(provider_factory):
    from turnstile.config import Config

    cfg = Config(
        _env_file=None,  # type: ignore[call-arg]
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
            provider=provider_factory(),
            hooks=[store.hook(session_id)],
            session_id=session_id,
            resume=store.load(session_id),
            keep_interrupted_context=True,  # the chatbot cancel semantics
        )
        return AssembledAgent(agent=agent, references=ReferenceCollector(), store=store)

    app = create_app(cfg)
    app.state.store = store
    app.state.registry = ConversationRegistry(cfg, store, assemble=scripted_assemble)
    return app


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


async def _sse_events(response) -> list[tuple[str, dict]]:
    events, name = [], None
    async for line in response.aiter_lines():
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and name:
            events.append((name, json.loads(line.split(":", 1)[1].strip())))
            name = None
    return events


async def _post_stream(client, conversation_id: str, text: str) -> list[tuple[str, dict]]:
    async with client.stream(
        "POST", f"/v1/conversations/{conversation_id}/messages", json={"text": text}
    ) as response:
        return await _sse_events(response)


async def _wait_for(predicate, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


# ── explicit cancel ────────────────────────────────────────────────────


async def test_cancel_endpoint_cancels_the_running_turn():
    provider = _GatedProvider(["never emitted"])
    app = _app(lambda: provider)
    async with _client(app) as client:
        turn = asyncio.create_task(_post_stream(client, "c1", "q1"))
        await _wait_for(lambda: (e := app.state.registry.get("c1")) is not None and e.in_flight)

        response = await client.post("/v1/conversations/c1/cancel")
        assert response.status_code == 202
        assert response.json() == {"status": "cancelling"}

        events = await turn  # the open stream carries the cancel terminally
    names = [n for n, _ in events]
    assert "cancelled" in names
    assert ("turn_complete", {"reason": "cancelled"}) in events
    assert events[-1][0] == "envelope" and events[-1][1]["signal"] == "no_answer"
    # keep_interrupted_context: the user's message survived into the snapshot
    snapshot = app.state.store.load("c1")
    assert snapshot is not None
    assert any(m.text == "q1" for m in snapshot.messages)
    await app.state.registry.shutdown_all()


async def test_cancel_with_nothing_running_is_idle_not_an_error():
    app = _app(lambda: _GatedProvider(["x"]))
    async with _client(app) as client:
        response = await client.post("/v1/conversations/never-seen/cancel")
        assert response.status_code == 200
        assert response.json() == {"status": "idle"}
    assert app.state.registry.active_ids() == []  # cancel never spawns


# ── disconnect detaches, never cancels ─────────────────────────────────


async def test_disconnect_lets_the_turn_finish_detached():
    provider = _GatedProvider(["the detached answer", "second answer"])
    app = _app(lambda: provider)
    async with _client(app) as client:
        turn = asyncio.create_task(_post_stream(client, "c1", "q1"))
        await _wait_for(lambda: (e := app.state.registry.get("c1")) is not None and e.in_flight)

        turn.cancel()  # the client vanishes mid-turn
        with pytest.raises(asyncio.CancelledError):
            await turn

        entry = app.state.registry.get("c1")
        assert entry is not None and entry.in_flight  # NOT cancelled: still running

        provider.release.set()  # the model finally answers, to nobody
        await _wait_for(lambda: not entry.in_flight)  # drainer saw turn_complete

        # the answer was persisted, not lost; the client refetches it
        refetch = await client.get("/v1/conversations/c1")
        assert refetch.status_code == 200
        body = refetch.json()
        assert body["in_flight"] is False and body["turn_counter"] == 1
        assert {"role": "assistant", "text": "the detached answer"} in body["messages"]

        # and the next turn streams clean — no stale events from turn 1
        events = await _post_stream(client, "c1", "q2")
        texts = [d["text"] for n, d in events if n == "text_delta"]
        assert texts == ["second answer"]
        assert [n for n, _ in events].count("turn_complete") == 1
    await app.state.registry.shutdown_all()


# ── the refetch surface ────────────────────────────────────────────────


async def test_get_unknown_conversation_is_404():
    app = _app(lambda: _GatedProvider(["x"]))
    async with _client(app) as client:
        response = await client.get("/v1/conversations/nope")
        assert response.status_code == 404
    assert app.state.registry.active_ids() == []  # GET never spawns


async def test_get_returns_the_persisted_conversation():
    provider = _GatedProvider(["a1"])
    provider.release.set()  # no gating needed here
    app = _app(lambda: provider)
    async with _client(app) as client:
        await _post_stream(client, "c1", "q1")
        body = (await client.get("/v1/conversations/c1")).json()
    assert body["conversation_id"] == "c1"
    assert body["turn_counter"] == 1 and body["in_flight"] is False
    assert body["messages"] == [
        {"role": "user", "text": "q1"},
        {"role": "assistant", "text": "a1"},
    ]
    await app.state.registry.shutdown_all()
