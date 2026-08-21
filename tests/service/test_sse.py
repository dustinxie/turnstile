"""SSE turn endpoint — real ASGI streaming over a scripted registry: exact
event framing, tool events, steer-on-mid-turn-POST, validation."""

import asyncio
import json

import httpx
import pytest

from turnstile.capabilities.persistence.memory_store import MemorySessionStore
from turnstile.kernel.dtos import Done, StopReason, TextDelta, ToolCall, ToolCallEvent
from turnstile.kernel.engine import Agent
from turnstile.kernel.testkit import EchoTool, ScriptedProvider
from turnstile.products.middleware.references import ReferenceCollector
from turnstile.root import AssembledAgent
from turnstile.service.app import create_app
from turnstile.service.registry import ConversationRegistry
from turnstile.service.routes import event_data, event_name

pytestmark = pytest.mark.service


def _app(provider_factory):
    """App whose registry assembles scripted agents — zero network."""
    from turnstile.config import Config

    cfg = Config(
        _env_file=None,  # type: ignore[call-arg]
        llm={"base_url": "https://ds4.example/v1", "model": "model-fast"},
        kb={
            "embedding_url": "https://gpu.example/embed",
            "milvus_url": "https://milvus.example/search",
            "collection": "c",
            "expr": "e",
        },
    )
    store = MemorySessionStore()

    def scripted_assemble(cfg, session_id, store):
        agent = Agent(
            provider=provider_factory(),
            tools={"echo": EchoTool()},
            hooks=[store.hook(session_id)],
            session_id=session_id,
            resume=store.load(session_id),
        )
        return AssembledAgent(agent=agent, references=ReferenceCollector(), store=store)

    app = create_app(cfg)
    app.state.registry = ConversationRegistry(cfg, store, assemble=scripted_assemble)
    return app


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


async def _sse_events(response) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, data) pairs, ignoring pings/comments."""
    events, name = [], None
    async for line in response.aiter_lines():
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and name:
            events.append((name, json.loads(line.split(":", 1)[1].strip())))
            name = None
    return events


# ── serialization ──────────────────────────────────────────────────────


def test_event_names_are_snake_cased_and_data_is_json():
    from turnstile.kernel import events as ev

    complete = ev.TurnComplete(reason=StopReason.STOPPED)
    assert event_name(complete) == "turn_complete"
    assert json.loads(event_data(complete))["reason"] == "stopped"  # enum flattened
    delta = ev.TextDelta(text="hi")
    assert (event_name(delta), json.loads(event_data(delta))["text"]) == ("text_delta", "hi")


# ── the stream ─────────────────────────────────────────────────────────


async def test_turn_streams_as_verbatim_sse_events():
    app = _app(lambda: ScriptedProvider(rounds=[[TextDelta("hel"), TextDelta("lo"), Done()]]))
    async with (
        _client(app) as client,
        client.stream("POST", "/v1/conversations/c1/messages", json={"text": "hi"}) as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = await _sse_events(response)

    names = [n for n, _ in events]
    assert names[-1] == "turn_complete"  # terminal, stream closed after it
    deltas = [d["text"] for n, d in events if n == "text_delta"]
    assert deltas == ["hel", "lo"]
    assert events[-1][1]["reason"] == "stopped"


async def test_tool_rounds_ride_the_same_stream():
    app = _app(
        lambda: ScriptedProvider(
            rounds=[
                [ToolCallEvent(ToolCall("t1", "echo", '{"x": 1}')), Done()],
                [TextDelta("done"), Done()],
            ]
        )
    )
    async with (
        _client(app) as client,
        client.stream(
            "POST", "/v1/conversations/c1/messages", json={"text": "run the tool"}
        ) as response,
    ):
        events = await _sse_events(response)

    names = [n for n, _ in events]
    assert "tool_started" in names and "tool_result_event" in names
    result = next(d for n, d in events if n == "tool_result_event")
    assert result["result"]["content"] == 'echo: {"x": 1}'  # nested DTO serialized


async def test_two_sequential_turns_reuse_the_conversation():
    app = _app(
        lambda: ScriptedProvider(rounds=[[TextDelta("a1"), Done()], [TextDelta("a2"), Done()]])
    )
    async with _client(app) as client:
        for expected in ("a1", "a2"):
            async with client.stream(
                "POST", "/v1/conversations/c1/messages", json={"text": "q"}
            ) as response:
                events = await _sse_events(response)
            text = "".join(d["text"] for n, d in events if n == "text_delta")
            assert text == expected
    # both turns snapshotted into ONE session
    snapshot = app.state.registry.get_or_create("c1").bundle.store.load("c1")
    assert snapshot is not None and snapshot.turn_counter == 2
    await app.state.registry.shutdown_all()


# ── steer on mid-turn POST ─────────────────────────────────────────────


class _GatedProvider:
    """Round 1 blocks until the test releases it (a running turn to steer
    into); the post-steer round answers normally. Implements the LlmProvider
    surface duck-typed."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.calls: list[list] = []

    def model_name(self) -> str:
        return "gated"

    def context_window(self) -> int:
        return 0

    def bind_session_id(self, session_id: str) -> None:
        pass

    async def chat_stream(self, messages, tools, options):
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            await self.release.wait()
            yield TextDelta("first ")
        else:
            yield TextDelta("and the steered part")
        yield Done()


async def test_mid_turn_post_steers_with_202():
    providers: list[_GatedProvider] = []

    def factory():
        provider = _GatedProvider()
        providers.append(provider)
        return provider

    app = _app(factory)
    async with _client(app) as client:

        async def run_first_turn():
            # ASGITransport buffers the whole response, so this pends until
            # the turn ends — run it as a task and interleave the steer.
            async with client.stream(
                "POST", "/v1/conversations/c1/messages", json={"text": "q1"}
            ) as first:
                assert first.status_code == 200
                return await _sse_events(first)

        turn = asyncio.create_task(run_first_turn())
        while not providers:  # the route has spawned the agent
            await asyncio.sleep(0.01)
        while not app.state.registry.get_or_create("c1").in_flight:
            await asyncio.sleep(0.01)

        second = await client.post("/v1/conversations/c1/messages", json={"text": "also this"})
        assert second.status_code == 202
        assert second.json() == {"status": "steered"}  # no 409, no second stream

        providers[0].release.set()  # let the gated round finish
        events = await turn

    # ONE turn, ONE terminal event; the steered prompt folded into round 2
    assert [n for n, _ in events].count("turn_complete") == 1
    text = "".join(d["text"] for n, d in events if n == "text_delta")
    assert text == "first and the steered part"
    steered_round = providers[0].calls[1]
    assert any("also this" in m.text for m in steered_round if m.text)
    await app.state.registry.shutdown_all()


# ── validation ─────────────────────────────────────────────────────────


async def test_empty_text_is_rejected_before_any_turn():
    app = _app(lambda: ScriptedProvider(rounds=[]))
    async with _client(app) as client:
        response = await client.post("/v1/conversations/c1/messages", json={"text": ""})
        assert response.status_code == 422
    assert app.state.registry.active_ids() == []  # nothing was spawned
