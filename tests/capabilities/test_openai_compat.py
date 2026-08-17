"""OpenAI-compat provider adapter — recorded SSE fixtures replayed through a
fake httpx transport: no network, byte-level wire assertions both directions."""

import json
from pathlib import Path

import httpx
import pytest

from turnstile.capabilities.providers.openai_compat import OpenAICompatProvider
from turnstile.kernel.dtos import (
    ChatOptions,
    Done,
    Malformed,
    Message,
    ProviderError,
    Reasoning,
    ResponseId,
    ResponseModel,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolCallEvent,
    ToolDef,
    UsageEvent,
)

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures"


def _provider(fixture: str | None = None, status: int = 200, body: str = ""):
    """Provider wired to a MockTransport that replays a fixture (or an error
    body) and records the outgoing request for wire assertions."""
    seen: list[httpx.Request] = []
    content = (FIXTURES / fixture).read_bytes() if fixture else body.encode()

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        headers = {"Content-Type": "text/event-stream"} if status == 200 else {}
        return httpx.Response(status, content=content, headers=headers)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatProvider(
        base_url="https://ds4.example/model/v1",
        model="model-fast",
        api_key="sk-test",
        context_window=128_000,
        client=client,
    )
    return provider, seen


async def _events(provider, messages=None, tools=None) -> list:
    events = []
    async for event in provider.chat_stream(
        messages or [Message.user("q")], tools or [], ChatOptions()
    ):
        events.append(event)
    return events


# ── happy-path text stream ─────────────────────────────────────────────


async def test_text_stream_maps_to_events():
    provider, _ = _provider("text_stream.sse")
    events = await _events(provider)
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["hello", " turnstile"]
    assert isinstance(events[-1], Done) and not events[-1].truncated
    ids = [e for e in events if isinstance(e, ResponseId)]
    models = [e for e in events if isinstance(e, ResponseModel)]
    assert len(ids) == 1 and ids[0].id == "chatcmpl-bffe559ff4137452"  # once, not per chunk
    assert len(models) == 1 and models[0].model == "model-fast"
    usage = next(e.usage for e in events if isinstance(e, UsageEvent))
    assert (usage.prompt, usage.completion, usage.cached) == (18, 5, 0)


async def test_request_wire_shape():
    provider, seen = _provider("text_stream.sse")
    tools = [ToolDef(name="kb_search", description="Search the KB", parameters={"type": "object"})]
    messages = [
        Message.system("persona"),
        Message.user("q1"),
        Message.assistant("", [ToolCall("c1", "kb_search", '{"query": "x"}')]),
        Message.tool_result("c1", "result text"),
        Message.assistant("prior answer"),
    ]
    await _events(provider, messages=messages, tools=tools)
    request = seen[0]
    assert request.url.path.endswith("/chat/completions")
    assert request.headers["authorization"] == "Bearer sk-test"
    payload = json.loads(request.content)
    assert payload["model"] == "model-fast"
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    wire = payload["messages"]
    assert [m["role"] for m in wire] == ["system", "user", "assistant", "tool", "assistant"]
    assert wire[2]["tool_calls"][0] == {
        "id": "c1",
        "type": "function",
        "function": {"name": "kb_search", "arguments": '{"query": "x"}'},
    }
    assert wire[3]["tool_call_id"] == "c1"
    assert payload["tools"][0]["function"]["name"] == "kb_search"


async def test_rendering_is_deterministic():
    # The prefix-cache contract crosses the adapter: same Messages -> same bytes.
    from turnstile.capabilities.providers.openai_compat import render_messages

    messages = [Message.user("q"), Message.assistant("a")]
    assert json.dumps(render_messages(messages)) == json.dumps(render_messages(messages))


# ── tool-call stream ───────────────────────────────────────────────────


async def test_tool_call_fragments_stream_live_and_assemble():
    provider, _ = _provider("tool_call_stream.sse")
    events = await _events(provider)
    # text preamble before the calls maps as ordinary deltas
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert text == "I'll search both sources in parallel for you.\n\n"
    # live display fragments forwarded as they arrived (two parallel calls)
    deltas = [e for e in events if isinstance(e, ToolCallDelta)]
    assert [d.index for d in deltas] == [0, 0, 0, 0, 1, 1, 1, 1]
    # complete calls assembled per index, arguments concatenated across chunks
    calls = [e.call for e in events if isinstance(e, ToolCallEvent)]
    assert calls == [
        ToolCall(
            id="chatcmpl-tool-8c50bcb5f285cc56",
            name="kb_search",
            arguments='{"query": "refund policy"}',
        ),
        ToolCall(
            id="chatcmpl-tool-9d2249e9f629a109",
            name="web_search",
            arguments='{"q": "refund policy site"}',
        ),
    ]
    # assembled calls come AFTER the fragments, BEFORE Done
    assert isinstance(events[-1], Done)
    assert events.index(calls and next(e for e in events if isinstance(e, ToolCallEvent))) > max(
        events.index(d) for d in deltas
    )


# ── truncation ─────────────────────────────────────────────────────────


async def test_truncation_sets_done_truncated():
    provider, _ = _provider("truncated.sse")
    events = await _events(provider)
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "**Title: The"
    assert isinstance(events[-1], Done) and events[-1].truncated  # finish_reason=length


# ── synthetic chunks (ds4 can't produce these on demand; see fixtures/README) ──


def _chunk(delta: dict, finish: str | None = None) -> str:
    choice = {"index": 0, "delta": delta, "finish_reason": finish}
    return "data: " + json.dumps({"id": "c", "model": "m", "choices": [choice]}) + "\n\n"


async def test_malformed_chunk_is_a_signal_not_content():
    body = (
        _chunk({"content": "before"})
        + "data: this line is not json (gateway hiccup)\n\n"
        + _chunk({"content": " after"})
        + _chunk({}, finish="stop")
        + "data: [DONE]\n\n"
    )
    provider, _ = _provider(body=body)
    events = await _events(provider)
    assert sum(isinstance(e, Malformed) for e in events) == 1  # garbage = signal
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["before", " after"]


async def test_reasoning_channel_maps():
    body = (
        _chunk({"reasoning_content": "need to search"})
        + _chunk({"content": "answer"})
        + _chunk({}, finish="stop")
        + "data: [DONE]\n\n"
    )
    provider, _ = _provider(body=body)
    events = await _events(provider)
    assert [e.text for e in events if isinstance(e, Reasoning)] == ["need to search"]
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["answer"]


async def test_cached_tokens_map_from_prompt_details():
    usage_chunk = (
        'data: {"id":"c","model":"m","choices":[],"usage":{"prompt_tokens":21,'
        '"completion_tokens":4,"total_tokens":25,'
        '"prompt_tokens_details":{"cached_tokens":16}}}\n\n'
    )
    body = _chunk({}, finish="stop") + usage_chunk + "data: [DONE]\n\n"
    provider, _ = _provider(body=body)
    events = await _events(provider)
    usage = next(e.usage for e in events if isinstance(e, UsageEvent))
    assert (usage.prompt, usage.completion, usage.cached) == (21, 4, 16)


# ── failed open ────────────────────────────────────────────────────────


async def test_non_200_raises_structured_provider_error():
    provider, _ = _provider(status=401, body='{"error": "Unauthorized: Access Denied"}')
    with pytest.raises(ProviderError) as excinfo:
        await _events(provider)
    assert excinfo.value.http_status == 401
    assert "Access Denied" in excinfo.value.message
