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
    ErrorEvent,
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


# ── error taxonomy (commit 2) ──────────────────────────────────────────


async def test_5xx_is_retryable_with_parsed_openai_error_body():
    provider, _ = _provider(
        status=503, body='{"error": {"message": "upstream saturated", "code": "overloaded"}}'
    )
    with pytest.raises(ProviderError) as excinfo:
        await _events(provider)
    error = excinfo.value
    assert error.retryable and error.http_status == 503
    assert error.code == "overloaded" and "upstream saturated" in error.message


async def test_4xx_is_terminal_and_vllm_flat_body_parses():
    provider, _ = _provider(
        status=400, body='{"object": "error", "message": "bad request", "code": 40004}'
    )
    with pytest.raises(ProviderError) as excinfo:
        await _events(provider)
    error = excinfo.value
    assert not error.retryable and error.code == "40004"


async def test_overflow_code_feeds_kernel_classifier():
    provider, _ = _provider(
        status=400,
        body='{"error": {"message": "maximum context length exceeded", "code": "context_length_exceeded"}}',
    )
    with pytest.raises(ProviderError) as excinfo:
        await _events(provider)
    assert excinfo.value.is_context_overflow()  # the kernel's overflow recovery keys on this


async def test_429_carries_retry_after_header():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, content=b'{"error": {"message": "slow down"}}', headers={"Retry-After": "45"}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatProvider(base_url="https://x/v1", model="m", client=client)
    with pytest.raises(ProviderError) as excinfo:
        await _events(provider)
    error = excinfo.value
    assert error.http_status == 429 and error.retryable and error.retry_after_secs == 45


async def test_transport_failure_at_open_is_retryable_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatProvider(base_url="https://x/v1", model="m", client=client)
    with pytest.raises(ProviderError) as excinfo:
        await _events(provider)
    assert excinfo.value.retryable and "connection refused" in excinfo.value.message


async def test_mid_stream_transport_death_becomes_error_event():
    good_chunk = (
        b'data: {"id":"c","model":"m","choices":[{"index":0,'
        b'"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
    )

    async def raising_body():
        yield good_chunk
        raise httpx.ReadError("connection reset")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=raising_body(), headers={"Content-Type": "text/event-stream"}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenAICompatProvider(base_url="https://x/v1", model="m", client=client)
    events = await _events(provider)  # must NOT raise: the stream had opened
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["partial"]
    errors = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(errors) == 1 and errors[0].error.retryable
    assert not any(isinstance(e, Done) for e in events)  # failed, not finished


# ── options mapping + affinity (commit 2) ──────────────────────────────


async def test_chat_options_map_to_wire_and_sideband_stays_off():
    provider, seen = _provider("text_stream.sse")
    options = ChatOptions(
        reasoning_effort="high",
        max_tokens=512,
        temperature=0.2,
        tool_choice="required",
        rate_limit_retry_owner="kernel",
    )
    async for _ in provider.chat_stream([Message.user("q")], [], options):
        pass
    payload = json.loads(seen[0].content)
    assert payload["max_tokens"] == 512
    assert payload["temperature"] == 0.2
    assert payload["reasoning_effort"] == "high"
    assert payload["tool_choice"] == "required"
    assert "rate_limit_retry_owner" not in json.dumps(payload)  # sideband never on wire


async def test_neutral_options_omit_everything_and_specific_tool_maps():
    provider, seen = _provider("text_stream.sse")
    async for _ in provider.chat_stream([Message.user("q")], [], ChatOptions()):
        pass
    payload = json.loads(seen[0].content)
    for absent in ("max_tokens", "temperature", "reasoning_effort", "tool_choice"):
        assert absent not in payload  # a neutral request carries no opinions

    provider2, seen2 = _provider("text_stream.sse")
    async for _ in provider2.chat_stream(
        [Message.user("q")], [], ChatOptions(tool_choice="kb_search")
    ):
        pass
    payload2 = json.loads(seen2[0].content)
    assert payload2["tool_choice"] == {"type": "function", "function": {"name": "kb_search"}}


async def test_bind_session_id_is_one_shot_affinity_header():
    provider, seen = _provider("text_stream.sse")
    async for _ in provider.chat_stream([Message.user("q")], [], ChatOptions()):
        pass
    assert "x-turnstile-session-id" not in seen[0].headers  # unbound: header omitted

    provider.bind_session_id("session-42")
    provider.bind_session_id("later-rebind-ignored")
    async for _ in provider.chat_stream([Message.user("q")], [], ChatOptions()):
        pass
    assert seen[1].headers["x-turnstile-session-id"] == "session-42"


async def test_enable_thinking_maps_to_the_vllm_chat_template_kwarg():
    # a per-call knob like reasoning_effort: None = omitted, False/True = sent
    # the vLLM/SGLang way (forwarded into the model's chat template)
    provider, seen = _provider("text_stream.sse")
    async for _ in provider.chat_stream([Message.user("q")], [], ChatOptions(max_tokens=200)):
        pass
    assert "chat_template_kwargs" not in json.loads(seen[0].content)
    async for _ in provider.chat_stream(
        [Message.user("q")], [], ChatOptions(max_tokens=200, enable_thinking=False)
    ):
        pass
    payload = json.loads(seen[1].content)
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["max_tokens"] == 200  # per-call options intact beside it
