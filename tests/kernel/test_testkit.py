"""Sanity tests for the shipped test doubles — they must behave as documented,
since every later engine/L1/L2 test builds on them."""

import asyncio

import pytest

from turnstile.kernel.dtos import (
    ChatOptions,
    Done,
    Gate,
    Message,
    ProviderError,
    SessionSnapshot,
    TextDelta,
    ToolCall,
    ToolContext,
)
from turnstile.kernel.ports import CompactionCheckpointError
from turnstile.kernel.testkit import (
    AlwaysStopProvider,
    BeforeOutcome,
    BlockUntilCancelTool,
    ConcurrencyProbeTool,
    ConcurrencyState,
    ContinueOnceHook,
    CountingTool,
    EchoTool,
    FailingTool,
    FnMiddleware,
    MemoryCheckpoint,
    RecorderHook,
    ScriptedProvider,
    SilentProvider,
    StepClock,
)

pytestmark = pytest.mark.unit


async def _drain(provider, messages=None, tools=None, options=None):
    events = []
    async for ev in provider.chat_stream(messages or [], tools or [], options or ChatOptions()):
        events.append(ev)
    return events


# ── providers ──────────────────────────────────────────────────────────


async def test_scripted_provider_pops_rounds_and_records_calls():
    p = ScriptedProvider(rounds=[[TextDelta("a"), Done()], [TextDelta("b"), Done()]])
    first = await _drain(p, messages=[Message.user("q")])
    second = await _drain(p)
    assert [e.text for e in first if isinstance(e, TextDelta)] == ["a"]
    assert [e.text for e in second if isinstance(e, TextDelta)] == ["b"]
    assert len(p.calls) == 2
    assert p.received_texts(0) == [("user", "q")]  # full wire recorded per call


async def test_scripted_provider_exhausted_script_yields_bare_done():
    p = ScriptedProvider(rounds=[])
    events = await _drain(p)
    assert len(events) == 1 and isinstance(events[0], Done)


async def test_scripted_provider_error_round_raises_before_any_event():
    p = ScriptedProvider(rounds=[ProviderError(message="boom", http_status=500)])
    with pytest.raises(ProviderError, match="boom"):
        await _drain(p)


async def test_always_stop_provider_repeats_forever():
    p = AlwaysStopProvider("done")
    for _ in range(3):
        events = await _drain(p)
        assert [type(e) for e in events] == [TextDelta, Done]


async def test_silent_provider_pends_after_prefix():
    p = SilentProvider(prefix=[TextDelta("partial")])
    got = []

    async def consume():
        async for ev in p.chat_stream([], [], ChatOptions()):
            got.append(ev)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(consume(), timeout=0.05)
    assert [e.text for e in got] == ["partial"]  # prefix arrived, then silence


# ── clock ──────────────────────────────────────────────────────────────


def test_step_clock_advances_deterministically():
    c = StepClock(step_ms=10)
    assert (c.now_millis(), c.now_millis(), c.now_millis()) == (10, 20, 30)


# ── tools ──────────────────────────────────────────────────────────────


async def test_echo_and_counting_tools():
    ctx = ToolContext(working_dir=".")
    assert (await EchoTool().execute('{"text": "x"}', ctx)).content == 'echo: {"text": "x"}'
    counter = CountingTool()
    await counter.execute("{}", ctx)
    await counter.execute("{}", ctx)
    assert counter.count == 2


async def test_concurrency_probe_records_overlap():
    state = ConcurrencyState()
    a = ConcurrencyProbeTool(state, name="a")
    b = ConcurrencyProbeTool(state, name="b")
    ctx = ToolContext(working_dir=".")
    await asyncio.gather(a.execute("{}", ctx), b.execute("{}", ctx))
    assert state.max_active == 2  # they overlapped
    state2 = ConcurrencyState()
    c = ConcurrencyProbeTool(state2, name="c")
    await c.execute("{}", ctx)
    assert state2.max_active == 1


async def test_block_until_cancel_tool_is_cooperative():
    ctx = ToolContext(working_dir=".")
    task = asyncio.create_task(BlockUntilCancelTool().execute("{}", ctx))
    await asyncio.sleep(0)
    assert not task.done()
    ctx.cancel.cancel()
    result = await asyncio.wait_for(task, timeout=1)
    assert result.is_error and "cancelled" in result.content


async def test_failing_tool_raises():
    with pytest.raises(RuntimeError, match="tool exploded"):
        await FailingTool().execute("{}", ToolContext(working_dir="."))


# ── hooks / middleware / checkpoint ────────────────────────────────────


async def test_recorder_hook_logs_seams():
    h = RecorderHook()
    await h.user_prompt_submit("q")
    await h.on_text_delta("d")
    await h.offer_continuation(None)  # type: ignore[arg-type]
    assert h.log == ["user_prompt_submit", "on_text_delta", "offer_continuation"]


async def test_continue_once_hook_fires_exactly_once():
    h = ContinueOnceHook("more")
    assert await h.offer_continuation(None) == "more"  # type: ignore[arg-type]
    assert await h.offer_continuation(None) is None  # type: ignore[arg-type]


async def test_fn_middleware_wires_callables():
    deny = FnMiddleware(before=lambda call, tool: BeforeOutcome(gate=Gate.DENY, reason="no"))
    outcome = await deny.before(ToolCall("1", "echo", "{}"), EchoTool(), rt=None)
    assert outcome.gate is Gate.DENY and outcome.reason == "no"
    passthrough = FnMiddleware()
    assert (await passthrough.before(ToolCall("1", "echo", "{}"), EchoTool(), rt=None)) is (
        BeforeOutcome.PROCEED
    )


def test_memory_checkpoint_stores_or_fails():
    ok = MemoryCheckpoint()
    snap = SessionSnapshot(version=1, messages=[])
    ok.save(snap)
    assert ok.saved == [snap]
    failing = MemoryCheckpoint(fail_with="disk full")
    with pytest.raises(CompactionCheckpointError, match="disk full"):
        failing.save(snap)
