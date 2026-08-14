"""Engine mock suite — streaming seams (commit 8 scope): reasoning channel,
signed blocks, misrouted-answer promotion, dropped-call honoring."""

import pytest

from turnstile.kernel import events as ev
from turnstile.kernel.dtos import (
    Done,
    Message,
    Reasoning,
    ReasoningSignature,
    StopReason,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)
from turnstile.kernel.engine import Agent
from turnstile.kernel.ports import LifecycleHooks
from turnstile.kernel.testkit import (
    CountingTool,
    FnHook,
    ScriptedProvider,
    StepClock,
)

pytestmark = pytest.mark.unit


async def _collect(agent: Agent, text: str = "q") -> list:
    handle = agent.spawn()
    await handle.commands.put(ev.SendMessage(text=text))
    events = []
    while True:
        event = await handle.events.get()
        events.append(event)
        if isinstance(event, ev.TurnComplete):
            break
    await handle.commands.put(ev.Shutdown())
    await handle.task
    return events


def _stored_via_hook() -> tuple[list[Message], FnHook]:
    stored: list[Message] = []
    return stored, FnHook(on_model_response=stored.append)


# ── reasoning channel ──────────────────────────────────────────────────


async def test_reasoning_streams_live_and_stores_flat():
    provider = ScriptedProvider(
        rounds=[
            [Reasoning("thinking "), Reasoning("hard"), TextDelta("answer"), Done()],
        ]
    )
    stored, keeper = _stored_via_hook()
    agent = Agent(provider=provider, hooks=[keeper], clock=StepClock(step_ms=5))
    events = await _collect(agent)
    live = "".join(e.text for e in events if isinstance(e, ev.Reasoning))
    assert live == "thinking hard"
    assert stored[0].reasoning == "thinking hard"
    assert stored[0].text == "answer"
    meta = stored[0].meta
    assert meta is not None
    assert meta.reasoning_elapsed_ms > 0  # thinking phase measured
    assert meta.elapsed_ms >= meta.reasoning_elapsed_ms


async def test_reasoning_redaction_consistent_live_and_stored():
    provider = ScriptedProvider(
        rounds=[
            [Reasoning("the SECRET step"), TextDelta("done"), Done()],
        ]
    )
    stored, keeper = _stored_via_hook()

    class Redact(LifecycleHooks):
        async def on_reasoning_delta(self, delta: str) -> str:
            return delta.replace("SECRET", "[redacted]")

    agent = Agent(provider=provider, hooks=[Redact(), keeper])
    events = await _collect(agent)
    live = "".join(e.text for e in events if isinstance(e, ev.Reasoning))
    assert "[redacted]" in live and "SECRET" not in live
    assert stored[0].reasoning == "the [redacted] step"


# ── signed reasoning blocks ────────────────────────────────────────────


async def test_reasoning_signature_finalizes_blocks_in_order():
    provider = ScriptedProvider(
        rounds=[
            [
                Reasoning("first thought"),
                ReasoningSignature(opaque="sig-1", provider="anthropic"),
                ReasoningSignature(opaque="redacted-blob", provider="anthropic"),  # no text
                Reasoning("second thought"),
                ReasoningSignature(opaque="sig-2", provider="anthropic"),
                TextDelta("the answer"),
                Done(),
            ]
        ]
    )
    stored, keeper = _stored_via_hook()
    agent = Agent(provider=provider, hooks=[keeper])
    await _collect(agent)
    blocks = stored[0].reasoning_blocks
    assert [(b.text, b.opaque) for b in blocks] == [
        ("first thought", "sig-1"),
        ("", "redacted-blob"),  # redacted block: opaque only
        ("second thought", "sig-2"),
    ]
    assert all(b.provider == "anthropic" for b in blocks)
    # flat reasoning keeps the full text for the plain-text echo path
    assert stored[0].reasoning == "first thoughtsecond thought"


# ── misrouted-answer promotion ─────────────────────────────────────────


async def test_reasoning_only_stop_promotes_to_content():
    provider = ScriptedProvider(
        rounds=[
            [Reasoning("the actual answer landed here"), Done()],
        ]
    )
    stored, keeper = _stored_via_hook()
    agent = Agent(provider=provider, hooks=[keeper])
    events = await _collect(agent)
    assert events[-1].reason is StopReason.STOPPED
    # promoted text reaches the driver live AND the stored body
    live_text = "".join(e.text for e in events if isinstance(e, ev.TextDelta))
    assert live_text == "the actual answer landed here"
    assert stored[0].text == "the actual answer landed here"
    assert stored[0].reasoning is None  # now the body; not double-stored


async def test_promotion_passes_the_content_scrub_seam():
    provider = ScriptedProvider(
        rounds=[
            [Reasoning("answer with SECRET inside"), Done()],
        ]
    )
    stored, keeper = _stored_via_hook()
    agent = Agent(
        provider=provider,
        hooks=[FnHook(on_text_delta=lambda d: d.replace("SECRET", "[redacted]")), keeper],
    )
    events = await _collect(agent)
    live_text = "".join(e.text for e in events if isinstance(e, ev.TextDelta))
    assert live_text == "answer with [redacted] inside"
    assert stored[0].text == "answer with [redacted] inside"


async def test_promotion_gated_off_for_tool_calls_signed_blocks_and_truncation():
    # tool-call round: reasoning stays reasoning
    provider = ScriptedProvider(
        rounds=[
            [Reasoning("planning"), ToolCallEvent(ToolCall("c1", "count", "{}")), Done()],
            [TextDelta("done"), Done()],
        ]
    )
    stored, keeper = _stored_via_hook()
    agent = Agent(provider=provider, tools={"count": CountingTool()}, hooks=[keeper])
    await _collect(agent)
    assert stored[0].reasoning == "planning" and stored[0].text == ""

    # signed-block round: promoting would desync blocks — excluded
    provider2 = ScriptedProvider(
        rounds=[
            [
                Reasoning("signed thinking"),
                ReasoningSignature(opaque="sig", provider="anthropic"),
                Done(),
            ]
        ]
    )
    stored2, keeper2 = _stored_via_hook()
    await _collect(Agent(provider=provider2, hooks=[keeper2]))
    assert stored2[0].text == "" and stored2[0].reasoning == "signed thinking"

    # truncated round: a cut-off response is not a stop — excluded. The
    # truncation auto-continuation (resilience commit) re-asks; script an EMPTY
    # round 2 so the empty-200 re-issue fires once too (the two tiers compose:
    # truncation-continue -> empty retry -> recover), then the real answer on
    # call 3. backoff_scale=0 keeps the retry sleep out of the suite runtime.
    provider3 = ScriptedProvider(
        rounds=[
            [Reasoning("cut off"), Done(truncated=True)],
            [Done()],  # content-free 200: re-issued, not mistaken for a stop
            [TextDelta("resumed and finished"), Done()],
        ]
    )
    stored3, keeper3 = _stored_via_hook()
    await _collect(Agent(provider=provider3, hooks=[keeper3], backoff_scale=0.0))
    assert len(provider3.calls) == 3  # truncated -> empty retry -> answered
    assert stored3[0].text == "" and stored3[0].reasoning == "cut off"
    meta3 = stored3[0].meta
    assert meta3 is not None
    assert meta3.finish_reason == "length"


# ── dropped calls never execute ────────────────────────────────────────


async def test_on_model_response_dropping_calls_prevents_execution():
    counter = CountingTool()
    provider = ScriptedProvider(
        rounds=[
            [TextDelta("let me check"), ToolCallEvent(ToolCall("c1", "count", "{}")), Done()],
        ]
    )

    def drop_calls(message: Message) -> None:
        message.tool_calls.clear()

    agent = Agent(
        provider=provider,
        tools={"count": counter},
        hooks=[FnHook(on_model_response=drop_calls)],
    )
    events = await _collect(agent)
    assert counter.count == 0  # the dropped call never executed
    assert not [e for e in events if isinstance(e, ev.ToolResultEvent)]
    assert events[-1].reason is StopReason.STOPPED  # no-calls branch taken
