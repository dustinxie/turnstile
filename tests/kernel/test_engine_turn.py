"""Engine mock suite — session loop + single-round turn lifecycle (commit 5 scope).

Every test runs the real engine against testkit doubles: no network, no model,
deterministic time.
"""

import pytest

from turnstile.kernel import events as ev
from turnstile.kernel.dtos import (
    Done,
    ErrorEvent,
    Message,
    PromptRejected,
    ProviderError,
    Role,
    SessionSnapshot,
    StopReason,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallEvent,
    UsageEvent,
)
from turnstile.kernel.engine import Agent
from turnstile.kernel.testkit import (
    ContinueOnceHook,
    EchoTool,
    FnHook,
    RecorderHook,
    ScriptedProvider,
    StepClock,
)

pytestmark = pytest.mark.unit


async def _run_turn_collect(agent: Agent, text: str) -> list:
    """Spawn, send one message, collect events through TurnComplete, shut down."""
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


def _texts(events: list) -> str:
    return "".join(e.text for e in events if isinstance(e, ev.TextDelta))


def _terminal(events: list) -> StopReason:
    return events[-1].reason


# ── the basic turn ─────────────────────────────────────────────────────


async def test_single_text_round_stops_clean():
    provider = ScriptedProvider(rounds=[[TextDelta("Hel"), TextDelta("lo"), Done()]])
    agent = Agent(provider=provider, persona="be helpful", clock=StepClock())
    events = await _run_turn_collect(agent, "hi")

    assert isinstance(events[0], ev.TurnStarted)
    assert _texts(events) == "Hello"
    assert _terminal(events) is StopReason.STOPPED
    # provider saw persona + user message
    assert provider.received_texts(0) == [("system", "be helpful"), ("user", "hi")]


async def test_assistant_message_stored_with_kernel_meta():
    provider = ScriptedProvider(
        rounds=[[TextDelta("answer"), UsageEvent(TokenUsage(prompt=90, completion=10)), Done()]],
        ctx_window=1000,
    )
    stored: list[Message] = []
    keeper = FnHook(on_model_response=stored.append)
    agent = Agent(provider=provider, hooks=[keeper], clock=StepClock(step_ms=7))
    events = await _run_turn_collect(agent, "q")

    meta = stored[0].meta
    assert meta is not None
    assert (meta.turn_id, meta.request_id, meta.round) == (1, 1, 1)
    assert meta.finish_reason == "stop"
    assert meta.tokens == TokenUsage(prompt=90, completion=10)
    assert meta.used_tokens == 90 and meta.ctx_window == 1000
    assert meta.utilization == pytest.approx(0.09)
    assert meta.elapsed_ms > 0  # StepClock advanced between start and stamp
    usage_events = [e for e in events if isinstance(e, ev.Usage)]
    assert usage_events and usage_events[0].meta.request_id == 1


async def test_second_turn_appends_and_ids_stay_monotonic():
    provider = ScriptedProvider(
        rounds=[
            [TextDelta("one"), Done()],
            [TextDelta("two"), Done()],
        ]
    )
    agent = Agent(provider=provider, persona="p")
    handle = agent.spawn()
    # Sequential sends: a SendMessage during a running turn would STEER into it
    # (commit 10) — this test needs two distinct turns.
    for text in ("q1", "q2"):
        await handle.commands.put(ev.SendMessage(text=text))
        while not isinstance(await handle.events.get(), ev.TurnComplete):
            pass
    await handle.commands.put(ev.Shutdown())
    await handle.task
    # second call saw the whole first turn: persona, q1, one, q2
    assert provider.received_texts(1) == [
        ("system", "p"),
        ("user", "q1"),
        ("assistant", "one"),
        ("user", "q2"),
    ]


# ── hooks over the turn ────────────────────────────────────────────────


async def test_recorder_hook_sees_full_seam_sequence():
    provider = ScriptedProvider(rounds=[[TextDelta("x"), Done()]])
    recorder = RecorderHook()
    agent = Agent(provider=provider, hooks=[recorder])
    await _run_turn_collect(agent, "q")
    assert recorder.log == [
        "session_start:resumed=False",
        "user_prompt_submit",
        "turn_start",
        "pre_request:round=1",
        "pre_request_options",
        "on_request:request_id=1",
        "on_text_delta",
        "on_model_response",
        "offer_continuation",
        "turn_complete:stopped",
        "session_end",
    ]


async def test_text_delta_transform_reaches_stream_and_storage():
    provider = ScriptedProvider(rounds=[[TextDelta("the SECRET plan"), Done()]])
    stored: list[Message] = []
    agent = Agent(
        provider=provider,
        hooks=[
            FnHook(on_text_delta=lambda d: d.replace("SECRET", "[redacted]")),
            FnHook(on_model_response=stored.append),
        ],
    )
    events = await _run_turn_collect(agent, "q")
    assert _texts(events) == "the [redacted] plan"  # live stream transformed
    assert stored[0].text == "the [redacted] plan"  # storage consistent


async def test_prompt_rejected_blocks_without_running_a_turn():
    provider = ScriptedProvider(rounds=[[TextDelta("never"), Done()]])
    recorder = RecorderHook()

    def reject(text: str) -> str:
        raise PromptRejected("policy")

    agent = Agent(provider=provider, hooks=[FnHook(user_prompt_submit=reject), recorder])
    events = await _run_turn_collect(agent, "bad prompt")
    assert _terminal(events) is StopReason.PROMPT_REJECTED
    assert any(isinstance(e, ev.Error) and "policy" in e.message for e in events)
    assert provider.calls == []  # model never called
    assert "turn_start" not in recorder.log  # no turn ran
    assert "turn_complete:prompt_rejected" not in recorder.log  # hook not fired


# ── tool-call degenerate (full dispatch = next commit) ─────────────────


async def test_known_tool_executes_and_result_feeds_next_round():
    provider = ScriptedProvider(
        rounds=[
            [ToolCallEvent(ToolCall("c1", "echo", '{"text": "hi"}')), Done()],
            [TextDelta("used the echo"), Done()],
        ]
    )
    agent = Agent(provider=provider, tools={"echo": EchoTool()})
    events = await _run_turn_collect(agent, "q")
    results = [e.result for e in events if isinstance(e, ev.ToolResultEvent)]
    assert len(results) == 1 and results[0].content == 'echo: {"text": "hi"}'
    assert _terminal(events) is StopReason.STOPPED
    # round 2 saw the tool result message paired to the call
    roles = [r for r, _ in provider.received_texts(1)]
    assert roles == ["user", "assistant", "tool"]


async def test_unknown_tool_yields_error_result_model_can_react_to():
    provider = ScriptedProvider(
        rounds=[
            [ToolCallEvent(ToolCall("c1", "ghost", "{}")), Done()],
            [TextDelta("ok, no such tool"), Done()],
        ]
    )
    agent = Agent(provider=provider)
    events = await _run_turn_collect(agent, "q")
    results = [e.result for e in events if isinstance(e, ev.ToolResultEvent)]
    assert results[0].is_error and "unknown or unmounted tool" in results[0].content
    assert _terminal(events) is StopReason.STOPPED


# ── provider failure (retry tiers = later commit) ──────────────────────


async def test_mid_stream_error_event_fails_turn_cleanly():
    provider = ScriptedProvider(
        rounds=[
            [
                TextDelta("partial"),
                ErrorEvent(ProviderError(message="upstream 502", http_status=502)),
            ],
        ]
    )
    agent = Agent(provider=provider)
    events = await _run_turn_collect(agent, "q")
    assert _terminal(events) is StopReason.PROVIDER_ERROR
    errors = [e for e in events if isinstance(e, ev.Error)]
    assert errors and errors[0].http_status == 502


async def test_failed_open_fails_turn_cleanly():
    provider = ScriptedProvider(rounds=[ProviderError(message="auth", http_status=401)])
    agent = Agent(provider=provider)
    events = await _run_turn_collect(agent, "q")
    assert _terminal(events) is StopReason.PROVIDER_ERROR


# ── continuation seam ──────────────────────────────────────────────────


async def test_continue_once_injects_synthetic_user_and_reruns():
    provider = ScriptedProvider(
        rounds=[
            [TextDelta("draft"), Done()],
            [TextDelta("final"), Done()],
        ]
    )
    agent = Agent(provider=provider, hooks=[ContinueOnceHook("keep going")])
    events = await _run_turn_collect(agent, "q")
    assert _terminal(events) is StopReason.STOPPED
    assert len(provider.calls) == 2
    # the injected continuation is a synthetic user message the model saw
    round2 = provider.received_texts(1)
    assert ("user", "keep going") in round2


# ── snapshot / resume ──────────────────────────────────────────────────


async def test_request_snapshot_returns_live_conversation():
    provider = ScriptedProvider(rounds=[[TextDelta("a"), Done()]])
    agent = Agent(provider=provider, persona="p")
    handle = agent.spawn()
    await handle.commands.put(ev.SendMessage(text="q"))
    while not isinstance(await handle.events.get(), ev.TurnComplete):
        pass
    await handle.commands.put(ev.RequestSnapshot())
    snap_event = await handle.events.get()
    assert isinstance(snap_event, ev.Snapshot)
    snap = snap_event.snapshot
    assert snap.turn_counter == 1 and snap.request_counter == 1
    assert [m.role for m in snap.messages] == [Role.SYSTEM, Role.USER, Role.ASSISTANT]
    await handle.commands.put(ev.Shutdown())
    await handle.task


async def test_resume_continues_ids_and_never_reinjects_persona():
    snap = SessionSnapshot(
        version=1,
        messages=[Message.system("p"), Message.user("old"), Message.assistant("done")],
        turn_counter=4,
        request_counter=9,
    )
    provider = ScriptedProvider(rounds=[[TextDelta("resumed"), Done()]])
    stored: list[Message] = []
    agent = Agent(
        provider=provider,
        persona="p",
        resume=snap,
        hooks=[FnHook(on_model_response=stored.append)],
    )
    events = await _run_turn_collect(agent, "new q")
    assert _terminal(events) is StopReason.STOPPED
    # persona appears ONCE (from the snapshot), and ids continue past the marks
    roles = [r for r, _ in provider.received_texts(0)]
    assert roles.count("system") == 1
    meta = stored[0].meta
    assert meta is not None
    assert meta.turn_id == 5 and meta.request_id == 10


async def test_unsupported_snapshot_version_degrades_to_fresh_start():
    snap = SessionSnapshot(version=99, messages=[Message.user("lost")])
    provider = ScriptedProvider(rounds=[[TextDelta("fresh"), Done()]])
    agent = Agent(provider=provider, persona="p", resume=snap)
    handle = agent.spawn()
    await handle.commands.put(ev.SendMessage(text="q"))
    events = []
    while True:
        event = await handle.events.get()
        events.append(event)
        if isinstance(event, ev.TurnComplete):
            break
    await handle.commands.put(ev.Shutdown())
    await handle.task
    assert any(isinstance(e, ev.Warning) and "unsupported snapshot" in e.message for e in events)
    assert provider.received_texts(0) == [("system", "p"), ("user", "q")]


# ── one-shot adapter ───────────────────────────────────────────────────


async def test_run_to_completion_aggregates_outcome():
    provider = ScriptedProvider(rounds=[[TextDelta("the answer"), Done()]])
    outcome = await Agent(provider=provider).run_to_completion("q")
    assert outcome.text == "the answer"
    assert outcome.stop is StopReason.STOPPED and outcome.error is None


async def test_run_to_completion_carries_failure():
    provider = ScriptedProvider(rounds=[ProviderError(message="boom", http_status=500)])
    outcome = await Agent(provider=provider).run_to_completion("q")
    assert outcome.stop is StopReason.PROVIDER_ERROR
    assert outcome.error == "boom" and outcome.http_status == 500
