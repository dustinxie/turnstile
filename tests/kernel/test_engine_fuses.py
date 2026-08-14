"""Engine mock suite — continuations + fuses (commit 7 scope): round cap +
interactive checkpoint, MAX_CONTINUATIONS, coarse REPEAT_LOOP, exact
ToolLoopPolicy, INTERNAL_CONTROL suppression."""

import pytest

from turnstile.kernel import events as ev
from turnstile.kernel.dtos import (
    Continuation,
    ContinuationKind,
    ContinuationVisibility,
    Conversation,
    Done,
    StopReason,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)
from turnstile.kernel.engine import (
    MAX_REPEAT_ROUNDS,
    REPEAT_LOOP_NUDGE,
    REPEAT_NUDGE_AT,
    Agent,
    ToolLoopPolicy,
)
from turnstile.kernel.ports import LifecycleHooks
from turnstile.kernel.testkit import (
    AlwaysContinueHook,
    AlwaysStopProvider,
    CountingTool,
    EchoTool,
    ScriptedProvider,
)

pytestmark = pytest.mark.unit


def _tool_round(*calls: ToolCall) -> list:
    return [*map(ToolCallEvent, calls), Done()]


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


def _same_call_rounds(n: int, name: str = "echo", args: str = "{}") -> list:
    return [_tool_round(ToolCall(f"c{i}", name, args)) for i in range(n)]


# ── round cap ──────────────────────────────────────────────────────────


async def test_max_rounds_fuse_trips():
    provider = ScriptedProvider(rounds=_same_call_rounds(10))
    agent = Agent(provider=provider, tools={"echo": EchoTool()}, max_rounds=3)
    events = await _collect(agent)
    assert events[-1].reason is StopReason.MAX_ROUNDS
    assert len(provider.calls) == 3  # fuse fired before a 4th model call
    assert any(isinstance(e, ev.Error) and "max rounds" in e.message for e in events)


async def test_round_cap_checkpoint_rearms_then_stops():
    provider = ScriptedProvider(rounds=_same_call_rounds(10, args='{"n": 1}'))
    agent = Agent(
        provider=provider,
        tools={"echo": EchoTool()},
        max_rounds=2,
        round_cap_checkpoint=True,
    )
    handle = agent.spawn()
    await handle.commands.put(ev.SendMessage(text="q"))
    answered = 0
    events = []
    while True:
        event = await handle.events.get()
        events.append(event)
        if isinstance(event, ev.Request) and event.kind == "round_cap_checkpoint":
            answered += 1
            cont = answered == 1  # continue once, then stop
            assert event.payload["cap"] in (2, 4) and event.payload["base"] == 2
            await handle.commands.put(ev.Respond(event.id, {"continue": cont}))
        if isinstance(event, ev.TurnComplete):
            break
    await handle.commands.put(ev.Shutdown())
    await handle.task
    assert answered == 2
    assert events[-1].reason is StopReason.MAX_ROUNDS
    assert len(provider.calls) == 4  # base 2, re-armed once -> 4 rounds ran


async def test_round_cap_checkpoint_fails_closed_without_answer():
    provider = ScriptedProvider(rounds=_same_call_rounds(5))
    agent = Agent(
        provider=provider,
        tools={"echo": EchoTool()},
        max_rounds=1,
        round_cap_checkpoint=True,
        request_timeout=0.01,
    )
    events = await _collect(agent)  # nobody answers the checkpoint Request
    assert events[-1].reason is StopReason.MAX_ROUNDS
    assert len(provider.calls) == 1


# ── continuation fuse + internal control ───────────────────────────────


async def test_max_continuations_fuse():
    agent = Agent(
        provider=AlwaysStopProvider("answer"),
        hooks=[AlwaysContinueHook()],
        max_continuations=2,
    )
    events = await _collect(agent)
    assert events[-1].reason is StopReason.MAX_CONTINUATIONS
    assert any(isinstance(e, ev.Error) and "continuations" in e.message for e in events)


async def test_internal_control_round_streams_nothing_and_stores_blank():
    class VerifyOnce(LifecycleHooks):
        def __init__(self):
            self.used = False

        async def offer_typed_continuation(self, convo: Conversation):
            if self.used:
                return None
            self.used = True
            return Continuation(
                text="verify your work",
                kind=ContinuationKind.VERIFY_CADENCE,
                visibility=ContinuationVisibility.INTERNAL_CONTROL,
            )

    provider = ScriptedProvider(
        rounds=[
            [TextDelta("visible answer"), Done()],
            [TextDelta("internal check chatter"), Done()],
        ]
    )
    agent = Agent(provider=provider, hooks=[VerifyOnce()])
    events = await _collect(agent)
    assert events[-1].reason is StopReason.STOPPED
    streamed = "".join(e.text for e in events if isinstance(e, ev.TextDelta))
    assert streamed == "visible answer"  # control round emitted nothing
    reasoned = [e for e in events if isinstance(e, ev.Reasoning)]
    assert not reasoned  # reasoning channel suppressed too


async def test_internal_control_stored_message_is_tagged_and_blank():
    class VerifyOnce(LifecycleHooks):
        def __init__(self):
            self.used = False

        async def offer_typed_continuation(self, convo: Conversation):
            if self.used:
                return None
            self.used = True
            return Continuation(
                text="verify",
                kind=ContinuationKind.VERIFY_CADENCE,
                visibility=ContinuationVisibility.INTERNAL_CONTROL,
            )

    provider = ScriptedProvider(
        rounds=[
            [TextDelta("answer"), Done()],
            [TextDelta("chatter"), Done()],
        ]
    )
    agent = Agent(provider=provider, hooks=[VerifyOnce()])
    handle = agent.spawn()
    await handle.commands.put(ev.SendMessage(text="q"))
    while not isinstance(await handle.events.get(), ev.TurnComplete):
        pass
    await handle.commands.put(ev.RequestSnapshot())
    snap = None
    while snap is None:
        event = await handle.events.get()
        if isinstance(event, ev.Snapshot):
            snap = event.snapshot
    await handle.commands.put(ev.Shutdown())
    await handle.task
    control = [m for m in snap.messages if m.internal_origin == "verify_cadence"]
    assert len(control) == 1 and control[0].text == ""


# ── coarse repeat fuse ─────────────────────────────────────────────────


async def test_repeat_loop_fuse_nudges_then_stops():
    provider = ScriptedProvider(rounds=_same_call_rounds(MAX_REPEAT_ROUNDS + 2))
    agent = Agent(provider=provider, tools={"echo": EchoTool()})
    events = await _collect(agent)
    assert events[-1].reason is StopReason.REPEAT_LOOP
    assert len(provider.calls) == MAX_REPEAT_ROUNDS
    # the nudge entered the wire after REPEAT_NUDGE_AT identical rounds
    nudge_round = provider.calls[REPEAT_NUDGE_AT].messages
    assert any(m.text == REPEAT_LOOP_NUDGE for m in nudge_round)


async def test_repeat_fuse_resets_when_pattern_changes():
    rounds = []
    for i in range(MAX_REPEAT_ROUNDS + 2):  # alternate args -> never 6 consecutive
        rounds.append(_tool_round(ToolCall(f"c{i}", "echo", f'{{"i": {i % 2}}}')))
    rounds.append([TextDelta("done"), Done()])
    agent = Agent(provider=ScriptedProvider(rounds=rounds), tools={"echo": EchoTool()})
    events = await _collect(agent)
    assert events[-1].reason is StopReason.STOPPED


# ── exact tool-loop guard ──────────────────────────────────────────────


def test_tool_loop_policy_validation():
    with pytest.raises(ValueError):
        ToolLoopPolicy(warning_threshold=1, stop_threshold=4)
    with pytest.raises(ValueError):
        ToolLoopPolicy(warning_threshold=4, stop_threshold=4)


async def test_exact_guard_warns_then_stops_on_no_progress():
    # EchoTool returns identical content for identical args -> no progress.
    provider = ScriptedProvider(rounds=_same_call_rounds(6))
    agent = Agent(
        provider=provider,
        tools={"echo": EchoTool()},
        tool_loop_policy=ToolLoopPolicy(warning_threshold=3, stop_threshold=4),
    )
    events = await _collect(agent)
    assert events[-1].reason is StopReason.TOOL_LOOP_DETECTED
    assert len(provider.calls) == 4  # stopped at the 4th identical observation
    warnings = [e for e in events if isinstance(e, ev.Warning)]
    assert any("possible tool loop" in w.message for w in warnings)
    assert any("tool loop detected" in w.message for w in warnings)


async def test_exact_guard_tolerates_changing_results():
    # CountingTool's content changes every call (count#N) -> progress observable.
    provider = ScriptedProvider(
        rounds=[
            *_same_call_rounds(4, name="count"),
            [TextDelta("done"), Done()],
        ]
    )
    agent = Agent(
        provider=provider,
        tools={"count": CountingTool()},
        tool_loop_policy=ToolLoopPolicy(warning_threshold=3, stop_threshold=4),
    )
    events = await _collect(agent)
    assert events[-1].reason is StopReason.STOPPED


async def test_exact_guard_resets_on_new_real_user_turn():
    # 3 identical no-progress rounds (streak = 3, warned) then a text stop; the
    # NEXT real user turn starts a fresh intent scope — one more identical round
    # must NOT trip the stop threshold of 4.
    provider = ScriptedProvider(
        rounds=[
            *_same_call_rounds(3),
            [TextDelta("t1 done"), Done()],
            _tool_round(ToolCall("n1", "echo", "{}")),
            [TextDelta("t2 done"), Done()],
        ]
    )
    agent = Agent(
        provider=provider,
        tools={"echo": EchoTool()},
        tool_loop_policy=ToolLoopPolicy(warning_threshold=3, stop_threshold=4),
    )
    handle = agent.spawn()
    terminals = []
    # Sequential sends: a SendMessage during a running turn would STEER into it
    # (commit 10) — this test needs two distinct real-user turns.
    for text in ("q1", "q2"):
        await handle.commands.put(ev.SendMessage(text=text))
        while True:
            event = await handle.events.get()
            if isinstance(event, ev.TurnComplete):
                terminals.append(event.reason)
                break
    await handle.commands.put(ev.Shutdown())
    await handle.task
    assert terminals == [StopReason.STOPPED, StopReason.STOPPED]
