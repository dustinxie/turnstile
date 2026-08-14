"""Engine mock suite — cancel + steer (commit 10 scope): cancel-as-undo vs
keep-interrupted-context, cancel checkpoints at every await, steer folding."""

import asyncio

import pytest

from turnstile.kernel import events as ev
from turnstile.kernel.dtos import (
    BeforeOutcome,
    Done,
    Gate,
    ProviderError,
    Role,
    StopReason,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolContext,
    ToolResult,
)
from turnstile.kernel.engine import INTERRUPTION_MARKER, Agent
from turnstile.kernel.ports import LifecycleHooks, Tool
from turnstile.kernel.testkit import (
    AsyncFnMiddleware,
    BlockUntilCancelTool,
    CommandInjectorTool,
    EchoTool,
    ScriptedProvider,
    StepClock,
)

pytestmark = pytest.mark.unit


def _tool_round(*calls: ToolCall) -> list:
    return [*map(ToolCallEvent, calls), Done()]


async def _drive(handle, *, send=None, cancel_after=None, until_terminals=1):
    """Pump the event queue; optionally send a command once `cancel_after`
    (an event type) is observed. Collects through N TurnComplete events."""
    events = []
    fired = False
    terminals = 0
    while terminals < until_terminals:
        event = await asyncio.wait_for(handle.events.get(), timeout=5)
        events.append(event)
        if not fired and cancel_after is not None and isinstance(event, cancel_after):
            await handle.commands.put(send)
            fired = True
        if isinstance(event, ev.TurnComplete):
            terminals += 1
    return events


async def _snapshot(handle):
    await handle.commands.put(ev.RequestSnapshot())
    while True:
        event = await asyncio.wait_for(handle.events.get(), timeout=5)
        if isinstance(event, ev.Snapshot):
            return event.snapshot


async def _shutdown(handle):
    await handle.commands.put(ev.Shutdown())
    await handle.task


class _HangTool(Tool):
    """Ignores the cancel token entirely — exercises the drop-as-backstop race."""

    def name(self) -> str:
        return "hang"

    def description(self) -> str:
        return ""

    def parameters_schema(self) -> dict:
        return {}

    async def execute(self, args: str, ctx: ToolContext) -> ToolResult:
        await asyncio.Event().wait()  # never returns, never polls cancel
        return ToolResult(call_id="", content="unreachable")


# ── cancel semantics ───────────────────────────────────────────────────


async def test_cancel_mid_tool_default_is_undo():
    provider = ScriptedProvider(rounds=[_tool_round(ToolCall("c1", "block", "{}"))])
    agent = Agent(provider=provider, tools={"block": BlockUntilCancelTool()}, persona="p")
    handle = agent.spawn()
    await handle.commands.put(ev.SendMessage(text="q"))
    events = await _drive(handle, send=ev.Cancel(), cancel_after=ev.ToolStarted)
    assert any(isinstance(e, ev.Cancelled) for e in events)
    assert events[-1].reason is StopReason.CANCELLED
    snap = await _snapshot(handle)
    await _shutdown(handle)
    # UNDO: prompt + partial work left no trace — only the persona survives
    assert [(m.role, m.text) for m in snap.messages] == [(Role.SYSTEM, "p")]


async def test_cancel_keep_interrupted_context_preserves_partial():
    provider = ScriptedProvider(rounds=[_tool_round(ToolCall("c1", "block", "{}"))])
    agent = Agent(
        provider=provider,
        tools={"block": BlockUntilCancelTool()},
        keep_interrupted_context=True,
    )
    handle = agent.spawn()
    await handle.commands.put(ev.SendMessage(text="q"))
    events = await _drive(handle, send=ev.Cancel(), cancel_after=ev.ToolStarted)
    assert events[-1].reason is StopReason.CANCELLED
    snap = await _snapshot(handle)
    await _shutdown(handle)
    texts = [m.text for m in snap.messages]
    assert "q" in texts  # the user's message survives
    assert INTERRUPTION_MARKER in texts  # the model is told what happened
    # every tool call is paired (cooperative tool answered on cancel)
    call_ids = {c.id for m in snap.messages for c in m.tool_calls}
    result_ids = {m.tool_call_id for m in snap.messages if m.tool_call_id}
    assert call_ids <= result_ids


async def test_cancel_drops_a_hung_tool_as_backstop():
    provider = ScriptedProvider(rounds=[_tool_round(ToolCall("c1", "hang", "{}"))])
    agent = Agent(provider=provider, tools={"hang": _HangTool()}, keep_interrupted_context=True)
    handle = agent.spawn()
    await handle.commands.put(ev.SendMessage(text="q"))
    events = await _drive(handle, send=ev.Cancel(), cancel_after=ev.ToolStarted)
    assert events[-1].reason is StopReason.CANCELLED
    results = [e.result for e in events if isinstance(e, ev.ToolResultEvent)]
    assert results and "side effects unknown" in results[0].content
    await _shutdown(handle)


async def test_cancel_interrupts_retry_backoff():
    provider = ScriptedProvider(
        rounds=[ProviderError(message="502", retryable=True, http_status=502)] * 4
    )
    # Real 3s backoff (scale 1) — the raced cancel must return long before it.
    agent = Agent(provider=provider, backoff_scale=1.0, clock=StepClock())
    handle = agent.spawn()
    await handle.commands.put(ev.SendMessage(text="q"))
    events = await asyncio.wait_for(
        _drive(handle, send=ev.Cancel(), cancel_after=ev.Warning), timeout=2
    )
    assert events[-1].reason is StopReason.CANCELLED
    await _shutdown(handle)


async def test_cancel_unblocks_parked_approval():
    async def approval(call, tool, rt):
        answer = await rt.request("approval", {"tool": tool.name()})
        if answer and answer.get("decision") == "allow":
            return BeforeOutcome.PROCEED
        return BeforeOutcome(Gate.DENY, "no answer — denied")

    provider = ScriptedProvider(rounds=[_tool_round(ToolCall("c1", "echo", "{}"))])
    agent = Agent(
        provider=provider,
        tools={"echo": EchoTool()},
        middleware=[AsyncFnMiddleware(before=approval)],
        keep_interrupted_context=True,
    )
    handle = agent.spawn()
    await handle.commands.put(ev.SendMessage(text="q"))
    events = await _drive(handle, send=ev.Cancel(), cancel_after=ev.Request)
    assert events[-1].reason is StopReason.CANCELLED  # never parked
    results = [e.result for e in events if isinstance(e, ev.ToolResultEvent)]
    assert results and "denied" in results[0].content  # fail-closed deny applied
    await _shutdown(handle)


async def test_cancel_rollback_covers_prepended_context():
    provider = ScriptedProvider(rounds=[_tool_round(ToolCall("c1", "block", "{}"))])
    agent = Agent(provider=provider, tools={"block": BlockUntilCancelTool()}, persona="p")
    handle = agent.spawn()
    await handle.commands.put(ev.SendMessageWithContext(text="q", context="recovery context"))
    events = await _drive(handle, send=ev.Cancel(), cancel_after=ev.ToolStarted)
    assert events[-1].reason is StopReason.CANCELLED
    snap = await _snapshot(handle)
    await _shutdown(handle)
    assert [m.text for m in snap.messages] == ["p"]  # context rolled back too


# ── steer ──────────────────────────────────────────────────────────────


async def test_steer_folds_into_running_turn():
    slot: list = []
    injector = CommandInjectorTool(slot, ev.SendMessage(text="also check the logs"))
    provider = ScriptedProvider(
        rounds=[
            _tool_round(ToolCall("c1", "inject", "{}")),
            [TextDelta("answered both"), Done()],
        ]
    )
    agent = Agent(provider=provider, tools={"inject": injector})
    handle = agent.spawn()
    slot.append(handle.commands)
    await handle.commands.put(ev.SendMessage(text="q"))
    events = await _drive(handle)  # ONE terminal: no second turn was opened
    await _shutdown(handle)
    steered = [e for e in events if isinstance(e, ev.Steered)]
    assert len(steered) == 1 and steered[0].count == 1
    assert steered[0].inputs[0].text == "also check the logs"
    # the steered prompt reached the model inside the SAME turn
    assert ("user", "also check the logs") in provider.received_texts(1)
    assert events[-1].reason is StopReason.STOPPED


async def test_steer_keeps_text_only_turn_alive():
    class SteerOnResponse(LifecycleHooks):
        def __init__(self):
            self.slot: list = []
            self.fired = False

        async def on_model_response(self, response):
            if not self.fired and self.slot:
                self.fired = True
                self.slot[0].put_nowait(ev.SendMessage(text="one more thing"))
                for _ in range(16):  # yield so the select drains the command
                    await asyncio.sleep(0)

    hook = SteerOnResponse()
    provider = ScriptedProvider(
        rounds=[
            [TextDelta("first answer"), Done()],
            [TextDelta("and the follow-up"), Done()],
        ]
    )
    agent = Agent(provider=provider, hooks=[hook])
    handle = agent.spawn()
    hook.slot.append(handle.commands)
    await handle.commands.put(ev.SendMessage(text="q"))
    events = await _drive(handle)  # one terminal — the turn stayed alive
    await _shutdown(handle)
    assert len(provider.calls) == 2
    assert any(isinstance(e, ev.Steered) for e in events)
    assert ("user", "one more thing") in provider.received_texts(1)
    assert events[-1].reason is StopReason.STOPPED


async def test_midturn_synthetic_queues_as_its_own_turn():
    slot: list = []
    injector = CommandInjectorTool(slot, ev.SendSyntheticMessage(text="auto follow-up"))
    provider = ScriptedProvider(
        rounds=[
            _tool_round(ToolCall("c1", "inject", "{}")),
            [TextDelta("turn one done"), Done()],
            [TextDelta("turn two done"), Done()],
        ]
    )
    agent = Agent(provider=provider, tools={"inject": injector})
    handle = agent.spawn()
    slot.append(handle.commands)
    await handle.commands.put(ev.SendMessage(text="q"))
    events = await _drive(handle, until_terminals=2)  # a SECOND turn ran
    await _shutdown(handle)
    assert [e.reason for e in events if isinstance(e, ev.TurnComplete)] == [
        StopReason.STOPPED,
        StopReason.STOPPED,
    ]
    assert not any(isinstance(e, ev.Steered) for e in events)  # queued, not folded
    # the synthetic prompt opened turn 2's wire
    assert ("user", "auto follow-up") in provider.received_texts(2)


async def test_midturn_shutdown_cancels_turn_and_ends_session():
    provider = ScriptedProvider(rounds=[_tool_round(ToolCall("c1", "block", "{}"))])
    agent = Agent(provider=provider, tools={"block": BlockUntilCancelTool()})
    handle = agent.spawn()
    await handle.commands.put(ev.SendMessage(text="q"))
    events = await _drive(handle, send=ev.Shutdown(), cancel_after=ev.ToolStarted)
    assert events[-1].reason is StopReason.CANCELLED
    await asyncio.wait_for(handle.task, timeout=5)  # session tore down cleanly
    # final snapshot emitted at session end
    snapshots = []
    while not handle.events.empty():
        event = handle.events.get_nowait()
        if isinstance(event, ev.Snapshot):
            snapshots.append(event)
    assert snapshots
