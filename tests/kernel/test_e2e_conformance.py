"""M1 capstone — end-to-end conformance journeys on testkit doubles.

Three scenarios exercising the whole kernel together (the living documentation
of how the pieces compose):

- HAPPY: persona + parallel tool batch + dedup + collector middleware + delta
  redaction + a stateful quality judge (retry then verdict — the L2-collector
  pattern) across two turns, with the append-only prefix-cache audit over every
  provider call (the port of the reference's cache_prefix.rs invariant).
- HOSTILE: prompt rejection, DENY_TURN policy boundary, transient-retry
  recovery, the coarse repeat fuse, and one-shot Outcome failure perception.
- RESUME: committed compaction -> snapshot -> tampered dangling call -> resume
  repairs pairing, keeps the epoch, never re-injects persona, and continues the
  monotonic id sequence.
"""

import asyncio
from itertools import pairwise

import pytest

from turnstile.kernel import events as ev
from turnstile.kernel.dtos import (
    AfterOutcome,
    BeforeOutcome,
    Conversation,
    Done,
    Gate,
    Message,
    PromptRejected,
    ProviderError,
    Reasoning,
    Role,
    StopReason,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)
from turnstile.kernel.engine import (
    MAX_REPEAT_ROUNDS,
    REPEAT_LOOP_NUDGE,
    Agent,
)
from turnstile.kernel.ports import LifecycleHooks
from turnstile.kernel.testkit import (
    ConcurrencyProbeTool,
    ConcurrencyState,
    EchoTool,
    FnHook,
    FnMiddleware,
    MemoryCheckpoint,
    RecordedCall,
    ScriptedProvider,
    StepClock,
    SummarizeOldestStrategy,
)

pytestmark = pytest.mark.unit


async def _drain_turn(handle) -> list:
    events = []
    while True:
        event = await asyncio.wait_for(handle.events.get(), timeout=5)
        events.append(event)
        if isinstance(event, ev.TurnComplete):
            return events


async def _snapshot(handle):
    await handle.commands.put(ev.RequestSnapshot())
    while True:
        event = await asyncio.wait_for(handle.events.get(), timeout=5)
        if isinstance(event, ev.Snapshot):
            return event.snapshot


def _assert_append_only(calls: list[RecordedCall]) -> None:
    """THE prefix-cache invariant: every provider call's wire starts with the
    previous call's wire, byte for byte — within rounds AND across turns."""
    for earlier, later in pairwise(calls):
        prefix = later.messages[: len(earlier.messages)]
        assert prefix == earlier.messages, "wire prefix diverged — cache poisoned"


class _Judge(LifecycleHooks):
    """Quality judge: critiques the first draft (retry via offer_continuation),
    records its verdict on ITSELF when accepting — the L2-collector pattern."""

    def __init__(self) -> None:
        self.verdict: str | None = None
        self._critiqued = False

    async def offer_continuation(self, convo: Conversation) -> str | None:
        if not self._critiqued:
            self._critiqued = True
            return "Your draft lacks a citation. Cite the KB result and finalize."
        if self.verdict is None:
            self.verdict = "ok"
        return None


# ── the happy journey ──────────────────────────────────────────────────


async def test_happy_journey_two_turns_full_composition():
    state = ConcurrencyState()
    citations: list[str] = []
    judge = _Judge()
    provider = ScriptedProvider(
        rounds=[
            # turn 1, round 1: a parallel read-only batch
            [
                ToolCallEvent(ToolCall("a", "kb_a", "{}")),
                ToolCallEvent(ToolCall("b", "kb_b", "{}")),
                Done(),
            ],
            # round 2: a follow-up search whose result the collector harvests
            [ToolCallEvent(ToolCall("c", "echo", '{"q": "policy doc"}')), Done()],
            # round 3: a draft with thinking + a secret the redactor must scrub
            [Reasoning("checking sources"), TextDelta("Draft: the SECRET summary"), Done()],
            # round 4: the post-critique final answer
            [TextDelta("Final answer, with citation."), Done()],
            # turn 2: a quick follow-up
            [TextDelta("You're welcome."), Done()],
        ],
        ctx_window=100_000,
    )
    agent = Agent(
        provider=provider,
        persona="You are the support bot.",
        tools={
            "kb_a": ConcurrencyProbeTool(state, name="kb_a", read_only=True),
            "kb_b": ConcurrencyProbeTool(state, name="kb_b", read_only=True),
            "echo": EchoTool(),
        },
        hooks=[
            FnHook(on_text_delta=lambda d: d.replace("SECRET", "[redacted]")),
            judge,
        ],
        middleware=[
            FnMiddleware(after=lambda r: (citations.append(r.content), AfterOutcome.PROCEED)[1])
        ],
        clock=StepClock(),
        session_id="conformance-1",
    )
    handle = agent.spawn()

    await handle.commands.put(ev.SendMessage(text="What is our refund policy?"))
    turn1 = await _drain_turn(handle)
    await handle.commands.put(ev.SendMessage(text="thanks"))
    turn2 = await _drain_turn(handle)
    snap = await _snapshot(handle)
    await handle.commands.put(ev.Shutdown())
    await handle.task

    # terminals: exactly one clean stop per turn
    assert turn1[-1].reason is StopReason.STOPPED
    assert turn2[-1].reason is StopReason.STOPPED
    assert sum(isinstance(e, ev.TurnComplete) for e in turn1) == 1

    # the parallel batch really overlapped, with honest batch perception
    assert state.max_active == 2
    batches = [e for e in turn1 if isinstance(e, ev.ToolBatchStarted)]
    assert len(batches) == 1 and all(c.parallel_safe for c in batches[0].calls)
    completed = next(e for e in turn1 if isinstance(e, ev.ToolBatchCompleted))
    assert (completed.ok, completed.total) == (2, 2)

    # collector middleware harvested the search results (L2-collector pattern)
    assert any("policy doc" in c for c in citations)
    # judge retried the draft once, then recorded its verdict on itself
    assert judge.verdict == "ok"
    critique_wire = provider.received_texts(3)
    assert any("lacks a citation" in text for _, text in critique_wire)

    # redaction reached the live stream AND storage
    streamed = "".join(e.text for e in turn1 if isinstance(e, ev.TextDelta))
    assert "[redacted]" in streamed and "SECRET" not in streamed
    assert not any("SECRET" in m.text for m in snap.messages)

    # THE prefix-cache audit: append-only wire across all 5 calls, both turns
    assert len(provider.calls) == 5
    _assert_append_only(provider.calls)

    # correlation ids: request_ids strictly increase; turn_ids partition 1/2
    metas = [m.meta for m in snap.messages if m.meta is not None]
    request_ids = [m.request_id for m in metas]
    assert request_ids == sorted(request_ids) and len(set(request_ids)) == len(request_ids)
    assert {m.turn_id for m in metas} == {1, 2}
    assert all(m.session_id == "conformance-1" for m in metas)
    # snapshot high-water marks match the live counters
    assert snap.turn_counter == 2 and snap.request_counter == max(request_ids)


# ── the hostile journey ────────────────────────────────────────────────


async def test_hostile_journey_gates_and_fuses():
    def guard(text: str) -> str:
        if text.startswith("!"):
            raise PromptRejected("commands are not questions")
        return text

    def tenancy(call: ToolCall, tool) -> BeforeOutcome:
        if '"tenant": "other"' in call.arguments:
            return BeforeOutcome(Gate.DENY_TURN, "cross-tenant read")
        return BeforeOutcome.PROCEED

    provider = ScriptedProvider(
        rounds=[
            # turn 2 (turn 1 is rejected before any call): a cross-tenant call
            [ToolCallEvent(ToolCall("x", "echo", '{"tenant": "other"}')), Done()],
            # turn 3: a transient failure, then the model loops the same call
            ProviderError(message="502", retryable=True, http_status=502),
            *[
                [ToolCallEvent(ToolCall(f"r{i}", "echo", '{"same": 1}')), Done()]
                for i in range(MAX_REPEAT_ROUNDS + 2)
            ],
        ]
    )
    agent = Agent(
        provider=provider,
        tools={"echo": EchoTool()},
        hooks=[FnHook(user_prompt_submit=guard)],
        middleware=[FnMiddleware(before=tenancy)],
        backoff_scale=0.0,
    )
    handle = agent.spawn()

    await handle.commands.put(ev.SendMessage(text="!rm -rf /"))
    rejected = await _drain_turn(handle)
    assert rejected[-1].reason is StopReason.PROMPT_REJECTED
    assert provider.calls == []  # the model was never consulted

    await handle.commands.put(ev.SendMessage(text="read the other tenant's data"))
    denied = await _drain_turn(handle)
    assert denied[-1].reason is StopReason.POLICY_DENIED
    results = [e.result for e in denied if isinstance(e, ev.ToolResultEvent)]
    assert results and "cross-tenant read" in results[0].content  # paired, visible

    await handle.commands.put(ev.SendMessage(text="try again"))
    looped = await _drain_turn(handle)
    assert looped[-1].reason is StopReason.REPEAT_LOOP
    assert any(isinstance(e, ev.Warning) and "retrying" in e.message for e in looped)
    # the nudge reached the model before the fuse gave up
    nudged_wires = [
        call for call in provider.calls if any(m.text == REPEAT_LOOP_NUDGE for m in call.messages)
    ]
    assert nudged_wires
    snap = await _snapshot(handle)
    await handle.commands.put(ev.Shutdown())
    await handle.task
    # every tool call in stored history is paired (API-valid after all of it)
    call_ids = {c.id for m in snap.messages for c in m.tool_calls}
    result_ids = {m.tool_call_id for m in snap.messages if m.tool_call_id}
    assert call_ids <= result_ids

    # one-shot failure perception: a dead provider can't look like success
    outcome = await Agent(
        provider=ScriptedProvider(rounds=[ProviderError(message="down", http_status=503)]),
        backoff_scale=0.0,
    ).run_to_completion("q")
    assert outcome.stop is StopReason.PROVIDER_ERROR
    assert outcome.error == "down" and outcome.http_status == 503


# ── the resume journey ─────────────────────────────────────────────────


async def test_resume_journey_epoch_pairing_and_monotonic_ids():
    checkpoint = MemoryCheckpoint()
    provider = ScriptedProvider(rounds=[[TextDelta("answer " * 30), Done()]] * 3)
    agent = Agent(
        provider=provider,
        persona="persona",
        compaction=SummarizeOldestStrategy(keep_recent=1),
        checkpoint=checkpoint,
    )
    handle = agent.spawn()
    for text in ("q1", "q2"):
        await handle.commands.put(ev.SendMessage(text=text))
        await _drain_turn(handle)
    await handle.commands.put(ev.Compact(focus=None))
    snap = await _snapshot(handle)
    await handle.commands.put(ev.Shutdown())
    await handle.task
    assert snap.cache_epoch == 1  # the compaction committed through the checkpoint
    assert checkpoint.saved

    # Simulate a session that died mid-turn: a dangling assistant tool call.
    snap.messages.append(Message.assistant("", [ToolCall("dangle", "t", "{}")]))

    resumed_provider = ScriptedProvider(rounds=[[TextDelta("resumed"), Done()]])
    resumed = Agent(provider=resumed_provider, persona="persona", resume=snap)
    handle2 = resumed.spawn()
    await handle2.commands.put(ev.SendMessage(text="q3"))
    events = await _drain_turn(handle2)
    snap2 = await _snapshot(handle2)
    await handle2.commands.put(ev.Shutdown())
    await handle2.task

    assert events[-1].reason is StopReason.STOPPED
    wire = resumed_provider.calls[0].messages
    # persona exactly once (from the snapshot — never re-injected)
    assert [m.text for m in wire if m.role is Role.SYSTEM] == ["persona"]
    # the committed summary survived the resume
    assert any(m.synthetic and "[compressed history]" in m.text for m in wire)
    # the dangling call was repaired before the first resumed request
    dangling_results = [m for m in wire if m.tool_call_id == "dangle" and m.role is Role.TOOL]
    assert len(dangling_results) == 1 and dangling_results[0].is_error
    # epoch and id sequence continue monotonically across the resume boundary
    assert snap2.cache_epoch == 1
    assert snap2.turn_counter == snap.turn_counter + 1
    assert snap2.request_counter > snap.request_counter
