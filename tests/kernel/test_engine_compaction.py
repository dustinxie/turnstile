"""Engine mock suite — compaction + snapshot/resume (commit 11 scope):
apply_plan invariants, triggers (manual / auto / overflow / pre-send),
checkpoint gating, epoch behavior."""

import asyncio

import pytest

from turnstile.kernel import events as ev
from turnstile.kernel.dtos import (
    CompactionPlan,
    Conversation,
    Done,
    ManualTrigger,
    Message,
    MessageMeta,
    OverflowTrigger,
    ProviderError,
    StopReason,
    TextDelta,
    TokenUsage,
    ToolCall,
)
from turnstile.kernel.engine import Agent
from turnstile.kernel.testkit import (
    MemoryCheckpoint,
    ScriptedProvider,
    SummarizeOldestStrategy,
)

pytestmark = pytest.mark.unit


def _convo(*messages: Message, epoch: int = 0) -> Conversation:
    return Conversation(messages=list(messages), cache_epoch=epoch)


def _filled(n: int = 6) -> Conversation:
    convo = _convo(Message.system("persona"), Message.user("the real ask"))
    for i in range(n):
        convo.push(Message.assistant(f"answer {i} " + "x" * 50))
    return convo


# ── apply_plan invariants (pure dtos) ──────────────────────────────────


def test_drain_replaced_with_summary_bumps_epoch():
    convo = _filled()
    floor = convo.sacred_floor()
    report = convo.apply_plan(CompactionPlan(drain_from=floor, drain_to=6, summary="[sum]"), floor)
    assert report.committed and report.epoch_after == 1
    assert convo.cache_epoch == 1
    assert convo.messages[floor].text == "[sum]" and convo.messages[floor].synthetic
    assert report.bytes_after < report.bytes_before
    assert report.removed == 3  # 4 drained, 1 summary inserted


def test_sacred_floor_clamps_the_drain():
    convo = _filled()
    floor = convo.sacred_floor()
    convo.apply_plan(CompactionPlan(drain_from=0, drain_to=6, summary="[s]"), floor)
    # persona + the first real user prompt survive every compaction
    assert convo.messages[0].text == "persona"
    assert convo.messages[1].text == "the real ask"


def test_net_loss_guard_refuses_growth_and_noop():
    convo = _filled(2)
    floor = convo.sacred_floor()
    before = [m.text for m in convo.messages]
    grow = convo.apply_plan(
        CompactionPlan(drain_from=floor, drain_to=floor + 1, summary="Z" * 5000), floor
    )
    assert not grow.committed and convo.cache_epoch == 0
    assert [m.text for m in convo.messages] == before  # byte-identical
    noop = convo.apply_plan(CompactionPlan(), floor)
    assert not noop.committed and convo.cache_epoch == 0  # no epoch burn


def test_rewrites_translate_original_indices():
    convo = _filled(4)  # indices: 0 sys, 1 user, 2..5 assistants
    floor = convo.sacred_floor()
    report = convo.apply_plan(
        CompactionPlan(
            drain_from=2,
            drain_to=4,
            summary="[s]",
            rewrites=[
                (0, "hacked persona"),  # sacred -> skipped
                (3, "gone"),  # inside drain -> skipped
                (5, "[stubbed tool dump]"),  # survivor -> translated past drain+summary
                (99, "oob"),  # out of range -> skipped
            ],
        ),
        floor,
    )
    assert report.committed
    assert convo.messages[0].text == "persona"  # sacred rewrite skipped
    texts = [m.text for m in convo.messages]
    assert "[stubbed tool dump]" in texts and "gone" not in texts


def test_resume_note_appended_as_synthetic():
    convo = _filled()
    floor = convo.sacred_floor()
    convo.apply_plan(
        CompactionPlan(drain_from=floor, drain_to=7, summary="[s]", resume_note="[resume here]"),
        floor,
    )
    assert convo.messages[-1].text == "[resume here]" and convo.messages[-1].synthetic


def test_drain_splitting_a_tool_pair_is_repaired():
    convo = _convo(
        Message.user("q"),
        Message.assistant("filler " + "y" * 200),
        Message.assistant("", [ToolCall("c1", "t", "{}")]),
        Message.tool_result("c1", "result"),
    )
    floor = convo.sacred_floor()
    # Drain the assistant carrying the call but keep its result -> orphan result
    report = convo.apply_plan(CompactionPlan(drain_from=1, drain_to=3, summary="[s]"), floor)
    assert report.committed
    # the orphan tool result was dropped by the kernel's pairing repair
    assert not any(m.tool_call_id for m in convo.messages)


def test_pressure_relief_scales_surviving_meta():
    convo = _filled(4)
    tail = convo.messages[-1]
    tail.meta = MessageMeta(
        tokens=TokenUsage(), elapsed_ms=1, used_tokens=100_000, utilization=0.8
    )
    floor = convo.sacred_floor()
    report = convo.apply_plan(CompactionPlan(drain_from=floor, drain_to=5, summary="[s]"), floor)
    assert report.committed
    relieved = convo.messages[-1].meta
    assert relieved is not None
    assert relieved.used_tokens < 100_000  # scaled by the shrink ratio
    assert relieved.utilization < 0.8


# ── engine triggers ────────────────────────────────────────────────────


async def _collect(agent: Agent, text: str = "q") -> list:
    handle = agent.spawn()
    await handle.commands.put(ev.SendMessage(text=text))
    events = []
    while True:
        event = await asyncio.wait_for(handle.events.get(), timeout=5)
        events.append(event)
        if isinstance(event, ev.TurnComplete):
            break
    await handle.commands.put(ev.Shutdown())
    await handle.task
    return events


async def test_manual_compact_commits_through_checkpoint():
    checkpoint = MemoryCheckpoint()
    provider = ScriptedProvider(rounds=[[TextDelta("long answer " * 30), Done()]] * 2)
    agent = Agent(
        provider=provider,
        compaction=SummarizeOldestStrategy(keep_recent=1),
        checkpoint=checkpoint,
    )
    handle = agent.spawn()
    for text in ("q1", "q2"):  # build drainable history across two turns
        await handle.commands.put(ev.SendMessage(text=text))
        while not isinstance(await asyncio.wait_for(handle.events.get(), 5), ev.TurnComplete):
            pass
    await handle.commands.put(ev.Compact(focus=None))
    started = compacted = None
    while compacted is None:
        event = await asyncio.wait_for(handle.events.get(), 5)
        if isinstance(event, ev.CompactionStarted):
            started = event
        if isinstance(event, ev.Compacted):
            compacted = event
    await handle.commands.put(ev.Shutdown())
    await handle.task
    assert started is not None  # will_summarize gated the progress line
    assert compacted.committed and compacted.epoch == 1
    assert isinstance(compacted.trigger, ManualTrigger)
    assert compacted.snapshot is not None  # exact post-compaction working set
    assert checkpoint.saved and checkpoint.saved[0].cache_epoch == 1


async def test_checkpoint_failure_refuses_the_commit():
    provider = ScriptedProvider(rounds=[[TextDelta("long answer " * 30), Done()]] * 2)
    agent = Agent(
        provider=provider,
        compaction=SummarizeOldestStrategy(keep_recent=1),
        checkpoint=MemoryCheckpoint(fail_with="disk full"),
    )
    handle = agent.spawn()
    for text in ("q1", "q2"):  # two turns: drainable history past the sacred floor
        await handle.commands.put(ev.SendMessage(text=text))
        while not isinstance(await asyncio.wait_for(handle.events.get(), 5), ev.TurnComplete):
            pass
    await handle.commands.put(ev.Compact(focus=None))
    failed = None
    while failed is None:
        event = await asyncio.wait_for(handle.events.get(), 5)
        if isinstance(event, ev.CompactionFailed):
            failed = event
        assert not isinstance(event, ev.Compacted)  # never committed
    snap = None
    await handle.commands.put(ev.RequestSnapshot())
    while snap is None:
        event = await asyncio.wait_for(handle.events.get(), 5)
        if isinstance(event, ev.Snapshot):
            snap = event.snapshot
    await handle.commands.put(ev.Shutdown())
    await handle.task
    assert "disk full" in failed.error
    assert snap.cache_epoch == 0  # live conversation and epoch untouched


async def test_noop_strategy_manual_compact_is_refused_without_progress_line():
    provider = ScriptedProvider(rounds=[[TextDelta("a"), Done()]])
    agent = Agent(provider=provider)  # NoCompaction default
    handle = agent.spawn()
    await handle.commands.put(ev.SendMessage(text="q"))
    while not isinstance(await asyncio.wait_for(handle.events.get(), 5), ev.TurnComplete):
        pass
    await handle.commands.put(ev.Compact(focus=None))
    compacted = None
    events = []
    while compacted is None:
        event = await asyncio.wait_for(handle.events.get(), 5)
        events.append(event)
        if isinstance(event, ev.Compacted):
            compacted = event
    await handle.commands.put(ev.Shutdown())
    await handle.task
    assert not compacted.committed and compacted.epoch == 0
    assert not any(isinstance(e, ev.CompactionStarted) for e in events)


async def test_auto_task_boundary_compacts_before_the_next_turn():
    provider = ScriptedProvider(
        rounds=[[TextDelta("ok"), Done()], [TextDelta("two"), Done()]],
        ctx_window=100,  # tiny window: turn 1's PROMPT crosses any threshold
    )
    agent = Agent(
        provider=provider,
        compaction=SummarizeOldestStrategy(keep_recent=1),
        compact_threshold=0.5,
    )
    handle = agent.spawn()
    # Pressure = prompt (input) tokens: a big q1 makes turn 1's recorded
    # used_tokens cross the threshold, so q2's boundary compacts FIRST.
    await handle.commands.put(ev.SendMessage(text="Q" * 600))
    while not isinstance(await asyncio.wait_for(handle.events.get(), 5), ev.TurnComplete):
        pass
    await handle.commands.put(ev.SendMessage(text="q2"))
    order = []
    while True:
        event = await asyncio.wait_for(handle.events.get(), 5)
        order.append(type(event).__name__)
        if isinstance(event, ev.TurnComplete):
            break
    await handle.commands.put(ev.Shutdown())
    await handle.task
    # compaction ran at the task boundary: Compacted BEFORE TurnStarted
    assert "Compacted" in order and "TurnStarted" in order
    assert order.index("Compacted") < order.index("TurnStarted")


async def test_hard_overflow_recovery_compacts_and_retries():
    overflow = ProviderError(
        message="prompt is too long: 999 tokens > 100",
        http_status=400,
        code="context_length_exceeded",
    )
    provider = ScriptedProvider(
        rounds=[overflow, [TextDelta("fits now"), Done()]],
    )
    agent = Agent(
        provider=provider,
        compaction=SummarizeOldestStrategy(keep_recent=1),
        backoff_scale=0.0,
        resume=None,
    )
    # seed drainable history so the overflow compaction can shrink something
    handle = agent.spawn()
    await handle.commands.put(ev.SendMessage(text="q"))
    events = []
    while True:
        event = await asyncio.wait_for(handle.events.get(), 5)
        events.append(event)
        if isinstance(event, ev.TurnComplete):
            break
    await handle.commands.put(ev.Shutdown())
    await handle.task
    assert events[-1].reason is StopReason.STOPPED
    assert len(provider.calls) == 2  # rejected once, retried after compaction
    warnings = [e for e in events if isinstance(e, ev.Warning)]
    assert any("context overflow" in w.message for w in warnings)
    compactions = [e for e in events if isinstance(e, ev.Compacted)]
    assert compactions and isinstance(compactions[0].trigger, OverflowTrigger)


async def test_compacted_epoch_rides_snapshots_and_resume():
    provider = ScriptedProvider(rounds=[[TextDelta("answer " * 40), Done()]] * 3)
    agent = Agent(provider=provider, compaction=SummarizeOldestStrategy(keep_recent=1))
    handle = agent.spawn()
    for text in ("q1", "q2"):  # two turns: drainable history past the sacred floor
        await handle.commands.put(ev.SendMessage(text=text))
        while not isinstance(await asyncio.wait_for(handle.events.get(), 5), ev.TurnComplete):
            pass
    await handle.commands.put(ev.Compact(focus=None))
    snap = None
    await handle.commands.put(ev.RequestSnapshot())
    while snap is None:
        event = await asyncio.wait_for(handle.events.get(), 5)
        if isinstance(event, ev.Snapshot):
            snap = event.snapshot
    await handle.commands.put(ev.Shutdown())
    await handle.task
    assert snap.cache_epoch == 1  # committed compaction persisted its epoch
    # resume restores the epoch (prefix generation) losslessly
    resumed = Agent(
        provider=ScriptedProvider(rounds=[[TextDelta("resumed"), Done()]]),
        resume=snap,
    )
    events = await _collect(resumed, "next")
    assert events[-1].reason is StopReason.STOPPED
