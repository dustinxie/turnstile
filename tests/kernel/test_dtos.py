"""L0 DTO behavior tests — pure data, no I/O, no engine."""

import pytest

from turnstile.kernel.dtos import (
    AfterOutcome,
    AutoTrigger,
    BeforeOutcome,
    CompactionPlan,
    Conversation,
    Gate,
    ImageContent,
    ManualTrigger,
    Message,
    MessageMeta,
    OverflowTrigger,
    ProviderError,
    Role,
    SessionSnapshot,
    StopReason,
    TokenUsage,
    ToolCall,
)

pytestmark = pytest.mark.unit


# ── TokenUsage.merge_max ───────────────────────────────────────────────


def test_merge_max_keeps_split_and_cumulative_fields():
    # Anthropic-style split: input early, cumulative output across later deltas.
    u = TokenUsage()
    u.merge_max(TokenUsage(prompt=100, completion=0, cached=10))  # message_start
    u.merge_max(TokenUsage(prompt=0, completion=20, cached=0))  # message_delta
    u.merge_max(TokenUsage(prompt=0, completion=50, cached=0))  # cumulative delta
    assert u == TokenUsage(prompt=100, completion=50, cached=10)

    # OpenAI-style single cumulative event: merge is a no-op equivalent.
    o = TokenUsage()
    o.merge_max(TokenUsage(prompt=200, completion=30, cached=5))
    assert o == TokenUsage(prompt=200, completion=30, cached=5)


# ── gate singletons ────────────────────────────────────────────────────


def test_gate_outcome_defaults():
    assert BeforeOutcome.PROCEED.gate is Gate.PROCEED
    assert BeforeOutcome().gate is Gate.PROCEED
    assert AfterOutcome.PROCEED.block_reason is None
    deny = BeforeOutcome(gate=Gate.DENY_TURN, reason="tenancy violation")
    assert deny.gate is Gate.DENY_TURN and deny.reason == "tenancy violation"


def test_outcome_singletons_are_poison_proof():
    # Outcomes are frozen: mutating the shared PROCEED must raise, so one
    # misbehaving middleware can't rewrite the singleton every caller shares.
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        BeforeOutcome.PROCEED.reason = "x"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        AfterOutcome.PROCEED.block_reason = "x"  # type: ignore[misc]


def test_stop_reason_has_all_terminal_causes():
    expected = {
        "stopped",
        "max_rounds",
        "max_continuations",
        "repeat_loop",
        "tool_loop_detected",
        "provider_error",
        "timeout",
        "cancelled",
        "prompt_rejected",
        "policy_denied",
        "rate_limited",
    }
    assert {r.value for r in StopReason} == expected


# ── ProviderError.is_context_overflow ──────────────────────────────────


def _err(http=None, code=None, msg=""):
    return ProviderError(message=msg, http_status=http, code=code)


def test_openai_code_is_overflow():
    assert _err(400, "context_length_exceeded", "...").is_context_overflow()


def test_anthropic_message_is_overflow():
    assert _err(400, None, "prompt is too long: 250000 tokens > 200000").is_context_overflow()


def test_gateway_shapes_are_overflow():
    assert _err(400, None, "Range of input length should be [1, 1000000].").is_context_overflow()
    assert _err(400, None, "too large for model with 8192 maximum context").is_context_overflow()


def test_auth_and_model_errors_are_not_overflow():
    assert not _err(401, "invalid_api_key", "bad key").is_context_overflow()
    assert not _err(400, "model_not_found", "no such model").is_context_overflow()
    assert not _err(429, None, "rate limited").is_context_overflow()
    # near-miss substring of the trailing-space needle must stay false
    assert not _err(
        400, None, "payload too large for model without streaming"
    ).is_context_overflow()


def test_provider_error_is_an_exception():
    with pytest.raises(ProviderError):
        raise _err(500, None, "boom")


# ── Message helpers ────────────────────────────────────────────────────


def test_constructors_set_roles_and_flags():
    assert Message.system("p").role is Role.SYSTEM
    assert Message.user("q").role is Role.USER
    assert Message.assistant("a").tool_calls == []
    tr = Message.tool_result("c1", "out", is_error=True)
    assert tr.role is Role.TOOL and tr.tool_call_id == "c1" and tr.is_error
    sy = Message.synthetic_user("nudge")
    assert sy.role is Role.USER and sy.synthetic


def test_estimate_tokens_heuristics():
    assert Message.user("x" * 400).estimate_tokens() == 104  # 400/4 + 4
    img = Message.user("hi", images=[ImageContent("image/png", "AA")])
    assert img.estimate_tokens() > 1600  # images dominate
    call = ToolCall(id="1", name="search", arguments='{"q": "y"}')
    with_call = Message.assistant("", [call])
    assert with_call.estimate_tokens() > Message.assistant("").estimate_tokens()


# ── Conversation invariants ────────────────────────────────────────────


def test_sacred_floor_covers_system_and_first_real_user():
    c = Conversation()
    c.push(Message.system("persona"))
    c.push(Message.synthetic_user("resume note"))  # NOT the anchor
    c.push(Message.user("the real ask"))
    c.push(Message.assistant("answer"))
    assert c.sacred_floor() == 3  # system + synthetic + real user, inclusive

    only_system = Conversation(messages=[Message.system("p")])
    assert only_system.sacred_floor() == 1
    assert Conversation().sacred_floor() == 0


def test_last_pressure_reads_latest_assistant_meta():
    c = Conversation()
    c.push(Message.user("hi"))
    assert c.last_pressure() == (0, 0, 0.0)
    a = Message.assistant("ans")
    a.meta = MessageMeta(
        tokens=TokenUsage(),
        elapsed_ms=5,
        ctx_window=128_000,
        used_tokens=40_000,
        utilization=0.3125,
    )
    c.push(a)
    assert c.last_pressure() == (128_000, 40_000, 0.3125)


def test_backfill_cancelled_pairs_dangling_calls():
    c = Conversation()
    calls = [ToolCall("a", "t", "{}"), ToolCall("b", "t", "{}")]
    c.push(Message.assistant("", calls))
    c.push(Message.tool_result("a", "done"))
    c.backfill_cancelled_tool_results()
    tail = c.messages[-1]
    assert tail.tool_call_id == "b" and tail.is_error and tail.text == "(cancelled)"
    assert len(c.messages) == 3  # append-only, no duplicates for "a"


def test_repair_pairing_drops_orphans_and_backfills_in_place():
    msgs = [
        Message.user("q"),
        Message.tool_result("ghost", "orphan"),  # no matching call -> dropped
        Message.assistant("", [ToolCall("a", "t", "{}"), ToolCall("b", "t", "{}")]),
        Message.tool_result("a", "ok"),
        Message.user("next"),  # closes the window before b resolved
    ]
    Conversation.repair_pairing(msgs)
    roles = [(m.role, m.tool_call_id) for m in msgs]
    assert roles == [
        (Role.USER, None),
        (Role.ASSISTANT, None),
        (Role.TOOL, "a"),
        (Role.TOOL, "b"),  # synthesized immediately after its window
        (Role.USER, None),
    ]
    assert msgs[3].text == "(cancelled)" and msgs[3].is_error


def test_repair_pairing_dedups_double_results_for_one_call():
    msgs = [
        Message.assistant("", [ToolCall("a", "t", "{}")]),
        Message.tool_result("a", "first"),
        Message.tool_result("a", "second"),  # illegal duplicate -> dropped
    ]
    Conversation.repair_pairing(msgs)
    assert [m.text for m in msgs if m.role is Role.TOOL] == ["first"]


# ── compaction plan / triggers ─────────────────────────────────────────


def test_compaction_plan_noop_detection():
    assert CompactionPlan().is_noop()
    assert not CompactionPlan(drain_from=0, drain_to=2).is_noop()
    assert not CompactionPlan(summary="s").is_noop()
    assert not CompactionPlan(rewrites=[(1, "stub")]).is_noop()
    assert not CompactionPlan(resume_note="n").is_noop()


def test_compact_triggers_carry_their_facts():
    assert AutoTrigger(utilization=0.9).utilization == 0.9
    assert ManualTrigger().focus is None
    assert OverflowTrigger(attempt=2).attempt == 2


# ── snapshot ───────────────────────────────────────────────────────────


def test_snapshot_derives_counter_high_water_marks():
    c = Conversation(cache_epoch=3)
    a1 = Message.assistant("x")
    a1.meta = MessageMeta(tokens=TokenUsage(), elapsed_ms=1, turn_id=2, request_id=5)
    a2 = Message.assistant("y")
    a2.meta = MessageMeta(tokens=TokenUsage(), elapsed_ms=1, turn_id=1, request_id=7)
    c.push(a1)
    c.push(a2)
    snap = SessionSnapshot.from_conversation(c)
    assert (snap.turn_counter, snap.request_counter) == (2, 7)
    assert snap.cache_epoch == 3 and snap.version == 1
    assert snap.messages is not c.messages  # a copy, not an alias
