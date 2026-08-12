"""L0 port contract tests — defaults, required methods, HookChain composition."""

import pytest

from turnstile.kernel.dtos import (
    AfterOutcome,
    BeforeOutcome,
    CompactionPlan,
    CompactionView,
    Continuation,
    ContinuationKind,
    Conversation,
    Gate,
    ManualTrigger,
    PromptRejected,
    RateLimitDecision,
    RateLimitHint,
    RiskLevel,
    ToolContext,
    ToolResult,
)
from turnstile.kernel.ports import (
    CompactionCheckpoint,
    CompactionCheckpointError,
    FixedClock,
    HookChain,
    LifecycleHooks,
    NoCompaction,
    Requester,
    SystemClock,
    Tool,
    ToolMiddleware,
)

pytestmark = pytest.mark.unit


# ── Tool defaults ──────────────────────────────────────────────────────


class _PlainTool(Tool):
    def name(self) -> str:
        return "plain"

    def description(self) -> str:
        return ""

    def parameters_schema(self) -> dict:
        return {"type": "object"}

    async def execute(self, args, ctx):
        return ToolResult(call_id="", content="ok")


class _ReadOnlyTool(_PlainTool):
    def read_only_hint(self) -> bool:
        return True


def test_parallel_safe_defaults_to_read_only_hint():
    assert not _PlainTool().parallel_safe("{}"), "default (no hint) is NOT parallel-safe"
    assert _ReadOnlyTool().parallel_safe("{}"), "a read-only tool IS parallel-safe"


def test_tool_advisory_defaults():
    t = _PlainTool()
    assert t.risk('{"cmd": "rm -rf /"}') is RiskLevel.SAFE  # advisory, conservative
    assert t.always_grant_scope('{"a": 1}') == '{"a": 1}'  # per-exact-args grant


def test_abstract_tool_cannot_instantiate():
    class Incomplete(Tool):  # no execute / name / ...
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


# ── LifecycleHooks defaults (the noop hook) ────────────────────────────


async def test_bare_hooks_are_noop():
    h = LifecycleHooks()  # no abstract methods -> instantiable = the noop
    assert await h.user_prompt_submit("hi") == "hi"
    assert await h.on_text_delta("chunk") == "chunk"
    assert await h.offer_continuation(Conversation()) is None
    assert (
        await h.on_rate_limit(
            RateLimitHint(http_status=429, retry_after_secs=None, terminal=False, attempt=1)
        )
        is None
    )


async def test_typed_continuation_wraps_legacy_text_as_generic():
    class Legacy(LifecycleHooks):
        async def offer_continuation(self, convo):
            return "continue"

    got = await Legacy().offer_typed_continuation(Conversation())
    assert got == Continuation(text="continue", kind=ContinuationKind.GENERIC)


# ── HookChain composition contract ─────────────────────────────────────


class _Recorder(LifecycleHooks):
    def __init__(self, tag: str, log: list[str], continuation: str | None = None):
        self.tag, self.log, self.continuation = tag, log, continuation

    async def user_prompt_submit(self, text: str) -> str:
        self.log.append(self.tag)
        return f"{text}+{self.tag}"

    async def on_text_delta(self, delta: str) -> str:
        return f"{delta}|{self.tag}"

    async def offer_continuation(self, convo):
        self.log.append(f"offer:{self.tag}")
        return self.continuation


async def test_chain_transforms_in_registration_order():
    log: list[str] = []
    chain = HookChain([_Recorder("a", log), _Recorder("b", log)])
    assert await chain.user_prompt_submit("q") == "q+a+b"  # later sees earlier's rewrite
    assert log == ["a", "b"]
    assert await chain.on_text_delta("x") == "x|a|b"


async def test_chain_short_circuits_on_first_block():
    log: list[str] = []

    class Blocker(LifecycleHooks):
        async def user_prompt_submit(self, text: str) -> str:
            raise PromptRejected("nope")

    chain = HookChain([_Recorder("a", log), Blocker(), _Recorder("c", log)])
    with pytest.raises(PromptRejected):
        await chain.user_prompt_submit("q")
    assert log == ["a"], "hooks after the block must not run"


async def test_chain_offer_continuation_all_observe_first_some_wins():
    log: list[str] = []
    chain = HookChain(
        [
            _Recorder("a", log, continuation=None),
            _Recorder("b", log, continuation="from-b"),
            _Recorder("c", log, continuation="from-c"),  # observed, but ignored
        ]
    )
    assert await chain.offer_continuation(Conversation()) == "from-b"
    assert log == ["offer:a", "offer:b", "offer:c"], "ALL hooks observe the turn end"


async def test_empty_chain_is_noop():
    chain = HookChain([])
    assert await chain.user_prompt_submit("hi") == "hi"
    assert await chain.offer_continuation(Conversation()) is None


async def test_chain_rate_limit_first_opinion_wins():
    class NoOpinion(LifecycleHooks):
        pass

    class Waits(LifecycleHooks):
        async def on_rate_limit(self, hint):
            return RateLimitDecision(wait_secs=5)

    chain = HookChain([NoOpinion(), Waits()])
    hint = RateLimitHint(http_status=429, retry_after_secs=None, terminal=False, attempt=1)
    decision = await chain.on_rate_limit(hint)
    assert decision is not None and decision.wait_secs == 5


# ── ToolMiddleware defaults ────────────────────────────────────────────


async def test_middleware_defaults_proceed():
    class Bare(ToolMiddleware):
        pass

    mw = Bare()
    from turnstile.kernel.dtos import ToolCall

    outcome = await mw.before(ToolCall("1", "t", "{}"), _PlainTool(), rt=None)
    assert outcome.gate is Gate.PROCEED
    assert outcome is BeforeOutcome.PROCEED
    assert await mw.after(ToolResult("1", "ok")) is AfterOutcome.PROCEED


# ── compaction ports ───────────────────────────────────────────────────


async def test_no_compaction_always_plans_noop():
    view = CompactionView(
        messages=[],
        trigger=ManualTrigger(),
        ctx_window=0,
        used_tokens=0,
        utilization=0.0,
        sacred_floor=0,
    )
    strategy = NoCompaction()
    assert not strategy.will_summarize(view)
    assert (await strategy.plan(view)) == CompactionPlan()


def test_checkpoint_error_gates_commit():
    class Failing(CompactionCheckpoint):
        def save(self, snapshot) -> None:
            raise CompactionCheckpointError("disk full")

    with pytest.raises(CompactionCheckpointError, match="disk full"):
        Failing().save(None)  # type: ignore[arg-type]


# ── clocks ─────────────────────────────────────────────────────────────


def test_fixed_clock_yields_zero_elapsed():
    c = FixedClock(42)
    assert c.now_millis() - c.now_millis() == 0


def test_system_clock_is_monotonic_nondecreasing():
    c = SystemClock()
    a = c.now_millis()
    b = c.now_millis()
    assert b >= a


# ── ToolContext.request degradation ────────────────────────────────────


async def test_tool_context_without_requester_returns_none():
    ctx = ToolContext(working_dir="/")
    assert await ctx.request("ask", {}) is None


async def test_tool_context_with_requester_forwards():
    class Echo(Requester):
        async def request(self, kind: str, payload: dict):
            return {"kind": kind, **payload}

    ctx = ToolContext(working_dir="/", requester=Echo())
    assert await ctx.request("approve", {"ok": 1}) == {"kind": "approve", "ok": 1}
