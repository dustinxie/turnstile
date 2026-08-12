"""L0 ports — the behavioral contract the loop calls through.

Seven interfaces + the Requester handle. L1 capabilities implement the I/O ports
(Tool, LlmProvider, CompactionStrategy, CompactionCheckpoint); L2 products
implement the discipline seams (LifecycleHooks, ToolMiddleware). The engine holds
ONLY these types — never a concrete class. Reference: docs/one-loop-two-layer-
agent.md Appendix; hook/middleware semantics: docs/kernel-loop-structure.md §I.2.

Two flavors: @abstractmethod = required (instantiation fails without it); a plain
body = default (override optional — a hook overriding nothing is a genuine no-op).
"""

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from turnstile.kernel.dtos import (
    AfterOutcome,
    BeforeOutcome,
    ChatOptions,
    CompactionPlan,
    CompactionView,
    Continuation,
    Conversation,
    Message,
    RateLimitDecision,
    RateLimitHint,
    RiskLevel,
    SessionSnapshot,
    StopReason,
    StreamEvent,
    ToolCall,
    ToolContext,
    ToolDef,
    ToolResult,
    TurnCtx,
)


class Tool(ABC):
    """A mounted, callable capability.

    TRUST MODEL (design doc §4): the kernel does NOT sandbox. Mounting a tool
    grants its execute() the host process's full ambient authority; OS-level
    isolation is the embedder's / an L1 capability's job. The kernel's one
    built-in bound is the tool-result size cap. A failing tool returns
    ToolResult(is_error=True); raised exceptions are caught into error results.
    """

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def parameters_schema(self) -> dict:
        """JSON schema shown to the model (the kernel derives ToolDef from it)."""

    @abstractmethod
    async def execute(self, args: str, ctx: ToolContext) -> ToolResult: ...

    # -- advisory / optional (arg-aware where it matters) -----------------

    def risk(self, args: str) -> RiskLevel:
        """Risk classification for THIS call — bash rates `rm -rf` RISKY and `ls`
        SAFE from its args. Advisory metadata only; the kernel never enforces."""
        return RiskLevel.SAFE

    def read_only_hint(self) -> bool:
        """Intrinsic 'no side effects' property. Default False (unknown)."""
        return False

    def parallel_safe(self, args: str) -> bool:
        """May THIS call run concurrently with others in the same batch?
        Side-effecting tools stay False and serialize behind the write-lock."""
        return self.read_only_hint()

    def always_grant_scope(self, args: str) -> str:
        """Scope string under which an 'Always' approval grant is remembered.
        Default = the exact args (each distinct call approved on its own — right
        for bash); a tool whose approval is meaningfully tool-wide returns a
        constant."""
        return args


class LlmProvider(ABC):
    """The model backend. The loop never names a vendor — it calls chat_stream
    once per round and consumes the event stream."""

    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        options: ChatOptions,
    ) -> AsyncIterator[StreamEvent]:
        """Open the stream for one round. Raises ProviderError on a FAILED OPEN
        (auth / connect / overflow / 429 — the loop branches on
        is_context_overflow / retryable / http_status); a stream that opened may
        still fail mid-flight via an ErrorEvent item."""

    def context_window(self) -> int:
        """Effective context window in tokens. 0 = unknown."""
        return 0

    def bind_session_id(self, session_id: str) -> None:  # noqa: B027 — deliberate no-op default
        """One-shot binding at spawn — an adapter forwards it as a gateway
        prefix-cache-affinity header. Default no-op: adapters that don't forward
        an affinity id, and test doubles, ignore it."""


class Requester(ABC):
    """Request-only handle into the driver Request/Respond round-trip, threaded
    into ToolContext so a plain tool can ask the driver a structured question.
    The concrete implementation rides on RequestCtx (events.py)."""

    @abstractmethod
    async def request(self, kind: str, payload: dict) -> Any:
        """Emit a Request and await the driver's Respond. Degrades to None when
        unanswered (timeout / cancel / dead driver) — callers proceed FAIL-CLOSED."""


class LifecycleHooks:
    """TURN-level seams — all 15 default no-op. Deliberately NOT abstract: every
    method has a working default, so LifecycleHooks() IS the noop hook and an
    implementation overrides only what it needs. Rule (design doc §4): gate,
    redact, observe — never strategy.

    Composition: register many; HookChain fans out in REGISTRATION ORDER
    (transforms chain; user_prompt_submit short-circuits on the first block;
    offer_continuation: all observe, first non-None wins).
    """

    async def session_start(self, convo: Conversation, resumed: bool) -> None:
        """Seed persona/context. PERMANENT. A seeding hook must early-return
        `if resumed` — the snapshot already carries its injection."""

    async def user_prompt_submit(self, text: str) -> str:
        """Rewrite/augment the prompt; raise PromptRejected(reason) to BLOCK
        (the message never enters history; no turn runs)."""
        return text

    async def turn_start(self, convo: Conversation) -> None:
        """Once per turn, before the first LLM call. Mutate history. PERMANENT."""

    async def pre_request(self, messages: list[Message], ctx: TurnCtx) -> None:
        """Mutate the per-request CLONE. EPHEMERAL; must be APPEND-ONLY at the
        tail — a non-append projection warns (prefix-cache poison)."""

    async def pre_request_options(
        self, messages: list[Message], options: ChatOptions, ctx: TurnCtx
    ) -> None:
        """Mutate this call's ChatOptions copy (EPHEMERAL sideband)."""

    async def on_request(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        options: ChatOptions,
        ctx: TurnCtx,
    ) -> None:
        """READ-ONLY observation of the exact final wire — telemetry / prefix-
        cache RCA home. Fires after pre_request projects, before the provider."""

    async def on_text_delta(self, delta: str) -> str:
        """Transform each streamed chunk BEFORE emit; post-hook bytes reach BOTH
        the live stream and storage (consistent redaction). Return "" to
        suppress. Cross-chunk redaction is the hook's own buffering job."""
        return delta

    async def on_reasoning_delta(self, delta: str) -> str:
        """Symmetric twin of on_text_delta for the thinking channel."""
        return delta

    async def on_model_response(self, response: Message) -> None:
        """Transform the STORED assistant message — including dropping/rewriting
        tool_calls, which the loop HONORS (a dropped call never executes).
        Post-stream: live bytes already went; meta is kernel-owned."""

    async def offer_continuation(self, convo: Conversation) -> str | None:
        """The model wants to stop. Return critique/follow-up text to inject a
        synthetic user message and CONTINUE (the quality-retry seam); None to
        accept. A judge records its verdict on ITS OWN state (L2-collector)."""
        return None

    async def offer_typed_continuation(self, convo: Conversation) -> Continuation | None:
        """Richer variant: kind + visibility (INTERNAL_CONTROL rounds stream
        nothing and store blank text). Default wraps offer_continuation."""
        text = await self.offer_continuation(convo)
        return Continuation(text) if text is not None else None

    async def turn_complete(self, convo: Conversation, reason: StopReason, ctx: TurnCtx) -> None:
        """EXACTLY ONCE per turn on EVERY terminal path (not on a blocked prompt
        — no turn ran). The seam for per-turn persistence / metrics."""

    async def on_error(self, error: str) -> None:
        """Observe a tool/provider error mid-turn (the turn may continue)."""

    async def on_rate_limit(self, hint: RateLimitHint) -> RateLimitDecision | None:
        """Verdict on a 429: wait-and-retry vs pause. None = no opinion — the
        kernel falls back to a conservative hint-derived decision."""
        return None

    async def session_end(self, convo: Conversation) -> None:
        """Session ends (any exit path). Read-only cleanup / telemetry."""


class HookChain(LifecycleHooks):
    """Composes MANY LifecycleHooks into one by fanning out each method over an
    ordered list — the seam that lets independent capabilities coexist. The
    engine holds exactly one hooks object either way.

    Contract (load-bearing): registration order everywhere; transforms CHAIN (a
    later hook sees the earlier rewrite); user_prompt_submit SHORT-CIRCUITS on
    the first block (PromptRejected propagates, later hooks don't run);
    offer_continuation runs ALL hooks (each observes the turn end) and the FIRST
    non-None wins; on_rate_limit returns the first non-None immediately. An
    empty chain behaves exactly like LifecycleHooks().
    """

    def __init__(self, hooks: list[LifecycleHooks]) -> None:
        self._hooks = list(hooks)

    async def session_start(self, convo: Conversation, resumed: bool) -> None:
        for h in self._hooks:
            await h.session_start(convo, resumed)

    async def user_prompt_submit(self, text: str) -> str:
        for h in self._hooks:  # first PromptRejected propagates (short-circuit)
            text = await h.user_prompt_submit(text)
        return text

    async def turn_start(self, convo: Conversation) -> None:
        for h in self._hooks:
            await h.turn_start(convo)

    async def pre_request(self, messages: list[Message], ctx: TurnCtx) -> None:
        for h in self._hooks:
            await h.pre_request(messages, ctx)

    async def pre_request_options(
        self, messages: list[Message], options: ChatOptions, ctx: TurnCtx
    ) -> None:
        for h in self._hooks:
            await h.pre_request_options(messages, options, ctx)

    async def on_request(
        self,
        messages: list[Message],
        tools: list[ToolDef],
        options: ChatOptions,
        ctx: TurnCtx,
    ) -> None:
        for h in self._hooks:
            await h.on_request(messages, tools, options, ctx)

    async def on_text_delta(self, delta: str) -> str:
        for h in self._hooks:
            delta = await h.on_text_delta(delta)
        return delta

    async def on_reasoning_delta(self, delta: str) -> str:
        for h in self._hooks:
            delta = await h.on_reasoning_delta(delta)
        return delta

    async def on_model_response(self, response: Message) -> None:
        for h in self._hooks:
            await h.on_model_response(response)

    async def offer_continuation(self, convo: Conversation) -> str | None:
        continuation: str | None = None
        for h in self._hooks:  # ALL observe; the first Some wins
            result = await h.offer_continuation(convo)
            if continuation is None:
                continuation = result
        return continuation

    async def offer_typed_continuation(self, convo: Conversation) -> Continuation | None:
        continuation: Continuation | None = None
        for h in self._hooks:
            result = await h.offer_typed_continuation(convo)
            if continuation is None:
                continuation = result
        return continuation

    async def turn_complete(self, convo: Conversation, reason: StopReason, ctx: TurnCtx) -> None:
        for h in self._hooks:
            await h.turn_complete(convo, reason, ctx)

    async def on_error(self, error: str) -> None:
        for h in self._hooks:
            await h.on_error(error)

    async def on_rate_limit(self, hint: RateLimitHint) -> RateLimitDecision | None:
        for h in self._hooks:
            decision = await h.on_rate_limit(hint)
            if decision is not None:
                return decision
        return None

    async def session_end(self, convo: Conversation) -> None:
        for h in self._hooks:
            await h.session_end(convo)


class ToolMiddleware:
    """TOOL-level seams wrapping each call; registration order is load-bearing
    (before-chain forward; argument-normalizers must precede every gate/approver,
    or the user approves different bytes than what executes). NOT abstract: both
    methods default to PROCEED, so a middleware overrides only the side it needs."""

    async def before(self, call: ToolCall, tool: Tool, rt: Any) -> BeforeOutcome:
        """May REWRITE the call in place (args; a name rewrite does not re-route)
        and round-trip the driver via rt.request(kind, payload) (approvals).
        Gate fold across the chain: first DENY / DENY_TURN blocks; ALLOW
        short-circuits the remaining gates; ASK defers to a downstream approval
        middleware (the kernel owns no prompt). `rt` is the RequestCtx broker."""
        return BeforeOutcome.PROCEED

    async def after(self, result: ToolResult) -> AfterOutcome:
        """Transform/observe the RAW (pre-size-cap) result in place; BLOCK(reason)
        feeds the reason back to the model. Natural collection point for the
        response envelope (the L2-collector pattern)."""
        return AfterOutcome.PROCEED


class CompactionStrategy(ABC):
    """PLAN-ONLY policy: proposes a CompactionPlan from a read-only view; the
    kernel remains the sole history writer (clamps, net-loss guard, epoch bump),
    so a buggy strategy cannot corrupt invariants."""

    @abstractmethod
    async def plan(self, view: CompactionView) -> CompactionPlan: ...

    def will_summarize(self, view: CompactionView) -> bool:
        """Cheap, side-effect-free pre-check (NO LLM call): will plan() do slow
        summary work? Gates the 'compacting…' progress event."""
        return False


class NoCompaction(CompactionStrategy):
    """The neutral default: never compacts (always a noop plan)."""

    async def plan(self, view: CompactionView) -> CompactionPlan:
        return CompactionPlan()


class CompactionCheckpointError(Exception):
    """A prepared compaction snapshot could not be durably saved — the plan is
    NOT committed (live history and epoch unchanged)."""


class CompactionCheckpoint(ABC):
    """The durable writer gating committed manual compactions and session resume.
    save() returns only after the snapshot is available to the resume path."""

    @abstractmethod
    def save(self, snapshot: SessionSnapshot) -> None:
        """Raise CompactionCheckpointError on failure."""


class Clock(ABC):
    """Injected monotonic milliseconds — timestamp STAMPING only (the loop's sole
    nondeterministic value; see kernel-loop-structure.md §II.8). Only differences
    between two reads are meaningful."""

    @abstractmethod
    def now_millis(self) -> int: ...


class SystemClock(Clock):
    """The real monotonic clock — the kernel default."""

    def __init__(self) -> None:
        self._origin = time.monotonic()

    def now_millis(self) -> int:
        return int((time.monotonic() - self._origin) * 1000)


class FixedClock(Clock):
    """Always returns the same value, so every elapsed_ms is 0 — makes a run's
    snapshots byte-reproducible for eval / replay."""

    def __init__(self, value: int = 0) -> None:
        self._value = value

    def now_millis(self) -> int:
        return self._value
