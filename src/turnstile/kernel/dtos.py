"""L0 DTOs — the shared vocabulary every layer speaks.

Reference definitions: docs/one-loop-two-layer-agent.md Appendix; structure tour:
docs/kernel-loop-structure.md Part I.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

# ── enums ──────────────────────────────────────────────────────────────


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class RiskLevel(Enum):
    """Advisory only — the loop knows risk, never enforces it."""

    SAFE = "safe"
    RISKY = "risky"


class StopReason(Enum):
    """How a turn terminated — exactly one TurnComplete per turn."""

    STOPPED = "stopped"  # normal: no tool calls, no continuation
    MAX_ROUNDS = "max_rounds"  # per-turn round fuse tripped
    MAX_CONTINUATIONS = "max_continuations"  # runaway offer_continuation fuse
    REPEAT_LOOP = "repeat_loop"  # coarse fuse: same call pattern, consecutive rounds
    TOOL_LOOP_DETECTED = "tool_loop_detected"  # opt-in exact no-progress guard
    PROVIDER_ERROR = "provider_error"  # failed open / mid-stream / empty-retry exhausted
    TIMEOUT = "timeout"  # stream-liveness timeout, reconnects exhausted
    CANCELLED = "cancelled"
    PROMPT_REJECTED = "prompt_rejected"  # user_prompt_submit blocked — no turn ran
    POLICY_DENIED = "policy_denied"  # middleware DENY_TURN; results stayed paired
    RATE_LIMITED = "rate_limited"  # 429 pause — NOT a failure; content preserved


class PromptRejected(Exception):
    """Raised by a user_prompt_submit hook to block the prompt (no turn runs)."""


# ── middleware gate DTOs ───────────────────────────────────────────────


class Gate(Enum):
    PROCEED = "proceed"  # continue the chain + normal approval flow
    ALLOW = "allow"  # force-approve: bypass remaining gates, no prompt
    ASK = "ask"  # defer to a downstream approval middleware
    DENY = "deny"  # block THIS call; reason -> model + driver
    DENY_TURN = "deny_turn"  # block + terminate the turn once the batch is paired


@dataclass(frozen=True)
class BeforeOutcome:
    """Immutable return value: the shared PROCEED singleton must be poison-proof.
    (Call rewrites ride on the mutable ToolCall, never on the outcome.)"""

    gate: Gate = Gate.PROCEED
    reason: str = ""

    PROCEED: ClassVar["BeforeOutcome"]  # singleton, assigned below


BeforeOutcome.PROCEED = BeforeOutcome()


@dataclass(frozen=True)
class AfterOutcome:
    """PROCEED, or BLOCK: reason fed back to the model as a synthetic user message.
    Immutable return value (result transforms ride on the mutable ToolResult)."""

    block_reason: str | None = None

    PROCEED: ClassVar["AfterOutcome"]


AfterOutcome.PROCEED = AfterOutcome()


# ── tool DTOs ──────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON-arguments string from the model


@dataclass(frozen=True)
class ToolDef:
    """What the model sees for a mounted tool."""

    name: str
    description: str
    parameters: dict  # JSON schema


@dataclass(frozen=True)
class ImageContent:
    """Neutral inline image; the L1 adapter owns the wire shape."""

    media_type: str  # e.g. "image/png"
    data: str  # base64 bytes


@dataclass
class ToolResult:
    call_id: str
    content: str  # size-capped by the kernel before store/emit
    is_error: bool = False
    images: list[ImageContent] = field(default_factory=list)
    # ^ TRANSIENT carrier: the loop lifts these onto ONE synthetic user message after
    #   the batch (providers reject images on the tool role); never stored on the
    #   tool-result message itself.


class CancellationToken:
    """Cooperative per-turn cancellation. A long tool polls `is_cancelled` or
    awaits `cancelled()`; the kernel also drops the execute future as a backstop."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    async def cancelled(self) -> None:
        await self._event.wait()


class ProgressSink:
    """Live progress channel a long-running tool MAY use; each emit becomes an
    AgentEvent ToolProgress. The no-arg default silently discards, so a tool can
    always call emit without branching."""

    def __init__(self, fn: Callable[[str], None] | None = None) -> None:
        self._fn = fn

    def emit(self, message: str) -> None:
        if self._fn is not None:
            self._fn(message)


@dataclass(frozen=True)
class ToolContext:
    """Handed to Tool.execute(). The kernel never chdir's the process and never
    sandboxes — see the trust model (design doc §4)."""

    working_dir: str
    cancel: CancellationToken = field(default_factory=CancellationToken)
    progress: ProgressSink = field(default_factory=ProgressSink)
    requester: object | None = None


# ── content sidecars ───────────────────────────────────────────────────


@dataclass(frozen=True)
class ReasoningBlock:
    """One SIGNED thinking unit (Anthropic/OpenAI/Gemini opaque round-trip)."""

    text: str  # may be empty (redacted block)
    opaque: str | None = None  # round-trip token, echoed VERBATIM, never re-encoded
    provider: str | None = None
    # ^ INVARIANT: opaque set => provider set — a token is PROVIDER-BOUND; an adapter
    #   echoes it only to its own backend. Plain-text reasoning paths never make these.


@dataclass
class TokenUsage:
    """Provider usage for one LLM call."""

    prompt: int = 0
    completion: int = 0
    cached: int = 0

    def merge_max(self, other: "TokenUsage") -> None:
        # Field-wise MAX fold across a round's (possibly multiple) Usage events —
        # correct for both one cumulative report (OpenAI) and split / cumulative-delta
        # reports (Anthropic): never double-counts, never drops an early-only field.
        self.prompt = max(self.prompt, other.prompt)
        self.completion = max(self.completion, other.completion)
        self.cached = max(self.cached, other.cached)


# ── message DTOs ───────────────────────────────────────────────────────


@dataclass
class MessageMeta:
    """Execution-stats sidecar; never rendered into text (prefix-cache safety)."""

    tokens: TokenUsage
    elapsed_ms: int
    reasoning_elapsed_ms: int = 0  # thinking-phase duration
    ctx_window: int = 0
    used_tokens: int = 0  # provider's prompt count, or byte-estimate fallback
    utilization: float = 0.0
    round: int = 0
    turn_id: int = 0  # correlation: which user turn produced this
    request_id: int = 0  # correlation: which LLM request (session-global)
    provider_response_id: str | None = None  # upstream handle for log cross-referencing
    provider_model: str | None = None  # reported model — gateway-misroute detection
    session_id: str | None = None
    finish_reason: str = ""  # "stop" | "tool_calls" | "length"


@dataclass
class Message:
    """Provider-neutral, losslessly persistable."""

    role: Role
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    is_error: bool = False  # true iff this is a FAILED tool result
    meta: MessageMeta | None = None
    synthetic: bool = False  # kernel-injected (summary / resume note / nudge)
    internal_origin: str | None = None  # e.g. "verify_cadence"
    reasoning: str | None = None  # flat stored thinking (plain-text path)
    reasoning_blocks: list[ReasoningBlock] = field(default_factory=list)  # signed path
    images: list[ImageContent] = field(default_factory=list)  # multimodal input

    # -- constructors ---------------------------------------------------

    @classmethod
    def system(cls, text: str) -> "Message":
        return cls(role=Role.SYSTEM, text=text)

    @classmethod
    def user(cls, text: str, images: list[ImageContent] | None = None) -> "Message":
        return cls(role=Role.USER, text=text, images=images or [])

    @classmethod
    def assistant(cls, text: str, tool_calls: list[ToolCall] | None = None) -> "Message":
        return cls(role=Role.ASSISTANT, text=text, tool_calls=tool_calls or [])

    @classmethod
    def tool_result(cls, call_id: str, content: str, is_error: bool = False) -> "Message":
        return cls(role=Role.TOOL, text=content, tool_call_id=call_id, is_error=is_error)

    @classmethod
    def synthetic_user(cls, text: str, images: list[ImageContent] | None = None) -> "Message":
        """Kernel-injected user-role message (compaction summary / resume note /
        nudge / tool images). User role, not System: a mid-conversation system
        message is rejected or re-homed by many providers; `synthetic=True` keeps
        it out of `sacred_floor` anchoring."""
        return cls(role=Role.USER, text=text, synthetic=True, images=images or [])

    # -- intrinsic helpers ----------------------------------------------

    def estimate_tokens(self) -> int:
        """Byte heuristic (~4 bytes/token; images ~1600 tokens each). FALLBACK
        ONLY, for when the provider omits a usage report — without it a
        non-reporting provider records utilization 0.0 forever and never compacts."""
        if self.images:
            return max(len(self.text) // 4, 1) + len(self.images) * 1600 + 4
        if self.role is Role.TOOL:
            byte_count = len(self.text) + 10  # small wrapper overhead
        elif self.tool_calls:
            calls = sum(len(c.name) + len(c.arguments) + 20 for c in self.tool_calls)
            byte_count = len(self.text) + calls + len(self.reasoning or "")
        else:
            byte_count = len(self.text)
        return max(byte_count // 4, 1) + 4


# ── streaming (a tagged union — the LlmProvider yields these) ──────────


@dataclass(eq=False)
class ProviderError(Exception):
    """Raised on a failed OPEN; carried by an ErrorEvent mid-stream."""

    message: str = ""
    retryable: bool = False  # 429/5xx/timeout may retry; False = terminal (auth/400)
    http_status: int | None = None
    code: str | None = None  # structured code, e.g. "context_length_exceeded"
    retry_after_secs: int | None = None  # real Retry-After header on a 429

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def is_context_overflow(self) -> bool:
        """Hard context-window overflow — the whole request was rejected as too
        long. Centralizes per-vendor signatures so no call site string-matches."""
        if self.code and self.code.lower() in (
            "context_length_exceeded",
            "string_above_max_length",
        ):
            return True
        if self.http_status == 400:
            m = self.message.lower()
            needles = (
                "context length",
                "context window",
                "maximum context",
                "prompt is too long",
                "reduce the length",
                "too many tokens",
                "range of input length",  # Bailian / Tencent gateway
                "maximum prompt length",  # Anthropic-style variant
                "too large for model with ",  # trailing space: don't match "...without..."
            )
            return any(n in m for n in needles)
        return False


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class Reasoning:
    text: str


@dataclass(frozen=True)
class ReasoningSignature:
    """End-of-block boundary for a SIGNED thinking unit — finalize one
    ReasoningBlock from the Reasoning deltas since the previous boundary."""

    opaque: str
    provider: str


@dataclass(frozen=True)
class ToolCallEvent:
    call: ToolCall


@dataclass(frozen=True)
class ToolCallDelta:
    """Display-only streaming fragment of a tool call; the whole ToolCall is
    still emitted separately for execution."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass(frozen=True)
class UsageEvent:
    """May repeat within one round; fold via TokenUsage.merge_max."""

    usage: TokenUsage


@dataclass(frozen=True)
class ResponseId:
    id: str


@dataclass(frozen=True)
class ResponseModel:
    model: str


@dataclass(frozen=True)
class ErrorEvent:
    """Mid-stream failure — cleanly fails the turn (or enters a retry tier)."""

    error: ProviderError


@dataclass(frozen=True)
class Malformed:
    """The adapter dropped an unparseable chunk. Diagnostic signal, not content."""


@dataclass(frozen=True)
class Done:
    truncated: bool = False  # cut by finish_reason=length


type StreamEvent = (
    TextDelta
    | Reasoning
    | ReasoningSignature
    | ToolCallEvent
    | ToolCallDelta
    | UsageEvent
    | ResponseId
    | ResponseModel
    | ErrorEvent
    | Malformed
    | Done
)


# ── per-request / per-turn ─────────────────────────────────────────────


@dataclass
class ChatOptions:
    """Neutral per-call knobs; None = adapter default. SIDEBAND: never part of
    the prefix bytes (cache-safe). The kernel forwards, the L1 adapter maps."""

    reasoning_effort: str | None = None  # low | medium | high | max
    enable_thinking: bool | None = (
        None  # reasoning models: thinking on/off; None = backend default
    )
    max_tokens: int | None = None
    temperature: float | None = None
    tool_choice: str = "auto"  # auto | required | none | <specific tool name>


class ContinuationKind(Enum):
    GENERIC = "generic"
    VERIFY_CADENCE = "verify_cadence"
    TRUNCATION_RESUME = "truncation_resume"


class ContinuationVisibility(Enum):
    NORMAL = "normal"
    INTERNAL_CONTROL = "internal_control"
    # ^ INTERNAL_CONTROL rounds stream nothing to the driver and store blank text
    #   with internal_origin set — control chatter stays out of user-visible history


@dataclass(frozen=True)
class Continuation:
    """Typed offer_continuation result."""

    text: str
    kind: ContinuationKind = ContinuationKind.GENERIC
    visibility: ContinuationVisibility = ContinuationVisibility.NORMAL


@dataclass(frozen=True)
class RateLimitHint:
    """What the kernel knows about a 429 at the moment it fires."""

    http_status: int | None
    retry_after_secs: int | None  # header preferred; body-text hint as fallback
    terminal: bool  # account/billing exhaustion — never auto-retry
    attempt: int  # 1-based consecutive incident this turn (kernel-owned)


@dataclass(frozen=True)
class RateLimitDecision:
    """Host verdict on a 429: wait_secs set -> WaitAndRetry; else Pause."""

    wait_secs: int | None = None
    reset_at_display: str = ""  # Pause-side display facts (may be empty)
    reset_label: str = ""
    secs_until_reset: int | None = None


@dataclass(frozen=True)
class TurnCtx:
    """Correlation + live pressure, passed to hooks. Frozen: read-only enforced."""

    session_id: str | None  # driver-owned identity; the kernel never mints it
    turn_id: int  # one user message -> one turn (constant across rounds)
    request_id: int  # unique per LLM request (bumps every round)
    round: int  # 1-based index of the LLM call within the turn
    max_rounds: int | None = None
    cache_epoch: int = 0  # prefix-generation marker
    context_window: int = 0  # from the LAST response's usage report (0 on round 1)
    used_tokens: int = 0  # ditto — pairs with context_window for live pressure
    # Per-turn structured data (quality verdict, citations) lives on the L2
    # collector objects themselves — the L2-collector pattern
    # (docs/kernel-loop-structure.md §I.2).


# ── compaction (strategy proposes, kernel disposes) ────────────────────


@dataclass(frozen=True)
class AutoTrigger:
    """Context-pressure driven: utilization of the window crossed the threshold."""

    utilization: float


@dataclass(frozen=True)
class ManualTrigger:
    """User-requested (e.g. /compact), optionally focused on a topic."""

    focus: str | None = None


@dataclass(frozen=True)
class OverflowTrigger:
    """Hard context-window overflow recovery; attempt drives the escalation ladder."""

    attempt: int = 0


type CompactTrigger = AutoTrigger | ManualTrigger | OverflowTrigger


@dataclass(frozen=True)
class CompactionView:
    """READ-ONLY view handed to CompactionStrategy.plan."""

    messages: list[Message]
    trigger: CompactTrigger
    ctx_window: int
    used_tokens: int
    utilization: float
    sacred_floor: int  # leading messages a plan must not drain (kernel re-clamps)


@dataclass(frozen=True)
class CompactionPlan:
    """A PROPOSAL; the kernel revalidates everything (clamps, net-loss guard,
    epoch bump)."""

    drain_from: int = 0  # replace messages[drain_from:drain_to] with the summary
    drain_to: int = 0
    summary: str | None = None  # inserted as ONE synthetic user message
    rewrites: list[tuple[int, str]] = field(default_factory=list)
    # ^ in-place text stubs (permanent microcompact); indices are ORIGINAL positions —
    #   the kernel translates past the drain/summary shift, skips drained/sacred targets
    resume_note: str | None = None  # appended as a trailing synthetic user message

    def is_noop(self) -> bool:
        return (
            self.drain_from >= self.drain_to
            and self.summary is None
            and not self.rewrites
            and self.resume_note is None
        )


@dataclass(frozen=True)
class CompactReport:
    """Audit record of one attempt; committed=False = refused (net-loss guard or
    noop): messages byte-identical, epoch unchanged."""

    epoch_before: int
    epoch_after: int
    removed: int
    bytes_before: int
    bytes_after: int
    committed: bool


# ── session persistence ────────────────────────────────────────────────


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)
    cache_epoch: int = 0  # bumped only by a committed compaction

    def push(self, message: Message) -> None:
        self.messages.append(message)

    def sacred_floor(self) -> int:
        """Number of LEADING messages compaction must never remove: a leading
        System message plus up to and INCLUDING the first NON-SYNTHETIC User
        message — persona + the original ask survive every compaction."""
        lead_system = 1 if self.messages and self.messages[0].role is Role.SYSTEM else 0
        for i, m in enumerate(self.messages):
            if m.role is Role.USER and not m.synthetic:
                return i + 1
        return lead_system

    def last_pressure(self) -> tuple[int, int, float]:
        """(context_window, used_tokens, utilization) from the most recent
        assistant message's recorded meta; zeros before any assistant turn."""
        for m in reversed(self.messages):
            if m.role is Role.ASSISTANT and m.meta is not None:
                return (m.meta.ctx_window, m.meta.used_tokens, m.meta.utilization)
        return (0, 0, 0.0)

    def backfill_cancelled_tool_results(self) -> None:
        """APPEND a synthetic '(cancelled)' error result for every assistant
        tool_call lacking a matching result — keeps the wire API-valid after a
        cancel mid-turn. Append-only: never mutates or reorders existing messages."""
        self._backfill_missing_tool_results("(cancelled)")

    def backfill_interrupted_tool_results(self) -> None:
        """Same repair for calls recovered from a failed provider stream that were
        never executed — distinct wording so a transport failure is not misreported
        as a user cancellation on resume."""
        self._backfill_missing_tool_results("(interrupted before execution)")

    def _backfill_missing_tool_results(self, content: str) -> None:
        seen_result_ids = {m.tool_call_id for m in self.messages if m.tool_call_id}
        missing: list[str] = []
        for m in self.messages:
            if m.role is Role.ASSISTANT:
                for call in m.tool_calls:
                    if call.id not in seen_result_ids and call.id not in missing:
                        missing.append(call.id)
        for call_id in missing:
            self.push(Message.tool_result(call_id, content, is_error=True))

    @staticmethod
    def repair_pairing(messages: list[Message]) -> None:
        """Make a message list API-VALID in place: every assistant tool_call gets
        EXACTLY ONE result placed immediately after its window; orphan tool
        results (no matching call) are dropped; missing results are synthesized
        as '(cancelled)' errors. Kernel-owned so a buggy strategy, hook
        projection, or legacy snapshot can never hand the provider an illegal
        payload. Sequential windows handle reused ids across turns."""
        rebuilt: list[Message] = []
        i = 0
        n = len(messages)
        while i < n:
            m = messages[i]
            i += 1
            if m.role is not Role.ASSISTANT or not m.tool_calls:
                if m.role is not Role.TOOL:  # a tool result outside a window = orphan
                    rebuilt.append(m)
                continue
            pending = [c.id for c in m.tool_calls]
            rebuilt.append(m)
            resolved: set[str] = set()
            while i < n and messages[i].role is Role.TOOL:
                result = messages[i]
                i += 1
                rid = result.tool_call_id
                if rid is not None and rid in pending and rid not in resolved:
                    resolved.add(rid)
                    rebuilt.append(result)
            for call_id in pending:
                if call_id not in resolved:
                    resolved.add(call_id)
                    rebuilt.append(Message.tool_result(call_id, "(cancelled)", is_error=True))
        messages[:] = rebuilt


SNAPSHOT_VERSION = 1
"""Persisted-conversation schema version. Bump whenever the serialized shape of
Message/Conversation changes in a way an older kernel could not read."""


@dataclass
class SessionSnapshot:
    """Versioned, LOSSLESS, resumable conversation snapshot — the durable contract."""

    version: int  # reader checks BEFORE interpreting messages
    messages: list[Message]
    cache_epoch: int = 0
    turn_counter: int = 0  # id high-water marks so resume stays monotonic
    request_counter: int = 0

    @classmethod
    def from_conversation(cls, convo: Conversation) -> "SessionSnapshot":
        turn_counter, request_counter = cls.derive_counters(convo.messages)
        return cls(
            version=SNAPSHOT_VERSION,
            messages=list(convo.messages),
            cache_epoch=convo.cache_epoch,
            turn_counter=turn_counter,
            request_counter=request_counter,
        )

    @staticmethod
    def derive_counters(messages: list[Message]) -> tuple[int, int]:
        """Max meta.turn_id / meta.request_id over the messages (0 when none) —
        the fallback derivation; a capturer that knows the live counters stamps
        higher values over these for turns that died before storing a response."""
        turn = request = 0
        for m in messages:
            if m.meta is not None:
                turn = max(turn, m.meta.turn_id)
                request = max(request, m.meta.request_id)
        return (turn, request)
