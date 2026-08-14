"""L0 driver protocol — AgentCommand / AgentEvent / Outcome + the RequestCtx broker.

The loop's public face (design doc §5): the driver sends commands, consumes
events. Everything here is a plain DTO (asdict-serializable), so the SAME
protocol works in-process and across a process/network boundary — the service
layer serializes the event stream onto SSE verbatim; class names map to wire
names there, not here.

Naming notes vs dtos.py: stream-level dtos.TextDelta/Reasoning are what the
PROVIDER yields to the loop; the same-named classes here are what the LOOP emits
to the driver (post-hook bytes). ToolResultEvent / Usage wrap dtos payloads
(dtos.ToolResult / dtos.MessageMeta) — renamed where they would collide.
"""

import asyncio
import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from turnstile.kernel.dtos import (
    CompactTrigger,
    ImageContent,
    MessageMeta,
    SessionSnapshot,
    StopReason,
    ToolCall,
    ToolResult,
)
from turnstile.kernel.ports import Requester

ROUND_CAP_CHECKPOINT_KIND = "round_cap_checkpoint"
"""Driver round-trip kind for the round-cap checkpoint (kernel-initiated: the
fuse pauses and asks 'continue past the cap?'). The driver answers
{"continue": bool}; a missing/None answer degrades to False (stop)."""


# ── driver → agent commands ────────────────────────────────────────────


@dataclass(frozen=True)
class SendMessage:
    """The user's next prompt. images: optional multimodal attachments."""

    text: str
    images: list[ImageContent] = field(default_factory=list)


@dataclass(frozen=True)
class SendMessageWithContext:
    """One REAL user prompt with host-owned synthetic context prepended to the
    SAME turn: the context is stored as a synthetic user message, then the real
    prompt normally — one command, one turn, one terminal. For deterministic
    resume/recovery context that must not become a second automated turn or leak
    into user-facing prompt projections."""

    text: str
    context: str
    images: list[ImageContent] = field(default_factory=list)


@dataclass(frozen=True)
class SendSyntheticMessage:
    """Host-injected synthetic prompt (e.g. an automated continuation). Same
    execution path as SendMessage, but stored as a synthetic user message so
    sacred_floor skips it and hosts can hide it from user-facing projections."""

    text: str


@dataclass(frozen=True)
class Respond:
    """Answer a pending Request event, correlated by id."""

    id: int
    value: Any


@dataclass(frozen=True)
class RequestSnapshot:
    """Ask the agent to emit a Snapshot event of the current conversation."""


@dataclass(frozen=True)
class Compact:
    """MANUAL compaction (e.g. a user /compact) — runs the injected strategy
    regardless of any auto threshold; a net-loss/no-op plan is still refused."""

    focus: str | None = None


@dataclass(frozen=True)
class Cancel:
    """Cooperatively cancel the running turn."""


@dataclass(frozen=True)
class Shutdown:
    """Tear the session down (session_end still fires)."""


type AgentCommand = (
    SendMessage
    | SendMessageWithContext
    | SendSyntheticMessage
    | Respond
    | RequestSnapshot
    | Compact
    | Cancel
    | Shutdown
)


# ── agent → driver events ──────────────────────────────────────────────


@dataclass(frozen=True)
class TurnStarted:
    """A turn began (perception granularity)."""


@dataclass(frozen=True)
class TextDelta:
    """A streamed answer chunk (post-hook bytes — consistent with storage)."""

    text: str


@dataclass(frozen=True)
class Reasoning:
    """A streamed thinking chunk (post-hook bytes)."""

    text: str


@dataclass(frozen=True)
class ToolCallStreaming:
    """Live display of a tool call the model is still emitting. Observational —
    the tool executes later from the complete call."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass(frozen=True)
class ToolBatchCall:
    """One call inside a ToolBatchStarted payload — what a driver needs to
    render a child row, incl. the honest 'in parallel' label."""

    id: str
    name: str
    arguments: str
    parallel_safe: bool


@dataclass(frozen=True)
class ToolBatchStarted:
    """>= 2 distinct calls fan out from one assistant message; fires before the
    per-call ToolStarted events so a driver renders one grouped block."""

    batch_id: str
    calls: list[ToolBatchCall]


@dataclass(frozen=True)
class ToolBatchCompleted:
    batch_id: str
    ok: int
    total: int
    elapsed_ms: int


@dataclass(frozen=True)
class ToolStarted:
    call: ToolCall


@dataclass(frozen=True)
class ToolProgress:
    """Live progress from a long-running tool mid-execution (ProgressSink.emit)."""

    call_id: str
    message: str


@dataclass(frozen=True)
class ToolResultEvent:
    result: ToolResult


@dataclass(frozen=True)
class Request:
    """Generic middleware/tool <-> driver round-trip; kernel is agnostic to
    kind/payload. Answered by the Respond command, correlated by id."""

    id: int
    kind: str
    payload: dict


@dataclass(frozen=True)
class Usage:
    """Per-LLM-call execution stats (mirrors the stored message sidecar)."""

    meta: MessageMeta


@dataclass(frozen=True)
class Snapshot:
    """Whole-conversation snapshot (reply to RequestSnapshot, or emitted at
    shutdown so a replacing owner never resumes from a stale checkpoint)."""

    snapshot: SessionSnapshot


@dataclass(frozen=True)
class SteeredInput:
    """One user input that was folded into the running turn."""

    text: str
    images: list[ImageContent] = field(default_factory=list)


@dataclass(frozen=True)
class Steered:
    """User prompt(s) folded ('steered') into the running turn at a round
    boundary — drivers relabel 'queued' to 'folded into current turn'."""

    count: int
    inputs: list[SteeredInput] = field(default_factory=list)


@dataclass(frozen=True)
class Warning:
    """Non-fatal advisory (truncation, retry notice, cache-prefix violation…).
    The turn continues."""

    message: str


@dataclass(frozen=True)
class Error:
    """A failure. http_status/code are the structured provider codes (None for
    kernel-internal errors)."""

    message: str
    http_status: int | None = None
    code: str | None = None


@dataclass(frozen=True)
class RateLimited:
    """A 429 pause or auto-resuming wait — a driver renders a non-error pause
    line, never a red error."""

    reset_at_display: str = ""
    reset_label: str = ""
    secs_until_reset: int | None = None
    auto_resuming: bool = False  # True = kernel sleeps then retries by itself
    server_message: str | None = None  # provider's own 429 reason, when actionable


@dataclass(frozen=True)
class Cancelled:
    """The turn was cooperatively cancelled; emitted just before the terminal
    TurnComplete, with any dangling tool calls already repaired."""


@dataclass(frozen=True)
class CompactionStarted:
    """A compaction is about to run (may include a slow LLM summary call)."""

    trigger: CompactTrigger


@dataclass(frozen=True)
class Compacted:
    """A compaction was ATTEMPTED. committed=False = refused (net-loss guard or
    no-op): history byte-identical, epoch unchanged."""

    trigger: CompactTrigger
    epoch: int
    removed: int
    bytes_before: int
    bytes_after: int
    committed: bool
    snapshot: SessionSnapshot | None = None  # exact post-compaction working set (manual)


@dataclass(frozen=True)
class CompactionFailed:
    """A prepared manual compaction could not be durably checkpointed; the live
    conversation and cache epoch are unchanged."""

    trigger: CompactTrigger
    error: str


@dataclass(frozen=True)
class TurnComplete:
    """THE terminal — exactly one per turn, on every exit path. reason is the
    failure-perception contract: a failed turn can never look like an empty
    success."""

    reason: StopReason


type AgentEvent = (
    TurnStarted
    | TextDelta
    | Reasoning
    | ToolCallStreaming
    | ToolBatchStarted
    | ToolBatchCompleted
    | ToolStarted
    | ToolProgress
    | ToolResultEvent
    | Request
    | Usage
    | Snapshot
    | Steered
    | Warning
    | Error
    | RateLimited
    | Cancelled
    | CompactionStarted
    | Compacted
    | CompactionFailed
    | TurnComplete
)


# ── one-shot aggregate ─────────────────────────────────────────────────


@dataclass
class Outcome:
    """Aggregated result for batch/CI drivers (run_to_completion). stop + error
    make a failed run impossible to mistake for an empty success."""

    text: str = ""
    tool_results: list[ToolResult] = field(default_factory=list)
    stop: StopReason = StopReason.STOPPED
    error: str | None = None  # last Error captured; None on a clean stop
    http_status: int | None = None
    error_code: str | None = None


# ── the Request/Respond broker ─────────────────────────────────────────


class RequestCtx:
    """Kernel-internal round-trip broker: lets middleware/tools emit events and
    perform an id-correlated request to the driver. The future that resolves a
    request lives ONLY here — never inside an event — which is what keeps the
    protocol serializable / wire-compatible.

    LIVENESS: request_timeout bounds how long request() awaits the driver's
    Respond; on expiry (or cancel_pending) the round-trip degrades to None so an
    awaiting middleware proceeds FAIL-CLOSED (an approval sees None -> deny)
    instead of parking the turn forever. None = unbounded (default).
    """

    def __init__(
        self,
        emit: Callable[[AgentEvent], None],
        request_timeout: float | None = None,
    ) -> None:
        self._emit = emit
        self._timeout = request_timeout
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._ids = itertools.count(1)

    def emit(self, event: AgentEvent) -> None:
        """Forward an event to the driver."""
        self._emit(event)

    async def request(self, kind: str, payload: dict) -> Any:
        """Emit Request{id, kind, payload} and await the matching Respond."""
        request_id = next(self._ids)
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        self._emit(Request(id=request_id, kind=kind, payload=payload))
        try:
            if self._timeout is None:
                return await future
            return await asyncio.wait_for(future, self._timeout)
        except TimeoutError:
            return None  # degrade like a dead driver; late Respond no-ops
        finally:
            self._pending.pop(request_id, None)

    def resolve(self, request_id: int, value: Any) -> None:
        """Route a driver Respond to the waiting requester. Unknown/late ids no-op."""
        future = self._pending.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(value)

    def cancel_pending(self) -> None:
        """Resolve EVERY pending request to None (fail-closed) — a cancel must
        also unblock a turn parked inside a middleware round-trip."""
        for future in self._pending.values():
            if not future.done():
                future.set_result(None)
        self._pending.clear()

    def requester(self) -> Requester:
        """A request-only handle for tools (rides in ToolContext)."""
        return _CtxRequester(self)


class _CtxRequester(Requester):
    """Thin handle exposing ONLY the round-trip (not emit/resolve/cancel)."""

    def __init__(self, ctx: RequestCtx) -> None:
        self._ctx = ctx

    async def request(self, kind: str, payload: dict) -> Any:
        return await self._ctx.request(kind, payload)
