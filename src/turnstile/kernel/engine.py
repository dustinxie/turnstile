"""The one loop — L0's engine, written entirely against ports and DTOs.

Mechanics spec: docs/kernel-loop-structure.md Part II. This module grows across
M1 commits; current coverage: session loop + spawn/handle (II driver protocol),
turn lifecycle (II.1), the single-round path of II.2 (project -> open -> consume
-> store -> branch), the unknown-tool degenerate of II.3, the offer_continuation
seam with its fuse, snapshot/resume seeding, and the run_to_completion adapter.
TODO(M1): full tool dispatch (II.3), remaining fuses (II.6), streaming seams
(II.2.7-8), resilience tiers (II.5), cancel/steer (II.4), compaction (II.7).
"""

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field, replace

from turnstile.kernel import events as ev
from turnstile.kernel.dtos import (
    SNAPSHOT_VERSION,
    CancellationToken,
    ChatOptions,
    ContinuationKind,
    ContinuationVisibility,
    Conversation,
    Done,
    ErrorEvent,
    Gate,
    Malformed,
    ManualTrigger,
    Message,
    MessageMeta,
    ProgressSink,
    PromptRejected,
    ProviderError,
    RateLimitDecision,
    RateLimitHint,
    Reasoning,
    ReasoningBlock,
    ReasoningSignature,
    ResponseId,
    ResponseModel,
    SessionSnapshot,
    StopReason,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    ToolCallEvent,
    ToolContext,
    ToolDef,
    ToolResult,
    TurnCtx,
    UsageEvent,
)
from turnstile.kernel.events import (
    ROUND_CAP_CHECKPOINT_KIND,
    AgentCommand,
    Outcome,
    RequestCtx,
)
from turnstile.kernel.ports import (
    Clock,
    CompactionCheckpoint,
    CompactionStrategy,
    HookChain,
    LifecycleHooks,
    LlmProvider,
    NoCompaction,
    SystemClock,
    Tool,
    ToolMiddleware,
)

# ── resilience tiers (II.5) ────────────────────────────────────────────

# Failed-open transient retries per round (the adapter's own fast transport
# retries sit BELOW this tier). Visible waits: 3/6/9s.
# First-event allowance on the liveness timeout: waiting for the FIRST stream
# event covers prefill, which is compute-bound and scales with context length;
# inter-token gaps are decode-bound and near-constant. One knob, two budgets:
# stream_timeout x THIS for time-to-first-token, stream_timeout plain between
# events — e.g. stream_timeout=5 gives a tight 5s inter-token bound with a
# 100s TTFT budget that survives a long prefill on a loaded box.
FIRST_EVENT_TIMEOUT_FACTOR = 20

MAX_PROVIDER_RETRIES = 3
# Mid-stream idle-timeout reconnects per round — only while NO content arrived
# (replaying after content risks duplicate output/side effects).
MAX_STREAM_RETRIES = 5
# Consecutive WaitAndRetry rate-limit sleeps per turn before a forced pause
# (livelock fuse for a broken hook / never-reopening window).
MAX_RATE_LIMIT_WAITS = 5
# Content-free 200s re-issued per TURN (an empty 200 opens fine and ends with a
# clean Done, so the error tiers never see it).
EMPTY_RESPONSE_MAX_RETRIES = 5
# Auto-continuations after an output-limit truncation per turn.
MAX_TRUNCATION_CONTINUATIONS = 2
# The FIRST anonymous transient 429 (no host verdict, no Retry-After) retries
# quietly after this wait — a one-off burst clears without banner spam.
SILENT_FIRST_RATE_LIMIT_WAIT = 1.0

TRUNCATION_RESUME_NUDGE = (
    "Output limit hit — your last response was cut off before finishing. If the "
    "task is already complete, reply with a short summary and stop (no tool "
    "calls). Otherwise resume where you left off, writing remaining content "
    "INCREMENTALLY (append the next section) rather than re-emitting it all in "
    "one response."
)

# Billing/account exhaustion signatures: such a 429 must never auto-retry.
_TERMINAL_RATE_LIMIT_CODES = frozenset(
    {
        "insufficient_quota",
        "billing_hard_limit_reached",
        "payment_required",
        "insufficient_balance",
        "1113",
    }
)
_TERMINAL_RATE_LIMIT_NEEDLES = (
    "insufficient quota",
    "insufficient balance",
    "billing hard limit",
    "payment required",
    "credit balance",
    "余额不足",
    "无可用资源包",
    "请充值",
)


def _is_terminal_rate_limit(error: ProviderError) -> bool:
    code = (error.code or "").lower()
    if code in _TERMINAL_RATE_LIMIT_CODES:
        return True
    message = error.message.lower()
    return any(needle in message for needle in _TERMINAL_RATE_LIMIT_NEEDLES)


def _effective_retry_after(error: ProviderError) -> int | None:
    """Authoritative Retry-After: the real header when the adapter surfaced one,
    else a best-effort 'try again in N seconds' sniff from the body text."""
    if error.retry_after_secs is not None:
        return error.retry_after_secs
    lower = error.message.lower()
    marker = "try again in "
    index = lower.find(marker)
    if index < 0:
        return None
    digits = ""
    for char in lower[index + len(marker) :]:
        if not char.isdigit():
            break
        digits += char
    return int(digits) if digits else None


INTERRUPTION_MARKER = (
    "[The previous response was interrupted by the user before completing. "
    "Reconsider the approach in light of this interruption before continuing.]"
)


class _TurnCancelled(Exception):
    """Internal control flow: a cancel checkpoint fired mid-turn. Caught by
    _run_turn's single handler, which funnels into the cancel terminal."""


# ── loop fuses (II.6) ──────────────────────────────────────────────────

# Always-on coarse repetition fuse: same model-emitted call pattern for too many
# CONSECUTIVE rounds, even when results change ("same choice regardless of
# results"). A course-correction nudge fires first.
MAX_REPEAT_ROUNDS = 6
REPEAT_NUDGE_AT = 3
REPEAT_LOOP_NUDGE = (
    "You have issued the SAME tool call with the SAME arguments several rounds "
    "in a row. Stop repeating it and change your approach. If the task is done, "
    "reply with a short summary and no tool calls. If you are blocked, explain "
    "what you need."
)
TOOL_LOOP_NUDGE = (
    "[Tool-loop guard] The same tool call or read-only batch has returned the "
    "same result(s) repeatedly. Do not repeat it unchanged. Reassess the task, "
    "use a different action, or explain why no further progress is possible."
)


@dataclass(frozen=True)
class ToolLoopPolicy:
    """Opt-in EXACT no-progress guard ("same call + same result"): warn at
    warning_threshold identical observations, stop at stop_threshold. The warning
    must leave the model a real chance to change course before the stop."""

    warning_threshold: int = 3
    stop_threshold: int = 4

    def __post_init__(self) -> None:
        if self.warning_threshold < 2:
            raise ValueError("tool-loop warning threshold must be at least 2")
        if self.warning_threshold >= self.stop_threshold:
            raise ValueError("tool-loop warning threshold must be below stop threshold")


class _ToolLoopState:
    """Session-owned exact-loop streak. Persists across synthetic turns (an
    automated goal can't evade the guard by opening fresh turns); a REAL user
    prompt resets it (new intent scope)."""

    def __init__(self, policy: ToolLoopPolicy) -> None:
        self.policy = policy
        self.last: tuple | None = None
        self.consecutive = 0

    def reset(self) -> None:
        self.last = None
        self.consecutive = 0

    def observe(self, fingerprint: tuple) -> str:
        if fingerprint == self.last:
            self.consecutive += 1
        else:
            self.last = fingerprint
            self.consecutive = 1
        if self.consecutive >= self.policy.stop_threshold:
            return "stop"
        if self.consecutive == self.policy.warning_threshold:
            return "warn"
        return "continue"


@dataclass
class AgentHandle:
    """Bidirectional session handle: send AgentCommand, receive AgentEvent."""

    commands: asyncio.Queue
    events: asyncio.Queue
    task: asyncio.Task


@dataclass
class Agent:
    """Assembled agent configuration. spawn() starts the session loop."""

    provider: LlmProvider
    tools: dict[str, Tool] = field(default_factory=dict)
    persona: str = ""
    hooks: list[LifecycleHooks] = field(default_factory=list)
    middleware: list[ToolMiddleware] = field(default_factory=list)
    max_rounds: int | None = None
    # SAFETY FUSE, defaults ON: an offer_continuation hook that always continues
    # is an infinite kernel-driven loop with no model agency to stop it.
    max_continuations: int | None = 50
    # When True, the max_rounds fuse becomes an interactive CHECKPOINT: the
    # kernel round-trips the driver ("continue past the cap?") and re-arms the
    # cap by the base amount on yes; a no/unanswered request stops fail-closed.
    round_cap_checkpoint: bool = False
    # Opt-in exact no-progress guard (product policy; the coarse REPEAT_LOOP
    # fuse is always on regardless).
    tool_loop_policy: ToolLoopPolicy | None = None
    # Phase-② concurrency cap for parallel-safe tools (side-effecting tools
    # always serialize regardless).
    max_parallel_tools: int = 4
    # Byte cap on a single tool result's content — the kernel's ONE built-in
    # safety bound (context bloat). 0 = unbounded.
    max_tool_result_bytes: int = 64 * 1024
    compaction: CompactionStrategy = field(default_factory=NoCompaction)
    checkpoint: CompactionCheckpoint | None = None
    chat_options: ChatOptions = field(default_factory=ChatOptions)
    clock: Clock = field(default_factory=SystemClock)
    session_id: str | None = None
    resume: SessionSnapshot | None = None
    working_dir: str = "."
    # KERNEL<->DRIVER round-trip bound, nothing to do with HTTP: how long a
    # Request (approval ask, round-cap checkpoint) awaits the driver's Respond
    # before degrading to None — the asker proceeds FAIL-CLOSED (approval sees
    # None -> deny). None (default) = wait forever; bound it whenever no human
    # is guaranteed to be attached to the driver.
    request_timeout: float | None = None
    # Cancel semantics: False (default) = CANCEL IS UNDO — the cancelled turn
    # rolls back to before its user message, leaving no trace. True = preserve
    # partial work (dangling calls backfilled '(cancelled)', an interruption
    # marker appended) — the right choice for a chatbot service where losing the
    # user's message on a flaky network is worse than keeping a half answer.
    keep_interrupted_context: bool = False
    # LIVENESS: max seconds to wait for the NEXT stream event (bounds first-token
    # AND inter-token latency). None = unbounded. Production should set it.
    stream_timeout: float | None = None
    # Multiplier on every retry/backoff sleep — tests set 0 for instant retries.
    backoff_scale: float = 1.0

    def spawn(self) -> AgentHandle:
        """Start the long-lived session loop; the driver owns the handle."""
        commands: asyncio.Queue = asyncio.Queue()
        events: asyncio.Queue = asyncio.Queue()
        if self.session_id is not None:
            self.provider.bind_session_id(self.session_id)
        session = _Session(self, events)
        task = asyncio.create_task(session.run(commands))
        return AgentHandle(commands=commands, events=events, task=task)

    async def run_to_completion(self, text: str, approve: bool = True) -> Outcome:
        """One-shot adapter for batch/CI drivers: send one message, auto-answer
        Requests, aggregate events into an Outcome, tear the session down."""
        handle = self.spawn()
        await handle.commands.put(ev.SendMessage(text=text))
        outcome = Outcome()
        while True:
            event = await handle.events.get()
            if isinstance(event, ev.TextDelta):
                outcome.text += event.text
            elif isinstance(event, ev.ToolResultEvent):
                outcome.tool_results.append(event.result)
            elif isinstance(event, ev.Request):
                decision = "allow" if approve else "deny"
                await handle.commands.put(ev.Respond(event.id, {"decision": decision}))
            elif isinstance(event, ev.Error):
                outcome.error = event.message
                outcome.http_status = event.http_status
                outcome.error_code = event.code
            elif isinstance(event, ev.TurnComplete):
                outcome.stop = event.reason
                await handle.commands.put(ev.Shutdown())
                break
        await handle.task
        return outcome


class _Session:
    """The running agent: owns the conversation, the counters, and the loop."""

    def __init__(self, agent: Agent, events: asyncio.Queue) -> None:
        self._a = agent
        self._events = events
        self._hooks: LifecycleHooks = HookChain(agent.hooks)
        self._rt = RequestCtx(events.put_nowait, agent.request_timeout)
        self._turn_counter = 0
        self._request_counter = 0
        self._pending_getter: asyncio.Task | None = None
        self._tool_loop_state = (
            _ToolLoopState(agent.tool_loop_policy) if agent.tool_loop_policy is not None else None
        )

    # ── session lifecycle ─────────────────────────────────────────────

    def _seed(self) -> tuple[Conversation, bool]:
        """Build the starting conversation: resume from a supported snapshot
        (repair pairing, seed counters — never re-inject persona) or start fresh
        with the persona as the leading system message."""
        snap = self._a.resume
        if snap is not None and snap.version == SNAPSHOT_VERSION:
            messages = list(snap.messages)
            Conversation.repair_pairing(messages)
            convo = Conversation(messages=messages, cache_epoch=snap.cache_epoch)
            derived_turn, derived_req = SessionSnapshot.derive_counters(messages)
            self._turn_counter = max(snap.turn_counter, derived_turn)
            self._request_counter = max(snap.request_counter, derived_req)
            return convo, True
        if snap is not None:
            self._rt.emit(
                ev.Warning(
                    f"unsupported snapshot version {snap.version} "
                    f"(kernel supports {SNAPSHOT_VERSION}); starting empty"
                )
            )
        convo = Conversation()
        if self._a.persona:
            convo.push(Message.system(self._a.persona))
        return convo, False

    def _capture_snapshot(self, convo: Conversation) -> SessionSnapshot:
        """Snapshot with LIVE counter stamps: a turn that died before storing an
        assistant message is invisible to meta derivation, but the counters know."""
        snap = SessionSnapshot.from_conversation(convo)
        snap.turn_counter = max(snap.turn_counter, self._turn_counter)
        snap.request_counter = max(snap.request_counter, self._request_counter)
        return snap

    async def run(self, commands: asyncio.Queue) -> None:
        convo, resumed = self._seed()
        await self._hooks.session_start(convo, resumed)
        pending: deque[AgentCommand] = deque()
        shutdown = False
        while not shutdown:
            command = pending.popleft() if pending else await self._next_command(commands)
            match command:
                case ev.Shutdown():
                    break
                case ev.Cancel():
                    self._rt.cancel_pending()  # no turn running: flush strays
                case ev.Respond(id=rid, value=value):
                    self._rt.resolve(rid, value)
                case ev.RequestSnapshot():
                    self._rt.emit(ev.Snapshot(self._capture_snapshot(convo)))
                case ev.Compact(focus=focus):
                    # TODO(M1 compaction commit): full plan/apply. NoCompaction
                    # (the default) always produces a refused no-op.
                    self._rt.emit(
                        ev.Compacted(
                            trigger=ManualTrigger(focus=focus),
                            epoch=convo.cache_epoch,
                            removed=0,
                            bytes_before=0,
                            bytes_after=0,
                            committed=False,
                        )
                    )
                case ev.SendMessage(text=text, images=images):
                    shutdown = await self._process_prompt(
                        convo, commands, pending, text, images, synthetic=False
                    )
                case ev.SendMessageWithContext(text=text, images=images, context=context):
                    shutdown = await self._process_prompt(
                        convo,
                        commands,
                        pending,
                        text,
                        images,
                        synthetic=False,
                        context=context,
                    )
                case ev.SendSyntheticMessage(text=text):
                    shutdown = await self._process_prompt(
                        convo, commands, pending, text, [], synthetic=True
                    )
        await self._hooks.session_end(convo)
        self._rt.emit(ev.Snapshot(self._capture_snapshot(convo)))
        if self._pending_getter is not None:  # don't leak a parked queue-getter
            self._pending_getter.cancel()
            self._pending_getter = None

    async def _next_command(self, commands: asyncio.Queue) -> AgentCommand:
        if self._pending_getter is not None:
            task, self._pending_getter = self._pending_getter, None
            return await task
        return await commands.get()

    def _getter(self, commands: asyncio.Queue) -> asyncio.Task:
        """Persistent queue-getter: survives across select rounds so a popped
        command is never lost to task cancellation."""
        if self._pending_getter is None:
            self._pending_getter = asyncio.create_task(commands.get())
        return self._pending_getter

    async def _process_prompt(
        self,
        convo: Conversation,
        commands: asyncio.Queue,
        pending: deque,
        text: str,
        images: list,
        synthetic: bool,
        context: str | None = None,
    ) -> bool:
        """Run one prompt as one turn while servicing mid-turn commands
        (Respond/Cancel/Shutdown live; the rest queue FIFO for the boundary).
        Returns True iff a Shutdown was observed."""
        try:
            text = await self._hooks.user_prompt_submit(text)
        except PromptRejected as reason:
            # No turn ran: no turn_start, no turn_complete hook — but the driver
            # still gets a terminal so a blocked prompt can't hang a client.
            self._rt.emit(ev.Error(message=f"prompt rejected: {reason}"))
            self._rt.emit(ev.TurnComplete(StopReason.PROMPT_REJECTED))
            return False
        if not synthetic and self._tool_loop_state is not None:
            # A REAL user submission starts a new intent scope; synthetic
            # continuations keep the accumulated exact-loop evidence.
            self._tool_loop_state.reset()
        # CANCEL = UNDO rollback point: history length BEFORE this turn pushed
        # anything (context and prompt roll back together).
        rollback_len = len(convo.messages)
        if context is not None:  # host-owned context rides the SAME turn
            convo.push(Message.synthetic_user(context))
        if synthetic:
            convo.push(Message.synthetic_user(text))
        else:
            convo.push(Message.user(text, images=images))

        cancel = CancellationToken()  # per-turn; every await races against it
        steer: deque[ev.SteeredInput] = deque()  # mid-turn prompts fold in here
        turn = asyncio.create_task(self._run_turn(convo, cancel, rollback_len, steer))
        shutdown = False
        while not turn.done():
            getter = self._getter(commands)
            await asyncio.wait({turn, getter}, return_when=asyncio.FIRST_COMPLETED)
            if getter.done():
                self._pending_getter = None
                command = getter.result()
                match command:
                    case ev.Respond(id=rid, value=value):
                        self._rt.resolve(rid, value)
                    case ev.Cancel():
                        # Both halves of a parked turn: the token covers stream /
                        # tools / sleeps; flushing pending requests (fail-closed
                        # None) unblocks a middleware approval round-trip the
                        # token cannot reach. A cancelled turn's queued steers
                        # die with it.
                        cancel.cancel()
                        self._rt.cancel_pending()
                        steer.clear()
                    case ev.Shutdown():
                        # Cooperative terminal, never a dropped future: the turn
                        # funnels through its normal cancel path first.
                        shutdown = True
                        cancel.cancel()
                        self._rt.cancel_pending()
                        steer.clear()
                    case ev.SendMessage(text=steer_text, images=steer_images):
                        # Mid-turn user prompt STEERS the running turn (folded at
                        # the next round boundary) instead of queueing a 2nd turn.
                        steer.append(ev.SteeredInput(text=steer_text, images=steer_images))
                    case other:
                        pending.append(other)
        await turn
        # Steers that arrived too late for any round boundary must not vanish:
        # they become ordinary queued prompts.
        for leftover in steer:
            pending.append(ev.SendMessage(text=leftover.text, images=leftover.images))
        return shutdown

    # ── the turn ──────────────────────────────────────────────────────

    async def _finish_turn(self, convo: Conversation, reason: StopReason, ctx: TurnCtx) -> None:
        """The single terminal funnel: hook then event, on EVERY exit path."""
        await self._hooks.turn_complete(convo, reason, ctx)
        self._rt.emit(ev.TurnComplete(reason))

    async def _run_turn(
        self,
        convo: Conversation,
        cancel: CancellationToken,
        rollback_len: int,
        steer: "deque[ev.SteeredInput]",
    ) -> None:
        try:
            await self._run_turn_rounds(convo, cancel, rollback_len, steer)
        except _TurnCancelled as cancelled:
            await self._finish_cancelled(convo, rollback_len, cancelled.args[0])

    async def _finish_cancelled(
        self, convo: Conversation, rollback_len: int, ctx: TurnCtx
    ) -> None:
        """The cancel funnel — every checkpoint lands here exactly once."""
        if self._a.keep_interrupted_context:
            # PRESERVE: keep partial work; pair every dangling call so the wire
            # stays API-valid; a synthetic user-role marker (wire-safe on all
            # adapters, skipped by sacred_floor) tells the model what happened.
            convo.backfill_cancelled_tool_results()
            convo.push(Message.synthetic_user(INTERRUPTION_MARKER))
        else:
            # UNDO (default): the prompt + all partial work leave no trace.
            del convo.messages[rollback_len:]
        self._rt.emit(ev.Cancelled())
        await self._finish_turn(convo, StopReason.CANCELLED, ctx)

    async def _run_turn_rounds(
        self,
        convo: Conversation,
        cancel: CancellationToken,
        rollback_len: int,
        steer: "deque[ev.SteeredInput]",
    ) -> None:
        await self._hooks.turn_start(convo)
        self._rt.emit(ev.TurnStarted())
        tools = dict(self._a.tools)  # per-turn snapshot of the mounted set
        tool_defs = [
            # ToolDef derivation from the Tool port — the menu from the kitchen.
            _tool_def(t)
            for t in tools.values()
        ]
        self._turn_counter += 1
        turn_id = self._turn_counter
        round_no = 0
        continuations = 0
        # Re-armable round cap (grows on a checkpoint "continue").
        round_cap = self._a.max_rounds
        # Coarse repetition fuse state (per turn).
        repeat_sig: str | None = None
        repeat_rounds = 0
        repeat_nudged = False
        # Typed-continuation carry: the NEXT round's response is internal-control.
        active_internal: tuple[ContinuationKind, ContinuationVisibility] | None = None
        # Resilience counters (II.5). Per-round budgets reset on a healthy round;
        # per-turn budgets (rate-limit waits, empty retries, truncation) don't.
        provider_retry = 0
        stream_retry = 0
        rate_limit_waits = 0
        empty_retries = 0
        truncation_continuations = 0

        while True:
            round_no += 1
            self._request_counter += 1
            ctx_window, used_tokens, _ = convo.last_pressure()
            ctx = TurnCtx(
                session_id=self._a.session_id,
                turn_id=turn_id,
                request_id=self._request_counter,
                round=round_no,
                max_rounds=self._a.max_rounds,
                cache_epoch=convo.cache_epoch,
                context_window=ctx_window,
                used_tokens=used_tokens,
            )
            if cancel.is_cancelled:  # between-rounds checkpoint
                raise _TurnCancelled(ctx)
            # ── STEER: fold mid-turn user prompts into THIS turn before the next
            # request — real user messages, append-only. A steer changes the
            # turn's intent, so repetition evidence resets.
            if steer:
                folded: list[ev.SteeredInput] = []
                while steer:
                    folded.append(steer.popleft())
                for item in folded:
                    convo.push(Message.user(item.text, images=list(item.images)))
                self._rt.emit(ev.Steered(count=len(folded), inputs=folded))
                if self._tool_loop_state is not None:
                    self._tool_loop_state.reset()
                repeat_sig = None
                repeat_rounds = 0
                repeat_nudged = False
            if round_cap is not None and round_no > round_cap:
                if self._a.round_cap_checkpoint:
                    answer = await self._rt.request(
                        ROUND_CAP_CHECKPOINT_KIND,
                        {
                            "round": round_no - 1,
                            "cap": round_cap,
                            "base": self._a.max_rounds,
                        },
                    )
                    if isinstance(answer, dict) and answer.get("continue"):
                        # Re-arm by the base amount (round_cap started as
                        # max_rounds, so it is the fallback base when unset).
                        base = self._a.max_rounds if self._a.max_rounds is not None else round_cap
                        round_cap += base
                    else:  # explicit stop OR fail-closed (no answer / timeout)
                        await self._finish_turn(convo, StopReason.MAX_ROUNDS, ctx)
                        return
                else:
                    self._rt.emit(ev.Error(message=f"max rounds ({round_cap}) reached"))
                    await self._finish_turn(convo, StopReason.MAX_ROUNDS, ctx)
                    return

            # Project the request: EPHEMERAL clone; hooks may append reminders.
            messages = list(convo.messages)
            await self._hooks.pre_request(messages, ctx)
            appended_only = (
                len(messages) >= len(convo.messages)
                and messages[: len(convo.messages)] == convo.messages
            )
            Conversation.repair_pairing(messages)
            if not appended_only:
                self._rt.emit(
                    ev.Warning(
                        "pre_request is not append-only: the outgoing prefix diverges "
                        "from stored history — this poisons the provider prefix cache"
                    )
                )
            options = replace(self._a.chat_options)
            options.rate_limit_retry_owner = "kernel"  # loop owns 429 waits
            await self._hooks.pre_request_options(messages, options, ctx)
            await self._hooks.on_request(messages, tool_defs, options, ctx)

            internal = active_internal
            active_internal = None
            suppress = (
                internal is not None and internal[1] is ContinuationVisibility.INTERNAL_CONTROL
            )
            started = self._a.clock.now_millis()
            stream = await self._consume_stream(
                messages, tool_defs, options, cancel, ctx, suppress=suppress
            )

            # ── liveness timeout: reconnect while content-free, else fail ──
            if isinstance(stream, _StreamTimeout):
                if not stream.partial.saw_content and stream_retry < MAX_STREAM_RETRIES:
                    stream_retry += 1
                    self._rt.emit(
                        ev.Warning(
                            f"stream idle timeout — reconnecting "
                            f"({stream_retry}/{MAX_STREAM_RETRIES})"
                        )
                    )
                    await self._sleep(min(0.2 * (2 ** (stream_retry - 1)), 8.0), cancel, ctx)
                    round_no -= 1  # a RETRY of the same logical round
                    continue
                if stream.partial.saw_content:
                    # Replaying after content risks duplicate output/side effects.
                    self._persist_partial(convo, stream.partial, suppress)
                    message = (
                        "stream timeout after partial response; the request "
                        "was not replayed; partial response preserved"
                    )
                else:
                    message = "stream timeout after automatic reconnects"
                await self._hooks.on_error(message)
                self._rt.emit(ev.Error(message=message))
                await self._finish_turn(convo, StopReason.TIMEOUT, ctx)
                return

            # ── provider failure: 429 policy, transient retries, or terminal ──
            if isinstance(stream, _StreamFailure):
                error = stream.error
                if error.http_status == 429:
                    outcome = await self._handle_rate_limit(
                        cancel, convo, ctx, error, stream.partial, suppress, rate_limit_waits
                    )
                    if outcome is None:
                        return  # terminal emitted (pause / fuse / post-content)
                    rate_limit_waits = outcome
                    provider_retry = 0  # 429 never consumes the transient budget
                    round_no -= 1
                    continue
                if not stream.opened and error.retryable and provider_retry < MAX_PROVIDER_RETRIES:
                    provider_retry += 1
                    wait = 3 * provider_retry  # 3/6/9s, visible
                    self._rt.emit(
                        ev.Warning(
                            f"provider error ({error.message}); retrying in {wait}s "
                            f"({provider_retry}/{MAX_PROVIDER_RETRIES})"
                        )
                    )
                    await self._sleep(wait, cancel, ctx)
                    round_no -= 1
                    continue
                if stream.partial.saw_content:
                    self._persist_partial(convo, stream.partial, suppress)
                await self._fail_provider(convo, ctx, error)
                return

            # A healthy round refills the per-round budgets; a natural stream end
            # means any rate-limit incident recovered.
            provider_retry = 0
            stream_retry = 0
            rate_limit_waits = 0

            # ── empty-200 fast retry: a content-free Done is NOT the model
            # choosing to stop — keyed on RAW provider content, so a redacting
            # hook can't make a real response look empty. ──
            if not stream.saw_content and not stream.truncated:
                if empty_retries < EMPTY_RESPONSE_MAX_RETRIES:
                    empty_retries += 1
                    flavor = (
                        "unparseable/garbled response"
                        if stream.saw_malformed
                        else "empty response"
                    )
                    self._rt.emit(
                        ev.Warning(
                            f"model returned an {flavor}; retrying "
                            f"({empty_retries}/{EMPTY_RESPONSE_MAX_RETRIES})"
                        )
                    )
                    await self._sleep(min((empty_retries + 1) // 2, 3), cancel, ctx)
                    round_no -= 1
                    continue
                message = (
                    f"model returned {EMPTY_RESPONSE_MAX_RETRIES} consecutive "
                    "empty responses (upstream fault, not a clean finish); resend "
                    "to try again"
                )
                await self._hooks.on_error(message)
                self._rt.emit(ev.Error(message=message))
                await self._finish_turn(convo, StopReason.PROVIDER_ERROR, ctx)
                return

            text, reasoning, calls = stream.text, stream.reasoning, stream.calls
            usage, truncated = stream.usage, stream.truncated

            # MISROUTED-ANSWER promotion: some gateways put the whole answer into
            # the reasoning channel and leave content empty. A stop-finish round
            # with ONLY plain-text reasoning promotes it to the body — through the
            # same on_text_delta seam a normal delta passes, so redaction holds.
            # Signed-block rounds are excluded (promoting would desync the blocks).
            if (
                not text
                and reasoning.strip()
                and not calls
                and not truncated
                and not stream.reasoning_blocks
            ):
                promoted = await self._hooks.on_text_delta(reasoning)
                if promoted:
                    text = promoted
                    reasoning = ""
                    if not suppress:
                        self._rt.emit(ev.TextDelta(promoted))

            finish_reason = "tool_calls" if calls else ("length" if truncated else "stop")
            window = self._a.provider.context_window()
            used = usage.prompt if usage.prompt > 0 else sum(m.estimate_tokens() for m in messages)
            assistant = Message.assistant(text, calls)
            assistant.reasoning = reasoning or None
            assistant.reasoning_blocks = stream.reasoning_blocks
            if internal is not None and internal[0] is ContinuationKind.VERIFY_CADENCE:
                # Control chatter stays out of user-visible history.
                assistant.internal_origin = "verify_cadence"
                assistant.text = ""
                assistant.reasoning = None
                assistant.reasoning_blocks = []
            assistant.meta = MessageMeta(
                tokens=usage,
                elapsed_ms=self._a.clock.now_millis() - started,
                reasoning_elapsed_ms=stream.reasoning_elapsed_ms,
                ctx_window=window,
                used_tokens=used,
                utilization=(used / window) if window > 0 else 0.0,
                round=round_no,
                turn_id=turn_id,
                request_id=ctx.request_id,
                provider_response_id=stream.response_id,
                provider_model=stream.response_model,
                session_id=self._a.session_id,
                finish_reason=finish_reason,
            )
            await self._hooks.on_model_response(assistant)
            self._rt.emit(ev.Usage(assistant.meta))
            calls = list(assistant.tool_calls)  # re-read: a dropped call never executes
            convo.push(assistant)

            if calls:
                stop, fingerprint = await self._dispatch_tools(
                    cancel, ctx, convo, calls, tools, turn_id, round_no
                )
                if stop is not None:
                    await self._finish_turn(convo, stop, ctx)
                    return
                if steer:
                    # A pending steer is real user intent: honor it before any
                    # repetition verdict — the evidence chain resets anyway.
                    if self._tool_loop_state is not None:
                        self._tool_loop_state.reset()
                    repeat_sig = None
                    repeat_rounds = 0
                    repeat_nudged = False
                    continue

                # ── exact no-progress guard (opt-in; owns the streak) ──
                exact_active = False
                if self._tool_loop_state is not None:
                    state = self._tool_loop_state
                    verdict = "continue"
                    if fingerprint is not None:
                        verdict = state.observe(fingerprint)
                    else:
                        state.reset()  # ineligible batch breaks the evidence chain
                    exact_active = state.consecutive > 1
                    if verdict == "warn":
                        self._rt.emit(
                            ev.Warning(
                                "possible tool loop: identical call and result "
                                f"{state.policy.warning_threshold} times; asking the "
                                "model to change course"
                            )
                        )
                        convo.push(Message.synthetic_user(TOOL_LOOP_NUDGE))
                    elif verdict == "stop":
                        self._rt.emit(
                            ev.Warning(
                                "tool loop detected: identical call and result "
                                f"{state.policy.stop_threshold} times; stopping"
                            )
                        )
                        await self._finish_turn(convo, StopReason.TOOL_LOOP_DETECTED, ctx)
                        return

                # ── always-on coarse repetition fuse ──
                signature = "\x01".join(
                    sorted(f"{c.name}\x00{_canonicalize_args(c.arguments)}" for c in calls)
                )
                if signature == repeat_sig:
                    repeat_rounds += 1
                else:
                    repeat_sig = signature
                    repeat_rounds = 1
                    repeat_nudged = False
                if not exact_active:  # an active exact streak owns these rounds
                    if repeat_rounds >= MAX_REPEAT_ROUNDS:
                        self._rt.emit(
                            ev.Error(
                                message=(
                                    "stopped: the model repeated the same tool-call "
                                    f"pattern for {repeat_rounds} consecutive rounds"
                                )
                            )
                        )
                        await self._finish_turn(convo, StopReason.REPEAT_LOOP, ctx)
                        return
                    if repeat_rounds >= REPEAT_NUDGE_AT and not repeat_nudged:
                        repeat_nudged = True
                        convo.push(Message.synthetic_user(REPEAT_LOOP_NUDGE))
                continue

            # A no-tool round is an observable break in repetition evidence.
            if self._tool_loop_state is not None:
                self._tool_loop_state.reset()
            repeat_sig = None
            repeat_rounds = 0
            repeat_nudged = False

            # A prompt steered in during this round keeps the turn alive: loop
            # back so the round-top drain folds it in and the model answers it.
            if steer:
                continue

            # ── truncation auto-continuation: output cut at the token limit
            # with no tool call is almost always unfinished work. Runs BEFORE
            # offer_continuation so a discipline hook can't pre-empt finishing. ──
            if truncated and truncation_continuations < MAX_TRUNCATION_CONTINUATIONS:
                truncation_continuations += 1
                convo.push(Message.synthetic_user(TRUNCATION_RESUME_NUDGE))
                continue

            continuation = await self._hooks.offer_typed_continuation(convo)
            if continuation is not None:
                if (
                    self._a.max_continuations is not None
                    and continuations >= self._a.max_continuations
                ):
                    self._rt.emit(
                        ev.Error(
                            message=(
                                f"max offer_continuation continuations "
                                f"({self._a.max_continuations}) reached"
                            )
                        )
                    )
                    await self._finish_turn(convo, StopReason.MAX_CONTINUATIONS, ctx)
                    return
                continuations += 1
                active_internal = (continuation.kind, continuation.visibility)
                convo.push(Message.synthetic_user(continuation.text))
                continue

            if truncated:
                # Unrecovered truncation ending the turn — the one case the user
                # must see: real work was cut off and is not being finished.
                self._rt.emit(ev.Warning("response truncated: finish_reason=length"))
            await self._finish_turn(convo, StopReason.STOPPED, ctx)
            return

    async def _consume_stream(
        self,
        messages,
        tool_defs,
        options,
        cancel: CancellationToken,
        ctx: TurnCtx,
        suppress: bool = False,
    ) -> "_RoundStream | _StreamFailure | _StreamTimeout":
        """One provider call: open + consume. Emits NO terminals — failures and
        timeouts return typed outcomes for _run_turn's retry policy to judge;
        a cancel mid-stream raises _TurnCancelled (nothing dangles: the round's
        accumulation is discarded, the funnel repairs history). suppress=True
        (an INTERNAL_CONTROL continuation round) keeps text/reasoning off the
        live stream AND out of the accumulation."""
        out = _RoundStream()
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        block_parts: list[str] = []  # buffer for the CURRENT signed block
        reasoning_started: int | None = None
        opened = False  # any event received => the open succeeded

        def _close_reasoning_phase() -> None:
            nonlocal reasoning_started
            if reasoning_started is not None:
                out.reasoning_elapsed_ms += self._a.clock.now_millis() - reasoning_started
                reasoning_started = None

        def _finalize() -> None:
            _close_reasoning_phase()
            out.text = "".join(text_parts)
            out.reasoning = "".join(reasoning_parts)

        iterator = aiter(self._a.provider.chat_stream(messages, tool_defs, options))
        cancel_wait = asyncio.ensure_future(cancel.cancelled())
        try:
            while True:
                # Race the next event against cancel and the liveness timeout —
                # every await inside the stream is bounded and cancellable.
                next_event = asyncio.ensure_future(anext(iterator))
                timeout = self._a.stream_timeout
                if timeout is not None and not opened:
                    timeout *= FIRST_EVENT_TIMEOUT_FACTOR  # prefill allowance
                done, _ = await asyncio.wait(
                    {next_event, cancel_wait},
                    timeout=timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cancel_wait in done:
                    next_event.cancel()
                    raise _TurnCancelled(ctx)
                if not done:  # liveness timeout: no event within stream_timeout
                    next_event.cancel()
                    _finalize()
                    return _StreamTimeout(partial=out)
                try:
                    event = next_event.result()
                except StopAsyncIteration:
                    break
                except ProviderError as error:
                    _finalize()
                    return _StreamFailure(error=error, opened=opened, partial=out)
                opened = True
                match event:
                    case TextDelta(text=delta):
                        out.saw_content = True  # RAW arrival, before any hook —
                        _close_reasoning_phase()  # a clearing hook != empty 200
                        delta = await self._hooks.on_text_delta(delta)
                        if delta and not suppress:
                            text_parts.append(delta)
                            self._rt.emit(ev.TextDelta(delta))
                    case Reasoning(text=delta):
                        out.saw_content = True
                        if reasoning_started is None:
                            reasoning_started = self._a.clock.now_millis()
                        delta = await self._hooks.on_reasoning_delta(delta)
                        if delta and not suppress:
                            reasoning_parts.append(delta)
                            block_parts.append(delta)  # post-hook bytes: a stored
                            self._rt.emit(ev.Reasoning(delta))  # block stays redacted
                    case ReasoningSignature(opaque=opaque, provider=provider):
                        # Finalize ONE signed block: the text since the previous
                        # boundary. A redacted block has no preceding deltas ->
                        # empty text. Pure storage — the text already streamed.
                        out.saw_content = True
                        if reasoning_started is None:
                            reasoning_started = self._a.clock.now_millis()
                        if not suppress:
                            out.reasoning_blocks.append(
                                ReasoningBlock(
                                    text="".join(block_parts),
                                    opaque=opaque,
                                    provider=provider,
                                )
                            )
                            block_parts.clear()
                    case ToolCallEvent(call=call):
                        out.saw_content = True
                        _close_reasoning_phase()
                        out.calls.append(call)
                    case ToolCallDelta():
                        out.saw_content = True
                        _close_reasoning_phase()
                        self._rt.emit(
                            ev.ToolCallStreaming(
                                index=event.index,
                                id=event.id,
                                name=event.name,
                                arguments=event.arguments,
                            )
                        )
                    case UsageEvent(usage=u):
                        out.usage.merge_max(u)
                    case ResponseId(id=rid):
                        out.response_id = rid
                    case ResponseModel(model=model):
                        out.response_model = model
                    case ErrorEvent(error=error):
                        _finalize()
                        return _StreamFailure(error=error, opened=True, partial=out)
                    case Malformed():
                        out.saw_malformed = True  # flavors the empty-retry wording
                    case Done(truncated=t):
                        out.truncated = t
                        break
        finally:
            cancel_wait.cancel()  # never leak the parked waiter
        _finalize()
        return out

    async def _fail_provider(self, convo, ctx, error: ProviderError) -> None:
        await self._hooks.on_error(error.message)
        self._rt.emit(
            ev.Error(
                message=error.message,
                http_status=error.http_status,
                code=error.code,
            )
        )
        await self._finish_turn(convo, StopReason.PROVIDER_ERROR, ctx)

    async def _sleep(self, seconds: float, cancel: CancellationToken, ctx: TurnCtx) -> None:
        """Every retry/backoff wait, scaled (tests set backoff_scale=0) and raced
        against cancel so Esc never sits out a backoff."""
        sleep = asyncio.ensure_future(asyncio.sleep(seconds * self._a.backoff_scale))
        cancel_wait = asyncio.ensure_future(cancel.cancelled())
        done, _ = await asyncio.wait({sleep, cancel_wait}, return_when=asyncio.FIRST_COMPLETED)
        sleep.cancel()
        cancel_wait.cancel()
        if cancel_wait in done:
            raise _TurnCancelled(ctx)

    def _persist_partial(
        self, convo: Conversation, partial: "_RoundStream", suppress: bool
    ) -> None:
        """Persist the replay-safe portion of a failed stream: the assembled
        assistant message (deduped complete calls), with every dangling call
        paired to an '(interrupted before execution)' error result so a resume
        never re-executes or sends an illegal payload."""
        seen: set[str] = set()
        calls = [c for c in partial.calls if not (c.id in seen or seen.add(c.id))]
        if not partial.text and not partial.reasoning and not calls:
            return
        message = Message.assistant(partial.text, calls)
        if suppress:  # control-round chatter stays invisible even when partial
            message.text = ""
            message.reasoning = None
        else:
            message.reasoning = partial.reasoning or None
            message.reasoning_blocks = partial.reasoning_blocks
        convo.push(message)
        convo.backfill_interrupted_tool_results()

    async def _handle_rate_limit(
        self,
        cancel: CancellationToken,
        convo: Conversation,
        ctx: TurnCtx,
        error: ProviderError,
        partial: "_RoundStream",
        suppress: bool,
        waits: int,
    ) -> int | None:
        """The 429 path. Returns the updated wait count when the round should be
        RE-ISSUED after a (possibly silent) wait; None when a terminal was
        emitted (post-content preserve, pause verdict, or the waits fuse)."""
        server_message = error.message.strip() or None
        # Once content reached the driver, a replay could duplicate output that
        # cannot be retracted — preserve the partial and pause cleanly.
        if partial.saw_content:
            self._persist_partial(convo, partial, suppress)
            self._rt.emit(
                ev.RateLimited(
                    auto_resuming=False,
                    server_message=server_message,
                )
            )
            await self._finish_turn(convo, StopReason.RATE_LIMITED, ctx)
            return None
        hint = RateLimitHint(
            http_status=error.http_status,
            retry_after_secs=_effective_retry_after(error),
            terminal=_is_terminal_rate_limit(error),
            attempt=waits + 1,
        )
        verdict = None if hint.terminal else await self._hooks.on_rate_limit(hint)
        jitter = (self._a.clock.now_millis() % 1000) / 1000  # deterministic w/ fakes
        decision = verdict if verdict is not None else RateLimitDecision.from_hint(hint, jitter)
        if decision.wait_secs is None:  # Pause: reset too far / terminal billing
            self._rt.emit(
                ev.RateLimited(
                    reset_at_display=decision.reset_at_display,
                    reset_label=decision.reset_label,
                    secs_until_reset=decision.secs_until_reset,
                    auto_resuming=False,
                    server_message=server_message,
                )
            )
            await self._finish_turn(convo, StopReason.RATE_LIMITED, ctx)
            return None
        waits += 1
        if waits > MAX_RATE_LIMIT_WAITS:  # livelock fuse: force a clean pause
            self._rt.emit(
                ev.RateLimited(
                    auto_resuming=False,
                    server_message=server_message,
                )
            )
            await self._finish_turn(convo, StopReason.RATE_LIMITED, ctx)
            return None
        quiet_first = verdict is None and hint.retry_after_secs is None and waits == 1
        if quiet_first:  # a one-off burst clears without banner spam
            await self._sleep(SILENT_FIRST_RATE_LIMIT_WAIT, cancel, ctx)
        else:
            self._rt.emit(
                ev.RateLimited(
                    secs_until_reset=decision.wait_secs,
                    auto_resuming=True,
                )
            )
            await self._sleep(decision.wait_secs, cancel, ctx)
        return waits

    # ── tool dispatch: three phases (II.3) ────────────────────────────

    async def _dispatch_tools(
        self,
        cancel: CancellationToken,
        ctx: TurnCtx,
        convo: Conversation,
        calls: list[ToolCall],
        tools: dict[str, Tool],
        turn_id: int,
        round_no: int,
    ) -> tuple[StopReason | None, tuple | None]:
        """① classify → ② execute (concurrent) → ③ apply (in order).
        Returns (stop, fingerprint): stop=POLICY_DENIED when a DENY_TURN fired;
        fingerprint is the exact-loop evidence (sorted per-call tool + canonical
        args + post-cap result + success state) when the batch is ELIGIBLE — one
        real execution or an all-parallel-safe batch, no stubs/blocks/images —
        else None (ineligible batches break the streak)."""
        # ── Phase ① CLASSIFY (in emission order) ──
        # Cancel checkpoint before any classification/batch perception: nothing
        # started, so nothing dangles — the funnel repairs the assistant message.
        if cancel.is_cancelled:
            raise _TurnCancelled(ctx)
        plans: list[_Skip | _Ready | _Execute] = []
        result_ids: set[str] = set()  # call_ids already resulted this batch (mode A)
        seen_calls: set[tuple[str, str]] = set()  # executed (name, canonical args) (mode B)
        deny_turn_seen = False
        for call in calls:
            # Dedup keys use the MODEL's original bytes, captured BEFORE any
            # middleware rewrite: two model-identical calls are duplicates
            # regardless of what middleware would later do to them.
            dedup_key = (call.name, _canonicalize_args(call.arguments))
            if call.id in result_ids:
                # Mode A: a second result for one id is an illegal payload — skip
                # ENTIRELY (no execute, no result row, nothing to repair).
                plans.append(_Skip())
                continue
            if deny_turn_seen:
                result_ids.add(call.id)
                plans.append(
                    _Ready(
                        ToolResult(
                            call_id=call.id,
                            content="blocked: another call in this batch terminated the turn by policy",
                            is_error=True,
                        )
                    )
                )
                continue
            if dedup_key in seen_calls:
                # Mode B: same tool+args under a NEW id — don't re-execute; a stub
                # keeps this id paired (API-valid) without repeating side effects.
                result_ids.add(call.id)
                plans.append(
                    _Ready(
                        ToolResult(
                            call_id=call.id,
                            content="[duplicate call — identical tool and arguments to an "
                            "earlier call this turn; result already returned above]",
                        )
                    )
                )
                continue
            tool = tools.get(call.name)
            if tool is None:
                result_ids.add(call.id)  # id paired; (name,args) NOT recorded — a
                plans.append(
                    _Ready(
                        ToolResult(  # later retry may legitimately run
                            call_id=call.id,
                            content=f"unknown or unmounted tool: {call.name}",
                            is_error=True,
                        )
                    )
                )
                continue
            blocked: tuple[str, bool] | None = None
            for mw in self._a.middleware:
                outcome = await mw.before(call, tool, self._rt)
                if outcome.gate is Gate.ALLOW:
                    break  # force-approve: skip remaining gates
                if outcome.gate is Gate.DENY:
                    blocked = (outcome.reason, False)
                    break
                if outcome.gate is Gate.DENY_TURN:
                    blocked = (outcome.reason, True)
                    break
                # PROCEED and ASK both continue the chain: the kernel owns no
                # prompt — a bare ASK defers to a downstream approval middleware.
            if blocked is not None:
                reason, terminate = blocked
                result_ids.add(call.id)
                plans.append(
                    _Ready(
                        ToolResult(call_id=call.id, content=f"blocked: {reason}", is_error=True),
                        terminate_turn=terminate,
                    )
                )
                if terminate:
                    deny_turn_seen = True
                    for i, prior in enumerate(plans[:-1]):
                        if isinstance(prior, _Execute):
                            plans[i] = _Ready(
                                ToolResult(
                                    call_id=prior.call.id,
                                    content="blocked: another call in this batch "
                                    "terminated the turn by policy",
                                    is_error=True,
                                )
                            )
                continue
            result_ids.add(call.id)
            seen_calls.add(dedup_key)
            plans.append(
                _Execute(tool=tool, call=call, parallel_safe=tool.parallel_safe(call.arguments))
            )

        # ── Batch perception: >= 2 non-duplicate calls -> grouped block ──
        distinct = len(result_ids)
        batch_id: str | None = None
        batch_started_at = 0
        if distinct >= 2:
            batch_id = f"batch_{turn_id}_{round_no}"
            batch_started_at = self._a.clock.now_millis()
            self._rt.emit(
                ev.ToolBatchStarted(
                    batch_id=batch_id,
                    calls=[
                        ev.ToolBatchCall(
                            id=p.call.id,
                            name=p.call.name,
                            arguments=p.call.arguments,
                            parallel_safe=p.parallel_safe,
                        )
                        for p in plans
                        if isinstance(p, _Execute)
                    ],
                )
            )

        # ── Phase ② EXECUTE ──
        # Consecutive parallel-safe plans run concurrently (bounded by the
        # semaphore); a side-effecting plan is an exclusive barrier. Results land
        # in plan order either way, so Phase ③ applies in emission order.
        results: list[ToolResult | None] = [
            p.result if isinstance(p, _Ready) else None for p in plans
        ]
        semaphore = asyncio.Semaphore(max(1, self._a.max_parallel_tools))
        i = 0
        while i < len(plans):
            plan = plans[i]
            if not isinstance(plan, _Execute):
                i += 1
                continue
            if plan.parallel_safe:
                group: list[tuple[int, _Execute]] = [(i, plan)]
                j = i + 1
                while j < len(plans):
                    nxt = plans[j]
                    if isinstance(nxt, _Execute) and not nxt.parallel_safe:
                        break  # a writer closes the concurrent window
                    if isinstance(nxt, _Execute):
                        group.append((j, nxt))
                    j += 1
                gathered = await asyncio.gather(
                    *[self._execute_one(execute, semaphore, cancel) for _, execute in group]
                )
                for (k, _), result in zip(group, gathered, strict=True):
                    results[k] = result
                i = j
            else:
                results[i] = await self._execute_one(plan, semaphore, cancel)
                i += 1

        # ── Phase ③ APPLY (in order) ──
        policy_denied = False
        batch_ok = 0
        lifted_images: list = []
        # Exact-loop eligibility: one real execution, or an all-parallel-safe
        # batch (emission order isn't semantic progress there). Stubs, blocks,
        # unknown tools, images, and post-blocks are ineligible.
        executes = [p for p in plans if isinstance(p, _Execute)]
        loop_eligible = 0 < len(plans) == len(executes) and (
            len(executes) == 1 or all(p.parallel_safe for p in executes)
        )
        loop_entries: list[tuple] = []
        for plan, result in zip(plans, results, strict=True):
            if isinstance(plan, _Skip) or result is None:
                continue
            post_block: str | None = None
            for mw in self._a.middleware:
                after = await mw.after(result)  # sees the RAW, pre-cap result
                if after.block_reason is not None and post_block is None:
                    post_block = after.block_reason
            _cap_tool_result(result, self._a.max_tool_result_bytes)
            if loop_eligible and isinstance(plan, _Execute):
                if result.images or post_block is not None:
                    loop_eligible = False
                else:
                    loop_entries.append(
                        (
                            plan.call.name,
                            _canonicalize_args(plan.call.arguments),
                            result.content,
                            result.is_error,
                        )
                    )
            if result.is_error:
                await self._hooks.on_error(result.content)
            elif batch_id is not None:
                batch_ok += 1
            if result.images:
                lifted_images.extend(result.images)
                result.images = []  # transient: never stored on the tool message
            self._rt.emit(ev.ToolResultEvent(result))
            convo.push(Message.tool_result(result.call_id, result.content, result.is_error))
            if post_block is not None:
                convo.push(Message.synthetic_user(post_block))
            if isinstance(plan, _Ready) and plan.terminate_turn:
                policy_denied = True

        if batch_id is not None:
            self._rt.emit(
                ev.ToolBatchCompleted(
                    batch_id=batch_id,
                    ok=batch_ok,
                    total=distinct,
                    elapsed_ms=self._a.clock.now_millis() - batch_started_at,
                )
            )
        if lifted_images:
            # Providers reject images on the tool role: ONE synthetic user message
            # after all results keeps the assistant's calls contiguous (API-valid).
            convo.push(
                Message.synthetic_user(
                    "[Images returned by the tool calls above are attached for you to view.]",
                    images=lifted_images,
                )
            )
        if cancel.is_cancelled:
            # Results that DID complete were applied above (their events fired);
            # now the cancel terminal takes over — the funnel repairs pairing.
            raise _TurnCancelled(ctx)
        fingerprint = tuple(sorted(loop_entries)) if loop_eligible and loop_entries else None
        return (StopReason.POLICY_DENIED if policy_denied else None, fingerprint)

    async def _execute_one(
        self, plan: "_Execute", semaphore: asyncio.Semaphore, cancel: CancellationToken
    ) -> ToolResult:
        async with semaphore:
            call = plan.call
            if cancel.is_cancelled:  # cancelled while queued behind the semaphore
                return ToolResult(
                    call_id=call.id,
                    content="(cancelled — never started)",
                    is_error=True,
                )
            self._rt.emit(ev.ToolStarted(call))
            tctx = ToolContext(
                working_dir=self._a.working_dir,
                cancel=cancel,  # cooperative: a long tool polls / awaits it
                progress=ProgressSink(
                    lambda message, call_id=call.id: self._rt.emit(
                        ev.ToolProgress(call_id=call_id, message=message)
                    )
                ),
                requester=self._rt.requester(),
            )
            # Race execute against cancel, execute-first biased: a tool that
            # already completed keeps its real result. A tool still pending when
            # cancel fires is dropped as a backstop — side effects unknown, and
            # the synthetic result says so.
            execute = asyncio.ensure_future(plan.tool.execute(call.arguments, tctx))
            cancel_wait = asyncio.ensure_future(cancel.cancelled())
            try:
                done, _ = await asyncio.wait(
                    {execute, cancel_wait}, return_when=asyncio.FIRST_COMPLETED
                )
                if execute not in done:
                    execute.cancel()
                    return ToolResult(
                        call_id=call.id,
                        content="(cancelled — side effects unknown)",
                        is_error=True,
                    )
                result = execute.result()
            except Exception as error:  # noqa: BLE001 — trust model: ANY tool failure becomes data
                return ToolResult(
                    call_id=call.id,
                    content=f"{type(error).__name__}: {error}",
                    is_error=True,
                )
            finally:
                cancel_wait.cancel()
            result.call_id = call.id
            return result


def _tool_def(tool: Tool) -> ToolDef:
    return ToolDef(
        name=tool.name(),
        description=tool.description(),
        parameters=tool.parameters_schema(),
    )


@dataclass
class _StreamFailure:
    """A provider failure this round. opened=False (raised before any event) may
    enter the transient retry tier; opened=True (mid-stream) is terminal for
    non-429s. 429s route to rate-limit policy either way."""

    error: ProviderError
    opened: bool
    partial: "_RoundStream"


@dataclass
class _StreamTimeout:
    """The liveness stream_timeout elapsed waiting for the next event."""

    partial: "_RoundStream"


@dataclass
class _RoundStream:
    """One round's accumulated stream: post-hook text/reasoning, signed blocks,
    calls, usage, and the RAW-content flags the retry tiers key on."""

    text: str = ""
    reasoning: str = ""
    reasoning_blocks: list[ReasoningBlock] = field(default_factory=list)
    calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    truncated: bool = False
    response_id: str | None = None
    response_model: str | None = None
    reasoning_elapsed_ms: int = 0
    # Did the PROVIDER stream any content (pre-hook)? A redacting hook that
    # clears every chunk must not make a real response look like an empty 200.
    saw_content: bool = False
    saw_malformed: bool = False


# ── dispatch plan variants (Phase ① output) ────────────────────────────


@dataclass
class _Skip:
    """Mode-A duplicate (same call_id already resulted) — no result row at all."""


@dataclass
class _Ready:
    """A ready-to-apply result: mode-B stub, middleware block, or unknown tool."""

    result: ToolResult
    terminate_turn: bool = False


@dataclass
class _Execute:
    """Runs in Phase ②. parallel_safe captured at classification time."""

    tool: Tool
    call: ToolCall
    parallel_safe: bool


def _canonicalize_args(arguments: str) -> str:
    """Call-identity canonicalization: object-key order and insignificant
    whitespace don't change identity; array order and malformed input still do."""
    try:
        return json.dumps(json.loads(arguments), sort_keys=True, separators=(",", ":"))
    except (ValueError, TypeError):
        return arguments


def _cap_tool_result(result: ToolResult, max_bytes: int) -> None:
    """Enforce the kernel's tool-result size cap IN PLACE: keep head + tail
    (signal usually lives at both ends), splice an elision marker between.
    Deterministic — same content + cap → byte-identical output (prefix-cache
    safe). max_bytes == 0 disables (unbounded)."""
    if max_bytes == 0:
        return
    raw = result.content.encode("utf-8")
    total = len(raw)
    if total <= max_bytes:
        return
    half = max_bytes // 2
    head = raw[:half].decode("utf-8", errors="ignore")
    tail = raw[total - half :].decode("utf-8", errors="ignore")
    elided = total - len(head.encode("utf-8")) - len(tail.encode("utf-8"))
    result.content = (
        f"{head}\n…[truncated: {elided} of {total} bytes elided by kernel cap]…\n{tail}"
    )
