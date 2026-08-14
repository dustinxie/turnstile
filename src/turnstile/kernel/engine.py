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
    Reasoning,
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
    request_timeout: float | None = None

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
        if context is not None:  # host-owned context rides the SAME turn
            convo.push(Message.synthetic_user(context))
        if synthetic:
            convo.push(Message.synthetic_user(text))
        else:
            convo.push(Message.user(text, images=images))

        turn = asyncio.create_task(self._run_turn(convo))
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
                        self._rt.cancel_pending()  # TODO(cancel commit): turn token
                    case ev.Shutdown():
                        shutdown = True
                        self._rt.cancel_pending()
                    case other:
                        pending.append(other)
        await turn
        return shutdown

    # ── the turn ──────────────────────────────────────────────────────

    async def _finish_turn(self, convo: Conversation, reason: StopReason, ctx: TurnCtx) -> None:
        """The single terminal funnel: hook then event, on EVERY exit path."""
        await self._hooks.turn_complete(convo, reason, ctx)
        self._rt.emit(ev.TurnComplete(reason))

    async def _run_turn(self, convo: Conversation) -> None:
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
            await self._hooks.pre_request_options(messages, options, ctx)
            await self._hooks.on_request(messages, tool_defs, options, ctx)

            internal = active_internal
            active_internal = None
            suppress = (
                internal is not None and internal[1] is ContinuationVisibility.INTERNAL_CONTROL
            )
            started = self._a.clock.now_millis()
            outcome = await self._consume_stream(
                messages, tool_defs, options, convo, ctx, suppress=suppress
            )
            if outcome is None:
                return  # terminal already emitted
            text, reasoning, calls, usage, truncated, response_id, response_model = outcome

            finish_reason = "tool_calls" if calls else ("length" if truncated else "stop")
            window = self._a.provider.context_window()
            used = usage.prompt if usage.prompt > 0 else sum(m.estimate_tokens() for m in messages)
            assistant = Message.assistant(text, calls)
            assistant.reasoning = reasoning or None
            if internal is not None and internal[0] is ContinuationKind.VERIFY_CADENCE:
                # Control chatter stays out of user-visible history.
                assistant.internal_origin = "verify_cadence"
                assistant.text = ""
                assistant.reasoning = None
            assistant.meta = MessageMeta(
                tokens=usage,
                elapsed_ms=self._a.clock.now_millis() - started,
                ctx_window=window,
                used_tokens=used,
                utilization=(used / window) if window > 0 else 0.0,
                round=round_no,
                turn_id=turn_id,
                request_id=ctx.request_id,
                provider_response_id=response_id,
                provider_model=response_model,
                session_id=self._a.session_id,
                finish_reason=finish_reason,
            )
            await self._hooks.on_model_response(assistant)
            self._rt.emit(ev.Usage(assistant.meta))
            calls = list(assistant.tool_calls)  # re-read: a dropped call never executes
            convo.push(assistant)

            if calls:
                stop, fingerprint = await self._dispatch_tools(
                    convo, calls, tools, turn_id, round_no
                )
                if stop is not None:
                    await self._finish_turn(convo, stop, ctx)
                    return

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

            await self._finish_turn(convo, StopReason.STOPPED, ctx)
            return

    async def _consume_stream(
        self, messages, tool_defs, options, convo, ctx, suppress: bool = False
    ):
        """One provider call: open + consume. Returns the round's accumulation,
        or None if a terminal was already emitted. suppress=True (an
        INTERNAL_CONTROL continuation round) keeps text/reasoning off the live
        stream AND out of the accumulation — control chatter is invisible.
        Retry tiers land later — today any provider failure is a clean
        PROVIDER_ERROR."""
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        calls: list = []
        usage = TokenUsage()
        truncated = False
        response_id: str | None = None
        response_model: str | None = None
        try:
            stream = self._a.provider.chat_stream(messages, tool_defs, options)
            async for event in stream:
                match event:
                    case TextDelta(text=delta):
                        delta = await self._hooks.on_text_delta(delta)
                        if delta and not suppress:
                            text_parts.append(delta)
                            self._rt.emit(ev.TextDelta(delta))
                    case Reasoning(text=delta):
                        delta = await self._hooks.on_reasoning_delta(delta)
                        if delta and not suppress:
                            reasoning_parts.append(delta)
                            self._rt.emit(ev.Reasoning(delta))
                    case ToolCallEvent(call=call):
                        calls.append(call)
                    case ToolCallDelta():
                        self._rt.emit(
                            ev.ToolCallStreaming(
                                index=event.index,
                                id=event.id,
                                name=event.name,
                                arguments=event.arguments,
                            )
                        )
                    case UsageEvent(usage=u):
                        usage.merge_max(u)
                    case ResponseId(id=rid):
                        response_id = rid
                    case ResponseModel(model=model):
                        response_model = model
                    case ErrorEvent(error=error):
                        await self._fail_provider(convo, ctx, error)
                        return None
                    case Malformed():
                        pass  # diagnostic only; empty-retry flavor lands later
                    case Done(truncated=t):
                        truncated = t
                        break
        except ProviderError as error:
            await self._fail_provider(convo, ctx, error)
            return None
        return (
            "".join(text_parts),
            "".join(reasoning_parts),
            calls,
            usage,
            truncated,
            response_id,
            response_model,
        )

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

    # ── tool dispatch: three phases (II.3) ────────────────────────────

    async def _dispatch_tools(
        self,
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
                    *[self._execute_one(execute, semaphore) for _, execute in group]
                )
                for (k, _), result in zip(group, gathered, strict=True):
                    results[k] = result
                i = j
            else:
                results[i] = await self._execute_one(plan, semaphore)
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
        fingerprint = tuple(sorted(loop_entries)) if loop_eligible and loop_entries else None
        return (StopReason.POLICY_DENIED if policy_denied else None, fingerprint)

    async def _execute_one(self, plan: "_Execute", semaphore: asyncio.Semaphore) -> ToolResult:
        async with semaphore:
            call = plan.call
            self._rt.emit(ev.ToolStarted(call))
            tctx = ToolContext(
                working_dir=self._a.working_dir,
                progress=ProgressSink(
                    lambda message, call_id=call.id: self._rt.emit(
                        ev.ToolProgress(call_id=call_id, message=message)
                    )
                ),
                requester=self._rt.requester(),
            )
            try:
                result = await plan.tool.execute(call.arguments, tctx)
            except Exception as error:  # noqa: BLE001 — trust model: ANY tool failure becomes data
                return ToolResult(
                    call_id=call.id,
                    content=f"{type(error).__name__}: {error}",
                    is_error=True,
                )
            result.call_id = call.id
            return result


def _tool_def(tool: Tool) -> ToolDef:
    return ToolDef(
        name=tool.name(),
        description=tool.description(),
        parameters=tool.parameters_schema(),
    )


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
