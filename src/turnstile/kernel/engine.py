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
from collections import deque
from dataclasses import dataclass, field, replace

from turnstile.kernel import events as ev
from turnstile.kernel.dtos import (
    SNAPSHOT_VERSION,
    ChatOptions,
    Conversation,
    Done,
    ErrorEvent,
    Malformed,
    ManualTrigger,
    Message,
    MessageMeta,
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
from turnstile.kernel.events import AgentCommand, Outcome, RequestCtx
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
            if self._a.max_rounds is not None and round_no > self._a.max_rounds:
                self._rt.emit(ev.Error(message=f"max rounds ({self._a.max_rounds}) reached"))
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

            started = self._a.clock.now_millis()
            outcome = await self._consume_stream(messages, tool_defs, options, convo, ctx)
            if outcome is None:
                return  # terminal already emitted
            text, reasoning, calls, usage, truncated, response_id, response_model = outcome

            finish_reason = "tool_calls" if calls else ("length" if truncated else "stop")
            window = self._a.provider.context_window()
            used = usage.prompt if usage.prompt > 0 else sum(m.estimate_tokens() for m in messages)
            assistant = Message.assistant(text, calls)
            assistant.reasoning = reasoning or None
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
                # Degenerate dispatch (full three-phase lands next commit):
                # every call resolves; unknown tools produce error results.
                for call in calls:
                    result = await self._execute_unknown(call, tools)
                    if result.is_error:
                        await self._hooks.on_error(result.content)
                    self._rt.emit(ev.ToolResultEvent(result))
                    convo.push(
                        Message.tool_result(result.call_id, result.content, result.is_error)
                    )
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
                convo.push(Message.synthetic_user(continuation.text))
                continue

            await self._finish_turn(convo, StopReason.STOPPED, ctx)
            return

    async def _consume_stream(self, messages, tool_defs, options, convo, ctx):
        """One provider call: open + consume. Returns the round's accumulation,
        or None if a terminal was already emitted. Retry tiers land later —
        today any provider failure is a clean PROVIDER_ERROR."""
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
                        if delta:
                            text_parts.append(delta)
                            self._rt.emit(ev.TextDelta(delta))
                    case Reasoning(text=delta):
                        delta = await self._hooks.on_reasoning_delta(delta)
                        if delta:
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

    async def _execute_unknown(self, call: ToolCall, tools: dict[str, Tool]) -> ToolResult:
        tool = tools.get(call.name)
        if tool is None:
            return ToolResult(
                call_id=call.id,
                content=f"unknown or unmounted tool: {call.name}",
                is_error=True,
            )
        tctx = ToolContext(working_dir=self._a.working_dir, requester=self._rt.requester())
        try:
            result = await tool.execute(call.arguments, tctx)
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
