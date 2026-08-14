"""Kernel test doubles — shipped so L1/L2 tests reuse the same fakes.

Python implementation fo Mock Provider/Tool/Hook/Middleware/Checkpoint.
Everything here talks ONLY kernel DTOs and ports — no I/O, no time, no network.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from turnstile.kernel.dtos import (
    AfterOutcome,
    BeforeOutcome,
    ChatOptions,
    Conversation,
    Done,
    Message,
    ProviderError,
    RateLimitDecision,
    RateLimitHint,
    SessionSnapshot,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolContext,
    ToolDef,
    ToolResult,
    TurnCtx,
)
from turnstile.kernel.ports import (
    Clock,
    CompactionCheckpoint,
    CompactionCheckpointError,
    LifecycleHooks,
    LlmProvider,
    Tool,
    ToolMiddleware,
)

# ── providers ──────────────────────────────────────────────────────────

type ScriptedRound = list[StreamEvent] | ProviderError
"""One chat_stream call's script: events to yield, or a ProviderError to raise
before any event (the failed-open shape)."""


@dataclass
class RecordedCall:
    """What the provider saw on one chat_stream call."""

    messages: list[Message]
    tools: list[ToolDef]
    options: ChatOptions


class ScriptedProvider(LlmProvider):
    """Scripted, recording provider: each chat_stream call pops the next round's
    script and records the FULL (messages, tools, options) it received — so a
    test can byte-compare the exact wire prefix across rounds (prefix-cache RCA)
    and assert which options reached the provider. An exhausted script yields a
    bare Done — the loop treats a content-free Done as an empty-200."""

    def __init__(self, rounds: list[ScriptedRound], ctx_window: int = 0) -> None:
        self._rounds = list(rounds)
        self._ctx_window = ctx_window
        self.calls: list[RecordedCall] = []

    def model_name(self) -> str:
        return "scripted"

    def context_window(self) -> int:
        return self._ctx_window

    async def chat_stream(
        self, messages: list[Message], tools: list[ToolDef], options: ChatOptions
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append(
            RecordedCall(messages=list(messages), tools=list(tools), options=options)
        )
        script: ScriptedRound = self._rounds.pop(0) if self._rounds else [Done()]
        if isinstance(script, ProviderError):
            raise script
        for event in script:
            yield event

    # -- assertion conveniences ------------------------------------------

    def received_texts(self, call: int) -> list[tuple[str, str]]:
        """(role.value, text) pairs the model saw on the given call."""
        return [(m.role.value, m.text) for m in self.calls[call].messages]


class AlwaysStopProvider(LlmProvider):
    """The same content-bearing stop on EVERY call, forever — a legitimate
    completion each round (never mistaken for an empty-200), for exercising the
    offer_continuation / continuation-fuse paths over many rounds."""

    def __init__(self, text: str = "done") -> None:
        self._text = text

    def model_name(self) -> str:
        return "always-stop"

    async def chat_stream(
        self, messages: list[Message], tools: list[ToolDef], options: ChatOptions
    ) -> AsyncIterator[StreamEvent]:
        yield TextDelta(self._text)
        yield Done()


class SilentProvider(LlmProvider):
    """Opens OK, yields an optional prefix, then PENDS FOREVER — a TCP half-open /
    model stall. Empty prefix bounds first-token latency; a non-empty one bounds
    inter-token latency. Drives the stream idle-timeout path."""

    def __init__(self, prefix: list[StreamEvent] | None = None) -> None:
        self._prefix = list(prefix or [])

    def model_name(self) -> str:
        return "silent"

    async def chat_stream(
        self, messages: list[Message], tools: list[ToolDef], options: ChatOptions
    ) -> AsyncIterator[StreamEvent]:
        for event in self._prefix:
            yield event
        await asyncio.Event().wait()  # never set — pends forever


class StallThenProvider(LlmProvider):
    """Pends forever on the first `stall_calls` calls (driving idle-timeout
    reconnects), then yields `events`. stall_calls=1 → first attempt times out,
    the reconnect succeeds; a large value exhausts the reconnect budget."""

    def __init__(self, stall_calls: int, events: list[StreamEvent]) -> None:
        self._stall_calls = stall_calls
        self._events = list(events)
        self.call_count = 0

    def model_name(self) -> str:
        return "stall-then"

    async def chat_stream(
        self, messages: list[Message], tools: list[ToolDef], options: ChatOptions
    ) -> AsyncIterator[StreamEvent]:
        self.call_count += 1
        if self.call_count <= self._stall_calls:
            await asyncio.Event().wait()  # pend forever; nothing yielded
        for event in self._events:
            yield event


# ── clock ──────────────────────────────────────────────────────────────


class StepClock(Clock):
    """Advances a fixed step per read — deterministic non-zero elapsed_ms."""

    def __init__(self, step_ms: int = 10) -> None:
        self._step = step_ms
        self._now = 0

    def now_millis(self) -> int:
        self._now += self._step
        return self._now


# ── tools ──────────────────────────────────────────────────────────────


class EchoTool(Tool):
    """Safe tool: echoes its arguments back."""

    def name(self) -> str:
        return "echo"

    def description(self) -> str:
        return "Echo the arguments back"

    def parameters_schema(self) -> dict:
        return {"type": "object", "properties": {"text": {"type": "string"}}}

    def read_only_hint(self) -> bool:
        return True

    async def execute(self, args: str, ctx: ToolContext) -> ToolResult:
        return ToolResult(call_id="", content=f"echo: {args}")


class CountingTool(Tool):
    """Counts actual executions — proves a dedup-suppressed duplicate call never
    ran (counter = calls let through, not calls the model emitted)."""

    def __init__(self, name: str = "count", read_only: bool = False) -> None:
        self._name = name
        self._read_only = read_only
        self.count = 0

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "Increments a counter each time it executes"

    def parameters_schema(self) -> dict:
        return {"type": "object", "properties": {"k": {"type": "string"}}}

    def read_only_hint(self) -> bool:
        return self._read_only

    async def execute(self, args: str, ctx: ToolContext) -> ToolResult:
        self.count += 1
        return ToolResult(call_id="", content=f"count#{self.count} args={args}")


@dataclass
class ConcurrencyState:
    """Shared across ConcurrencyProbeTool instances of one test."""

    active: int = 0
    max_active: int = 0
    order: list[str] = field(default_factory=list)


class ConcurrencyProbeTool(Tool):
    """Records how many executions overlap — proves parallel-safe calls run
    concurrently (max_active > 1) and side-effecting calls serialize (== 1)."""

    def __init__(
        self, state: ConcurrencyState, name: str = "probe", read_only: bool = True
    ) -> None:
        self._state = state
        self._name = name
        self._read_only = read_only

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "Records execution overlap"

    def parameters_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    def read_only_hint(self) -> bool:
        return self._read_only

    async def execute(self, args: str, ctx: ToolContext) -> ToolResult:
        s = self._state
        s.order.append(f"start:{self._name}")
        s.active += 1
        s.max_active = max(s.max_active, s.active)
        for _ in range(4):  # yield so overlapping executions interleave
            await asyncio.sleep(0)
        s.active -= 1
        s.order.append(f"end:{self._name}")
        return ToolResult(call_id="", content=f"probe:{self._name}")


class BlockUntilCancelTool(Tool):
    """Parks until the turn's cancel token fires, then reports cooperatively —
    drives the cancel-mid-execute paths."""

    def name(self) -> str:
        return "block_until_cancel"

    def description(self) -> str:
        return "Waits for cooperative cancellation"

    def parameters_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, args: str, ctx: ToolContext) -> ToolResult:
        await ctx.cancel.cancelled()
        return ToolResult(call_id="", content="(cancelled cooperatively)", is_error=True)


class CommandInjectorTool(Tool):
    """Injects a pre-configured AgentCommand back into the session MID-TURN, the
    first time it executes — a deterministic mid-turn injection point (the
    kernel runs execute between the assistant's tool_call and the round ending).
    `commands_slot` is a one-element list filled with handle.commands AFTER
    spawn (the queue only exists then). Yields to the event loop after sending
    so the session's mid-turn select drains the command while the turn is still
    in flight."""

    def __init__(self, commands_slot: list, command: object, yields: int = 16) -> None:
        self._slot = commands_slot
        self._command = command
        self._yields = yields
        self._fired = False

    def name(self) -> str:
        return "inject"

    def description(self) -> str:
        return "Injects an AgentCommand mid-turn (test only)"

    def parameters_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, args: str, ctx: ToolContext) -> ToolResult:
        if not self._fired and self._slot:
            self._fired = True
            self._slot[0].put_nowait(self._command)
            for _ in range(self._yields):
                await asyncio.sleep(0)
        return ToolResult(call_id="", content="injected")


class FailingTool(Tool):
    """Raises from execute — proves the loop converts exceptions into error
    ToolResults instead of unwinding the turn (the must-not-raise adaptation)."""

    def name(self) -> str:
        return "failing"

    def description(self) -> str:
        return "Always raises"

    def parameters_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, args: str, ctx: ToolContext) -> ToolResult:
        raise RuntimeError("tool exploded")


# ── hooks ──────────────────────────────────────────────────────────────


class RecorderHook(LifecycleHooks):
    """Records every lifecycle callback — proves the loop wires the FULL hook
    surface (no dead seams)."""

    def __init__(self) -> None:
        self.log: list[str] = []

    async def session_start(self, convo: Conversation, resumed: bool) -> None:
        self.log.append(f"session_start:resumed={resumed}")

    async def user_prompt_submit(self, text: str) -> str:
        self.log.append("user_prompt_submit")
        return text

    async def turn_start(self, convo: Conversation) -> None:
        self.log.append("turn_start")

    async def pre_request(self, messages: list[Message], ctx: TurnCtx) -> None:
        self.log.append(f"pre_request:round={ctx.round}")

    async def pre_request_options(
        self, messages: list[Message], options: ChatOptions, ctx: TurnCtx
    ) -> None:
        self.log.append("pre_request_options")

    async def on_request(self, messages, tools, options, ctx: TurnCtx) -> None:
        self.log.append(f"on_request:request_id={ctx.request_id}")

    async def on_text_delta(self, delta: str) -> str:
        self.log.append("on_text_delta")
        return delta

    async def on_reasoning_delta(self, delta: str) -> str:
        self.log.append("on_reasoning_delta")
        return delta

    async def on_model_response(self, response: Message) -> None:
        self.log.append("on_model_response")

    async def offer_continuation(self, convo: Conversation) -> str | None:
        self.log.append("offer_continuation")
        return None

    async def turn_complete(self, convo, reason, ctx: TurnCtx) -> None:
        self.log.append(f"turn_complete:{reason.value}")

    async def on_error(self, error: str) -> None:
        self.log.append(f"on_error:{error}")

    async def on_rate_limit(self, hint: RateLimitHint) -> RateLimitDecision | None:
        self.log.append("on_rate_limit")
        return None

    async def session_end(self, convo: Conversation) -> None:
        self.log.append("session_end")


class ContinueOnceHook(LifecycleHooks):
    """Injects one 'keep going' the first time the model tries to stop — proves
    turn-level injection changes the loop."""

    def __init__(self, text: str = "keep going") -> None:
        self._text = text
        self._used = False

    async def offer_continuation(self, convo: Conversation) -> str | None:
        if self._used:
            return None
        self._used = True
        return self._text


class AlwaysContinueHook(LifecycleHooks):
    """Never accepts the stop — drives the MAX_CONTINUATIONS fuse."""

    async def offer_continuation(self, convo: Conversation) -> str | None:
        return "again"


class FnHook(LifecycleHooks):
    """Generic seam override: pass any subset of hook behaviors as callables.
    Subsumes the reference's one-off Redact/DropTools/RejectPrompt/TailReminder
    hooks — a test states its behavior inline."""

    def __init__(
        self,
        on_text_delta: Callable[[str], str] | None = None,
        on_model_response: Callable[[Message], None] | None = None,
        user_prompt_submit: Callable[[str], str] | None = None,
        pre_request: Callable[[list[Message], TurnCtx], None] | None = None,
        on_rate_limit: Callable[[RateLimitHint], RateLimitDecision | None] | None = None,
    ) -> None:
        self._on_text_delta = on_text_delta
        self._on_model_response = on_model_response
        self._user_prompt_submit = user_prompt_submit
        self._pre_request = pre_request
        self._on_rate_limit = on_rate_limit

    async def on_text_delta(self, delta: str) -> str:
        return self._on_text_delta(delta) if self._on_text_delta else delta

    async def on_model_response(self, response: Message) -> None:
        if self._on_model_response:
            self._on_model_response(response)

    async def user_prompt_submit(self, text: str) -> str:
        return self._user_prompt_submit(text) if self._user_prompt_submit else text

    async def pre_request(self, messages: list[Message], ctx: TurnCtx) -> None:
        if self._pre_request:
            self._pre_request(messages, ctx)

    async def on_rate_limit(self, hint: RateLimitHint) -> RateLimitDecision | None:
        return self._on_rate_limit(hint) if self._on_rate_limit else None


# ── middleware ─────────────────────────────────────────────────────────


class FnMiddleware(ToolMiddleware):
    """Generic tool middleware from callables (sync or returning awaitables not
    needed — the loop awaits us, we call the fn synchronously). Subsumes the
    reference's ArgRewrite/BlockTool/Truncate one-offs."""

    def __init__(
        self,
        before: Callable[[ToolCall, Tool], BeforeOutcome] | None = None,
        after: Callable[[ToolResult], AfterOutcome] | None = None,
    ) -> None:
        self._before = before
        self._after = after

    async def before(self, call: ToolCall, tool: Tool, rt) -> BeforeOutcome:
        return self._before(call, tool) if self._before else BeforeOutcome.PROCEED

    async def after(self, result: ToolResult) -> AfterOutcome:
        return self._after(result) if self._after else AfterOutcome.PROCEED


class AsyncFnMiddleware(ToolMiddleware):
    """Like FnMiddleware but the before fn is async and receives rt — for
    approval-style middlewares that round-trip the driver."""

    def __init__(
        self, before: Callable[[ToolCall, Tool, object], Awaitable[BeforeOutcome]]
    ) -> None:
        self._before_fn = before

    async def before(self, call: ToolCall, tool: Tool, rt) -> BeforeOutcome:
        return await self._before_fn(call, tool, rt)


# ── checkpoint ─────────────────────────────────────────────────────────


class MemoryCheckpoint(CompactionCheckpoint):
    """In-memory checkpoint: records every saved snapshot; optionally fails —
    proves a failed save refuses the compaction commit."""

    def __init__(self, fail_with: str | None = None) -> None:
        self.saved: list[SessionSnapshot] = []
        self._fail_with = fail_with

    def save(self, snapshot: SessionSnapshot) -> None:
        if self._fail_with is not None:
            raise CompactionCheckpointError(self._fail_with)
        self.saved.append(snapshot)
