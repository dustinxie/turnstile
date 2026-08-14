"""Engine mock suite — three-phase tool dispatch (commit 6 scope): dedup gates,
middleware gate fold, parallel-safe concurrency, size cap, batch events, images."""

import pytest

from turnstile.kernel import events as ev
from turnstile.kernel.dtos import (
    AfterOutcome,
    BeforeOutcome,
    Done,
    Gate,
    ImageContent,
    Role,
    StopReason,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolContext,
    ToolResult,
)
from turnstile.kernel.engine import Agent, _canonicalize_args, _cap_tool_result
from turnstile.kernel.ports import Tool
from turnstile.kernel.testkit import (
    AsyncFnMiddleware,
    ConcurrencyProbeTool,
    ConcurrencyState,
    CountingTool,
    EchoTool,
    FnMiddleware,
    ScriptedProvider,
)

pytestmark = pytest.mark.unit


def _tool_round(*calls: ToolCall) -> list:
    return [*map(ToolCallEvent, calls), Done()]


async def _collect(agent: Agent, text: str = "q") -> list:
    handle = agent.spawn()
    await handle.commands.put(ev.SendMessage(text=text))
    events = []
    while True:
        event = await handle.events.get()
        events.append(event)
        if isinstance(event, ev.TurnComplete):
            break
    await handle.commands.put(ev.Shutdown())
    await handle.task
    return events


def _results(events: list) -> list[ToolResult]:
    return [e.result for e in events if isinstance(e, ev.ToolResultEvent)]


# ── dedup gates ────────────────────────────────────────────────────────


async def test_mode_a_same_call_id_skipped_entirely():
    counter = CountingTool()
    provider = ScriptedProvider(
        rounds=[
            _tool_round(ToolCall("dup", "count", "{}"), ToolCall("dup", "count", "{}")),
            [TextDelta("done"), Done()],
        ]
    )
    agent = Agent(provider=provider, tools={"count": counter})
    events = await _collect(agent)
    assert counter.count == 1  # executed once
    assert len(_results(events)) == 1  # ONE result row — a second would be illegal
    # round 2's wire has exactly one tool message for the id
    tool_msgs = [r for r, _ in provider.received_texts(1) if r == "tool"]
    assert len(tool_msgs) == 1


async def test_mode_b_same_args_new_id_gets_stub_without_reexecution():
    counter = CountingTool()
    provider = ScriptedProvider(
        rounds=[
            _tool_round(
                ToolCall("c1", "count", '{"a": 1, "b": 2}'),
                ToolCall("c2", "count", '{"b": 2, "a": 1}'),  # same canonical identity
            ),
            [TextDelta("done"), Done()],
        ]
    )
    agent = Agent(provider=provider, tools={"count": counter})
    events = await _collect(agent)
    assert counter.count == 1
    results = _results(events)
    assert len(results) == 2  # BOTH ids paired (API-valid)
    stub = next(r for r in results if r.call_id == "c2")
    assert not stub.is_error and "duplicate call" in stub.content


def test_canonicalize_args_ignores_key_order_not_array_order():
    assert _canonicalize_args('{"a": 1, "b": 2}') == _canonicalize_args('{"b":2,"a":1}')
    assert _canonicalize_args("[1, 2]") != _canonicalize_args("[2, 1]")
    assert _canonicalize_args("not json") == "not json"


# ── middleware gate fold ───────────────────────────────────────────────


async def test_deny_blocks_execution_and_model_sees_reason():
    counter = CountingTool()
    provider = ScriptedProvider(
        rounds=[
            _tool_round(ToolCall("c1", "count", "{}")),
            [TextDelta("understood"), Done()],
        ]
    )
    deny = FnMiddleware(before=lambda call, tool: BeforeOutcome(Gate.DENY, "rbac: no"))
    agent = Agent(provider=provider, tools={"count": counter}, middleware=[deny])
    events = await _collect(agent)
    assert counter.count == 0
    result = _results(events)[0]
    assert result.is_error and result.content == "blocked: rbac: no"
    assert ("tool", "blocked: rbac: no") in provider.received_texts(1)
    assert events[-1].reason is StopReason.STOPPED  # turn continued


async def test_allow_short_circuits_remaining_gates():
    order: list[str] = []

    def allow(call, tool):
        order.append("allow")
        return BeforeOutcome(Gate.ALLOW)

    def never(call, tool):
        order.append("never")
        return BeforeOutcome(Gate.DENY, "should not run")

    provider = ScriptedProvider(
        rounds=[
            _tool_round(ToolCall("c1", "echo", "{}")),
            [TextDelta("done"), Done()],
        ]
    )
    agent = Agent(
        provider=provider,
        tools={"echo": EchoTool()},
        middleware=[FnMiddleware(before=allow), FnMiddleware(before=never)],
    )
    events = await _collect(agent)
    assert order == ["allow"]  # second gate never consulted
    assert not _results(events)[0].is_error


async def test_ask_defers_to_downstream_approval_middleware():
    async def approval(call, tool, rt):
        answer = await rt.request("approval", {"tool": tool.name()})
        if answer and answer.get("decision") == "allow":
            return BeforeOutcome.PROCEED
        return BeforeOutcome(Gate.DENY, "denied")

    ask = FnMiddleware(before=lambda call, tool: BeforeOutcome(Gate.ASK, "force prompt"))
    provider_yes = ScriptedProvider(
        rounds=[
            _tool_round(ToolCall("c1", "echo", "{}")),
            [TextDelta("ok"), Done()],
        ]
    )
    agent = Agent(
        provider=provider_yes,
        tools={"echo": EchoTool()},
        middleware=[ask, AsyncFnMiddleware(before=approval)],
    )
    outcome = await agent.run_to_completion("q", approve=True)
    assert outcome.tool_results and not outcome.tool_results[0].is_error

    provider_no = ScriptedProvider(
        rounds=[
            _tool_round(ToolCall("c1", "echo", "{}")),
            [TextDelta("ok"), Done()],
        ]
    )
    agent = Agent(
        provider=provider_no,
        tools={"echo": EchoTool()},
        middleware=[ask, AsyncFnMiddleware(before=approval)],
    )
    outcome = await agent.run_to_completion("q", approve=False)
    assert outcome.tool_results[0].is_error  # fail-closed denial


async def test_deny_turn_pairs_batch_then_terminates_policy_denied():
    counter = CountingTool()

    def gate(call, tool):
        if call.name == "count" and '"forbidden"' in call.arguments:
            return BeforeOutcome(Gate.DENY_TURN, "hard tenancy boundary")
        return BeforeOutcome.PROCEED

    provider = ScriptedProvider(
        rounds=[
            _tool_round(
                ToolCall("c1", "count", '{"k": "ok"}'),
                ToolCall("c2", "count", '{"k": "forbidden"}'),
                ToolCall("c3", "count", '{"k": "after"}'),
            ),
        ]
    )
    agent = Agent(
        provider=provider, tools={"count": counter}, middleware=[FnMiddleware(before=gate)]
    )
    events = await _collect(agent)
    assert events[-1].reason is StopReason.POLICY_DENIED
    results = {r.call_id: r for r in _results(events)}
    assert set(results) == {"c1", "c2", "c3"}  # every id paired
    assert "terminated the turn by policy" in results["c1"].content  # prior converted
    assert "hard tenancy boundary" in results["c2"].content
    assert "terminated the turn by policy" in results["c3"].content
    assert counter.count == 0  # nothing executed


async def test_before_rewrite_reaches_tool():
    received: list[str] = []

    class Capture(Tool):
        def name(self) -> str:
            return "capture"

        def description(self) -> str:
            return ""

        def parameters_schema(self) -> dict:
            return {}

        async def execute(self, args: str, ctx: ToolContext) -> ToolResult:
            received.append(args)
            return ToolResult(call_id="", content="ok")

    def rewrite(call, tool):
        call.arguments = '{"normalized": true}'
        return BeforeOutcome.PROCEED

    provider = ScriptedProvider(
        rounds=[
            _tool_round(ToolCall("c1", "capture", '{"raw": 1}')),
            [TextDelta("done"), Done()],
        ]
    )
    agent = Agent(
        provider=provider, tools={"capture": Capture()}, middleware=[FnMiddleware(before=rewrite)]
    )
    await _collect(agent)
    assert received == ['{"normalized": true}']


async def test_after_block_feeds_reason_back_as_synthetic_user():
    provider = ScriptedProvider(
        rounds=[
            _tool_round(ToolCall("c1", "echo", "{}")),
            [TextDelta("done"), Done()],
        ]
    )
    blocker = FnMiddleware(after=lambda result: AfterOutcome(block_reason="disregard that"))
    agent = Agent(provider=provider, tools={"echo": EchoTool()}, middleware=[blocker])
    await _collect(agent)
    wire = provider.received_texts(1)
    tool_index = next(i for i, (r, _) in enumerate(wire) if r == "tool")
    assert wire[tool_index + 1] == ("user", "disregard that")


# ── concurrency ────────────────────────────────────────────────────────


async def test_parallel_safe_calls_overlap_side_effecting_serialize():
    read_state = ConcurrencyState()
    write_state = ConcurrencyState()
    tools: dict[str, Tool] = {
        "r1": ConcurrencyProbeTool(read_state, name="r1", read_only=True),
        "r2": ConcurrencyProbeTool(read_state, name="r2", read_only=True),
        "w1": ConcurrencyProbeTool(write_state, name="w1", read_only=False),
        "w2": ConcurrencyProbeTool(write_state, name="w2", read_only=False),
    }
    provider = ScriptedProvider(
        rounds=[
            _tool_round(
                ToolCall("a", "r1", "{}"),
                ToolCall("b", "r2", "{}"),
                ToolCall("c", "w1", "{}"),
                ToolCall("d", "w2", "{}"),
            ),
            [TextDelta("done"), Done()],
        ]
    )
    agent = Agent(provider=provider, tools=tools)
    events = await _collect(agent)
    assert read_state.max_active == 2  # read-only pair overlapped
    assert write_state.max_active == 1  # writers serialized
    # results applied in emission order regardless of concurrency
    assert [r.call_id for r in _results(events)] == ["a", "b", "c", "d"]


# ── batch events / size cap / images ───────────────────────────────────


async def test_batch_events_wrap_multi_call_rounds():
    provider = ScriptedProvider(
        rounds=[
            _tool_round(ToolCall("c1", "echo", '{"a":1}'), ToolCall("c2", "echo", '{"a":2}')),
            [TextDelta("done"), Done()],
        ]
    )
    agent = Agent(provider=provider, tools={"echo": EchoTool()})
    events = await _collect(agent)
    started = [e for e in events if isinstance(e, ev.ToolBatchStarted)]
    completed = [e for e in events if isinstance(e, ev.ToolBatchCompleted)]
    assert len(started) == 1 and len(completed) == 1
    assert [c.parallel_safe for c in started[0].calls] == [True, True]
    assert completed[0].ok == 2 and completed[0].total == 2
    # single-call rounds get no batch wrapper
    provider2 = ScriptedProvider(
        rounds=[
            _tool_round(ToolCall("c1", "echo", "{}")),
            [TextDelta("x"), Done()],
        ]
    )
    events2 = await _collect(Agent(provider=provider2, tools={"echo": EchoTool()}))
    assert not [e for e in events2 if isinstance(e, ev.ToolBatchStarted)]


def test_cap_tool_result_keeps_head_and_tail():
    result = ToolResult(call_id="c", content="A" * 600 + "MIDDLE" + "Z" * 600)
    _cap_tool_result(result, 200)
    assert result.content.startswith("A" * 100)
    assert result.content.endswith("Z" * 100)
    assert "elided by kernel cap" in result.content and "MIDDLE" not in result.content
    unbounded = ToolResult(call_id="c", content="x" * 1000)
    _cap_tool_result(unbounded, 0)
    assert len(unbounded.content) == 1000  # 0 = unbounded


async def test_tool_images_lift_onto_one_synthetic_user_message():
    class VisionTool(Tool):
        def name(self) -> str:
            return "shot"

        def description(self) -> str:
            return ""

        def parameters_schema(self) -> dict:
            return {}

        async def execute(self, args: str, ctx: ToolContext) -> ToolResult:
            return ToolResult(
                call_id="", content="took screenshot", images=[ImageContent("image/png", "QUJD")]
            )

    provider = ScriptedProvider(
        rounds=[
            _tool_round(ToolCall("c1", "shot", "{}")),
            [TextDelta("nice screenshot"), Done()],
        ]
    )
    agent = Agent(provider=provider, tools={"shot": VisionTool()})
    events = await _collect(agent)
    assert _results(events)[0].images == []  # transient: not on the result
    round2 = provider.calls[1].messages
    image_msgs = [m for m in round2 if m.images]
    assert len(image_msgs) == 1
    assert image_msgs[0].role is Role.USER and image_msgs[0].synthetic
    # ordering: tool result first, then the image carrier (calls stay contiguous)
    roles = [m.role.value for m in round2]
    assert roles.index("tool") < roles.index("user", roles.index("tool"))
