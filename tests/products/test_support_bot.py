"""support_bot spec — composition assertions + a run_to_completion journey on
testkit doubles (the M3 guard from architecture §6)."""

from types import SimpleNamespace

import pytest

from turnstile.kernel.dtos import Done, StopReason, TextDelta, ToolCall, ToolCallEvent
from turnstile.kernel.engine import Agent
from turnstile.kernel.testkit import EchoTool, ScriptedProvider
from turnstile.products.spec import AgentSpec
from turnstile.products.specs.support_bot import SupportBotSpec

pytestmark = pytest.mark.unit

CFG = SimpleNamespace()  # duck-typed cfg: specs read attributes, never a class


def _assemble(spec: AgentSpec, cfg, provider, catalog) -> Agent:
    """Spec -> Agent, the way root will do it in c5: mount by name from a
    shared catalog, spec supplies persona/hooks/middleware/options."""
    tools = {name: catalog[name] for name in spec.mount_names(cfg)}
    for tool in spec.extra_tools(cfg):
        tools[tool.name()] = tool
    return Agent(
        provider=provider,
        tools=tools,
        persona=spec.persona(cfg),
        hooks=spec.hooks(cfg),
        middleware=spec.middleware(cfg),
        chat_options=spec.chat_options(cfg),
    )


class _KbTool(EchoTool):
    def name(self) -> str:
        return "kb_search"


class _WebTool(EchoTool):
    def name(self) -> str:
        return "web_search"


def _catalog() -> dict:
    return {"kb_search": _KbTool(), "web_search": _WebTool()}


# ── composition ────────────────────────────────────────────────────────


def test_spec_identity_and_mounts():
    spec = SupportBotSpec()
    assert spec.id() == "support_bot"
    assert spec.mount_names(CFG) == ["kb_search", "web_search"]
    assert spec.extra_tools(CFG) == []  # nothing product-specific yet


def test_web_kill_switch_drops_web_search():
    spec = SupportBotSpec()
    no_web = SimpleNamespace(web_enabled=False)
    assert not spec.allow_web_search(no_web)
    assert spec.mount_names(no_web) == ["kb_search"]
    assert spec.allow_web_search(CFG)  # absent attribute = enabled


def test_persona_pins_kb_first_and_no_invention():
    persona = SupportBotSpec().persona(CFG)
    assert "kb_search first" in persona
    assert "never invent" in persona
    # scope gate: off-topic questions get the fixed statement, no search
    assert "HR & benefits questions only" in persona and "do\nnot search" in persona
    # citations: echo the server-assigned [n], never a hand-written source list
    assert "bracketed number" in persona and "Do not write your own list of sources" in persona


def test_chat_options_pin_zero_temperature():
    options = SupportBotSpec().chat_options(CFG)
    assert options.temperature == 0.0
    assert options.tool_choice == "auto"  # model decides when to search


# ── run_to_completion journey (scripted provider, real engine) ────────


async def test_kb_first_journey_tool_round_then_answer():
    provider = ScriptedProvider(
        rounds=[
            [ToolCallEvent(ToolCall("c1", "kb_search", '{"query": "leave benefits"}')), Done()],
            [TextDelta("PTO accrues at 1.5 days/month [pto.md]."), Done()],
        ]
    )
    agent = _assemble(SupportBotSpec(), CFG, provider, _catalog())
    outcome = await agent.run_to_completion("what is my leave benefits")

    assert outcome.stop is StopReason.STOPPED and outcome.error is None
    assert outcome.tool_results[0].content == 'echo: {"query": "leave benefits"}'
    assert "PTO accrues" in outcome.text
    # the wire carried the spec's composition: persona + both tool defs + knobs
    call = provider.calls[0]
    assert call.messages[0].text.startswith("You are an internal HR & benefits assistant")
    assert [t.name for t in call.tools] == ["kb_search", "web_search"]
    assert call.options.temperature == 0.0


async def test_no_web_variant_never_advertises_web_search():
    provider = ScriptedProvider(rounds=[[TextDelta("answer"), Done()]])
    agent = _assemble(SupportBotSpec(), SimpleNamespace(web_enabled=False), provider, _catalog())
    await agent.run_to_completion("q")
    assert [t.name for t in provider.calls[0].tools] == ["kb_search"]
