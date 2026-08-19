"""Reference collector — parsing against real tool-rendering shapes (the kb
fixture format, the Exa text format) + a full-engine collection journey."""

import pytest

from turnstile.kernel.dtos import (
    AfterOutcome,
    BeforeOutcome,
    Done,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolContext,
    ToolResult,
)
from turnstile.kernel.engine import Agent
from turnstile.kernel.ports import Tool
from turnstile.kernel.testkit import EchoTool, ScriptedProvider
from turnstile.products.middleware.references import Reference, ReferenceCollector

pytestmark = pytest.mark.unit

# real rendering shapes (kb_search._render / Exa LLM-ready text)
KB_CONTENT = (
    "[1] Benefits/FTNT 2026 OE Webinar FAQ Final v2.pdf#L521 (score 0.033)\n"
    "Q: Is there a specific paternity leave policy?\n\n"
    "[2] Handbook & Labor Postings/2024 Handbook.pdf#L613 (score 0.016)\n"
    "Following is a brief description of employee benefits."
)
WEB_CONTENT = (
    'Search results for "leave policy":\n\n'
    "Title: Time Off and Leaves\n"
    "URL: https://fortinet.sharepoint.com/sites/HRUS/SitePages/Time-Off.aspx\n"
    "Published: N/A\n"
    "Highlights:\nOur leave policies...\n\n---\n\n"
    "Title: FMLA overview\n"
    "URL: https://www.dol.gov/agencies/whd/fmla\n"
    "Highlights:\nFederal leave law..."
)


class _NamedTool(EchoTool):
    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name


async def _seen(collector: ReferenceCollector, tool_name: str, content: str, **kw) -> None:
    call = ToolCall(id=f"c-{tool_name}", name=tool_name, arguments="{}")
    assert await collector.before(call, _NamedTool(tool_name), None) is BeforeOutcome.PROCEED
    result = ToolResult(call_id=call.id, content=content, **kw)
    assert await collector.after(result) is AfterOutcome.PROCEED


# ── parsing ────────────────────────────────────────────────────────────


async def test_kb_refs_parse_from_the_real_rendering():
    collector = ReferenceCollector()
    await _seen(collector, "kb_search", KB_CONTENT)
    assert collector.take() == [
        Reference(tool="kb_search", ref="Benefits/FTNT 2026 OE Webinar FAQ Final v2.pdf#L521"),
        Reference(tool="kb_search", ref="Handbook & Labor Postings/2024 Handbook.pdf#L613"),
    ]


async def test_web_titles_and_urls_parse_from_the_exa_shape():
    collector = ReferenceCollector()
    await _seen(collector, "web_search", WEB_CONTENT)
    references = collector.take()
    assert references[0] == Reference(
        tool="web_search",
        ref="Time Off and Leaves",
        url="https://fortinet.sharepoint.com/sites/HRUS/SitePages/Time-Off.aspx",
    )
    assert references[1].url == "https://www.dol.gov/agencies/whd/fmla"


async def test_duplicates_collapse_errors_and_unlisted_tools_are_skipped():
    collector = ReferenceCollector()
    await _seen(collector, "kb_search", KB_CONTENT)
    await _seen(collector, "kb_search", KB_CONTENT)  # re-hit: same refs
    await _seen(collector, "kb_search", "[9] broken.pdf#L1 (score 0.5)\nboom", is_error=True)
    await _seen(collector, "calculator", "Title: not a search\nURL: https://x")
    references = collector.take()
    assert len(references) == 2  # deduped; error + unlisted tool contributed nothing


async def test_take_drains_for_the_next_turn():
    collector = ReferenceCollector()
    await _seen(collector, "kb_search", KB_CONTENT)
    assert len(collector.take()) == 2
    assert collector.take() == []  # drained


# ── full-engine journey ────────────────────────────────────────────────


class _KbDouble(Tool):
    def name(self) -> str:
        return "kb_search"

    def description(self) -> str:
        return "kb"

    def parameters_schema(self) -> dict:
        return {"type": "object"}

    def read_only_hint(self) -> bool:
        return True

    async def execute(self, args: str, ctx: ToolContext) -> ToolResult:
        return ToolResult(call_id="", content=KB_CONTENT)


async def test_collector_harvests_during_a_real_turn():
    provider = ScriptedProvider(
        rounds=[
            [ToolCallEvent(ToolCall("c1", "kb_search", '{"query": "leave"}')), Done()],
            [TextDelta("Per the FAQ, leave accrues [1]."), Done()],
        ]
    )
    collector = ReferenceCollector()
    agent = Agent(provider=provider, tools={"kb_search": _KbDouble()}, middleware=[collector])
    outcome = await agent.run_to_completion("leave benefits?")
    assert "leave accrues" in outcome.text
    refs = [r.ref for r in collector.take()]
    assert refs == [
        "Benefits/FTNT 2026 OE Webinar FAQ Final v2.pdf#L521",
        "Handbook & Labor Postings/2024 Handbook.pdf#L613",
    ]
