"""Reference collector — the numbering authority: turn-global renumbering of
real tool-rendering shapes (kb_search._render, Exa text), per-FILE numbers,
the n -> source map, and a full-engine journey where the model cites what
it was shown."""

import pytest

from turnstile.capabilities.persistence.memory_store import MemorySessionStore
from turnstile.kernel.dtos import (
    AfterOutcome,
    BeforeOutcome,
    Done,
    Role,
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

# real rendering shapes (kb_search._render with region prefix / Exa text)
KB_CONTENT = (
    "[1] hrus::Benefits/FTNT 2026 OE Webinar FAQ Final v2.pdf#L521 (score 0.033)\n"
    "Q: Is there a specific paternity leave policy?\n\n"
    "[2] hrus::Handbook & Labor Postings/2024 Handbook.pdf#L613 (score 0.016)\n"
    "Following is a brief description of employee benefits."
)
# a second call: one NEW document plus another chunk of a doc already seen
KB_CONTENT_2 = (
    "[1] hrus::Benefits/FTNT 2026 OE Webinar FAQ Final v2.pdf#L88 (score 0.041)\n"
    "Q: How is PTO accrued?\n\n"
    "[2] hrus::Holidays/2026 Holiday Calendar.pdf#L3 (score 0.020)\n"
    "Fixed holidays are listed below."
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

FAQ = "Benefits/FTNT 2026 OE Webinar FAQ Final v2.pdf"
HANDBOOK = "Handbook & Labor Postings/2024 Handbook.pdf"
CALENDAR = "Holidays/2026 Holiday Calendar.pdf"


class _NamedTool(EchoTool):
    def __init__(self, name: str) -> None:
        self._name = name

    def name(self) -> str:
        return self._name


async def _seen(collector: ReferenceCollector, tool_name: str, content: str, **kw) -> ToolResult:
    """Run one tool result through the collector; returns the (possibly
    rewritten) result so tests can inspect what the model would see."""
    call = ToolCall(id=f"c-{tool_name}-{len(content)}", name=tool_name, arguments="{}")
    assert await collector.before(call, _NamedTool(tool_name), None) is BeforeOutcome.PROCEED
    result = ToolResult(call_id=call.id, content=content, **kw)
    assert await collector.after(result) is AfterOutcome.PROCEED
    return result


# ── parsing + the map ──────────────────────────────────────────────────


async def test_kb_headers_parse_into_numbered_references():
    collector = ReferenceCollector()
    await _seen(collector, "kb_search", KB_CONTENT)
    assert collector.take() == [
        Reference(n=1, tool="kb_search", ref=FAQ, region="hrus", fragment="L521"),
        Reference(n=2, tool="kb_search", ref=HANDBOOK, region="hrus", fragment="L613"),
    ]


async def test_web_titles_and_urls_parse_and_get_numbers_too():
    collector = ReferenceCollector()
    result = await _seen(collector, "web_search", WEB_CONTENT)
    references = collector.take()
    assert references[0] == Reference(
        n=1,
        tool="web_search",
        ref="Time Off and Leaves",
        url="https://fortinet.sharepoint.com/sites/HRUS/SitePages/Time-Off.aspx",
    )
    assert references[1].n == 2 and references[1].url == "https://www.dol.gov/agencies/whd/fmla"
    # the model sees the numbers stamped onto the title lines
    assert "[1] Title: Time Off and Leaves" in result.content
    assert "[2] Title: FMLA overview" in result.content


# ── the numbering authority ────────────────────────────────────────────


async def test_second_call_continues_numbering_instead_of_colliding():
    collector = ReferenceCollector()
    first = await _seen(collector, "kb_search", KB_CONTENT)
    second = await _seen(collector, "kb_search", KB_CONTENT_2)

    # what the model sees: no two documents share a number across calls
    assert first.content.startswith(f"[1] hrus::{FAQ}#L521 (score 0.033)")
    assert f"\n\n[2] hrus::{HANDBOOK}#L613" in first.content
    assert second.content.startswith(f"[1] hrus::{FAQ}#L88")  # same doc -> same number
    assert f"\n\n[3] hrus::{CALENDAR}#L3" in second.content  # new doc -> next number

    # the map: three documents, per FILE — the FAQ's second chunk added
    # nothing new and keeps the first-cited anchor
    references = collector.take()
    assert [(r.n, r.ref) for r in references] == [(1, FAQ), (2, HANDBOOK), (3, CALENDAR)]
    assert references[0].fragment == "L521"


async def test_numbering_is_per_file_not_per_chunk():
    collector = ReferenceCollector()
    two_chunks = (
        "[1] hrus::pto.md#L12 (score 0.910)\nPTO accrues at 1.5 days/month.\n\n"
        "[2] hrus::pto.md#L40 (score 0.840)\nCarry-over caps at 10 days."
    )
    result = await _seen(collector, "kb_search", two_chunks)
    assert "[1] hrus::pto.md#L12" in result.content
    assert "[1] hrus::pto.md#L40" in result.content  # same document, same number
    assert "[2]" not in result.content
    assert [(r.n, r.ref, r.fragment) for r in collector.take()] == [(1, "pto.md", "L12")]


async def test_kb_and_web_share_one_number_space():
    collector = ReferenceCollector()
    await _seen(collector, "kb_search", KB_CONTENT)
    web = await _seen(collector, "web_search", WEB_CONTENT)
    assert "[3] Title: Time Off and Leaves" in web.content  # continues after the two kb docs
    assert [r.n for r in collector.take()] == [1, 2, 3, 4]


async def test_region_less_refs_still_number():
    # a collection whose doc_id carries no "<region>#" prefix renders bare
    collector = ReferenceCollector()
    await _seen(collector, "kb_search", "[1] docs/guide.md#L5 (score 0.5)\ntext")
    assert collector.take() == [
        Reference(n=1, tool="kb_search", ref="docs/guide.md", region=None, fragment="L5")
    ]


async def test_errors_and_unlisted_tools_are_left_alone():
    collector = ReferenceCollector()
    err = await _seen(collector, "kb_search", "[9] broken.pdf#L1 (score 0.5)\nboom", is_error=True)
    other = await _seen(collector, "calculator", "Title: not a search\nURL: https://x")
    assert err.content.startswith("[9] broken.pdf")  # untouched
    assert other.content.startswith("Title: not a search")  # untouched
    assert collector.take() == []


async def test_take_drains_and_numbering_restarts_next_turn():
    collector = ReferenceCollector()
    await _seen(collector, "kb_search", KB_CONTENT)
    assert len(collector.take()) == 2
    assert collector.take() == []  # drained
    result = await _seen(collector, "kb_search", KB_CONTENT_2)
    assert result.content.startswith("[1] ")  # a new turn starts at 1 again


# ── the turn-end touchpoint: finish() ──────────────────────────────────


def _link(reference: Reference) -> str | None:
    """A driver-style link: web keeps its URL, kb docs get a fake token URL."""
    if reference.url:
        return reference.url
    return f"/v1/files/tok-{reference.n}" + (
        f"#{reference.fragment}" if reference.fragment else ""
    )


async def test_finish_appends_a_references_section_for_cited_numbers_only():
    collector = ReferenceCollector()
    await _seen(collector, "kb_search", KB_CONTENT)
    await _seen(collector, "kb_search", KB_CONTENT_2)  # -> [3] calendar
    answer, structured = collector.finish(
        "Leave accrues per the FAQ [1]; holidays are fixed [3].", _link
    )

    assert answer == (
        "Leave accrues per the FAQ [1]; holidays are fixed [3].\n\n"
        "### References\n"
        "- [1] [FTNT 2026 OE Webinar FAQ Final v2.pdf](/v1/files/tok-1#L521)\n"
        "- [3] [2026 Holiday Calendar.pdf](/v1/files/tok-3#L3)"
    )
    # the structured list carries EVERY numbered doc; the uncited handbook
    # is ground truth of retrieval, flagged rather than dropped
    assert structured == [
        {
            "n": 1,
            "title": "FTNT 2026 OE Webinar FAQ Final v2.pdf",
            "url": "/v1/files/tok-1#L521",
            "cited": True,
        },
        {"n": 2, "title": "2024 Handbook.pdf", "url": "/v1/files/tok-2#L613", "cited": False},
        {"n": 3, "title": "2026 Holiday Calendar.pdf", "url": "/v1/files/tok-3#L3", "cited": True},
    ]
    assert collector.take() == []  # finish drains like take


async def test_finish_resolves_by_number_never_by_name():
    collector = ReferenceCollector()
    await _seen(collector, "kb_search", KB_CONTENT)
    # the model paraphrases the document ("the FAQ") and invents [7]
    answer, structured = collector.finish("Per the FAQ [2] and elsewhere [7].", _link)
    assert "- [2] [2024 Handbook.pdf]" in answer  # [2] IS the handbook, whatever the prose says
    assert "[7]" in answer and "tok-7" not in answer  # hallucinated: plain text, no source line
    assert [e["cited"] for e in structured] == [False, True]


async def test_finish_drops_the_models_own_source_list():
    collector = ReferenceCollector()
    await _seen(collector, "kb_search", KB_CONTENT)
    disobedient = (
        "Leave accrues [1].\n\n### Sources\n[1] some-made-up-name.pdf\n[2] not even retrieved.pdf"
    )
    answer, _ = collector.finish(disobedient, _link)
    assert "made-up" not in answer and "not even retrieved" not in answer
    assert answer.count("### ") == 1  # exactly our section
    assert answer.endswith("- [1] [FTNT 2026 OE Webinar FAQ Final v2.pdf](/v1/files/tok-1#L521)")


async def test_finish_without_citations_or_links_stays_plain():
    collector = ReferenceCollector()
    await _seen(collector, "kb_search", KB_CONTENT)
    answer, structured = collector.finish("I could not find that.", _link)
    assert answer == "I could not find that."  # nothing cited -> no section
    assert all(not e["cited"] for e in structured) and len(structured) == 2

    await _seen(collector, "web_search", WEB_CONTENT)
    answer, _ = collector.finish("See the policy page [1].", lambda r: None)  # driver: no links
    assert answer.endswith("### References\n- [1] Time Off and Leaves")  # plain title, no link


# ── full-engine journey ────────────────────────────────────────────────


class _KbDouble(Tool):
    def __init__(self) -> None:
        self.calls = 0

    def name(self) -> str:
        return "kb_search"

    def description(self) -> str:
        return "kb"

    def parameters_schema(self) -> dict:
        return {"type": "object"}

    def read_only_hint(self) -> bool:
        return True

    async def execute(self, args: str, ctx: ToolContext) -> ToolResult:
        self.calls += 1
        return ToolResult(call_id="", content=KB_CONTENT if self.calls == 1 else KB_CONTENT_2)


async def test_model_sees_renumbered_excerpts_across_a_real_turn():
    provider = ScriptedProvider(
        rounds=[
            [ToolCallEvent(ToolCall("c1", "kb_search", '{"query": "leave"}')), Done()],
            [ToolCallEvent(ToolCall("c2", "kb_search", '{"query": "holidays"}')), Done()],
            [TextDelta("Leave accrues per the FAQ [1]; holidays are fixed [3]."), Done()],
        ]
    )
    collector = ReferenceCollector()
    store = MemorySessionStore()
    agent = Agent(
        provider=provider,
        tools={"kb_search": _KbDouble()},
        middleware=[collector],
        hooks=[store.hook("s")],
        session_id="s",
    )
    outcome = await agent.run_to_completion("leave and holidays?")
    assert "[3]" in outcome.text

    # the tool messages the model read (as persisted) carry turn-global numbers
    snapshot = store.load("s")
    assert snapshot is not None
    tool_texts = [m.text for m in snapshot.messages if m.role is Role.TOOL]
    assert tool_texts[0].startswith(f"[1] hrus::{FAQ}#L521")
    assert f"[3] hrus::{CALENDAR}#L3" in tool_texts[1]

    # and the map resolves what the model cited, by number
    by_n = {r.n: r for r in collector.take()}
    assert by_n[1].ref == FAQ and by_n[3].ref == CALENDAR
