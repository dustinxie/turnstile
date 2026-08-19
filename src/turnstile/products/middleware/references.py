"""Reference collector — ground-truth sources for the envelope.

The persona asks the model to cite sources in prose; that is readability,
not evidence — model-written citations can name documents never retrieved.
The envelope's `references` field must be ground truth, and the only ground
truth for "what did this answer draw on" is what the tools actually
returned. `ToolMiddleware.after` is the one seam that sees every raw tool
result (pre-size-cap), so this collector parses refs there — code over real
results, zero model involvement.

Pure observer: both seams always PROCEED, results are never altered.
Collected references land on the collector's OWN state (L2-collector); the
driver drains them with take() after TurnComplete.

Parsers are registered per tool name, so a new retrieval tool joins by
adding one entry — the collector itself stays source-agnostic.
"""

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from turnstile.kernel.dtos import AfterOutcome, BeforeOutcome, ToolCall, ToolResult
from turnstile.kernel.ports import Tool, ToolMiddleware

# kb_search rendering: "[n] <ref> (score s)" header lines (score optional).
_KB_REF_RE = re.compile(r"^\[\d+\] (.+?)(?: \(score [\d.]+\))?$", re.MULTILINE)
# web_search rendering: "Title: ..." / "URL: ..." blocks (Exa LLM-ready text).
_WEB_RE = re.compile(r"^Title: (?P<title>.+)\nURL: (?P<url>\S+)$", re.MULTILINE)


@dataclass(frozen=True)
class Reference:
    """L2-local (design doc §4): one source the answer actually drew on."""

    tool: str  # which tool surfaced it ("kb_search", "web_search", ...)
    ref: str  # document path / page title
    url: str | None = None  # openable link when the source has one


def _kb_references(tool: str, content: str) -> Iterator[Reference]:
    for ref in _KB_REF_RE.findall(content):
        yield Reference(tool=tool, ref=ref)


def _web_references(tool: str, content: str) -> Iterator[Reference]:
    for match in _WEB_RE.finditer(content):
        yield Reference(tool=tool, ref=match["title"], url=match["url"])


# tool name -> parser over its rendered result. Unlisted tools contribute
# nothing (a calculator has no sources to cite).
_PARSERS: dict[str, Callable[[str, str], Iterator[Reference]]] = {
    "kb_search": _kb_references,
    "web_search": _web_references,
}


class ReferenceCollector(ToolMiddleware):
    """Harvests sources from retrieval-tool results as they happen."""

    def __init__(self) -> None:
        self._tool_by_call: dict[str, str] = {}
        self._references: list[Reference] = []
        self._seen: set[tuple[str, str]] = set()

    async def before(self, call: ToolCall, tool: Tool, rt: object) -> BeforeOutcome:
        # after() receives only the ToolResult (call_id, content) — remember
        # which tool owns each call id so the parser is picked by identity,
        # not by content sniffing.
        self._tool_by_call[call.id] = tool.name()
        return BeforeOutcome.PROCEED

    async def after(self, result: ToolResult) -> AfterOutcome:
        tool = self._tool_by_call.get(result.call_id, "")
        parse = _PARSERS.get(tool)
        if parse is not None and not result.is_error:
            for reference in parse(tool, result.content):
                self._add(reference)
        return AfterOutcome.PROCEED

    def _add(self, reference: Reference) -> None:
        key = (reference.tool, reference.url or reference.ref)
        if key not in self._seen:  # first occurrence wins; re-hits add nothing
            self._seen.add(key)
            self._references.append(reference)

    def take(self) -> list[Reference]:
        """Drain: the collected references, clearing state for the next turn.
        The driver calls this once after TurnComplete (the envelope build)."""
        references = self._references
        self._references = []
        self._seen.clear()
        self._tool_by_call.clear()
        return references
