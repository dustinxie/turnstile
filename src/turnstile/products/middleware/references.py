"""Reference collector — the citation NUMBERING AUTHORITY and ground truth.

Citations work like a bibliography: someone numbers the list, the writer
cites the numbers. Here the server numbers and the model cites. Retrieval
tools render their hits with per-call numbers ("[1] ..." restarting every
call), so two kb_search calls in one turn would both offer a "[1]" — the
model could not cite unambiguously. This middleware sits on the one seam
that sees every raw tool result (`ToolMiddleware.after`, which MAY rewrite
the result in place) and:

  - REWRITES the excerpt numbers to turn-global ones before the model sees
    them: call one -> [1..k], call two -> [k+1..]; the same document is
    always the same number (numbering is PER FILE, not per chunk — a
    Reference section lists documents, and the model must never be able to
    cite one document under two numbers);
  - RECORDS the authoritative map n -> Reference (tool, region, path,
    fragment). Whatever "[n]" the model later writes resolves by NUMBER
    lookup against this map — never by matching document names, which the
    model paraphrases or omits. An "[n]" it was never shown has no entry
    and stays plain text.

The model's own citations are readability, not evidence: the only ground
truth for "what did this answer draw on" is what the tools returned, and
that is what lands here — code over real results, zero model involvement.

Collected references land on the collector's OWN state (L2-collector); the
driver drains them with take() after TurnComplete. Parsers are registered
per tool name, so a new retrieval tool joins by adding one entry.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from turnstile.kernel.dtos import AfterOutcome, BeforeOutcome, ToolCall, ToolResult
from turnstile.kernel.ports import Tool, ToolMiddleware

# kb_search rendering: "[n] [<region>::]<path>[#L<line>] [(score s)]" header
# lines. The "::" region separator cannot collide with path slashes.
_KB_HEADER_RE = re.compile(
    r"^\[(?P<n>\d+)\] (?:(?P<region>[^\s:/]+)::)?(?P<path>.+?)(?:#(?P<fragment>L\d+))?"
    r"(?P<score> \(score [\d.]+\))?$",
    re.MULTILINE,
)
# web_search rendering: "Title: ..." / "URL: ..." blocks (Exa LLM-ready text).
_WEB_RE = re.compile(r"^(?P<title>Title: .+)\nURL: (?P<url>\S+)$", re.MULTILINE)


@dataclass(frozen=True)
class Reference:
    """L2-local (design doc §4): one document the answer could draw on, under
    its turn-global citation number."""

    n: int
    tool: str  # which tool surfaced it ("kb_search", "web_search", ...)
    ref: str  # kb: path within the region's store; web: page title
    url: str | None = None  # openable link when the source already has one
    region: str | None = None  # kb: the store's region tier ("hrus")
    fragment: str | None = None  # kb: first cited chunk anchor ("L521")


class ReferenceCollector(ToolMiddleware):
    """Renumbers retrieval results turn-globally and keeps the n -> source map."""

    def __init__(self) -> None:
        self._tool_by_call: dict[str, str] = {}
        self._by_key: dict[tuple, Reference] = {}  # (tool, identity) -> its Reference
        self._rewriters: dict[str, Callable[[str, str], str]] = {
            "kb_search": self._rewrite_kb,
            "web_search": self._rewrite_web,
        }

    async def before(self, call: ToolCall, tool: Tool, rt: object) -> BeforeOutcome:
        # after() receives only the ToolResult (call_id, content) — remember
        # which tool owns each call id so the parser is picked by identity,
        # not by content sniffing.
        self._tool_by_call[call.id] = tool.name()
        return BeforeOutcome.PROCEED

    async def after(self, result: ToolResult) -> AfterOutcome:
        tool = self._tool_by_call.get(result.call_id, "")
        rewrite = self._rewriters.get(tool)
        if rewrite is not None and not result.is_error:
            result.content = rewrite(tool, result.content)  # in place: the seam's contract
        return AfterOutcome.PROCEED

    def _number(self, key: tuple, make: Callable[[int], Reference]) -> int:
        """Get-or-assign the turn-global number for one document."""
        found = self._by_key.get(key)
        if found is None:
            found = make(len(self._by_key) + 1)
            self._by_key[key] = found
        return found.n

    def _rewrite_kb(self, tool: str, content: str) -> str:
        def renumber(match: re.Match) -> str:
            region, path = match["region"], match["path"]
            n = self._number(
                (tool, region, path),
                lambda n: Reference(
                    n=n, tool=tool, ref=path, region=region, fragment=match["fragment"]
                ),
            )
            header = f"[{n}] " + (f"{region}::" if region else "") + path
            if match["fragment"]:
                header += f"#{match['fragment']}"
            return header + (match["score"] or "")

        return _KB_HEADER_RE.sub(renumber, content)

    def _rewrite_web(self, tool: str, content: str) -> str:
        def renumber(match: re.Match) -> str:
            url = match["url"]
            n = self._number(
                (tool, url),
                lambda n: Reference(n=n, tool=tool, ref=match["title"][len("Title: ") :], url=url),
            )
            return f"[{n}] {match.group(0)}"

        return _WEB_RE.sub(renumber, content)

    def take(self) -> list[Reference]:
        """Drain: every document numbered this turn, in number order, clearing
        state for the next turn. The driver calls this once after TurnComplete
        (the envelope build), or via finish()."""
        references = sorted(self._by_key.values(), key=lambda r: r.n)
        self._by_key.clear()
        self._tool_by_call.clear()
        return references

    def finish(
        self, answer: str, link: Callable[[Reference], str | None]
    ) -> tuple[str, list[dict]]:
        """The turn-end touchpoint: the ACCEPTED answer in, the answer with its
        References section out, plus the structured list for the envelope.

        Inline "[n]" markers resolve by NUMBER against this turn's map — the
        model only ever echoes numbers it was shown; one it invents has no
        entry and stays plain text. The section lists only documents the
        body actually cites (the bibliography rule), while the returned list
        carries EVERY numbered document with a `cited` flag — what the
        retrieval surfaced is ground truth whether or not the model used it.
        A source list the model wrote anyway is dropped: ours is
        authoritative. `link` is the driver's: it turns a Reference into an
        openable URL (file tokens need the requesting principal, which only
        the driver knows) — this class never touches secrets or config.
        Drains the collector like take()."""
        references = self.take()
        body = _OWN_SOURCES_RE.sub("", answer).rstrip()
        cited = {int(n) for n in _INLINE_CITE_RE.findall(body)}

        structured = [
            {
                "n": r.n,
                "title": _title(r),
                "url": link(r),
                "cited": r.n in cited,
                # the raw source, so a later reader (history) can re-mint a link
                # for its own principal instead of trusting a stored, expiring one
                "tool": r.tool,
                "region": r.region,
                "path": r.ref,
                "fragment": r.fragment,
            }
            for r in references
        ]
        # One list item per document: consecutive bare lines would be ONE
        # markdown paragraph (soft breaks) and render run together.
        lines = [
            f"- [{e['n']}] [{e['title']}]({e['url']})"
            if e["url"]
            else f"- [{e['n']}] {e['title']}"
            for e in structured
            if e["cited"]
        ]
        if lines:
            body = f"{body}\n\n### References\n" + "\n".join(lines)
        return body, structured


# Any inline "[n]" citation marker in the answer body.
_INLINE_CITE_RE = re.compile(r"\[(\d+)\]")
# A source list the model wrote despite instructions: a trailing "### Sources"
# / "### References" heading (any level) through end of text.
_OWN_SOURCES_RE = re.compile(
    r"\n#{1,6}\s+(?:Sources?|References?)\s*\n.*\Z", re.DOTALL | re.IGNORECASE
)


def _title(reference: Reference) -> str:
    """kb: the document's file name (the path is a store detail); web: the
    page title."""
    if reference.tool == "kb_search":
        return reference.ref.rsplit("/", 1)[-1]
    return reference.ref
