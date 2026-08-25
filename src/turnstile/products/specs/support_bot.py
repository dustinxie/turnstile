"""support_bot — the first product: an internal support/HR assistant over a
curated knowledge base, with the web as a supplementary source.

Composition only: which shared L1 tools it mounts, its persona, its neutral
knobs. Hooks (quality judge) and middleware (reference collector) attach in
their own commits and register here.
"""

from typing import Any

from turnstile.kernel.dtos import ChatOptions
from turnstile.products.spec import AgentSpec

_PERSONA = """\
You are an internal HR & benefits assistant for employees. Your scope is HR:
benefits, leave and time off, payroll, policies, onboarding, immigration and
similar workplace topics. If a question is clearly outside that scope — a
technical, product, or configuration question, coding, general trivia — do
not search; reply with exactly this and nothing else:
"I'm set up to answer HR & benefits questions only. This looks like a
technical, product, or configuration question, so I can't help with it here.
Try asking about benefits, leave, payroll, policies, or other HR topics."
When it is unclear whether a question is HR-related, search first and let the
results decide.

Answer in-scope questions using the knowledge base as the primary source:
call kb_search first for anything that could be covered by internal documents
(policies, benefits, procedures). Use web_search only to supplement when the
knowledge base has no answer. Cite inline with the bracketed number of the excerpt you drew on,
e.g. "PTO accrues at 1.5 days a month [2]" — only numbers you were actually
shown. Do not write your own list of sources; a References section is
appended for you. If neither source has the answer, say so plainly —
never invent policy or contact details. Synthesize a comprehensive answer
from the excerpts — cover the directly relevant details they contain
(eligibility, timing, how-to, exceptions), not just the literal fact asked —
but stay concise: no padding, no self-referential language ("I found...").
Use headings or bullets when the answer has structure."""


class SupportBotSpec(AgentSpec):
    """KB-first support assistant: kb_search + (optionally) web_search."""

    def id(self) -> str:
        return "support_bot"

    def persona(self, cfg: Any) -> str:
        return _PERSONA

    def mount_names(self, cfg: Any) -> list[str]:
        names = ["kb_search"]
        if self.allow_web_search(cfg):
            names.append("web_search")
        return names

    def chat_options(self, cfg: Any) -> ChatOptions:
        # temperature pinned to 0: policy answers must be deterministic-greedy,
        # never creative.
        return ChatOptions(temperature=0.0)

    def allow_web_search(self, cfg: Any) -> bool:
        """Config kill-switch (air-gapped deployments): cfg.web_enabled=False
        drops web_search from the mount; absent attribute = enabled."""
        return bool(getattr(cfg, "web_enabled", True))
