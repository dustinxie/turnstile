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
You are an internal support assistant. Answer employees' questions using the
knowledge base as the primary source: call kb_search first for anything that
could be covered by internal documents (policies, benefits, procedures,
product docs). Use web_search only to supplement when the knowledge base has
no answer. Cite your sources: reference the document names/URLs your answer
came from. If neither source has the answer, say so plainly — never invent
policy or contact details. Keep answers concise and complete."""


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
