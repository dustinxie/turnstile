"""AgentSpec — the L2 composition seam (design doc §3).

One subclass per product. The application root (the one place allowed to
know all products) constructs the right spec from config and injects it
into the neutral assembly — the loop and the assembly name NO product;
products are pure plug-ins, siblings that never import each other.

`cfg` is deliberately untyped (Any): the concrete config class lives in
root, which products must not import (dependency rule §2) — a spec reads
the attributes it needs, duck-typed.

`mount_names` is PER-PRODUCT STATIC (design doc §3): per-query dynamic
mounting decided by code is the router reborn inside the assembly root.
"""

from abc import ABC, abstractmethod
from typing import Any

from turnstile.kernel.dtos import ChatOptions
from turnstile.kernel.ports import LifecycleHooks, Tool, ToolMiddleware


class AgentSpec(ABC):
    """A product: a persona + a composition of shared L1 capabilities."""

    @abstractmethod
    def id(self) -> str:
        """Stable product identifier (config selects a spec by this)."""

    @abstractmethod
    def persona(self, cfg: Any) -> str:
        """The system prompt."""

    @abstractmethod
    def mount_names(self, cfg: Any) -> list[str]:
        """Which shared L1 tools to expose — names into root's tool catalog."""

    def extra_tools(self, cfg: Any) -> list[Tool]:
        """Product-specific tools mounted alongside the shared ones."""
        return []

    def hooks(self, cfg: Any) -> list[LifecycleHooks]:
        """Product discipline: judges, injectors — registration order matters."""
        return []

    def middleware(self, cfg: Any) -> list[ToolMiddleware]:
        """Tool-call discipline: gates, collectors — registration order matters."""
        return []

    def chat_options(self, cfg: Any) -> ChatOptions:
        """Per-product neutral knobs (None fields = backend defaults rule)."""
        return ChatOptions()

    def allow_web_search(self, cfg: Any) -> bool:
        """Capability toggle: may this product search the web?"""
        return True
