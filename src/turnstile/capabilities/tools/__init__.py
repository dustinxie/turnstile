"""L1 tool capabilities: real-I/O implementations of the kernel Tool port.

`RECIPES` is the tool registry — the one binding of mounted name -> class +
config section. It lives HERE, next to the tools it builds, so adding a tool
is one module plus one line in this package and root never changes. A recipe
receives the deployment cfg DUCK-TYPED (same discipline as AgentSpec: read
attributes, never import the config class — the boundary conversion to plain
scalars happens inside the recipe).
"""

from collections.abc import Callable
from typing import Any

from turnstile.capabilities.tools.kb_search import KbSearchTool
from turnstile.capabilities.tools.web_search import WebSearchTool
from turnstile.kernel.ports import Tool

RECIPES: dict[str, Callable[[Any], Tool]] = {
    "kb_search": lambda cfg: KbSearchTool(**cfg.kb.model_dump()),
    "web_search": lambda cfg: WebSearchTool(api_key=cfg.web_api_key),
}
