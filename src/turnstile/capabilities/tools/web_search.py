"""web_search tool — keyword web search via the Exa MCP search API.

Exa backend: `https://mcp.exa.ai/mcp` is reachable keyless (an optional
API key raises rate limits) and returns clean LLM-ready result text, so
o HTML scraping and no key management. The call is a stateless one-shot
JSON-RPC `tools/call` — deliberately NOT mounted through the MCP session
wrapper (no handshake, no session lifecycle for a single POST).

Empirical backend choice (2026-08-18, 4-query side-by-side vs Serper/
Google): parity or better — Exa's page-content highlights repeatedly
carried the actual answer where Google's 160-char snippets truncated it.
A Serper backend can be added later behind the same tool name.

Read-only => SAFE. Every failure is ToolResult(is_error=True), never a
raise.
"""

import json

import httpx

from turnstile.kernel.dtos import ToolContext, ToolResult
from turnstile.kernel.ports import Tool

EXA_MCP_URL = "https://mcp.exa.ai/mcp"

_MAX_RESULTS_DEFAULT = 8
_MAX_RESULTS_CAP = 20


class WebSearchTool(Tool):
    """Web search over Exa's MCP endpoint, one JSON-RPC call per search."""

    def __init__(
        self,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """`api_key=None` uses Exa's keyless tier. Injectable client for
        tests (MockTransport) and pooling."""
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(timeout=25.0)

    def name(self) -> str:
        return "web_search"

    def description(self) -> str:
        return (
            "Search the web for information — returns titles, URLs, and "
            "page-content highlights. Use for information not available in "
            "the knowledge base. max_results caps the list (default 8)."
        )

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {
                    "type": "integer",
                    "description": "Max results (default 8)",
                },
            },
            "required": ["query"],
        }

    def read_only_hint(self) -> bool:
        return True

    async def execute(self, args: str, ctx: ToolContext) -> ToolResult:
        try:
            parsed = json.loads(args or "{}")
        except ValueError as e:
            return ToolResult(
                call_id="", content=f"invalid web_search arguments: {e}", is_error=True
            )
        query = parsed.get("query") if isinstance(parsed, dict) else None
        if not query or not isinstance(query, str):
            return ToolResult(
                call_id="", content="web_search requires a 'query' string argument", is_error=True
            )
        raw_max = parsed.get("max_results", _MAX_RESULTS_DEFAULT)
        max_results = raw_max if isinstance(raw_max, int) else _MAX_RESULTS_DEFAULT
        max_results = max(1, min(max_results, _MAX_RESULTS_CAP))

        url = EXA_MCP_URL if self._api_key is None else f"{EXA_MCP_URL}?exaApiKey={self._api_key}"
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "web_search_exa",
                "arguments": {"query": query, "numResults": max_results},
            },
        }
        try:
            response = await self._client.post(
                url,
                json=body,
                headers={"Accept": "application/json, text/event-stream"},
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            return ToolResult(
                call_id="", content=f"web_search failed for '{query}': {e}", is_error=True
            )

        text = _extract_result_text(response.text)
        if not text:
            return ToolResult(
                call_id="",
                content=(
                    f"web_search returned no usable results for '{query}' "
                    f"({len(response.content)} bytes received)"
                ),
                is_error=True,
            )
        return ToolResult(call_id="", content=f'Search results for "{query}":\n\n{text}')


def _extract_result_text(body: str) -> str:
    """The LLM-ready text out of an Exa MCP response — an SSE stream whose
    `data:` line carries one JSON-RPC message with `result.content[0].text`
    (a plain-JSON body is accepted too). "" when nothing usable."""
    stripped = body.strip()
    payloads: list[dict] = []
    if stripped.startswith("{"):
        try:
            payloads.append(json.loads(stripped))
        except ValueError:
            return ""
    else:
        for line in stripped.splitlines():
            if not line.startswith("data:"):
                continue
            try:
                message = json.loads(line[len("data:") :].strip())
            except ValueError:
                continue
            if isinstance(message, dict):
                payloads.append(message)
    for message in payloads:
        if message.get("error"):
            return ""
        content = (message.get("result") or {}).get("content") or []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                return str(block["text"]).strip()
    return ""
