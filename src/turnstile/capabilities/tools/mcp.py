"""MCP tool wrapper — surfaces a discovered MCP tool as a kernel Tool.

Thin marshaling over an MCP client session (the `mcp` SDK or anything
speaking its types): `list_tools` discovery mounts one adapter per remote
tool under the `mcp__{server}__{tool}` name; `call_tool` results flatten
into the kernel ToolResult (text blocks joined, image blocks carried as
ImageContent, everything else dropped). Every failure — bad arguments, a
dead server, a server-declared error — maps to ToolResult(is_error=True);
raw MCP exceptions never cross the port boundary.

Trust model: an MCP server is external code, so an unannotated tool is
RISKY (an approval middleware gates it); a server-declared
`readOnlyHint: true` tool has no side effects and is SAFE.
"""

import hashlib
import json
from typing import Protocol

from mcp import types as mcp_types

from turnstile.kernel.dtos import ImageContent, RiskLevel, ToolContext, ToolResult
from turnstile.kernel.ports import Tool

# Conservative OpenAI-compatible upper bound for a function name.
MAX_MCP_TOOL_NAME_LEN = 64


class McpSession(Protocol):
    """The two calls the wrapper needs from `mcp.ClientSession` — a Protocol so
    tests script an in-process double instead of standing up a transport."""

    async def list_tools(
        self, *, params: mcp_types.PaginatedRequestParams | None = None
    ) -> mcp_types.ListToolsResult: ...

    async def call_tool(self, name: str, arguments: dict | None = None) -> mcp_types.Result: ...


def sanitize_name_segment(segment: str) -> str:
    """Replace every character OpenAI/litellm forbids in a function name
    (anything outside [a-zA-Z0-9_-]) with '-'. Real MCP servers routinely
    declare names containing spaces, dots, colons or CJK characters;
    unsanitized they fail the whole request with a 400."""
    return "".join(c if c.isascii() and (c.isalnum() or c in "_-") else "-" for c in segment)


def mcp_tool_full_name(server: str, tool: str) -> str:
    """The name the LLM sees. Keeps the `mcp__{server}__{tool}` shape for
    already-valid names; invalid or overlong names get a stable hash suffix
    derived from the length-delimited original identity, so two names that
    sanitize to the same readable prefix remain distinct."""
    raw = f"mcp__{server}__{tool}"
    if len(raw) <= MAX_MCP_TOOL_NAME_LEN and all(
        c.isascii() and (c.isalnum() or c in "_-") for c in raw
    ):
        return raw

    readable = f"mcp__{sanitize_name_segment(server)}__{sanitize_name_segment(tool)}"
    hasher = hashlib.sha256()
    hasher.update(len(server.encode()).to_bytes(8, "big"))
    hasher.update(server.encode())
    hasher.update(len(tool.encode()).to_bytes(8, "big"))
    hasher.update(tool.encode())
    suffix = "__" + hasher.hexdigest()[:32]
    prefix_len = MAX_MCP_TOOL_NAME_LEN - len(suffix)
    return readable[:prefix_len] + suffix


class McpToolAdapter(Tool):
    """One discovered MCP tool as a kernel Tool; calls route through the shared
    session that owns the server connection."""

    def __init__(self, session: McpSession, server_name: str, info: mcp_types.Tool) -> None:
        self._session = session
        self._server = server_name
        self._tool = info.name
        self._full_name = mcp_tool_full_name(server_name, info.name)
        if info.description:
            self._description = f"[MCP:{server_name}] {info.description}"
        else:
            self._description = (
                f"MCP tool from server '{server_name}'. See input schema for details."
            )
        self._schema = info.input_schema
        annotations = info.annotations
        self._read_only = bool(annotations and annotations.read_only_hint)

    def name(self) -> str:
        return self._full_name

    def description(self) -> str:
        return self._description

    def parameters_schema(self) -> dict:
        return self._schema

    def risk(self, args: str) -> RiskLevel:
        """External code is RISKY by default — the approval middleware gates it
        (the kernel never sandboxes). A server-declared read-only tool has no
        side effects, so it is SAFE (no approval)."""
        return RiskLevel.SAFE if self._read_only else RiskLevel.RISKY

    def read_only_hint(self) -> bool:
        return self._read_only

    def always_grant_scope(self, args: str) -> str:
        """'Always' approves THIS tool regardless of the call's arguments — an
        empty scope keys the grant on the tool name alone. The default per-args
        grant would re-prompt on every differing query."""
        return ""

    async def execute(self, args: str, ctx: ToolContext) -> ToolResult:
        trimmed = args.strip()
        if not trimmed or trimmed == "{}":
            arguments: dict = {}
        else:
            try:
                arguments = json.loads(trimmed)
            except ValueError as e:
                return ToolResult(
                    call_id="", content=f"invalid MCP tool arguments: {e}", is_error=True
                )
            if not isinstance(arguments, dict):
                return ToolResult(
                    call_id="",
                    content="invalid MCP tool arguments: not a JSON object",
                    is_error=True,
                )

        try:
            result = await self._session.call_tool(self._tool, arguments)
        except Exception as e:  # noqa: BLE001 — a dead server is an error RESULT, never a raise
            return ToolResult(call_id="", content=f"MCP call failed: {e}", is_error=True)

        if not isinstance(result, mcp_types.CallToolResult):
            # e.g. InputRequiredResult — an interaction this wrapper does not
            # support; fail closed rather than misreading it as content.
            return ToolResult(
                call_id="",
                content=f"unsupported MCP result: {type(result).__name__}",
                is_error=True,
            )

        texts: list[str] = []
        images: list[ImageContent] = []
        for block in result.content:
            if isinstance(block, mcp_types.TextContent):
                texts.append(block.text)
            elif isinstance(block, mcp_types.ImageContent):
                images.append(ImageContent(media_type=block.mime_type, data=block.data))
            # other block kinds (embedded resources, ...) are dropped
        content = "\n".join(texts)
        if result.is_error:
            return ToolResult(call_id="", content=f"MCP tool error: {content}", is_error=True)
        return ToolResult(call_id="", content=content, images=images)


async def mount_mcp_tools(session: McpSession, server_name: str) -> list[McpToolAdapter]:
    """Discover a connected server's tools (all pages) and wrap each as a kernel
    Tool. Mounted-name collisions fail closed — an external server must not
    replace another tool under an already-mounted name."""
    infos: list[mcp_types.Tool] = []
    cursor: str | None = None
    while True:
        params = mcp_types.PaginatedRequestParams(cursor=cursor) if cursor else None
        page = await session.list_tools(params=params)
        infos.extend(page.tools)
        cursor = page.next_cursor
        if cursor is None:
            break

    adapters: list[McpToolAdapter] = []
    seen: set[str] = set()
    for info in infos:
        adapter = McpToolAdapter(session, server_name, info)
        if adapter.name() in seen:
            raise ValueError(f"MCP tool alias collision: {adapter.name()}")
        seen.add(adapter.name())
        adapters.append(adapter)
    return adapters
