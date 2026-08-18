"""MCP tool wrapper — a scripted in-process session double returning real
`mcp.types` objects: no transport, no server process."""

import json

import pytest
from mcp import types as mcp_types

from turnstile.capabilities.tools.mcp import (
    MAX_MCP_TOOL_NAME_LEN,
    McpToolAdapter,
    mcp_tool_full_name,
    mount_mcp_tools,
    sanitize_name_segment,
)
from turnstile.kernel.dtos import ImageContent, RiskLevel, ToolContext

pytestmark = pytest.mark.unit


class ScriptedSession:
    """McpSession double: canned list_tools pages + a call_tool script (a
    result to return or an exception to raise); records every call."""

    def __init__(
        self,
        pages: list[mcp_types.ListToolsResult] | None = None,
        result: mcp_types.Result | None = None,
        error: Exception | None = None,
    ):
        self.pages = pages or []
        self.result = result
        self.error = error
        self.calls: list[tuple[str, dict | None]] = []

    async def list_tools(
        self, *, params: mcp_types.PaginatedRequestParams | None = None
    ) -> mcp_types.ListToolsResult:
        index = 0 if params is None or params.cursor is None else int(params.cursor)
        return self.pages[index]

    async def call_tool(self, name: str, arguments: dict | None = None) -> mcp_types.Result:
        self.calls.append((name, arguments))
        if self.error is not None:
            raise self.error
        assert self.result is not None, "call_tool invoked without a scripted result"
        return self.result


def _info(name="query", description="Search the docs", read_only=False):
    return mcp_types.Tool(
        name=name,
        description=description,
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        annotations=mcp_types.ToolAnnotations(read_only_hint=read_only) if read_only else None,
    )


def _adapter(info=None, session=None) -> McpToolAdapter:
    return McpToolAdapter(session or ScriptedSession(), "docs", info or _info())


def _text_result(*texts, is_error=False):
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=t) for t in texts],
        is_error=is_error,
    )


# ── discovery / mounting ───────────────────────────────────────────────


async def test_mount_wraps_each_discovered_tool():
    session = ScriptedSession(
        pages=[mcp_types.ListToolsResult(tools=[_info("query"), _info("read_file")])]
    )
    adapters = await mount_mcp_tools(session, "docs")
    assert [a.name() for a in adapters] == ["mcp__docs__query", "mcp__docs__read_file"]
    assert adapters[0].description() == "[MCP:docs] Search the docs"
    assert adapters[0].parameters_schema()["properties"] == {"q": {"type": "string"}}


async def test_mount_follows_pagination():
    session = ScriptedSession(
        pages=[
            mcp_types.ListToolsResult(tools=[_info("a")], next_cursor="1"),
            mcp_types.ListToolsResult(tools=[_info("b")]),
        ]
    )
    adapters = await mount_mcp_tools(session, "docs")
    assert [a.name() for a in adapters] == ["mcp__docs__a", "mcp__docs__b"]


async def test_mount_fails_closed_on_alias_collision():
    # Two names that sanitize apart but collide would let an external server
    # replace another tool under an approved name; identical names collide.
    session = ScriptedSession(
        pages=[mcp_types.ListToolsResult(tools=[_info("query"), _info("query")])]
    )
    with pytest.raises(ValueError, match="alias collision"):
        await mount_mcp_tools(session, "docs")


async def test_empty_description_gets_a_usable_default():
    adapter = _adapter(_info(description=""))
    assert "MCP tool from server 'docs'" in adapter.description()


# ── advisory metadata ──────────────────────────────────────────────────


def test_unannotated_tool_is_risky_and_not_read_only():
    adapter = _adapter()
    assert adapter.risk("{}") is RiskLevel.RISKY
    assert not adapter.read_only_hint()
    assert not adapter.parallel_safe("{}")


def test_read_only_hint_tool_is_safe():
    adapter = _adapter(_info(read_only=True))
    assert adapter.risk("{}") is RiskLevel.SAFE
    assert adapter.read_only_hint()
    assert adapter.parallel_safe("{}")


def test_always_grant_scope_is_tool_wide_not_per_args():
    adapter = _adapter()
    assert adapter.always_grant_scope('{"q": "a"}') == adapter.always_grant_scope('{"q": "b"}')
    assert adapter.always_grant_scope("{}") == ""


# ── execute: marshaling both directions ────────────────────────────────


async def test_execute_parses_args_and_joins_text_blocks():
    session = ScriptedSession(result=_text_result("first", "second"))
    result = await _adapter(session=session).execute('{"q": "refunds"}', ToolContext("/tmp"))
    assert session.calls == [("query", {"q": "refunds"})]
    assert result.content == "first\nsecond"
    assert not result.is_error and result.images == []


async def test_execute_empty_args_become_empty_object():
    session = ScriptedSession(result=_text_result("ok"))
    adapter = _adapter(session=session)
    await adapter.execute("", ToolContext("/tmp"))
    await adapter.execute("  {} ", ToolContext("/tmp"))
    assert session.calls == [("query", {}), ("query", {})]


async def test_execute_maps_images_and_drops_unknown_blocks():
    resource = mcp_types.EmbeddedResource(
        type="resource",
        resource=mcp_types.TextResourceContents(uri="mcp://docs/x", text="ignored"),
    )
    session = ScriptedSession(
        result=mcp_types.CallToolResult(
            content=[
                mcp_types.TextContent(type="text", text="caption"),
                mcp_types.ImageContent(type="image", data="QUJD", mime_type="image/png"),
                resource,
            ]
        )
    )
    result = await _adapter(session=session).execute("{}", ToolContext("/tmp"))
    assert result.content == "caption"
    assert result.images == [ImageContent(media_type="image/png", data="QUJD")]


async def test_invalid_arguments_are_an_error_result_not_a_call():
    session = ScriptedSession(result=_text_result("never"))
    adapter = _adapter(session=session)
    bad_json = await adapter.execute("{not json", ToolContext("/tmp"))
    not_object = await adapter.execute(json.dumps([1, 2]), ToolContext("/tmp"))
    assert bad_json.is_error and "invalid MCP tool arguments" in bad_json.content
    assert not_object.is_error and "not a JSON object" in not_object.content
    assert session.calls == []  # neither reached the server


async def test_server_declared_error_maps_to_error_result():
    session = ScriptedSession(result=_text_result("boom", is_error=True))
    result = await _adapter(session=session).execute("{}", ToolContext("/tmp"))
    assert result.is_error and "boom" in result.content


async def test_session_exception_becomes_error_result_never_raises():
    session = ScriptedSession(error=RuntimeError("server went away"))
    result = await _adapter(session=session).execute("{}", ToolContext("/tmp"))
    assert result.is_error and "server went away" in result.content


async def test_non_call_result_fails_closed():
    session = ScriptedSession(result=mcp_types.Result())
    result = await _adapter(session=session).execute("{}", ToolContext("/tmp"))
    assert result.is_error and "unsupported MCP result" in result.content


# ── mounted-name sanitation (OpenAI function-name rules) ───────────────


def test_valid_short_names_keep_the_readable_shape():
    assert mcp_tool_full_name("docs", "query-v2") == "mcp__docs__query-v2"


def test_sanitize_replaces_forbidden_characters():
    assert sanitize_name_segment("docs w/ spaces") == "docs-w--spaces"
    for name in (
        mcp_tool_full_name("docs w/ spaces", "query#result"),
        mcp_tool_full_name("文档服务", "读取文件"),
    ):
        assert name.startswith("mcp__")
        assert all(c.isascii() and (c.isalnum() or c in "_-") for c in name)


def test_colliding_readable_names_stay_distinct_and_stable():
    dotted = mcp_tool_full_name("docs", "read.file")
    spaced = mcp_tool_full_name("docs", "read file")
    assert dotted != spaced
    assert dotted == mcp_tool_full_name("docs", "read.file")  # stable across runs
    assert len(dotted) <= MAX_MCP_TOOL_NAME_LEN


def test_overlong_names_are_bounded_and_distinct():
    first = mcp_tool_full_name("server" * 20, "tool" * 20)
    second = mcp_tool_full_name("server" * 20, "tool" * 20 + "x")
    assert len(first) == MAX_MCP_TOOL_NAME_LEN == len(second)
    assert first != second
