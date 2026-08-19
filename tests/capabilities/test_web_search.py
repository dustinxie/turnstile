"""web_search tool — MockTransport replays the recorded Exa MCP response:
byte-level JSON-RPC wire assertions out, real SSE bytes back."""

import json
from pathlib import Path

import httpx
import pytest

from turnstile.capabilities.tools.web_search import EXA_MCP_URL, WebSearchTool
from turnstile.kernel.dtos import RiskLevel, ToolContext

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures"
RECORDED = (FIXTURES / "exa_search_response.txt").read_bytes()


def _tool(body: bytes = RECORDED, status: int = 200, api_key: str | None = None):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, content=body, headers={"Content-Type": "text/event-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return WebSearchTool(api_key=api_key, client=client), seen


def _args(query="FortiGate 7.6 release notes", **kw) -> str:
    return json.dumps({"query": query, **kw})


# ── wire shape ─────────────────────────────────────────────────────────


async def test_jsonrpc_wire_shape_keyless():
    tool, seen = _tool()
    await tool.execute(_args(), ToolContext("/tmp"))
    request = seen[0]
    assert str(request.url) == EXA_MCP_URL  # keyless: no exaApiKey param
    assert request.headers["accept"] == "application/json, text/event-stream"
    assert json.loads(request.content) == {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "web_search_exa",
            "arguments": {"query": "FortiGate 7.6 release notes", "numResults": 8},
        },
    }


async def test_api_key_rides_as_query_param():
    tool, seen = _tool(api_key="exa-k1")
    await tool.execute(_args(), ToolContext("/tmp"))
    assert str(seen[0].url) == f"{EXA_MCP_URL}?exaApiKey=exa-k1"


async def test_max_results_is_clamped():
    tool, seen = _tool()
    await tool.execute(_args(max_results=99), ToolContext("/tmp"))
    await tool.execute(_args(max_results=0), ToolContext("/tmp"))
    nums = [json.loads(r.content)["params"]["arguments"]["numResults"] for r in seen]
    assert nums == [20, 1]


# ── recorded response (real bytes; see fixtures/README) ────────────────


async def test_recorded_exa_response_renders():
    tool, _ = _tool()
    result = await tool.execute(_args(), ToolContext("/tmp"))
    assert not result.is_error
    assert result.content.startswith('Search results for "FortiGate 7.6 release notes":\n\n')
    assert "Title: FortiGate / FortiOS 7.6" in result.content
    assert "URL: https://docs.fortinet.com/product/fortigate/7.6" in result.content


# ── error taxonomy: always an error RESULT, never a raise ──────────────


async def test_bad_arguments_never_reach_the_network():
    tool, seen = _tool()
    not_json = await tool.execute("{oops", ToolContext("/tmp"))
    no_query = await tool.execute("{}", ToolContext("/tmp"))
    assert not_json.is_error and "invalid web_search arguments" in not_json.content
    assert no_query.is_error and "requires a 'query'" in no_query.content
    assert seen == []


async def test_http_failure_is_an_error_result():
    tool, _ = _tool(status=503)
    result = await tool.execute(_args(), ToolContext("/tmp"))
    assert result.is_error and "web_search failed" in result.content


async def test_jsonrpc_error_payload_is_an_error_result():
    error = b'data: {"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"rate limited"}}\n\n'
    tool, _ = _tool(body=error)
    result = await tool.execute(_args(), ToolContext("/tmp"))
    assert result.is_error and "no usable results" in result.content


async def test_empty_or_garbage_body_is_an_error_result():
    for body in (b"", b"data: not json\n\n", b'data: {"result":{"content":[]}}\n\n'):
        tool, _ = _tool(body=body)
        result = await tool.execute(_args(), ToolContext("/tmp"))
        assert result.is_error and "no usable results" in result.content


async def test_plain_json_body_is_accepted_too():
    body = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "Title: X"}]}}
    ).encode()
    tool, _ = _tool(body=body)
    result = await tool.execute(_args(), ToolContext("/tmp"))
    assert not result.is_error and "Title: X" in result.content


# ── advisory metadata ──────────────────────────────────────────────────


def test_web_search_is_safe_read_only_and_parallel():
    tool, _ = _tool()
    assert tool.name() == "web_search"
    assert tool.read_only_hint()
    assert tool.parallel_safe("{}")
    assert tool.risk("{}") is RiskLevel.SAFE
    assert tool.parameters_schema()["required"] == ["query"]
