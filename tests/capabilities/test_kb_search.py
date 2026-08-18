"""kb_search tool — MockTransport answers both endpoints (embedding + Milvus
hybrid search): byte-level wire assertions out, canned or recorded rows back."""

import json
from pathlib import Path

import httpx
import pytest

from turnstile.capabilities.tools.kb_search import KbSearchTool
from turnstile.kernel.dtos import RiskLevel, ToolContext

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures"

EMBED_URL = "https://gpu.example/embedding"
SEARCH_URL = "https://milvus.example/hybrid_search_generic"
EXPR = 'doc_id in ["hrus#e35025e3-58c2-4d6c-8e59-4f62277b3e6e"]'
VECTOR = [0.1, 0.2, 0.3]

ROWS = [
    {
        "content": "PTO accrues at 1.5 days/month.",
        "ref": "pto.md#L12",
        "doc_id": "hrus#e35",
        "score": 0.91,
    },
    {
        "content": "Carry-over caps at 10 days.",
        "ref": "pto.md#L40",
        "doc_id": "hrus#e35",
        "score": 0.84,
    },
]


def _tool(
    embed_response=None,
    search_response=None,
    embed_status=200,
    search_status=200,
    token="mv-test",
    **kw,
):
    """Tool wired to a MockTransport that answers both endpoints and records
    each outgoing request for wire assertions."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if str(request.url) == EMBED_URL:
            body = embed_response if embed_response is not None else [VECTOR]
            return httpx.Response(embed_status, json=body)
        body = search_response if search_response is not None else ROWS
        return httpx.Response(search_status, json=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = KbSearchTool(
        embedding_url=EMBED_URL,
        milvus_url=SEARCH_URL,
        milvus_token=token,
        collection="agentassist_user_datasource",
        expr=EXPR,
        client=client,
        **kw,
    )
    return tool, seen


def _args(query="pto policy") -> str:
    return json.dumps({"query": query})


# ── wire shape ─────────────────────────────────────────────────────────


async def test_embedding_request_wire_shape():
    tool, seen = _tool()
    await tool.execute(_args(), ToolContext("/tmp"))
    embed_request = seen[0]
    assert str(embed_request.url) == EMBED_URL
    assert json.loads(embed_request.content) == {
        "query": ["pto policy"],  # batch of one, matching the [[vec]] response
        "batch_size": 1,
        "normalize": True,
        "dim_size": 4096,
    }


async def test_search_request_wire_shape_and_expr_verbatim():
    tool, seen = _tool(limit=7, dim_size=2048)
    await tool.execute(_args(), ToolContext("/tmp"))
    assert json.loads(seen[0].content)["dim_size"] == 2048
    search_request = seen[1]
    assert str(search_request.url) == SEARCH_URL
    assert search_request.headers["authorization"] == "Bearer mv-test"
    payload = json.loads(search_request.content)
    assert payload == {
        "collection_names": ["agentassist_user_datasource"],
        "query_embedding": VECTOR,  # the [[vec]] batch unwrapped to one vector
        "k": 7,
        "query": "pto policy",
        "expr": EXPR,  # opaque pass-through: the tool never builds or parses it
        "output_fields": ["content", "ref", "doc_id"],
    }


async def test_empty_token_omits_authorization_header():
    # Live-probe finding: "Bearer " with no token is an illegal header value.
    tool, seen = _tool(token="")
    result = await tool.execute(_args(), ToolContext("/tmp"))
    assert not result.is_error
    assert "authorization" not in seen[1].headers


# ── recorded responses (real bytes; see fixtures/README) ───────────────


async def test_recorded_responses_replay_end_to_end():
    embed = json.loads((FIXTURES / "kb_embedding_response.json").read_text())
    rows = json.loads((FIXTURES / "kb_search_response.json").read_text())
    tool, seen = _tool(embed_response=embed, search_response=rows, dim_size=64, limit=5)
    result = await tool.execute(_args("what is my leave benefits"), ToolContext("/tmp"))
    assert not result.is_error
    # the recorded [[vec]] batch is unwrapped and shipped verbatim
    payload = json.loads(seen[1].content)
    assert payload["query_embedding"] == embed[0]
    assert len(payload["query_embedding"]) == 64  # fixture recorded at dim_size=64
    # real rows carry an extra `weighted_score` key — rendering must not care
    assert result.content.startswith(
        "[1] Benefits/Health Resources/2026 Benefits/"
        "FTNT 2026 OE Webinar FAQ Final v2.pdf#L521 (score 0.033)"
    )
    assert "paternity leave policy" in result.content
    assert "[5] " in result.content


# ── result rendering ───────────────────────────────────────────────────


async def test_hits_render_as_numbered_cited_excerpts():
    tool, _ = _tool()
    result = await tool.execute(_args(), ToolContext("/tmp"))
    assert not result.is_error
    assert "[1] pto.md#L12 (score 0.910)" in result.content
    assert "PTO accrues at 1.5 days/month." in result.content
    assert "[2] pto.md#L40" in result.content


async def test_envelope_response_shape_is_unwrapped():
    tool, _ = _tool(search_response={"results": ROWS})
    result = await tool.execute(_args(), ToolContext("/tmp"))
    assert not result.is_error and "[2] pto.md#L40" in result.content


async def test_no_hits_is_a_plain_answer_not_an_error():
    tool, _ = _tool(search_response=[])
    result = await tool.execute(_args(), ToolContext("/tmp"))
    assert not result.is_error
    assert result.content == "kb_search: no results"


# ── error taxonomy: always an error RESULT, never a raise ──────────────


async def test_bad_arguments_never_reach_the_network():
    tool, seen = _tool()
    not_json = await tool.execute("{not json", ToolContext("/tmp"))
    no_query = await tool.execute("{}", ToolContext("/tmp"))
    assert not_json.is_error and "invalid kb_search arguments" in not_json.content
    assert no_query.is_error and "requires a 'query'" in no_query.content
    assert seen == []


async def test_embedding_failure_is_an_error_result():
    tool, seen = _tool(embed_status=503)
    result = await tool.execute(_args(), ToolContext("/tmp"))
    assert result.is_error and "embedding failed" in result.content
    assert len(seen) == 1  # never reached the search endpoint


async def test_search_failure_is_an_error_result():
    tool, _ = _tool(search_status=500)
    result = await tool.execute(_args(), ToolContext("/tmp"))
    assert result.is_error and "search failed" in result.content


async def test_malformed_embedding_response_is_an_error_result():
    tool, _ = _tool(embed_response={"not": "a batch"})
    result = await tool.execute(_args(), ToolContext("/tmp"))
    assert result.is_error and "embedding failed" in result.content


# ── advisory metadata ──────────────────────────────────────────────────


def test_kb_search_is_safe_read_only_and_parallel():
    tool, _ = _tool()
    assert tool.name() == "kb_search"
    assert tool.read_only_hint()
    assert tool.parallel_safe("{}")
    assert tool.risk("{}") is RiskLevel.SAFE
    assert tool.parameters_schema()["required"] == ["query"]
