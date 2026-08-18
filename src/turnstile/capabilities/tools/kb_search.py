"""KB search tool — query -> embedding server -> Milvus hybrid search.

A pure retrieval leg. The tool embeds the model's query text and searches
ONE Milvus collection under a FIXED scope filter (`expr`) that is passed
through to the search service verbatim. It never resolves users, owners,
or grants — the embedder decides scope at mount time (root builds `expr`
per session), so the model can never widen its own retrieval scope: the
only input it controls is the query string.

Wire contract (matches the internal GPU stack):
  embedding server  POST {query: [text], batch_size, normalize, dim_size}
                    -> [[vec]] (batch of one)
  hybrid search     POST {collection_names, query_embedding, k, query,
                    expr, output_fields} -> rows [{content, ref, doc_id,
                    score}] (flat list, or {results: [...]} envelope) —
                    dense + BM25 legs combined server-side under `score`.

TODO(M2, reranker): optional second-stage cross-encoder pass — a
`rerank_url` param (qwen3-reranker-8b: POST {query, documents} ->
relevance-ordered indices, optionally {index, relevance_score}); reorder
rows between _search and _render, best-effort (rerank failure -> keep
retrieval order, never an error). Then raise the retrieval `limit` to
20-25 at the embedder and render only the top slice post-rerank.
"""

import json

import httpx

from turnstile.kernel.dtos import ToolContext, ToolResult
from turnstile.kernel.ports import Tool

_OUTPUT_FIELDS = ["content", "ref", "doc_id"]


class KbSearchTool(Tool):
    """Scoped vector+keyword search over an indexed document collection."""

    def __init__(
        self,
        *,
        embedding_url: str,
        dim_size: int = 4096,
        milvus_url: str,
        milvus_token: str,
        collection: str,
        expr: str,
        limit: int = 10,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        """`embedding_url` and `milvus_url` are FULL endpoint URLs. `dim_size`
        must match the collection schema (a mismatch fails the search, or
        worse, scores garbage). `expr` is opaque here — e.g.
        'doc_id in ["hrus#<ds_id>"]' — and rides the request untouched."""
        self._embedding_url = embedding_url
        self._dim_size = dim_size
        self._milvus_url = milvus_url
        self._milvus_token = milvus_token
        self._collection = collection
        self._expr = expr
        self._limit = limit
        # Injectable client: tests pass one with a MockTransport answering
        # both endpoints; production shares a connection pool.
        self._client = client or httpx.AsyncClient(verify=False, timeout=15.0)

    def name(self) -> str:
        return "kb_search"

    def description(self) -> str:
        return (
            "Search the knowledge base for document chunks relevant to a "
            "query. Returns the best-matching excerpts with their source "
            "file references."
        )

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for, phrased as a concise search query.",
                }
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
                call_id="", content=f"invalid kb_search arguments: {e}", is_error=True
            )
        query = parsed.get("query") if isinstance(parsed, dict) else None
        if not query or not isinstance(query, str):
            return ToolResult(
                call_id="", content="kb_search requires a 'query' string argument", is_error=True
            )

        try:
            vector = await self._embed(query)
        except (httpx.HTTPError, ValueError) as e:
            return ToolResult(
                call_id="", content=f"kb_search embedding failed: {e}", is_error=True
            )

        try:
            rows = await self._search(query, vector)
        except (httpx.HTTPError, ValueError, TypeError) as e:
            return ToolResult(call_id="", content=f"kb_search search failed: {e}", is_error=True)

        if not rows:
            return ToolResult(call_id="", content="kb_search: no results")
        return ToolResult(call_id="", content=self._render(rows))

    async def _embed(self, query: str) -> list[float]:
        """One query in, one unit vector out (server answers [[vec]])."""
        response = await self._client.post(
            self._embedding_url,
            json={
                "query": [query],
                "batch_size": 1,
                "normalize": True,
                "dim_size": self._dim_size,
            },
        )
        response.raise_for_status()
        batch = response.json()
        if not isinstance(batch, list) or not batch:
            raise ValueError(f"unexpected embedding response shape: {type(batch).__name__}")
        return batch[0]

    async def _search(self, query: str, vector: list[float]) -> list[dict]:
        headers = {"Authorization": f"Bearer {self._milvus_token}"} if self._milvus_token else {}
        response = await self._client.post(
            self._milvus_url,
            json={
                "collection_names": [self._collection],
                "query_embedding": vector,
                "k": self._limit,
                "query": query,
                "expr": self._expr,
                "output_fields": _OUTPUT_FIELDS,
            },
            headers=headers,
        )
        response.raise_for_status()
        raw = response.json()
        # Flat list today; single-collection {results: [...]} envelope tomorrow.
        rows = raw.get("results") if isinstance(raw, dict) else raw
        if not isinstance(rows, list):
            raise TypeError(f"unexpected search response shape: {type(raw).__name__}")
        return rows

    @staticmethod
    def _render(rows: list[dict]) -> str:
        """Numbered excerpts the model can cite: `ref` is the source pointer
        ("<filename>#L<line>"), score kept for relative relevance."""
        parts: list[str] = []
        for n, row in enumerate(rows, start=1):
            ref = row.get("ref") or row.get("doc_id") or "unknown source"
            score = row.get("score")
            header = f"[{n}] {ref}" + (f" (score {score:.3f})" if score is not None else "")
            parts.append(f"{header}\n{row.get('content', '')}")
        return "\n\n".join(parts)
