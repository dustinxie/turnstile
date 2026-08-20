"""Live end-to-end — a real question answered by the assembled agent.

The whole stack in one turn: root builds the support_bot from config, the
model decides to call kb_search, the tool embeds the query and searches the
real HR US datasource in Milvus, the model answers from those chunks, the
judge grades it, and the reference collector reports the documents the
answer actually drew on.

Marked `integration`: `make check` never collects it. Run it on a box with
routes to both hosts via `make test-all` (or `pytest -m integration -v`).
Endpoints are env-overridable so a different deployment can reuse the test.
"""

import os

import pytest

from turnstile.config import Config
from turnstile.kernel.dtos import StopReason
from turnstile.root import assemble

pytestmark = pytest.mark.integration

LLM_BASE_URL = os.environ.get("DS4_BASE_URL", "https://10.83.135.205/model-long/v1")
GPU_HOST = os.environ.get("GPU_HOST", "https://10.83.135.206")
# The HR US SharePoint datasource (owner#ds_id is the Milvus row key).
HRUS_EXPR = 'doc_id in ["hrus#e35025e3-58c2-4d6c-8e59-4f62277b3e6e"]'


def _cfg() -> Config:
    return Config(
        _env_file=None,  # type: ignore[call-arg]
        llm={"base_url": LLM_BASE_URL, "model": "model-fast", "context_window": 128_000},
        judge_llm={"base_url": LLM_BASE_URL, "model": "model-fast", "request_timeout": 30.0},
        kb={
            "embedding_url": f"{GPU_HOST}/api/v1/embedding/qwen3-embedding-8b",
            "milvus_url": f"{GPU_HOST}/api/v2/vectordb/hybrid_search_generic",
            "collection": "agentassist_user_datasource",
            "expr": HRUS_EXPR,
            "limit": 5,
        },
        # Keep the turn on the knowledge base: this asserts KB retrieval, not
        # the model's taste in web results.
        web_enabled=False,
    )


async def test_leave_benefits_question_end_to_end():
    bundle = assemble(_cfg(), session_id="live-e2e")
    outcome = await bundle.agent.run_to_completion("what is my leave benefits")

    assert outcome.error is None, outcome.error
    assert outcome.stop is StopReason.STOPPED
    assert outcome.text.strip(), "the turn produced no answer"

    # the model reached for the knowledge base, and the tool returned chunks
    assert outcome.tool_results, "kb_search was never called"
    assert not outcome.tool_results[0].is_error, outcome.tool_results[0].content

    # ground-truth references: what the retrieval actually surfaced
    references = bundle.references.take()
    assert references, "no references collected from the kb_search result"
    assert {r.tool for r in references} == {"kb_search"}

    # the judge graded the answer (a recorded verdict, pass or fail)
    assert bundle.judge is not None
    assert bundle.judge.last_verdict is not None, "judge recorded no verdict"
    assert 0.0 <= bundle.judge.last_verdict.score <= 1.0

    # the turn was snapshotted for resume
    snapshot = bundle.store.load("live-e2e")
    assert snapshot is not None and snapshot.turn_counter >= 1
