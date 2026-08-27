"""The envelope — signal mapping unit tests + streamed journeys: unjudged
live-text turns, judged withhold-until-turn-complete turns, and the never-fake-ok
paths (cancel, low quality)."""

import json

import httpx
import pytest

from turnstile.capabilities.persistence.memory_store import MemorySessionStore
from turnstile.kernel.dtos import (
    SNAPSHOT_VERSION,
    Done,
    Message,
    SessionSnapshot,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolContext,
    ToolResult,
)
from turnstile.kernel.engine import Agent
from turnstile.kernel.ports import Tool
from turnstile.kernel.testkit import ScriptedProvider
from turnstile.products.hooks.quality_judge import QualityJudgeHook
from turnstile.products.middleware.references import ReferenceCollector
from turnstile.root import AssembledAgent
from turnstile.service.app import create_app
from turnstile.service.envelope import build_envelope
from turnstile.service.registry import ConversationRegistry

pytestmark = pytest.mark.service

KB_CONTENT = "[1] Handbook.pdf#L755 (score 0.033)\nVacation policy text."


def _snap(*messages: Message) -> SessionSnapshot:
    return SessionSnapshot(version=SNAPSHOT_VERSION, messages=list(messages))


class _Verdict:
    def __init__(self, score: float, passed: bool) -> None:
        self.score, self.passed = score, passed


class _JudgeLike:
    def __init__(self, verdict) -> None:
        self.last_verdict = verdict


# ── signal mapping (unit) ──────────────────────────────────────────────


def test_signals_never_fake_ok():
    answered = _snap(Message.user("q"), Message.assistant("a"))
    # clean turn, no judge deployed -> unjudged (absence of grading != pass)
    assert build_envelope("c", answered, "stopped")["signal"] == "unjudged"
    # judge deployed but eval failed open -> still unjudged
    assert build_envelope("c", answered, "stopped", judge=_JudgeLike(None))["signal"] == "unjudged"
    # graded pass / fail
    ok = build_envelope("c", answered, "stopped", judge=_JudgeLike(_Verdict(0.9, True)))
    low = build_envelope("c", answered, "stopped", judge=_JudgeLike(_Verdict(0.1, False)))
    assert (ok["signal"], ok["score"]) == ("ok", 0.9)
    assert (low["signal"], low["score"]) == ("low_quality", 0.1)
    # dirty exits: cancelled / fuse -> no_answer even with a passing verdict
    cancelled = build_envelope("c", answered, "cancelled", judge=_JudgeLike(_Verdict(0.9, True)))
    assert cancelled["signal"] == "no_answer"
    # clean stop but nothing said -> no_answer
    assert build_envelope("c", _snap(Message.user("q")), "stopped")["signal"] == "no_answer"


def test_answer_is_the_accepted_one_not_a_draft_concat():
    retried = _snap(
        Message.user("q"),
        Message.assistant("thin draft"),
        Message.synthetic_user("judged insufficient: no specifics"),
        Message.assistant("the accepted answer"),
    )
    envelope = build_envelope("c", retried, "stopped")
    assert envelope["answer"] == "the accepted answer"


def test_no_collector_means_answer_untouched_and_no_references():
    # the lego piece is absent (root wired no collector): the model's inline
    # markers stay as typed, no section, empty list
    envelope = build_envelope(
        "c", _snap(Message.user("q"), Message.assistant("see [1]")), "stopped"
    )
    assert envelope["answer"] == "see [1]"
    assert envelope["references"] == []


# ── streamed journeys ──────────────────────────────────────────────────


class _KbDouble(Tool):
    def __init__(self, content: str = KB_CONTENT) -> None:
        self._content = content

    def name(self) -> str:
        return "kb_search"

    def description(self) -> str:
        return "kb"

    def parameters_schema(self) -> dict:
        return {"type": "object"}

    def read_only_hint(self) -> bool:
        return True

    async def execute(self, args: str, ctx: ToolContext) -> ToolResult:
        return ToolResult(call_id="", content=self._content)


def _app(provider_factory, judge_factory=None, kb_content=KB_CONTENT, file_root=None):
    from turnstile.config import Config

    cfg = Config(
        _env_file=None,  # type: ignore[call-arg]
        file_root=file_root,
        llm={"base_url": "https://ds4.example/v1", "model": "model-fast"},
        kb={
            "embedding_url": "https://e/x",
            "milvus_url": "https://m/x",
            "collection": "c",
            "expr": "e",
        },
    )
    store = MemorySessionStore()

    def scripted_assemble(cfg, session_id, store):
        references = ReferenceCollector()
        judge = judge_factory() if judge_factory else None
        agent = Agent(
            provider=provider_factory(),
            tools={"kb_search": _KbDouble(kb_content)},
            hooks=([judge] if judge else []) + [store.hook(session_id)],
            middleware=[references],
            session_id=session_id,
            resume=store.load(session_id),
            keep_interrupted_context=True,
        )
        return AssembledAgent(agent=agent, references=references, store=store, judge=judge)

    app = create_app(cfg)
    app.state.store = store
    app.state.registry = ConversationRegistry(cfg, store, assemble=scripted_assemble)
    return app


async def _post_stream(app, conversation_id: str, text: str) -> list[tuple[str, dict]]:
    async with (
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client,
        client.stream(
            "POST", f"/v1/conversations/{conversation_id}/messages", json={"text": text}
        ) as response,
    ):
        events, name = [], None
        async for line in response.aiter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:") and name:
                events.append((name, json.loads(line.split(":", 1)[1].strip())))
                name = None
        return events


def _grade(score: float, critique: str = "") -> list:
    return [TextDelta(json.dumps({"score": score, "critique": critique})), Done()]


async def test_unjudged_turn_streams_live_and_closes_with_the_envelope():
    app = _app(
        lambda: ScriptedProvider(
            rounds=[
                [ToolCallEvent(ToolCall("t1", "kb_search", '{"query": "v"}')), Done()],
                [TextDelta("the "), TextDelta("answer [1]"), Done()],
            ]
        )
    )
    events = await _post_stream(app, "c1", "q")
    names = [n for n, _ in events]
    assert "text_delta" in names  # no judge -> live streaming
    assert names[-2:] == ["turn_complete", "envelope"]  # envelope closes the stream
    envelope = events[-1][1]
    # the References section rides the envelope's answer; no file store and
    # no region in this fixture -> a plain (unlinked) title
    assert envelope["answer"] == "the answer [1]\n\n### References\n- [1] Handbook.pdf"
    assert envelope["signal"] == "unjudged" and envelope["score"] is None
    assert envelope["references"] == [
        {
            "n": 1,
            "title": "Handbook.pdf",
            "url": None,
            "cited": True,
            # the raw source rides along so history can re-mint links later
            "tool": "kb_search",
            "region": None,
            "path": "Handbook.pdf",
            "fragment": "L755",
        }
    ]
    await app.state.registry.shutdown_all()


async def test_cited_kb_docs_link_to_recipient_bound_file_tokens(tmp_path):
    # a real store: the cited doc exists under <root>/<region>/<path>
    (tmp_path / "hrus" / "Policies").mkdir(parents=True)
    (tmp_path / "hrus" / "Policies" / "Handbook.pdf").write_bytes(b"%PDF handbook")
    kb = "[1] hrus::Policies/Handbook.pdf#L755 (score 0.033)\nVacation policy text."
    app = _app(
        lambda: ScriptedProvider(
            rounds=[
                [ToolCallEvent(ToolCall("t1", "kb_search", "{}")), Done()],
                [TextDelta("Vacation is covered [1]."), Done()],
            ]
        ),
        kb_content=kb,
        file_root=str(tmp_path),
    )
    events = await _post_stream(app, "c1", "q")
    envelope = events[-1][1]
    (reference,) = envelope["references"]
    assert reference["title"] == "Handbook.pdf" and reference["cited"] is True
    assert reference["url"].startswith("/v1/files/") and reference["url"].endswith("#L755")
    assert f"- [1] [Handbook.pdf]({reference['url']})" in envelope["answer"]  # list item, md link

    # the link opens for the principal it was minted for (dev mode: anonymous)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        served = await c.get(reference["url"].split("#", 1)[0])
    assert served.status_code == 200 and served.content == b"%PDF handbook"
    await app.state.registry.shutdown_all()


async def test_judged_turn_withholds_text_and_reports_ok():
    app = _app(
        lambda: ScriptedProvider(rounds=[[TextDelta("graded answer"), Done()]]),
        judge_factory=lambda: QualityJudgeHook(ScriptedProvider(rounds=[_grade(0.9)])),
    )
    events = await _post_stream(app, "c1", "q")
    names = [n for n, _ in events]
    assert "text_delta" not in names  # withhold-until-turn-complete: no live drafts
    envelope = events[-1][1]
    assert envelope == {
        "conversation_id": "c1",
        "answer": "graded answer",
        "signal": "ok",
        "score": 0.9,
        "references": [],
        "stop_reason": "stopped",
    }
    await app.state.registry.shutdown_all()


async def test_judged_retry_reports_the_accepted_answer_only():
    app = _app(
        lambda: ScriptedProvider(
            rounds=[[TextDelta("thin"), Done()], [TextDelta("full answer"), Done()]]
        ),
        judge_factory=lambda: QualityJudgeHook(
            ScriptedProvider(rounds=[_grade(0.05, "no specifics"), _grade(0.95)])
        ),
    )
    events = await _post_stream(app, "c1", "q")
    envelope = events[-1][1]
    assert envelope["answer"] == "full answer"  # never "thin" + "full answer"
    assert envelope["signal"] == "ok"
    await app.state.registry.shutdown_all()


async def test_references_are_per_turn_not_cumulative():
    app = _app(
        lambda: ScriptedProvider(
            rounds=[
                [ToolCallEvent(ToolCall("t1", "kb_search", "{}")), Done()],
                [TextDelta("a1"), Done()],
                [TextDelta("a2, from memory"), Done()],  # turn 2: no tool call
            ]
        )
    )
    first = await _post_stream(app, "c1", "q1")
    second = await _post_stream(app, "c1", "q2")
    assert len(first[-1][1]["references"]) == 1
    assert second[-1][1]["references"] == []  # turn 1's refs were drained
    await app.state.registry.shutdown_all()


async def test_reloaded_conversation_carries_verdict_and_relinked_references(tmp_path):
    # the sidecar makes the badge + References survive a reload; kb links are
    # re-minted for the reader (stored tokens would expire before the chat)
    (tmp_path / "hrus" / "Policies").mkdir(parents=True)
    (tmp_path / "hrus" / "Policies" / "Handbook.pdf").write_bytes(b"%PDF handbook")
    kb = "[1] hrus::Policies/Handbook.pdf#L755 (score 0.033)\nVacation policy text."
    app = _app(
        lambda: ScriptedProvider(
            rounds=[
                [ToolCallEvent(ToolCall("t1", "kb_search", "{}")), Done()],
                [TextDelta("Vacation is covered [1]."), Done()],
            ]
        ),
        judge_factory=lambda: QualityJudgeHook(ScriptedProvider(rounds=[_grade(0.8)])),
        kb_content=kb,
        file_root=str(tmp_path),
    )
    events = await _post_stream(app, "c1", "q")
    live = events[-1][1]
    assert live["signal"] == "ok"

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        view = (await c.get("/v1/conversations/c1")).json()
        answer = [m for m in view["messages"] if m["role"] == "assistant"][-1]
        assert (answer["signal"], answer["score"]) == ("ok", 0.8)
        (ref,) = answer["references"]
        assert ref["title"] == "Handbook.pdf" and ref["cited"] is True
        assert ref["url"].startswith("/v1/files/") and ref["url"].endswith("#L755")
        assert ref["url"] != live["references"][0]["url"]  # minted fresh, not the stored one
        served = await c.get(ref["url"].split("#", 1)[0])
        assert served.status_code == 200  # and it opens
        # user rows carry no sidecar
        assert "signal" not in next(m for m in view["messages"] if m["role"] == "user")
    await app.state.registry.shutdown_all()
