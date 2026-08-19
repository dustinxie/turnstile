"""Quality judge — unit grading paths + a full-engine retry journey where a
low-scored answer triggers one critique-driven re-run."""

import json

import pytest

from turnstile.kernel.dtos import Conversation, Done, Message, ProviderError, Reasoning, TextDelta
from turnstile.kernel.engine import Agent
from turnstile.kernel.testkit import ScriptedProvider, StallThenProvider
from turnstile.products.hooks.quality_judge import QualityJudgeHook, _parse_grade

pytestmark = pytest.mark.unit


def _grade_stream(score: float, critique: str = "") -> list:
    return [TextDelta(json.dumps({"score": score, "critique": critique})), Done()]


def _convo(question="what is my leave benefits", answer="PTO accrues monthly.") -> Conversation:
    convo = Conversation()
    convo.push(Message.user(question))
    convo.push(Message.assistant(answer))
    return convo


async def _offered(judge: QualityJudgeHook, convo: Conversation) -> str | None:
    await judge.turn_start(convo)
    return await judge.offer_continuation(convo)


# ── grading paths ──────────────────────────────────────────────────────


async def test_good_score_accepts_and_records_passing_verdict():
    judge = QualityJudgeHook(ScriptedProvider(rounds=[_grade_stream(0.9, "solid")]))
    assert await _offered(judge, _convo()) is None
    assert judge.last_verdict is not None
    assert judge.last_verdict.passed and judge.last_verdict.score == 0.9


async def test_low_score_returns_critique_then_exhausts_retries():
    judge = QualityJudgeHook(
        ScriptedProvider(rounds=[_grade_stream(0.1, "misses the carry-over cap")] * 2),
        max_retries=1,
    )
    convo = _convo()
    first = await _offered(judge, convo)
    assert first is not None and "misses the carry-over cap" in first
    assert judge.last_verdict is not None and not judge.last_verdict.passed
    # same turn, second offer: retries exhausted -> accept despite low score
    assert await judge.offer_continuation(convo) is None


async def test_eval_failure_is_fail_open_and_records_nothing():
    failing = ScriptedProvider(rounds=[ProviderError(message="judge backend down")])
    judge = QualityJudgeHook(failing)
    assert await _offered(judge, _convo()) is None
    assert judge.last_verdict is None  # absent verdict = never a fake ok


async def test_stalled_eval_times_out_fail_open():
    # The judge runs at the END of a turn: a hung eval must not hold the
    # finished answer, so the wall-clock bound expires into the fail-open path.
    stalled = StallThenProvider(stall_calls=1, events=[])
    judge = QualityJudgeHook(stalled, timeout_seconds=0.01)
    assert await _offered(judge, _convo()) is None
    assert judge.last_verdict is None


async def test_unparseable_grade_is_fail_open():
    judge = QualityJudgeHook(ScriptedProvider(rounds=[[TextDelta("not json at all"), Done()]]))
    assert await _offered(judge, _convo()) is None
    assert judge.last_verdict is None


async def test_blank_answer_is_not_graded():
    judge = QualityJudgeHook(ScriptedProvider(rounds=[]))
    convo = Conversation()
    convo.push(Message.user("q"))
    assert await _offered(judge, convo) is None
    assert judge.last_verdict is None


async def test_turn_start_resets_retries_and_verdict():
    judge = QualityJudgeHook(
        ScriptedProvider(rounds=[_grade_stream(0.1, "thin")] * 3), max_retries=1
    )
    convo = _convo()
    assert await _offered(judge, convo) is not None  # retry 1 used
    assert await judge.offer_continuation(convo) is None  # exhausted
    # new turn: budget back, stale verdict cleared before regrade
    assert await _offered(judge, convo) is not None
    assert judge.last_verdict is not None


def test_parse_grade_tolerates_fences_and_clamps():
    assert _parse_grade('```json\n{"score": 0.7, "critique": "ok"}\n```') == (0.7, "ok")
    assert _parse_grade('the grade is {"score": 3, "critique": ""} thanks') == (1.0, "")
    assert _parse_grade('{"score": "high"}') is None
    assert _parse_grade("") is None


# ── full-engine journey: critique drives a re-run ──────────────────────


async def test_low_score_reruns_turn_and_second_answer_lands():
    main = ScriptedProvider(
        rounds=[
            [TextDelta("Leave exists."), Done()],  # thin first answer
            [TextDelta("PTO accrues at 1.5 days/month; carry-over caps at 10."), Done()],
        ]
    )
    judge = QualityJudgeHook(
        ScriptedProvider(
            rounds=[_grade_stream(0.05, "no specifics"), _grade_stream(0.95, "complete")]
        )
    )
    outcome = await Agent(provider=main, hooks=[judge]).run_to_completion("leave benefits?")

    assert len(main.calls) == 2  # the critique forced a second round
    # the synthetic user message carried the critique into the model's context
    injected = [m.text for m in main.calls[1].messages if m.role.value == "user"]
    assert any("no specifics" in text for text in injected)
    assert "carry-over caps" in outcome.text
    assert judge.last_verdict is not None and judge.last_verdict.passed  # final grade wins


async def test_grade_in_the_reasoning_channel_still_counts():
    # some vLLM reasoning parsers stream a thinking model's whole output as
    # reasoning_content even with thinking off — the grade is wherever it landed
    grade = json.dumps({"score": 0.9, "critique": "fine"})
    judge = QualityJudgeHook(
        ScriptedProvider(rounds=[[Reasoning(grade[:5]), Reasoning(grade[5:]), Done()]])
    )
    convo = _convo("q", "a")
    assert await judge.offer_continuation(convo) is None
    assert judge.last_verdict is not None and judge.last_verdict.score == 0.9
