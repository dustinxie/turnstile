"""Quality judge — the offer_continuation retry seam + the envelope's verdict.

When the model wants to stop, a small eval-LLM call grades the answer
against the question (JSON {"score": 0..1, "critique": "..."}). Low score
-> return critique text, which the kernel injects as a synthetic user
message and runs another round (bounded here by max_retries, and by the
kernel's max_continuations fuse regardless). Good score -> None, accept.

The verdict lands on the hook's OWN state (`last_verdict`) — the
L2-collector the service reads after TurnComplete for the envelope's
`signal`. No verdict recorded (fuse exit, judge never ran, eval failure)
reads as "not ok" downstream — never a fake ok.

FAIL-OPEN on eval failure (provider error, timeout, unparseable grade):
accept the answer and record nothing — a broken judge must not block turns
or spin retries.
"""

import asyncio
import json
from dataclasses import dataclass

from turnstile.kernel.dtos import (
    ChatOptions,
    Conversation,
    Reasoning,
    Role,
    TextDelta,
)
from turnstile.kernel.ports import LifecycleHooks, LlmProvider

_JUDGE_SYSTEM = """\
You are a strict quality judge for an internal support assistant. Score how
well the ASSISTANT ANSWER actually answers the USER QUESTION: correctness,
completeness, and whether it cites its sources. Output ONLY JSON:
{"score": <0.0-1.0>, "critique": "<one or two sentences: what is missing or wrong>"}"""

# A single-decision JSON grade: thinking OFF — a reasoning model (Nemotron-3)
# otherwise spends the whole budget thinking and returns no content.
_EVAL_OPTIONS = ChatOptions(max_tokens=200, temperature=0.0, enable_thinking=False)


@dataclass(frozen=True)
class QualityVerdict:
    """Product-local (deliberately NOT an L0 DTO): the judge's grade of the
    turn's final answer, read off the hook after TurnComplete."""

    score: float
    passed: bool
    critique: str = ""


class QualityJudgeHook(LifecycleHooks):
    """offer_continuation judge over an injected eval provider."""

    def __init__(
        self,
        provider: LlmProvider,
        threshold: float = 0.2,
        max_retries: int = 1,
        timeout_seconds: float = 15.0,
    ) -> None:
        """`threshold`: scores strictly below it request a retry. `max_retries`:
        retry rounds per turn this hook will ask for (the kernel's
        max_continuations fuse still bounds the total regardless).
        `timeout_seconds`: wall-clock bound on one grade — this hook runs at
        the END of a turn, so a stalled eval would hold a finished answer;
        expiry falls through the fail-open path like any other eval failure.
        Root passes the judge backend's configured request_timeout here, since
        the adapter's socket-level timeout cannot bound a slow token drip."""
        self._provider = provider
        self._threshold = threshold
        self._max_retries = max_retries
        self._timeout = timeout_seconds
        self._retries_used = 0
        self.last_verdict: QualityVerdict | None = None

    async def turn_start(self, convo: Conversation) -> None:
        # Per-turn state: a fresh turn gets fresh retries and no stale verdict
        # (a fuse-exited turn must not inherit the previous turn's "ok").
        self._retries_used = 0
        self.last_verdict = None

    async def offer_continuation(self, convo: Conversation) -> str | None:
        question, answer = _last_exchange(convo)
        if not answer:
            return None  # nothing to grade (blank/internal round)

        try:
            raw = await asyncio.wait_for(self._grade(question, answer), self._timeout)
            parsed = _parse_grade(raw)
        except Exception:  # noqa: BLE001 — fail-open: a broken judge must not block turns
            return None
        if parsed is None:
            return None  # unparseable grade: fail-open, record nothing
        score, critique = parsed

        passed = score >= self._threshold
        self.last_verdict = QualityVerdict(score=score, passed=passed, critique=critique)
        if passed or self._retries_used >= self._max_retries:
            return None
        self._retries_used += 1
        return (
            f"Your previous answer was judged insufficient "
            f"(score {score:.2f}): {critique}\n"
            f"Improve the answer — address the critique directly."
        )

    async def _grade(self, question: str, answer: str) -> str:
        from turnstile.kernel.dtos import Message

        prompt = f"USER QUESTION:\n{question}\n\nASSISTANT ANSWER:\n{answer}"
        content: list[str] = []
        reasoning: list[str] = []
        async for event in self._provider.chat_stream(
            [Message.system(_JUDGE_SYSTEM), Message.user(prompt)], [], _EVAL_OPTIONS
        ):
            if isinstance(event, TextDelta):
                content.append(event.text)
            elif isinstance(event, Reasoning):
                reasoning.append(event.text)
        # Some vLLM reasoning parsers route a thinking model's WHOLE streamed
        # output into the reasoning channel (even with thinking disabled).
        # The grade is wherever the model put it: content first, else reasoning.
        return "".join(content) or "".join(reasoning)


def _last_exchange(convo: Conversation) -> tuple[str, str]:
    """(last real user question, accumulated assistant text after it)."""
    question = ""
    answer_parts: list[str] = []
    for message in convo.messages:
        if message.role is Role.USER and message.text:
            question = message.text
            answer_parts = []
        elif message.role is Role.ASSISTANT and message.text:
            answer_parts.append(message.text)
    return question, "\n".join(answer_parts)


def _parse_grade(raw: str) -> tuple[float, str] | None:
    """{"score", "critique"} out of the eval response; tolerates prose/code
    fences around the JSON (reasoning models wrap output). None = unusable."""
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    score = data.get("score")
    if not isinstance(score, int | float):
        return None
    clamped = max(0.0, min(1.0, float(score)))
    return clamped, str(data.get("critique") or "")
