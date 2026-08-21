"""The response envelope — the service-added closing event of every turn.

Everything a chat client needs to FINISH rendering a turn, in one frame:
the final answer (from the snapshot, so a judge-retried turn reports the
accepted answer, never a concatenation of drafts), a deterministic quality
`signal`, and the ground-truth `references` drained from the collector (the
L2-collector pattern, kernel-loop-structure.md §I.2).

The signal is ENDPOINT-SET, never model-claimed:
  no_answer    - the turn did not end cleanly (cancel / fuse / error), or
                 produced no text. NEVER a fake ok (architecture §2).
  ok           - the judge graded the answer at/above threshold.
  low_quality  - the judge graded it below threshold and retries ran out;
                 `score` carries the grade.
  unjudged     - an answer with no verdict: judge not deployed, or its
                 eval failed open. Distinct from ok — absence of grading
                 is not a pass.

Judge and references arrive duck-typed off the AssembledAgent bundle — the
service never imports product classes (the c6 contract).
"""

from typing import Any

from turnstile.kernel.dtos import Role, SessionSnapshot


def final_answer(snapshot: SessionSnapshot | None) -> str:
    """The turn's accepted answer: the LAST assistant message. Judge retries
    leave rejected drafts earlier in history; the final one is the answer."""
    if snapshot is None:
        return ""
    for message in reversed(snapshot.messages):
        if message.role is Role.ASSISTANT and message.text and not message.synthetic:
            return message.text
    return ""


def build_envelope(
    conversation_id: str,
    snapshot: SessionSnapshot | None,
    stop_reason: str,
    judge: Any = None,
    references: list[Any] | None = None,
) -> dict:
    answer = final_answer(snapshot)
    verdict = getattr(judge, "last_verdict", None)

    if stop_reason != "stopped" or not answer:
        signal = "no_answer"
    elif verdict is None:
        signal = "unjudged"
    elif verdict.passed:
        signal = "ok"
    else:
        signal = "low_quality"

    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "signal": signal,
        "score": getattr(verdict, "score", None),
        "references": [{"tool": r.tool, "ref": r.ref, "url": r.url} for r in (references or [])],
        "stop_reason": stop_reason,
    }
