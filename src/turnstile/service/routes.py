"""The turn endpoint — POST a message, receive the turn as Server-Sent Events.

The SSE stream IS the driver protocol serialized verbatim (architecture.md
§2): one SSE event per AgentEvent — `event:` carries the snake_cased event
class, `data:` its fields as JSON — ending with the terminal turn_complete.
(The service-added envelope event lands in M4-c5; withhold-until-turn-complete rides
with it.)

A POST while this conversation's turn is still streaming STEERS it
(kernel-native: the prompt folds in at the next round boundary) and answers
202 immediately — the folded content arrives on the already-open stream; the
events queue is single-consumer, so there is never a second pump.
"""

import asyncio
import json
import re
from dataclasses import asdict, is_dataclass
from enum import Enum

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from turnstile.kernel import events as ev
from turnstile.service.envelope import build_envelope

router = APIRouter()

_SNAKE_RE = re.compile(r"(?<!^)(?=[A-Z])")


class MessageIn(BaseModel):
    text: str = Field(min_length=1)


def event_name(event: object) -> str:
    """AgentEvent class -> SSE event name (TextDelta -> text_delta)."""
    return _SNAKE_RE.sub("_", type(event).__name__).lower()


def event_data(event: object) -> str:
    """AgentEvent -> JSON. Nested DTOs recurse via asdict; enums flatten to
    their values; anything exotic degrades to str rather than erroring the
    stream mid-turn."""
    payload = asdict(event) if is_dataclass(event) and not isinstance(event, type) else {}
    return json.dumps(
        payload,
        default=lambda o: o.value if isinstance(o, Enum) else str(o),
    )


@router.post("/conversations/{conversation_id}/messages")
async def post_message(conversation_id: str, body: MessageIn, request: Request):
    registry = request.app.state.registry
    await registry.evict_idle()  # opportunistic sweep — no background task needed
    entry = registry.get_or_create(conversation_id)

    if entry.in_flight:
        # Mid-turn POST = steer. No 409, no second turn, no second stream.
        await entry.handle.commands.put(ev.SendMessage(text=body.text))
        return JSONResponse({"status": "steered"}, status_code=202)

    entry.in_flight = True
    # withhold-until-turn-complete (architecture §2): a judge can retry the
    # answer mid-turn, so streamed text is provisional — text deltas are
    # withheld until TurnComplete, whose envelope carries the accepted
    # answer. Without a judge, text streams live. All non-text events
    # stream either way.
    withhold_text = entry.bundle.judge is not None

    async def stream():
        try:
            await entry.handle.commands.put(ev.SendMessage(text=body.text))
            while True:
                event = await entry.handle.events.get()
                if withhold_text and isinstance(event, ev.TextDelta):
                    continue
                yield {"event": event_name(event), "data": event_data(event)}
                if isinstance(event, ev.TurnComplete):
                    # hooks ran before the event (engine contract), so the
                    # snapshot already holds this turn — the envelope reads
                    # the ACCEPTED answer from it, never a concat of drafts.
                    envelope = build_envelope(
                        conversation_id,
                        entry.bundle.store.load(conversation_id),
                        stop_reason=event.reason.value,
                        judge=entry.bundle.judge,
                        references=entry.bundle.references.take(),
                    )
                    yield {"event": "envelope", "data": json.dumps(envelope)}
                    entry.in_flight = False  # clean end: the slot frees here
                    return
        except (asyncio.CancelledError, GeneratorExit):
            # DISCONNECT != CANCEL (architecture §2): the client vanished but
            # the turn keeps running detached — cancelling here would erase the
            # user's message on every flaky network. A drainer adopts the event
            # stream so the queue never carries stale events into the next
            # pump, and in_flight stays True (a POST meanwhile still steers).
            _adopt_detached(entry)
            raise

    return EventSourceResponse(stream())


# Strong refs to detached drainers (a bare create_task is GC-bait).
_DETACHED: set[asyncio.Task] = set()


def _adopt_detached(entry) -> None:
    task = asyncio.create_task(_drain_detached(entry))
    _DETACHED.add(task)
    task.add_done_callback(_DETACHED.discard)


async def _drain_detached(entry) -> None:
    """Consume the abandoned turn's events until it completes (they are not
    lost — the snapshot hook persists the turn; a reconnecting client
    refetches via GET). Also exits if the session task dies (eviction), so a
    drainer can never leak."""
    while True:
        getter = asyncio.ensure_future(entry.handle.events.get())
        done, _ = await asyncio.wait(
            {getter, entry.handle.task}, return_when=asyncio.FIRST_COMPLETED
        )
        if getter in done:
            if isinstance(getter.result(), ev.TurnComplete):
                entry.bundle.references.take()  # discard: refs are per-turn
                entry.in_flight = False
                return
        else:
            getter.cancel()  # session gone (evicted/shutdown): nothing to drain
            entry.in_flight = False
            return


@router.post("/conversations/{conversation_id}/cancel")
async def cancel_turn(conversation_id: str, request: Request):
    """EXPLICIT cancel — the only way a turn is ever cancelled (a dropped
    connection never is). The kernel checkpoints the cancel and, with
    keep_interrupted_context, preserves the partial work."""
    entry = request.app.state.registry.get(conversation_id)
    if entry is None or not entry.in_flight:
        return JSONResponse({"status": "idle"})  # nothing running; not an error
    await entry.handle.commands.put(ev.Cancel())
    return JSONResponse({"status": "cancelling"}, status_code=202)


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, request: Request):
    """The refetch surface: the persisted conversation, straight from the
    snapshot store. This is what a client reads after a reconnect — it never
    spawns an agent and never touches a running turn."""
    snapshot = request.app.state.store.load(conversation_id)
    if snapshot is None:
        return JSONResponse({"detail": "unknown conversation"}, status_code=404)
    entry = request.app.state.registry.get(conversation_id)
    return {
        "conversation_id": conversation_id,
        "turn_counter": snapshot.turn_counter,
        "in_flight": bool(entry and entry.in_flight),
        "messages": [
            {"role": m.role.value, "text": m.text}
            for m in snapshot.messages
            if m.text and not m.synthetic
        ],
    }
