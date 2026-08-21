"""The turn endpoint — POST a message, receive the turn as Server-Sent Events.

The SSE stream IS the driver protocol serialized verbatim (architecture.md
§2): one SSE event per AgentEvent — `event:` carries the snake_cased event
class, `data:` its fields as JSON — ending with the terminal turn_complete.
(The service-added envelope event lands in M4-c5; buffer-until-judged rides
with it.)

A POST while this conversation's turn is still streaming STEERS it
(kernel-native: the prompt folds in at the next round boundary) and answers
202 immediately — the folded content arrives on the already-open stream; the
events queue is single-consumer, so there is never a second pump.
"""

import json
import re
from dataclasses import asdict, is_dataclass
from enum import Enum

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from turnstile.kernel import events as ev

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

    async def stream():
        try:
            await entry.handle.commands.put(ev.SendMessage(text=body.text))
            while True:
                event = await entry.handle.events.get()
                yield {"event": event_name(event), "data": event_data(event)}
                if isinstance(event, ev.TurnComplete):
                    break
        finally:
            entry.in_flight = False

    return EventSourceResponse(stream())
