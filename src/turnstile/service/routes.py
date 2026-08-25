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
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from turnstile.kernel import events as ev
from turnstile.service.auth import require_user
from turnstile.service.envelope import build_envelope
from turnstile.service.files import mint_file_token

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


def _guard_owner(request: Request, conversation_id: str, principal: str) -> None:
    """404 on someone else's conversation — never 403, which would leak that
    the id exists. Unclaimed ids pass (the caller may become the owner)."""
    owner = request.app.state.store.owner(conversation_id)
    if owner is not None and owner != principal:
        raise HTTPException(status_code=404, detail="unknown conversation")


@router.post("/conversations/{conversation_id}/messages")
async def post_message(
    conversation_id: str,
    body: MessageIn,
    request: Request,
    principal: str = Depends(require_user),
):
    _guard_owner(request, conversation_id, principal)
    registry = request.app.state.registry
    await registry.evict_idle()  # opportunistic sweep — no background task needed
    # First toucher claims the conversation; the guard above already 404'd
    # a non-owner, so a mismatch here can only be a concurrent first claim.
    if request.app.state.store.claim(conversation_id, principal) != principal:
        raise HTTPException(status_code=404, detail="unknown conversation")
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
                        references=entry.bundle.references,
                        link=_file_link(request, principal),
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


def _file_link(request: Request, principal: str):
    """The driver's half of citation links: a Reference -> openable URL. Web
    sources carry their own URL; kb documents get a recipient-bound file
    token (files.py) when a file store is configured, else no link — the
    References section then lists them as plain titles."""
    cfg: Any = request.app.state.cfg
    file_root = getattr(cfg, "file_root", None)
    secret = getattr(cfg, "jwt_secret", None)

    def link(reference: Any) -> str | None:
        if reference.url:
            return reference.url
        if not file_root or not reference.region or not reference.ref:
            return None
        token = mint_file_token(secret, reference.region, reference.ref, principal)
        anchor = f"#{reference.fragment}" if reference.fragment else ""
        return f"/v1/files/{token}{anchor}"

    return link


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
async def cancel_turn(
    conversation_id: str, request: Request, principal: str = Depends(require_user)
):
    """EXPLICIT cancel — the only way a turn is ever cancelled (a dropped
    connection never is). The kernel checkpoints the cancel and, with
    keep_interrupted_context, preserves the partial work."""
    _guard_owner(request, conversation_id, principal)
    entry = request.app.state.registry.get(conversation_id)
    if entry is None or not entry.in_flight:
        return JSONResponse({"status": "idle"})  # nothing running; not an error
    await entry.handle.commands.put(ev.Cancel())
    return JSONResponse({"status": "cancelling"}, status_code=202)


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str, request: Request, principal: str = Depends(require_user)
):
    """The refetch surface: the persisted conversation, straight from the
    snapshot store. This is what a client reads after a reconnect — it never
    spawns an agent and never touches a running turn."""
    _guard_owner(request, conversation_id, principal)
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
