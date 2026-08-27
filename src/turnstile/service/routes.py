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
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from turnstile.kernel import events as ev
from turnstile.kernel.dtos import Role
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
                    _record_turn_meta(entry, conversation_id, envelope)
                    entry.in_flight = False  # clean end: the slot frees here
                    return
        except (asyncio.CancelledError, GeneratorExit):
            # DISCONNECT != CANCEL (architecture §2): the client vanished but
            # the turn keeps running detached — cancelling here would erase the
            # user's message on every flaky network. A drainer adopts the event
            # stream so the queue never carries stale events into the next
            # pump, and in_flight stays True (a POST meanwhile still steers).
            _adopt_detached(entry, conversation_id)
            raise

    return EventSourceResponse(stream())


def _record_turn_meta(entry, conversation_id: str, envelope: dict) -> None:
    """Persist the turn's L2 verdict + references beside the snapshot (never
    inside it — design doc: collector data stays out of kernel DTOs) so a
    reloaded conversation shows the badge and the References again. Keyed by
    the turn the snapshot just recorded. Link URLs are NOT stored: a file
    token expires long before the conversation does; GET re-mints them."""
    snapshot = entry.bundle.store.load(conversation_id)
    if snapshot is None or envelope["stop_reason"] != "stopped":
        return
    references = [
        {**r, "url": r["url"] if r.get("tool") != "kb_search" else None}
        for r in envelope["references"]
    ]
    entry.bundle.store.save_turn_meta(
        conversation_id,
        snapshot.turn_counter,
        {"signal": envelope["signal"], "score": envelope["score"], "references": references},
    )


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


def _adopt_detached(entry, conversation_id: str) -> None:
    task = asyncio.create_task(_drain_detached(entry, conversation_id))
    _DETACHED.add(task)
    task.add_done_callback(_DETACHED.discard)


async def _drain_detached(entry, conversation_id: str) -> None:
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
                # nobody is watching, but the turn still happened: record its
                # verdict + references for the reconnecting client's refetch
                # (links re-minted on read, so no principal is needed here)
                envelope = build_envelope(
                    conversation_id,
                    entry.bundle.store.load(conversation_id),
                    stop_reason=getter.result().reason.value,
                    judge=entry.bundle.judge,
                    references=entry.bundle.references,
                )
                _record_turn_meta(entry, conversation_id, envelope)
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


_TITLE_CHARS = 60


@router.get("/conversations")
async def list_conversations(request: Request, principal: str = Depends(require_user)):
    """The history panel's surface: the principal's conversations, newest
    claim first. Ownership is the ONLY filter — foreign ids are simply not
    in the list (the same no-existence-leak rule as the 404s). `title` is a
    display convenience: the first user message, truncated; a conversation
    whose first turn is still running has no snapshot yet and lists as
    untitled with in_flight=true."""
    store = request.app.state.store
    registry = request.app.state.registry
    items = []
    for conversation_id in reversed(store.owned_by(principal)):
        snapshot = store.load(conversation_id)
        entry = registry.get(conversation_id)
        first_user = next(
            (m.text for m in (snapshot.messages if snapshot else []) if m.role is Role.USER),
            "",
        )
        items.append(
            {
                "conversation_id": conversation_id,
                "title": first_user[:_TITLE_CHARS],
                "turn_counter": snapshot.turn_counter if snapshot else 0,
                "in_flight": bool(entry and entry.in_flight),
            }
        )
    return {"conversations": items}


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
    turn_meta = request.app.state.store.load_turn_meta(conversation_id)
    link = _file_link(request, principal)
    messages = []
    for i, m in enumerate(snapshot.messages):
        if not m.text or m.synthetic:
            continue
        row: dict = {"role": m.role.value, "text": m.text}
        turn = m.meta.turn_id if m.meta is not None else 0
        # the turn's FINAL assistant message carries the turn's verdict +
        # references (a judge-retried turn has earlier drafts; those get none)
        if m.role is Role.ASSISTANT and turn in turn_meta and _is_last_of_turn(snapshot, i, turn):
            meta = turn_meta[turn]
            row["signal"], row["score"] = meta["signal"], meta["score"]
            row["references"] = [_relink(r, link) for r in meta["references"]]
        messages.append(row)
    return {
        "conversation_id": conversation_id,
        "turn_counter": snapshot.turn_counter,
        "in_flight": bool(entry and entry.in_flight),
        "messages": messages,
    }


def _is_last_of_turn(snapshot, index: int, turn: int) -> bool:
    for later in snapshot.messages[index + 1 :]:
        if later.role is Role.ASSISTANT and later.meta is not None and later.meta.turn_id == turn:
            return False
    return True


def _relink(reference: dict, link) -> dict:
    """A stored reference with a fresh link for THIS reader: kb documents get a
    file token minted now (bound to the requesting principal); web sources
    keep their URL."""
    if reference.get("tool") == "kb_search" and reference.get("region") and reference.get("path"):
        source = SimpleNamespace(
            url=None,
            region=reference["region"],
            ref=reference["path"],
            fragment=reference.get("fragment"),
        )
        return {**reference, "url": link(source)}
    return reference
