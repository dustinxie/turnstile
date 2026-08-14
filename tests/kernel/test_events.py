"""Driver-protocol tests — command/event DTOs, Outcome defaults, RequestCtx broker."""

import asyncio
import dataclasses

import pytest

from turnstile.kernel.dtos import StopReason
from turnstile.kernel.events import (
    Cancel,
    Outcome,
    Request,
    RequestCtx,
    SendMessage,
    SendMessageWithContext,
    TurnComplete,
)
from turnstile.kernel.ports import Requester

pytestmark = pytest.mark.unit


# ── protocol DTOs ──────────────────────────────────────────────────────


def test_commands_are_frozen_values():
    cmd = SendMessage(text="hi")
    assert cmd.images == []
    with pytest.raises(dataclasses.FrozenInstanceError):
        cmd.text = "rewritten"  # type: ignore[misc]
    assert Cancel() == Cancel()  # value semantics


def test_send_message_with_context_carries_both_parts():
    cmd = SendMessageWithContext(text="continue", context="hidden recovery")
    assert (cmd.text, cmd.context, cmd.images) == ("continue", "hidden recovery", [])


def test_events_are_asdict_serializable():
    ev = TurnComplete(reason=StopReason.STOPPED)
    assert dataclasses.asdict(ev) == {"reason": StopReason.STOPPED}


def test_outcome_default_is_clean_stop():
    out = Outcome()
    assert out.stop is StopReason.STOPPED and out.error is None
    # failure perception: a failed run carries both the stop and the cause
    failed = Outcome(stop=StopReason.PROVIDER_ERROR, error="HTTP 500", http_status=500)
    assert failed.error == "HTTP 500"


# ── RequestCtx broker ──────────────────────────────────────────────────


def _ctx(timeout: float | None = None) -> tuple[RequestCtx, list]:
    emitted: list = []
    return RequestCtx(emitted.append, request_timeout=timeout), emitted


async def test_request_emits_and_resolves_by_id():
    ctx, emitted = _ctx()

    async def driver():
        while not emitted:  # wait for the Request event to land
            await asyncio.sleep(0)
        req = emitted[0]
        assert isinstance(req, Request) and req.kind == "approval"
        ctx.resolve(req.id, {"decision": "allow"})

    driver_task = asyncio.create_task(driver())
    value = await ctx.request("approval", {"tool": "bash"})
    await driver_task
    assert value == {"decision": "allow"}


async def test_request_ids_are_monotonic():
    ctx, emitted = _ctx()

    async def answer_all():
        while len(emitted) < 2:
            await asyncio.sleep(0)
        for req in list(emitted):
            ctx.resolve(req.id, req.id)

    task = asyncio.create_task(answer_all())
    first, second = await asyncio.gather(ctx.request("a", {}), ctx.request("b", {}))
    await task
    assert second == first + 1


async def test_request_timeout_degrades_to_none_and_cleans_pending():
    ctx, emitted = _ctx(timeout=0.01)
    value = await ctx.request("approval", {})
    assert value is None
    late_id = emitted[0].id
    ctx.resolve(late_id, {"decision": "allow"})  # late Respond must no-op, not raise


async def test_cancel_pending_flushes_fail_closed():
    ctx, _ = _ctx()
    task = asyncio.create_task(ctx.request("approval", {}))
    while not ctx._pending:  # private peek: reach the parked state deterministically
        await asyncio.sleep(0)
    ctx.cancel_pending()
    assert await task is None  # parked round-trip unblocked with the degraded value


async def test_requester_is_a_request_only_port_handle():
    ctx, _emitted = _ctx(timeout=0.01)
    handle = ctx.requester()
    assert isinstance(handle, Requester)
    assert await handle.request("ask", {"q": 1}) is None  # degrades identically
    assert not hasattr(handle, "resolve") and not hasattr(handle, "cancel_pending")
