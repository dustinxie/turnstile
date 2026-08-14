"""Engine mock suite — resilience tiers (commit 9 scope): transient retries,
the 429 path, stream timeouts, empty-200 retries, truncation continuation,
partial-stream persistence. All backoffs run with backoff_scale=0."""

import pytest

from turnstile.kernel import events as ev
from turnstile.kernel.dtos import (
    Done,
    ErrorEvent,
    ProviderError,
    RateLimitDecision,
    RateLimitHint,
    StopReason,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)
from turnstile.kernel.engine import (
    EMPTY_RESPONSE_MAX_RETRIES,
    MAX_PROVIDER_RETRIES,
    MAX_RATE_LIMIT_WAITS,
    MAX_STREAM_RETRIES,
    TRUNCATION_RESUME_NUDGE,
    Agent,
)
from turnstile.kernel.testkit import (
    EchoTool,
    FnHook,
    ScriptedProvider,
    SilentProvider,
    StallThenProvider,
    StepClock,
)

pytestmark = pytest.mark.unit


async def _collect(agent: Agent, text: str = "q") -> list:
    handle = agent.spawn()
    await handle.commands.put(ev.SendMessage(text=text))
    events = []
    while True:
        event = await handle.events.get()
        events.append(event)
        if isinstance(event, ev.TurnComplete):
            break
    await handle.commands.put(ev.Shutdown())
    await handle.task
    return events


def _agent(provider, **kwargs) -> Agent:
    return Agent(provider=provider, backoff_scale=0.0, clock=StepClock(), **kwargs)


def _retryable(msg="upstream 502"):
    return ProviderError(message=msg, retryable=True, http_status=502)


def _rate_limited(msg="rate limited", retry_after=None, code=None):
    return ProviderError(
        message=msg, retryable=True, http_status=429, code=code, retry_after_secs=retry_after
    )


# ── transient open retries ─────────────────────────────────────────────


async def test_retryable_open_errors_retry_then_succeed():
    provider = ScriptedProvider(
        rounds=[
            _retryable(),
            _retryable(),
            [TextDelta("recovered"), Done()],
        ]
    )
    events = await _collect(_agent(provider))
    assert events[-1].reason is StopReason.STOPPED
    assert len(provider.calls) == 3
    retry_notices = [e for e in events if isinstance(e, ev.Warning) and "retrying" in e.message]
    assert len(retry_notices) == 2


async def test_retry_budget_exhaustion_fails_the_turn():
    provider = ScriptedProvider(rounds=[_retryable()] * 6)
    events = await _collect(_agent(provider))
    assert events[-1].reason is StopReason.PROVIDER_ERROR
    assert len(provider.calls) == MAX_PROVIDER_RETRIES + 1  # initial + retries


async def test_non_retryable_open_fails_fast():
    provider = ScriptedProvider(
        rounds=[
            ProviderError(message="bad key", retryable=False, http_status=401),
        ]
    )
    events = await _collect(_agent(provider))
    assert events[-1].reason is StopReason.PROVIDER_ERROR
    assert len(provider.calls) == 1  # no retry on auth errors


async def test_mid_stream_error_after_content_persists_partial():
    provider = ScriptedProvider(
        rounds=[
            [
                TextDelta("partial answer"),
                ToolCallEvent(ToolCall("c1", "echo", "{}")),
                ErrorEvent(ProviderError(message="died", http_status=502)),
            ],
        ]
    )
    agent = _agent(provider, tools={"echo": EchoTool()})
    handle = agent.spawn()
    await handle.commands.put(ev.SendMessage(text="q"))
    while not isinstance(await handle.events.get(), ev.TurnComplete):
        pass
    await handle.commands.put(ev.RequestSnapshot())
    snap = None
    while snap is None:
        event = await handle.events.get()
        if isinstance(event, ev.Snapshot):
            snap = event.snapshot
    await handle.commands.put(ev.Shutdown())
    await handle.task
    texts = [(m.role.value, m.text) for m in snap.messages]
    assert ("assistant", "partial answer") in texts  # partial preserved
    # the dangling call got an interrupted pairing, never executed
    tool_rows = [m for m in snap.messages if m.tool_call_id == "c1"]
    assert len(tool_rows) == 1
    assert "interrupted before execution" in tool_rows[0].text and tool_rows[0].is_error


# ── the 429 path ───────────────────────────────────────────────────────


async def test_hook_verdict_wait_retries_with_banner():
    provider = ScriptedProvider(
        rounds=[
            _rate_limited(),
            [TextDelta("after wait"), Done()],
        ]
    )
    hook = FnHook(on_rate_limit=lambda hint: RateLimitDecision(wait_secs=7))
    events = await _collect(_agent(provider, hooks=[hook]))
    assert events[-1].reason is StopReason.STOPPED
    banners = [e for e in events if isinstance(e, ev.RateLimited)]
    assert len(banners) == 1
    assert banners[0].auto_resuming and banners[0].secs_until_reset == 7


async def test_anonymous_first_429_retries_quietly():
    provider = ScriptedProvider(
        rounds=[
            _rate_limited(),  # no verdict, no Retry-After -> silent first retry
            _rate_limited(),  # second incident surfaces normally
            [TextDelta("ok"), Done()],
        ]
    )
    events = await _collect(_agent(provider))
    assert events[-1].reason is StopReason.STOPPED
    banners = [e for e in events if isinstance(e, ev.RateLimited)]
    assert len(banners) == 1  # first wait was silent, only the second showed
    assert banners[0].auto_resuming


async def test_terminal_billing_429_pauses_without_retry():
    provider = ScriptedProvider(
        rounds=[
            _rate_limited(msg="insufficient balance, please recharge", code="insufficient_quota"),
        ]
    )
    events = await _collect(_agent(provider))
    assert events[-1].reason is StopReason.RATE_LIMITED
    assert len(provider.calls) == 1
    banner = next(e for e in events if isinstance(e, ev.RateLimited))
    assert not banner.auto_resuming
    assert banner.server_message is not None
    assert "insufficient balance" in banner.server_message


async def test_far_reset_pauses_instead_of_waiting():
    provider = ScriptedProvider(rounds=[_rate_limited(retry_after=600)])
    events = await _collect(_agent(provider))
    assert events[-1].reason is StopReason.RATE_LIMITED  # 600s > auto-wait window
    assert len(provider.calls) == 1


async def test_rate_limit_waits_fuse_forces_pause():
    provider = ScriptedProvider(rounds=[_rate_limited()] * 10)
    hook = FnHook(on_rate_limit=lambda hint: RateLimitDecision(wait_secs=0))
    events = await _collect(_agent(provider, hooks=[hook]))
    assert events[-1].reason is StopReason.RATE_LIMITED
    assert len(provider.calls) == MAX_RATE_LIMIT_WAITS + 1  # fuse after 5 waits


async def test_429_after_content_preserves_and_pauses():
    provider = ScriptedProvider(
        rounds=[
            [TextDelta("already streamed"), ErrorEvent(_rate_limited())],
        ]
    )
    events = await _collect(_agent(provider))
    assert events[-1].reason is StopReason.RATE_LIMITED
    assert len(provider.calls) == 1  # no replay once content reached the driver


# ── stream liveness ────────────────────────────────────────────────────


async def test_idle_timeout_reconnects_then_succeeds():
    provider = StallThenProvider(1, [TextDelta("late but fine"), Done()])
    events = await _collect(_agent(provider, stream_timeout=0.02))
    assert events[-1].reason is StopReason.STOPPED
    assert provider.call_count == 2
    assert any(isinstance(e, ev.Warning) and "reconnecting" in e.message for e in events)


async def test_idle_timeout_budget_exhaustion_times_out():
    provider = StallThenProvider(99, [TextDelta("never"), Done()])
    events = await _collect(_agent(provider, stream_timeout=0.02))
    assert events[-1].reason is StopReason.TIMEOUT
    assert provider.call_count == MAX_STREAM_RETRIES + 1


async def test_timeout_after_content_preserves_partial_no_replay():
    provider = SilentProvider(prefix=[TextDelta("half an answer")])
    events = await _collect(_agent(provider, stream_timeout=0.02))
    assert events[-1].reason is StopReason.TIMEOUT
    assert any(
        isinstance(e, ev.Error) and "partial response preserved" in e.message for e in events
    )


# ── empty-200 retries ──────────────────────────────────────────────────


async def test_empty_200_retries_then_recovers():
    provider = ScriptedProvider(rounds=[[Done()], [Done()], [TextDelta("late"), Done()]])
    events = await _collect(_agent(provider))
    assert events[-1].reason is StopReason.STOPPED
    assert len(provider.calls) == 3
    notices = [e for e in events if isinstance(e, ev.Warning) and "empty response" in e.message]
    assert len(notices) == 2


async def test_empty_200_exhaustion_is_a_provider_error_not_a_stop():
    provider = ScriptedProvider(rounds=[[Done()]] * 10)
    events = await _collect(_agent(provider))
    assert events[-1].reason is StopReason.PROVIDER_ERROR  # never a silent STOPPED
    assert len(provider.calls) == EMPTY_RESPONSE_MAX_RETRIES + 1


async def test_hook_cleared_text_is_not_mistaken_for_empty():
    provider = ScriptedProvider(rounds=[[TextDelta("secret stuff"), Done()]])
    censor = FnHook(on_text_delta=lambda d: "")  # clears EVERY chunk
    events = await _collect(_agent(provider, hooks=[censor]))
    assert events[-1].reason is StopReason.STOPPED
    assert len(provider.calls) == 1  # provider DID produce content: no retry


# ── truncation continuation ────────────────────────────────────────────


async def test_truncation_nudges_resume_then_finishes():
    provider = ScriptedProvider(
        rounds=[
            [TextDelta("first half"), Done(truncated=True)],
            [TextDelta(" second half"), Done()],
        ]
    )
    events = await _collect(_agent(provider))
    assert events[-1].reason is StopReason.STOPPED
    assert len(provider.calls) == 2
    assert any(
        m.text == TRUNCATION_RESUME_NUDGE for m in provider.calls[1].messages
    )  # nudge reached the model
    assert not any(
        isinstance(e, ev.Warning) and "truncated" in e.message for e in events
    )  # recovered: no alarming banner


async def test_truncation_budget_exhausts_with_visible_warning():
    provider = ScriptedProvider(
        rounds=[
            [TextDelta("a"), Done(truncated=True)],
            [TextDelta("b"), Done(truncated=True)],
            [TextDelta("c"), Done(truncated=True)],
        ]
    )
    events = await _collect(_agent(provider))
    assert events[-1].reason is StopReason.STOPPED
    assert len(provider.calls) == 3  # initial + 2 continuations
    assert any(isinstance(e, ev.Warning) and "response truncated" in e.message for e in events)


# ── from_hint fallback (dtos) ──────────────────────────────────────────


def _hint(retry_after=None, terminal=False, attempt=1):
    return RateLimitHint(
        http_status=429, retry_after_secs=retry_after, terminal=terminal, attempt=attempt
    )


def test_from_hint_waits_when_reset_imminent():
    assert RateLimitDecision.from_hint(_hint(retry_after=45)).wait_secs == 45


def test_from_hint_pauses_when_reset_far_or_terminal():
    assert RateLimitDecision.from_hint(_hint(retry_after=600)).wait_secs is None
    assert RateLimitDecision.from_hint(_hint(terminal=True)).wait_secs is None


def test_from_hint_backoff_ladder_and_jitter_bounds():
    for attempt, secs in [(1, 3), (2, 6), (3, 12), (4, 24), (5, 48), (6, 48)]:
        assert RateLimitDecision.from_hint(_hint(attempt=attempt), 0.5).wait_secs == secs
    assert RateLimitDecision.from_hint(_hint(attempt=3), 0.0).wait_secs == 9
    assert RateLimitDecision.from_hint(_hint(attempt=3), 1.0).wait_secs == 15


# ── retry ownership sideband ───────────────────────────────────────────


async def test_kernel_stamps_rate_limit_retry_ownership():
    provider = ScriptedProvider(rounds=[[TextDelta("x"), Done()]])
    await _collect(_agent(provider))
    assert provider.calls[0].options.rate_limit_retry_owner == "kernel"
