"""Live ds4 smoke — one real streaming round-trip through the provider.

Marked `integration`: the commit gate (`make check` -> `make test`,
`-m "not integration"`) never collects it; run it deliberately on a box
with a route to ds4 via `make test-all` (or `pytest -m integration -v`).

Target: the tokenless vLLM route (`/model-long/v1`, alias `model-fast`) —
`/model/v1` keeps a bearer gate; see tests/capabilities/fixtures/README.md.
"""

import os

import pytest

from turnstile.capabilities.providers.openai_compat import OpenAICompatProvider
from turnstile.kernel.dtos import (
    ChatOptions,
    Done,
    Message,
    ResponseId,
    ResponseModel,
    TextDelta,
    UsageEvent,
)

pytestmark = pytest.mark.integration

DS4_BASE_URL = os.environ.get("DS4_BASE_URL", "https://10.83.135.205/model-long/v1")


async def test_chat_stream_round_trip_against_ds4():
    provider = OpenAICompatProvider(
        base_url=DS4_BASE_URL,
        model="model-fast",
        context_window=128_000,
        request_timeout=60.0,
    )
    provider.bind_session_id("turnstile-live-smoke")
    events = []
    async for event in provider.chat_stream(
        [
            Message.system("You are a helpful assistant."),
            Message.user("Reply with exactly: smoke ok"),
        ],
        [],
        ChatOptions(max_tokens=20, temperature=0.0),
    ):
        events.append(event)

    # the full event mapping, against live vLLM bytes
    assert isinstance(events[-1], Done) and not events[-1].truncated
    assert sum(isinstance(e, ResponseId) for e in events) == 1
    models = [e.model for e in events if isinstance(e, ResponseModel)]
    assert models == ["model-fast"]
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "smoke ok" in text.lower()
    usage = next(e.usage for e in events if isinstance(e, UsageEvent))
    assert usage.prompt > 0 and usage.completion > 0
