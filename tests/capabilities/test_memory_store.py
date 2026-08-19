"""In-memory session store — store semantics plus real-engine integration:
the turn_complete hook persists, the snapshot resumes a second agent."""

import pytest

from turnstile.capabilities.persistence.memory_store import MemorySessionStore
from turnstile.kernel.dtos import (
    SNAPSHOT_VERSION,
    Done,
    Message,
    Role,
    SessionSnapshot,
    TextDelta,
)
from turnstile.kernel.ports import CompactionCheckpoint
from turnstile.kernel.testkit import ScriptedProvider

pytestmark = pytest.mark.unit


def _snap(*texts: str) -> SessionSnapshot:
    return SessionSnapshot(version=SNAPSHOT_VERSION, messages=[Message.user(t) for t in texts])


# ── store semantics ────────────────────────────────────────────────────


def test_save_load_round_trip_and_unknown_session_is_none():
    store = MemorySessionStore()
    snapshot = _snap("q1")
    store.save("s1", snapshot)
    assert store.load("s1") is snapshot
    assert store.load("never-seen") is None


def test_keep_latest_overwrites():
    store = MemorySessionStore()
    store.save("s1", _snap("q1"))
    latest = _snap("q1", "q2")
    store.save("s1", latest)
    assert store.load("s1") is latest  # exactly one snapshot per session


def test_sessions_are_isolated_and_drop_forgets():
    store = MemorySessionStore()
    store.save("s1", _snap("mine"))
    store.save("s2", _snap("theirs"))
    store.drop("s1")
    store.drop("s1")  # idempotent
    assert store.load("s1") is None
    assert store.load("s2") is not None  # untouched


def test_bound_checkpoint_writes_the_sessions_slot():
    store = MemorySessionStore()
    checkpoint = store.checkpoint("s1")
    assert isinstance(checkpoint, CompactionCheckpoint)  # Agent.checkpoint-ready
    checkpoint.save(_snap("compacted"))
    loaded = store.load("s1")
    assert loaded is not None and loaded.messages[0].text == "compacted"


# ── engine integration: persist every turn, resume from the store ─────


async def test_hook_saves_a_snapshot_per_turn_and_resume_is_monotonic():
    from turnstile.kernel.engine import Agent

    store = MemorySessionStore()
    provider = ScriptedProvider(rounds=[[TextDelta("a1"), Done()], [TextDelta("a2"), Done()]])
    agent = Agent(provider=provider, hooks=[store.hook("s1")], session_id="s1")
    await agent.run_to_completion("q1")

    first = store.load("s1")
    assert first is not None and first.version == SNAPSHOT_VERSION
    assert [(m.role, m.text) for m in first.messages] == [
        (Role.USER, "q1"),
        (Role.ASSISTANT, "a1"),
    ]
    assert first.turn_counter == 1

    # a NEW agent resumes from the stored snapshot (nothing shared in memory)
    resumed = Agent(provider=provider, hooks=[store.hook("s1")], resume=first)
    await resumed.run_to_completion("q2")

    second = store.load("s1")  # keep-latest: overwritten by turn 2's snapshot
    assert second is not None
    assert [m.text for m in second.messages] == ["q1", "a1", "q2", "a2"]
    assert second.turn_counter == 2  # monotonic across the resume boundary
    assert second.request_counter > first.request_counter
