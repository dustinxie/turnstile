"""In-memory session store — the M2 stand-in for the M7 Redis checkpoint.

Keep-latest snapshot per session, in a plain dict (architecture §4: each
snapshot is cumulative, so latest-only suffices for resume; keep-latest =
overwrite). Single-process, single-event-loop by design (§2: one uvicorn
worker) and saves are awaited inside the turn, so there is no concurrent
access to lock against.

Three faces, one store:
  - the store itself: save/load/drop by session id (the driver's resume path),
  - checkpoint(session_id): a session-bound CompactionCheckpoint for
    Agent.checkpoint (gates committed manual compactions),
  - hook(session_id): a turn_complete LifecycleHooks that snapshots the
    conversation after every turn — the per-turn persistence seam.

M7 swaps this class for the Redis implementation behind the same surface;
nothing above L1 changes.
"""

from turnstile.kernel.dtos import Conversation, SessionSnapshot, StopReason, TurnCtx
from turnstile.kernel.ports import CompactionCheckpoint, LifecycleHooks


class MemorySessionStore:
    """Session-keyed keep-latest snapshot store."""

    def __init__(self) -> None:
        self._latest: dict[str, SessionSnapshot] = {}
        # Conversation ownership rides the session store because it has
        # exactly the snapshot's lifetime: it must survive registry eviction
        # (the resumed conversation is still the same principal's) and die
        # with the session. M7's Redis store carries it the same way.
        self._owners: dict[str, str] = {}

    def save(self, session_id: str, snapshot: SessionSnapshot) -> None:
        self._latest[session_id] = snapshot

    def claim(self, session_id: str, owner: str) -> str:
        """First claimant wins; returns the session's owner AFTER the call —
        callers compare it to their principal (mismatch = not yours)."""
        return self._owners.setdefault(session_id, owner)

    def owner(self, session_id: str) -> str | None:
        return self._owners.get(session_id)

    def owned_by(self, owner: str) -> list[str]:
        """The principal's sessions, oldest claim first (dict order). The
        listing surface reads this — foreign ids never leave the store."""
        return [sid for sid, who in self._owners.items() if who == owner]

    def load(self, session_id: str) -> SessionSnapshot | None:
        """The resume path: the latest snapshot, or None for a new/expired
        session (the caller starts fresh — never reconstructs from elsewhere)."""
        return self._latest.get(session_id)

    def drop(self, session_id: str) -> None:
        """Forget a session (idle eviction). Missing id is a no-op."""
        self._latest.pop(session_id, None)
        self._owners.pop(session_id, None)

    def checkpoint(self, session_id: str) -> CompactionCheckpoint:
        """A session-bound CompactionCheckpoint to pass as Agent.checkpoint."""
        return _BoundCheckpoint(self, session_id)

    def hook(self, session_id: str) -> LifecycleHooks:
        """A session-bound turn_complete hook: saves a snapshot on every
        terminal path, so a crash between turns loses at most the live turn."""
        return _SnapshotHook(self, session_id)


class _BoundCheckpoint(CompactionCheckpoint):
    """CompactionCheckpoint over one session's slot. An in-memory dict write
    cannot fail, so save never raises CompactionCheckpointError here."""

    def __init__(self, store: MemorySessionStore, session_id: str) -> None:
        self._store = store
        self._session_id = session_id

    def save(self, snapshot: SessionSnapshot) -> None:
        self._store.save(self._session_id, snapshot)


class _SnapshotHook(LifecycleHooks):
    """The per-turn persistence seam (ports.py: turn_complete is exactly this).
    Counters come from SessionSnapshot.from_conversation's meta derivation —
    the documented fallback; good enough until the M7 store, where the capturer
    can stamp live counters for turns that died before storing a response."""

    def __init__(self, store: MemorySessionStore, session_id: str) -> None:
        self._store = store
        self._session_id = session_id

    async def turn_complete(self, convo: Conversation, reason: StopReason, ctx: TurnCtx) -> None:
        self._store.save(self._session_id, SessionSnapshot.from_conversation(convo))
