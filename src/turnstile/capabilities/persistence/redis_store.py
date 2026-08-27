"""Redis session store — the M7 store: conversations outlive the process.

Same surface as MemorySessionStore (architecture §4: the driver and root see
one store shape; the M2 stand-in swaps out with zero change above L1):
  - save/load/drop by session id (the driver's resume path),
  - claim/owner/owned_by — conversation OWNERSHIP, same lifetime as the snapshot,
  - checkpoint(session_id) / hook(session_id) — the kernel seams.

Layout (one prefix, so several deployments may share a Redis):
  {prefix}:snap:{session_id}   -> snapshot JSON (snapshot_codec), keep-latest
  {prefix}:owner:{session_id}  -> principal
  {prefix}:owned:{principal}   -> SET of session ids (the listing surface)
Snapshot and owner carry the SAME TTL — the resume window (§4): a session
that ages out is gone as a unit; 0 = no expiry. The owned-set entry for an
expired session is dropped lazily when the listing finds no snapshot.

The client is the SYNC redis client on purpose: every call site (routes,
registry, root) is a sub-millisecond local round trip and the store's
surface is synchronous today; going async is a surface change for later.
Durability across Redis restarts is Redis's job (RDB/AOF, see deploy/).
"""

import redis

from turnstile.capabilities.persistence import snapshot_codec
from turnstile.kernel.dtos import Conversation, SessionSnapshot, StopReason, TurnCtx
from turnstile.kernel.ports import CompactionCheckpoint, CompactionCheckpointError, LifecycleHooks


class RedisSessionStore:
    """Session-keyed keep-latest snapshot store over Redis."""

    def __init__(self, url: str, ttl_seconds: int = 0, prefix: str = "turnstile") -> None:
        """`url`: redis://[:password@]host:port/db. `ttl_seconds`: the resume
        window for snapshot + ownership (0 = never expire). `prefix`: key
        namespace, so one Redis can host several deployments."""
        self._r = redis.Redis.from_url(url, decode_responses=True)
        self._ttl = ttl_seconds
        self._prefix = prefix

    # ── keys ──
    def _snap(self, sid: str) -> str:
        return f"{self._prefix}:snap:{sid}"

    def _owner(self, sid: str) -> str:
        return f"{self._prefix}:owner:{sid}"

    def _owned(self, principal: str) -> str:
        return f"{self._prefix}:owned:{principal}"

    def _get(self, key: str) -> str | None:
        # decode_responses=True makes these str at runtime; the client's typing
        # still says bytes | str, so coerce once here
        value = self._r.get(key)
        return None if value is None else str(value)

    def _set(self, key: str, value: str) -> None:
        if self._ttl > 0:
            self._r.set(key, value, ex=self._ttl)
        else:
            self._r.set(key, value)

    # ── snapshots ──
    def save(self, session_id: str, snapshot: SessionSnapshot) -> None:
        self._set(self._snap(session_id), snapshot_codec.dumps(snapshot))
        if self._ttl > 0:  # the owner ages with its newest snapshot
            self._r.expire(self._owner(session_id), self._ttl)

    def load(self, session_id: str) -> SessionSnapshot | None:
        """The resume path: the latest snapshot, or None for a new/expired
        session (the caller starts fresh — never reconstructs from elsewhere)."""
        raw = self._r.get(self._snap(session_id))
        return snapshot_codec.loads(raw) if raw is not None else None

    def drop(self, session_id: str) -> None:
        """Forget a session (idle eviction). Missing id is a no-op."""
        owner = self._get(self._owner(session_id))
        pipe = self._r.pipeline()
        pipe.delete(self._snap(session_id), self._owner(session_id))
        if owner is not None:
            pipe.srem(self._owned(owner), session_id)
        pipe.execute()

    # ── ownership ──
    def claim(self, session_id: str, owner: str) -> str:
        """First claimant wins; returns the session's owner AFTER the call —
        callers compare it to their principal (mismatch = not yours)."""
        key = self._owner(session_id)
        if self._r.set(key, owner, nx=True, ex=self._ttl if self._ttl > 0 else None):
            self._r.sadd(self._owned(owner), session_id)
            return owner
        return self._get(key) or owner  # raced with an expiry: treat as ours

    def owner(self, session_id: str) -> str | None:
        return self._get(self._owner(session_id))

    def owned_by(self, owner: str) -> list[str]:
        """The principal's sessions, oldest claim first. Sessions whose keys
        have aged out are pruned from the set as they are met."""
        members = self._r.smembers(self._owned(owner))
        sids = sorted(str(m) for m in members)  # sets are unordered
        alive: list[str] = []
        for sid in sids:
            if self._r.exists(self._owner(sid)):
                alive.append(sid)
            else:
                self._r.srem(self._owned(owner), sid)
        return alive

    # ── kernel seams ──
    def checkpoint(self, session_id: str) -> CompactionCheckpoint:
        return _BoundCheckpoint(self, session_id)

    def hook(self, session_id: str) -> LifecycleHooks:
        return _SnapshotHook(self, session_id)

    def ping(self) -> bool:
        """Boot-time reachability check (root fails loudly, not on first turn)."""
        return bool(self._r.ping())


class _BoundCheckpoint(CompactionCheckpoint):
    def __init__(self, store: RedisSessionStore, session_id: str) -> None:
        self._store, self._session_id = store, session_id

    def save(self, snapshot: SessionSnapshot) -> None:
        try:
            self._store.save(self._session_id, snapshot)
        except redis.RedisError as e:  # the gate: a compaction is committed only if saved
            raise CompactionCheckpointError(str(e)) from e


class _SnapshotHook(LifecycleHooks):
    def __init__(self, store: RedisSessionStore, session_id: str) -> None:
        self._store, self._session_id = store, session_id

    async def turn_complete(self, convo: Conversation, reason: StopReason, ctx: TurnCtx) -> None:
        self._store.save(self._session_id, SessionSnapshot.from_conversation(convo))
