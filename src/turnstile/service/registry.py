"""Worker-local conversation registry — HTTP conversation id ↔ running agent.

One entry per conversation: the AssembledAgent bundle plus its spawned
handle. Created on the conversation's first request (resume comes free —
assemble() reads the shared store); reused for every later request, so a
mid-turn POST steers the running turn instead of starting a second one
(kernel-native, no 409). Evicted after `session_ttl_seconds` idle: the
snapshot hook already persisted every turn, so eviction is just a graceful
Shutdown — the next request for that id resumes from the store.

Worker-local by design (architecture.md §2): one uvicorn worker, no shared
mutable state across processes; scaling later is conversation-affinity
routing, not a distributed registry. Single event loop + no awaits between
the lookup and the insert in get_or_create = no lock needed.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from turnstile import root
from turnstile.kernel import events as ev
from turnstile.kernel.engine import AgentHandle
from turnstile.root import AssembledAgent

# Bound on a graceful Shutdown before the session task is cancelled outright:
# eviction must never hang the sweep behind one stuck turn.
_SHUTDOWN_GRACE_SECONDS = 5.0


@dataclass
class _Entry:
    bundle: AssembledAgent
    handle: AgentHandle
    last_used: float = field(default=0.0)


class ConversationRegistry:
    """id -> (bundle, handle), with idle-TTL eviction and graceful shutdown."""

    def __init__(
        self,
        cfg: Any,
        store: Any,
        assemble: Callable[..., AssembledAgent] = root.assemble,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        """`cfg` is root's config, held opaquely. `assemble`/`now` are
        injectable for tests (a scripted bundle factory / a fake clock)."""
        self._cfg = cfg
        self._store = store
        self._assemble = assemble
        self._now = now
        # 0 = no expiry (config contract).
        self._ttl = float(getattr(cfg, "session_ttl_seconds", 0) or 0)
        self._entries: dict[str, _Entry] = {}

    def get_or_create(self, conversation_id: str) -> _Entry:
        """The conversation's live entry, spawning (or resuming) on first
        sight. Touches last_used — every request resets the idle clock."""
        entry = self._entries.get(conversation_id)
        if entry is None:
            bundle = self._assemble(self._cfg, session_id=conversation_id, store=self._store)
            entry = _Entry(bundle=bundle, handle=bundle.agent.spawn())
            self._entries[conversation_id] = entry
        entry.last_used = self._now()
        return entry

    def active_ids(self) -> list[str]:
        return list(self._entries)

    async def evict_idle(self) -> list[str]:
        """Shut down conversations idle past the TTL; returns the evicted ids.
        Cheap sweep — the driver calls it opportunistically (each request) or
        from a periodic task. State is safe to drop: every completed turn was
        snapshotted, so the next request resumes from the store."""
        if self._ttl <= 0:
            return []
        cutoff = self._now() - self._ttl
        expired = [cid for cid, entry in self._entries.items() if entry.last_used <= cutoff]
        for conversation_id in expired:
            await self._shutdown(self._entries.pop(conversation_id))
        return expired

    async def shutdown_all(self) -> None:
        """Drain the registry (app shutdown). Same guarantees as eviction."""
        entries, self._entries = list(self._entries.values()), {}
        for entry in entries:
            await self._shutdown(entry)

    async def _shutdown(self, entry: _Entry) -> None:
        await entry.handle.commands.put(ev.Shutdown())
        try:
            await asyncio.wait_for(entry.handle.task, _SHUTDOWN_GRACE_SECONDS)
        except (TimeoutError, asyncio.CancelledError):
            entry.handle.task.cancel()  # a stuck turn must not hang the sweep
