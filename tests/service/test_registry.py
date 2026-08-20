"""Conversation registry — spawn/reuse identity, TTL eviction with graceful
shutdown, resume-after-eviction through the shared store. Bundles come from a
scripted assemble factory (testkit provider), never the network."""

from types import SimpleNamespace

import pytest

from turnstile.capabilities.persistence.memory_store import MemorySessionStore
from turnstile.kernel import events as ev
from turnstile.kernel.dtos import Done, TextDelta
from turnstile.kernel.engine import Agent
from turnstile.kernel.testkit import ScriptedProvider
from turnstile.products.middleware.references import ReferenceCollector
from turnstile.root import AssembledAgent
from turnstile.service.registry import ConversationRegistry

pytestmark = pytest.mark.service


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _scripted_assemble(cfg, session_id: str, store: MemorySessionStore) -> AssembledAgent:
    """Root-shaped factory over testkit doubles: same signature, same resume
    and persistence wiring, scripted provider instead of the network."""
    provider = ScriptedProvider(rounds=[[TextDelta(f"answer for {session_id}"), Done()]] * 5)
    agent = Agent(
        provider=provider,
        hooks=[store.hook(session_id)],
        session_id=session_id,
        resume=store.load(session_id),
    )
    return AssembledAgent(agent=agent, references=ReferenceCollector(), store=store)


def _registry(ttl: float = 60.0):
    clock = _Clock()
    store = MemorySessionStore()
    cfg = SimpleNamespace(session_ttl_seconds=ttl)
    registry = ConversationRegistry(cfg, store, assemble=_scripted_assemble, now=clock)
    return registry, clock, store


async def _run_turn(entry, text: str) -> str:
    await entry.handle.commands.put(ev.SendMessage(text=text))
    answer = ""
    while True:
        event = await entry.handle.events.get()
        if isinstance(event, ev.TextDelta):
            answer += event.text
        elif isinstance(event, ev.TurnComplete):
            return answer


# ── identity ───────────────────────────────────────────────────────────


async def test_same_id_returns_the_same_running_entry():
    registry, _, _ = _registry()
    first = registry.get_or_create("c1")
    again = registry.get_or_create("c1")
    assert again is first  # one conversation = one live agent, steer-capable
    assert registry.active_ids() == ["c1"]
    await registry.shutdown_all()


async def test_distinct_ids_get_distinct_agents_over_one_store():
    registry, _, store = _registry()
    one, two = registry.get_or_create("c1"), registry.get_or_create("c2")
    assert one.handle is not two.handle
    assert one.bundle.store is two.bundle.store is store  # shared snapshots
    await registry.shutdown_all()


# ── eviction + resume ──────────────────────────────────────────────────


async def test_idle_conversations_evict_and_active_ones_survive():
    registry, clock, _ = _registry(ttl=60.0)
    registry.get_or_create("idle")
    clock.now += 61
    survivor = registry.get_or_create("busy")  # touched now
    assert await registry.evict_idle() == ["idle"]
    assert registry.active_ids() == ["busy"]
    assert registry.get_or_create("busy") is survivor  # untouched by the sweep
    await registry.shutdown_all()


async def test_touch_resets_the_idle_clock():
    registry, clock, _ = _registry(ttl=60.0)
    registry.get_or_create("c1")
    clock.now += 59
    registry.get_or_create("c1")  # activity just before the deadline
    clock.now += 59
    assert await registry.evict_idle() == []  # 118s old but touched at 59
    await registry.shutdown_all()


async def test_ttl_zero_never_evicts():
    registry, clock, _ = _registry(ttl=0)
    registry.get_or_create("c1")
    clock.now += 10_000_000
    assert await registry.evict_idle() == []
    await registry.shutdown_all()


async def test_eviction_shuts_down_gracefully_and_next_request_resumes():
    registry, clock, _ = _registry(ttl=60.0)
    entry = registry.get_or_create("c1")
    answer = await _run_turn(entry, "q1")
    assert answer == "answer for c1"

    clock.now += 61
    assert await registry.evict_idle() == ["c1"]
    assert entry.handle.task.done()  # Shutdown honored, session loop exited

    # the turn was snapshotted before eviction; a fresh entry resumes from it
    revived = registry.get_or_create("c1")
    assert revived is not entry
    resumed = revived.bundle.agent.resume
    assert resumed is not None
    assert [m.text for m in resumed.messages] == ["q1", "answer for c1"]
    await registry.shutdown_all()


async def test_shutdown_all_drains_everything():
    registry, _, _ = _registry()
    entries = [registry.get_or_create(f"c{i}") for i in range(3)]
    await registry.shutdown_all()
    assert registry.active_ids() == []
    assert all(e.handle.task.done() for e in entries)
