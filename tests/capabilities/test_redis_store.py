"""Redis session store — the codec round-trips every DTO field losslessly,
and the store honors the MemorySessionStore surface against a REAL local
redis-server (spawned per module on a free port; skipped when the binary is
absent). Marked unit: it is fast and needs no network."""

import shutil
import socket
import subprocess
import time

import pytest
import redis

from turnstile.capabilities.persistence import snapshot_codec
from turnstile.capabilities.persistence.redis_store import RedisSessionStore
from turnstile.kernel.dtos import (
    SNAPSHOT_VERSION,
    ImageContent,
    Message,
    MessageMeta,
    ReasoningBlock,
    Role,
    SessionSnapshot,
    TokenUsage,
    ToolCall,
)

pytestmark = pytest.mark.unit


def _snapshot() -> SessionSnapshot:
    meta = MessageMeta(
        tokens=TokenUsage(prompt=120, completion=33, cached=100),
        elapsed_ms=812,
        reasoning_elapsed_ms=40,
        ctx_window=128_000,
        used_tokens=1200,
        utilization=0.0094,
        round=2,
        turn_id=3,
        request_id=7,
        provider_response_id="resp-1",
        provider_model="model-fast",
        session_id="s1",
        finish_reason="stop",
    )
    return SessionSnapshot(
        version=SNAPSHOT_VERSION,
        cache_epoch=1,
        turn_counter=3,
        request_counter=7,
        messages=[
            Message.system("persona — with unicode ✓"),
            Message.user("q", images=[ImageContent("image/png", "aGVsbG8=")]),
            Message.assistant("", [ToolCall("c1", "kb_search", '{"query": "x"}')]),
            Message.tool_result("c1", "[1] hrus::a.pdf#L1\nchunk", is_error=False),
            Message.synthetic_user("judge critique"),
            Message(
                role=Role.ASSISTANT,
                text="answer [1]",
                meta=meta,
                reasoning="thought",
                reasoning_blocks=[ReasoningBlock("t", opaque="sig", provider="anthropic")],
                internal_origin="verify_cadence",
            ),
        ],
    )


def test_codec_round_trips_every_field():
    snap = _snapshot()
    back = snapshot_codec.loads(snapshot_codec.dumps(snap))
    assert back == snap  # dataclass equality across the whole tree
    assert back.messages[1].images[0].media_type == "image/png"
    assert back.messages[5].meta is not None and back.messages[5].meta.tokens.cached == 100


# ── against a real redis-server ────────────────────────────────────────


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def redis_url():
    binary = shutil.which("redis-server")
    if binary is None:
        pytest.skip("redis-server not installed")
    port = _free_port()
    proc = subprocess.Popen(
        [binary, "--port", str(port), "--save", "", "--appendonly", "no", "--loglevel", "warning"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"redis://127.0.0.1:{port}/0"
    client = redis.Redis.from_url(url)
    for _ in range(50):
        try:
            if client.ping():
                break
        except redis.RedisError:
            time.sleep(0.05)
    else:
        proc.kill()
        pytest.skip("redis-server did not come up")
    yield url
    proc.kill()
    proc.wait()


def test_save_load_drop_and_keep_latest(redis_url):
    store = RedisSessionStore(redis_url, prefix="t1")
    assert store.ping()
    assert store.load("c1") is None
    snap = _snapshot()
    store.save("c1", snap)
    assert store.load("c1") == snap
    later = SessionSnapshot(version=SNAPSHOT_VERSION, messages=[Message.user("later")])
    store.save("c1", later)
    assert store.load("c1") == later  # keep-latest = overwrite
    store.drop("c1")
    assert store.load("c1") is None
    store.drop("c1")  # missing id is a no-op


def test_ownership_first_claim_wins_and_listing_is_per_principal(redis_url):
    store = RedisSessionStore(redis_url, prefix="t2")
    assert store.owner("c1") is None
    assert store.claim("c1", "alice") == "alice"
    assert store.claim("c1", "bob") == "alice"  # first claimant wins
    assert store.claim("c2", "bob") == "bob"
    assert store.claim("c0", "alice") == "alice"
    assert store.owned_by("alice") == ["c0", "c1"]  # sorted, alice's only
    assert store.owned_by("bob") == ["c2"]
    store.drop("c1")
    assert store.owner("c1") is None and store.owned_by("alice") == ["c0"]


def test_ttl_ages_snapshot_and_owner_out_together(redis_url):
    store = RedisSessionStore(redis_url, ttl_seconds=1, prefix="t3")
    store.claim("c1", "alice")
    store.save("c1", _snapshot())
    assert store.load("c1") is not None and store.owned_by("alice") == ["c1"]
    time.sleep(1.3)
    assert store.load("c1") is None  # the resume window closed
    assert store.owner("c1") is None
    assert store.owned_by("alice") == []  # pruned lazily from the listing


def test_prefix_isolates_deployments(redis_url):
    a = RedisSessionStore(redis_url, prefix="dep-a")
    b = RedisSessionStore(redis_url, prefix="dep-b")
    a.claim("c1", "alice")
    a.save("c1", _snapshot())
    assert b.load("c1") is None and b.owner("c1") is None


async def test_hook_and_checkpoint_write_the_sessions_slot(redis_url):
    from turnstile.kernel.dtos import Done, TextDelta
    from turnstile.kernel.engine import Agent
    from turnstile.kernel.testkit import ScriptedProvider

    store = RedisSessionStore(redis_url, prefix="t4")
    agent = Agent(
        provider=ScriptedProvider(rounds=[[TextDelta("hi"), Done()]]),
        hooks=[store.hook("s")],
        checkpoint=store.checkpoint("s"),
        session_id="s",
    )
    await agent.run_to_completion("q")
    snap = store.load("s")
    assert snap is not None and snap.turn_counter == 1
    assert [m.text for m in snap.messages if m.role is Role.ASSISTANT] == ["hi"]
    # resume from the persisted snapshot: monotonic counters
    resumed = Agent(
        provider=ScriptedProvider(rounds=[[TextDelta("again"), Done()]]),
        hooks=[store.hook("s")],
        session_id="s",
        resume=store.load("s"),
    )
    await resumed.run_to_completion("q2")
    snap2 = store.load("s")
    assert snap2 is not None and snap2.turn_counter == 2


def test_root_wires_retention_from_the_redis_section_not_eviction(redis_url):
    # redis.ttl_seconds = retention (resume window); session_ttl_seconds only
    # evicts the live agent from memory — two knobs, deliberately separate
    from turnstile.config import Config
    from turnstile.root import build_store

    cfg = Config(
        _env_file=None,  # type: ignore[call-arg]
        session_ttl_seconds=60,
        redis={"url": redis_url, "ttl_seconds": 86400 * 30, "prefix": "t6"},
        llm={"base_url": "https://ds4.example/v1", "model": "model-fast"},
        kb={
            "embedding_url": "https://e/x",
            "milvus_url": "https://m/x",
            "collection": "c",
            "expr": "e",
        },
    )
    store = build_store(cfg.redis)  # the store needs only its own section
    assert isinstance(store, RedisSessionStore)
    assert store._ttl == 86400 * 30  # retention, not the 60s eviction knob
