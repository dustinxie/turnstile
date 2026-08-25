"""AuthN + conversation ownership — 401 paths, the anonymous dev mode, and
foreign conversations answering 404 (never 403: existence must not leak).
Plus the flat authorization model: the signed role claim and the admin gate
(401 = not authenticated, 403 = authenticated but not admin)."""

import httpx
import pytest
from fastapi import Depends

from turnstile.capabilities.persistence.memory_store import MemorySessionStore
from turnstile.kernel.dtos import Done, TextDelta
from turnstile.kernel.engine import Agent
from turnstile.kernel.testkit import ScriptedProvider
from turnstile.products.middleware.references import ReferenceCollector
from turnstile.root import AssembledAgent
from turnstile.service.app import create_app
from turnstile.service.auth import mint_token, require_admin
from turnstile.service.registry import ConversationRegistry

pytestmark = pytest.mark.service

SECRET = "test-secret-0123456789abcdef-0123456789"  # >=32 bytes: HS256 floor


def _app(jwt_secret: str | None):
    from turnstile.config import Config

    cfg = Config(
        _env_file=None,  # type: ignore[call-arg]
        jwt_secret=jwt_secret,
        llm={"base_url": "https://ds4.example/v1", "model": "model-fast"},
        kb={
            "embedding_url": "https://e/x",
            "milvus_url": "https://m/x",
            "collection": "c",
            "expr": "e",
        },
    )
    store = MemorySessionStore()

    def scripted_assemble(cfg, session_id, store):
        agent = Agent(
            provider=ScriptedProvider(rounds=[[TextDelta("answer"), Done()]] * 9),
            hooks=[store.hook(session_id)],
            session_id=session_id,
            resume=store.load(session_id),
        )
        return AssembledAgent(agent=agent, references=ReferenceCollector(), store=store)

    app = create_app(cfg)
    app.state.store = store
    app.state.registry = ConversationRegistry(cfg, store, assemble=scripted_assemble)

    # Stand-in for the future admin-only surface: no production route uses
    # require_admin yet, so the gate is proven on a probe route here.
    @app.get("/admin-probe")
    async def admin_probe(principal: str = Depends(require_admin)) -> dict:  # pyright: ignore[reportUnusedFunction]
        return {"admin": principal}

    return app


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def _bearer(user: str, secret: str = SECRET, ttl: int = 3600, role: str = "user") -> dict:
    return {"Authorization": f"Bearer {mint_token(secret, user, ttl, role=role)}"}


async def _post(client, cid: str, headers: dict | None = None) -> int:
    async with client.stream(
        "POST", f"/v1/conversations/{cid}/messages", json={"text": "q"}, headers=headers or {}
    ) as response:
        await response.aread()
        return response.status_code


# ── authn ──────────────────────────────────────────────────────────────


async def test_auth_off_everything_is_anonymous():
    app = _app(jwt_secret=None)
    async with _client(app) as client:
        assert await _post(client, "c1") == 200  # no header needed in dev mode
        assert (await client.get("/v1/conversations/c1")).status_code == 200
    await app.state.registry.shutdown_all()


async def test_auth_on_rejects_missing_bad_and_expired_tokens():
    app = _app(jwt_secret=SECRET)
    async with _client(app) as client:
        assert await _post(client, "c1") == 401  # no header
        assert await _post(client, "c1", {"Authorization": "Bearer garbage"}) == 401
        wrong = "wrong-secret-0123456789abcdef-01234567"
        assert await _post(client, "c1", _bearer("alice", secret=wrong)) == 401
        assert await _post(client, "c1", _bearer("alice", ttl=-10)) == 401  # expired
        assert (await client.get("/v1/conversations/c1")).status_code == 401
        assert (await client.post("/v1/conversations/c1/cancel")).status_code == 401
    assert app.state.registry.active_ids() == []  # nothing spawned by rejects


async def test_valid_token_resolves_the_principal_and_works():
    app = _app(jwt_secret=SECRET)
    async with _client(app) as client:
        assert await _post(client, "c1", _bearer("alice")) == 200
        assert app.state.store.owner("c1") == "alice"  # first toucher claimed it
    await app.state.registry.shutdown_all()


async def test_health_stays_open():
    app = _app(jwt_secret=SECRET)
    async with _client(app) as client:
        assert (await client.get("/health")).status_code == 200  # container probes


# ── the role claim + admin gate ────────────────────────────────────────


async def test_admin_token_passes_the_gate():
    app = _app(jwt_secret=SECRET)
    async with _client(app) as client:
        response = await client.get("/admin-probe", headers=_bearer("alice", role="admin"))
    assert response.status_code == 200
    assert response.json() == {"admin": "alice"}


async def test_user_role_is_403_not_401():
    # a real identity that isn't admin: authenticated (not 401), forbidden.
    # The role is SIGNED into the token — a client cannot elevate itself.
    app = _app(jwt_secret=SECRET)
    async with _client(app) as client:
        assert (await client.get("/admin-probe", headers=_bearer("bob"))).status_code == 403
        # default mint role is "user" — an unspecified role never elevates
        headers = {"Authorization": f"Bearer {mint_token(SECRET, 'bob', 3600)}"}
        assert (await client.get("/admin-probe", headers=headers)).status_code == 403


async def test_admin_gate_still_authenticates_first():
    app = _app(jwt_secret=SECRET)
    async with _client(app) as client:
        assert (await client.get("/admin-probe")).status_code == 401  # no token
        wrong = "wrong-secret-0123456789abcdef-01234567"
        forged = _bearer("mallory", secret=wrong, role="admin")  # wrong key -> dead
        assert (await client.get("/admin-probe", headers=forged)).status_code == 401


async def test_dev_mode_has_no_admin():
    # auth off: user surfaces run anonymous, but there is no identity to
    # elevate — admin surfaces stay closed until real auth is configured
    app = _app(jwt_secret=None)
    async with _client(app) as client:
        assert (await client.get("/admin-probe")).status_code == 403


async def test_role_claim_does_not_disturb_user_routes():
    # an admin token is still a valid user token for ordinary surfaces
    app = _app(jwt_secret=SECRET)
    async with _client(app) as client:
        assert await _post(client, "c1", _bearer("alice", role="admin")) == 200
        assert app.state.store.owner("c1") == "alice"
    await app.state.registry.shutdown_all()


# ── ownership ──────────────────────────────────────────────────────────


async def test_foreign_conversation_is_404_on_every_route():
    app = _app(jwt_secret=SECRET)
    async with _client(app) as client:
        assert await _post(client, "c1", _bearer("alice")) == 200

        # bob sees alice's conversation as nonexistent — 404, never 403
        assert await _post(client, "c1", _bearer("bob")) == 404
        assert (
            await client.get("/v1/conversations/c1", headers=_bearer("bob"))
        ).status_code == 404
        cancel = await client.post("/v1/conversations/c1/cancel", headers=_bearer("bob"))
        assert cancel.status_code == 404

        # alice still owns it; bob gets his own id untouched
        assert (
            await client.get("/v1/conversations/c1", headers=_bearer("alice"))
        ).status_code == 200
        assert await _post(client, "c2", _bearer("bob")) == 200
    await app.state.registry.shutdown_all()


async def test_ownership_survives_eviction_via_the_store():
    # eviction drops the live agent but NOT the snapshot/owner — the resumed
    # conversation still belongs to its principal
    app = _app(jwt_secret=SECRET)
    async with _client(app) as client:
        assert await _post(client, "c1", _bearer("alice")) == 200
        await app.state.registry.shutdown_all()  # stand-in for idle eviction
        assert app.state.registry.active_ids() == []
        assert await _post(client, "c1", _bearer("bob")) == 404  # still alice's
        assert await _post(client, "c1", _bearer("alice")) == 200  # resumes fine
    await app.state.registry.shutdown_all()
