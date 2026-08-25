"""Tokenized citation file serving — mint/serve round-trip, recipient
binding, expiry, tamper, traversal, and the presence-switch. Every failure
after authentication is ONE 404 (no-existence-leak convention)."""

import httpx
import pytest

from turnstile.config import Config
from turnstile.service.app import create_app
from turnstile.service.auth import mint_token
from turnstile.service.files import mint_file_token, resolve_file_token

pytestmark = pytest.mark.service

SECRET = "test-secret-0123456789abcdef-0123456789"  # >=32 bytes: HS256 floor


@pytest.fixture()
def store(tmp_path):
    """A two-region store with a same-name collision — the reason the
    region tier exists."""
    (tmp_path / "hrus").mkdir()
    (tmp_path / "hrcanada").mkdir()
    (tmp_path / "hrus" / "leave_benefits.pdf").write_bytes(b"%PDF-1.7 us-version")
    (tmp_path / "hrcanada" / "leave_benefits.pdf").write_bytes(b"%PDF-1.7 ca-version")
    (tmp_path / "hrus" / "holidays.txt").write_text("fixed holidays")
    return tmp_path


def _cfg(store_path=None, jwt_secret=SECRET) -> Config:
    return Config(
        _env_file=None,  # type: ignore[call-arg]
        jwt_secret=jwt_secret,
        file_root=str(store_path) if store_path else None,
        llm={"base_url": "https://ds4.example/v1", "model": "model-fast"},
        kb={
            "embedding_url": "https://e/x",
            "milvus_url": "https://m/x",
            "collection": "c",
            "expr": "e",
        },
    )


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def _bearer(user: str) -> dict:
    return {"Authorization": f"Bearer {mint_token(SECRET, user)}"}


def _url(region: str, filename: str, sub: str, ttl: int = 600) -> str:
    return f"/v1/files/{mint_file_token(SECRET, region, filename, sub, ttl_seconds=ttl)}"


# ── the happy path ─────────────────────────────────────────────────────


async def test_minted_link_serves_the_file_inline(store):
    app = create_app(_cfg(store))
    async with _client(app) as client:
        response = await client.get(
            _url("hrus", "leave_benefits.pdf", "alice"), headers=_bearer("alice")
        )
    assert response.status_code == 200
    assert response.content == b"%PDF-1.7 us-version"
    assert response.headers["content-type"] == "application/pdf"  # mime-guessed
    assert response.headers["content-disposition"].startswith("inline")  # read, not download


async def test_region_tier_disambiguates_same_name_docs(store):
    app = create_app(_cfg(store))
    async with _client(app) as client:
        ca = await client.get(
            _url("hrcanada", "leave_benefits.pdf", "alice"), headers=_bearer("alice")
        )
    assert ca.content == b"%PDF-1.7 ca-version"  # same filename, other region


# ── everything else is 404 ─────────────────────────────────────────────


async def test_foreign_recipient_is_404(store):
    # bob authenticates fine but the link was minted for alice
    app = create_app(_cfg(store))
    async with _client(app) as client:
        url = _url("hrus", "holidays.txt", "alice")
        assert (await client.get(url, headers=_bearer("bob"))).status_code == 404
        assert (await client.get(url, headers=_bearer("alice"))).status_code == 200  # still hers


async def test_expired_garbage_and_dangling_links_are_404(store):
    app = create_app(_cfg(store))
    async with _client(app) as client:
        headers = _bearer("alice")
        expired = _url("hrus", "holidays.txt", "alice", ttl=-10)
        assert (await client.get(expired, headers=headers)).status_code == 404
        assert (await client.get("/v1/files/garbage", headers=headers)).status_code == 404
        dangling = _url("hrus", "no_such_doc.pdf", "alice")  # valid token, file gone
        assert (await client.get(dangling, headers=headers)).status_code == 404


async def test_traversal_segments_never_resolve(store):
    # the token is authenticated, so these can only come from our own
    # minting — defense in depth still refuses them
    (store.parent / "outside.txt").write_text("secret")
    app = create_app(_cfg(store))
    async with _client(app) as client:
        headers = _bearer("alice")
        for region, filename in [("..", "outside.txt"), ("hrus", "../hrus/holidays.txt")]:
            url = _url(region, filename, "alice")
            assert (await client.get(url, headers=headers)).status_code == 404, (region, filename)


async def test_authentication_still_comes_first(store):
    app = create_app(_cfg(store))
    async with _client(app) as client:
        assert (await client.get(_url("hrus", "holidays.txt", "alice"))).status_code == 401


# ── wiring ─────────────────────────────────────────────────────────────


async def test_no_file_root_means_no_files_route(store):
    app = create_app(_cfg(store_path=None))
    async with _client(app) as client:
        url = _url("hrus", "holidays.txt", "alice")
        assert (await client.get(url, headers=_bearer("alice"))).status_code == 404  # not mounted


def test_tokens_are_opaque_and_tamper_proof():
    token = mint_file_token(SECRET, "hrus", "leave_benefits.pdf", "alice")
    assert "hrus" not in token and "leave" not in token and "alice" not in token  # encrypted
    assert resolve_file_token(SECRET, token[:-2] + "xx") is None  # any bit flip kills it
    assert resolve_file_token("other-secret-0123456789abcdef-012345", token) is None
    claim = resolve_file_token(SECRET, token)
    assert claim is not None
    assert (claim["r"], claim["f"], claim["s"]) == ("hrus", "leave_benefits.pdf", "alice")
