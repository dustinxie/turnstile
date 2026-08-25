"""SSO login — the SAML seam is stubbed (init_saml_auth); everything above
it runs for real: presence-switch mounting, ACS validation outcomes, the
fortinet-identity gate, role assignment from admin_users, and that the
minted JWT actually passes require_user/require_admin."""

import httpx
import jwt as pyjwt
import pytest

from turnstile.config import Config
from turnstile.service import sso
from turnstile.service.app import create_app

pytestmark = pytest.mark.service

SECRET = "test-secret-0123456789abcdef-0123456789"  # >=32 bytes: HS256 floor

_SAML = {
    "sp_entity_id": "https://bot.example/sso/metadata",
    "sp_acs_url": "https://bot.example/sso/acs",
    "idp_entity_id": "https://fac.example/idp",
    "idp_sso_url": "https://fac.example/idp/sso",
    "idp_x509_cert": "MIIF-fake-cert",
}


def _cfg(**overrides) -> Config:
    base = {
        "jwt_secret": SECRET,
        "saml": _SAML,
        "llm": {"base_url": "https://ds4.example/v1", "model": "model-fast"},
        "kb": {
            "embedding_url": "https://e/x",
            "milvus_url": "https://m/x",
            "collection": "c",
            "expr": "e",
        },
    }
    return Config(_env_file=None, **{**base, **overrides})  # type: ignore[arg-type]


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


class _StubAuth:
    """What the seam returns in tests: a canned IdP outcome."""

    def __init__(self, attributes=None, errors=(), authenticated=True):
        self._attributes = attributes or {}
        self._errors = list(errors)
        self._authenticated = authenticated

    def process_response(self) -> None:
        pass

    def get_errors(self) -> list:
        return self._errors

    def get_last_error_reason(self) -> str:
        return self._errors[0] if self._errors else ""

    def is_authenticated(self) -> bool:
        return self._authenticated

    def get_attributes(self) -> dict:
        return self._attributes

    def login(self) -> str:
        return "https://fac.example/idp/sso?SAMLRequest=stub"


def _stub_seam(monkeypatch, **stub_kwargs) -> None:
    monkeypatch.setattr(sso, "init_saml_auth", lambda req, saml: _StubAuth(**stub_kwargs))


async def _acs(client) -> httpx.Response:
    return await client.post("/sso/acs", data={"SAMLResponse": "stub", "RelayState": "/"})


# ── mounting ───────────────────────────────────────────────────────────


async def test_no_saml_config_means_no_sso_routes():
    app = create_app(_cfg(saml=None))
    async with _client(app) as client:
        assert (await client.get("/sso")).status_code == 404
        assert (await _acs(client)).status_code == 404
        assert (await client.get("/sso/metadata")).status_code == 404


def test_saml_without_jwt_secret_fails_at_boot():
    with pytest.raises(ValueError, match="jwt_secret"):
        _cfg(jwt_secret=None)


async def test_initiate_redirects_to_the_idp(monkeypatch):
    _stub_seam(monkeypatch)
    app = create_app(_cfg())
    async with _client(app) as client:
        response = await client.get("/sso", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"].startswith("https://fac.example/idp/sso")


# ── the ACS: SAML in, our JWT out ──────────────────────────────────────


async def test_acs_mints_a_working_user_token(monkeypatch):
    _stub_seam(
        monkeypatch,
        attributes={"username": ["Alice"], "email": ["alice@fortinet.com"]},
    )
    app = create_app(_cfg())
    async with _client(app) as client:
        response = await _acs(client)
        assert response.status_code == 200
        token = response.json()["token"]

        claims = pyjwt.decode(token, SECRET, algorithms=["HS256"])
        assert claims["sub"] == "alice"  # lowercased
        assert claims["role"] == "user"

        # the minted credential works against the real user surface
        get = await client.get(
            "/v1/conversations/nope", headers={"Authorization": f"Bearer {token}"}
        )
        assert get.status_code == 404  # authenticated (not 401), just unknown id


async def test_admin_users_mints_the_admin_role(monkeypatch):
    _stub_seam(
        monkeypatch,
        attributes={"username": ["boss"], "email": ["boss@fortinet.com"]},
    )
    app = create_app(_cfg(admin_users="boss, other.admin"))
    async with _client(app) as client:
        token = (await _acs(client)).json()["token"]
    assert pyjwt.decode(token, SECRET, algorithms=["HS256"])["role"] == "admin"


async def test_acs_failures_are_one_generic_401(monkeypatch):
    app = create_app(_cfg())
    cases = [
        {"errors": ["invalid_response"]},  # signature/Destination/expiry failed
        {"authenticated": False},  # processed but not authenticated
        {"attributes": {}},  # no identity attributes at all
        {"attributes": {"username": ["eve"], "email": ["eve@evil.example"]}},  # foreign domain
        {"attributes": {"email": ["ghost@fortinet.com"]}},  # email but no username
    ]
    for stub_kwargs in cases:
        _stub_seam(monkeypatch, **stub_kwargs)
        async with _client(app) as client:
            response = await _acs(client)
        assert response.status_code == 401, stub_kwargs
        assert response.json()["detail"] == "sso failed"  # no oracle


# ── metadata (real python3-saml, no stub) ──────────────────────────────


async def test_metadata_serves_sp_xml():
    app = create_app(_cfg())
    async with _client(app) as client:
        response = await client.get("/sso/metadata")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    body = response.text
    assert "https://bot.example/sso/metadata" in body  # our entityId
    assert "https://bot.example/sso/acs" in body  # the ACS contract
