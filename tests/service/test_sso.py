"""SSO login — the SAML seam is stubbed (init_saml_auth); everything above
it runs for real: presence-switch mounting, ACS validation outcomes, the
username-is-the-identity rule, role assignment from admin_users, and that
the minted JWT actually passes require_user/require_admin."""

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

    def login(self, return_to=None) -> str:
        self.return_to = return_to
        return "https://fac.example/idp/sso?SAMLRequest=stub"


def _stub_seam(monkeypatch, **stub_kwargs) -> None:
    monkeypatch.setattr(sso, "init_saml_auth", lambda req, saml: _StubAuth(**stub_kwargs))


async def _acs(client, relay: str = "/") -> httpx.Response:
    return await client.post("/sso/acs", data={"SAMLResponse": "stub", "RelayState": relay})


def _token_from(response: httpx.Response) -> str:
    """The ACS hands the token over as a URL fragment on a 302."""
    assert response.status_code == 302, (response.status_code, response.text)
    location = response.headers["location"]
    assert "#token=" in location, location
    return location.split("#token=", 1)[1]


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
        token = _token_from(response)
        assert response.headers["location"].startswith("/#token=")  # default return_url

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
        token = _token_from(await _acs(client))
    assert pyjwt.decode(token, SECRET, algorithms=["HS256"])["role"] == "admin"


async def test_acs_failures_are_one_generic_401(monkeypatch):
    app = create_app(_cfg())
    cases = [
        {"errors": ["invalid_response"]},  # signature/Destination/expiry failed
        {"authenticated": False},  # processed but not authenticated
        {"attributes": {}},  # no identity attributes at all
        {"attributes": {"email": ["ghost@fortinet.com"]}},  # email but no username: the
        # username IS the identity; email is not consulted
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


# ── where the browser lands ────────────────────────────────────────────


async def test_return_path_rides_relay_state_and_rejects_open_redirects(monkeypatch):
    _stub_seam(monkeypatch, attributes={"username": ["alice"], "email": ["alice@fortinet.com"]})
    app = create_app(_cfg(saml={**_SAML, "return_url": "https://chat.example/app"}))
    async with _client(app) as client:
        # a same-origin next path wins over the configured landing page
        location = (await _acs(client, relay="/chat/c1")).headers["location"]
        assert location.startswith("/chat/c1#token=")
        # anything not a same-origin path falls back to return_url — RelayState
        # is attacker-writable; an absolute URL there would be an open redirect
        for bad in ["https://evil.example/", "//evil.example", "javascript:alert(1)", ""]:
            location = (await _acs(client, relay=bad)).headers["location"]
            assert location.startswith("https://chat.example/app#token="), bad


async def test_initiate_forwards_next_as_relay_state(monkeypatch):
    seen: list[_StubAuth] = []

    def seam(req, saml):
        stub = _StubAuth()
        seen.append(stub)
        return stub

    monkeypatch.setattr(sso, "init_saml_auth", seam)
    app = create_app(_cfg())
    async with _client(app) as client:
        await client.get("/sso", params={"next": "/chat/c1"}, follow_redirects=False)
        await client.get("/sso", params={"next": "https://evil.example"}, follow_redirects=False)
    assert [s.return_to for s in seen] == ["/chat/c1", None]
