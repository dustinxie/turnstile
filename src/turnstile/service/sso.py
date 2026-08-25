"""SSO login — SAML in, our JWT out.

FortiAuthenticator (the IdP) authenticates the employee; THIS service issues
the credential (auth.mint_token — the one minting path). The browser flow:

    GET  /sso           -> 302 to the IdP's login page
    POST /sso/acs       <- the IdP posts the signed SAMLResponse here;
                           we validate it and answer {"token": <JWT>}
    GET  /sso/metadata  -> our SP metadata XML (pasted into FAC once)

Routes are mounted UNVERSIONED (not under /v1) and only when `saml` is
configured: the ACS URL is signed into FAC's assertions — a contract with
the IdP that must not move on an API version bump (same logic as /health).

The token comes back as JSON for now (no frontend exists; curl-era honest).
When the frontend lands, this becomes a redirect with a #token= fragment —
one line here, zero protocol change.

AuthZ at login is deliberately thin: any @fortinet.com identity gets a
user token; usernames in `admin_users` get role=admin (the flat model,
auth.require_admin). No user table, no auto-provisioning — that can ride
the M7 sessions table if ever needed.
"""

import logging
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from onelogin.saml2.auth import OneLogin_Saml2_Auth

from turnstile.service.auth import mint_token

logger = logging.getLogger(__name__)

router = APIRouter()

_BINDING_POST = "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
_BINDING_REDIRECT = "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"


def _settings(saml: Any) -> dict:
    """python3-saml settings dict from the duck-typed `saml` config section."""
    return {
        "sp": {
            "entityId": saml.sp_entity_id,
            "assertionConsumerService": {"url": saml.sp_acs_url, "binding": _BINDING_POST},
            "singleLogoutService": {"url": saml.sp_sls_url, "binding": _BINDING_REDIRECT},
            "NameIDFormat": "urn:oasis:names:tc:SAML:1.1:nameid-format:unspecified",
            "x509cert": saml.sp_x509_cert,
            "privateKey": saml.sp_private_key,
        },
        "idp": {
            "entityId": saml.idp_entity_id,
            "singleSignOnService": {"url": saml.idp_sso_url, "binding": _BINDING_REDIRECT},
            "singleLogoutService": {"url": saml.idp_sls_url, "binding": _BINDING_REDIRECT},
            "x509cert": saml.idp_x509_cert,
        },
        "debug": saml.debug,
    }


def _prepare_request(request: Request) -> dict:
    """FastAPI request -> the shape python3-saml expects.

    Reads X-Forwarded-* + X-FortiSSO-Uri so Destination validation works
    behind nginx (TLS terminated + path rewritten upstream).

    SECURITY: these headers are client-spoofable on the wire. The deployment
    contract is that nginx (a) re-sets all of them via proxy_set_header —
    overriding client-supplied values — and (b) is the only path to this
    port. A directly reachable FastAPI port lets an attacker spoof
    Destination and replay assertions; verify network isolation in prod.
    """
    url = urlparse(str(request.url))
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    return {
        "https": "on" if proto == "https" else "off",
        "http_host": request.headers.get("x-forwarded-host", request.headers.get("host", "")),
        "server_port": request.headers.get("x-forwarded-port")
        or url.port
        or (443 if proto == "https" else 80),
        # The IdP signs the PUBLIC path into Destination; behind a rewrite
        # the internal path won't match — nginx must send the public one.
        "script_name": request.headers.get("x-fortisso-uri") or request.url.path,
        "get_data": dict(request.query_params),
        "post_data": {},
        "query_string": request.url.query,
    }


def init_saml_auth(req: dict, saml: Any) -> OneLogin_Saml2_Auth:
    """The SAML seam: tests stub THIS to fake IdP responses — everything
    above it (routes, extraction, minting) is exercised for real."""
    return OneLogin_Saml2_Auth(req, old_settings=_settings(saml))


@router.get("/sso")
async def initiate_sso(request: Request) -> RedirectResponse:
    """Kick off login: 302 to the IdP with a signed AuthnRequest."""
    auth = init_saml_auth(_prepare_request(request), request.app.state.cfg.saml)
    return RedirectResponse(url=auth.login(), status_code=302)


@router.post("/sso/acs")
async def sso_acs(request: Request) -> dict:
    """Assertion Consumer Service: validate the IdP's SAMLResponse, then
    mint OUR credential. Every failure is one generic 401 — the ACS must
    not be an oracle for what part of a forged assertion was wrong."""
    cfg: Any = request.app.state.cfg
    req = _prepare_request(request)
    # SAML POST binding sends SAMLResponse/RelayState single-valued;
    # dict() takes first-occurrence via Starlette's FormData.
    req["post_data"] = dict(await request.form())

    auth = init_saml_auth(req, cfg.saml)
    auth.process_response()
    if auth.get_errors() or not auth.is_authenticated():
        logger.warning(
            "SAML validation failed: %s (%s)", auth.get_errors(), auth.get_last_error_reason()
        )
        raise HTTPException(status_code=401, detail="sso failed")

    attributes = auth.get_attributes()  # may carry PII — never log above DEBUG
    username = _first(attributes, "username", "Username").lower()
    email = _first(attributes, "email", "Email").lower()
    if not username or not email.endswith("@fortinet.com"):
        logger.warning("SAML assertion lacked a fortinet identity (user=%r)", username)
        raise HTTPException(status_code=401, detail="sso failed")

    admins = {u.strip() for u in cfg.admin_users.split(",") if u.strip()}
    role = "admin" if username in admins else "user"
    return {"token": mint_token(cfg.jwt_secret, username, role=role)}


@router.get("/sso/metadata")
async def sso_metadata(request: Request) -> Response:
    """Our SP metadata XML — pasted into the IdP's config once at setup."""
    settings = OneLogin_Saml2_Auth({}, old_settings=_settings(request.app.state.cfg.saml))
    metadata = settings.get_settings().get_sp_metadata()
    return Response(content=metadata, media_type="application/xml")


def _first(attributes: dict, *keys: str) -> str:
    """First value of the first present key ('' when absent) — SAML attrs
    are lists, and FAC deployments vary the key casing."""
    for key in keys:
        values = attributes.get(key) or []
        if values and isinstance(values[0], str):
            return values[0]
    return ""
