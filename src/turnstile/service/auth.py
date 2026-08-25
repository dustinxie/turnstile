"""AuthN — who is making this request.

JWT (HS256) verified against the deployment's `jwt_secret`; the principal is
the token's `sub` claim. Secret unset = auth OFF: every request resolves to
the ANONYMOUS principal, so ownership logic stays uniform in dev.

The IdP story (SAML/FortiAuthenticator) terminates at a future login route
IN THIS SERVICE that mints these same tokens — FAC authenticates, we issue
the credential. Until then, mint by hand for service accounts and demos:

    uv run python -c "from turnstile.service.auth import mint_token; \
        print(mint_token('$JWT_SECRET', 'alice'))"

AuthZ beyond identity (which tools, which data) is deliberately absent: the
RBAC middleware arrives with the first side-effecting tool; today's tools
are read-only and the only enforcement needed is conversation OWNERSHIP,
done in the routes against the session store.
"""

import time
from typing import Any

import jwt
from fastapi import HTTPException, Request

ANONYMOUS = "anonymous"
_ALGORITHM = "HS256"


def mint_token(secret: str, sub: str, ttl_seconds: int = 24 * 3600, role: str = "user") -> str:
    """A signed credential for `sub` — the ONE minting path: the by-hand
    one-liner above, the future SSO login route, and the tests.

    `role` is the flat authorization model (user | admin): signed into the
    token so the client cannot forge it, read by require_admin. Deliberately
    NOT an RBAC framework — per-resource permissions stay deferred to the
    RBAC middleware (arrives with the first side-effecting tool)."""
    now = int(time.time())
    claims = {"sub": sub, "role": role, "iat": now, "exp": now + ttl_seconds}
    return jwt.encode(claims, secret, _ALGORITHM)


def _verify_claims(request: Request, secret: str) -> dict:
    """Parse + verify the request's Bearer JWT. Any failure is 401 (one
    generic detail — no oracle for which check failed)."""
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="missing bearer token")
    try:
        claims = jwt.decode(token, secret, algorithms=[_ALGORITHM])
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail="invalid token") from e
    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub:
        raise HTTPException(status_code=401, detail="invalid token")
    return claims


async def require_user(request: Request) -> str:
    """FastAPI dependency: the verified principal for this request.

    Auth off (no jwt_secret) -> ANONYMOUS. Auth on -> a valid, unexpired
    Bearer JWT is required; anything else is 401."""
    cfg: Any = request.app.state.cfg  # duck-typed, same discipline as everywhere
    secret = getattr(cfg, "jwt_secret", None)
    if not secret:
        return ANONYMOUS
    return _verify_claims(request, secret)["sub"]


async def require_admin(request: Request) -> str:
    """FastAPI dependency for the few admin-only routes: require_user's
    verification PLUS the role claim. 401 = not authenticated; 403 = a real
    identity that simply isn't admin (role is signed — a client cannot
    elevate itself). Dev mode (auth off) has no identity to elevate -> 403:
    admin surfaces need real auth even where user surfaces run anonymous."""
    cfg: Any = request.app.state.cfg
    secret = getattr(cfg, "jwt_secret", None)
    if not secret:
        raise HTTPException(status_code=403, detail="admin only")
    claims = _verify_claims(request, secret)
    if claims.get("role") != "admin":
        raise HTTPException(status_code=403, detail="admin only")
    return claims["sub"]
