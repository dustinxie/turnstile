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


def mint_token(secret: str, sub: str, ttl_seconds: int = 24 * 3600) -> str:
    """A signed credential for `sub` — the ONE minting path: the by-hand
    one-liner above, the future SSO login route, and the tests."""
    now = int(time.time())
    return jwt.encode({"sub": sub, "iat": now, "exp": now + ttl_seconds}, secret, _ALGORITHM)


async def require_user(request: Request) -> str:
    """FastAPI dependency: the verified principal for this request.

    Auth off (no jwt_secret) -> ANONYMOUS. Auth on -> a valid, unexpired
    Bearer JWT is required; anything else is 401 (one generic detail — no
    oracle for which check failed)."""
    cfg: Any = request.app.state.cfg  # duck-typed, same discipline as everywhere
    secret = getattr(cfg, "jwt_secret", None)
    if not secret:
        return ANONYMOUS

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
    return sub
