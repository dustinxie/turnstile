"""Citation file serving — one endpoint, opaque expiring recipient-bound tokens.

The Reference section links documents as `/v1/files/<token>`; the token IS
the (encrypted) claim `{region, filename, sub, exp}` — Fernet, so it is
opaque in the URL (no path, no region, no filename leaks), tamper-proof
(decryption fails on any bit flip), and STATELESS: no token table, no
lookup, which is what lets this land before the persistence milestone.
What statelessness gives up — revoking one leaked link early — is accepted;
rotating the secret kills all outstanding links if it ever matters.

Recipient binding without state: `sub` is baked in at mint time (the
conversation owner the references were built for) and compared against the
authenticated principal on serve. Someone else's link -> 404, never 403 —
the same no-existence-leak convention as conversation ownership. Binding is
to the principal, not the login session: the same user re-clicking an old
link after re-login still works.

Key material derives from `jwt_secret` (sha256 -> Fernet key): one secret
to operate. Dev mode (no secret) uses a process-random key — links work
within one process lifetime, die on restart; fine for local testing.

The store itself is `<file_root>/<region>/<filename>` — server-side layout,
never visible to clients (see config.py). Serving is Content-Disposition
INLINE: citations open readable in the browser, not as downloads.
"""

import base64
import hashlib
import json
import mimetypes
import time
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from turnstile.service.auth import require_user

router = APIRouter()

TOKEN_TTL_SECONDS = 24 * 3600  # matches the JWT TTL: a chat's links outlive its day

_dev_key: bytes | None = None


def _fernet(secret: str | None) -> Fernet:
    """Deterministic key from the deployment secret, so every caller (the
    route here, root's minting weld) builds the same cipher. No secret =
    one process-random key, shared module-wide for the same reason."""
    if secret:
        return Fernet(base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest()))
    global _dev_key
    if _dev_key is None:
        _dev_key = Fernet.generate_key()
    return Fernet(_dev_key)


def mint_file_token(
    secret: str | None,
    region: str,
    filename: str,
    sub: str,
    ttl_seconds: int = TOKEN_TTL_SECONDS,
) -> str:
    """The claim, encrypted. Compact keys keep the URL short."""
    claim = {"r": region, "f": filename, "s": sub, "e": int(time.time()) + ttl_seconds}
    return _fernet(secret).encrypt(json.dumps(claim).encode()).decode()


def resolve_file_token(secret: str | None, token: str) -> dict | None:
    """Decrypt + expiry check. None for ANY defect (garbage, tampered,
    wrong key, expired) — the route folds them all into one 404."""
    try:
        raw = _fernet(secret).decrypt(token.encode())
        claim = json.loads(raw)
    except (InvalidToken, ValueError):
        return None
    if not isinstance(claim, dict) or int(claim.get("e", 0)) < time.time():
        return None
    return claim


def _safe_relative(rel: str) -> bool:
    """A relative path that stays put: no absolute, no parent hops, no
    backslash tricks. The region is one segment; the document path may nest
    (kb refs are paths like "Benefits/2026/FAQ.pdf"). The token is
    authenticated so a bad value can only come from our own minting —
    defense in depth anyway, ahead of the resolved-containment check."""
    parts = rel.split("/")
    return bool(rel) and not rel.startswith("/") and "\\" not in rel and ".." not in parts


@router.get("/files/{token}")
async def get_file(token: str, request: Request, principal: str = Depends(require_user)):
    """Serve one cited document. Every failure after authentication is the
    same 404 — an invalid, expired, foreign, or dangling link must be
    indistinguishable from one that never existed."""
    cfg: Any = request.app.state.cfg
    claim = resolve_file_token(getattr(cfg, "jwt_secret", None), token)
    if claim is None or claim.get("s") != principal:
        raise HTTPException(status_code=404, detail="not found")

    region, filename = str(claim.get("r", "")), str(claim.get("f", ""))
    if "/" in region or not (_safe_relative(region) and _safe_relative(filename)):
        raise HTTPException(status_code=404, detail="not found")
    root = Path(cfg.file_root).resolve()
    path = (root / region / filename).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="not found")

    mime, _ = mimetypes.guess_type(path.name)
    return FileResponse(
        path=path,
        media_type=mime or "application/octet-stream",
        # inline: the user clicked a citation to READ the source, not save it
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )
