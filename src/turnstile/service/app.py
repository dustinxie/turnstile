"""FastAPI app factory — the web driver's entry point.

`create_app()` reads config via root (the service never imports the config
layer — the c6 contract), builds the process-wide snapshot store, and wires
the routes. The process starts at the driver; root is a function it calls,
not an entry point (architecture.md §2).

    uvicorn --factory turnstile.service.app:create_app

Tests inject their own `cfg` so no environment is needed.
"""

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from turnstile import root
from turnstile.service import files, routes, sso
from turnstile.service.registry import ConversationRegistry


def create_app(cfg: Any = None) -> FastAPI:
    """Build the service. `cfg` is root's config object, held opaquely
    (duck-typed — same discipline as AgentSpec); None = read the env."""
    if cfg is None:
        cfg = root.load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        # Graceful drain: every running session gets Shutdown (bounded), so
        # in-flight turns finish their snapshots before the process exits.
        await app.state.registry.shutdown_all()

    app = FastAPI(title="turnstile", docs_url=None, redoc_url=None, lifespan=lifespan)
    # Process-wide state: ONE snapshot store outlives every conversation
    # (root.build_store's contract); the registry maps conversation ids onto
    # running agents assembled from cfg + that store.
    app.state.cfg = cfg
    app.state.store = root.build_store()
    app.state.registry = ConversationRegistry(cfg, app.state.store)
    # API is versioned (/v1/conversations/...); /health is deliberately
    # OUTSIDE the prefix — probes pin to the container, not the API contract,
    # and must not move when the API version bumps.
    app.include_router(routes.router, prefix="/v1")
    # SSO is presence-switched (same pattern as the judge) and UNVERSIONED:
    # the ACS URL is a contract with the IdP — it must not move with /v1.
    if getattr(cfg, "saml", None) is not None:
        app.include_router(sso.router)
    # Citation file serving is presence-switched on the store's existence;
    # ordinary API surface, so it lives under /v1 (unlike /sso).
    if getattr(cfg, "file_root", None):
        app.include_router(files.router, prefix="/v1")

    @app.get("/health")
    async def health() -> dict:
        # Liveness + which product this deployment serves. Deliberately no
        # upstream probes: a slow LLM must not flap the container's health.
        # `commit`/`docker_tag` are build provenance stamped into the image
        # env by the Dockerfile (GIT_COMMIT/DOCKER_TAG build args) — "what is
        # deployed?" answerable over HTTP.
        return {
            "status": "ok",
            "spec": cfg.spec_id,
            "commit": os.environ.get("GIT_COMMIT", "unknown"),
            "docker_tag": os.environ.get("DOCKER_TAG", "unknown"),
            # what the frontend needs to shape its login UI: is a token
            # required at all, and is there an SSO route to send people to
            "auth": bool(getattr(cfg, "jwt_secret", None)),
            "sso": getattr(cfg, "saml", None) is not None,
        }

    return app
