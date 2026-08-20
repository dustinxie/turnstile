"""FastAPI app factory — the web driver's entry point.

`create_app()` reads config via root (the service never imports the config
layer — the c6 contract), builds the process-wide snapshot store, and wires
the routes. The process starts at the driver; root is a function it calls,
not an entry point (architecture.md §2).

    uvicorn --factory turnstile.service.app:create_app

Tests inject their own `cfg` so no environment is needed.
"""

from typing import Any

from fastapi import FastAPI

from turnstile import root


def create_app(cfg: Any = None) -> FastAPI:
    """Build the service. `cfg` is root's config object, held opaquely
    (duck-typed — same discipline as AgentSpec); None = read the env."""
    if cfg is None:
        cfg = root.load_config()

    app = FastAPI(title="turnstile", docs_url=None, redoc_url=None)
    # Process-wide state: ONE snapshot store outlives every conversation
    # (root.build_store's contract); cfg is what assemble() will consume
    # per conversation (M4-c2's registry).
    app.state.cfg = cfg
    app.state.store = root.build_store()

    @app.get("/health")
    async def health() -> dict:
        # Liveness + which product this deployment serves. Deliberately no
        # upstream probes: a slow LLM must not flap the container's health.
        return {"status": "ok", "spec": cfg.spec_id}

    return app
