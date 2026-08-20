"""App factory — in-process ASGI tests (httpx transport, no sockets)."""

import httpx
import pytest

from turnstile.config import Config
from turnstile.root import build_store
from turnstile.service.app import create_app

pytestmark = pytest.mark.service


def _cfg(**overrides) -> Config:
    base = {
        "llm": {"base_url": "https://ds4.example/v1", "model": "model-fast"},
        "kb": {
            "embedding_url": "https://gpu.example/embed",
            "milvus_url": "https://milvus.example/search",
            "collection": "agentassist_user_datasource",
            "expr": 'doc_id in ["hrus#e35025e3"]',
        },
    }
    return Config(_env_file=None, **{**base, **overrides})  # type: ignore[arg-type]


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")


async def test_health_reports_ok_and_the_served_spec():
    app = create_app(_cfg())
    async with _client(app) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "spec": "support_bot"}


async def test_injected_cfg_is_used_verbatim():
    app = create_app(_cfg(spec_id="support_bot", web_enabled=False))
    assert app.state.cfg.web_enabled is False  # held opaquely, not rebuilt


async def test_each_app_owns_one_process_wide_store():
    app = create_app(_cfg())
    assert type(app.state.store) is type(build_store())
    assert app.state.store is app.state.store  # stable attribute, built once
    assert create_app(_cfg()).state.store is not app.state.store  # per app, not global
