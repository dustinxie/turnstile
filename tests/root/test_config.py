"""Typed config — env round-trip, defaults, validation, the judge section."""

import pytest
from pydantic import ValidationError

from turnstile.config import Config

pytestmark = pytest.mark.unit

ENV = {
    "LLM__BASE_URL": "https://10.83.135.205/model-long/v1",
    "LLM__MODEL": "model-fast",
    "KB__EMBEDDING_URL": "https://10.83.135.206/api/v1/embedding/qwen3-embedding-8b",
    "KB__MILVUS_URL": "https://10.83.135.206/api/v2/vectordb/hybrid_search_generic",
    "KB__COLLECTION": "agentassist_user_datasource",
    "KB__EXPR": 'doc_id in ["hrus#e35025e3-58c2-4d6c-8e59-4f62277b3e6e"]',
}


def _load(monkeypatch, **overrides) -> Config:
    """Config from a clean environment: only the vars under test, and no
    .env underneath (a developer's file must not steer the suite)."""
    for key in list(ENV) + list(overrides):
        monkeypatch.delenv(key, raising=False)
    for key, value in {**ENV, **overrides}.items():
        monkeypatch.setenv(key, value)
    return Config(_env_file=None)  # type: ignore[call-arg]


# ── env round-trip ─────────────────────────────────────────────────────


def test_nested_env_vars_populate_the_sections(monkeypatch):
    cfg = _load(monkeypatch)
    assert cfg.llm.base_url.endswith("/model-long/v1")
    assert cfg.llm.model == "model-fast"
    assert cfg.kb.collection == "agentassist_user_datasource"
    # the scope filter survives verbatim, quotes and brackets included
    assert cfg.kb.expr == 'doc_id in ["hrus#e35025e3-58c2-4d6c-8e59-4f62277b3e6e"]'


def test_defaults_fill_the_unset_knobs(monkeypatch):
    cfg = _load(monkeypatch)
    assert cfg.spec_id == "support_bot"
    assert cfg.web_enabled is True  # the spec's duck-typed toggle
    assert (cfg.kb.dim_size, cfg.kb.limit, cfg.kb.milvus_token) == (4096, 10, "")
    assert cfg.web_api_key is None  # Exa keyless tier
    assert cfg.session_ttl_seconds == 24 * 3600
    assert cfg.stream_timeout == 5.0  # liveness default: silent streams reconnect
    assert cfg.llm.context_window == 0  # unknown until configured


def test_scalars_coerce_from_env_strings(monkeypatch):
    cfg = _load(
        monkeypatch,
        WEB_ENABLED="0",
        KB__LIMIT="25",
        LLM__CONTEXT_WINDOW="128000",
        JUDGE__THRESHOLD="0.5",
        JUDGE__ENABLED="0",
    )
    assert cfg.web_enabled is False  # the air-gap kill-switch, from "0"
    assert cfg.kb.limit == 25
    assert cfg.llm.context_window == 128_000
    # configured but switched off — the ops toggle keeps the settings
    assert cfg.judge.threshold == 0.5 and cfg.judge.enabled is False


# ── validation: fail loudly at boot, never mid-turn ────────────────────


def test_missing_required_endpoint_fails_to_construct(monkeypatch):
    for key in ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LLM__BASE_URL", "https://x/v1")
    monkeypatch.setenv("LLM__MODEL", "m")
    with pytest.raises(ValidationError) as excinfo:
        Config(_env_file=None)  # type: ignore[call-arg]
    assert "kb" in str(excinfo.value)  # the whole kb section is required


def test_out_of_range_threshold_is_rejected(monkeypatch):
    with pytest.raises(ValidationError):
        _load(monkeypatch, JUDGE__THRESHOLD="1.5")


# ── the judge's own backend ────────────────────────────────────────────


def test_absent_judge_backend_means_no_judge(monkeypatch):
    cfg = _load(monkeypatch)
    assert cfg.judge_llm is None  # presence of the BACKEND is the switch
    assert cfg.judge.threshold == 0.2  # policy still has its defaults


def test_judge_backend_is_a_sibling_of_the_chat_backend(monkeypatch):
    cfg = _load(
        monkeypatch,
        LLM__API_KEY="sk-main",
        JUDGE_LLM__BASE_URL="https://cheap.example/v1",
        JUDGE_LLM__MODEL="model-small",
        JUDGE_LLM__API_KEY="sk-judge",
        JUDGE_LLM__REQUEST_TIMEOUT="15",
        JUDGE__THRESHOLD="0.4",
    )
    # JUDGE_LLM__* and JUDGE__* must not bleed into each other
    assert cfg.judge_llm is not None
    assert cfg.judge_llm.base_url == "https://cheap.example/v1"
    assert cfg.judge_llm.model == "model-small"
    assert cfg.judge_llm.api_key == "sk-judge"  # its own gateway, its own key
    # one timeout knob per backend: root drives both the adapter's socket
    # timeout and the hook's wall-clock bound from it
    assert cfg.judge_llm.request_timeout == 15.0
    assert cfg.judge.threshold == 0.4  # policy section untouched by the backend
    assert (cfg.llm.model, cfg.llm.api_key) == ("model-fast", "sk-main")  # untouched


def test_partial_judge_backend_fails_loudly(monkeypatch):
    # No borrowing from the chat backend: naming a judge model without its
    # endpoint is a misconfiguration, not an inheritance.
    with pytest.raises(ValidationError) as excinfo:
        _load(monkeypatch, JUDGE_LLM__MODEL="model-small")
    assert "base_url" in str(excinfo.value)
