"""root.assemble — the composition seam: config + spec in, one bundle out."""

import inspect

import pytest

from turnstile.capabilities.providers.openai_compat import OpenAICompatProvider
from turnstile.capabilities.tools.kb_search import KbSearchTool
from turnstile.capabilities.tools.web_search import WebSearchTool
from turnstile.config import Config
from turnstile.kernel.dtos import Message, SessionSnapshot
from turnstile.products.hooks.quality_judge import QualityJudgeHook
from turnstile.products.middleware.references import ReferenceCollector
from turnstile.root import assemble, build_store

pytestmark = pytest.mark.unit

BASE = {
    "llm": {
        "base_url": "https://ds4.example/v1",
        "model": "model-fast",
        "context_window": 128_000,
    },
    "kb": {
        "embedding_url": "https://gpu.example/embed",
        "milvus_url": "https://milvus.example/search",
        "collection": "agentassist_user_datasource",
        "expr": 'doc_id in ["hrus#e35025e3"]',
    },
}


def _cfg(**overrides) -> Config:
    return Config(_env_file=None, **{**BASE, **overrides})  # type: ignore[arg-type]


# ── config section <-> constructor parity ─────────────────────────────
# assemble() splats sections into constructors (**section.model_dump()), which
# silently makes each section's field set the constructor's signature. These
# pin that contract in BOTH directions: a config field the constructor doesn't
# take would TypeError at boot; a constructor param missing from config would
# silently fall back to its default (worse — nothing would ever set it).


@pytest.mark.parametrize(
    ("section_type", "constructor"),
    [
        pytest.param(type(_cfg().llm), OpenAICompatProvider, id="llm->provider"),
        pytest.param(type(_cfg().kb), KbSearchTool, id="kb->kb_search"),
    ],
)
def test_splatted_sections_mirror_their_constructors(section_type, constructor):
    config_fields = set(section_type.model_fields)
    params = set(inspect.signature(constructor.__init__).parameters) - {"self", "client"}
    assert config_fields == params, (
        f"{section_type.__name__} and {constructor.__name__} drifted apart: "
        f"config-only={config_fields - params or '{}'} "
        f"constructor-only={params - config_fields or '{}'}"
    )


# ── composition ────────────────────────────────────────────────────────


def test_bundle_carries_the_agent_and_its_collectors():
    bundle = assemble(_cfg())
    assert isinstance(bundle.references, ReferenceCollector)
    assert bundle.references in bundle.agent.middleware  # wired, not just returned
    assert isinstance(bundle.agent.provider, OpenAICompatProvider)
    assert bundle.agent.provider.model_name() == "model-fast"
    assert bundle.agent.provider.context_window() == 128_000
    assert bundle.agent.persona.startswith("You are an internal HR & benefits assistant")
    assert bundle.agent.chat_options.temperature == 0.0  # the spec's knob
    assert bundle.agent.keep_interrupted_context is True  # chatbot cancel semantics
    assert bundle.agent.stream_timeout == 5.0  # liveness: silent streams reconnect


def test_every_spec_mount_resolves_in_the_tool_registry():
    # Cross-registry guardrail: a spec (products registry) naming a tool the
    # L1 registry has no recipe for would only explode at boot; catch it here.
    from turnstile.capabilities.tools import RECIPES
    from turnstile.products.specs import SPECS

    cfg = _cfg()  # web enabled: the widest mount every spec can ask for
    for spec_type in SPECS.values():
        spec = spec_type()
        unresolvable = set(spec.mount_names(cfg)) - set(RECIPES)
        assert not unresolvable, f"{spec.id()}: no recipe for {unresolvable}"


def test_spec_mounts_are_resolved_from_the_catalog():
    tools = assemble(_cfg()).agent.tools
    assert sorted(tools) == ["kb_search", "web_search"]
    assert isinstance(tools["kb_search"], KbSearchTool)
    assert isinstance(tools["web_search"], WebSearchTool)


def test_config_reaches_the_tool_that_needs_it():
    # the opaque scope filter travels config -> root -> tool untouched
    kb = assemble(_cfg()).agent.tools["kb_search"]
    assert isinstance(kb, KbSearchTool)
    assert kb._expr == 'doc_id in ["hrus#e35025e3"]'
    assert kb._collection == "agentassist_user_datasource"


def test_web_kill_switch_drops_the_tool_from_the_mount():
    bundle = assemble(_cfg(web_enabled=False))
    assert sorted(bundle.agent.tools) == ["kb_search"]


def test_unknown_spec_id_fails_at_boot():
    with pytest.raises(ValueError, match="unknown spec_id 'nope'"):
        assemble(_cfg(spec_id="nope"))


# ── the judge is optional ──────────────────────────────────────────────


def test_no_judge_backend_means_no_judge():
    bundle = assemble(_cfg())
    assert bundle.judge is None
    assert not any(isinstance(h, QualityJudgeHook) for h in bundle.agent.hooks)


def test_judge_backend_builds_and_registers_the_hook():
    bundle = assemble(
        _cfg(
            judge_llm={
                "base_url": "https://cheap.example/v1",
                "model": "model-small",
                "request_timeout": 5.0,
            },
            judge={"threshold": 0.4, "max_retries": 2},
        )
    )
    assert isinstance(bundle.judge, QualityJudgeHook)
    assert bundle.judge in bundle.agent.hooks
    assert bundle.judge._threshold == 0.4 and bundle.judge._max_retries == 2
    assert bundle.judge._timeout == 5.0  # one knob bounds socket AND whole grade
    assert bundle.judge._provider.model_name() == "model-small"  # its own backend
    assert bundle.agent.provider.model_name() == "model-fast"  # chat backend untouched


def test_disabled_judge_is_not_assembled():
    bundle = assemble(
        _cfg(
            judge_llm={"base_url": "https://cheap.example/v1", "model": "model-small"},
            judge={"enabled": False},
        )
    )
    assert bundle.judge is None  # the ops toggle, settings kept


# ── session store: persistence + resume ────────────────────────────────


def test_persistence_hook_is_registered_last():
    hooks = assemble(_cfg()).agent.hooks
    assert type(hooks[-1]).__name__ == "_SnapshotHook"  # snapshots what earlier hooks did


def test_fresh_session_has_nothing_to_resume():
    bundle = assemble(_cfg(), session_id="s1")
    assert bundle.agent.resume is None
    assert bundle.agent.session_id == "s1"


def test_assemble_resumes_from_a_stored_snapshot():
    store = build_store()
    snapshot = SessionSnapshot(version=1, messages=[Message.user("q1")], turn_counter=1)
    store.save("s1", snapshot)

    resumed = assemble(_cfg(), session_id="s1", store=store)
    other = assemble(_cfg(), session_id="s2", store=store)

    assert resumed.agent.resume is snapshot
    assert other.agent.resume is None  # sessions stay isolated
    assert resumed.store is store  # the shared store, not a fresh one
