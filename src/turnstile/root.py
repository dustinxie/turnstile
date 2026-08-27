"""Application root — config in, an assembled agent out.

The one module allowed to know every layer (architecture.md §1): it reads
the typed config, builds the L1 capabilities, picks the L2 spec, and hands
back a driver-neutral bundle. Web, CLI and batch drivers all call the same
`assemble()` and get the identical object.

Root holds NO registries and NO literals — SPECS lives with the specs
(products/specs), RECIPES with the tools (capabilities/tools); adding an
agent or a tool never edits this file. Root only welds: cfg.spec_id picks
the spec from SPECS, the spec's mount_names pick recipes from RECIPES, and
each recipe converts cfg to plain scalars at the boundary (design doc §4 —
no adapter ever names a config type).
"""

from dataclasses import dataclass

from turnstile.capabilities.persistence.memory_store import MemorySessionStore
from turnstile.capabilities.persistence.redis_store import RedisSessionStore
from turnstile.capabilities.providers.openai_compat import OpenAICompatProvider
from turnstile.capabilities.tools import RECIPES
from turnstile.config import Config, RedisConfig
from turnstile.kernel.engine import Agent
from turnstile.kernel.ports import LifecycleHooks, Tool, ToolMiddleware
from turnstile.products.hooks.quality_judge import QualityJudgeHook
from turnstile.products.middleware.references import ReferenceCollector

# Registries live with what they register (SPECS in specs/__init__, RECIPES
# in tools/__init__); root only welds. Adding an agent or a tool never edits
# this file.
from turnstile.products.spec import AgentSpec
from turnstile.products.specs import SPECS

# The store the driver holds: Redis when configured, else the in-memory
# stand-in — same surface (architecture §4), chosen once in build_store().
SessionStore = MemorySessionStore | RedisSessionStore


@dataclass(frozen=True)
class AssembledAgent:
    """What every driver receives. `agent` runs turns; `judge` and
    `references` are the L2 collectors read AFTER TurnComplete to build the
    response envelope (kernel-loop-structure.md §I.2); `store` holds the
    session snapshots. A driver touches product objects only through this
    bundle, never by importing product classes."""

    agent: Agent
    references: ReferenceCollector
    store: SessionStore
    judge: QualityJudgeHook | None = None


def load_config() -> Config:
    """Read the deployment config from the environment (+ optional .env).
    Lives here so DRIVERS never import the config layer (the c6 contract):
    a driver calls root.load_config() and holds the result opaquely."""
    return Config()  # type: ignore[call-arg] — fields come from the env


def build_store(redis: RedisConfig | None = None) -> SessionStore:
    """The snapshot store outlives any one conversation, so a driver builds
    it once and passes it into every assemble() (M4's registry does this).
    Redis when its section is configured (conversations outlive the process;
    retention is redis.ttl_seconds), else the in-memory stand-in. The section
    mirrors the constructor field-for-field (splat + parity test, like llm/kb).
    Reachability is checked here so a bad REDIS__URL fails at boot, not on the
    first turn."""
    if redis is not None:
        store = RedisSessionStore(**redis.model_dump())
        store.ping()
        return store
    return MemorySessionStore()


def assemble(
    cfg: Config,
    session_id: str = "default",
    store: SessionStore | None = None,
) -> AssembledAgent:
    """Build one conversation's agent. Resumes from `store` when it holds a
    snapshot for `session_id`; otherwise starts fresh."""
    spec = _pick_spec(cfg)
    store = store if store is not None else build_store()

    # Config sections mirror the constructors field-for-field, so the dump
    # splats 1:1. Still plain scalars crossing into L1 — never a config type.
    # Trade-off vs explicit kwargs: pyright cannot check **dict, so a config/
    # constructor rename surfaces as a TypeError at boot (assemble tests catch
    # it) instead of a red squiggle here.
    provider = OpenAICompatProvider(**cfg.llm.model_dump())

    tools = _mount(spec, cfg)

    references = ReferenceCollector()
    judge = _build_judge(cfg)
    hooks: list[LifecycleHooks] = [*spec.hooks(cfg)]
    if judge is not None:
        hooks.append(judge)
    # The persistence seam goes LAST: it snapshots what every earlier hook
    # already did to the turn.
    hooks.append(store.hook(session_id))
    middleware: list[ToolMiddleware] = [*spec.middleware(cfg), references]

    agent = Agent(
        provider=provider,
        tools=tools,
        persona=spec.persona(cfg),
        hooks=hooks,
        middleware=middleware,
        chat_options=spec.chat_options(cfg),
        checkpoint=store.checkpoint(session_id),
        session_id=session_id,
        resume=store.load(session_id),
        # A chatbot keeps a cancelled turn's context: losing the user's
        # message on a flaky network is worse than keeping half an answer
        # (architecture.md §2).
        keep_interrupted_context=True,
        # Liveness: a silent stream reconnects instead of stalling the turn.
        stream_timeout=cfg.stream_timeout,
    )
    return AssembledAgent(agent=agent, references=references, store=store, judge=judge)


def _pick_spec(cfg: Config) -> AgentSpec:
    spec_type = SPECS.get(cfg.spec_id)
    if spec_type is None:
        known = ", ".join(sorted(SPECS))
        raise ValueError(f"unknown spec_id {cfg.spec_id!r}; known specs: {known}")
    return spec_type()


def _mount(spec: AgentSpec, cfg: Config) -> dict[str, Tool]:
    """The spec names tools (its knowledge); the L1 registry holds the build
    recipes (its knowledge); root only welds the two — constructing exactly
    what the spec named, nothing else. An unknown name fails at boot — a
    product asking for a capability this deployment cannot build must not
    silently run without it."""
    tools: dict[str, Tool] = {}
    for name in spec.mount_names(cfg):
        recipe = RECIPES.get(name)
        if recipe is None:
            known = ", ".join(sorted(RECIPES))
            raise ValueError(f"spec {spec.id()!r} mounts unknown tool {name!r}; catalog: {known}")
        tools[name] = recipe(cfg)
    for extra in spec.extra_tools(cfg):
        tools[extra.name()] = extra
    return tools


def _build_judge(cfg: Config) -> QualityJudgeHook | None:
    """No judge backend configured (or grading switched off) = no judge. The
    envelope then reports no verdict rather than a fabricated pass."""
    if cfg.judge_llm is None or not cfg.judge.enabled:
        return None
    return QualityJudgeHook(
        provider=OpenAICompatProvider(**cfg.judge_llm.model_dump()),
        threshold=cfg.judge.threshold,
        max_retries=cfg.judge.max_retries,
        # One knob: the same number bounds the socket and the whole grade.
        timeout_seconds=cfg.judge_llm.request_timeout,
    )
