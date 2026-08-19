"""Typed configuration — environment in, one frozen object out.

Root's input (architecture.md §1): every endpoint, credential and knob the
assembly needs, validated once at startup so a misconfigured deployment
fails loudly at boot instead of mid-turn. Products never import this class
— specs read attributes duck-typed (design doc §3), which is why the
product-facing toggles (`web_enabled`) live at the top level.

Env names are the field names, `__` separating a section from its field:

    LLM__BASE_URL=https://10.83.135.205/model-long/v1
    LLM__MODEL=model-fast
    JUDGE_LLM__MODEL=model-small        # the judge's backend
    JUDGE__THRESHOLD=0.2                # the judge's policy
    KB__EXPR=doc_id in ["hrus#e35025e3-..."]
    WEB_ENABLED=0

Note that `JUDGE__*` and `JUDGE_LLM__*` are different sections: the judge's
policy and the backend it grades with.

A `.env` file is read when present (developer convenience; never committed).
"""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LlmConfig(BaseModel):
    """The chat backend (OpenAI-compatible: vLLM / LiteLLM / gateway)."""

    base_url: str
    model: str
    api_key: str | None = None
    # 0 = unknown; drives the kernel's compaction utilization math.
    context_window: int = 0
    request_timeout: float = 120.0


class KbConfig(BaseModel):
    """kb_search: the embedding leg + the Milvus hybrid-search leg.

    `expr` is the opaque scope filter passed to the search service verbatim
    (e.g. 'doc_id in ["hrus#<ds_id>"]'). It is FIXED per deployment here;
    per-user identity->scope resolution arrives with M4 auth.
    """

    embedding_url: str
    dim_size: int = 4096  # must match the collection schema
    milvus_url: str
    milvus_token: str = ""  # empty = tokenless deployment, header omitted
    collection: str
    expr: str
    limit: int = 10


class JudgeConfig(BaseModel):
    """Quality-judge POLICY — when to retry, how strict. The backend it grades
    with is a separate section (`judge_llm`)."""

    enabled: bool = True
    threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    max_retries: int = Field(default=1, ge=0)


class Config(BaseSettings):
    """The whole deployment, validated at boot."""

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    # Which product root assembles (AgentSpec.id()).
    spec_id: str = "support_bot"
    # Product-facing toggle read duck-typed by AgentSpec.allow_web_search.
    web_enabled: bool = True
    # web_search backend: Exa. Unset = its keyless tier.
    web_api_key: str | None = None
    # The resume window: past it a conversation is view-only (architecture.md
    # §4). 0 = no expiry.
    session_ttl_seconds: int = 24 * 3600
    # LIVENESS bound on the CHAT stream: max seconds between stream events,
    # enforced by the kernel (a plain pass-through to Agent.stream_timeout —
    # deliberately NOT an LlmConfig field, since no provider constructor takes
    # it). The FIRST event of an attempt gets 20x this (engine
    # FIRST_EVENT_TIMEOUT_FACTOR): TTFT includes prefill, which scales with
    # context. A wedged connection becomes an idle-timeout reconnect (budget
    # 5) instead of a minutes-long stall. None = unbounded.
    stream_timeout: float | None = 5.0

    # Backends are siblings. The chat backend is required; the judge's is
    # optional and stands alone — grading an answer and producing it are
    # different jobs, so nothing is inherited between them. Point both at the
    # same URL if a deployment really does serve both from one place, and set
    # the judge's request_timeout low (~15s): it runs at the END of a turn, so
    # a stalled eval holds a finished answer.
    llm: LlmConfig
    judge_llm: LlmConfig | None = None  # absent = no judge assembled
    kb: KbConfig
    # Policy only; `enabled` is the ops toggle that keeps the settings while
    # turning grading off.
    judge: JudgeConfig = JudgeConfig()
