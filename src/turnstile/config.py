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

from pydantic import BaseModel, Field, model_validator
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


class SamlConfig(BaseModel):
    """SAML SSO (FortiAuthenticator IdP): the SP we present + the IdP we
    trust. Section absent = SSO routes not mounted. Certs/keys are PEM
    strings (env-injected; never committed)."""

    # Ourselves, the Service Provider. acs_url is SIGNED INTO assertions by
    # the IdP — it is a contract with FAC, which is why /sso lives outside
    # the /v1 API prefix (it must not move on an API version bump).
    sp_entity_id: str
    sp_acs_url: str
    sp_sls_url: str = ""
    sp_x509_cert: str = ""
    sp_private_key: str = ""
    # The IdP we trust (FAC).
    idp_entity_id: str
    idp_sso_url: str
    idp_sls_url: str = ""
    idp_x509_cert: str
    debug: bool = False
    # Where the browser lands after a successful login: the frontend origin
    # (+ optional path). The token rides the URL FRAGMENT ("#token=..."):
    # fragments never reach servers or logs, only the page's own JS.
    return_url: str = "/"


class RedisConfig(BaseModel):
    """The session store (architecture §4): snapshots + ownership live in
    Redis so conversations outlive the process. Section absent = the
    in-memory store (dev; everything dies with the process). The resume
    window is `session_ttl_seconds` (top level) — one TTL for eviction and
    expiry alike."""

    url: str  # redis://[:password@]host:port/db
    prefix: str = "turnstile"  # key namespace; several deployments may share one Redis
    # RETENTION: how long a conversation stays resumable/listed after its last
    # turn (the resume window, §4). Distinct from session_ttl_seconds, which
    # only evicts the live agent from process memory. 0 = never expire.
    ttl_seconds: int = 30 * 24 * 3600


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
    # AuthN switch: the HS256 secret our JWTs are signed with (use >=32 random
    # bytes; pyjwt warns below that). Unset = auth OFF (dev mode: every request
    # is the "anonymous" principal). Tokens come from auth.mint_token — by hand
    # (see its docstring) or via the SAML/FAC login route (service/sso.py: the
    # IdP authenticates; THIS service issues its own credential). Also the key
    # material for citation file tokens (service/files.py).
    jwt_secret: str | None = None
    # The flat authorization model: usernames (comma-separated) whose
    # SSO-minted tokens carry role=admin; everyone else is role=user.
    # Hand-minted admin tokens pass role="admin" explicitly. Not RBAC.
    admin_users: str = ""
    # Citation file store: the documents the kb collection was indexed from,
    # laid out as <file_root>/<region>/<filename> — the region tier exists
    # because one deployment serves several fixed kb scopes (us/canada/...)
    # and same-name docs must not collide. Server-side only; clients reach
    # files exclusively through opaque tokens (GET /v1/files/{token}).
    # Unset = the files endpoint is not mounted.
    file_root: str | None = None

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
    saml: SamlConfig | None = None  # absent = no SSO routes mounted
    redis: RedisConfig | None = None  # absent = in-memory session store (dev)

    @model_validator(mode="after")
    def _sso_needs_a_signing_secret(self) -> "Config":
        # SSO's whole job is minting our JWTs — configured without the
        # secret it could only fail at first login; fail at boot instead.
        if self.saml is not None and not self.jwt_secret:
            raise ValueError("saml is configured but jwt_secret is unset; SSO mints JWTs")
        return self
