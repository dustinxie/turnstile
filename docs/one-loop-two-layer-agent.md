# Design Doc: One-Loop, Two-Layer Agent Architecture (L0 / L1 / L2)

**Status:** proposal
**Author context:** restructuring our chatbot as a clean layered stack

> **The name:** *one loop* (the single orchestration engine at L0) driving *two layers*
> of pluggable code above it — L1 capabilities (craft the hammer) and L2 products (what we ship).

## 1. Motivation

Three goals drive this restructure:

1. **Less latency** — a streaming-first loop with well-defined seams lets us optimize
   the hot path (stream tokens as they arrive, run parallel-safe tools concurrently,
   cache stable prompt prefixes) without disturbing product logic. Today these
   optimizations are tangled into the monolith, so each one risks everything.
2. **More flexible reaction** — give the model more direct access to the query so
   behavior emerges from the model plus pluggable tools/hooks. New behavior comes from
   adding a capability or a hook (mid-turn continuation, dynamic tool mounting,
   per-product discipline).
3. **Better session management** — persistence, resume, and context compaction become
   first-class L0 concerns (`Conversation` / `SessionSnapshot` DTOs + the
   `CompactionCheckpoint` port) with swappable L1 backends, instead of ad-hoc state
   scattered through the code.

The restructure separates them into three layers with a strict one-way dependency rule:
a `kernel → capabilities → products` structure. Each goal maps to a layer boundary —
latency work lives in the L0 loop + L1 capabilities, new reactions are L1/L2 plug-ins,
and session management is an L0 port with L1 implementations.

## 2. The three layers

```
        ┌──────────────────────────────────────────────────┐
  L2    │  Product assembly — the chatbot(s)               │  "what we ship"
        │  picks + composes L1 components, sets persona    │
        └───────────────────┬──────────────────────────────┘
                            │ depends on ▼
        ┌───────────────────┴──────────────────────────────┐
  L1    │  Capabilities — concrete implementations         │  "craft the hammer"
        │  marshal real I/O (LLM API, files, net, DB)      │
        │  into/out of L0 types                            │
        └───────────────────┬──────────────────────────────┘
                            │ depends on ▼
        ┌───────────────────┴──────────────────────────────┐
  L0    │  Kernel — the protocol + the engine              │  "the contract + engine"
        │  interfaces (ports) + dataclasses (DTOs) + loop  │
        └──────────────────────────────────────────────────┘
```

**Dependency rule (the invariant that makes this work):** arrows point **down only**.
L2 depends on L1 and L0; L1 depends on L0; **L0 depends on nothing**. No layer ever
imports a layer above it. No sideways edge between two L2 products. Violating this is
the one thing code review must reject.

### L0 — Kernel: protocol + engine

Four things:

1. **Ports (interfaces)** — the behavioral contract: method signatures with no concrete
   implementation (or default no-ops). In Python an interface is an `abc.ABC` with
   `@abstractmethod` (nominal, enforced at instantiation), or a `typing.Protocol` for
   structural "duck-typed" contracts. Example ports: `Tool`, `LlmProvider`,
   `LifecycleHooks`, `ToolMiddleware`, `CompactionStrategy`, `CompactionCheckpoint`,
   `Clock`.
2. **Data types (dataclasses)** — the structural contract; the shared vocabulary every
   layer speaks (Python `@dataclass`es / DTOs). Examples: `Message`, `ToolResult`,
   `ToolDef`, `StreamEvent`, `ChatOptions`, `Conversation`, `SessionSnapshot`,
   `StopReason`.
3. **The one loop** — the concrete orchestration engine, written **entirely against
   the ports and DTOs**, never against a concrete type. It fires hooks, dispatches
   tools, drives the streaming turn/round state machine, handles continuation and
   compaction. In a mature implementation this is a single engine module of several
   thousand lines. (Mechanics: `kernel-loop-structure.md` Part II.)
4. **The driver protocol** — the serializable `AgentCommand` / `AgentEvent` DTO pair
   plus the session handle: how any driver (TUI, web layer, batch runner) talks to
   the running loop. Part of the L0 contract, same freeze discipline as the ports
   (§5).

Key insight: **L0 is not "just interfaces."** It owns the loop. It's a *microkernel* —
a small privileged core that orchestrates, with all concrete capability delegated
through the ports. The loop calls `Tool` / `LlmProvider` objects through dynamic
dispatch (duck typing) and has zero knowledge of HTTP, filesystems, or which model is
behind the provider.

Two flavors of port:
- **Required** (`@abstractmethod` — `Tool.execute`, `LlmProvider.chat_stream`): no
  default body — L1 *must* implement, or instantiation fails.
- **Default no-op** (`LifecycleHooks` methods): default `pass` bodies, so a hook that
  overrides nothing is a genuine no-op — which is why hooks are optional.

### L1 — Capabilities: make the contract real

L1 implements the L0 ports. Its specific job is **marshaling**: turn real-world I/O
into L0 DTOs and back.

- An `LlmProvider` capability parses an SSE/HTTP response stream → emits `StreamEvent`s.
- A `Tool` capability runs a filesystem read / DB query / API call → packs a `ToolResult`.

L1 is where all the real-world mess lives (retries, TLS, auth, rate limits, parsing).
The loop upstream never sees any of it — it sees only the clean DTOs. (In hexagonal-
architecture terms, L1 capabilities are the *adapters* and the L0 interfaces are the *ports*.)

### L2 — Product assembly: the chatbot(s)

L2 is what we ship. It **assembles**: picks which L1 components to mount, sets the
persona/system prompt, chooses the model, wires the discipline hooks. Each distinct
product/agent is one L2 unit.

Example: each assistant product (e.g. a support bot, a sales bot) is one L2 — all
compose the same L1 tools + providers differently, each with its own persona and
mounted toolset. They are **siblings**; none depends on another.

## 3. The composition seam (how L2 stays decoupled)

To keep L2 products from depending on each other and to keep the loop neutral, **inject
a spec** instead of branching on a hardcoded enum. The pattern (`AgentSpec`):

```python
from abc import ABC, abstractmethod


class AgentSpec(ABC):
    @abstractmethod
    def id(self) -> str: ...
    @abstractmethod
    def persona(self, cfg) -> str: ...  # system prompt
    @abstractmethod
    def mount_names(self, cfg) -> list[str]: ...  # which L1 tools to expose
    @abstractmethod
    def extra_tools(self, cfg) -> list[Tool]: ...  # product-specific tools
    @abstractmethod
    def hooks(self, cfg) -> list[LifecycleHooks]: ...  # product discipline

    def allow_web_search(self, cfg) -> bool:  # capability toggle (default)
        return True
```

Each L2 product subclasses `AgentSpec`. The **application root** (the entry point, the
one place allowed to know all products) constructs the right spec from config and
injects it into the neutral L0 assembly. Result: the loop and assembly name **no**
product; products are pure plug-ins.

For our chatbot: each "mode" or "assistant persona" becomes one `AgentSpec` subclass —
same engine, same capabilities, different composition.

**Registries live with what they register; root only welds.** The spec registry
(`SPECS`, keyed by `AgentSpec.id()` — what config selects) lives in the specs
package; the tool registry (`RECIPES`, one construction per tool, taking the
deployment cfg duck-typed and converting to plain scalars at that boundary)
lives in the tools package. Root imports both and welds: the chosen spec's
`mount_names` pick recipes, and only what the spec named gets constructed.
Adding an agent or a tool never edits root — each is one module plus one
registry line in its own package. An unknown name fails at boot.

`mount_names` is **per-product static** — per-query dynamic mounting decided by code
is the router reborn inside the assembly root; if dynamic tool selection is wanted,
expose a meta-tool and let the model choose. *Runtime* remounting between turns is
still supported at the kernel level: tools live in a registry whose mounted set is an
immutable, revision-stamped snapshot taken once per turn, with a separate publisher
handle that atomically replaces it — so a background capability reconciler can change
the catalog, and a turn never observes a half-updated tool set (the defs advertised
to the model and the executable set are always the same snapshot).

## 4. Rules / anti-patterns (enforce in review)

- **Classify every new component first.** Before writing code, state whether it is an
  **L1 capability** (a concrete tool/provider/backend) or an **L2 product** (a
  composition + persona), and whether it requires **any L0 change** (a new port method
  or DTO field). Default answer for L0 is *no*: an L0 change alters the contract every
  layer depends on, so it needs explicit justification and the highest review scrutiny.
  Most new work is L1 (add a capability) or L2 (add a spec) and touches L0 zero.
- **Review ladder** (heaviest to free):
  1. **L0 contract — full design review.** ANY add / delete / change of an L0 port or
     seam mechanism, not just DTOs: a new port class, a removed one, a hook/middleware
     method added or dropped, firing order, gate-fold rules, DTO shape.
  2. **New L1 capability — classification review.** Cheap but deliberate: state the
     capability, confirm it is product-agnostic, confirm zero L0 change rides along.
  3. **L2 product change — normal review** inside the product.
  4. **L2-local dataclass — free** (rule below).
- **L2-local dataclass.** A dataclass private to one L2 module is FREE —
  add/change/delete at will. Zero contract surface: no kernel type mentions it, no
  port returns it, no sibling product sees it; its only readers are the product itself
  and the root/driver via the assembled bundle (the L2-collector pattern — a judge's
  verdict, a citations list). The line not to cross: the moment a kernel port or another
  layer would need to name the type, it's an L0 proposal — review it as one. And never
  smuggle such data through existing contract DTOs to avoid the class.
- **No upward imports.** L0 must not name L1; L1 must not name any L2. If assembly code
  needs a product type, that's a signal to move it behind a port or up to the binary
  root.
- **No sideways L2→L2 edge.** Products are siblings composed at the root, never
  dependent on each other. (A product depending on another product is the edge to
  design out — compose them at the root instead.)
- **The loop touches only ports + DTOs.** Any `if provider == "openai"` in the loop is a
  layering violation — push it into a capability.
- **Hooks and middleware may gate, redact, and observe — they must never do strategy.**
  No rewording the query, no picking sources, no judging sufficiency; strategy belongs
  to the model. A hook doing strategy is a layering violation even when the imports are
  clean — this is exactly how the old pipeline thicket regrows (review R1).
- **Tools are trusted code.** The kernel does not sandbox: mounting a tool grants its
  `execute` the host process's full ambient authority. OS-level isolation (containers,
  seccomp, a separate process) is the embedder's or an L1 capability's job. The
  kernel's one built-in bound is the tool-result size cap (§II.7). A failing tool
  returns `ToolResult(is_error=True)`; the loop additionally catches raised exceptions
  into error results (errors must-not-panic contract).
- **Product-specific config stays out of L0/L1.** Each product's config is owned by its
  L2 spec (constructed at the root), not baked into shared types. Where a shared config
  type must cross into a capability, convert it at the boundary (a mirror-type /
  boundary conversion) so the capability never depends on the config layer.

## 5. The driver protocol (how anything outside talks to the loop)

The loop's public face is a **bidirectional, serializable Command/Event handle** — not
a bare function call. The driver (TUI, web layer, batch runner) holds two queues; the
kernel owns everything between them.

- **Spawn**: assembling an agent yields a handle `{commands, events}` plus the running
  session task. The driver sends `AgentCommand`s and consumes `AgentEvent`s. Both are
  plain DTOs (JSON-serializable), so the SAME protocol works in-process and across a
  process/network boundary — step 5's FastAPI layer serializes the event stream onto
  SSE/WebSocket verbatim instead of inventing a second schema.
- **Commands**: `SendMessage{text, images}` · `SendSyntheticMessage{text}`
  (host-injected continuation; enters history as a synthetic user message) ·
  `Respond{id, value}` · `Snapshot` · `Compact{focus}` · `Cancel` · `Shutdown`.
- **Events** (the driver's whole perception): `TurnStarted`, `TextDelta`, `Reasoning`,
  `ToolCallStreaming`, `ToolBatchStarted/Completed`, `ToolStarted`, `ToolProgress`,
  `ToolResult`, `Usage`, `Request`, `Snapshot`, `Steered`, `Warning`, `Error`,
  `RateLimited`, `Cancelled`, `CompactionStarted/Compacted/CompactionFailed`, and the
  terminal `TurnComplete{reason}`.
- **Request/Respond round-trip**: a middleware or tool can ask the driver a structured
  question (`Request{id, kind, payload}`) and await the matching `Respond{id, value}` —
  id-correlated, so it works identically in-process and over the wire. This is where
  approval prompts live. A configurable `request_timeout` degrades an unanswered
  request to null, and the awaiting middleware proceeds FAIL-CLOSED (an approval sees
  null → deny) instead of parking the turn forever; `Cancel` flushes all pending
  requests to null the same way.
- **Failure perception**: every turn ends with exactly one `TurnComplete{reason}`
  (`StopReason`), and the one-shot adapter (`run_to_completion`) aggregates
  `Outcome{text, tool_results, stop, error, http_status, error_code}` — a failed turn
  can never masquerade as an empty success.

Python mapping: two `asyncio.Queue`s (or an async iterator over events); the
command/event types are the same dataclass vocabulary as the rest of L0.

## 6. Worked mapping — our chatbot onto the layers

| Concern in today's chatbot | Lands in | Form |
|--|--|--|
| turn/message loop, tool-call dispatch | **L0** | the one engine + `LlmProvider`/`Tool` ports + `Message`/`ToolResult` DTOs |
| OpenAI/Anthropic/vendor HTTP client | **L1** | `LlmProvider` capability marshaling HTTP ↔ `StreamEvent` |
| file/KB/DB/search tools | **L1** | `Tool` capabilities marshaling I/O ↔ `ToolResult` |
| retrieval / RAG source access | **L1** | `Tool` (or provider-side) capability |
| system prompt, persona, guardrails | **L2** | `AgentSpec.persona` + `hooks` |
| "which tools does this bot get" | **L2** | `AgentSpec.mount_names` / `extra_tools` |
| product variants (support bot vs sales bot) | **L2** | one `AgentSpec` subclass each, siblings |
| session persistence | **L1** (impl) via **L0** port | `CompactionCheckpoint` |
| answer-quality gate ("is it actually done?") | **L2** hook via **L0** seam | `offer_continuation` — Some→continue (retry with critique), None→accept; final verdict kept on the judge hook's own state |
| API response envelope (signal / confidence / references) | **L2** web layer | reads its own L2 collector objects (judge verdict, citations middleware) after `TurnComplete` — the L2-collector pattern (`kernel-loop-structure.md` §I.2); absent verdict → degraded default |
| tool approval / RBAC gate | **L2** middleware via **L0** seam | `ToolMiddleware.before` → Proceed/Allow/Ask/Deny/DenyTurn; approval prompt rides the Request/Respond round-trip (§5) |
| rate-limit policy (wait vs pause) | **L2** hook via **L0** seam | `on_rate_limit` → `RateLimitDecision`; the kernel owns the cancellable wait + livelock fuse (§II.5) |

## 7. Why this is worth the restructure

1. **Testability** — the loop tests against mock `Tool`/`LlmProvider`/`Clock` at the L0
   seams; no network, no real model, deterministic time. L1 capabilities test against real
   I/O in isolation. (The `Clock` port covers timestamp stamping only — see `kernel-loop-structure.md` §II.8 for
   the exact determinism scope.)
2. **Provider/vendor swaps are L1-only** — a new LLM vendor is one new `LlmProvider`
   capability; the loop and every product are untouched.
3. **New product = one L2 spec** — no forking the engine, no touching I/O.
4. **Blast radius is bounded** — a bug in a tool capability can't corrupt the loop; a
   product's persona can't leak into another product.
5. **One vocabulary** — the L0 DTOs are the lingua franca across every layer, so there's
   no translation/versioning drift between components.
6. **Curbs bloat under AI-assisted coding** — every changelist declares which layer it
   touches, so its blast radius is visible up front. A diff that sprawls across layers,
   or reaches into L0 without cause, is an immediate review flag rather than something
   spotted after the fact. The layer boundaries turn "does this change belong here?"
   into a mechanical check, keeping AI-generated changes small and on-target instead of
   accreting incidental logic.

## 8. Plan for restructuring

Build **bottom-up** — each layer is fully testable before the next exists. Step 1 starts
from the **Appendix** inventory.

1. **Define the L0 contract** — the ports (interfaces) + DTOs, starting from the Appendix
   list. This is the hardest layer to change later, so get the **streaming shape**
   (`StreamEvent` variants) and the DTO fields right first. *Deliverable:* interfaces +
   dataclasses that type-check, no implementations yet.
2. **Implement the one loop (L0 engine)** — orchestration written entirely against the
   step-1 ports/DTOs: turn/round state machine, tool dispatch, hook firing, streaming,
   continuation, compaction. Test against **mock ports** (a fake `LlmProvider` emitting
   scripted `StreamEvent`s, fake `Tool`s, a fake `Clock`) — no network, no model.
   *Deliverable:* a unit-tested engine that runs a full turn on mocks.
3. **Define the L1 components** — the real capabilities implementing the ports: the
   `LlmProvider` adapter(s) (vendor HTTP ↔ `StreamEvent`), the `Tool` set (files, KB/DB,
   search, retrieval), and the persistence backend behind `CompactionCheckpoint`. Each
   tested in isolation against real I/O. *Deliverable:* one working capability per port.
4. **Define the L2 assembly** — the `AgentSpec` interface + one subclass per product
   (which L1 tools to mount, persona, model, hooks, capability toggles), plus the
   **application-root wiring**: read config → construct the chosen spec → inject into the
   L0 assembly. *Deliverable:* a runnable agent selected by config.
5. **Chatbot service — L2 behind a FastAPI server** — wrap the assembled agent in the web
   layer: **AuthN** (who the user is), **AuthZ** (which products / tools / data they may
   use), and **streaming transport** (SSE or WebSocket carrying `StreamEvent`s to the
   client). Map one HTTP session ↔ one agent `Conversation`. *Deliverable:* a deployable
   endpoint streaming a live conversation.
6. **Persistence — DB schema / tables** — the concrete store behind the
   `CompactionCheckpoint` port and session management: conversations, messages,
   snapshots, per-user/session metadata for resume. (An L1 backend — the port is defined
   in step 1; this supplies the table-backed implementation.) *Deliverable:* sessions
   survive restarts and resume cleanly.

**Watch for (easy to miss):**
- **Config schema** — where product selection + provider credentials + per-product
  settings live; consumed at the application root in step 4.
- **Observability** — telemetry/logging fits as a `ToolMiddleware` or `LifecycleHooks`
  hook, so it's added without touching the loop.
- **Rate-limit / error policy** — already an L0 seam (`on_rate_limit`, `on_error`);
  decide the concrete behavior when wiring L1 providers in step 3.
- **Answer-quality contract (review R2) is L2 composition, not L0 surface** — the
  response envelope's structured data (quality signal, citations) is delivered by the
  L2-collector pattern (`kernel-loop-structure.md` §I.2): stateful hook/middleware objects the web layer
  composes and reads after `TurnComplete`. The signal stays structured and is
  collected as the data flows past, never re-mined at stop time. Collectors are
  step-4/5 work; L0 needs nothing for this.

## Appendix — reference L0 port + DTO inventory

**Ports (interfaces)** — `@abstractmethod` = required (L1 must implement); a plain
body = default (override optional).

```python
from abc import ABC, abstractmethod
from typing import Any, Optional, AsyncIterator


class Tool(ABC):  # a callable capability (TRUSTED code — §4)
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def description(self) -> str: ...
    @abstractmethod
    def parameters_schema(self) -> dict: ...  # JSON schema shown to the model
    @abstractmethod
    async def execute(self, args: str, ctx: ToolContext) -> ToolResult: ...
    # advisory / optional defaults (arg-aware where it matters — bash rates `rm -rf`
    # RISKY and `ls` SAFE from its args):
    def risk(self, args: str) -> RiskLevel:
        return RiskLevel.SAFE

    def read_only_hint(self) -> bool:
        return False  # intrinsic "no side effects"

    def parallel_safe(self, args: str) -> bool:
        return self.read_only_hint()

    def always_grant_scope(self, args: str) -> str:
        return args

    # ^ scope string under which an "Always" approval grant is remembered. Default =
    #   the exact args (each distinct call approved on its own — right for bash); a
    #   tool whose approval is meaningfully tool-wide (edit_file…) returns a constant.


class LlmProvider(ABC):  # the model backend
    @abstractmethod
    def model_name(self) -> str: ...
    @abstractmethod
    def chat_stream(
        self, messages: list[Message], tools: list[ToolDef], options: ChatOptions
    ) -> AsyncIterator[StreamEvent]: ...
    # ^ raises ProviderError on a FAILED OPEN (auth / connect / overflow / 429 — the
    #   loop branches on e.is_context_overflow() / e.retryable / e.http_status); a
    #   stream that opened may still fail mid-flight via a StreamEvent Error item.
    def context_window(self) -> int:
        return 0  # 0 = unknown

    def bind_session_id(self, session_id: str) -> None: ...

    # ^ one-shot binding at spawn (gateway prefix-cache affinity header); default no-op


class LifecycleHooks(ABC):  # TURN-level seams — ALL default no-op
    # COMPOSITION: many hooks fan out in REGISTRATION ORDER — transforms chain (a
    # later hook sees the earlier rewrite); user_prompt_submit short-circuits on the
    # first block; offer_continuation: all observe, first non-None wins.
    async def session_start(self, convo: Conversation, resumed: bool) -> None: ...
    # ^ mutate to seed persona/context. PERMANENT. A seeding hook must early-return
    #   `if resumed` — the snapshot already carries its injection (double-seed guard).
    async def user_prompt_submit(self, text: str) -> str:
        return text

    # ^ rewrite/augment; raise PromptRejected(reason) to BLOCK → no turn runs (§II.1)
    async def turn_start(self, convo: Conversation) -> None: ...  # once per turn; PERMANENT
    async def pre_request(self, messages: list[Message], ctx: TurnCtx) -> None: ...
    # ^ mutate the per-request CLONE. EPHEMERAL; must be APPEND-ONLY at the tail —
    #   a non-append projection warns (prefix-cache poison, §II.2)
    async def pre_request_options(
        self, messages: list[Message], options: ChatOptions, ctx: TurnCtx
    ) -> None: ...
    # ^ mutate this call's ChatOptions copy (EPHEMERAL sideband)
    async def on_request(self, messages, tools, options, ctx: TurnCtx) -> None: ...
    # ^ READ-ONLY observation of the exact final wire — telemetry / cache-RCA home
    async def on_text_delta(self, delta: str) -> str:
        return delta

    async def on_reasoning_delta(self, delta: str) -> str:
        return delta

    # ^ transform each chunk BEFORE emit; post-hook bytes reach BOTH the live stream
    #   and storage (consistent redaction). Return "" to suppress the chunk.
    #   Cross-chunk redaction (secret split across deltas) is the hook's buffering job.
    async def on_model_response(self, response: Message) -> None: ...
    # ^ transform the STORED message — incl. dropping/rewriting tool_calls, which the
    #   loop HONORS (a dropped call never executes). Post-stream: live bytes already went.
    async def offer_continuation(self, convo: Conversation) -> Optional[str]:
        return None

    # ^ Some(text) → inject synthetic user msg + continue; None → accept the stop.
    #   The quality-retry seam: a judge hook critiques the draft answer and returns
    #   the critique to re-run (bounded by MAX_CONTINUATIONS); its final verdict
    #   lives on the hook's OWN state — the L2-collector pattern (`kernel-loop-structure.md` §I.2).
    async def offer_typed_continuation(self, convo: Conversation) -> Optional["Continuation"]:
        text = await self.offer_continuation(convo)
        return Continuation(text) if text is not None else None

    # ^ richer variant: kind + visibility (INTERNAL_CONTROL rounds stream nothing and
    #   store blank text with internal_origin set — control chatter stays invisible)
    async def turn_complete(
        self, convo: Conversation, reason: StopReason, ctx: TurnCtx
    ) -> None: ...
    # ^ EXACTLY ONCE per turn on EVERY terminal path (not on a blocked prompt — no turn ran)
    async def on_error(self, error: str) -> None: ...  # pure observation
    async def on_rate_limit(self, hint: "RateLimitHint") -> Optional["RateLimitDecision"]:
        return None

    # ^ None = no opinion → kernel falls back to a conservative hint-derived decision
    async def session_end(self, convo: Conversation) -> None: ...


class ToolMiddleware(ABC):  # TOOL-level seams; registration order load-bearing
    async def before(self, call: ToolCall, tool: Tool, rt: "RequestCtx") -> "BeforeOutcome":
        return BeforeOutcome.PROCEED

    # ^ may REWRITE the call in place (args; a name rewrite does not re-route) and
    #   round-trip the driver via rt.request(kind, payload) (approval prompts, §5).
    #   Gate fold across the chain: first DENY / DENY_TURN blocks; ALLOW short-circuits
    #   the remaining gates; ASK defers to a downstream approval middleware (the kernel
    #   owns no prompt).
    async def after(self, result: ToolResult) -> "AfterOutcome":
        return AfterOutcome.PROCEED

    # ^ transform/observe the RAW (pre-size-cap) result in place; BLOCK(reason) feeds
    #   the reason back to the model as a synthetic user message. Also the natural
    #   COLLECTION POINT for the response envelope: a citations middleware appends
    #   references to ITS OWN state as results flow past (the L2-collector pattern,
    #   §I.2) — citations exist at the moment a search result passes through here;
    #   a stop-time hook re-mining the conversation is lossy after truncation/caps.


class CompactionStrategy(ABC):  # PLAN-ONLY policy; the kernel applies (§II.7)
    @abstractmethod
    async def plan(self, view: "CompactionView") -> "CompactionPlan": ...
    def will_summarize(self, view: "CompactionView") -> bool:
        return False

    # ^ cheap, side-effect-free pre-check (NO LLM call): will plan() do slow summary
    #   work? Gates the "compacting…" progress event so a no-op never shows one.


class CompactionCheckpoint(ABC):  # durable writer gating committed manual compaction
    @abstractmethod
    def save(self, snapshot: SessionSnapshot) -> None: ...

    # ^ raise CompactionCheckpointError on failure → the prepared plan is NOT committed
    #   (live history and epoch unchanged)


class Clock(ABC):  # timestamp STAMPING only — see §II.8
    @abstractmethod
    def now_millis(self) -> int: ...


class RequestCtx:  # kernel-PROVIDED broker (concrete, not a port)
    def emit(self, event: "AgentEvent") -> None: ...
    async def request(self, kind: str, payload: dict) -> Any: ...

    # ^ emits Request{id, kind, payload}, awaits the matching Respond{id, value}.
    #   Unanswered (request_timeout / cancel / dead driver) degrades to None so the
    #   awaiting middleware proceeds FAIL-CLOSED (an approval sees None → deny).
```

**Data types (dataclasses)** — field names + semantics proven in the reference
kernel; trim/extend for our needs:

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── enums ──────────────────────────────────────────────────────────────
class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class RiskLevel(Enum):  # advisory only — the loop knows risk, never enforces
    SAFE = "safe"
    RISKY = "risky"


class StopReason(Enum):  # how a turn terminated — exactly one TurnComplete per turn
    STOPPED = "stopped"  # normal: no tool calls, no continuation
    MAX_ROUNDS = "max_rounds"  # per-turn round fuse tripped
    MAX_CONTINUATIONS = "max_continuations"  # runaway offer_continuation fuse
    REPEAT_LOOP = "repeat_loop"  # coarse fuse: same call pattern, 6 rounds (§II.6)
    TOOL_LOOP_DETECTED = "tool_loop_detected"  # opt-in exact no-progress guard (§II.6)
    PROVIDER_ERROR = "provider_error"  # failed open / mid-stream / empty-retry exhausted
    TIMEOUT = "timeout"  # stream-liveness timeout, reconnects exhausted
    CANCELLED = "cancelled"
    PROMPT_REJECTED = "prompt_rejected"  # user_prompt_submit blocked — no turn ran
    POLICY_DENIED = "policy_denied"  # middleware DENY_TURN; results stayed paired
    RATE_LIMITED = "rate_limited"  # 429 pause — NOT a failure; content preserved


class PromptRejected(Exception):  # raised by a user_prompt_submit hook to block the prompt
    pass


# ── middleware gate DTOs (§II.3) ────────────────────────────────────────
class Gate(Enum):
    PROCEED = "proceed"  # continue the chain + normal approval flow
    ALLOW = "allow"  # force-approve: bypass remaining gates, no prompt
    ASK = "ask"  # defer to a downstream approval middleware
    DENY = "deny"  # block THIS call; reason → model + driver
    DENY_TURN = "deny_turn"  # block + terminate the turn once the batch is paired


@dataclass
class BeforeOutcome:
    gate: Gate = Gate.PROCEED
    reason: str = ""


BeforeOutcome.PROCEED = BeforeOutcome()


@dataclass
class AfterOutcome:  # PROCEED, or BLOCK: reason fed back to the model
    block_reason: Optional[str] = None


AfterOutcome.PROCEED = AfterOutcome()


# ── tool DTOs ──────────────────────────────────────────────────────────
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON-arguments string from the model


@dataclass
class ToolDef:  # what the model sees for a mounted tool
    name: str
    description: str
    parameters: dict  # JSON schema


@dataclass
class ImageContent:  # neutral inline image; the L1 adapter owns the wire shape
    media_type: str  # e.g. "image/png"
    data: str  # base64 bytes


@dataclass
class ToolResult:
    call_id: str
    content: str  # size-capped by the kernel before store/emit (§II.7)
    is_error: bool = False
    images: list[ImageContent] = field(default_factory=list)
    # ^ TRANSIENT carrier: the loop lifts these onto ONE synthetic user message after
    #   the batch (providers reject images on the tool role); never stored on the
    #   tool-result message itself.


@dataclass
class ToolContext:  # handed to Tool.execute()
    working_dir: str  # per-agent pin or shared mutable handle; never process chdir
    cancel: "CancellationToken"  # cooperative; the kernel also drops the future as backstop
    progress: "ProgressSink"  # emit(msg) → ToolProgress event; no-op default
    requester: Optional["Requester"] = None  # ask the driver a structured question (§5)


# ── content sidecars ───────────────────────────────────────────────────
@dataclass
class ReasoningBlock:  # one SIGNED thinking unit (Anthropic/OpenAI/Gemini opaque)
    text: str  # may be empty (redacted block)
    opaque: Optional[str] = None  # round-trip token, echoed VERBATIM, never re-encoded
    provider: Optional[str] = None
    # ^ INVARIANT: opaque set ⇒ provider set — a token is PROVIDER-BOUND; an adapter
    #   echoes it only to its own backend. Plain-text reasoning paths never make these.


@dataclass
class TokenUsage:  # provider usage for one LLM call
    prompt: int = 0
    completion: int = 0
    cached: int = 0

    def merge_max(self, other: "TokenUsage") -> None:
        # Field-wise MAX fold across a round's (possibly multiple) Usage events —
        # correct for both one cumulative report (OpenAI) and split / cumulative-delta
        # reports (Anthropic): never double-counts, never drops an early-only field.
        self.prompt = max(self.prompt, other.prompt)
        self.completion = max(self.completion, other.completion)
        self.cached = max(self.cached, other.cached)


# ── message DTOs ───────────────────────────────────────────────────────
@dataclass
class MessageMeta:  # execution-stats sidecar; never rendered into text
    tokens: TokenUsage
    elapsed_ms: int
    reasoning_elapsed_ms: int = 0  # thinking-phase duration
    ctx_window: int = 0
    used_tokens: int = 0  # provider's prompt count, or byte-estimate fallback
    utilization: float = 0.0
    round: int = 0
    turn_id: int = 0  # correlation: which user turn produced this
    request_id: int = 0  # correlation: which LLM request (session-global)
    provider_response_id: Optional[str] = None  # upstream handle for log cross-referencing
    provider_model: Optional[str] = None  # reported model — gateway-misroute detection
    session_id: Optional[str] = None
    finish_reason: str = ""  # "stop" | "tool_calls" | "length"


@dataclass
class Message:  # provider-neutral, losslessly persistable
    role: Role
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    is_error: bool = False  # true iff this is a FAILED tool result
    meta: Optional[MessageMeta] = None
    synthetic: bool = False  # kernel-injected (summary / resume note / nudge)
    internal_origin: Optional[str] = None  # e.g. "verify_cadence"
    reasoning: Optional[str] = None  # flat stored thinking (plain-text path)
    reasoning_blocks: list[ReasoningBlock] = field(default_factory=list)  # signed path
    images: list[ImageContent] = field(default_factory=list)  # multimodal input


# ── streaming (a tagged union — the LlmProvider yields these) ──────────
@dataclass
class ProviderError(Exception):  # raised on a failed OPEN; carried by Error mid-stream
    message: str = ""
    retryable: bool = False  # 429/5xx/timeout may retry; False = terminal (auth/400)
    http_status: Optional[int] = None
    code: Optional[str] = None  # structured code, e.g. "context_length_exceeded"
    retry_after_secs: Optional[int] = None  # real Retry-After header on a 429

    def is_context_overflow(self) -> bool: ...

    # ^ centralized per-vendor overflow signatures so no call site string-matches


# StreamEvent = TextDelta(str) | Reasoning(str)
#             | ReasoningSignature(opaque, provider)      # finalize one signed block
#             | ToolCall(ToolCall)
#             | ToolCallDelta(index, id?, name?, arguments)  # display-only streaming frag
#             | Usage(TokenUsage)                         # may repeat; fold via merge_max
#             | ResponseId(str) | ResponseModel(str)
#             | Error(ProviderError)                      # mid-stream failure
#             | Malformed                                 # dropped unparseable chunk (diagnostic)
#             | Done(truncated: bool)                     # truncated = finish_reason "length"
# Model as a class hierarchy or `type StreamEvent = TextDelta | Reasoning | ...`.


# ── per-request / per-turn ─────────────────────────────────────────────
@dataclass
class ChatOptions:  # neutral per-call knobs; None = adapter default. SIDEBAND:
    # never part of the prefix bytes (cache-safe).
    reasoning_effort: Optional[str] = None  # low | medium | high | max
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    tool_choice: str = "auto"  # auto | required | none | <specific tool name>


class ContinuationKind(Enum):
    GENERIC = "generic"
    VERIFY_CADENCE = "verify_cadence"
    TRUNCATION_RESUME = "truncation_resume"


class ContinuationVisibility(Enum):
    NORMAL = "normal"
    INTERNAL_CONTROL = "internal_control"
    # ^ INTERNAL_CONTROL rounds stream nothing to the driver and store blank text with
    #   internal_origin set — control chatter stays out of user-visible history


@dataclass
class Continuation:  # typed offer_continuation result
    text: str
    kind: ContinuationKind = ContinuationKind.GENERIC
    visibility: ContinuationVisibility = ContinuationVisibility.NORMAL


@dataclass
class RateLimitHint:  # what the kernel knows about a 429 when it fires
    http_status: Optional[int]
    retry_after_secs: Optional[int]  # header preferred; body-text hint as fallback
    terminal: bool  # account/billing exhaustion — never auto-retry
    attempt: int  # 1-based consecutive incident this turn (kernel-owned)


@dataclass
class RateLimitDecision:  # host verdict: wait_secs set → WaitAndRetry; else Pause
    wait_secs: Optional[int] = None
    reset_at_display: str = ""  # Pause-side display facts (may be empty)
    reset_label: str = ""
    secs_until_reset: Optional[int] = None


@dataclass
class TurnCtx:  # correlation + live pressure, passed to hooks
    session_id: Optional[str]  # driver-owned identity; the kernel never mints it
    turn_id: int  # one user message → one turn (constant across rounds)
    request_id: int  # unique per LLM request (bumps every round)
    round: int  # 1-based index of the LLM call within the turn
    max_rounds: Optional[int] = None
    cache_epoch: int = 0  # prefix-generation marker
    context_window: int = 0  # from the LAST response's usage report (0 on round 1)
    used_tokens: int = 0  # ditto — pairs with context_window for live pressure
    # Per-turn structured data (quality verdict, citations) lives on the L2 collector
    # objects themselves; the driver that composed them reads them after TurnComplete
    # (the L2-collector pattern, §I.2). BEST-EFFORT on fuse exits: a judge that never
    # ran leaves verdict=None, and the envelope mapper owns the degraded default
    # (signal=no_answer) — never a fake ok.


# ── compaction (§II.7 invariants; strategy proposes, kernel disposes) ───
# CompactTrigger = Auto(utilization) | Manual(focus?) | Overflow(attempt)


@dataclass
class CompactionView:  # READ-ONLY view handed to CompactionStrategy.plan
    messages: list[Message]
    trigger: "CompactTrigger"
    ctx_window: int
    used_tokens: int
    utilization: float
    sacred_floor: int  # leading messages a plan must not drain (kernel re-clamps)


@dataclass
class CompactionPlan:  # a PROPOSAL; the kernel revalidates everything
    drain_from: int = 0  # replace messages[drain_from:drain_to] with the summary
    drain_to: int = 0
    summary: Optional[str] = None  # inserted as ONE synthetic user message
    rewrites: list[tuple[int, str]] = field(default_factory=list)
    # ^ in-place text stubs (permanent microcompact); indices are ORIGINAL positions —
    #   the kernel translates past the drain/summary shift, skips drained/sacred targets
    resume_note: Optional[str] = None  # appended as a trailing synthetic user message


@dataclass
class CompactReport:  # audit record of one attempt; committed=False = refused
    epoch_before: int
    epoch_after: int
    removed: int
    bytes_before: int
    bytes_after: int
    committed: bool


# ── session persistence ────────────────────────────────────────────────
@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)
    cache_epoch: int = 0  # bumped only by a committed compaction
    # kernel-owned helpers (invariants, not policy): sacred_floor(), repair_pairing(),
    # backfill_cancelled_tool_results(), last_pressure()


@dataclass
class SessionSnapshot:  # versioned, LOSSLESS, resumable
    version: int  # reader checks BEFORE interpreting messages
    messages: list[Message]
    cache_epoch: int = 0
    turn_counter: int = 0  # id high-water marks so resume stays monotonic
    request_counter: int = 0


# ── driver protocol DTOs (§5) ──────────────────────────────────────────
# AgentCommand = SendMessage(text, images) | SendSyntheticMessage(text)
#              | Respond(id, value) | Snapshot | Compact(focus?) | Cancel | Shutdown
#
# AgentEvent   = TurnStarted | TextDelta(str) | Reasoning(str)
#              | ToolCallStreaming(index, id?, name?, arguments)
#              | ToolBatchStarted(batch_id, calls) | ToolBatchCompleted(batch_id, ok, total, elapsed_ms)
#              | ToolStarted(call) | ToolProgress(call_id, message) | ToolResult(result)
#              | Request(id, kind, payload) | Usage(MessageMeta) | Snapshot(SessionSnapshot)
#              | Steered(count, inputs) | Warning(str) | Error(message, http_status?, code?)
#              | RateLimited(reset_at_display, reset_label, secs_until_reset?, auto_resuming, server_message?)
#              | Cancelled | CompactionStarted(trigger)
#              | Compacted(trigger, epoch, removed, bytes_before, bytes_after, committed, snapshot?)
#              | CompactionFailed(trigger, error)
#              | TurnComplete(reason: StopReason)        # the terminal — exactly one per turn


@dataclass
class Outcome:  # one-shot adapter aggregate (batch / CI drivers)
    text: str = ""
    tool_results: list[ToolResult] = field(default_factory=list)
    stop: StopReason = StopReason.STOPPED
    error: Optional[str] = None  # last Error captured; None on a clean stop
    http_status: Optional[int] = None
    error_code: Optional[str] = None
```

The single-engine loop manipulates only these DTOs and calls only these ports — zero
concrete knowledge below. That is the whole L0 contract.
