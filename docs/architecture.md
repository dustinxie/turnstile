# Architecture: Chatbot Service on the One-Loop Kernel

**Status:** planned
**Companions:** `one-loop-two-layer-agent.md` (L0/L1/L2 design + DTO appendix) ·
`kernel-loop-structure.md` (loop structure & mechanics)

Stack decisions this doc pins: **FastAPI + SSE** for the service, **uv** for package
management, **alembic** for DB migration, and a **regression-test-guarded** commit
workflow.

Assumptions: Postgres behind SQLAlchemy/alembic; LLM = self-hosted vLLM behind an
OpenAI-compatible endpoint (optionally via LiteLLM gateway); Python 3.12.

## 1. Repo layout (uv project, layers = packages)

```
turnstile/
├── pyproject.toml            # uv-managed; py3.12 pinned
├── uv.lock                   # committed
├── alembic.ini
├── migrations/               # alembic env.py + versions/
├── src/turnstile/
│   ├── kernel/               # L0 — imports NOTHING outside stdlib
│   │   ├── dtos.py           #   the appendix inventory
│   │   ├── ports.py
│   │   ├── events.py         #   AgentCommand / AgentEvent / Outcome
│   │   ├── engine.py         #   the one loop (kernel-loop-structure.md Part II)
│   │   └── testkit.py        #   ScriptedProvider, FakeClock, recorder hooks —
│   │                         #   shipped, so L1/L2 tests reuse the same doubles
│   ├── capabilities/         # L1 — implements ports, real I/O
│   │   ├── providers/        #   openai_compat.py (vLLM/LiteLLM, SSE ↔ StreamEvent)
│   │   ├── tools/            #   kb_search.py, web_search.py, mcp.py (thin Tool wrapper)
│   │   ├── persistence/      #   memory_store.py (M2 stand-in), redis_checkpoint.py
│   │   │                     #   (CompactionCheckpoint → Redis, M7)
│   │   └── compaction/       #   summarize_strategy.py
│   ├── products/             # L2 — specs + discipline
│   │   ├── specs/            #   support_bot.py, … (one AgentSpec each, siblings)
│   │   ├── hooks/            #   quality_judge.py
│   │   └── middleware/       #   rbac.py, references.py
│   ├── service/              # L2 web DRIVER (one of possibly several drivers)
│   │   ├── app.py            #   FastAPI app factory: create_app() → root.assemble()
│   │   ├── routes.py         #   POST /v1/conversations/{id}/messages → SSE
│   │   ├── auth.py           #   AuthN/AuthZ dependencies
│   │   └── envelope.py       #   TurnComplete + AssembledAgent reads → response schema
│   ├── root.py               # application root: config → AgentSpec → AssembledAgent
│   │                         #   (agent handle + judge + citations bundle); driver-
│   │                         #   neutral — a CLI/batch driver calls the same assemble()
│   └── config.py
└── tests/
    ├── kernel/               # step-2 mock suite — the regression bed
    ├── capabilities/
    ├── products/
    └── service/              # httpx ASGI-transport tests incl. SSE
```

The package boundaries ARE the layer boundaries: **L0 must not import L1; L1 must
not import L2; no L2 imports a sibling L2.** Arrows point down only. Concretely:
`kernel` imports nothing internal; `capabilities` imports `kernel` only; `products`
imports both; product specs never import each other — the application root
(`turnstile/root.py`) is the one place allowed to know all products. (Root holds
no registries itself: `SPECS` lives in `products/specs/`, tool `RECIPES` in
`capabilities/tools/`; root imports and welds them, so adding an agent or a
tool never edits root.) Within the top
tier: `root` may import everything (composition is its job — that's its exemption);
`service` imports `root` + `kernel` only, never `products`/`capabilities` — it
touches product objects solely as fields of the `AssembledAgent` bundle that
`root.assemble(config)` returns (agent handle + judge + citations collectors), so
any driver (web, CLI, batch) gets the identical bundle. Drivers are siblings too:
a future `cli/` next to `service/` never imports it (or vice versa) — both consume
`root.assemble()`. Enforced mechanically — see §5.

## 2. Service: FastAPI + SSE

- **Startup**: `uvicorn turnstile.service.app` → `create_app()` reads config →
  `root.assemble(config)` → routes drive the returned `AssembledAgent`. The process
  starts at the driver; root is a function it calls, not an entry point.
- **One streaming endpoint per turn**: `POST /v1/conversations/{id}/messages` returns
  Server-Sent Events via `sse-starlette` (heartbeats + framing for free). The SSE
  stream is the driver protocol (design doc §5) serialized **verbatim** — one SSE
  event per `AgentEvent` (`event: text_delta`, `event: tool_started`, …), the
  terminal `event: turn_complete`, then one service-added `event: envelope`.
- **Envelope assembly** (`envelope.py`): the accepted answer from the snapshot, with a
  `### References` section appended; `signal` from the quality judge's recorded
  verdict; `references` `[{n, title, url, cited}]` from the reference collector — the
  L2-collector pattern (`kernel-loop-structure.md` §I.2), read via the `AssembledAgent`
  bundle from `root.assemble()` (service never imports product classes). Absent
  verdict (fuse exit) → `signal=no_answer`, never a fake ok.
- **Citations — the server numbers, the model cites**: the reference collector
  (middleware) renumbers retrieval excerpts turn-globally and per FILE before the
  model sees them and keeps the authoritative `n → document` map; at turn end it
  resolves the model's inline `[n]` by NUMBER (never by document name), lists only
  cited documents in the section, and drops any source list the model wrote itself.
  kb documents link to `GET /v1/files/{token}` — an opaque, expiring, recipient-bound
  (Fernet) token over `<file_root>/<region>/<path>`; web sources keep their URL. The
  driver mints links (it knows the principal); the collector never touches secrets.
  No collector wired = answer untouched, `references: []`.
- **History listing**: `GET /v1/conversations` → the principal's own conversations
  (`{conversation_id, title, turn_counter, in_flight}`, newest first; title = first user
  message). Ownership is the only filter — foreign ids are absent, never 403'd.
- **Session mapping**: one HTTP conversation id ↔ one kernel `Conversation` + its
  spawned agent handle, held in a **worker-local registry**. Created on the
  conversation's first POST (or resumed from the latest snapshot); evicted on idle
  TTL → final snapshot + `Shutdown`. A POST while a turn is still running is
  **steered** (kernel-native): the prompt folds into the running turn at the next
  round boundary — no 409, no second turn.
- **Decision — disconnect ≠ cancel**: a dropped SSE connection does NOT cancel the
  turn (the kernel's default cancel-as-undo would erase the user's message on every
  flaky mobile network). The turn runs to completion detached and is persisted; the
  client refetches on reconnect. Cancel is only ever explicit
  (`POST /v1/conversations/{id}/cancel` → `AgentCommand.Cancel`).
- **AuthN/AuthZ**: FastAPI dependencies resolve the user, then which products /
  tools / data they may use; data-level enforcement stays in the RBAC middleware
  (the model can't bypass the actuator wrapper).
- **Decision — buffer vs stream-then-revise**: a retry-capable policy (the quality
  judge today) can reject a draft whose `TextDelta`s already streamed. Start with
  **withhold-until-turn-complete** (with a judge deployed the service withholds
  text events until `TurnComplete`, whose envelope carries the accepted answer;
  simpler, correct);
  revisit streaming-with-revision-events when latency demands it. The retry half
  stays kernel-native (`offer_continuation`); withholding is driver presentation
  over the kernel's truthful event stream.
- **Decision — process model**: the L2-collector pattern requires the driver
  in-process with the kernel, and conversations are single-flight. Start with **one
  uvicorn worker** (asyncio concurrency across conversations). Scaling later means
  **conversation-affinity routing** across workers (sticky by conversation id) —
  never shared mutable state between workers. Same affinity idea also feeds the
  provider prefix cache (`bind_session_id` → gateway pins one upstream).

## 3. Packaging: uv

- `pyproject.toml` + committed `uv.lock`; `uv sync` / `uv run pytest` everywhere,
  including CI — one resolver, no requirements.txt drift.
- Runtime deps: `fastapi`, `sse-starlette`, `sqlalchemy[asyncio]`, `asyncpg`,
  `alembic`, `redis`, `httpx`, `pydantic-settings`.
- `[dependency-groups] dev`: `pytest`, `pytest-asyncio`, `ruff`, `import-linter`.
- Python pinned 3.12 (`requires-python` + `.python-version`).

## 4. Persistence: Redis (snapshots) + Postgres/alembic (projection)

- **Dual store, split by question answered** (decision 2026-08-18): the
  `SessionSnapshot` is the KERNEL-STATE store — hot, session-keyed, expiring —
  and lives in **Redis**; the `messages` projection is the AUDIT/HISTORY store —
  durable, queryable — and lives in **Postgres** (SQLAlchemy 2.0 async + asyncpg,
  alembic owns schema evolution; tables: `conversations`, `messages` (lossless
  `Message` rows), `sessions` / per-user metadata).
- **Why Redis for snapshots**: the snapshot's access pattern is a short-lived
  key-value object that goes colder with time. Keep-latest retention = a plain
  `SET session:{id}`; the idle-eviction semantics of §2 = native TTL; the blob is
  opaque versioned JSON — nothing relational. The L1 `redis_checkpoint.py`
  implements `CompactionCheckpoint` + snapshot load behind the port, so the
  M2 in-memory stand-in swaps out with zero change above L1.
  Ops guards: `maxmemory-policy noeviction` (memory pressure must not silently
  kill ACTIVE sessions); enable AOF if resume across a Redis restart matters.
- **The snapshot is authoritative for resume.** The kernel resumes ONLY from the
  latest snapshot (version check, counter high-water marks, `repair_pairing`).
  `messages` is an append-only PROJECTION for queries / audit / UI listing and is
  NEVER read to reconstruct kernel state — reconstruction from rows is wrong in
  general: (1) `derive_counters` under-counts turns that died before storing a
  response (the live capturer stamps higher counters into the snapshot), so
  resume would reuse ids; (2) the projection retains pre-compaction rows, so a
  rebuild resurrects what compaction removed; (3) `cache_epoch` is Conversation
  state, not a message attribute. Both stores write from the same seam: a
  `turn_complete` hook saves the snapshot and appends the projection rows; a
  committed compaction additionally saves through the `CompactionCheckpoint` gate.
- **Snapshot lifetime = the resume window.** A TTL'd-out (or evicted) snapshot
  makes that conversation permanently non-continuable — deliberately accepted:
  it stays VIEWABLE via the projection, just not resumable. Pick the TTL as the
  product's "come back and continue" window.
- **Snapshot retention: keep latest only.** Each snapshot is the whole conversation
  (cumulative — snapshot N ⊇ snapshot N−1 up to compaction), so keeping all of them
  costs O(turns × conversation size); latest-only suffices for resume.
  TODO(later): last-N retention if undo/debugging wants it; externalize `images` to
  object storage before snapshotting if the product accepts image input (base64
  blobs inflate snapshots by MBs per photo).
- Rules: autogenerate diffs are reviewed, never blind-applied; a migration and the
  code that uses it land in the **same commit**; migrations tested up **and** down.

## 5. Regression-test-guarded workflow

- **CI gate on every commit**: `ruff check` + `uv run pytest` + `import-linter`.
  Local gate is `make check`; CI host wiring is deferred (TODO: pick runner).
- **Commit subject format**: `[module] description` — e.g. `[ruff] lint gate`,
  `[kernel] engine round loop`. One commit = one thing.
- **import-linter contracts** encode the dependency rule (design doc §2) as a
  failing build: `kernel` → nothing internal; `capabilities` → `kernel` only;
  `products` → `kernel` + `capabilities`; `service` → `root` + `kernel` only;
  `root` → anything (the composition exemption); no `products` ↔ `products` edge.
  The one review-rejectable violation becomes machine-checked.
- **Feature commit = code + tests in one diff.** A feature diff with no new test is
  review-rejectable.
- **Bugfix = failing test first**, then the fix, same commit.
- **Existing tests are immutable by default.** A commit editing an existing
  assertion is a contract change — reviewed as such, never slipped in with a
  feature. Existing behavior regressing = the suite goes red = the commit doesn't
  land.
- **Test tiers** (pytest markers):
  - `unit` — kernel + products on `testkit` doubles; fast, zero I/O; always run.
  - `service` — in-process ASGI via httpx transport, incl. SSE framing; always run.
  - `integration` — real Postgres (disposable container), optionally a real
    provider endpoint; pre-merge / nightly.
- The kernel mock suite (`tests/kernel/`) is the permanent regression bed: turn and
  round lifecycle, tool dispatch (dedup, parallel-safe concurrency, gates), every
  fuse, hook composition order, steering, cancel semantics, resume monotonicity.

## 6. Milestones (maps to design doc §8; each lands green tests)

| M | Delivers | Tests that guard it |
|---|---|---|
| M0 | Scaffold: uv project, layout, CI, import-linter contracts | lint + boundary contracts green on empty packages |
| M1 | **L0**: dtos / ports / events + engine + testkit (steps 1–2) | the kernel mock suite — turn/round, dispatch, fuses, hooks, middleware, steering, cancel. The de-risk milestone: hold all L1/L2 work until green |
| M2 | **L1**: OpenAI-compat provider, KB/search tools, in-memory checkpoint (step 3) | provider adapter vs recorded SSE fixtures; each tool vs real I/O in isolation |
| M3 | **L2 assembly**: `AgentSpec` + first product spec + root wiring + headless driver (step 4) | `run_to_completion` product tests on scripted provider |
| M4 | **Service**: FastAPI + SSE + auth + envelope with judge/citations collectors (step 5); **docker service image** (multi-stage: `uv sync --frozen --no-dev`, uvicorn entrypoint) | ASGI SSE tests: event framing, envelope defaults on fuse exits, detached completion on disconnect + explicit cancel, steer on mid-turn POST; image builds + container answers a health check |
| M5 | **Features** (persistence-free, so it lands first): flat role claim + admin gate; SAML/FAC SSO login minting our JWTs; tokenized citation-file endpoint (Fernet, recipient-bound); reference numbering + `### References` section in the envelope | admin gate 401/403 paths; ACS journeys on a stubbed SAML seam; file tokens (binding, expiry, traversal); turn-global per-file numbering + section rendering, lego-removal test |
| M6 | **Web UI** (`frontend/`, Vite + React + TS): `GET /v1/conversations` listing; chat window over fetch-SSE (envelope-driven rendering, signal badge, stop/steer); markdown + `[n]` citations with blob-opened file links; history panel; SPA served by the container | list endpoint ownership/ordering tests; `npm run check` (typecheck, lint, vitest with msw-mocked SSE, build) per `[frontend]` commit |
| M7 | **Persistence**: Redis snapshot checkpoint (TTL resume window) + Postgres/alembic messages projection + resume (step 6) | snapshot round-trip incl. TTL expiry → view-only, resume monotonicity, migration up/down |
