# turnstile

One-loop agent engine built on a two-layer contract: L0 kernel and L1
capabilities. Products compose on top — a chatbot, a CLI, a web UI, whatever
the deployment needs — each picking its own tools, persona, and driver.

## Philosophy

- **No LLM framework.** No LangChain, LangGraph, LlamaIndex, CrewAI, AutoGen,
  or Semantic Kernel — the runtime deps are an HTTP client, FastAPI, and the
  MCP SDK. An agent loop is a `while` loop over a message list; a framework
  buys abstraction we would have to fight to keep the loop honest.
- **One loop, two layers.** A single kernel engine (`kernel/engine.py`) drives
  every turn, written only against ports and DTOs — it knows nothing about
  HTTP, files, or which model answers. L1 marshals real I/O into those types.
- **A minimal shared language.** The frozen L0 surface is 8 ports — 6 with
  abstract methods, plus `LifecycleHooks` and `ToolMiddleware` whose methods all
  default to no-ops — and 69 dataclasses across `dtos.py` + `events.py`. Every
  module speaks that vocabulary; nothing else crosses a layer line.
- **The judgment lives in the model, not the code.** Hooks and middleware may
  gate, redact, and observe; the moment code makes a choice the model could
  have made from context, it is strategy and belongs to the model.

## The three layers

```
        ┌─────────────────────────────────────────────────┐
  L2    │  Product assembly — what we ship                │  "the product"
        │  picks + composes L1 components, sets persona   │
        └───────────────────┬─────────────────────────────┘
                            │ depends on ▼
        ┌───────────────────┴─────────────────────────────┐
  L1    │  Capabilities — concrete implementations        │  "craft the hammer"
        │  marshal real I/O (LLM API, files, net, DB)     │
        │  into and out of L0 types                       │
        └───────────────────┬─────────────────────────────┘
                            │ depends on ▼
        ┌───────────────────┴─────────────────────────────┐
  L0    │  Kernel — the protocol and the engine           │  "contract + engine"
        │  ports + DTOs + the one loop + driver protocol  │
        └─────────────────────────────────────────────────┘
```

- **L0 is not "just interfaces" — it owns the loop.** A microkernel: ports,
  DTOs, the streaming turn/round engine, and the driver protocol. It reaches
  every capability through dynamic dispatch and knows nothing about HTTP,
  filesystems, or which model is behind the provider.
- **L1 marshals.** An `LlmProvider` parses an SSE/HTTP stream into
  `StreamEvent`s; a `Tool` runs a KB query or an MCP call and packs a
  `ToolResult`. Every real-world mess — retries, TLS, auth, parsing — stops
  here; the loop upstream sees only clean DTOs.
- **L2 assembles.** It mounts the tools, sets the persona and model, and wires
  the discipline hooks. Products are siblings: none depends on another.
- **Arrows point down only.** L2 depends on L1 and L0, L1 depends on L0, L0
  depends on nothing. Violating this is the one thing code review must reject.

## Loop structure

Three levels of one skeleton: the round loop, the two injection seams, and the
plumbing that keeps it alive across failures and restarts.

### 1. The round loop

- One user message = one **turn**; each model call inside it = one **round**.
- The loop runs rounds until the model replies with plain text and no tool
  calls — the model decides when it is done.
- Errors are data: a failed tool returns a `ToolResult`, never an exception up
  the stack, so the model can react and retry.

```
user text ──▶ append to Conversation
                    │
        ┌───────────▼──────────┐
        │  call the model      │◀────────────┐
        │  (stream the reply)  │             │
        └───────────┬──────────┘             │
                    │                        │
           reply has tool calls?             │
             │              │                │
            no             yes               │
             ▼              ▼                │
        TurnComplete   execute tools,        │
        (STOPPED)      append results ───────┘
```

### 2. Hooks and middleware

- **Hooks shape the turn** — one port, 15 methods, all default no-op: rewrite
  or reject the prompt, tune this call's options, transform each streamed
  chunk, drop a tool call before it runs, keep the model going at stop.
- **Middleware wraps each tool call** — `before` returns a gate
  (`PROCEED` / `ALLOW` / `ASK` / `DENY` / `DENY_TURN`), `after` transforms the
  raw result. It wraps the actuator, so the model cannot bypass it.
- **Fuses bound the trust**: `MAX_ROUNDS`, `MAX_CONTINUATIONS`, `REPEAT_LOOP`,
  `TOOL_LOOP_DETECTED` — every exit funnels through one `TurnComplete`.

```
user text ─▶ user_prompt_submit ─▶ append to Conversation
              (rewrite / block → PROMPT_REJECTED)
     ┌────────────────────────────▼─────────────────────────┐
     │ pre_request        (hooks mutate a CLONE, ephemeral) │◀─────┐
     │ call model ── stream ──▶ on_text_delta (per chunk)   │      │
     │ on_model_response  (may drop / rewrite tool_calls)   │      │
     └──────────┬───────────────────────────┬───────────────┘      │
                │ no tool calls             │ tool calls           │
                ▼                           ▼                      │
       offer_continuation        middleware.before(call, tool)     │
       (critique → re-run,        gate ─▶ execute ─▶ .after ───────┘
        None → accept)
                ▼
       TurnComplete(reason)  ← the terminal event, exactly one per turn
```

### 3. The plumbing

- **Failures are typed, not guessed.** `ProviderError` carries `retryable`,
  `http_status`, `retry_after_secs`, and `is_context_overflow()` — the loop
  matches on that one type to pick retry, compact-and-retry, the rate-limit
  path, or fail. A 429 is a conversation: the host answers with wait-or-pause,
  the kernel owns only the mechanics and the livelock fuse.
- **A session survives the process.** `SessionSnapshot` holds lossless
  messages, `cache_epoch`, and the id high-water marks, so a resumed session
  continues its sequence instead of minting duplicates. Compaction is a
  protocol — the strategy proposes a plan, the kernel revalidates it and
  commits only a net loss of bytes.
- **None of this changes the loop's shape.** The driver protocol
  (`AgentCommand` / `AgentEvent`), the content sidecars (`ImageContent`,
  signed `ReasoningBlock`), and the tool-context companions
  (`CancellationToken`, `ProgressSink`, `Requester`) are all additive — the
  bet is that levels 2 and 3 never bend level 1.

## Structure and dependency

    ──▶ imports

    turnstile.service ──▶ turnstile.root ──▶ turnstile.products      (L2)
     the web driver       the composition             │
                          root: the ONE               ▼
                          module that may     turnstile.capabilities (L1)
                          import every layer           │
                                                       ▼
                                              turnstile.kernel       (L0)

- **The dependency rule, as imports.** L0 imports nothing internal — not L1,
  not L2, not the service, not config. L1 imports the kernel only. No L2 spec
  imports another.
- **Only `root` composes.** `root.assemble()` is the single place allowed to
  import every layer, and the only place deployment config is read — config
  never crosses into a capability or a product spec.
- **The driver stays generic.** `turnstile.service` talks to `root` and the
  kernel DTOs only; it never names a product or a capability type.
- **The rule is a failing build, not a convention.** Six import-linter
  contracts in `pyproject.toml` encode it; `make contracts` is part of the gate.

## Run

    uv sync && make check              # the gate: lint, format, types, contracts, tests
    make test-all                      # + live integration tests (GPU hosts, file mirror)
    uv run uvicorn --factory turnstile.service.app:create_app   # reads .env (see config.py)
    make image TAG=dev                 # docker image; /health reports commit + tag

## Docs

- [`docs/one-loop-two-layer-agent.md`](docs/one-loop-two-layer-agent.md) — the
  L0/L1/L2 design and the frozen L0 port + DTO inventory (Appendix)
- [`docs/kernel-loop-structure.md`](docs/kernel-loop-structure.md) — the kernel
  loop: DTO structure in three levels (Part I) and runtime mechanics (Part II)
- [`docs/architecture.md`](docs/architecture.md) — repo layout, service design,
  persistence, workflow, milestones

## Credits

This work is motivated by — and in large part a direct conversion of —
[**atomcode**](https://atomgit.com/atomgit_atomcode/atomcode), an open-source
terminal AI coding agent written in Rust (MIT). Its `atomcode-kernel` crate is
the reference implementation cited throughout `docs/`.

What came across: the L0/L1/L2 split, the one-loop engine and its turn/round
state machine, the hook and middleware seams, the fuse and `StopReason` set,
and most of the DTO vocabulary. `kernel/testkit.py` is a port of that crate's
`testkit.rs`, kept shipped so L1 and L2 tests reuse the same doubles.

The translation is close to line-for-line where the two languages allow: traits
become `ABC`s and `Protocol`s, `Result<T, E>` becomes return-or-raise, `&mut`
in-place mutation becomes "mutate the object" or "return the value" depending
on the type, and Rust's must-not-panic tool discipline becomes "any tool
exception is caught and turned into an error result". Where we deliberately
diverge, the docs say so — the split first-event / inter-token stream timeout
(`kernel/engine.py`), and the L2-collector read-back a web driver needs and a
CLI does not.

The architecture is theirs; the mistakes in this port are ours. Upstream's MIT
notice is reproduced in [`LICENSE-THIRD-PARTY`](LICENSE-THIRD-PARTY), which the
license requires to travel with a port this direct.

