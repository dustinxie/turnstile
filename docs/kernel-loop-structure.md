# Kernel Loop — Structure & Mechanics

Companion to `one-loop-two-layer-agent.md` (the L0/L1/L2 design doc; full DTO
reference definitions live in its Appendix). This file explains the kernel loop
itself, twice over: **Part I** builds the DTO structure up in three levels of
importance — read it top-down to learn the system; **Part II** specifies the runtime
mechanics inside one turn — the step-2 build list.

## Part I — The DTO structure in three levels

### I.1 Level 1 — the irreducible loop (6 DTOs + 1 enum)

Strip everything away and the kernel is this:

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
             │              │                │
             ▼              ▼                │
        TurnComplete   execute tools,        │
        (STOPPED)      append results ───────┘
                       (next round)
```

One user message = one **turn**. Each model call inside it = one **round**. The loop
runs rounds until the model answers with plain text and no tool calls — the model
deciding it's done, which is the whole thesis: the judgment loop lives in the model,
the kernel just carries messages back and forth. Degenerate case: no tools mounted →
exactly one round (append user message, stream the answer, `STOPPED`).

The six DTOs that make this work:

1. **`Message`** — the atom. `role` (system/user/assistant/tool) + `text`, plus
   `tool_calls` when the assistant wants tools and `tool_call_id` when it's a tool's
   answer. Everything the model ever sees is a list of these.
2. **`Conversation`** — the state. Just `messages: list[Message]`, append-only. Every
   round sends the whole list to the model.
3. **`ToolCall`** — the model's request: `id`, `name`, raw JSON `arguments` string.
   The id matters: every call must get exactly one matching result or the provider
   rejects the payload.
4. **`ToolResult`** — the answer to one call: `call_id`, `content`, `is_error`.
   Errors are *data* — an error result goes back to the model so it can react, never
   an exception up the stack.
5. **`ToolDef`** — what the model is told a tool looks like (name, description, JSON
   schema). The menu, as opposed to the kitchen: connecting a real backend (web
   search, an MCP server) means implementing the `Tool` **port** in L1 — a thin
   wrapper marshaling `execute(args) → ToolResult` (an MCP server's `readOnlyHint`
   maps to `read_only_hint()`); the kernel derives the `ToolDef` from it.
6. **`StreamEvent`** — how the model's reply arrives: a stream of `TextDelta` /
   `ToolCall` items ending in `Done`. The loop consumes these; the provider produces
   them. This union is the entire seam between the loop and any vendor.

And one enum to close the turn: **`StopReason`** — at this level just `STOPPED`
(model finished on its own). Every turn ends with exactly one of these.

That's a complete working agent: 6 DTOs + 1 enum, two ports implied (`LlmProvider`
yields StreamEvents, `Tool` maps args → ToolResult), and a while-loop. Everything
below is armor bolted onto this skeleton without changing its shape.

### I.2 Level 2 — the control surfaces: who else gets a say

Level 1 was model ↔ tools. Level 2 adds the two injection seams and the safety
rails. Same skeleton, annotated:

```
user text ─▶ user_prompt_submit ─▶ append to Conversation
                (hooks: rewrite / block → PROMPT_REJECTED)
             ┌─────────────────────────────▼─────────────────────────┐
             │ round: mint TurnCtx (ids, cache_epoch, live pressure) │
             │  pre_request (hooks mutate a CLONE, ephemeral)        │◀──────┐
             │  call model with ChatOptions ── stream ──▶            │       │
             │  on_text_delta (hooks transform each chunk)           │       │
             │  on_model_response (hook may drop/rewrite tool_calls) │       │
             └───────────────┬────────────────────────┬──────────────┘       │
                             │ no tool calls          │ tool calls           │
                             ▼                        ▼                      │
                   offer_continuation      middleware.before(call, tool, rt) │
                   (judge hook: critique ─┐  Gate: PROCEED/ALLOW/ASK/        │
                    → Some → re-run ──────┤        DENY/DENY_TURN            │
                    good → None,          │  execute → middleware.after ─────┘
                    verdict → self)       │  (citations middleware → self)
                             │            └────────────────────▶ (next round)
                             ▼
                   TurnComplete(reason: StopReason)      ← the terminal event
                             │
                             ▼
              web layer (driver): answer = accumulated TextDeltas,
              signal = judge.verdict, references = collector.docs
              (reads its own L2 collector objects — see below)
              [fuses watch every round: MAX_ROUNDS, MAX_CONTINUATIONS,
               REPEAT_LOOP, TOOL_LOOP_DETECTED]
```

The new DTOs, in importance order:

1. **`TurnCtx`** — the round's identity card, handed to every hook call:
   `session_id / turn_id / request_id / round` (monotonic counters — deterministic
   log stitching), `max_rounds`, `cache_epoch`, and live pressure
   (`context_window`, `used_tokens` from the last usage report).
2. **`Gate` / `BeforeOutcome` / `AfterOutcome`** — the tool-level policy vocabulary
   (see "Middleware" below).
3. **The full `StopReason` set + fuses** — the loop trusts the model but not
   infinitely: `MAX_ROUNDS` (round cap), `MAX_CONTINUATIONS` (a continuation hook
   that never says stop is a bug, not a workload), `REPEAT_LOOP` (same call pattern
   6 rounds, nudge at 3), `TOOL_LOOP_DETECTED` (opt-in: same call + same result =
   no progress; warn 3 / stop 4), plus `PROMPT_REJECTED`, `POLICY_DENIED`,
   `RATE_LIMITED`, `PROVIDER_ERROR`, `TIMEOUT`, `CANCELLED`. Every fuse exits
   through the single finish funnel — one `turn_complete`, one terminal event.
4. **`ChatOptions`** — per-call knobs (`reasoning_effort`, `max_tokens`,
   `temperature`, `tool_choice`): L2 sets values, the L1 adapter maps them to its
   wire format or ignores them. Sideband — never in the prefix bytes.
5. **`MessageMeta` + `TokenUsage`** — the stats sidecar stamped on every assistant
   message: usage, elapsed, window, utilization, correlation ids, `finish_reason`.
   Never rendered into text (prefix-cache safety). This is what the compaction
   trigger and `TurnCtx`'s live-pressure fields read.
6. **`Continuation`** (typed) — `offer_continuation` upgraded: text + `kind`
   (generic / verify-cadence / truncation-resume) + `visibility`
   (`INTERNAL_CONTROL` rounds stream nothing to the driver and store blank text —
   control chatter stays out of user-visible history).

Rule of thumb for the two seams: **hooks** shape the *turn*, **middleware** wraps
each *tool call*. Both talk to the loop exclusively in these DTOs. And both live
under the design doc's §4 rule — **gate, redact, observe; never strategy**. The
test: if the code makes a choice the model could have made by reading the context
(reword the query, pick sources, judge a result mid-flight), it's strategy and it
belongs to the model.

#### The hook inventory (one port, 15 methods, all default no-op)

| Stage | Hook | What it's for |
|---|---|---|
| session | `session_start(convo, resumed)` | seed persona/context into a fresh conversation; skip when resuming (snapshot already has it) |
| | `session_end(convo)` | flush telemetry, close resources |
| turn entry | `user_prompt_submit(text)` | rewrite the prompt (metadata, guardrails) or raise `PromptRejected` to refuse it |
| | `turn_start(convo)` | one-time permanent history mutation as the turn opens |
| per round, pre-call | `pre_request(messages, ctx)` | shape THIS request on a throwaway clone: append tail reminders; must stay append-only |
| | `pre_request_options(messages, options, ctx)` | tune this call's knobs (effort, tool_choice) |
| | `on_request(messages, tools, options, ctx)` | read-only view of the exact final wire — telemetry, prefix-cache RCA (see box) |
| per round, streaming | `on_text_delta(delta)` / `on_reasoning_delta(delta)` | transform each chunk pre-emit (redaction); `""` drops the chunk |
| per round, post-stream | `on_model_response(msg)` | edit the stored message; dropping a tool call means it never executes |
| turn end | `offer_continuation(convo)` / typed variant | model wants to stop: return critique text to keep it going, `None` to accept |
| | `turn_complete(convo, reason, ctx)` | the turn ended, however it ended — persistence, metrics; exactly once |
| anytime | `on_error(str)` | observe tool/provider errors (turn may continue) |
| | `on_rate_limit(hint)` | wait-vs-pause verdict on a 429, or `None` for the kernel fallback |

Multiple hook objects compose in registration order: transforms chain (a later hook
sees the earlier rewrite), `user_prompt_submit` short-circuits on the first block,
`offer_continuation` first-Some wins. All are plain in-process functions — µs–ms —
except where an implementation deliberately spends more (an LLM-judge continuation,
a human-approval round-trip).

> **Box: `on_request` and prefix-cache RCA.** Providers/gateways cache the KV state
> of a prompt's leading bytes (vLLM: per-16-token blocks, hash-chained
> `h_n = hash(h_{n-1} ‖ block_n)`; reuse = longest matching block prefix). An agent
> loop should hit that cache every round — unless some hook perturbs earlier bytes,
> which fails *silently* (same answers, full price, `TokenUsage.cached` ≈ 0). The
> RCA tool: in `on_request`, log a cumulative hash per message boundary; compare
> consecutive requests — the first diverging index names the message whose bytes
> changed. `cache_epoch` scopes the comparison (a committed compaction legitimately
> rewrites history). `LlmProvider.bind_session_id` completes the picture: the
> gateway pins a session to one upstream worker so the cache is actually reachable.

#### Middleware in one box

`ToolMiddleware` is the around-wrapper for one tool call — two methods, both default
pass-through:

- **`before(call, tool, rt)`** — may rewrite the call in place (normalize args) and
  round-trip the driver via `rt.request()` (approval prompts). Returns a gate:
  `PROCEED` (next middleware), `ALLOW` (force-approve, skip remaining gates), `ASK`
  (defer to a downstream approval middleware — the kernel owns no prompt),
  `DENY(reason)` (error result the model sees and can react to), `DENY_TURN(reason)`
  (block + end the turn → `POLICY_DENIED` — the hard boundary against retrying
  another spelling of a forbidden thing).
- **`after(result)`** — transform the raw (pre-size-cap) result; `BLOCK(reason)`
  feeds the reason back to the model as a synthetic user message.

Ordering is load-bearing and contractual: registration order, first deny stops the
chain — so argument-normalizers must register before every gate/approver, or the
user approves different bytes than what executes. Why a separate port from hooks:
approval stays out of the kernel (`Tool.risk()` is advisory; the kernel knows risk,
never enforces), the policy/strategy split becomes structural (the model chooses
*when* to search; middleware decides *whose* docs it sees — it wraps the actuator,
so the model can't bypass it), and middleware gets only the call/result — it
*can't* do turn-level strategy.

#### The quality-retry loop (`offer_continuation`)

The seam fires only when the model stops on its own (a round with zero tool calls).
`Some(text)` → the kernel appends `text` as a synthetic user message and runs
another round — the model retries *with the critique in context*; `None` → the stop
is accepted. `MAX_CONTINUATIONS` backstops a hook that never says stop.

The reference CLI's production judges are **cheap code, no LLM**: `VerifyCadenceHook`
(model stopped after editing code without running a check → one nudge "run `cargo
check`"; own state so it nudges once per edit-batch) and `TodoHook` (open task-list
items at stop → one close-out nudge). In the coding domain the evaluator is
executable — `cargo check` *is* the judge. A QA chatbot has no compiler for "is this
answer good and grounded", so our judge on the same seam is an **eval-LLM call**:

```python
class QualityJudge(LifecycleHooks):
    def __init__(self, eval_llm):
        self.eval_llm = eval_llm
        self.verdict = None
        self.tries = 0

    async def offer_continuation(self, convo):
        v = await self.eval_llm.judge(question=..., draft=last_assistant_text(convo))
        if v.score >= 0.2 or self.tries >= 1:  # low bar to retry; own budget, not the fuse
            self.verdict = v  # ok / low_confidence / no_answer
            return None
        self.tries += 1
        return f"Answer scored low: {v.reason}. Address this and revise."
```

Conventions carried from the reference impls: the judge reads only `convo`; returns
a *specific, actionable* critique (a bare "re-run" is a blind re-roll — same context,
same answer); keeps its own one-shot/retry state (the kernel fuse is a backstop, not
the policy); and its verdict thresholds map onto the envelope signal
(`< 0.2` retry → exhausted = `no_answer`; `0.2–0.5` no retry but `low_confidence`;
else `ok`). This judge is the sanctioned edge of the no-strategy rule: it judges
sufficiency *overtly, at the turn's end* — its critique enters the conversation and
the model decides how to respond — never a silent mid-flight overrule. One product
caution: don't add "skip the eval when the query looks trivial" heuristics
mid-flight (the k5 scope-heuristic eval-skip incident); if eval-skip is wanted, make
it an explicit per-product `AgentSpec` toggle.

#### The L2-collector pattern (structured data out of the loop)

A product that needs structured per-turn data (quality verdict, citations for the
response envelope) composes **stateful L2 hook/middleware objects**: a citations
middleware appends references to its own field as search results pass through
`after`; the judge above records `self.verdict`. The web layer *constructed* those
objects, so after the terminal `TurnComplete{reason}` event it simply reads them:

```python
collector, judge = CitationsCollector(), QualityJudge(eval_llm)
agent = assemble(spec, middleware=[collector], hooks=[judge])
async for ev in events:
    ...stream TextDeltas to the client...
    if isinstance(ev, TurnComplete):
        envelope = build(answer_text, ev.reason,
                         signal=judge.verdict or NO_ANSWER,   # consumer-owned default
                         references=collector.docs)
```

Absent verdict → degraded default (`signal=no_answer`), never a fake ok — a fuse
exit may mean the judge never ran. Preconditions: the driver runs in-process with
the kernel (our web-service shape) and one in-flight turn per conversation (the
session loop already guarantees this). A CLI composes similar stateful L2 objects
(approval grants, persistence hooks) but never needs the read-back — its consumer is
a human who already read the stream.

### I.3 Level 3 — the plumbing: survivable, resumable, multi-modal

Nothing here changes the loop's shape; these DTOs exist because networks fail,
processes restart, and answers aren't always text.

**Resilience (feeds Part II §II.5):**

1. **`ProviderError`** — the structured failure: `retryable` (5xx/timeout vs
   terminal auth/400), `http_status`, `code`, `retry_after_secs`, plus
   `is_context_overflow()` centralizing every vendor's "prompt too long" signature.
   The open-failure `match` branches on this one type: retry, compact-and-retry,
   rate-limit path, or fail.
2. **`RateLimitHint` / `RateLimitDecision`** — the 429 conversation between kernel
   and host: hint = what the kernel knows (status, Retry-After, billing-vs-throttle,
   attempt #); decision = the verdict (`wait_secs` → cancellable sleep + re-issue,
   else pause cleanly as `RATE_LIMITED`). Only the *host* knows quota windows; the
   kernel owns just the mechanics and the livelock fuse.

**Persistence & compaction:**

3. **`SessionSnapshot`** — the durable conversation: `version` (checked before
   interpreting), full lossless `messages`, `cache_epoch`, and
   `turn_counter`/`request_counter` high-water marks so a resumed session continues
   its id sequence instead of minting duplicates.
4. **`CompactTrigger` / `CompactionView` / `CompactionPlan` / `CompactReport`** —
   the compaction protocol: *why* (auto pressure / manual / overflow recovery),
   *what the strategy sees* (read-only messages + pressure facts + `sacred_floor`),
   *what it proposes* (drain range, summary, in-place rewrites, resume note — a
   proposal the kernel revalidates), *what happened* (bytes before/after,
   `committed` — a refused net-loss plan burns nothing).

**Driver protocol:**

5. **`AgentCommand` / `AgentEvent`** — the two serializable unions; the entire
   in/out surface (design doc §5). The long tail of events exists for UI honesty:
   `ToolBatchStarted/Completed` (grouped rendering), `ToolProgress` (a long tool's
   heartbeat), `Steered` (your prompt was folded into the running turn), `Warning`
   vs `Error`, `CompactionStarted/Compacted`.
6. **`Outcome`** — the one-shot aggregate for batch/CI drivers: text + tool results
   + `stop` + last error with structured code. Pure failure perception — a failed
   headless run can't look like an empty success.

**Content sidecars:**

7. **`ImageContent`** — neutral base64 image + MIME. Rides user messages in, and
   tool results out (the loop lifts tool images onto a synthetic user message —
   providers reject images on the tool role). The adapter owns the wire shape.
8. **`ReasoningBlock`** — a signed thinking unit: text + opaque round-trip token +
   provider attribution. Thinking models reject requests unless prior reasoning is
   echoed back byte-exact; the kernel stores the mechanism, the L1 adapter owns the
   echo policy.

**Tool-context companions:**

9. **`CancellationToken`** — cooperative cancel; a long tool polls it, the kernel
   drops the future as backstop.
10. **`ProgressSink`** — `emit(msg)` → `ToolProgress` event; no-op default, so
    tools report unconditionally.
11. **`Requester`** — the request-only handle into the Request/Respond round-trip,
    so a plain tool can ask the driver a structured question.

Full map: **level 1** = the conversation loop; **level 2** = control; **level 3** =
survival and transport. The design bet is that levels 2–3 never alter level 1's
skeleton.


## Part II — Loop mechanics: inside one turn

What the engine does between `SendMessage` and `TurnComplete`. These are the step-2
build list (design doc §8 plan).

### II.1 Turn lifecycle

1. **Prompt gate** — `user_prompt_submit` hooks run in registration order, chaining
   text rewrites; the first block wins → `Error` + `TurnComplete(PROMPT_REJECTED)`.
   The message never enters history; `turn_start`/`turn_complete` do not fire (no
   turn ran).
2. **Task-boundary auto-compaction** — if the last turn's recorded prompt tokens,
   recomputed against the LIVE model window (so model switches re-evaluate), cross the
   configured threshold, compact BEFORE the new user message enters history. This is
   the cache-safe trigger point: a committed compaction opens a new epoch, then the
   turn appends onto the compacted history.
3. **Rollback point** — record history length before pushing the user message, so
   cancel-as-undo can erase the whole turn.
4. **Rounds** (§II.2) until a terminal: a no-tool-calls stop, a fuse, an error, or
   cancel. Exactly one `turn_complete` hook + `TurnComplete` event per turn, on EVERY
   terminal path — a single finish funnel.

### II.2 One round

1. Mint `request_id`, build `TurnCtx` — before the round fuse, so even a fuse terminal
   hands `turn_complete` a ctx.
2. **Round-cap fuse** — `max_rounds` exceeded → `MAX_ROUNDS`. Optionally an
   interactive checkpoint instead: a driver `Request` ("continue past the cap?")
   re-arms the cap by the base amount on yes, stops fail-closed on no/no-answer.
3. **Drain the steer buffer** (§II.4) — fold mid-turn user prompts into history.
4. **Project the request** — clone stored messages (EPHEMERAL); `pre_request` hooks
   mutate the clone; the loop repairs tool pairing on the projection and verifies it
   is APPEND-ONLY over the stored prefix (a violation warns — it poisons the provider
   prefix cache); `pre_request_options` mutates the per-call `ChatOptions` copy;
   `on_request` observes the exact final wire, read-only.
5. **Pre-send emergency compaction** — if the estimated request reaches the effective
   input budget (window − output reserve − margin) and a completed exchange exists to
   drain, compact-and-reproject (bounded attempts) instead of firing a doomed request;
   otherwise emit a once-per-turn over-window advisory.
6. **Open the stream** — raced against cancel. Open failures branch: context overflow
   → compact-and-retry the round; 429 → the rate-limit path (§II.5); other retryable →
   bounded visible retries; terminal → `PROVIDER_ERROR`.
7. **Consume the stream** — every next-event await raced against cancel and the idle
   `stream_timeout`. Text/reasoning deltas run through their transform hooks BEFORE
   emit and accumulation, so the live stream and the stored message are consistently
   redacted (a cleared chunk is suppressed entirely). Tool calls collect; usage events
   fold field-wise-max (handles split vs cumulative provider reporting);
   `Done(truncated)` closes the round.
8. **Store the response** — build the assistant `Message` with kernel-owned `meta`
   (usage, elapsed, window, utilization, correlation ids, finish_reason).
   `on_model_response` may transform it INCLUDING dropping/rewriting `tool_calls`; the
   loop re-reads the calls from the (possibly edited) message, so a dropped call is
   never executed. Misrouted-answer recovery: a stop-finish round with empty content
   but non-empty plain reasoning promotes the reasoning to content, through the same
   delta-transform seam.
9. **Branch** — tool calls present → dispatch (§II.3), then next round. None →
   truncation auto-continuation if the output was cut at the token limit (bounded;
   nudges toward incremental file writes); else `offer_continuation` hooks (all
   observe, first `Some` wins; `MAX_CONTINUATIONS` fuse; injected as a synthetic user
   message); else `TurnComplete(STOPPED)`.

### II.3 Tool dispatch — three phases

- **① Classify** every call, in emission order: (a) duplicate `call_id` → skip
  entirely (two results for one id is an illegal payload); (b) same
  name + canonicalized args under a NEW id → stub result, no re-execution; (c)
  unknown/unmounted tool → error result; (d) otherwise run the middleware `before`
  chain — `PROCEED` = next gate, `ALLOW` = short-circuit approve, `ASK` = defer to a
  downstream approval middleware (the kernel owns no prompt), `DENY` = error result
  the model sees, `DENY_TURN` = error result + terminate the turn once every call in
  the batch is paired (`POLICY_DENIED`). Arg rewrites ride on the mutable call; dedup
  keys use the model's ORIGINAL bytes, pre-rewrite.
- **② Execute** concurrently: `parallel_safe(args)` calls share a read-lock;
  side-effecting calls take the exclusive write-lock (a barrier); a semaphore caps
  concurrency (default 4). Each execution polls cancel; `ToolStarted` fires only for
  tools that actually run.
- **③ Apply** in emission order: `after` middleware chain (sees the raw result;
  `Block{reason}` feeds the reason back to the model as a synthetic user message),
  then the kernel size cap, then emit + store. Tool-returned images are lifted onto
  ONE synthetic user message after all results land (providers reject images on the
  tool role).
- Batches of ≥2 distinct calls emit `ToolBatchStarted`/`ToolBatchCompleted` so a
  driver renders one grouped block.

### II.4 Steering and cancellation

- **Steer**: a `SendMessage` arriving MID-turn folds into the RUNNING turn at the next
  round boundary as a real user message (`Steered` event). It changes the turn's
  intent, so the repetition fuses reset; a steer landing on a no-tool-call round keeps
  the turn alive so the model answers it in-turn. Synthetic prompts, `Snapshot`, and
  `Compact` queue FIFO and run at the turn boundary instead.
- **Cancel**: one per-turn token, optionally a child of an external parent token so a
  parent agent's cancel propagates into subagents. Every await — open, stream, sleeps,
  tool execution — is raced against it, and pending driver requests flush to null
  (fail-closed) so a parked approval prompt can't freeze the turn. Default semantics:
  **cancel = undo** — roll history back to the rollback point, no trace. Opt-in
  alternative preserves the partial work: backfill `(cancelled)` results for dangling
  tool calls (wire stays API-valid) plus a synthetic interruption marker.

### II.5 Resilience tiers

The provider adapter's own fast transport retries sit BELOW all of these.

| Failure | Kernel response | Budget |
|---|---|---|
| retryable open error (5xx/transport) | visible warning + cancellable backoff (3/6/9s), re-issue same round | 3 / round |
| 429 rate limit | `on_rate_limit` hook verdict, else hint fallback: `WaitAndRetry{secs}` (cancellable sleep, re-issue) or `Pause` (clean `RATE_LIMITED` stop, content preserved). First anonymous 429 retries quietly (1s, no banner) | 5 waits / turn (livelock fuse) |
| context overflow at open | compact (`Overflow` trigger, escalating) + retry round | 3 / round |
| stream idle timeout | reconnect same round with exponential backoff — only while NO content has arrived; after content: persist the partial, `TIMEOUT`. The FIRST event of each attempt gets `stream_timeout × FIRST_EVENT_TIMEOUT_FACTOR` (20): TTFT covers prefill, which scales with context; inter-token gaps are decode-bound and keep the tight bound. | 5 / round |
| empty 200 (no content at all) | short backoff + re-issue; keyed on PROVIDER content, not post-hook text (a redacting hook ≠ an empty response) | 5 / turn; exhaustion → `PROVIDER_ERROR` with size-aware message |
| output truncated (`length`), no tool calls | synthetic resume nudge steering toward incremental file writes | 2 / turn |

Mid-stream provider errors and post-content 429s persist the partial assistant message
first (tool calls paired with synthetic error results), so a resume never replays a
half-executed round or sends a dangling tool call.

### II.6 Termination — the StopReason set

`STOPPED` (normal) · `MAX_ROUNDS` · `MAX_CONTINUATIONS` · `REPEAT_LOOP` ·
`TOOL_LOOP_DETECTED` · `PROVIDER_ERROR` · `TIMEOUT` · `CANCELLED` ·
`PROMPT_REJECTED` · `POLICY_DENIED` · `RATE_LIMITED`.

The two repetition fuses are distinct by definition:

- **REPEAT_LOOP** — always-on coarse fuse: the same order-independent (name, args)
  round signature for 6 consecutive rounds, with a course-correction nudge injected at
  3. Catches "same choice regardless of results".
- **TOOL_LOOP_DETECTED** — opt-in exact guard (product policy): fingerprint = tool +
  canonicalized args + effective cwd + result content + success state; warn at 3
  identical, stop at 4. Catches "no observable progress". Real user input (prompt or
  steer) resets the streak; synthetic continuations don't — an automated goal can't
  evade the guard by opening fresh turns.

### II.7 Kernel-owned safety and invariants

- **Tool-result size cap** (default 64 KiB): head+tail truncation with an elision
  marker, deterministic. The one built-in bound on mounted tools — bounds context
  bloat; everything else is the trust model (design doc §4).
- **API-validity repair is kernel-owned**, never a strategy's job: every tool_call
  paired with exactly one result, orphans dropped, danglings backfilled — applied to
  request projections, resume seeds, and compaction candidates alike.
- **Compaction invariants** (strategy proposes, kernel disposes): sacred floor (the
  leading system message + first REAL user message are never drained), a
  compute-then-commit net-loss guard (commit only if strictly smaller; a refused plan
  burns no epoch), `cache_epoch` bumped exactly once per committed compaction, and
  pressure-relief scaling so a committed compaction doesn't immediately re-trigger.
- **Prefix-cache discipline**: stored history is append-only between epochs;
  `pre_request` projections are checked append-only (warning on violation);
  `ChatOptions` are a sideband request param, never part of the prefix bytes.

### II.8 Determinism scope

`Clock` covers timestamp STAMPING only — `elapsed_ms`, the loop's sole
nondeterministic value; a fixed clock makes a run's snapshots byte-reproducible. Ids
are monotonic counters (`turn_id` per user turn; `request_id` per LLM call,
session-global; resume seeds both from snapshot high-water marks). Liveness timers,
retry backoffs, and rate-limit sleeps are event-loop timers raced against cancel —
deliberately OUTSIDE `Clock`'s scope; tests exercise them with short real durations.
