# Review: One-Loop, Two-Layer Agent Architecture (L0 / L1 / L2)

**Reviewing:** `docs/one-loop-two-layer-agent.md` (commit `49c2b8cf`)
**Status:** review — approve direction, two additions requested before L0 code
**Context:** written after the k5 HR-path triage (2026-07-31): delegation dead-end,
scope-heuristic eval-skip, KB-only refinement rounds, three parallel LLM config
systems. Those incidents are the evidence base for this review.

## Verdict

The design is faithful to its own thesis — surface the LLM to the top, feed it the
query plus available resources, keep the harness thin — and the dependency rule would
have structurally prevented most of what the HR triage debugged. Approve the
direction. One soft spot where the old thicket can regrow (hooks), and two gaps worth
closing before L0 freezes (answer-quality contract, latency claim scoping).

## Where it delivers the thesis

1. **The one loop replaces the hand-coded judgment chain.** Today's
   `router → rephrase → scope-dispatch → evaluate_node → suggested_queries` becomes
   model turns: the model reads a tool result and decides to search again,
   differently, against another corpus. Round-2 refinement emerges instead of being
   plumbed. This is the judgment that was never removable — the pipeline already
   concedes it by making `evaluate_node` an LLM call, then overruling its output with
   scope strings, used-set filters, and perf flags.

2. **Policy/strategy split is architecturally enforced, not aspirational.**
   `ToolMiddleware.before` (Proceed/Allow/Ask/Deny) is where RBAC and the region
   pivot live; the tool implementation enforces the `ds_owner` filter. The model
   chooses *when* to search; it cannot choose *whose* docs it sees.

3. **`LlmProvider` kills the split-config disease by construction.** Base URL and
   model id are paired inside one capability. A config key can never again be stapled
   onto an unrelated endpoint (the background quality-eval 404: a `models_mapping`
   key sent as a model id to `EVAL_LLM_BASE_URL`).

4. **The dependency rule rejects, at review time, the cross-layer reach.** The exact
   smell fixed during the triage — `evaluate_node` reading routing's
   `ctx_skill_outputs` contextvar — is an upward import under this design and dies
   in review. §6's "declare which layer this diff touches" is the mechanical form of
   the AGENTS.md discipline.

5. **Concrete validation — the k5 HR path.** The entire accumulated HR fix set
   (`force_hr_qa` branch-0, `datasource_only` scope pin, `keep_retrieval_eval`,
   `skip_web_search` suppression) collapses into **one L2 spec**: persona +
   `mount_names=["search_hr_docs"]` + region middleware. Five patches become one
   composition. That collapse is the design working.

## Rebuttals

### R1 — Hooks are the new habitat for the old thicket (needs a §4 rule)

`user_prompt_submit → return text`, `pre_request` (mutate messages),
`on_model_response` (transform stored message) — each is a *legal* place to
reintroduce rephrase steps, query rewriting, and source steering the model never
sees. Today's pipeline did not start as a thicket; it started as one rephrase step.

**Requested addition to §4:**

> Hooks and middleware may **gate, redact, and observe** — they must never reword
> the query, pick sources, or judge sufficiency. Strategy belongs to the model; a
> hook doing strategy is a layering violation even when the imports are clean.

Without this rule the endpoint is one-loop-plus-N-heuristic-hooks, and the
composition-of-branches unpredictability returns wearing a cleaner architecture.

### R2 — No answer-quality contract for API consumers (needs a DTO)

The old eval, for all its sins, exported a structured `confidence` that the tachevu
email workflow's escalation branch consumes (`ok / no_answer / no_supporting_docs /
low_confidence`). In the loop world that signal must come from somewhere — a final
structured-output turn, a `StopReason`-adjacent verdict, or an explicit `signal`
DTO. The doc is silent on it. This is a *contract* (hard to change later), so it
belongs in the Appendix inventory before L0 freezes, not in a follow-up.

### R3 — Latency claim needs honesty-scoping (§1)

Streaming, parallel-safe tools, and prefix caching are real wins, but the loop's
rounds are **serial** LLM calls: a trivial query the pipeline answers in one
synthesis shot may cost 2–3 round-trips. The honest claim is better tail-latency and
quality on hard queries, plus optimizability of the hot path — not strictly-less
latency everywhere. Scope the claim now, or the first pipeline-vs-loop benchmark
gets weaponized against the whole restructure.

## Nits

- **`AgentSpec.mount_names` must stay per-product static.** Per-*query* dynamic
  mounting decided by code is the router reborn inside the assembly root. If dynamic
  tool selection is wanted, expose a meta-tool and let the model do it.
- **"Several thousand lines" of L0 engine is a real build.** Defensible in-house
  (self-hosted vLLM; the pcap-rca `run_tshark` tool-loop already proved the pattern
  on this codebase), but step 2's mock-tested engine is the milestone that de-risks
  everything downstream — hold L1/L2 work until it is green.

## Requested changes summary

| # | Where | Change | Blocking? |
|---|-------|--------|-----------|
| 1 | §4 rules | Add hooks-can't-do-strategy rule (R1) | Yes — before any hook lands |
| 2 | Appendix DTOs | Add answer-quality/`signal` contract (R2) | Yes — before L0 freeze |
| 3 | §1 motivation | Scope the latency claim (R3) | No — wording |
| 4 | §3 | Note mount_names is per-product static | No — wording |
