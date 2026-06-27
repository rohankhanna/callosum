# Handoff → benchmark suite: continuous unseeded model×test rate matrix

Status: Draft handoff (for the benchmark suite agent/operator). This is a
request, not an edit. callosum does not modify benchmark suite; the owning
repo's `The Project Documentation`,  gate, public/private boundary, and
verification path remain in force.

## Why callosum is asking

callosum is adding a statistical auto-promotion gate for AI-authored
changes (see callosum `docs/adr/2026-06-27-auto-dev-ticket-queue-
statistical-promotion.md`). Tier 1 of that gate needs a model-behavior
signal callosum must NOT produce itself — a continuous model×test
**rate** matrix. By callosum's repo-intent boundary, a reproducible
model-quality benchmark is a different product and belongs in
benchmark suite. callosum will consume it read-only.

## What is requested (intended change)

A continuously-maintained matrix of **unseeded behavior rates**:

- For each `(test, model, weight_identity)`, sample the model **N times
  with no seed** and record a **pass-rate** with a sample count and an
  uncertainty band — not a single boolean. Production is unseeded;
  seeding would measure behavior production never exhibits.
- A **background daemon on the GPU host** keeps the matrix populated
  opportunistically (GPU-idle-gated, serial to avoid OOM — mirror the
  pattern callosum's `capability/scheduler.py` already uses). This is
  never a CPU / GitHub-CI job.
- **Full coverage is the convergence target, not an instantaneous
  guarantee:** new test → backfill across all models; new model (or new
  `weight_identity`) → run the full suite. Prioritize stalest /
  least-sampled / newest first. Expose per-cell sample-count + freshness
  so consumers know what is still thin.
- The suite is **append-only and growing**; results record pass AND fail
  (the daemon does not gate — gating is the consumer's job).

## Read interface callosum needs (the contract)

Expose a read-only query (file artifact, SQLite, or loopback endpoint —
benchmark suite' choice) answering, per `(model, weight_identity,
suite_version)`:

- `coverage_complete: bool` — every test in the suite version has ≥ a
  minimum sample count for this model+weight_identity.
- per-test: `pass_rate`, `sample_count`, `uncertainty_band`,
  `last_sampled_at`.

callosum's gate computes `benchmarks_green := coverage_complete AND
(pass_rate ≥ τ at ≥ N with band not straddling τ)`. callosum owns τ, N,
and the gate; benchmark suite owns the suite, the sampling, and the matrix.

## Acceptance criteria

- Matrix records unseeded multi-sample rates (not booleans, no seed),
  keyed by `weight_identity`.
- Background GPU-local daemon maintains coverage and exposes
  sample-count + freshness per cell.
- A stable read interface returns `coverage_complete` + per-test rate/
  band/count/last-sampled.
- The owning repo runs its OWN build-vs-buy  scan first:
  **DeepEval / Promptfoo / Confident AI are strong ADOPT candidates as
  the runner** — do not hand-roll an eval framework if one fits. The
  novel part is unseeded-rate accumulation + GPU-local continuous
  coverage + `weight_identity` keying, not the test-execution plumbing.

## Likely files / areas (owning repo's call)

A sampling daemon, a results store (rates + counts + freshness), a suite
registry, and the read interface. Names and layout are benchmark suite'
decision.

## Boundary constraints

- No LLM-judge in the rate signal callosum consumes for its gate (callosum
  keeps LLM-judge peer-quality as a separate *routing* input, never in the
  promotion gate).
- Candidate to absorb later: callosum's capability probe suite
  (`tool_call_shape`, `reasoning_channel`) is benchmark-shaped but
  currently routing-coupled (it emits `adapter_hint`s callosum consumes).
  Raise migration as a separate decision; do not assume it.
- Verification: keep benchmark suite' own canonical verification path; this
  handoff adds a consumer, not a coupling that bypasses it.
