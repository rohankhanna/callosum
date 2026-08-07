# ADR: Tiered merge/promotion gate harness (Tier-1 CPU / Tier-2 GPU rate matrix / Tier-3 shadow canary)

Date: 2026-07-03

## Status

Accepted

> Note: this ADR was authored 2026-08-13 to record a decision enacted
> 2026-07-03 (merged to `main` as `75fef44`). It records a past decision;
> the implementation landed prior to this document. The date above is the
> decision date.

## Context

Callosum's auto-dev/routing work produced many feature branches merging
to `main`. There was no mechanical merge gate: a branch could land with
red unit tests, lint/type drift, or a regression in a previously fixed
bug, and the breakage would only surface later in live routing. At the
same time, the project wanted a promotion path for model-behavior
changes that needed GPU confirmation before being trusted for live
routing, and a shadow canary layer that could statistically auto-revert
a bad cell without a human watching.

A single "everything must pass" gate was the wrong shape. The fast,
deterministic, every-merge checks (unit tests, lint, types, the
inaugural bug regressions) must block every merge, but GPU behavior
confirmation is slow, resumable, and not always available (it depends on
a sibling repo's published rate matrix), and shadow canary evaluation is
statistical and ongoing — neither should block a merge or run on every
commit.

## Decision

Adopt a three-tier gate harness (`src/callosum/gate/`: `types`,
`harness`, `tier1`, `tier2`, `tier3`), invoked via `callosum gate` or
`scripts/run_gate.py`, where only Tier-1 blocks merges:

- **Tier-1 — BLOCKING, fast, deterministic.** Runs the two inaugural bug
  regressions first (the exploration doom-loop fix and the cell-level
  failover fix) as a fast-fail, then the full `tests/unit` suite, `ruff
  check .`, and `mypy --strict src/callosum`. Exit 0 = green-for-merge;
  1 = merge-blocked. This is the only tier whose redness blocks a merge.
- **Tier-2 — resumable, NOT blocking.** `FileRateMatrixReader` consumes
  a GPU behavior rate matrix produced by the sibling `benchmark suite`
  repo (read-only; callosum never builds it). Fail-closes to `pending`
  when the matrix is absent — a pending Tier-2 does NOT block a
  Tier-1-green merge. Configured via `callosum gate --tier2-*` flags
  (expected tests, models + weights, min samples, threshold, suite
  version). Bounded daily GPU windows; resumable via a disk checkpoint.
- **Tier-3 — shadow/canary, NOT blocking.** `ShadowCanaryGuard` keeps
  per-cell rolling counts on disk and returns a statistical auto-revert
  decision (Wilson lower-bound margin + hard zero-passes revert). It
  returns a decision; it never flips the live flag itself.

The gate gates **merge**, not autonomy activation. The auto-dev
promotion master switch (`CALLOSUM_AUTO_PROMOTION_ENABLED`) stays OFF
independently — a green gate does not turn autonomy on.

## Consequences

**Positive:**

- `main` stays green by construction: a red Tier-1 blocks the merge,
  covering regressions, lint, and types every time.
- Slow / sometimes-unavailable evidence (GPU behavior, shadow canary)
  feeds the gate without blocking it — Tier-2 pending and Tier-3
  statistical decisions are merge-non-blocking.
- The inaugural bug regressions run first, so a regression in a
  previously fixed bug fails fast before the full suite.

**Negative / accepted:**

- Tier-1 runs `ruff check .` but NOT `ruff format --check`; formatting
  drift is not merge-blocking at the gate. The canonical `make verify`
  path adds `format-check` and is the stricter flow (documented in
  `CONTRIBUTING.md`); run `make verify`, not just the gate, before
  merging.
- Tier-2's value depends on the sibling `benchmark suite` repo publishing
  the rate matrix; when it is absent, Tier-2 is pending (not green), so
  GPU confirmation is best-effort, not guaranteed.

**Reversibility:** the gate is a normal git-history change. Removing a
tier or changing the blocking set is a new decision; this ADR records
the three-tier, only-Tier-1-blocks shape as the agreed one.