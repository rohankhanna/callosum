# ADR: cell_mean_prior predictor — shadow-only, magnitude-preserving per-cell mean

Date: 2026-08-04

## Status

Accepted (shadow-only — not rolled out to live routing)

> Note: this ADR was authored 2026-08-13 to record a decision enacted
> 2026-08-04 (introducing commit `ac9522b`, merged to `main` as
> `64215e1`). It records a past decision; the implementation landed
> prior to this document. The date above is the decision date. A later
> 2026-08-06 finding (recorded under Consequences) changed the rollout
> case but not the decision to land the predictor shadow-only.

## Context

After the embedding/KNN predictor was ripped out (2026-08-03) as
non-contextual — the bandit, not a lookup, is the right routing
instrument — the remaining per-cell predictor was
`CellMajorityPriorPredictor` (`src/callosum/routing/predictor/cell_prior.py`).
It buckets each quality label `{-1, 0, +1}` to its **sign** and predicts
the per-cell majority sign. A 2026-08-03 read-only shadow eval (807
labeled rows, 25 cells) showed why that instrument is weak: most cells'
majority sign equals the global majority sign, so the per-cell
majority-sign prior degenerates to a near-global prior (LOO lift
`+0.009`, far below the `0.05` rollout gate). The root cause is the
**instrument**, not the data: sign-bucketing throws away label
**magnitude** — a cell averaging `+0.2` and one averaging `+0.8` both
collapse to sign `+1` and predict the same `P(satisfy)=1.0`.

The same eval's *routing* view (per-cell mean of the continuous outcome
in `[-1, +1]`) looked far stronger: a per-cell **mean** preserves
magnitude, so `P(satisfy) = (mean + 1) / 2` lets a `+0.2` cell
(`P=0.6`) route below a `+0.8` cell (`P=0.9`). On that day's data the
mean showed `Spearman ρ = 1.0` between LOO-predicted and realized cell
means and a `+0.582` best-vs-random routing gain — framing that read
as a stable, strong per-cell signal.

## Decision

Land a second per-cell prior, `CellMeanPriorPredictor` (id
`cell_mean_prior`), **shadow-only**:

- `reload` computes the per-cell **arithmetic mean** of `row.outcome`
  (magnitude-preserving); `predict` maps it to `P(satisfy) = (mean + 1)
  / 2` clamped to `[0, 1]`. Cells with no labels stay at the cold-start
  `0.5` prior so a new cell is never unreachable.
- Keep `CellMajorityPriorPredictor` registered for A/B — this is an
  addition, not a replacement.
- **No live-routing change**: `RoutingConfig.quality_predictor` default
  stays `"uniform"`. The predictor is inert until an operator selects it.
- Ship a reusable shadow-eval harness, `routing/eval/cell_shadow_eval.py`
  (leave-one-out), replacing the `_knn_shadow_eval` harness deleted in
  the embedding/KNN rip — the rollout gate had lost its measurement
  probe. `scripts/check_cell_shadow_gate.py` is a read-only gate *tool*
  (repointing the gate to it is the operator's call, deliberately not
  baked in).

## Consequences

**Positive:**

- A magnitude-preserving per-cell prior is now available behind
  configuration, A/B-able against the sign-bucketing prior and the
  uniform no-op, without touching live routing.
- The LOO shadow harness restores a routing-relevant measurement probe
  (per-cell mean spread + stability + best-vs-random gain), replacing
  the one deleted with the KNN rip.

**Negative / accepted — the 2026-08-06 correction (the rollout case
collapsed):**

A fresh read-only re-run of `check_cell_shadow_gate.py` on the current
DB (same 807 rows / 25 cells, but per-cell means now span the **full**
`-1..+1` range — a cell at realized mean `+1.0`, another at `-1.0`,
which the 2026-08-03 spread did not reach) showed the
`routing_gain` gate is **ceiling-saturated**:

| predictor | routing_gain | vs uniform |
|---|---|---|
| `uniform` (no-op) | 1.14994 | — (baseline) |
| `cell_mean_prior` | 1.14918 | −0.00076 (loses) |
| `cell_majority_prior` | 1.14904 | −0.00090 (loses) |

Under the harness's outcome model, "route row `r` to cell `c`" scores
`cell_mean[c]` — a **row-independent, cell-level constant**. The
optimal policy collapses to "always pick the single highest-mean cell,"
which `uniform` also does via argmax tie-break. So the gate *passes*
`uniform`: it is not a gate, and the learned priors add **negative**
value vs the no-op. The `Spearman ρ = 1.0` for the mean prior is
**tautological** — `P = (mean + 1) / 2` is a monotonic transform of the
mean, so its ranking *is* the realized-mean ranking by construction; it
measures self-consistency, not generalization. The `+0.582` figure and
the "ρ = 1.0 ⇒ stable signal" framing are **not a valid rollout basis**.

**Structural limit.** A per-cell prior can only add value where *which
cell is best varies per row*, which requires counterfactual outcomes for
un-routed cells — absent from a single-outcome-per-row request log. So
the data-side lever is **ceiling-bound by construction**; the live flip
and the metric redefinition (the gate must be repointed from "vs random"
to "vs the uniform no-op," under which `cell_mean_prior` fails) stay
operator-gated (work tracker ).

**Reversibility:** normal git-history change. The predictor is behind
configuration (`quality_predictor`); the default `uniform` makes it
inert until selected. The shadow harness is read-only. This ADR records
the land-shadow-only + magnitude-preserving-mean + A/B-keep-majority
shape as the agreed one, **and** records that the 2026-08-06 re-run
collapsed the rollout case — a future agent should not re-walk the
"per-cell mean is a strong signal" path without first confronting the
ceiling-saturation finding and the counterfactual-outcome structural
limit. Related: the bandit direction
([[project_router_relevance_bandit_direction]] in memory) is the
non-ceiling-bound lever.