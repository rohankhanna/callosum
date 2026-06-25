# Per-cell exploration quota

> Status: in progress (). Supersedes the removed synthetic
> exploration tier ().

## Why this exists

The learning router (`routing/router.py`) is supposed to route each request to
the best `(model, reasoning_effort)` **cell** for the cost. It can't, because
its quality signal is empty: `quality_score` is NULL on ~100% of request rows,
so the predictor is the `uniform` stub and the cost-weighted selector collapses
all traffic to one cell (today: `model-a0e8`). See `docs/glossary.md` for the I/O
and routing vocabulary used here.

To earn cost-efficient routing you must first **measure** how each cell
performs — which requires routing some traffic to cells the selector would not
naturally pick. That is exploration, and its price is paid up front. The two
data consumers that need this coverage:

- the **peer-quality matrix** (one cell rates a prior *different-cell text
  turn*; `peer_quality.py`), and
- the **quality predictor** (`routing/predictor/knn.py`).

The **root cause** the soak hit (): real Codex traffic
routes all message-text turns to one cell and uses other cells only for tool
turns (no judgeable text), so cross-cell text pairs — the only thing the matrix
feeds on — essentially never co-occur. A coverage floor manufactures them.

## What was removed

The synthetic traffic tier (`auto-learning-synthetic`) was deleted: it was a
dead, unbuilt emitter (no worker loop ever consumed its `synthetic_*` config;
the May-2026 batch came from an external driver, was non-streaming, stored no
text, and was never labeled — so it produced coverage counts but never a
quality label). Its reusable coverage machinery — `CellCoverage` and the
least-covered-first `exploration_order` — is **kept** and repurposed as the
quota-floor engine.

## Design

A per-cell **minimum-usage floor** on organic traffic, enforced inside the
router after the cost-weighted selector picks its optimal cell.

| Decision | Choice |
|---|---|
| Quota unit | **per cell** `(model, reasoning_effort)` — full grid coverage, accepting that callosum overrides the client's requested effort |
| Rate | **bootstrap then taper** — force ~5% of compatible turns per under-covered cell until it reaches the per-cell sample threshold, then drop that cell to a 1% maintenance floor |
| Hard turns | **protected** — turns at/above the router's `_task_difficulty` cutoff are never force-routed; they take the optimal cell |
| Eligibility | floor enforced only on **text-producing turns**, never pure tool/exec turns (a tool turn yields no opinion; upgrading its effort is pure cost) |
| Lane scope | enforced over the **post-gate candidate pool**, which the gate stack has already filtered to the lane — so `auto` floors all cells, `remote-only` floors remote cells, `local-only` floors local cells, for free |

### Mechanism

1. Router selects the optimal `chosen` cell as today.
2. If the turn is text-eligible and below the hard-difficulty cutoff, read the
   rolling per-cell served-share + sample-count over a recent window
   (`CellCoverage` / `coverage_from_db`, scoped to the lane's candidate set).
3. If any compatible candidate cell is **below its floor** (bootstrap floor
   while under threshold, else maintenance floor), steer to the most-deficient
   such cell (`exploration_order`) and override `reasoning_effort` to it.
4. Stamp provenance (`routing_mode = "quota-explore"`) so forced turns are
   distinguishable in the log: excludable from "what would the router naturally
   do" baselines, while their quality labels are still usable.

### After data accrues

Once cells clear the per-cell sample threshold and the peer-quality matrix has
enough labeled opinions ( P3 thresholds), revisit flipping
the predictor from `uniform` to `knn` so routing becomes quality-driven and the
floor drops to maintenance. The floor is the bootstrap, not the destination.
