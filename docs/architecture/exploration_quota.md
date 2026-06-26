# Per-cell exploration quota

> Status: live (). Replaces the removed synthetic exploration
> tier ().

## Why this exists

The learning router routes each request to a `(model, reasoning_effort)`
**cell**. Its quality signal starts empty (`quality_score` NULL on ~all rows).

As of 2026-06-27 the router does **explore-then-exploit**: with no differentiated
quality prediction (cold start, or no KNN neighbours for this prompt) it picks a
**random** capable cell — honest exploration, no hand-tuned prior. Once the
predictor returns differentiated scores it exploits (cheapest capable cell above
the quality bar). So cold-start traffic already spreads across the grid on its
own (`routing/router.py`).

The quota is the **floor** on top of that: random exploration only covers cells
*in expectation*, and once exploitation kicks in it would otherwise starve the
non-best cells. The quota guarantees every compatible cell keeps getting at least
its share of recent traffic, so the quality signal keeps refreshing across the
whole grid instead of decaying. See `docs/glossary.md` for vocabulary.

## The rule

**Every compatible cell, within its lane, must get at least its share of a fixed
exploration budget split evenly across the lane's candidate cells —
`exploration_budget_pct / n_candidates` of recent traffic. Mandatory; no
exceptions.**

When a candidate cell is below its floor, the turn is steered to it
(least-sampled first); once every candidate meets its floor, routing is
untouched. Lane scope is implicit — the enforcer runs on the router's post-gate
candidate pool, which is already filtered to the lane (`auto` floors all cells,
`remote-only` remote cells, `local-only` local cells).

### Why a split budget, not a flat per-cell floor

A flat per-cell floor `f` reserves `n * f` of all traffic, which grows with the
grid and is only feasible while `n <= 1/f` (a flat 1% breaks past 100 cells, and
well before that it crowds out the router's own cost/quality pick). Splitting a
fixed *budget* `E` evenly — each cell gets `E / n` — keeps the **total** forced
exploration bounded by `E` no matter how many cells exist. At `n=10` the default
`E = 0.10` reproduces the prior flat 1% per cell; at `n=17` it is ~0.59% per cell
(10% total) where a flat 1% would have reserved 17%. The honest trade-off: larger
grids explore each cell less often under the same budget. `effective_floor_pct`
(`routing/quota.py`) computes the split; `quota_floor_pct` is an optional absolute
minimum floor (0 = pure even split) as a safety net on very large grids.

## Mechanism

1. The router picks its cost/quality-optimal `chosen` cell as usual.
2. `cell_sample_counts` reads each candidate cell's successful-served count over
   the `quota_window_seconds` window (30 days).
3. `select_quota_deficit_cell` (`routing/quota.py`): if any candidate's share of
   that traffic is below the effective floor (`effective_floor_pct(n_candidates,
   budget_pct=exploration_budget_pct, min_floor_pct=quota_floor_pct)`), return the
   least-sampled such cell; else `None`.
4. If a deficit cell is returned, route the turn there and stamp
   `effective_routing_mode = "quota_explore"` so forced turns are distinguishable
   in the log (excludable from natural-routing baselines, quality labels still
   usable).

There is no count cap, no bootstrap rate, no difficulty exception, and no
tool-turn exception: a turn forced to an under-floor cell is forced regardless of
its shape. The exploration budget (and the optional minimum floor) are the only
knobs.

## Observability

`/status` carries `router.exploration_quota`: the `enabled` flag,
`exploration_budget_pct`, the effective `floor_pct` (= budget / live cell count),
per-cell `{samples, share, under_floor}` (least-sampled first), and
`{total_samples, cells_total, cells_under_floor}`. Forced turns are also counted
under `effective_routing_mode = "quota_explore"` in `/status`'s per-mode stats.
The grid is the live, upstream-advertised, version-ranked cell set (de-listed
models drop out automatically).

## Config

`[auto_router]` in `~/.config/callosum/config.toml`:
`exploration_quota_enabled` (on/off), `exploration_budget_pct` (default 0.10 —
total exploration budget split evenly across the lane's candidates),
`quota_floor_pct` (default 0.0 — optional absolute minimum per-cell floor),
`quota_window_seconds` (default 30 days).
