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

## Temporary xhigh cap

While the peer-quality and cost models are still stabilizing, automatic routing
also applies a temporary `xhigh` guardrail before the router and exploration
quota see the candidate pool. If recent successful `xhigh` traffic is at or
above `xhigh_cap_pct` (default 1%) over `xhigh_cap_window_seconds` (default seven
days), or if the next `xhigh` choice would push the share over that cap, `xhigh`
cells are removed from automatic routing while non-`xhigh` alternatives are
available. Explicit `callosum:<source>/<model>:xhigh` pins are not filtered; if
the only viable lane is `xhigh`, the pool is preserved rather than fabricating
an outage.

This is intentionally a temporary operating cap, not a learned quality claim.
work tracker tracks the later removal once the learned router has enough reliable
cost/quality evidence to spend high reasoning effort deliberately.

## Feasibility + cooldown (the doom-loop fix, )

The floor says "every compatible cell must get its share of traffic" — but
"compatible" used to mean only *capability + context-window fit*. A slow local
cell that *fits* a 262K-token prompt but cannot *finish* it within the local
stall-guard first-byte timeout (`CALLOSUM_LOCAL_FIRST_BYTE_TIMEOUT_S`, default
180s) would be forced onto the turn, run to the timeout, return `status=0`,
and record no completed sample. A cell with zero completed samples stays under
its floor forever, so the even-split quota re-targeted the **same** slow cell
indefinitely — GPU pegged, every turn failed. Live-confirmed 2026-06-28.

The fix is two halves, shipped together (regression test:
`tests/unit/routing/test_exploration_doom_loop.py`, a failing→green tiered-gate
case seeding ):

1. **Feasibility-aware candidate filter** (`src/callosum/routing/feasibility.py`).
   A cell is eligible for *forced* exploration only when the forward time
   estimator () predicts its p95 completion sits under the
   stall-guard first-byte budget. The estimator is promoted from a soft
   scheduling tie-break to a **hard constraint on the exploration path**
   (cold-start random selection in `router.py` + quota forcing in `app.py`);
   the exploit (cost/quality) path keeps it as a soft tie-break, so a
   learned-optimal cell is never hard-excluded for being slow.

   **Cold-cell grace — the deliberate tradeoff.** A cell whose latency
   prediction rests on a model-pooled / global-prior / flat-fallback prior
   rather than its own measured fit is *not* hard-excluded on that prior. The
   prior is uncertain, and excluding on it would starve exploration of exactly
   the cold cells we most need to sample (zero measured rows). Such a cell
   stays eligible for one forced attempt; if it then times out, the cooldown
   below bounds the damage to one wasted turn. The cost is at most one timed-out
   forced turn per cold cell — the price of not starving exploration.

2. **Post-timeout cooldown** (`cell_grid.recent_quota_cooldown_cells` +
   `quota.select_quota_deficit_cell(cooldown=…)`). A cell that just timed out
   on a `quota_explore` turn (most recent forced attempt `status != 200` with
   no `status=200` row since, within the cooldown window) is skipped for the
   next selection cycle. The floor's purpose is *coverage*, and coverage needs
   a *completed* sample, not a timeout — so the cell re-arms only after it
   records a real completed sample (organic traffic on a small turn counts too;
   any completion proves the cell can finish something). This is the backstop to
   feasibility: it bounds a cold cell's wasted forced turns to one even when the
   latency prior could not rule it out.

If every candidate is a known-slow measured cell (all over budget) the quota
forces nothing and the router's optimal pick stands — burning a forced turn on
a certain timeout is worse than not forcing. If every candidate is cooling down
the quota likewise forces nothing.

## Observability

`/status` carries `router.exploration_quota`: the `enabled` flag,
`exploration_budget_pct`, the effective `floor_pct` (= budget / live cell count),
per-cell `{samples, share, under_floor}` (least-sampled first), and
`{total_samples, cells_total, cells_under_floor}`. Forced turns are also counted
under `effective_routing_mode = "quota_explore"` in `/status`'s per-mode stats.
The grid is the live, upstream-advertised, version-ranked cell set (de-listed
models drop out automatically).

`/status` also carries `router.xhigh_cap` with the enabled flag, cap percentage,
and cap window.

## Config

`[auto_router]` in `~/.config/callosum/config.toml`:
`exploration_quota_enabled` (on/off), `exploration_budget_pct` (default 0.10 —
total exploration budget split evenly across the lane's candidates),
`quota_floor_pct` (default 0.0 — optional absolute minimum per-cell floor),
`quota_window_seconds` (default 30 days), `xhigh_cap_enabled` (default on),
`xhigh_cap_pct` (default 0.01), and `xhigh_cap_window_seconds` (default seven
days). Doom-loop fix knobs (): `exploration_feasibility_enabled`
(default on — escape hatch to fall back to window-fit-only filtering) and
`exploration_cooldown_enabled` / `exploration_cooldown_window_seconds` (default
on / 600s — how long a timed-out forced turn keeps a cell out of the forced
pool). The feasibility budget itself is the env-tunable
`CALLOSUM_LOCAL_FIRST_BYTE_TIMEOUT_S` (default 180s).
