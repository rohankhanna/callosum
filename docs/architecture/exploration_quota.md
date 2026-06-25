# Per-cell exploration quota

> Status: live (). Replaces the removed synthetic exploration
> tier ().

## Why this exists

The learning router routes each request to a `(model, reasoning_effort)`
**cell**. Its quality signal starts empty (`quality_score` NULL on ~all rows),
so the predictor is uniform and the cost-weighted selector collapses all traffic
to one cell (e.g. `model-a0e8`). To learn whether a cheaper cell is good enough, the
router has to actually use it sometimes. See `docs/glossary.md` for vocabulary.

## The rule

**Every compatible cell, within its lane, must get at least `quota_floor_pct`
(default 1%) of recent traffic. Mandatory; no exceptions.**

When a candidate cell is below its floor, the turn is steered to it
(least-sampled first); once every candidate meets its floor, routing is
untouched. Lane scope is implicit — the enforcer runs on the router's post-gate
candidate pool, which is already filtered to the lane (`auto` floors all cells,
`remote-only` remote cells, `local-only` local cells).

## Mechanism

1. The router picks its cost/quality-optimal `chosen` cell as usual.
2. `cell_sample_counts` reads each candidate cell's successful-served count over
   the `quota_window_seconds` window (30 days).
3. `select_quota_deficit_cell` (`routing/quota.py`): if any candidate's share of
   that traffic is below `quota_floor_pct`, return the least-sampled such cell;
   else `None`.
4. If a deficit cell is returned, route the turn there and stamp
   `effective_routing_mode = "quota_explore"` so forced turns are distinguishable
   in the log (excludable from natural-routing baselines, quality labels still
   usable).

There is no count cap, no bootstrap rate, no difficulty exception, and no
tool-turn exception: a turn forced to an under-floor cell is forced regardless of
its shape. The floor is the only knob.

## Observability

`/status` carries `router.exploration_quota`: the `enabled` flag, `floor_pct`,
per-cell `{samples, share, under_floor}` (least-sampled first), and
`{total_samples, cells_total, cells_under_floor}`. Forced turns are also counted
under `effective_routing_mode = "quota_explore"` in `/status`'s per-mode stats.
The grid is the live, upstream-advertised, version-ranked cell set (de-listed
models drop out automatically).

## Config

`[auto_router]` in `~/.config/callosum/config.toml`:
`exploration_quota_enabled` (on/off), `quota_floor_pct` (default 0.01),
`quota_window_seconds` (default 30 days).
