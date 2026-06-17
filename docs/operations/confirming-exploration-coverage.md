# Confirming arm-level exploration coverage

Arm-level exploration (`src/callosum/routing/exploration.py`) biases the
**synthetic** auto-learning tier so each synthetic request targets the
*least-sampled* `(model, reasoning_effort)` cell. The goal is even coverage of
the whole cell grid, which is what the quality predictor, the measured
cost model (`routing/cost_model.py`), and the future contextual bandit need.

Exploration only acts on requests whose `requested_model` is
`auto-learning-synthetic`. It does **not** generate that traffic — an external
driver (e.g. the CE/dev loop) must send synthetic requests. If no synthetic
traffic is flowing, coverage will not grow no matter how exploration is
configured. (As of 2026-06-17 the synthetic emitter is external/dormant; the
last synthetic request was 2026-05-18.)

The request log lives at `~/.local/state/callosum/requests.sqlite`.

## 1. Is synthetic traffic flowing at all?

```sql
SELECT datetime(MAX(ts_start), 'unixepoch') AS last_synthetic
FROM requests
WHERE routing_mode = 'auto-learning-synthetic';
```

If `last_synthetic` is stale, re-enable the synthetic driver first — nothing
below will move until it is.

## 2. Is coverage spreading across cells (not collapsing)?

```sql
SELECT model, reasoning_effort, COUNT(*) AS n
FROM requests
WHERE routing_mode = 'auto-learning-synthetic'
  AND status = 200
  AND ts_start > strftime('%s','now') - 7*86400
GROUP BY model, reasoning_effort
ORDER BY n ASC;          -- least-covered cells are the next exploration targets
```

Healthy exploration shows **many distinct cells** with counts converging
toward each other over time (round-robin over under-sampled arms). A single
dominant row with everything else near zero means exploration is off
(`exploration_enabled=false`), or the grid has collapsed to one cell, or no
synthetic traffic is arriving.

`cell_grid.coverage_from_db(path, cells, routing_mode="auto-learning-synthetic")`
is the same query the router uses to pick the next target; counts here are the
counts it sees.

## 3. Are the requests big enough to move the quota needle?

`weekly_used_percent` is **integer-resolution**, so a small request logs a
per-call delta of 0 and contributes no usable cost signal. Under-covered cells
must receive requests large enough to produce a non-zero delta, or the
measured cost model never gets evidence for them.

Per-cell non-zero-delta yield (recent window, crossover excluded):

```sql
SELECT model,
       COUNT(*) AS n,
       SUM(CASE WHEN weekly_used_percent_after > weekly_used_percent_before
                THEN 1 ELSE 0 END) AS nonzero_delta,
       ROUND(AVG(weekly_used_percent_after - weekly_used_percent_before), 4) AS avg_delta
FROM requests
WHERE status = 200
  AND quota_reset_crossover = 0
  AND weekly_used_percent_before IS NOT NULL
  AND weekly_used_percent_after  IS NOT NULL
  AND ts_start > strftime('%s','now') - 30*86400
GROUP BY model
ORDER BY avg_delta;
```

A model needs at least `cost_rank_min_nonzero_samples` (default 10) **non-zero**
deltas before its measured mean is trusted by `CostRankProvider`; below that it
falls back to the catalog-priority cold-start prior. If `nonzero_delta` stays
near zero for a cell, the synthetic prompts for that cell are too small — size
them up so exploration produces real cost signal, not just more zero-delta
noise.

> Caveat: per-request delta attribution (`before` = last snapshot, `after` =
> this response) is clean only on **serialized** credential traffic;
> concurrency blurs single-row attribution and is not flagged in the data. The
> cost model averages over many rows, which is robust for *ordering* purposes;
> a strict per-credential serialization filter is a documented follow-up.

## 4. Is the measured cost_rank actually differentiating models?

Once a few cells clear the non-zero-delta threshold, organic remote routing
should stop collapsing onto a single arm and prefer the cheaper-burn model.
Compare served models for organic (`auto` / `auto-learning`) traffic over time;
a spread that tracks `avg_delta` order from step 3 confirms the cost overlay is
live. The flat `cost_rank=10` in the backends is now only the cold-start
fallback used until measured data exists.
