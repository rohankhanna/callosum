# Confirming the forward time estimator

The forward time estimator (`src/callosum/routing/time_estimator.py`) predicts,
per request, how long a codex call will take in **milliseconds**, for BOTH
remote and local cells. It is the sibling of the
[cost estimator](confirming-cost-estimator.md): both consume the SAME shared
`OutputTokenForecaster` and the same request-log substrate, so they cannot
diverge on assumed output length.

Model: `t ≈ a·input_tokens + b·output_tokens + c`. v1 fits `a == b` (one slope
per total token) plus the intercept `c` against the log's `latency_ms`; the
TTFB-vs-decode split (`a ≠ b`) is deferred until first-byte time is captured,
because only total latency is logged today.

Two things make time **different from cost**:

* **Local cells are NOT zeroed.** Local models are often the slow path — the
  live log shows local lanes running 5–50× slower than remote cells — so local
  latency is where the estimate matters most.
* **Latency is always observable**, so there is no integer-resolution /
  unverifiable case. Every estimate and `finalize()` is `verifiable=True`;
  cold-start-vs-measured provenance lives in the `source` field, not the
  verifiable flag.

The request log lives at `~/.local/state/callosum/requests.sqlite`.

## 1. Aggregate predicted-vs-actual (the calibration check)

One request's latency is dominated by **irreducible dispersion** (server load,
concurrency, queueing) that the coefficients cannot model, so the estimator is
validated **only in aggregate**, never per single request.
`aggregate_time_accuracy()` sums predicted vs actual latency over many rows per
model and overall:

```python
from pathlib import Path
from callosum.routing.time_estimator import TimeModelProvider, aggregate_time_accuracy

db = Path.home() / ".local/state/callosum/requests.sqlite"
p = TimeModelProvider(db, min_samples=10, window_seconds=10**9)  # all-time
acc = aggregate_time_accuracy(db, p, window_seconds=10**9)
for model, r in sorted(acc.items(), key=lambda kv: -kv[1].n_rows):
    ratio = f"{r.ratio:.3f}" if r.ratio is not None else "n/a"
    print(f"{model:40} rows={r.n_rows:6d} "
          f"pred_s={r.predicted_total_ms/1000:9.0f} actual_s={r.actual_total_ms/1000:9.0f} "
          f"ratio={ratio}")
```

`ratio = predicted_total / actual_total`; **1.0 = perfect aggregate
calibration**. Prediction uses the MEAN fit at each row's *actual* token counts
(no residual shift), so an in-sample least-squares fit calibrates to ≈1.0 —
this isolates the coefficient model from both forecaster error and the band
logic.

Observed on the live log (2026-06-17, all-time, in-sample): overall ratio
**0.998**; every cell with its own measured fit (the five remote models plus
`model-a0a1`) at **1.000**. The under-sampled local
lanes (`model-a0b0` 6 rows, `model-a0b5-...` 2 rows) sit *below*
`min_samples` and therefore ride the model/global prior, so their ratios are
off (0.13–1.17) — the expected cold-start behavior, not a bug.

A drift far from 1.0 for a *measured* cell means its latency regime has shifted
(e.g. a model swap or a busier host) — shorten `window_seconds` so the fit
tracks the current regime.

## 2. Which cells have enough signal to be trusted (vs cold-start prior)?

A cell's measured fit is trusted only after `time_estimate_min_samples`
(default 10) **timing rows with a usage block**. Below that it falls back to
the model pool, then the global pool (slowed for local cells by
`time_estimate_local_slowdown`), then the flat
`time_estimate_fallback_ms_per_token` / `time_estimate_fallback_base_ms`.

```sql
SELECT model, reasoning_effort,
       COUNT(*) AS n_rows,
       SUM(prompt_tokens IS NOT NULL AND completion_tokens IS NOT NULL) AS with_tokens,
       AVG(latency_ms) AS avg_ms
FROM requests
WHERE status = 200 AND latency_ms IS NOT NULL AND latency_ms > 0
  AND ts_start > strftime('%s','now') - 30*86400
GROUP BY model, reasoning_effort
ORDER BY n_rows DESC;
```

> **Local usage-block gap (token-count fallback).** The fit needs
> `prompt_tokens` + `completion_tokens` to form the `total_tokens` feature.
> Latency itself is always logged, but the live log shows some local lanes
> emit *no* usage block: the `*-responses-proxy` lane populates tokens ~94% of
> the time, while direct `*-ollama` / `*-vllm` lanes frequently do not
> (`model-a0b0` 6/41 rows, `model-a0a4` 0/9). Those rows
> are excluded from the *fit* (feature side) but their latency is still
> recorded by `finalize()`. At predict-time the input side comes from
> `features._approx_tokens` (always available), so prediction still works via
> the pooled prior until a lane's own usage instrumentation improves.
> Capturing usage blocks on the direct local lanes is the way to graduate them
> off the prior; the TTFB split () is the orthogonal
> refinement that would let `a ≠ b`.

## 3. Spot-checking a pre-flight estimate

```python
from callosum.cell_grid import Cell
from callosum.routing.time_estimator import TimeUsageEstimator, TimeModelProvider
from callosum.routing.usage_estimate import EstimateInput, OutputTokenForecaster

db = ...  # requests.sqlite path
forecaster = OutputTokenForecaster(db)
est = TimeUsageEstimator(TimeModelProvider(db, min_samples=10))

cell = Cell("model-a0e8", "medium")
inp = EstimateInput(cell=cell, input_tokens=8000, output=forecaster.forecast(cell, 8000))
e = est.estimate(inp)
print(f"≈{e.low/1000:.1f}s–{e.high/1000:.1f}s (source={e.source})")
```

The band carries BOTH output-length uncertainty (forecast p50→p95) and per-cell
**residual dispersion** (the spread of latency at fixed tokens), and the point
uses the *median* residual because latency is right-skewed. A `source` starting
with `fallback` / `global-prior` means the estimate rests on a cold-start prior
rather than the cell's own measured fit — surface it as a rough guess. A
**local** cell returns a real positive estimate, never 0.

## 4. Relation to the stall guard (not changed here)

`CALLOSUM_LOCAL_FIRST_BYTE_TIMEOUT_S` and `..._IDLE_TIMEOUT_S` are static GUARD
thresholds that abort a stalled stream; they are **not** estimates and are
untouched by this work. The natural next consumer is to make them data-driven
per cell from this estimator's post-hoc per-cell p95 instead of global
constants — a documented follow-up, not wired here (the same way the cost
session deferred the `/status` range surface and the bandit reward term).

## Configuration

All knobs live under `AutoRouterConfig` (`src/callosum/config.py`):
`time_estimate_enabled`, `time_estimate_min_samples`,
`time_estimate_window_seconds`, `time_estimate_fallback_ms_per_token`,
`time_estimate_fallback_base_ms`, `time_estimate_local_slowdown`,
`time_estimate_refresh_seconds`, `time_estimate_overrides`
(`{model_slug: [ms_per_token, base_ms]}`, wins outright), plus the shared
forecaster's `output_forecast_min_obs` and `output_forecast_fallback_ratio`.
