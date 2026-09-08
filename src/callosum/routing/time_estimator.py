"""Forward time estimator: per-request wall-clock latency prediction.

This is the TIME half of the usage estimator,
implementing the shared UsageEstimator contract in
routing/usage_estimate.py and mirroring the COST estimator
(routing/cost_estimator.py) so the two stay parallel. It predicts how
long a codex request will take, in milliseconds, for BOTH remote and local
cells.

Unlike cost, time is NOT zeroed for local cells: local models are often the
SLOW path (the live log shows local lanes running 5–50× slower than remote
cells), so latency estimation matters most there. Every served request also
has a fully-observed latency_ms label, so — unlike the integer-% quota
delta on the cost side — time has no unverifiable case: finalize always
records a usable observation.

Model: t ≈ a·input_tokens + b·output_tokens + c,
i.e. TTFB-ish ingest that scales with input, decode that scales with output,
and a fixed per-call overhead c. v1 fits a == b == m (a single
per-total-token slope) plus the intercept c against total latency_ms.
The a≠b split — TTFB-per-input vs decode-per-output — is deliberately
deferred: only TOTAL latency is logged today, which makes the two
coefficients unidentifiable, so separating them needs the TTFB capture
tracked as . CellTimeModel keeps ab as
separate fields so that refit is mechanical (exactly how CellCostModel
keeps input_rate/output_rate split). This mirrors the cost v1, which
likewise blended its two rates pending the same refinement.

Output tokens come from the SHARED OutputTokenForecast (p50/p95), which
propagates into the estimate's range. Because latency has high irreducible
dispersion at fixed token counts (server load, concurrency, queueing), the
range ALSO carries residual percentiles from the per-cell fit — the band is
"output uncertainty × latency-dispersion", not output uncertainty alone. The
point uses the MEDIAN residual rather than the mean, since latency is
right-skewed (preferring percentiles over the mean per ).

Resolution discipline matches cost: a cell uses its own measured fit once it
has min_samples rows; below that it falls back to the model pool, then
the global pool (tilted SLOWER for local cells — the time analogue of cost's
catalog-priority tilt, since "local is the slow path" is a real cold-start
prior), then a flat config fallback. An operator override wins outright.

RELATION TO THE STALL GUARD: CALLOSUM_LOCAL_FIRST_BYTE_TIMEOUT_S /
..._IDLE_TIMEOUT_S are static GUARD thresholds, not estimates. This
module is a forward ESTIMATE; those guards could later become data-driven
per-cell (consume this estimator's POST-HOC per-cell p95), but that is a
future consumer, not this module's scope. The router now consumes the
pre-flight p50 estimate as a bounded scheduling tie-break inside the same cost
and quality bucket; the pre-flight ETA range is exposed separately by
/v1/eta.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from callosum.cell_grid import Cell, is_completion_model
from callosum.routing.local_performance import LocalPerformanceModel
from callosum.routing.usage_estimate import (
    Estimate,
    EstimateInput,
    FinalizedObservation,
    OutputTokenForecast,
    _percentile,
)

logger = logging.getLogger(__name__)

UNIT = "ms"

_DEFAULT_WINDOW_SECONDS = 30 * 24 * 3600


def _default_is_remote(cell: Cell) -> bool:
    """Default remote-cell predicate: a gpt-X.Y completion model is a
    remote cloud cell; anything else (local ollama/vllm-style slugs) is a
    local cell. Matches the cell-grid completion filter (and the cost
    estimator) so the two stay consistent."""
    return is_completion_model(cell.model)


# --------------------------------------------------------------------------- #
# Per-cell time model                                                         #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CellTimeModel:
    """Resolved per-cell latency coefficients, plus residual band + provenance.

    t ≈ a·input + b·output + c in milliseconds. v1 sets a == b (a
    single slope per total token); the fields stay separate so a TTFB-vs-decode
    refit can populate them independently without changing
    call sites. resid_p50 / resid_p95 are the median and upper-tail
    residuals (ms) measured around the fit — they carry the irreducible
    dispersion (load/concurrency) the coefficients cannot explain, and build
    the honest upper edge of the range. source is "cell-measured" /
    "model-measured" / "global-prior" / "override" / "fallback"; n_obs is
    the supporting row count (0 for priors/fallback).
    """

    a: float
    b: float
    c: float
    resid_p50: float
    resid_p95: float
    source: str
    n_obs: int

    def predict(self, input_tokens: float, output_tokens: float) -> float:
        """Expected (mean) latency in ms at the given token counts, before the
        residual band is applied."""
        return self.a * input_tokens + self.b * output_tokens + self.c


@dataclass(frozen=True, slots=True)
class _Sample:
    """One observed timing row: total tokens in, total latency out."""

    total_tokens: float
    latency_ms: float


def _build_timing_samples(
    usage_log_path: Path,
    *,
    window_seconds: int,
    now: float | None,
) -> dict[tuple[str, str], list[_Sample]]:
    """Per-cell (total_tokens, latency_ms) samples from the request log.

    Only status=200 rows with a positive latency and both token counts
    present feed the fit (the token counts are needed to form the slope's
    total_tokens feature). Latency itself is always logged, but rows
    lacking a usage block — which the live log shows happens on some
    under-instrumented local lanes — cannot be fit and lean on the pooled
    priors instead (documented in confirming-time-estimator.md). Unlike the
    cost side, LOCAL cells are kept: time is not zeroed for them.
    """
    if not usage_log_path.exists():
        return {}
    cutoff = (now if now is not None else time.time()) - window_seconds
    conn = sqlite3.connect(usage_log_path)
    try:
        rows = conn.execute(
            "SELECT model, reasoning_effort,"
            " COALESCE(total_tokens, prompt_tokens + completion_tokens) AS total,"
            " latency_ms"
            " FROM requests"
            " WHERE status = 200"
            "   AND latency_ms IS NOT NULL AND latency_ms > 0"
            "   AND ts_start > ?"
            "   AND model IS NOT NULL"
            "   AND prompt_tokens IS NOT NULL"
            "   AND completion_tokens IS NOT NULL",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()
    out: dict[tuple[str, str], list[_Sample]] = {}
    for model, effort, total, latency in rows:
        if model is None or total is None or latency is None:
            continue
        out.setdefault((model, effort or ""), []).append(_Sample(total_tokens=float(total), latency_ms=float(latency)))
    return out


def _fit(samples: list[_Sample], *, min_samples: int, source: str) -> CellTimeModel | None:
    """Fit latency ≈ m·total_tokens + c by ordinary least squares, plus the
    residual percentiles, or None when there is too little signal.

    A negative slope (more tokens → less time) is unphysical noise; it is
    clamped to a flat fit (m = 0, c = mean latency) rather than trusted
    or propagated. The residuals are taken around the final (possibly clamped)
    line so the band reflects the model actually used.
    """
    n = len(samples)
    if n < min_samples:
        return None
    xs = [s.total_tokens for s in samples]
    ys = [s.latency_ms for s in samples]
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 0:
        # No variation in token counts: can't separate a slope, predict the
        # mean and let the residual band carry the spread.
        m = 0.0
        c = ybar
    else:
        sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys, strict=True))
        m = sxy / sxx
        if m < 0:
            m = 0.0
        c = ybar - m * xbar
    resids = sorted(y - (m * x + c) for x, y in zip(xs, ys, strict=True))
    return CellTimeModel(
        a=m,
        b=m,
        c=c,
        resid_p50=_percentile(resids, 0.50),
        resid_p95=_percentile(resids, 0.95),
        source=source,
        n_obs=n,
    )


class TimeModelProvider:
    """Caches per-cell timing samples and resolves a CellTimeModel per cell.

    Resolution order (data once it exists, prior until then, operator whenever
    they say) — the time analogue of CostModelProvider:

      1. operator overrides ({model: [ms_per_token, base_ms]}) — win
         outright;
      2. the cell's own least-squares fit, once it has min_samples rows;
      3. the model's pooled fit (across reasoning efforts), same threshold;
      4. the global pooled fit, tilted SLOWER for local cells (the
         "local is the slow path" cold-start prior);
      5. a flat fallback_ms_per_token / fallback_base_ms when the log
         has no usable signal at all.

    Recomputed at most once per refresh_seconds to keep the aggregate scan
    off the hot path (mirrors CostModelProvider / CostRankProvider).
    """

    def __init__(
        self,
        usage_log_path: Path,
        *,
        overrides: dict[str, list[float]] | None = None,
        enabled: bool = True,
        min_samples: int = 10,
        window_seconds: int = _DEFAULT_WINDOW_SECONDS,
        fallback_ms_per_token: float = 12.0,
        fallback_base_ms: float = 500.0,
        local_slowdown: float = 4.0,
        refresh_seconds: int = 3600,
        is_remote: Callable[[Cell], bool] = _default_is_remote,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._path = usage_log_path
        self._overrides = dict(overrides or {})
        self._enabled = enabled
        self._min_samples = min_samples
        self._window_seconds = window_seconds
        self._fallback_ms_per_token = fallback_ms_per_token
        self._fallback_base_ms = fallback_base_ms
        self._local_slowdown = max(1.0, local_slowdown)
        self._refresh_seconds = refresh_seconds
        self._is_remote = is_remote
        self._clock = clock
        self._cell_samples: dict[tuple[str, str], list[_Sample]] = {}
        # Resolved-model memo, cleared on each refresh. A fit is O(n) over the
        # cell's samples, so without this every model_for (and every row of the
        # aggregate report) would re-fit from scratch — O(n²) over a big log.
        self._fit_cache: dict[tuple[str, str], CellTimeModel] = {}
        self._last_built: float | None = None

    def _maybe_refresh(self) -> None:
        now = self._clock()
        if self._last_built is None or (now - self._last_built) >= self._refresh_seconds:
            self._cell_samples = _build_timing_samples(self._path, window_seconds=self._window_seconds, now=None)
            self._fit_cache = {}
            self._last_built = now

    def _model_samples(self, model: str) -> list[_Sample]:
        out: list[_Sample] = []
        for (m, _effort), samples in self._cell_samples.items():
            if m == model:
                out.extend(samples)
        return out

    def _global_samples(self) -> list[_Sample]:
        out: list[_Sample] = []
        for samples in self._cell_samples.values():
            out.extend(samples)
        return out

    def _override_model(self, model: str) -> CellTimeModel | None:
        coeffs = self._overrides.get(model)
        if not coeffs or len(coeffs) < 2:
            return None
        per_token, base = float(coeffs[0]), float(coeffs[1])
        # A pinned model asserts its curve; the band is a flat fraction of the
        # base overhead so the range is non-degenerate but not data-claimed.
        return CellTimeModel(
            a=per_token,
            b=per_token,
            c=base,
            resid_p50=0.0,
            resid_p95=base,
            source="override",
            n_obs=0,
        )

    def _tilt_for_local(self, model: CellTimeModel, cell: Cell) -> CellTimeModel:
        """Slow down a remote-dominated global prior for a local cell. Identity
        for remote cells and when the slowdown is 1.0."""
        if self._is_remote(cell) or self._local_slowdown == 1.0:
            return model
        s = self._local_slowdown
        return CellTimeModel(
            a=model.a * s,
            b=model.b * s,
            c=model.c * s,
            resid_p50=model.resid_p50 * s,
            resid_p95=model.resid_p95 * s,
            source=model.source,
            n_obs=model.n_obs,
        )

    def model_for(self, cell: Cell) -> CellTimeModel:
        """Resolve the CellTimeModel for a cell. Overrides apply even when
        dynamic derivation is disabled."""
        override = self._override_model(cell.model)
        if override is not None:
            return override
        if not self._enabled:
            return CellTimeModel(
                a=self._fallback_ms_per_token,
                b=self._fallback_ms_per_token,
                c=self._fallback_base_ms,
                resid_p50=0.0,
                resid_p95=self._fallback_base_ms,
                source="fallback",
                n_obs=0,
            )
        self._maybe_refresh()
        cached = self._fit_cache.get(cell.as_tuple())
        if cached is not None:
            return cached
        resolved = self._resolve(cell)
        self._fit_cache[cell.as_tuple()] = resolved
        return resolved

    def _resolve(self, cell: Cell) -> CellTimeModel:
        cell_fit = _fit(
            self._cell_samples.get(cell.as_tuple(), []),
            min_samples=self._min_samples,
            source="cell-measured",
        )
        if cell_fit is not None:
            return cell_fit
        model_fit = _fit(
            self._model_samples(cell.model),
            min_samples=self._min_samples,
            source="model-measured",
        )
        if model_fit is not None:
            return model_fit
        global_fit = _fit(
            self._global_samples(),
            min_samples=self._min_samples,
            source="global-prior",
        )
        if global_fit is not None:
            return self._tilt_for_local(global_fit, cell)
        return CellTimeModel(
            a=self._fallback_ms_per_token,
            b=self._fallback_ms_per_token,
            c=self._fallback_base_ms,
            resid_p50=0.0,
            resid_p95=self._fallback_base_ms,
            source="fallback",
            n_obs=0,
        )


# --------------------------------------------------------------------------- #
# The estimator                                                               #
# --------------------------------------------------------------------------- #


class TimeUsageEstimator:
    """UsageEstimator predicting wall-clock latency per request, in ms.

    estimate returns a range-first Estimate in ms (point = p50,
    high = p95). The point pushes the forecast's p50 output count through the
    per-cell model and shifts by the median residual; the high uses the p95
    output count and the p95 residual, so the band carries BOTH output-length
    uncertainty and latency dispersion. Local cells are NOT zeroed.

    Latency is always observable, so unlike cost there is no unverifiable case:
    every estimate and finalize is verifiable=True. The source
    field (not the verifiable flag) carries cold-start-vs-measured provenance.
    """

    def __init__(
        self,
        provider: TimeModelProvider,
        *,
        is_remote: Callable[[Cell], bool] = _default_is_remote,
        local_performance_model: Callable[[Cell], LocalPerformanceModel | None] | None = None,
    ) -> None:
        self._provider = provider
        self._is_remote = is_remote
        self._local_performance_model = local_performance_model

    @property
    def id(self) -> str:
        return "time-v1"

    @property
    def unit(self) -> str:
        return UNIT

    def estimate(self, inp: EstimateInput) -> Estimate:
        if not self._is_remote(inp.cell) and self._local_performance_model is not None:
            local_model = self._local_performance_model(inp.cell)
            if local_model is not None:
                point = local_model.turn_latency_ms(
                    input_tokens=inp.input_tokens,
                    output_tokens=int(inp.output.p50),
                )
                high = local_model.turn_latency_ms(
                    input_tokens=inp.input_tokens,
                    output_tokens=int(inp.output.p95),
                )
                regime = local_model.regime_for(inp.input_tokens)
                return Estimate(
                    point=max(0.0, point),
                    low=max(0.0, point),
                    high=max(max(0.0, point), high),
                    unit=UNIT,
                    source=f"local-{regime.value}+{inp.output.source}",
                    verifiable=True,
                )
        model = self._provider.model_for(inp.cell)
        forecast: OutputTokenForecast = inp.output
        point = model.predict(inp.input_tokens, forecast.p50) + model.resid_p50
        high = model.predict(inp.input_tokens, forecast.p95) + model.resid_p95
        point = max(0.0, point)
        high = max(point, high)
        source = f"{model.source}+{forecast.source}"
        # Latency is always observable post-hoc — time has no integer-resolution
        # / unverifiable problem (unlike cost). Provenance lives in `source`.
        return Estimate(
            point=point,
            low=point,
            high=high,
            unit=UNIT,
            source=source,
            verifiable=True,
        )

    def finalize(
        self,
        request_id: int,
        *,
        cell: Cell,
        observed_output_tokens: int | None,
        observed_value: float | None,
        verifiable: bool = True,
    ) -> FinalizedObservation:
        """Record realized latency for one served request.

        observed_value is the realized latency_ms (caller computes
        (ts_end - ts_start) * 1000). Latency is always observable, so this
        is ALWAYS verifiable=True — the verifiable argument is accepted
        for protocol symmetry with the cost estimator but ignored. There is no
        local-zero branch: local cells carry real (often large) latency. The
        request log already persists the row, so this returns the observation
        rather than re-storing it.
        """
        obs = FinalizedObservation(
            request_id=request_id,
            cell=cell,
            observed_output_tokens=observed_output_tokens,
            observed_value=observed_value,
            unit=UNIT,
            verifiable=True,
            source="latency",
        )
        logger.debug(
            "time.finalize request_id=%s cell=%s/%s observed_ms=%s",
            request_id,
            cell.model,
            cell.reasoning_effort,
            obs.observed_value,
        )
        return obs


# --------------------------------------------------------------------------- #
# Aggregate predicted-vs-actual calibration                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TimeAccuracyRow:
    """Aggregate predicted-vs-actual latency for one model (or the overall pool).

    This is ALWAYS an aggregate over many rows — never a per-single-request
    comparison (one request's latency is dominated by irreducible load /
    concurrency dispersion the coefficients cannot model). ratio is
    predicted_total / actual_total (1.0 = perfect aggregate calibration);
    None when actual_total is 0. Prediction uses the MEAN fit (no residual
    shift) so an in-sample fit calibrates to ≈1.0, isolating the coefficient
    model from the band logic.
    """

    model: str
    n_rows: int
    predicted_total_ms: float
    actual_total_ms: float
    ratio: float | None


def aggregate_time_accuracy(
    usage_log_path: Path,
    provider: TimeModelProvider,
    *,
    window_seconds: int = _DEFAULT_WINDOW_SECONDS,
    now: float | None = None,
) -> dict[str, TimeAccuracyRow]:
    """Compare summed predicted vs summed actual latency over the log.

    For each eligible row the predicted latency is computed from the time
    model at the row's ACTUAL input/output token counts (isolating the
    coefficient model from forecaster error), then summed per model and
    overall and compared against the summed actual latency_ms. LOCAL cells
    are INCLUDED — time matters most there.

    Returns a map keyed by model slug plus an "__overall__" row.
    """
    result: dict[str, TimeAccuracyRow] = {}
    if not usage_log_path.exists():
        return result
    cutoff = (now if now is not None else time.time()) - window_seconds
    conn = sqlite3.connect(usage_log_path)
    try:
        rows = conn.execute(
            "SELECT model, reasoning_effort, prompt_tokens, completion_tokens, latency_ms"
            " FROM requests"
            " WHERE status = 200"
            "   AND latency_ms IS NOT NULL AND latency_ms > 0"
            "   AND ts_start > ?"
            "   AND model IS NOT NULL"
            "   AND prompt_tokens IS NOT NULL"
            "   AND completion_tokens IS NOT NULL",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()

    acc: dict[str, list[float]] = {}  # model -> [pred_sum, actual_sum, n_rows]

    def _bump(key: str, pred: float, actual: float) -> None:
        slot = acc.setdefault(key, [0.0, 0.0, 0.0])
        slot[0] += pred
        slot[1] += actual
        slot[2] += 1

    for model, effort, prompt, completion, latency in rows:
        cell = Cell(model=model, reasoning_effort=effort or "")
        prompt = int(prompt or 0)
        completion = int(completion or 0)
        time_model = provider.model_for(cell)
        pred = time_model.predict(prompt, completion)
        actual = float(latency or 0.0)
        _bump(model, pred, actual)
        _bump("__overall__", pred, actual)

    for key, (pred_sum, actual_sum, n_rows) in acc.items():
        result[key] = TimeAccuracyRow(
            model=key,
            n_rows=int(n_rows),
            predicted_total_ms=pred_sum,
            actual_total_ms=actual_sum,
            ratio=(pred_sum / actual_sum) if actual_sum > 0 else None,
        )
    return result
