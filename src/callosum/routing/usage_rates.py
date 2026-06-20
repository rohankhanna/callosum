"""Empirical quota usage-rate reporting from the request log.

This module is deliberately separate from cost_estimator.py. The forward
estimator keeps its simple blended rate for routing stability; this reporting
layer exposes what the request log can honestly identify about cached vs
uncached input rates in each opaque Codex quota-meter unit.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from callosum.cell_grid import Cell, is_completion_model
from callosum.routing.cost_labels import (
    FIVE_HOURLY_METER,
    WEEKLY_METER,
    QuotaMeterSpec,
    cost_label_quality_summary,
    cutoff_for_window,
    quantized_cost_windows,
)

UNIT = "weekly_used_percent"
METERS = (FIVE_HOURLY_METER, WEEKLY_METER)
BASIS = "empirical_request_log"
LIMITATIONS = [
    "five_hourly_used_percent and weekly_used_percent are independent opaque ChatGPT/Codex plan-quota "
    "units, not API dollar pricing",
    "quota meters are integer-percent quantized; small requests may have unverifiable zero deltas",
    "zero-delta rows are pending burn evidence and are aggregated into serialized meter-tick windows before fitting",
    "cached token behavior is measured only from this proxy's request log when identifiable",
    "upstream prompt-cache policy, prefix matching, ordering behavior, and expiry are unverified by quota data",
    "quota reset crossover rows are excluded",
    "high-confidence quota rates use backend-serialized request windows; concurrent same-backend rows are excluded",
]


@dataclass(frozen=True, slots=True)
class _Sample:
    row_id: int
    model: str
    effort: str
    uncached_input: float
    cached_input: float
    output: float
    reasoning: float
    delta: float
    ts_start: float


def usage_rate_report(
    usage_log_path: Path,
    *,
    cells: list[Cell] | None = None,
    min_usable_samples: int = 4,
    window_seconds: int = 30 * 24 * 3600,
    now: float | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready per-cell empirical quota-rate table."""
    if not usage_log_path.exists():
        return {
            "available": False,
            "reason": "usage_log_unavailable",
            "unit": UNIT,
            "basis": BASIS,
            "limitations": LIMITATIONS,
            "rates": [],
            "meters": {},
            "relationships": [],
        }
    cutoff = cutoff_for_window(window_seconds, now=now)
    meter_reports = {
        meter.name: _meter_report(
            usage_log_path,
            meter=meter,
            cutoff=cutoff,
            cells=cells,
            min_usable_samples=min_usable_samples,
            window_seconds=window_seconds,
        )
        for meter in METERS
    }
    weekly = meter_reports[WEEKLY_METER.name]
    return {
        # Backward-compatible weekly surface.
        "available": weekly["available"],
        "reason": weekly["reason"],
        "unit": WEEKLY_METER.unit,
        "basis": BASIS,
        "limitations": LIMITATIONS,
        "metadata": weekly["metadata"],
        "rates": weekly["rates"],
        # New multi-meter surface.
        "meters": meter_reports,
        "relationships": meter_relationship_report(
            usage_log_path,
            cutoff=cutoff,
            cells=cells,
            min_usable_samples=min_usable_samples,
        ),
    }


def _meter_report(
    usage_log_path: Path,
    *,
    meter: QuotaMeterSpec,
    cutoff: float,
    cells: list[Cell] | None,
    min_usable_samples: int,
    window_seconds: int,
) -> dict[str, Any]:
    label_summary = cost_label_quality_summary(usage_log_path, cutoff=cutoff, meter=meter)
    rows = _load_rows(usage_log_path, cutoff, meter=meter)
    windows = quantized_cost_windows(usage_log_path, cutoff=cutoff, meter=meter)
    by_cell: dict[tuple[str, str], list[_Sample]] = {}
    total_by_cell: dict[tuple[str, str], int] = {}
    latest_by_cell: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (row.model, row.effort)
        total_by_cell[key] = total_by_cell.get(key, 0) + 1
        latest_by_cell[key] = max(latest_by_cell.get(key, 0.0), row.ts_start)
    for window in windows:
        key = (window.model, window.reasoning_effort)
        by_cell.setdefault(key, []).append(
            _Sample(
                row_id=window.tick_row_id,
                model=window.model,
                effort=window.reasoning_effort,
                uncached_input=window.uncached_input,
                cached_input=window.cached_input,
                output=window.output,
                reasoning=window.reasoning,
                delta=window.delta,
                ts_start=window.ts_start,
            )
        )

    if cells is not None:
        ordered_keys = {(cell.model, cell.reasoning_effort or "") for cell in cells}
    else:
        ordered_keys = set(total_by_cell) | set(by_cell)

    rates = []
    for model, effort in sorted(ordered_keys):
        cell = Cell(model=model, reasoning_effort=effort)
        if not is_completion_model(model):
            rates.append(
                _local_rate(
                    cell,
                    total_by_cell.get((model, effort), 0),
                    latest_by_cell.get((model, effort)),
                    unit=meter.unit,
                )
            )
            continue
        samples = by_cell.get((model, effort), [])
        rates.append(
            _fit_rate(
                cell,
                total_samples=total_by_cell.get((model, effort), 0),
                samples=samples,
                min_usable_samples=min_usable_samples,
                updated_at=latest_by_cell.get((model, effort)),
                unit=meter.unit,
            )
        )
    return {
        "available": bool(rates),
        "reason": None if rates else "no_usage_samples",
        "unit": meter.unit,
        "basis": BASIS,
        "limitations": LIMITATIONS,
        "metadata": {
            "meter": meter.name,
            "measured_from_weekly_quota_log": True,
            "integer_percent_quantized": True,
            "upstream_cache_policy_unverified": True,
            "window_seconds": window_seconds,
            "min_usable_samples": min_usable_samples,
            "cost_label_attribution": label_summary.to_meter_metadata(meter),
        },
        "rates": rates,
    }


def _load_rows(path: Path, cutoff: float, *, meter: QuotaMeterSpec) -> list[_Sample]:
    conn = sqlite3.connect(path)
    try:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(requests)").fetchall()}
        if meter.before_column not in columns or meter.after_column not in columns:
            return []
        db_rows = conn.execute(
            "SELECT id, model, reasoning_effort, prompt_tokens, completion_tokens,"
            " cached_tokens, reasoning_tokens,"
            f" {meter.after_column} - {meter.before_column} AS delta,"
            " ts_start"
            " FROM requests"
            " WHERE status = 200"
            "   AND quota_reset_crossover = 0"
            "   AND ts_start > ?"
            "   AND model IS NOT NULL"
            f"   AND {meter.before_column} IS NOT NULL"
            f"   AND {meter.after_column} IS NOT NULL"
            "   AND prompt_tokens IS NOT NULL"
            "   AND completion_tokens IS NOT NULL",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()
    out: list[_Sample] = []
    for row_id, model, effort, prompt, completion, cached, reasoning, delta, ts_start in db_rows:
        prompt_f = max(float(prompt or 0), 0.0)
        cached_f = min(max(float(cached or 0), 0.0), prompt_f)
        out.append(
            _Sample(
                row_id=int(row_id),
                model=str(model),
                effort=str(effort or ""),
                uncached_input=prompt_f - cached_f,
                cached_input=cached_f,
                output=max(float(completion or 0), 0.0),
                reasoning=max(float(reasoning or 0), 0.0),
                delta=float(delta or 0.0),
                ts_start=float(ts_start or 0.0),
            )
        )
    return out


def _component(rate: float | None, *, source: str, unit: str, note: str | None = None) -> dict[str, Any]:
    available = rate is not None
    return {
        "available": available,
        "rate": round(rate, 12) if rate is not None else None,
        "unit": f"{unit}_per_token",
        "source": source if available else "insufficient_data",
        "note": note,
    }


def _local_rate(cell: Cell, total_samples: int, updated_at: float | None, *, unit: str) -> dict[str, Any]:
    zero = {
        "available": True,
        "rate": 0.0,
        "unit": f"{unit}_per_token",
        "source": "local_zero",
        "note": "local cells do not consume ChatGPT/Codex plan quota",
    }
    return {
        "model": cell.model,
        "reasoning_effort": cell.reasoning_effort or None,
        "samples": {"total": total_samples, "usable": total_samples},
        "input_uncached": zero,
        "input_cached": zero,
        "output": zero,
        "reasoning": zero,
        "cache_effect": {
            "available": False,
            "rate_difference": None,
            "unit": f"{unit}_per_token",
            "source": "not_applicable",
        },
        "source": "local_zero",
        "confidence": "not_applicable",
        "updated_at": _iso(updated_at),
    }


def _fit_rate(
    cell: Cell,
    *,
    total_samples: int,
    samples: list[_Sample],
    min_usable_samples: int,
    updated_at: float | None,
    unit: str,
) -> dict[str, Any]:
    if len(samples) < min_usable_samples:
        return _insufficient_rate(
            cell, total_samples, len(samples), updated_at, "too_few_verifiable_samples", unit=unit
        )

    features = [
        ("input_uncached", np.asarray([s.uncached_input for s in samples], dtype=float)),
        ("input_cached", np.asarray([s.cached_input for s in samples], dtype=float)),
        ("output", np.asarray([s.output for s in samples], dtype=float)),
        ("reasoning", np.asarray([s.reasoning for s in samples], dtype=float)),
    ]
    active = [(name, col) for name, col in features if float(np.sum(np.abs(col))) > 0.0 and float(np.var(col)) > 0.0]
    if not active:
        return _insufficient_rate(
            cell, total_samples, len(samples), updated_at, "no_independent_token_variation", unit=unit
        )
    x = np.column_stack([col for _name, col in active])
    rank = int(np.linalg.matrix_rank(x))
    if rank < len(active):
        return _insufficient_rate(
            cell, total_samples, len(samples), updated_at, "collinear_token_features", unit=unit
        )
    y = np.asarray([s.delta for s in samples], dtype=float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    coeffs = {name: max(float(value), 0.0) for (name, _col), value in zip(active, coef, strict=True)}

    def rate_for(name: str) -> float | None:
        return coeffs.get(name)

    uncached = rate_for("input_uncached")
    cached = rate_for("input_cached")
    cache_note = None
    if uncached is not None and cached is not None and cached > uncached:
        cached = None
        cache_note = "cached_input_rate_exceeds_uncached_input_rate"
    confidence = "measured" if uncached is not None and cached is not None else "insufficient_data"
    cache_effect = {
        "available": uncached is not None and cached is not None,
        "rate_difference": round(uncached - cached, 12) if uncached is not None and cached is not None else None,
        "unit": f"{unit}_per_token",
        "source": (
            "measured_from_weekly_quota_log"
            if uncached is not None and cached is not None
            else "insufficient_data"
        ),
        "note": cache_note,
    }
    return {
        "model": cell.model,
        "reasoning_effort": cell.reasoning_effort or None,
        "samples": {"total": total_samples, "usable": len(samples)},
        "input_uncached": _component(uncached, source="measured_from_quota_log", unit=unit),
        "input_cached": _component(cached, source="measured_from_quota_log", unit=unit, note=cache_note),
        "output": _component(rate_for("output"), source="measured_from_quota_log", unit=unit),
        "reasoning": _component(rate_for("reasoning"), source="measured_from_quota_log", unit=unit),
        "cache_effect": cache_effect,
        "source": "measured_from_weekly_quota_log" if confidence == "measured" else "insufficient_data",
        "confidence": confidence,
        "updated_at": _iso(updated_at),
    }


def _insufficient_rate(
    cell: Cell,
    total_samples: int,
    usable_samples: int,
    updated_at: float | None,
    reason: str,
    unit: str,
) -> dict[str, Any]:
    return {
        "model": cell.model,
        "reasoning_effort": cell.reasoning_effort or None,
        "samples": {"total": total_samples, "usable": usable_samples},
        "input_uncached": _component(None, source="insufficient_data", unit=unit, note=reason),
        "input_cached": _component(None, source="insufficient_data", unit=unit, note=reason),
        "output": _component(None, source="insufficient_data", unit=unit, note=reason),
        "reasoning": _component(None, source="insufficient_data", unit=unit, note=reason),
        "cache_effect": {
            "available": False,
            "rate_difference": None,
            "unit": f"{unit}_per_token",
            "source": "insufficient_data",
            "note": reason,
        },
        "source": "insufficient_data",
        "confidence": "insufficient_data",
        "updated_at": _iso(updated_at),
    }


def meter_relationship_report(
    usage_log_path: Path,
    *,
    cutoff: float,
    cells: list[Cell] | None,
    min_usable_samples: int,
) -> list[dict[str, Any]]:
    five = quantized_cost_windows(usage_log_path, cutoff=cutoff, meter=FIVE_HOURLY_METER)
    weekly = quantized_cost_windows(usage_log_path, cutoff=cutoff, meter=WEEKLY_METER)
    five_by_cell = _window_sums_by_cell(five)
    weekly_by_cell = _window_sums_by_cell(weekly)
    if cells is not None:
        ordered_keys = {(cell.model, cell.reasoning_effort or "") for cell in cells}
    else:
        ordered_keys = set(five_by_cell) | set(weekly_by_cell)
    out: list[dict[str, Any]] = []
    for model, effort in sorted(ordered_keys):
        f_delta, f_count, f_rows = five_by_cell.get((model, effort), (0.0, 0, 0))
        w_delta, w_count, w_rows = weekly_by_cell.get((model, effort), (0.0, 0, 0))
        identifiable = f_count >= min_usable_samples and w_count >= min_usable_samples and f_delta > 0 and w_delta > 0
        note = None
        if not identifiable:
            note = "requires enough closed tick windows for both meters in the same cell"
        out.append(
            {
                "model": model,
                "reasoning_effort": effort or None,
                "samples": {
                    "five_hourly_windows": f_count,
                    "weekly_windows": w_count,
                    "five_hourly_rows": f_rows,
                    "weekly_rows": w_rows,
                },
                "five_hourly_per_weekly_ratio": round(f_delta / w_delta, 12) if identifiable else None,
                "weekly_per_five_hourly_ratio": round(w_delta / f_delta, 12) if identifiable else None,
                "confidence": "measured" if identifiable else "insufficient_data",
                "note": note,
            }
        )
    return out


def _window_sums_by_cell(
    windows: list[Any],
) -> dict[tuple[str, str], tuple[float, int, int]]:
    out: dict[tuple[str, str], tuple[float, int, int]] = {}
    for window in windows:
        key = (window.model, window.reasoning_effort)
        delta, count, rows = out.get(key, (0.0, 0, 0))
        out[key] = (delta + window.delta, count + 1, rows + window.row_count)
    return out


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
