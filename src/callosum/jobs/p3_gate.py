"""P3 promotion-gate evaluation for peer-quality KNN routing.

This module is the machine-checkable release condition for flipping the
quality predictor from uniform to knn (work tracker ). It
consumes a peer-quality shadow report and asserts the P3 release conditions.

P3 release model (grounded in  + the 2026-07-01 measurement):

  1. VOLUME      (necessary)  embedded peer-quality labels >= MIN_LABELS
  2. COVERAGE    (necessary)  >= MIN_CELLS cells each with >= MIN_PER_CELL labels
  3. METRIC      (binding)     leave-one-out KNN exact agreement must beat the
                              majority-class baseline by MIN_LIFT (a margin)

The binding gate is #3. Volume and coverage alone do NOT justify the flip:
accruing near-single-class labels to higher N does not help, because a
near-constant target is unlearnable and the trivial majority baseline
dominates. A DISCRIMINATION diagnostic (max single-class fraction) is
reported but NOT blocking — the metric gate is its rigorous form.

This is shadow-mode evaluation: it never switches live routing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from callosum.jobs.apply_peer_quality_labels import run as run_labeler
from callosum.routing.labeler.peer_quality import DEFAULT_SHADOW_CANDIDATE, peer_quality_shadow_report

# Defaults grounded in the operator decision (, 2026-06-26)
# and the code minimums (routing/labeler/peer_quality.py: MIN_SHADOW_*).
DEFAULT_MIN_LABELS = 100  # operator: "re-decide the flip at n>=100"
DEFAULT_MIN_CELLS = 2  # MIN_SHADOW_CELLS
DEFAULT_MIN_PER_CELL = 10  # MIN_SHADOW_EVALS
DEFAULT_MIN_LIFT = 0.05  # "provably beats the majority-class baseline by a margin"
DEFAULT_MAX_CLASS_FRACTION = 0.70  # discrimination diagnostic, not blocking
# The gate should evaluate the full current peer-label pool by default. The
# live 2026-07-08 measurement showed that bounded recent slices were unstable:
# the same label pool could narrowly fail at 50 rows, pass at 200, then fail
# on the full pool. A bounded sample remains available for ad-hoc diagnostics.
DEFAULT_SAMPLE_LIMIT = 0


def _label_distribution(report: dict[str, Any]) -> dict[str, int]:
    ev = report.get("knn_shadow_eval") or {}
    dist = ev.get("label_distribution") or {}
    return {str(k): int(v) for k, v in dist.items()}


def _max_class_fraction(dist: dict[str, int]) -> float:
    total = sum(dist.values())
    if total <= 0:
        return 1.0
    return max(dist.values()) / total


def evaluate_gate(
    report: dict[str, Any],
    *,
    min_labels: int = DEFAULT_MIN_LABELS,
    min_cells: int = DEFAULT_MIN_CELLS,
    min_per_cell: int = DEFAULT_MIN_PER_CELL,
    min_lift: float = DEFAULT_MIN_LIFT,
    max_class_fraction: float = DEFAULT_MAX_CLASS_FRACTION,
    shadow_candidate: str = DEFAULT_SHADOW_CANDIDATE,
) -> dict[str, Any]:
    """Evaluate the P3 release conditions against a shadow report dict.

    Returns a structured result with per-gate met/observed/threshold and an
    all_met boolean. all_met is True iff volume, coverage, and the
    binding metric gate are all met.
    """
    ev = report.get("knn_shadow_eval") or {}
    cells = report.get("cells") or []
    embedded = int(report.get("peer_labeled_requests_with_embeddings") or 0)

    cells_with_enough = [c for c in cells if int(c.get("labels", 0)) >= min_per_cell]

    # Gate 1: volume
    volume_met = embedded >= min_labels

    # Gate 2: coverage
    coverage_met = len(cells_with_enough) >= min_cells

    # Gate 3: metric (binding)
    metric_available = bool(ev.get("available"))
    lift = float(ev.get("knn_lift_over_majority") or 0.0)
    beats = bool(ev.get("beats_majority_baseline"))
    metric_met = metric_available and (lift > min_lift) and beats
    metric_reason: str | None = None
    if not metric_available:
        metric_reason = ev.get("reason") or "knn_shadow_eval unavailable"
    elif not beats:
        metric_reason = "knn does not beat majority-class baseline"
    elif not (lift > min_lift):
        metric_reason = f"lift {lift:.3f} <= min_lift {min_lift:.3f}"

    # Diagnostic: discrimination (not blocking)
    dist = _label_distribution(report)
    max_frac = _max_class_fraction(dist)
    discrimination_met = max_frac <= max_class_fraction

    blocking_on = []
    if not volume_met:
        blocking_on.append("volume")
    if not coverage_met:
        blocking_on.append("coverage")
    if not metric_met:
        blocking_on.append("metric")

    all_met = volume_met and coverage_met and metric_met

    baseline = ev.get("baseline_majority_class") or {}
    shadow_candidate_gate = _shadow_candidate_gate(
        ev,
        candidate=shadow_candidate,
        min_lift=min_lift,
    )
    return {
        "all_met": all_met,
        "blocking_on": blocking_on,
        "knn_shadow_ready": bool(report.get("knn_shadow_ready")),
        "gates": {
            "volume": {
                "met": volume_met,
                "observed": embedded,
                "threshold": min_labels,
            },
            "coverage": {
                "met": coverage_met,
                "observed_cells_with_min_per_cell": len(cells_with_enough),
                "threshold_cells": min_cells,
                "threshold_per_cell": min_per_cell,
            },
            "metric": {
                "met": metric_met,
                "available": metric_available,
                "observed_lift": lift,
                "threshold_lift": min_lift,
                "beats_majority_baseline": beats,
                "reason": metric_reason,
            },
            "discrimination_diagnostic": {
                "met": discrimination_met,
                "observed_max_class_fraction": round(max_frac, 4),
                "threshold_max_class_fraction": max_class_fraction,
                "label_distribution": dist,
            },
        },
        "observed": {
            "captured_opinions": int(report.get("captured_opinions") or 0),
            "embedded_labels": embedded,
            "cells": len(cells),
            "knn_exact_rate": float(ev["exact_rate"]) if ev.get("exact_rate") is not None else None,
            "baseline_majority_exact_rate": (
                float(baseline["exact_rate"]) if baseline.get("exact_rate") is not None else None
            ),
            "knn_lift_over_majority": lift,
            # Diagnostic-only comparators for . These help
            # evaluate whether a per-cell prior or KNN/prior blend has useful
            # signal without weakening the binding P3 gate above, which remains
            # the current KNN lift over the majority-class baseline.
            "shadow_comparators": ev.get("shadow_comparators") or {},
            "shadow_candidate_gate": shadow_candidate_gate,
        },
    }


def _shadow_candidate_gate(
    shadow_eval: dict[str, Any],
    *,
    candidate: str,
    min_lift: float,
) -> dict[str, Any]:
    """Evaluate one configured shadow candidate without changing P3 release.

    This is deliberately diagnostic.  all_met and blocking_on remain
    bound to the live KNN metric above until an operator explicitly changes
    the production predictor and its work tracker release condition.
    """

    comparators = shadow_eval.get("shadow_comparators") or {}
    selected = comparators.get("shadow_candidate_predictor") or {}
    selected_kind = str(selected.get("kind") or "")
    expected_kind = candidate if candidate == DEFAULT_SHADOW_CANDIDATE else "knn_cell_prior_blend"
    if selected_kind != expected_kind:
        return {
            "candidate": candidate,
            "available": False,
            "met": False,
            "reason": "configured shadow candidate is absent or mismatched",
        }
    exact_rate = selected.get("exact_rate")
    lift = selected.get("lift_over_majority")
    beats = bool(selected.get("beats_majority_baseline"))
    available = exact_rate is not None and lift is not None
    exact_rate_float = float(exact_rate) if exact_rate is not None else None
    lift_float = float(lift) if lift is not None else None
    met = available and beats and lift_float is not None and lift_float > min_lift
    reason: str | None = None
    if not available:
        reason = "configured shadow candidate unavailable"
    elif not beats:
        reason = "configured shadow candidate does not beat majority-class baseline"
    elif not met:
        reason = f"lift {lift_float:.3f} <= min_lift {min_lift:.3f}"
    return {
        "candidate": candidate,
        "available": available,
        "met": met,
        "exact_rate": exact_rate_float,
        "lift_over_majority": lift_float,
        "beats_majority_baseline": beats,
        "threshold_lift": min_lift,
        "reason": reason,
    }


def run(
    *,
    db_path: Path,
    checkpoint_path: Path | None = None,
    apply_labels: bool = False,
    min_labels: int = DEFAULT_MIN_LABELS,
    min_cells: int = DEFAULT_MIN_CELLS,
    min_per_cell: int = DEFAULT_MIN_PER_CELL,
    min_lift: float = DEFAULT_MIN_LIFT,
    max_class_fraction: float = DEFAULT_MAX_CLASS_FRACTION,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    label_batch_size: int = 200,
    shadow_candidate: str = DEFAULT_SHADOW_CANDIDATE,
) -> tuple[dict[str, Any], int]:
    """Run the P3 gate check against db_path.

    When apply_labels is True, run the shadow labeler first (requires
    checkpoint_path) to convert newly accrued opinions into embedded
    labels before evaluating the gate.

    Returns (result_dict, exit_code) where exit_code is 0 iff all
    release conditions are met.
    """
    if apply_labels:
        if checkpoint_path is None:
            raise ValueError("apply_labels=True requires checkpoint_path")
        rc = run_labeler(
            db_path=db_path,
            checkpoint_path=checkpoint_path,
            batch_size=label_batch_size,
            max_rows=None,
            min_opinions=1,
            dry_run=False,
        )
        if rc != 0:
            raise RuntimeError(f"labeler failed with rc={rc}")

    report = peer_quality_shadow_report(
        db_path,
        sample_limit=sample_limit,
        shadow_candidate=shadow_candidate,
    )
    result = evaluate_gate(
        report,
        min_labels=min_labels,
        min_cells=min_cells,
        min_per_cell=min_per_cell,
        min_lift=min_lift,
        max_class_fraction=max_class_fraction,
        shadow_candidate=shadow_candidate,
    )
    return result, (0 if result["all_met"] else 1)
