from __future__ import annotations

from callosum.jobs.p3_gate import DEFAULT_MIN_LIFT, DEFAULT_SAMPLE_LIMIT, evaluate_gate


def _report(
    *,
    embedded: int = 520,
    cells: list[dict] | None = None,
    eval_available: bool = True,
    lift: float = 0.0,
    beats: bool = False,
    label_distribution: dict[str, int] | None = None,
    knn_ready: bool = True,
) -> dict:
    if cells is None:
        # 13 cells, each >= 10 labels by default
        cells = [{"cell": f"c{i}", "labels": 30} for i in range(13)]
    if label_distribution is None:
        label_distribution = {"-1": 10, "0": 39, "1": 1}
    return {
        "knn_shadow_ready": knn_ready,
        "peer_labeled_requests_with_embeddings": embedded,
        "captured_opinions": 2737,
        "cells": cells,
        "knn_shadow_eval": {
            "available": eval_available,
            "exact_rate": 0.78,
            "baseline_majority_class": {"exact_rate": 0.78},
            "knn_lift_over_majority": lift,
            "beats_majority_baseline": beats,
            "label_distribution": label_distribution,
        },
    }


def test_gate_fails_on_metric_when_volume_and_coverage_met() -> None:
    # Mirrors the 2026-07-01 live measurement: lift 0.0, does not beat baseline.
    report = _report(embedded=520, lift=0.0, beats=False)
    result = evaluate_gate(report)

    assert result["all_met"] is False
    assert result["blocking_on"] == ["metric"]
    assert result["gates"]["volume"]["met"] is True
    assert result["gates"]["coverage"]["met"] is True
    assert result["gates"]["metric"]["met"] is False
    assert result["gates"]["metric"]["reason"] == "knn does not beat majority-class baseline"
    # discrimination is diagnostic, not blocking
    assert result["gates"]["discrimination_diagnostic"]["met"] is False
    assert "discrimination" not in result["blocking_on"]


def test_gate_passes_when_lift_beats_margin() -> None:
    report = _report(
        embedded=520, lift=DEFAULT_MIN_LIFT + 0.01, beats=True, label_distribution={"-1": 20, "0": 20, "1": 20}
    )
    result = evaluate_gate(report)

    assert result["all_met"] is True
    assert result["blocking_on"] == []
    assert result["gates"]["metric"]["met"] is True


def test_lift_equal_to_margin_does_not_pass() -> None:
    report = _report(lift=DEFAULT_MIN_LIFT, beats=True)
    result = evaluate_gate(report)
    # strict > margin
    assert result["gates"]["metric"]["met"] is False
    assert "metric" in result["blocking_on"]


def test_volume_below_floor_blocks_even_with_good_metric() -> None:
    report = _report(embedded=50, lift=DEFAULT_MIN_LIFT + 0.1, beats=True)
    result = evaluate_gate(report)
    assert result["gates"]["volume"]["met"] is False
    assert "volume" in result["blocking_on"]
    assert result["all_met"] is False


def test_coverage_blocks_when_too_few_cells_meet_per_cell_floor() -> None:
    # only one cell reaches the per-cell floor
    cells = [{"cell": "c0", "labels": 30}] + [{"cell": f"c{i}", "labels": 3} for i in range(1, 5)]
    report = _report(embedded=520, cells=cells, lift=DEFAULT_MIN_LIFT + 0.1, beats=True)
    result = evaluate_gate(report)
    assert result["gates"]["coverage"]["met"] is False
    assert "coverage" in result["blocking_on"]
    assert result["all_met"] is False


def test_metric_unavailable_blocks_with_reason() -> None:
    report = _report(eval_available=False, lift=0.0, beats=False, label_distribution={})
    report["knn_shadow_eval"]["reason"] = "insufficient_embedded_labels"
    result = evaluate_gate(report)
    assert result["gates"]["metric"]["met"] is False
    assert result["gates"]["metric"]["reason"] == "insufficient_embedded_labels"
    assert result["gates"]["metric"]["available"] is False


def test_single_class_labels_fail_metric_via_zero_lift() -> None:
    # all-neutral labels: majority baseline matches KNN, lift 0 -> metric fails.
    report = _report(embedded=520, lift=0.0, beats=False, label_distribution={"-1": 0, "0": 50, "1": 0})
    result = evaluate_gate(report)
    assert result["gates"]["metric"]["met"] is False
    assert result["gates"]["discrimination_diagnostic"]["observed_max_class_fraction"] == 1.0
    assert result["gates"]["discrimination_diagnostic"]["met"] is False


def test_thresholds_are_configurable() -> None:
    # lift must STRICTLY exceed min_lift. With min_lift=0.0, lift=0.0 still
    # fails (0.0 > 0.0 is False) even when beats is True; any positive lift
    # passes.
    report = _report(lift=0.0, beats=True)
    result = evaluate_gate(report, min_lift=0.0)
    assert result["gates"]["metric"]["met"] is False
    report2 = _report(lift=0.01, beats=True)
    result2 = evaluate_gate(report2, min_lift=0.0)
    assert result2["gates"]["metric"]["met"] is True
    # beats=False still blocks regardless of lift
    report3 = _report(lift=0.2, beats=False)
    result3 = evaluate_gate(report3, min_lift=0.0)
    assert result3["gates"]["metric"]["met"] is False


def test_observed_block_reports_lift_and_rates() -> None:
    report = _report(embedded=520, lift=0.12, beats=True)
    result = evaluate_gate(report)
    assert result["observed"]["knn_lift_over_majority"] == 0.12
    assert result["observed"]["knn_exact_rate"] == 0.78
    assert result["observed"]["baseline_majority_exact_rate"] == 0.78
    assert result["observed"]["embedded_labels"] == 520
    assert result["observed"]["cells"] == 13


def test_default_sample_limit_uses_full_label_pool() -> None:
    assert DEFAULT_SAMPLE_LIMIT == 0
