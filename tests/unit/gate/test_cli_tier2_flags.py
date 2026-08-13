"""Unit tests for the `callosum gate --tier2-*` flag -> Tier2Config wiring.

Pins the contract that the CLI flags populate Tier2Config exactly (expected_tests,
models with optional weight_identity, min_samples, threshold, suite_version) and
that an unflagged invocation yields an empty config — which the runner's guard
turns into pending rather than a vacuous green (work tracker ).
"""

from __future__ import annotations

import argparse

from callosum.cli import _tier2_config_from_args


def _args(**overrides: object) -> argparse.Namespace:
    base: dict[str, object] = {
        "tier2_expected_tests": None,
        "tier2_models": None,
        "tier2_min_samples": 30,
        "tier2_threshold": 0.9,
        "tier2_suite_version": "behavior-v1",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_unflagged_invocation_yields_empty_config() -> None:
    cfg = _tier2_config_from_args(_args())
    assert cfg.expected_tests == ()
    assert cfg.models == ()
    assert cfg.min_samples == 30
    assert cfg.threshold == 0.9
    assert cfg.suite_version == "behavior-v1"


def test_expected_tests_and_models_populate() -> None:
    cfg = _tier2_config_from_args(
        _args(
            tier2_expected_tests=["mmlu-pro"],
            tier2_models=["model-a0a1"],
        )
    )
    assert cfg.expected_tests == ("mmlu-pro",)
    assert cfg.models == (("model-a0a1", None),)


def test_model_weight_identity_parsed() -> None:
    cfg = _tier2_config_from_args(_args(tier2_models=["model-a0b5:default", "model-a0c7:"]))
    assert cfg.models == (("model-a0b5", "default"), ("model-a0c7", None))


def test_threshold_and_min_samples_override() -> None:
    cfg = _tier2_config_from_args(_args(tier2_min_samples=50, tier2_threshold=0.75, tier2_suite_version="behavior-v2"))
    assert cfg.min_samples == 50
    assert cfg.threshold == 0.75
    assert cfg.suite_version == "behavior-v2"
