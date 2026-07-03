"""Unit tests for the Tier-2 resumable GPU rate-matrix consumer.

All on-disk artifacts use tmp_path. The reader is fail-safe (a missing or
malformed matrix reports no coverage); the runner is resumable (a partial
matrix reports pending and writes a checkpoint so an operator can watch
coverage accrue across daily GPU windows). A missing matrix must NOT be red —
it fail-closes to pending so it never blocks a Tier-1-green merge.
"""

from __future__ import annotations

import json
from pathlib import Path

from callosum.gate.tier2 import (
    FileRateMatrixReader,
    ResumableTier2Runner,
    Tier2Checkpoint,
    Tier2Config,
    wilson_lower_bound,
)
from callosum.gate.types import TierStatus

TESTS = ("behavior::completes_under_budget", "behavior::no_looped_failure")
MODEL = "model-a0b5"
WEIGHT = "default"


def _matrix_path(tmp_path: Path) -> Path:
    return tmp_path / "rate-matrix.json"


def _write_matrix(path: Path, cell: dict[str, dict[str, int]], *, suite: str = "behavior-v1") -> None:
    payload = {
        "suite_version": suite,
        "min_samples": 30,
        "models": {MODEL: {WEIGHT: cell}},
    }
    path.write_text(json.dumps(payload))


def _config(tmp_path: Path, **kw: object) -> Tier2Config:
    return Tier2Config(
        matrix_path=_matrix_path(tmp_path),
        checkpoint_path=tmp_path / "checkpoint.json",
        suite_version="behavior-v1",
        expected_tests=TESTS,
        models=((MODEL, WEIGHT),),
        min_samples=30,
        threshold=0.9,
        **kw,  # type: ignore[arg-type]
    )


def test_missing_matrix_is_pending_not_red(tmp_path: Path) -> None:
    res = ResumableTier2Runner(_config(tmp_path)).run()
    assert res.status is TierStatus.PENDING
    assert "not yet published" in res.reason
    # Checkpoint still recorded so the operator sees the resumable state.
    assert (tmp_path / "checkpoint.json").exists()


def test_malformed_matrix_is_pending(tmp_path: Path) -> None:
    _matrix_path(tmp_path).write_text("{not json")
    res = ResumableTier2Runner(_config(tmp_path)).run()
    assert res.status is TierStatus.PENDING


def test_wrong_suite_version_is_pending(tmp_path: Path) -> None:
    _write_matrix(_matrix_path(tmp_path), {t: {"passes": 200, "samples": 200} for t in TESTS}, suite="other-v2")
    res = ResumableTier2Runner(_config(tmp_path)).run()
    assert res.status is TierStatus.PENDING


def test_partial_coverage_is_pending_and_checkpoints(tmp_path: Path) -> None:
    _write_matrix(
        _matrix_path(tmp_path),
        {TESTS[0]: {"passes": 200, "samples": 200}, TESTS[1]: {"passes": 10, "samples": 10}},
    )
    res = ResumableTier2Runner(_config(tmp_path)).run()
    assert res.status is TierStatus.PENDING
    assert "coverage incomplete" in res.reason
    cp = json.loads((tmp_path / "checkpoint.json").read_text())
    assert cp["coverage"][MODEL][WEIGHT][TESTS[1]]["samples"] == 10


def test_complete_coverage_all_green_is_green(tmp_path: Path) -> None:
    _write_matrix(_matrix_path(tmp_path), {t: {"passes": 200, "samples": 200} for t in TESTS})
    res = ResumableTier2Runner(_config(tmp_path)).run()
    assert res.status is TierStatus.GREEN
    assert "1 cell(s) x 2 test(s)" in res.reason


def test_complete_coverage_but_below_threshold_is_red(tmp_path: Path) -> None:
    # 150/200 -> lower bound well under 0.9 threshold.
    _write_matrix(_matrix_path(tmp_path), {t: {"passes": 150, "samples": 200} for t in TESTS})
    res = ResumableTier2Runner(_config(tmp_path)).run()
    assert res.status is TierStatus.RED
    assert "rate below threshold" in res.reason


def test_reader_coverage_and_rate(tmp_path: Path) -> None:
    _write_matrix(_matrix_path(tmp_path), {TESTS[0]: {"passes": 50, "samples": 60}})
    reader = FileRateMatrixReader(_matrix_path(tmp_path), expected_tests=TESTS, min_samples=30)
    assert reader.present
    assert reader.suite_version == "behavior-v1"
    # One test short of min_samples -> coverage incomplete.
    assert reader.coverage_complete(MODEL, WEIGHT, "behavior-v1") is False
    assert reader.test_rate(TESTS[0], MODEL, WEIGHT) == (50, 60)
    assert reader.test_rate("missing-test", MODEL, WEIGHT) == (0, 0)


def test_null_weight_identity_round_trips(tmp_path: Path) -> None:
    payload = {
        "suite_version": "behavior-v1",
        "min_samples": 30,
        "models": {MODEL: {"null": {TESTS[0]: {"passes": 200, "samples": 200}}}},
    }
    _matrix_path(tmp_path).write_text(json.dumps(payload))
    cfg = Tier2Config(
        matrix_path=_matrix_path(tmp_path),
        checkpoint_path=tmp_path / "cp.json",
        expected_tests=(TESTS[0],),
        models=((MODEL, None),),
        min_samples=30,
    )
    res = ResumableTier2Runner(cfg).run()
    assert res.status is TierStatus.GREEN


def test_wilson_matches_promotion_semantics() -> None:
    # Perfect rate on finite n is < 1.0; zero samples is 0.0.
    assert wilson_lower_bound(0, 0) == 0.0
    assert 0.95 < wilson_lower_bound(200, 200) < 1.0


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    cp = Tier2Checkpoint(123.0, "abc", "behavior-v1", {})
    path = tmp_path / "cp.json"
    cp.write(path)
    loaded = json.loads(path.read_text())
    assert loaded["matrix_digest"] == "abc"
    assert loaded["last_read_epoch_s"] == 123.0
