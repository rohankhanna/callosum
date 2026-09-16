"""Tier 2 — resumable GPU behaviour rate-matrix consumer (read-only).

The unseeded, statistical model x test behaviour rate matrix is produced by
an external benchmark suite on a GPU under bounded daily windows and consumed
read-only here. Callosum never builds it; an external producer publishes the
matrix artifact this reader expects.

Resumable / interruptible (per learning ): the
producer runs inside bounded daily GPU windows and may be paused/reclaimed at
any time, so the matrix grows across windows. This consumer tolerates partial
coverage — it reads whatever is published, reports pending until coverage
is complete, and writes a checkpoint so an operator can watch coverage accrue
across sessions. A missing matrix fail-closes to pending (the safe hold),
never red: an absent statistical matrix must not block a Tier-1-green merge.

Tier 2 is not merge-blocking on its own (GateReport.green is Tier 1 only).
It feeds the auto-promotion decision, which stays OFF
(CALLOSUM_AUTO_PROMOTION_ENABLED).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from callosum.gate.types import Tier, TierResult, TierStatus

# Expected rate-matrix artifact schema (what benchmark suite must publish):
#   {
#     "suite_version": "behavior-v1",
#     "generated_at": "2026-07-03T12:00:00Z",   # RFC 3339 UTC
#     "min_samples": 30,
#     "models": {
#       "<model_id>": {
#         "<weight_identity>|null": {
#           "<test_name>": {"passes": int, "samples": int}, ...
#         }, ...
#       }, ...
#     }
#   }
# All rates are UNSEEDED repeated-sampling pass-rates (the lower bound of the
# uncertainty band, not a boolean). See the external handoff for the full
# contract.

DEFAULT_MATRIX_PATH = Path(
    os.environ.get(
        "CALLOSUM_RATE_MATRIX_PATH",
        str(Path.home() / ".local/share/benchmark suite/rate-matrix.json"),
    )
)
DEFAULT_CHECKPOINT_PATH = Path(
    os.environ.get(
        "CALLOSUM_RATE_MATRIX_CHECKPOINT",
        str(Path.home() / ".local/share/callosum/gate/tier2-checkpoint.json"),
    )
)
# Advisory: the bounded daily GPU window the producer commits to. Recorded in
# the checkpoint for operator visibility; the consumer is read-only and does
# not enforce it.
DEFAULT_DAILY_WINDOW_S = 3600.0


def wilson_lower_bound(passes: int, samples: int, *, z: float = 1.96) -> float:
    """Lower bound of the Wilson score interval for a pass proportion.

    Duplicated from callosum.dev_loop.promotion (which ships on the
    auto-dev branch and merges later) so this tier is self-contained on main.
    Kept in sync by the unit tests in both modules.
    """
    if samples <= 0:
        return 0.0
    p = passes / samples
    z2 = z * z
    denom = 1.0 + z2 / samples
    center = (p + z2 / (2 * samples)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / samples + z2 / (4 * samples * samples))
    return max(0.0, center - margin)


@dataclass(frozen=True, slots=True)
class MatrixEntry:
    passes: int
    samples: int


class RateMatrixReader(Protocol):
    """Read-only view of the unseeded model x test rate matrix."""

    def coverage_complete(self, model: str, weight_identity: str | None, suite_version: str) -> bool: ...
    def test_rate(self, test: str, model: str, weight_identity: str | None) -> tuple[int, int]: ...


class FileRateMatrixReader:
    """Concrete reader for the JSON artifact published by benchmark suite.

    Fail-safe: a missing or malformed matrix reports no coverage and zero
    samples for every query, so the gate holds at the coverage check (the
    documented degrade-to-pending behaviour)."""

    def __init__(self, matrix_path: Path, *, expected_tests: tuple[str, ...], min_samples: int) -> None:
        self._path = matrix_path
        self._expected = expected_tests
        self._min_samples = min_samples
        self._raw: dict[str, object] | None = self._load()

    def _load(self) -> dict[str, object] | None:
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        return data

    @property
    def present(self) -> bool:
        return self._raw is not None

    @property
    def suite_version(self) -> str:
        raw = self._raw or {}
        v = raw.get("suite_version")
        return str(v) if isinstance(v, str) else ""

    def _cell(self, model: str, weight_identity: str | None) -> dict[str, MatrixEntry]:
        raw = self._raw or {}
        models = raw.get("models")
        if not isinstance(models, dict):
            return {}
        m = models.get(model)
        if not isinstance(m, dict):
            return {}
        key = weight_identity if weight_identity is not None else "null"
        w = m.get(key)
        if not isinstance(w, dict):
            return {}
        out: dict[str, MatrixEntry] = {}
        for test, entry in w.items():
            if not isinstance(entry, dict):
                continue
            passes = entry.get("passes", 0)
            samples = entry.get("samples", 0)
            if isinstance(passes, int) and isinstance(samples, int):
                out[str(test)] = MatrixEntry(passes, samples)
        return out

    def coverage_complete(self, model: str, weight_identity: str | None, suite_version: str) -> bool:
        if self._raw is None:
            return False
        if suite_version and self.suite_version and self.suite_version != suite_version:
            return False
        cell = self._cell(model, weight_identity)
        if not cell:
            return False
        for test in self._expected:
            entry = cell.get(test)
            if entry is None or entry.samples < self._min_samples:
                return False
        return True

    def test_rate(self, test: str, model: str, weight_identity: str | None) -> tuple[int, int]:
        cell = self._cell(model, weight_identity)
        entry = cell.get(test)
        if entry is None:
            return (0, 0)
        return (entry.passes, entry.samples)


@dataclass(frozen=True, slots=True)
class Tier2Config:
    matrix_path: Path = DEFAULT_MATRIX_PATH
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH
    suite_version: str = "behavior-v1"
    expected_tests: tuple[str, ...] = ()
    models: tuple[tuple[str, str | None], ...] = ()  # (model_id, weight_identity) pairs to evaluate
    min_samples: int = 30
    threshold: float = 0.9
    daily_window_s: float = DEFAULT_DAILY_WINDOW_S


@dataclass(frozen=True, slots=True)
class Tier2Checkpoint:
    """Persisted progress across daily windows. An operator reads this to see
    how coverage is accruing without re-deriving it from the matrix."""

    last_read_epoch_s: float
    matrix_digest: str
    suite_version: str
    coverage: dict[str, dict[str, dict[str, dict[str, int]]]]  # model -> weight -> test -> {passes,samples}

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_read_epoch_s": self.last_read_epoch_s,
            "matrix_digest": self.matrix_digest,
            "suite_version": self.suite_version,
            "coverage": self.coverage,
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")

    @classmethod
    def empty(cls, suite_version: str) -> Tier2Checkpoint:
        return cls(0.0, "", suite_version, {})


def _matrix_digest(path: Path) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return hashlib.sha256(data).hexdigest()


class ResumableTier2Runner:
    """Reads the published rate matrix, records a coverage checkpoint, and
    reports Tier 2 status. Never writes to the matrix (read-only consumer)."""

    def __init__(self, config: Tier2Config) -> None:
        self._config = config

    def run(self) -> TierResult:
        cfg = self._config
        reader = FileRateMatrixReader(cfg.matrix_path, expected_tests=cfg.expected_tests, min_samples=cfg.min_samples)
        if not reader.present:
            return self._pending("rate matrix not yet published by the sibling benchmarks repo")
        if not cfg.expected_tests or not cfg.models:
            # Refuse to report a vacuous green over zero configured cells. If
            # the operator has not declared expected_tests/models, stay pending
            # so a published matrix is never misread as "0 cell(s) x 0 test(s)
            # above threshold" — a real pass. Configure via `callosum gate
            # --tier2-expected-test ... --tier2-model ...`.
            return self._pending(
                "tier2 unconfigured: set --tier2-expected-test and --tier2-model before evaluating cells"
            )
        coverage = self._snapshot(reader)
        Tier2Checkpoint(time.time(), _matrix_digest(cfg.matrix_path), reader.suite_version, coverage).write(
            cfg.checkpoint_path
        )
        # Evaluate every configured (model, weight) cell. Pending until ALL
        # cells have complete coverage; red only when coverage IS complete and
        # a test's lower bound falls under the threshold.
        incomplete: list[str] = []
        worst: list[str] = []
        for model, weight in cfg.models:
            if not reader.coverage_complete(model, weight, cfg.suite_version):
                incomplete.append(f"{model}@{weight}")
                continue
            for test in cfg.expected_tests:
                passes, samples = reader.test_rate(test, model, weight)
                lb = wilson_lower_bound(passes, samples)
                if lb < cfg.threshold:
                    worst.append(f"{model}@{weight}::{test} lb={lb:.3f} ({passes}/{samples})")
        if incomplete:
            # The real coverage snapshot is already checkpointed above; do NOT
            # call _pending here (it would overwrite the snapshot with an empty
            # marker). Report pending so the operator sees coverage is still
            # accruing across daily GPU windows.
            return TierResult(
                Tier.TIER2,
                TierStatus.PENDING,
                f"coverage incomplete, awaiting more daily GPU windows: {', '.join(incomplete)}",
            )
        if worst:
            return TierResult(Tier.TIER2, TierStatus.RED, "rate below threshold: " + "; ".join(worst))
        return TierResult(
            Tier.TIER2,
            TierStatus.GREEN,
            f"tier2 green: {len(cfg.models)} cell(s) x {len(cfg.expected_tests)} test(s) above threshold",
        )

    def _snapshot(self, reader: FileRateMatrixReader) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
        out: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
        for model, weight in self._config.models:
            weight_key = weight if weight is not None else "null"
            per_test: dict[str, dict[str, int]] = {}
            for test in self._config.expected_tests:
                passes, samples = reader.test_rate(test, model, weight)
                per_test[test] = {"passes": passes, "samples": samples}
            if model not in out:
                out[model] = {}
            out[model][weight_key] = per_test
        return out

    def _pending(self, reason: str) -> TierResult:
        # Still record a checkpoint so the operator sees "matrix absent" as a
        # resumable state, not a silent hole.
        with contextlib.suppress(OSError):
            Tier2Checkpoint.empty(self._config.suite_version).write(self._config.checkpoint_path)
        return TierResult(Tier.TIER2, TierStatus.PENDING, reason)
