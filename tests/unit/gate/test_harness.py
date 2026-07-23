"""Unit tests for the top-level tiered-gate runner (run_gate).

Proves the documented contract: Tier 1 is the only merge-blocking tier; a red
Tier 1 marks the gate merge-blocked; a pending Tier 2 (no rate matrix yet) does
NOT block a Tier-1-green merge. Subprocess invocation is stubbed.
"""

from __future__ import annotations

from pathlib import Path

from callosum.gate.harness import GateConfig, run_gate
from callosum.gate.tier1 import SubprocessOutput, Tier1Config
from callosum.gate.tier2 import Tier2Config
from callosum.gate.tier3 import Tier3Config
from callosum.gate.types import Tier, TierStatus

REPO_ROOT = str(Path(__file__).resolve().parents[3])


def _classify(argv: list[str]) -> str:
    joined = " ".join(argv)
    if "test_min_coverage_doom_loop" in joined or "test_cell_level_retry" in joined:
        return "inaugural"
    if "mypy" in argv:
        return "mypy"
    if "ruff" in argv:
        return "ruff"
    if "pytest" in argv:
        return "unit"
    return "unknown"


class FakeRunner:
    def __init__(self, returncodes: dict[str, int]) -> None:
        self._codes = returncodes
        self.last_output: SubprocessOutput | None = None

    def __call__(self, argv: list[str], *, cwd: str, env: dict[str, str]) -> int:
        name = _classify(argv)
        rc = self._codes.get(name, 0)
        self.last_output = SubprocessOutput(rc, "" if rc == 0 else f"fail {name}", "")
        return rc


def _green_runner() -> FakeRunner:
    return FakeRunner({"inaugural": 0, "unit": 0, "ruff": 0, "mypy": 0})


def _config(tmp_path: Path) -> GateConfig:
    return GateConfig(
        tier1=Tier1Config(),
        # Tier 2: point at an absent matrix so it deterministically reports pending.
        tier2=Tier2Config(matrix_path=tmp_path / "absent-matrix.json", checkpoint_path=tmp_path / "cp.json"),
        tier3=Tier3Config(cell_flag="cell::flag"),
        repo_root=REPO_ROOT,
    )


def test_tier1_green_with_tier2_pending_does_not_block_merge(tmp_path: Path) -> None:
    report = run_gate(_config(tmp_path), runner=_green_runner())
    assert report.green is True
    assert report.merge_blocked is False
    statuses = {r.tier: r.status for r in report.results}
    assert statuses[Tier.TIER1] is TierStatus.GREEN
    assert statuses[Tier.TIER2] is TierStatus.PENDING  # no matrix yet
    assert statuses[Tier.TIER3] is TierStatus.PENDING  # cold canary, no samples
    assert "tier1 green" in report.summary


def test_tier1_red_blocks_merge(tmp_path: Path) -> None:
    report = run_gate(_config(tmp_path), runner=FakeRunner({"inaugural": 1, "unit": 0, "ruff": 0, "mypy": 0}))
    assert report.green is False
    assert report.merge_blocked is True
    assert "MERGE BLOCKED" in report.summary


def test_tier1_only_subset(tmp_path: Path) -> None:
    report = run_gate(_config(tmp_path), tiers=(Tier.TIER1,), runner=_green_runner())
    assert report.green is True
    assert len(report.results) == 1
    assert report.results[0].tier is Tier.TIER1


def test_tier2_complete_green_reports_green_not_blocking(tmp_path: Path) -> None:
    import json

    matrix = tmp_path / "matrix.json"
    tests = ("behavior::a",)
    matrix.write_text(
        json.dumps(
            {
                "suite_version": "behavior-v1",
                "min_samples": 30,
                "models": {"m": {"default": {tests[0]: {"passes": 200, "samples": 200}}}},
            }
        )
    )
    cfg = GateConfig(
        tier1=Tier1Config(),
        tier2=Tier2Config(
            matrix_path=matrix,
            checkpoint_path=tmp_path / "cp.json",
            expected_tests=tests,
            models=(("m", "default"),),
        ),
        tier3=Tier3Config(cell_flag="cell::flag"),
        repo_root=REPO_ROOT,
    )
    report = run_gate(cfg, runner=_green_runner())
    assert report.green is True  # tier1 green gates the merge
    statuses = {r.tier: r.status for r in report.results}
    assert statuses[Tier.TIER2] is TierStatus.GREEN
