"""Unit tests for the Tier-1 deterministic CPU gate.

The subprocess layer is stubbed via a fake CommandRunner so the tier's
branching logic (fast-fail on the inaugural cases, ordering of checks, the
exact inaugural case set) is tested deterministically without spawning real
pytest/ruff/mypy processes. A separate end-to-end test runs the real Tier 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

from callosum.gate.tier1 import INAUGURAL_CASES, SubprocessOutput, Tier1Config, resolve_gate_python, run_tier1
from callosum.gate.types import TierStatus

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
    """Returns canned return codes per check; captures the tail output on
    failure so _detail has something to read."""

    def __init__(self, returncodes: dict[str, int]) -> None:
        self._codes = returncodes
        self.calls: list[list[str]] = []
        self.last_output: SubprocessOutput | None = None

    def __call__(self, argv: list[str], *, cwd: str, env: dict[str, str]) -> int:
        self.calls.append(list(argv))
        name = _classify(argv)
        rc = self._codes.get(name, 0)
        out = "" if rc == 0 else f"FAILED {name}\nassertion error in {name}"
        self.last_output = SubprocessOutput(rc, out, "")
        return rc


def test_inaugural_cases_are_the_two_bug_regression_tests() -> None:
    assert INAUGURAL_CASES == ("tests/unit/test_cell_level_retry.py",)


def test_all_green_runs_all_four_checks_in_order() -> None:
    r = FakeRunner({"inaugural": 0, "unit": 0, "ruff": 0, "mypy": 0})
    result = run_tier1(Tier1Config(), repo_root=REPO_ROOT, runner=r)
    assert result.status is TierStatus.GREEN
    assert result.green
    names = [c.name for c in result.checks]
    assert names == ["inaugural cases", "unit suite", "ruff", "mypy --strict"]


def test_inaugural_red_fast_fails_before_unit_suite() -> None:
    r = FakeRunner({"inaugural": 1, "unit": 0, "ruff": 0, "mypy": 0})
    result = run_tier1(Tier1Config(), repo_root=REPO_ROOT, runner=r)
    assert result.status is TierStatus.RED
    assert result.reason == "inaugural regression case(s) failed"
    # Only the inaugural check ran.
    assert [c.name for c in result.checks] == ["inaugural cases"]
    assert result.checks[0].status is TierStatus.RED
    assert result.checks[0].detail.startswith("FAILED inaugural")
    # The unit suite was NOT attempted.
    assert not any("tests/unit" in " ".join(c) for c in r.calls[1:])


def test_unit_red_fails_at_unit_suite() -> None:
    r = FakeRunner({"inaugural": 0, "unit": 1, "ruff": 0, "mypy": 0})
    result = run_tier1(Tier1Config(), repo_root=REPO_ROOT, runner=r)
    assert result.status is TierStatus.RED
    assert result.reason == "unit suite failed"
    assert [c.name for c in result.checks] == ["inaugural cases", "unit suite"]


def test_ruff_red_fails_at_ruff() -> None:
    r = FakeRunner({"inaugural": 0, "unit": 0, "ruff": 1, "mypy": 0})
    result = run_tier1(Tier1Config(), repo_root=REPO_ROOT, runner=r)
    assert result.status is TierStatus.RED
    assert result.reason == "ruff failed"
    assert [c.name for c in result.checks] == ["inaugural cases", "unit suite", "ruff"]


def test_mypy_red_fails_at_mypy() -> None:
    r = FakeRunner({"inaugural": 0, "unit": 0, "ruff": 0, "mypy": 1})
    result = run_tier1(Tier1Config(), repo_root=REPO_ROOT, runner=r)
    assert result.status is TierStatus.RED
    assert result.reason == "mypy --strict failed"
    assert [c.name for c in result.checks] == ["inaugural cases", "unit suite", "ruff", "mypy --strict"]


def test_full_suite_can_be_disabled() -> None:
    r = FakeRunner({"inaugural": 0, "ruff": 0, "mypy": 0})
    result = run_tier1(Tier1Config(full_suite=False), repo_root=REPO_ROOT, runner=r)
    assert result.status is TierStatus.GREEN
    assert [c.name for c in result.checks] == ["inaugural cases", "ruff", "mypy --strict"]


def test_inaugural_cases_pass_through_to_pytest() -> None:
    r = FakeRunner({"inaugural": 0, "unit": 0, "ruff": 0, "mypy": 0})
    run_tier1(Tier1Config(), repo_root=REPO_ROOT, runner=r)
    inaugural_argv = r.calls[0]
    assert "-m" in inaugural_argv and inaugural_argv[2] == "pytest"
    assert INAUGURAL_CASES[0] in inaugural_argv


def test_resolve_gate_python_prefers_repo_venv_when_present(tmp_path: Path) -> None:
    # The pipx entry point runs under a venv that lacks the dev group; the
    # repo .venv is the deterministic interpreter that has pytest/ruff/mypy.
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "python"
    venv_python.write_text("#!/usr/bin/env python3\n")
    venv_python.chmod(0o755)
    assert resolve_gate_python(str(tmp_path)) == str(venv_python)


def test_resolve_gate_python_falls_back_to_sys_executable_without_venv(tmp_path: Path) -> None:
    # No repo .venv present (e.g. CI without a checked-in venv): behavior is
    # unchanged — fall back to sys.executable so the gate still runs.
    assert resolve_gate_python(str(tmp_path)) == sys.executable


def test_resolve_gate_python_ignores_non_executable_venv_python(tmp_path: Path) -> None:
    # A .venv/bin/python that exists but is not executable should not be
    # selected (e.g. a stale/broken venv); fall back to sys.executable.
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    venv_python = venv_bin / "python"
    venv_python.write_text("")
    venv_python.chmod(0o644)  # not executable
    assert resolve_gate_python(str(tmp_path)) == sys.executable
