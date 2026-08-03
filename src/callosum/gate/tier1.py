"""Tier 1 — deterministic CPU code gate (fast, blocking).

Runs, in order:

  1. the **inaugural regression cases** — the two bug regression tests that
     seed the gate (the coverage doom-loop fix and the cell-level failover
     fix). A red inaugural check fails the tier immediately, before spending
     time on the full suite.
  2. the full deterministic unit suite (tests/unit by default — fast,
     ~24s, no live backends).
  3. ruff check.
  4. mypy --strict on the source tree.

All four must be green for Tier 1 to be green. Tier 1 is the only merge-
blocking tier. Subprocess invocation goes through an injectable CommandRunner
so the harness is unit-testable without spawning real processes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

from callosum.gate.types import CheckResult, CommandRunner, Tier, TierResult, TierStatus

# The two bug regression tests that seed the gate. Both already ship green on
# main; the gate wires them as the inaugural Tier-1 members (work tracker
# ). Paths are repo-relative.
INAUGURAL_CASES: tuple[str, ...] = (
    "tests/unit/routing/test_min_coverage_doom_loop.py",  # Bug1: feasibility filter + post-timeout cooldown
    "tests/unit/test_cell_level_retry.py",  # Bug2: request-scoped dispatch retry budget / cell failover
)


@dataclass(frozen=True, slots=True)
class SubprocessOutput:
    returncode: int
    stdout: str
    stderr: str


class SubprocessRunner:
    """Default CommandRunner: runs subprocess.run capturing stdout/stderr.
    The last invocation's output is stashed on last_output so a caller
    that wants the failure detail can read it after the fact."""

    def __init__(self) -> None:
        self.last_output: SubprocessOutput | None = None

    def __call__(self, argv: list[str], *, cwd: str, env: dict[str, str]) -> int:
        proc = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)
        self.last_output = SubprocessOutput(proc.returncode, proc.stdout, proc.stderr)
        return proc.returncode


def _detail(runner: CommandRunner, max_tail: int = 800) -> str:
    """Pull the tail of the last command's output if the runner exposes it."""
    last = getattr(runner, "last_output", None)
    if last is None:
        return ""
    blob = (last.stdout or "") + (last.stderr or "")
    return blob.strip()[-max_tail:]


def resolve_gate_python(repo_root: str | None = None) -> str:
    """Pick the interpreter the gate's Tier-1 subprocesses should use.

    The gate invokes pytest/ruff/mypy as python -m ...
    subprocesses, so the interpreter must carry callosum's
    [dependency-groups] dev deps. The pipx-installed callosum
    entry point runs under the pipx venv python, which has callosum
    (editable) but NOT the dev group — so the gate reports Tier-1 red
    (subprocesses fail instantly, rc=1) when launched via pipx. The repo
    .venv (created by uv sync --group dev) is the deterministic
    interpreter that has the dev group.

    Prefer <repo_root>/.venv/bin/python when it exists and is
    executable; otherwise fall back to sys.executable so behavior is
    unchanged where no repo venv is present (e.g. CI without a checked-in
    venv, or an explicit non-venv run). The default Tier1Config.python
    factory remains sys.executable so unit tests that construct a
    Tier1Config and inject a fake runner are unaffected.
    """
    candidate = os.path.join(repo_root or "", ".venv", "bin", "python")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return sys.executable


@dataclass(frozen=True, slots=True)
class Tier1Config:
    inaugural_cases: tuple[str, ...] = INAUGURAL_CASES
    full_suite: bool = True
    full_suite_path: str = "tests/unit"
    run_ruff: bool = True
    run_mypy_strict: bool = True
    src_dir: str = "src/callosum"
    python: str = field(default_factory=lambda: sys.executable)
    extra_env: dict[str, str] = field(default_factory=dict)


def _env(config: Tier1Config) -> dict[str, str]:
    env = dict(os.environ)
    # The repo runs tests with PYTHONPATH=src (see Makefile `test`).
    env["PYTHONPATH"] = "src"
    env.update(config.extra_env)
    return env


def _check(
    name: str,
    runner: CommandRunner,
    argv: list[str],
    *,
    cwd: str,
    env: dict[str, str],
) -> tuple[TierStatus, int, float, str]:
    start = time.perf_counter()
    rc = runner(argv, cwd=cwd, env=env)
    elapsed = time.perf_counter() - start
    status = TierStatus.GREEN if rc == 0 else TierStatus.RED
    return status, rc, elapsed, ("" if rc == 0 else _detail(runner))


def run_tier1(config: Tier1Config, *, repo_root: str, runner: CommandRunner | None = None) -> TierResult:
    """Run the deterministic CPU gate. Returns red on the first failing
    check after recording it (so the report shows which check broke)."""
    r: CommandRunner = runner if runner is not None else SubprocessRunner()
    env = _env(config)
    checks: list[CheckResult] = []

    # 1. Inaugural regression cases — the gate seed. Fast-fail if red.
    status, rc, elapsed, detail = _check(
        "inaugural cases",
        r,
        [config.python, "-m", "pytest", *config.inaugural_cases, "-q"],
        cwd=repo_root,
        env=env,
    )
    checks.append(CheckResult("inaugural cases", status, rc, detail, elapsed))
    if status is TierStatus.RED:
        return TierResult(Tier.TIER1, TierStatus.RED, "inaugural regression case(s) failed", tuple(checks))

    # 2. Full deterministic unit suite.
    if config.full_suite:
        status, rc, elapsed, detail = _check(
            "unit suite",
            r,
            [config.python, "-m", "pytest", config.full_suite_path, "-q"],
            cwd=repo_root,
            env=env,
        )
        checks.append(CheckResult("unit suite", status, rc, detail, elapsed))
        if status is TierStatus.RED:
            return TierResult(Tier.TIER1, TierStatus.RED, "unit suite failed", tuple(checks))

    # 3. ruff check .
    if config.run_ruff:
        status, rc, elapsed, detail = _check(
            "ruff",
            r,
            [config.python, "-m", "ruff", "check", "."],
            cwd=repo_root,
            env=env,
        )
        checks.append(CheckResult("ruff", status, rc, detail, elapsed))
        if status is TierStatus.RED:
            return TierResult(Tier.TIER1, TierStatus.RED, "ruff failed", tuple(checks))

    # 4. mypy --strict on the source tree.
    if config.run_mypy_strict:
        status, rc, elapsed, detail = _check(
            "mypy --strict",
            r,
            [config.python, "-m", "mypy", "--strict", config.src_dir],
            cwd=repo_root,
            env=env,
        )
        checks.append(CheckResult("mypy --strict", status, rc, detail, elapsed))
        if status is TierStatus.RED:
            return TierResult(Tier.TIER1, TierStatus.RED, "mypy --strict failed", tuple(checks))

    return TierResult(Tier.TIER1, TierStatus.GREEN, "tier1 green: inaugural + unit + ruff + mypy", tuple(checks))
