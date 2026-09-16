"""End-to-end Tier-1 gate test: real subprocesses, real pytest on the two
inaugural bug regression cases.

This is the proof that the gate goes green on main with the two bug tests wired
as the inaugural Tier-1 members. It
uses the real SubprocessRunner (no stubs) but scopes the full unit suite
OFF and mypy to the new gate package so the test stays fast while still
exercising the real pytest/ruff/mypy subprocess path. The first-tier green is
the merge-blocking signal.
"""

from __future__ import annotations

from pathlib import Path

from callosum.gate.harness import GateConfig, run_gate
from callosum.gate.tier1 import Tier1Config
from callosum.gate.types import Tier, TierStatus

REPO_ROOT = str(Path(__file__).resolve().parents[3])


def test_tier1_goes_green_on_main_with_real_subprocesses() -> None:
    config = GateConfig(
        tier1=Tier1Config(
            full_suite=False,  # keep the test fast; the inaugural cases + ruff + mypy are the real proof
            src_dir="src/callosum/gate",
        ),
        repo_root=REPO_ROOT,
    )
    report = run_gate(config, tiers=(Tier.TIER1,))
    assert report.green, f"tier1 should be green on main: {report.summary}"
    assert not report.merge_blocked
    tier1 = report.results[0]
    assert tier1.status is TierStatus.GREEN
    names = [c.name for c in tier1.checks]
    assert names == ["inaugural cases", "ruff", "mypy --strict"]
    # Every check passed with rc 0.
    assert all(c.returncode == 0 for c in tier1.checks)
