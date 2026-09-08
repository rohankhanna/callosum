"""Tiered merge/promotion gate harness.

The gate that must go GREEN before auto-dev / routing feature branches merge
to main (work tracker ). Three tiers, ordered by cost and
decisiveness:

  * **Tier 1** — deterministic CPU code tests. pytest on the inaugural
    regression cases (then the full unit/integration suite) + ruff check
    + mypy --strict. Fast and blocking: a red Tier 1 blocks the merge.
  * **Tier 2** — GPU-local repeated-sampling behaviour RATE matrix (unseeded,
    statistical). Produced by the sibling benchmark suite repo and consumed
    READ-ONLY here; callosum never builds it. Resumable / interruptible so a
    bounded daily GPU window can be paused and resumed across sessions. Not
    merge-blocking on its own — it feeds the auto-promotion decision; an
    absent matrix fail-closes to pending (the safe hold).
  * **Tier 3** — shadow / canary of a candidate per-cell behaviour on the live
    router behind a per-cell flag, with auto-revert when the shadow's rolling
    health falls below the control arm by a margin.

The harness is pure orchestration: it runs each tier, collects a per-tier
result, and reports whether the merge is blocked. The auto-promotion master
switch (CALLOSUM_AUTO_PROMOTION_ENABLED) stays OFF — this gate gates
MERGE, not autonomy activation.
"""

from __future__ import annotations

from callosum.gate.harness import GateConfig, run_gate
from callosum.gate.tier1 import INAUGURAL_CASES, Tier1Config, run_tier1
from callosum.gate.tier2 import FileRateMatrixReader, ResumableTier2Runner, Tier2Config
from callosum.gate.tier3 import ShadowCanaryGuard, Tier3Config, Tier3State
from callosum.gate.types import CheckResult, GateReport, Tier, TierResult, TierStatus

__all__ = [
    "INAUGURAL_CASES",
    "CheckResult",
    "FileRateMatrixReader",
    "GateConfig",
    "GateReport",
    "ResumableTier2Runner",
    "ShadowCanaryGuard",
    "Tier",
    "Tier1Config",
    "Tier2Config",
    "Tier3Config",
    "Tier3State",
    "TierResult",
    "TierStatus",
    "run_gate",
    "run_tier1",
]
