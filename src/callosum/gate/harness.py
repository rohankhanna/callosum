"""Top-level tiered-gate runner: deterministic orchestration over the tiers.

Tier 1 is the only merge-blocking tier; a red Tier 1 marks the whole gate
merge_blocked. Tier 2 / Tier 3 ship as resumable scaffolding and report
their own status, but a non-green Tier 2 (no rate matrix yet) does NOT block a
Tier-1-green merge — the gate gates merge on deterministic code tests, not on a
statistical matrix that may not exist yet. The shared result types live in
callosum.gate.types to keep this module import-cycle-free from the tiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from callosum.gate.tier1 import Tier1Config, run_tier1
from callosum.gate.tier2 import Tier2Config
from callosum.gate.tier3 import ShadowCanaryGuard, Tier3Config, Tier3State
from callosum.gate.types import CommandRunner, GateReport, Tier, TierResult, TierStatus


@dataclass(frozen=True, slots=True)
class GateConfig:
    tier1: Tier1Config = field(default_factory=lambda: Tier1Config())
    tier2: Tier2Config = field(default_factory=lambda: Tier2Config())
    tier3: Tier3Config = field(default_factory=lambda: Tier3Config(cell_flag=""))
    repo_root: str = "."


def _summarize(results: tuple[TierResult, ...]) -> str:
    parts = [f"{r.tier.value}={r.status.value}" for r in results]
    tier1_red = any(r.tier is Tier.TIER1 and r.status is TierStatus.RED for r in results)
    head = "MERGE BLOCKED" if tier1_red else "tier1 green"
    return f"{head}: " + ", ".join(parts)


def _run_tier3(config: Tier3Config, *, state: Tier3State | None) -> TierResult:
    """Tier 3 evaluates the live shadow/canary guard and reports whether the
    candidate per-cell flag should auto-revert. It never flips the live flag
    itself — it returns a decision; the caller (operator or the autonomy
    pipeline with the master switch ON) applies it."""
    guard = ShadowCanaryGuard.from_state(state) if state is not None else ShadowCanaryGuard(config=config)
    decision = guard.evaluate()
    if decision.revert:
        return TierResult(Tier.TIER3, TierStatus.RED, decision.reason)
    if decision.shadow_samples < config.min_samples:
        return TierResult(Tier.TIER3, TierStatus.PENDING, decision.reason)
    return TierResult(Tier.TIER3, TierStatus.GREEN, decision.reason)


def run_gate(
    config: GateConfig,
    *,
    tiers: tuple[Tier, ...] = (Tier.TIER1, Tier.TIER2, Tier.TIER3),
    runner: CommandRunner | None = None,
    tier3_state: Tier3State | None = None,
) -> GateReport:
    """Run the requested tiers and return a merge decision.

    runner is injected into Tier 1 (the subprocess-based tier). tier3_state
    is injected into Tier 3 so a long-lived guard can carry its persisted
    rolling-health state across runs.
    """
    out: list[TierResult] = []
    if Tier.TIER1 in tiers:
        out.append(run_tier1(config.tier1, repo_root=config.repo_root, runner=runner))
    if Tier.TIER2 in tiers:
        from callosum.gate.tier2 import ResumableTier2Runner  # local import: optional tier

        out.append(ResumableTier2Runner(config.tier2).run())
    if Tier.TIER3 in tiers:
        out.append(_run_tier3(config.tier3, state=tier3_state))
    results = tuple(out)
    tier1_red = any(r.tier is Tier.TIER1 and r.status is TierStatus.RED for r in results)
    return GateReport(merge_blocked=tier1_red, results=results, summary=_summarize(results))
