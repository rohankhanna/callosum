"""Shared gate result types. Import-cycle-free base module: the tier modules
(tier1/tier2/tier3) import these without importing the orchestration
in harness, which imports the tier modules back."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class Tier(StrEnum):
    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"


class TierStatus(StrEnum):
    """Per-tier outcome.

    green — the tier passed. red — the tier failed (merge-blocking for
    Tier 1). pending — the tier could not reach a verdict because a
    dependency is not ready (e.g. no rate matrix published yet); not blocking.
    skipped — the operator did not request this tier.
    """

    GREEN = "green"
    RED = "red"
    PENDING = "pending"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One named check inside a tier (e.g. "inaugural cases", "ruff")."""

    name: str
    status: TierStatus
    returncode: int
    detail: str = ""
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status is TierStatus.GREEN


@dataclass(frozen=True, slots=True)
class TierResult:
    tier: Tier
    status: TierStatus
    reason: str
    checks: tuple[CheckResult, ...] = ()
    duration_s: float = 0.0

    @property
    def green(self) -> bool:
        return self.status is TierStatus.GREEN


@dataclass(frozen=True, slots=True)
class GateReport:
    merge_blocked: bool
    results: tuple[TierResult, ...]
    summary: str

    @property
    def green(self) -> bool:
        """Green-for-merge when Tier 1 is green. Tier 2/3 report their own
        status but a pending (no-matrix) Tier 2 does not block the merge."""
        tier1 = next((r for r in self.results if r.tier is Tier.TIER1), None)
        return tier1 is not None and tier1.green


class CommandRunner(Protocol):
    """Subprocess runner injected into Tier 1 for testability. Returns the
    process exit code; the default SubprocessRunner stashes captured
    stdout/stderr on last_output."""

    def __call__(self, argv: list[str], *, cwd: str, env: dict[str, str]) -> int: ...
