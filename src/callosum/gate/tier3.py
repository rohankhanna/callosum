"""Tier 3 — live shadow / canary of a candidate per-cell behaviour, with
auto-revert.

A candidate behaviour (e.g. a per-cell transform flag) ships shadowed behind a
per-cell flag on the live router: a fraction of eligible traffic for that cell
runs the candidate (the **shadow** arm) while the rest runs the incumbent
behaviour (the **control** arm). The guard watches a rolling pass-rate for
each arm and decides whether the flag should **auto-revert** to the incumbent.

Auto-revert is statistical, not boolean: the shadow is judged worse only when
enough unseeded samples put the lower end of its uncertainty band below the
control arm's lower bound by a configured margin. A thin-but-bad shadow is
pending (keep collecting), not an automatic revert — but a zero-passes
shadow with enough samples is a hard revert (the candidate is catastrophically
broken).

The guard never flips the live flag itself — it returns a decision; the
caller (an operator, or the auto-promotion pipeline with the master switch ON)
applies it. Rolling counts are persisted to disk so the guard survives daemon
restarts (state preserved across shutdowns/restarts, per the repo's
state-resume requirement).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from callosum.gate.tier2 import wilson_lower_bound


@dataclass(frozen=True, slots=True)
class Tier3Config:
    """Configuration for one shadow/canary guard. Counts default to zero
    (cold start); a persisted Tier3State overrides them via
    ShadowCanaryGuard.from_state."""

    cell_flag: str  # the per-cell flag under canary, e.g. "model-a0d4::inband_reasoning"
    control_passes: int = 0
    control_samples: int = 0
    shadow_passes: int = 0
    shadow_samples: int = 0
    min_samples: int = 30  # below this, keep canarying (pending), don't revert
    revert_margin: float = 0.05  # shadow_lb < control_lb - margin -> revert
    hard_revert_min_samples: int = 10  # zero-passes shadow reverts only after this many samples
    state_path: Path | None = None  # where to persist rolling counts


@dataclass(frozen=True, slots=True)
class Tier3State:
    """Persisted rolling counts for one guard. Survives restarts."""

    control_passes: int
    control_samples: int
    shadow_passes: int
    shadow_samples: int
    last_updated_epoch_s: float
    state_path: Path

    def write(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "control_passes": self.control_passes,
            "control_samples": self.control_samples,
            "shadow_passes": self.shadow_passes,
            "shadow_samples": self.shadow_samples,
            "last_updated_epoch_s": self.last_updated_epoch_s,
        }
        with self.state_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")

    @classmethod
    def load(cls, path: Path) -> Tier3State | None:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        return cls(
            control_passes=int(data.get("control_passes", 0)),
            control_samples=int(data.get("control_samples", 0)),
            shadow_passes=int(data.get("shadow_passes", 0)),
            shadow_samples=int(data.get("shadow_samples", 0)),
            last_updated_epoch_s=float(data.get("last_updated_epoch_s", 0.0)),
            state_path=path,
        )


@dataclass(frozen=True, slots=True)
class CanaryDecision:
    revert: bool
    reason: str
    shadow_samples: int
    control_lb: float
    shadow_lb: float


class ShadowCanaryGuard:
    """Evaluates one shadow/canary arm against its control and returns an
    auto-revert decision. Pure logic over the rolling counts; persistence is
    delegated to Tier3State."""

    def __init__(
        self,
        *,
        config: Tier3Config,
        control_passes: int | None = None,
        control_samples: int | None = None,
        shadow_passes: int | None = None,
        shadow_samples: int | None = None,
    ) -> None:
        self._config = config
        self._control_passes = config.control_passes if control_passes is None else control_passes
        self._control_samples = config.control_samples if control_samples is None else control_samples
        self._shadow_passes = config.shadow_passes if shadow_passes is None else shadow_passes
        self._shadow_samples = config.shadow_samples if shadow_samples is None else shadow_samples

    @classmethod
    def from_state(cls, state: Tier3State) -> ShadowCanaryGuard:
        cfg = Tier3Config(
            cell_flag="from-state",
            state_path=state.state_path,
        )
        return cls(
            config=cfg,
            control_passes=state.control_passes,
            control_samples=state.control_samples,
            shadow_passes=state.shadow_passes,
            shadow_samples=state.shadow_samples,
        )

    def evaluate(self) -> CanaryDecision:
        cfg = self._config
        s_pass, s_n = self._shadow_passes, self._shadow_samples
        c_pass, c_n = self._control_passes, self._control_samples
        shadow_lb = wilson_lower_bound(s_pass, s_n)
        control_lb = wilson_lower_bound(c_pass, c_n)

        # Hard revert: a zero-passes shadow with enough samples is
        # catastrophically broken — revert immediately, do not keep canarying.
        if s_n >= cfg.hard_revert_min_samples and s_pass == 0:
            return CanaryDecision(
                revert=True,
                reason=f"hard revert: shadow arm has 0 passes over {s_n} samples",
                shadow_samples=s_n,
                control_lb=control_lb,
                shadow_lb=shadow_lb,
            )

        # Not enough shadow data to judge — keep canarying (pending).
        if s_n < cfg.min_samples:
            return CanaryDecision(
                revert=False,
                reason=f"pending: shadow has {s_n}/{cfg.min_samples} samples",
                shadow_samples=s_n,
                control_lb=control_lb,
                shadow_lb=shadow_lb,
            )

        # Statistical revert: shadow lower bound sits below the control lower
        # bound by more than the configured margin.
        if shadow_lb < control_lb - cfg.revert_margin:
            return CanaryDecision(
                revert=True,
                reason=(
                    f"auto-revert: shadow lb {shadow_lb:.3f} < control lb {control_lb:.3f} "
                    f"- margin {cfg.revert_margin:.3f}"
                ),
                shadow_samples=s_n,
                control_lb=control_lb,
                shadow_lb=shadow_lb,
            )

        return CanaryDecision(
            revert=False,
            reason=f"keep canary: shadow lb {shadow_lb:.3f} within margin of control lb {control_lb:.3f}",
            shadow_samples=s_n,
            control_lb=control_lb,
            shadow_lb=shadow_lb,
        )
