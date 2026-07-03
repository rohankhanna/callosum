"""Unit tests for the Tier-3 shadow/canary guard with auto-revert.

Pure logic over rolling counts; Tier3State persistence is exercised against
tmp_path. Auto-revert is statistical: a thin-but-bad shadow stays
pending (keep canarying), a zero-passes shadow with enough samples hard-
reverts, and a shadow whose lower bound falls below the control by a margin
soft-reverts.
"""

from __future__ import annotations

from pathlib import Path

from callosum.gate.tier3 import ShadowCanaryGuard, Tier3Config, Tier3State
from callosum.gate.types import Tier, TierStatus  # noqa: F401  (Tier re-export sanity)


def _guard(
    *,
    s_pass: int,
    s_n: int,
    c_pass: int = 200,
    c_n: int = 200,
    min_samples: int = 30,
    revert_margin: float = 0.05,
    hard_revert_min_samples: int = 10,
) -> ShadowCanaryGuard:
    return ShadowCanaryGuard(
        config=Tier3Config(
            cell_flag="cell::flag",
            control_passes=c_pass,
            control_samples=c_n,
            shadow_passes=s_pass,
            shadow_samples=s_n,
            min_samples=min_samples,
            revert_margin=revert_margin,
            hard_revert_min_samples=hard_revert_min_samples,
        )
    )


def test_cold_start_is_pending_keep_canary() -> None:
    d = _guard(s_pass=0, s_n=0).evaluate()
    assert d.revert is False
    assert d.shadow_samples == 0
    assert "pending" in d.reason


def test_thin_bad_shadow_is_pending_not_revert() -> None:
    # 5 samples, 0 passes — below hard_revert_min_samples (10) AND min_samples (30).
    d = _guard(s_pass=0, s_n=5).evaluate()
    assert d.revert is False
    assert "pending" in d.reason


def test_zero_passes_with_enough_samples_hard_reverts() -> None:
    d = _guard(s_pass=0, s_n=10).evaluate()
    assert d.revert is True
    assert "hard revert" in d.reason


def test_shadow_below_control_by_margin_soft_reverts() -> None:
    # Control 200/200 (lb ~0.98). Shadow 150/200 -> lb ~0.66. Margin 0.05.
    d = _guard(s_pass=150, s_n=200).evaluate()
    assert d.revert is True
    assert "auto-revert" in d.reason


def test_shadow_within_margin_keeps_canary() -> None:
    # Control 200/200 (lb ~0.98). Shadow 195/200 (lb ~0.95). Margin 0.05 -> keep.
    d = _guard(s_pass=195, s_n=200).evaluate()
    assert d.revert is False
    assert "keep canary" in d.reason


def test_from_state_hydrates_counts() -> None:
    path = Path("/tmp/ignored-state.json")
    state = Tier3State(
        control_passes=200,
        control_samples=200,
        shadow_passes=0,
        shadow_samples=10,
        last_updated_epoch_s=42.0,
        state_path=path,
    )
    guard = ShadowCanaryGuard.from_state(state)
    d = guard.evaluate()
    assert d.revert is True  # 0 passes / 10 samples -> hard revert
    assert d.shadow_samples == 10


def test_state_persistence_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = Tier3State(
        control_passes=100,
        control_samples=110,
        shadow_passes=80,
        shadow_samples=100,
        last_updated_epoch_s=1.5,
        state_path=path,
    )
    state.write()
    loaded = Tier3State.load(path)
    assert loaded is not None
    assert loaded.control_passes == 100
    assert loaded.shadow_samples == 100
    assert loaded.last_updated_epoch_s == 1.5


def test_state_load_missing_file_is_none(tmp_path: Path) -> None:
    assert Tier3State.load(tmp_path / "absent.json") is None


def test_state_load_malformed_is_none(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{nope")
    assert Tier3State.load(path) is None
