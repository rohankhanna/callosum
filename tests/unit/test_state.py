from __future__ import annotations

from pathlib import Path

from callosum.backend import UsageSnapshot
from callosum.codex_quota import CodexQuotaSnapshot
from callosum.state import StateStore


def _quota(**overrides: object) -> CodexQuotaSnapshot:
    base = dict(
        plan_type="pro",
        active_limit="weekly",
        five_hourly_used_percent=1,
        weekly_used_percent=98,
        five_hourly_window_minutes=300,
        weekly_window_minutes=10080,
        five_hourly_reset_at=1234567890,
        weekly_reset_at=1234999999,
        five_hourly_reset_after_seconds=120,
        weekly_reset_after_seconds=86400,
        five_hourly_over_weekly_limit_percent=0,
        credits_balance="0",
        credits_has_credits=False,
        credits_unlimited=False,
        observed_at=1234567890.0,
    )
    base.update(overrides)
    return CodexQuotaSnapshot(**base)  # type: ignore[arg-type]


def test_save_and_load_usage_roundtrip(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    snapshot = UsageSnapshot(
        remaining_fraction=0.42,
        cooldown_until_ts=1234567890.0,
        weekly_exhausted=True,
        probed_at_ts=1234567890.0,
    )
    store.save_usage("alpha", snapshot)
    loaded = store.load_usage("alpha")
    assert loaded == snapshot


def test_load_returns_none_when_file_missing(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    assert store.load_usage("ghost") is None


def test_load_returns_none_when_file_corrupted(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    usage_dir = tmp_path / "usage"
    usage_dir.mkdir(parents=True)
    (usage_dir / "alpha.json").write_text("not json")
    assert store.load_usage("alpha") is None


def test_save_and_load_quota_roundtrip(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    snapshot = _quota()
    store.save_quota("alpha", snapshot)
    loaded = store.load_quota("alpha")
    assert loaded == snapshot


def test_load_quota_returns_none_when_file_missing(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    assert store.load_quota("ghost") is None


def test_load_quota_returns_none_when_file_corrupted(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    quota_dir = tmp_path / "quota"
    quota_dir.mkdir(parents=True)
    (quota_dir / "alpha.json").write_text("not json")
    assert store.load_quota("alpha") is None


def test_save_is_atomic_on_overwrite(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    first = UsageSnapshot(remaining_fraction=0.9, cooldown_until_ts=None, weekly_exhausted=False, probed_at_ts=1.0)
    second = UsageSnapshot(remaining_fraction=0.1, cooldown_until_ts=10.0, weekly_exhausted=True, probed_at_ts=2.0)
    store.save_usage("alpha", first)
    store.save_usage("alpha", second)
    assert store.load_usage("alpha") == second
