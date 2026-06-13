from __future__ import annotations

from pathlib import Path

from callosum.backend import UsageSnapshot
from callosum.state import StateStore


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


def test_save_is_atomic_on_overwrite(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    first = UsageSnapshot(remaining_fraction=0.9, cooldown_until_ts=None, weekly_exhausted=False, probed_at_ts=1.0)
    second = UsageSnapshot(remaining_fraction=0.1, cooldown_until_ts=10.0, weekly_exhausted=True, probed_at_ts=2.0)
    store.save_usage("alpha", first)
    store.save_usage("alpha", second)
    assert store.load_usage("alpha") == second
