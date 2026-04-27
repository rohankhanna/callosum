from __future__ import annotations

import sqlite3
from pathlib import Path

from codex_proxy.config import AutoRouterConfig
from codex_proxy.synthetic import (
    DailyCounts,
    daily_counts,
    should_fire,
    synthetic_body,
    synthetic_target_for_today,
)


def _make_log(db: Path, rows: list[tuple[float, str]]) -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_start REAL NOT NULL,
            routing_mode TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO requests (ts_start, routing_mode) VALUES (?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_synthetic_body_uses_virtual_model_name() -> None:
    body = synthetic_body()
    assert body["model"] == "auto-learning-synthetic"
    assert body["stream"] is False
    # Responses-API shape: input is a list of message blocks.
    assert isinstance(body["input"], list) and len(body["input"]) == 1
    assert body["input"][0]["role"] == "user"


def test_target_floor_dominates_when_organic_is_low() -> None:
    cfg = AutoRouterConfig(
        synthetic_floor_per_day=20,
        synthetic_pct_of_organic=0.05,  # 5% of 100 = 5
        synthetic_hard_ceiling_per_day=200,
    )
    counts = DailyCounts(organic=100, synthetic=0)
    # max(20, 5) = 20
    assert synthetic_target_for_today(counts, cfg) == 20


def test_target_pct_dominates_when_organic_is_high() -> None:
    cfg = AutoRouterConfig(
        synthetic_floor_per_day=20,
        synthetic_pct_of_organic=0.05,  # 5% of 1000 = 50
        synthetic_hard_ceiling_per_day=200,
    )
    counts = DailyCounts(organic=1000, synthetic=0)
    assert synthetic_target_for_today(counts, cfg) == 50


def test_target_hard_ceiling_caps_pct_growth() -> None:
    cfg = AutoRouterConfig(
        synthetic_floor_per_day=20,
        synthetic_pct_of_organic=0.5,  # 50% of 10000 = 5000
        synthetic_hard_ceiling_per_day=200,  # ceiling wins
    )
    counts = DailyCounts(organic=10000, synthetic=0)
    assert synthetic_target_for_today(counts, cfg) == 200


def test_should_fire_false_when_synthetic_at_target() -> None:
    cfg = AutoRouterConfig(
        synthetic_floor_per_day=10,
        synthetic_pct_of_organic=0.0,
        synthetic_hard_ceiling_per_day=100,
    )
    assert should_fire(DailyCounts(organic=0, synthetic=10), cfg) is False
    assert should_fire(DailyCounts(organic=0, synthetic=9), cfg) is True


def test_should_fire_false_when_disabled_by_defaults() -> None:
    # All-zero defaults = topper disabled, regardless of organic volume.
    cfg = AutoRouterConfig()
    assert should_fire(DailyCounts(organic=1000, synthetic=0), cfg) is False


def test_daily_counts_only_today(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    # Pick now_ts at a known UTC day boundary so we can craft "yesterday" rows.
    now_ts = 1_700_000_000.0  # 2023-11-14 ~22:13 UTC
    yesterday_ts = now_ts - 86400.0
    _make_log(
        db,
        [
            (yesterday_ts, "auto-learning"),  # yesterday — excluded
            (yesterday_ts, "auto-learning-synthetic"),  # yesterday — excluded
            (now_ts - 60.0, "auto-learning"),
            (now_ts - 30.0, "auto-learning"),
            (now_ts - 10.0, "auto-learning-synthetic"),
            (now_ts - 5.0, "pass-through"),  # different mode — excluded
        ],
    )
    counts = daily_counts(db, now_ts=now_ts)
    assert counts.organic == 2
    assert counts.synthetic == 1


def test_daily_counts_missing_log_returns_zero(tmp_path: Path) -> None:
    counts = daily_counts(tmp_path / "no.sqlite", now_ts=0.0)
    assert counts == DailyCounts(organic=0, synthetic=0)
