from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from codex_proxy.codex_quota import CodexQuotaSnapshot
from codex_proxy.config import AutoRouterConfig
from codex_proxy.synthetic import (
    DailyCounts,
    _aggressive_synthetic_body,
    _should_enter_aggressive,
    cold_start_fire_count,
    cold_start_target,
    daily_counts,
    organic_burn_rate_pct_per_hour,
    synthetic_body,
    weekly_exhaustion_fire_count,
    weekly_exhaustion_fire_rate,
)


def _make_log(db: Path, rows: list[tuple[float, str, str | None]]) -> None:
    """Create a minimal requests table for daily-counts/burn-rate queries."""
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_start REAL NOT NULL,
            backend_id TEXT,
            routing_mode TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO requests (ts_start, backend_id, routing_mode) VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def _make_burn_log(db: Path, rows: list[dict]) -> None:
    """Schema with the columns burn-rate query reads."""
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_start REAL NOT NULL,
            backend_id TEXT NOT NULL,
            routing_mode TEXT,
            status INTEGER NOT NULL,
            quota_reset_crossover INTEGER NOT NULL DEFAULT 0,
            weekly_used_percent_before INTEGER,
            weekly_used_percent_after INTEGER
        )
        """
    )
    for r in rows:
        conn.execute(
            "INSERT INTO requests (ts_start, backend_id, routing_mode, status,"
            " quota_reset_crossover, weekly_used_percent_before,"
            " weekly_used_percent_after) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                r["ts_start"],
                r["backend_id"],
                r["routing_mode"],
                r.get("status", 200),
                r.get("crossover", 0),
                r.get("before"),
                r.get("after"),
            ),
        )
    conn.commit()
    conn.close()


def _quota(
    *,
    weekly_used: int | None = 50,
    weekly_reset_at: int | None = None,
    five_hourly_used: int | None = 30,
) -> CodexQuotaSnapshot:
    return CodexQuotaSnapshot(
        plan_type="plus",
        active_limit="premium",
        five_hourly_used_percent=five_hourly_used,
        weekly_used_percent=weekly_used,
        five_hourly_window_minutes=300,
        weekly_window_minutes=10080,
        five_hourly_reset_at=None,
        weekly_reset_at=weekly_reset_at,
        five_hourly_reset_after_seconds=None,
        weekly_reset_after_seconds=None,
        five_hourly_over_weekly_limit_percent=None,
        credits_balance=None,
        credits_has_credits=False,
        credits_unlimited=False,
        observed_at=time.time(),
    )


def test_synthetic_body_uses_virtual_model_name() -> None:
    body = synthetic_body()
    assert body["model"] == "auto-learning-synthetic"
    assert body["stream"] is False
    assert isinstance(body["input"], list) and len(body["input"]) == 1
    # Codex's Responses API requires non-empty `instructions` AND `store: False`
    # — omitting either yields a 400.
    assert isinstance(body.get("instructions"), str)
    assert body["instructions"]  # non-empty
    assert body.get("store") is False


# --------- cold-start fallback ---------------------------------------------


def test_cold_start_target_floor_dominates_when_organic_low() -> None:
    cfg = AutoRouterConfig(
        synthetic_floor_per_day=20,
        synthetic_pct_of_organic=0.05,
        synthetic_hard_ceiling_per_day=200,
    )
    assert cold_start_target(DailyCounts(organic=100, synthetic=0), cfg) == 20


def test_cold_start_fire_count_zero_when_disabled_by_defaults() -> None:
    cfg = AutoRouterConfig()
    assert cold_start_fire_count(DailyCounts(organic=999, synthetic=0), cfg) == 0


def test_cold_start_fire_count_capped_by_max_per_tick() -> None:
    cfg = AutoRouterConfig(synthetic_floor_per_day=1000, max_synthetics_per_tick=5)
    # Deficit is huge but tick cap is 5.
    assert cold_start_fire_count(DailyCounts(organic=0, synthetic=0), cfg) == 5


def test_daily_counts_per_backend(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    now_ts = 1_700_000_000.0
    yesterday_ts = now_ts - 86400.0
    _make_log(
        db,
        [
            (yesterday_ts, "alpha", "auto-learning"),  # excluded — yesterday
            (now_ts - 60, "alpha", "auto-learning"),
            (now_ts - 30, "alpha", "auto-learning-synthetic"),
            (now_ts - 30, "beta", "auto-learning"),  # different backend
            (now_ts - 5, "alpha", "pass-through"),  # excluded — wrong mode
        ],
    )
    counts = daily_counts(db, now_ts=now_ts, backend_id="alpha")
    assert counts == DailyCounts(organic=1, synthetic=1)


# --------- weekly-exhaustion controller ------------------------------------


def test_weekly_exhaust_zero_when_no_quota() -> None:
    cfg = AutoRouterConfig()
    n = weekly_exhaustion_fire_count(
        _quota(weekly_used=None, weekly_reset_at=None),
        organic_rate_pct_per_hour=0.0,
        now_ts=1000.0,
        cfg=cfg,
    )
    assert n == 0


def test_weekly_exhaust_zero_when_at_target() -> None:
    cfg = AutoRouterConfig(weekly_target_pct=95.0)
    now = 1_700_000_000.0
    n = weekly_exhaustion_fire_count(
        _quota(weekly_used=95, weekly_reset_at=int(now + 3600 * 24)),
        organic_rate_pct_per_hour=0.0,
        now_ts=now,
        cfg=cfg,
    )
    assert n == 0


def test_weekly_exhaust_zero_when_5h_near_exhausted() -> None:
    cfg = AutoRouterConfig(five_hourly_pause_pct=95.0)
    now = 1_700_000_000.0
    n = weekly_exhaustion_fire_count(
        _quota(weekly_used=20, weekly_reset_at=int(now + 86400), five_hourly_used=99),
        organic_rate_pct_per_hour=0.0,
        now_ts=now,
        cfg=cfg,
    )
    assert n == 0


def test_weekly_exhaust_fires_when_room_remains() -> None:
    cfg = AutoRouterConfig(
        pct_per_synthetic_estimate=0.1,
        prediction_safety_margin=1.0,
        max_synthetics_per_tick=10,
        synthetic_check_interval_seconds=300,
        weekly_target_pct=95.0,
        five_hourly_pause_pct=95.0,
    )
    now = 1_700_000_000.0
    # 50% weekly used, 24h to reset, no organic. burnable = 95-50 = 45 pct.
    # synthetics_total = 45 / 0.1 = 450. tick_hours = 5/60 ≈ 0.0833.
    # synthetics_this_tick = 450 * 0.0833 / 24 ≈ 1.56 → round to 2. Capped at 10.
    n = weekly_exhaustion_fire_count(
        _quota(weekly_used=50, weekly_reset_at=int(now + 86400)),
        organic_rate_pct_per_hour=0.0,
        now_ts=now,
        cfg=cfg,
    )
    assert n == 2


def test_weekly_exhaust_zero_when_organic_will_consume_remainder() -> None:
    cfg = AutoRouterConfig(
        pct_per_synthetic_estimate=0.1,
        prediction_safety_margin=1.0,
        weekly_target_pct=95.0,
    )
    now = 1_700_000_000.0
    # 50% used, 24h to reset, organic burns 2%/hr → 48% projected → 95-50-48 = -3.
    n = weekly_exhaustion_fire_count(
        _quota(weekly_used=50, weekly_reset_at=int(now + 86400)),
        organic_rate_pct_per_hour=2.0,
        now_ts=now,
        cfg=cfg,
    )
    assert n == 0


def test_weekly_exhaust_safety_margin_makes_it_more_conservative() -> None:
    cfg = AutoRouterConfig(
        pct_per_synthetic_estimate=0.1,
        prediction_safety_margin=2.0,  # double the projected human burn
        weekly_target_pct=95.0,
    )
    now = 1_700_000_000.0
    # 50% used, 24h to reset, organic 1%/hr → projected 24 * margin 2.0 = 48.
    # burnable = 95 - 50 - 48 = -3 → 0.
    n = weekly_exhaustion_fire_count(
        _quota(weekly_used=50, weekly_reset_at=int(now + 86400)),
        organic_rate_pct_per_hour=1.0,
        now_ts=now,
        cfg=cfg,
    )
    assert n == 0


def test_organic_burn_rate_returns_zero_for_missing_log(tmp_path: Path) -> None:
    rate = organic_burn_rate_pct_per_hour(
        tmp_path / "no.sqlite",
        backend_id="alpha",
        now_ts=1000.0,
        window_hours=168.0,
    )
    assert rate == 0.0


def test_organic_burn_rate_sums_deltas_for_backend(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    now_ts = 1_700_000_000.0
    rows = [
        # Two organic alpha calls inside window: delta 1+2 = 3 pct over 100h
        # → 3/100 = 0.03 %/hr
        {
            "ts_start": now_ts - 3600,
            "backend_id": "alpha",
            "routing_mode": "auto-learning",
            "before": 50,
            "after": 51,
        },
        {
            "ts_start": now_ts - 1800,
            "backend_id": "alpha",
            "routing_mode": "pass-through",
            "before": 51,
            "after": 53,
        },
        # Outside window — excluded
        {
            "ts_start": now_ts - 3600 * 200,
            "backend_id": "alpha",
            "routing_mode": "auto-learning",
            "before": 30,
            "after": 31,
        },
        # Different backend — excluded
        {
            "ts_start": now_ts - 60,
            "backend_id": "beta",
            "routing_mode": "auto-learning",
            "before": 10,
            "after": 99,
        },
        # Reset crossover — excluded
        {
            "ts_start": now_ts - 60,
            "backend_id": "alpha",
            "routing_mode": "auto-learning",
            "crossover": 1,
            "before": 99,
            "after": 1,
        },
        # Synthetic — excluded (only organic + pass-through count toward
        # human burn prediction)
        {
            "ts_start": now_ts - 60,
            "backend_id": "alpha",
            "routing_mode": "auto-learning-synthetic",
            "before": 53,
            "after": 54,
        },
    ]
    _make_burn_log(db, rows)
    rate = organic_burn_rate_pct_per_hour(db, backend_id="alpha", now_ts=now_ts, window_hours=100.0)
    assert rate == 0.03


# --------- leaky-bucket fractional rate (the bug fix) -----------------------


def test_weekly_exhaust_returns_fractional_rate_under_one() -> None:
    """The bug: previous integer-rounded version returned 0 forever when the
    target rate was < 0.5 per tick. The float rate version returns the
    actual fractional rate so the worker can accumulate it.
    """
    cfg = AutoRouterConfig(
        pct_per_synthetic_estimate=0.1,
        prediction_safety_margin=1.0,
        max_synthetics_per_tick=10,
        synthetic_check_interval_seconds=300,
        weekly_target_pct=95.0,
    )
    now = 1_700_000_000.0
    # Mirrors the real-world bug: 25% used, ~140h remaining, low organic.
    rate = weekly_exhaustion_fire_rate(
        _quota(weekly_used=25, weekly_reset_at=int(now + 140 * 3600)),
        organic_rate_pct_per_hour=0.14,
        now_ts=now,
        cfg=cfg,
    )
    # Should be between 0 and 1 — exactly the case the int rounding broke.
    assert 0.0 < rate < 1.0
    # Old fire_count would have rounded this to 0; new code should preserve it.
    assert (
        weekly_exhaustion_fire_count(
            _quota(weekly_used=25, weekly_reset_at=int(now + 140 * 3600)),
            organic_rate_pct_per_hour=0.14,
            now_ts=now,
            cfg=cfg,
        )
        == 0
    )  # backward-compat int rounding still returns 0
    # But the rate itself is non-zero — that's what the worker accumulator
    # uses to actually fire over many ticks.
    assert rate > 0.1


async def test_synthetic_topper_accumulator_fires_over_many_ticks() -> None:
    """Functional check: with a fractional rate of ~0.3 per tick, after 10
    ticks the worker should have fired ~3 synthetics — not 0 (the old bug)
    and not 10 (over-firing).
    """
    import asyncio

    from codex_proxy.backend import HealthStatus, UsageSnapshot
    from codex_proxy.codex_quota import CodexQuotaSnapshot
    from codex_proxy.fakes import InMemoryFakeBackend
    from codex_proxy.synthetic import SyntheticTopper

    # Backend with a quota state that produces ~0.28 rate per tick (matches
    # the live state where the bug manifested).
    backend = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0e7"}),
        health=HealthStatus(available=True, reason="ok"),
        usage=UsageSnapshot(
            remaining_fraction=0.7,
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=0.0,
        ),
    )
    fake_clock = [1_700_000_000.0]
    backend._fake_quota = CodexQuotaSnapshot(  # type: ignore[attr-defined]
        plan_type="plus",
        active_limit="premium",
        five_hourly_used_percent=1,
        weekly_used_percent=25,
        five_hourly_window_minutes=300,
        weekly_window_minutes=10080,
        five_hourly_reset_at=None,
        weekly_reset_at=int(fake_clock[0] + 140 * 3600),
        five_hourly_reset_after_seconds=None,
        weekly_reset_after_seconds=None,
        five_hourly_over_weekly_limit_percent=None,
        credits_balance=None,
        credits_has_credits=False,
        credits_unlimited=False,
        observed_at=fake_clock[0],
    )

    fired_calls: list[str] = []

    async def fake_dispatch(body: dict, backend_id: str) -> None:
        fired_calls.append(backend_id)

    cfg = AutoRouterConfig(
        pct_per_synthetic_estimate=0.1,
        prediction_safety_margin=1.0,
        max_synthetics_per_tick=10,
        synthetic_check_interval_seconds=300,
        weekly_target_pct=95.0,
    )
    # Use a tmp_path-like path for the usage_log; doesn't need rows for this
    # test (organic_rate query handles missing file by returning 0.0).
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "u.sqlite"
        topper = SyntheticTopper(
            cfg=cfg,
            usage_log_path=log_path,
            backends=[backend],
            dispatch=fake_dispatch,
            clock=lambda: fake_clock[0],
        )
        # Run 10 ticks manually (don't start the asyncio loop).
        for _ in range(10):
            await topper._tick()
        # Should have fired some synthetics — strictly between 0 (old bug)
        # and 10 (over-firing). With rate ~0.28, expect ~2-3 over 10 ticks.
        assert 1 <= len(fired_calls) <= 5, (
            f"expected 1-5 synthetics fired across 10 ticks, got {len(fired_calls)}"
        )
        await topper.stop()
    _ = asyncio  # silence unused-import lint


# --------- aggressive exhaustion mode ----------------------------------------


def test_should_enter_aggressive_at_threshold() -> None:
    cfg = AutoRouterConfig(aggressive_exhaustion_pct=98.0)
    # At threshold
    assert _should_enter_aggressive(_quota(weekly_used=98), cfg)
    # Above threshold
    assert _should_enter_aggressive(_quota(weekly_used=99), cfg)
    # Below threshold
    assert not _should_enter_aggressive(_quota(weekly_used=97), cfg)
    # No snap
    assert not _should_enter_aggressive(None, cfg)
    # No weekly percentage
    assert not _should_enter_aggressive(_quota(weekly_used=None), cfg)


def test_aggressive_body_always_valid() -> None:
    for _ in range(60):
        body = _aggressive_synthetic_body()
        assert body["store"] is False
        assert isinstance(body.get("instructions"), str)
        assert body["instructions"]  # non-empty
        assert body["model"] == "auto-learning-synthetic"
        assert isinstance(body.get("input"), list)
        assert len(body["input"]) > 0


def test_aggressive_body_hits_all_tiers() -> None:
    small_count = 0
    medium_count = 0
    large_count = 0

    for _ in range(200):
        body = _aggressive_synthetic_body()
        # Detect tier by input length
        # Small: single input element, medium: 3 input elements, large: 1 but with very long text
        input_len = len(body["input"])
        max_tokens = body.get("max_tokens", 0)

        if input_len > 1:
            # Multiple messages = medium tier
            medium_count += 1
        elif max_tokens > 200:
            # Single message with high max_tokens = large tier
            large_count += 1
        else:
            # Default small tier
            small_count += 1

    # All tiers should appear statistically
    assert small_count > 10, f"expected > 10 small, got {small_count}"
    assert medium_count > 10, f"expected > 10 medium, got {medium_count}"
    assert large_count > 10, f"expected > 10 large, got {large_count}"


async def test_aggressive_fires_until_consecutive_429s(tmp_path: Path) -> None:
    from fastapi import HTTPException

    from codex_proxy.backend import HealthStatus, UsageSnapshot
    from codex_proxy.fakes import InMemoryFakeBackend
    from codex_proxy.synthetic import SyntheticTopper

    backend = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0e7"}),
        health=HealthStatus(available=True, reason="ok"),
        usage=UsageSnapshot(
            remaining_fraction=0.01,
            cooldown_until_ts=None,
            weekly_exhausted=True,
            probed_at_ts=0.0,
        ),
    )
    backend._fake_quota = _quota(weekly_used=99)  # type: ignore[attr-defined]

    dispatch_calls = []

    async def fake_dispatch(body: dict, backend_id: str) -> None:
        dispatch_calls.append(backend_id)
        raise HTTPException(status_code=429, detail="rate limited")

    cfg = AutoRouterConfig(
        aggressive_exhaustion_pct=98.0,
        aggressive_exhaustion_consecutive_429s=3,
    )

    log_path = tmp_path / "u.sqlite"
    topper = SyntheticTopper(
        cfg=cfg,
        usage_log_path=log_path,
        backends=[backend],
        dispatch=fake_dispatch,
    )

    await topper._tick()
    # Should have fired exactly 3 requests (the confirm threshold)
    assert len(dispatch_calls) == 3
    # Should exit aggressive mode after confirming
    assert not topper._in_aggressive_mode.get("alpha", False)
    await topper.stop()


async def test_aggressive_resets_streak_on_success(tmp_path: Path) -> None:
    from fastapi import HTTPException

    from codex_proxy.backend import HealthStatus, UsageSnapshot
    from codex_proxy.fakes import InMemoryFakeBackend
    from codex_proxy.synthetic import SyntheticTopper

    backend = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0e7"}),
        health=HealthStatus(available=True, reason="ok"),
        usage=UsageSnapshot(
            remaining_fraction=0.01,
            cooldown_until_ts=None,
            weekly_exhausted=True,
            probed_at_ts=0.0,
        ),
    )
    backend._fake_quota = _quota(weekly_used=99)  # type: ignore[attr-defined]

    call_sequence = ["ok", "ok", "429", "ok"]
    call_index = [0]

    async def fake_dispatch(body: dict, backend_id: str) -> None:
        response = call_sequence[call_index[0] % len(call_sequence)]
        call_index[0] += 1
        if response == "429":
            raise HTTPException(status_code=429, detail="rate limited")

    cfg = AutoRouterConfig(
        aggressive_exhaustion_pct=98.0,
        aggressive_exhaustion_consecutive_429s=3,
    )

    log_path = tmp_path / "u.sqlite"
    topper = SyntheticTopper(
        cfg=cfg,
        usage_log_path=log_path,
        backends=[backend],
        dispatch=fake_dispatch,
    )

    await topper._tick()
    # Should have 4 dispatch calls: ok, ok, 429, ok
    # After the final ok, streak should be 0
    assert topper._aggressive_consecutive_429s["alpha"] == 0
    await topper.stop()


async def test_aggressive_falls_back_on_non_429(tmp_path: Path) -> None:
    from codex_proxy.backend import HealthStatus, UsageSnapshot
    from codex_proxy.fakes import InMemoryFakeBackend
    from codex_proxy.synthetic import SyntheticTopper

    backend = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0e7"}),
        health=HealthStatus(available=True, reason="ok"),
        usage=UsageSnapshot(
            remaining_fraction=0.01,
            cooldown_until_ts=None,
            weekly_exhausted=True,
            probed_at_ts=0.0,
        ),
    )
    backend._fake_quota = _quota(weekly_used=99)  # type: ignore[attr-defined]

    async def fake_dispatch(body: dict, backend_id: str) -> None:
        raise RuntimeError("network error")

    cfg = AutoRouterConfig(aggressive_exhaustion_pct=98.0)

    log_path = tmp_path / "u.sqlite"
    topper = SyntheticTopper(
        cfg=cfg,
        usage_log_path=log_path,
        backends=[backend],
        dispatch=fake_dispatch,
    )

    await topper._tick()
    # Should exit aggressive mode on non-429 error
    assert not topper._in_aggressive_mode.get("alpha", False)
    await topper.stop()


async def test_normal_pacing_unchanged_below_threshold(tmp_path: Path) -> None:
    from codex_proxy.backend import HealthStatus, UsageSnapshot
    from codex_proxy.fakes import InMemoryFakeBackend
    from codex_proxy.synthetic import SyntheticTopper

    now_ts = 1_700_000_000.0

    backend = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0e7"}),
        health=HealthStatus(available=True, reason="ok"),
        usage=UsageSnapshot(
            remaining_fraction=0.5,
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=0.0,
        ),
    )
    # Provide weekly_reset_at so the pacing controller fires synthetics
    backend._fake_quota = _quota(  # type: ignore[attr-defined]
        weekly_used=50, weekly_reset_at=int(now_ts + 86400)
    )

    dispatch_calls = []

    async def fake_dispatch(body: dict, backend_id: str) -> None:
        dispatch_calls.append(body)

    cfg = AutoRouterConfig(
        aggressive_exhaustion_pct=98.0,
        pct_per_synthetic_estimate=0.1,
        prediction_safety_margin=1.0,
        weekly_target_pct=95.0,
        max_synthetics_per_tick=10,
        synthetic_check_interval_seconds=300,
    )

    log_path = tmp_path / "u.sqlite"
    topper = SyntheticTopper(
        cfg=cfg,
        usage_log_path=log_path,
        backends=[backend],
        dispatch=fake_dispatch,
        clock=lambda: now_ts,
    )

    await topper._tick()
    # Should not enter aggressive mode
    assert not topper._in_aggressive_mode.get("alpha", False)
    # Should use normal pacing (small payloads only)
    assert len(dispatch_calls) > 0
    for body in dispatch_calls:
        # Normal payload has single input element
        assert len(body["input"]) == 1
    await topper.stop()
