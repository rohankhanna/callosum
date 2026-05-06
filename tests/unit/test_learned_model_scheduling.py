"""Tests for adaptive learned model scheduling with two-phase ramp."""

from __future__ import annotations

import time

import pytest

from codex_proxy.app import _learned_model_cap_pct


class TestInitialPhase:
    """Tests for the initial 90-day ramp from proxy startup (no model release)."""

    def test_cold_start_day_1(self) -> None:
        """On day 1 of proxy startup, cap is 1%."""
        now = time.time()
        cap = _learned_model_cap_pct(startup_timestamp=now, model_release_timestamp=None)
        assert 0.99 < cap < 1.01

    def test_startup_day_45(self) -> None:
        """After 45 days, cap should be ~50%: 1 + (45/90)*89 = 45.5%."""
        now = time.time()
        startup_45_days_ago = now - (45 * 86400)
        cap = _learned_model_cap_pct(startup_timestamp=startup_45_days_ago, model_release_timestamp=None)
        assert 45.0 < cap < 46.0

    def test_startup_day_90(self) -> None:
        """After 90 days, cap should be fully ramped to 90%."""
        now = time.time()
        startup_90_days_ago = now - (90 * 86400)
        cap = _learned_model_cap_pct(startup_timestamp=startup_90_days_ago, model_release_timestamp=None)
        assert cap == 90.0

    def test_startup_day_100(self) -> None:
        """After 100+ days, cap stays at 90%."""
        now = time.time()
        startup_100_days_ago = now - (100 * 86400)
        cap = _learned_model_cap_pct(startup_timestamp=startup_100_days_ago, model_release_timestamp=None)
        assert cap == 90.0

    def test_no_timestamps(self) -> None:
        """With no timestamps at all, cap defaults to 1% (conservative for cold start)."""
        cap = _learned_model_cap_pct(startup_timestamp=None, model_release_timestamp=None)
        assert cap == 1.0


class TestModelReleasePhase:
    """Tests for the 30-day ramp after model release (75% → 90%)."""

    def test_model_release_day_0(self) -> None:
        """On model release day, cap resets to 75%."""
        now = time.time()
        cap = _learned_model_cap_pct(startup_timestamp=None, model_release_timestamp=now)
        assert 74.99 < cap < 75.01

    def test_model_release_day_15(self) -> None:
        """After 15 days post-release, cap should be ~82.5%: 75 + (15/30)*15 = 82.5%."""
        now = time.time()
        release_15_days_ago = now - (15 * 86400)
        cap = _learned_model_cap_pct(startup_timestamp=None, model_release_timestamp=release_15_days_ago)
        assert 82.4 < cap < 82.6

    def test_model_release_day_30(self) -> None:
        """After 30 days post-release, cap should be fully ramped to 90%."""
        now = time.time()
        release_30_days_ago = now - (30 * 86400)
        cap = _learned_model_cap_pct(startup_timestamp=None, model_release_timestamp=release_30_days_ago)
        assert cap == 90.0

    def test_model_release_day_60(self) -> None:
        """After 60+ days post-release, cap stays at 90%."""
        now = time.time()
        release_60_days_ago = now - (60 * 86400)
        cap = _learned_model_cap_pct(startup_timestamp=None, model_release_timestamp=release_60_days_ago)
        assert cap == 90.0


class TestPhaseInteraction:
    """Tests for interaction between startup and model release phases."""

    def test_model_release_overrides_startup(self) -> None:
        """When model release is set, it takes precedence over startup phase."""
        now = time.time()
        # Startup was 100 days ago (would be at 90%)
        startup_100_days_ago = now - (100 * 86400)
        # But model released 5 days ago (should be ramping from 75%)
        release_5_days_ago = now - (5 * 86400)

        cap = _learned_model_cap_pct(
            startup_timestamp=startup_100_days_ago,
            model_release_timestamp=release_5_days_ago,
        )
        # Should use model release phase: 75 + (5/30)*15 = 77.5%
        assert 77.4 < cap < 77.6

    def test_startup_after_release_holdout(self) -> None:
        """After release phase reaches 90%, subsequent updates still use 90%."""
        now = time.time()
        startup_100_days_ago = now - (100 * 86400)
        release_100_days_ago = now - (100 * 86400)

        cap = _learned_model_cap_pct(
            startup_timestamp=startup_100_days_ago,
            model_release_timestamp=release_100_days_ago,
        )
        # Both phases would give 90%, so definitely 90%
        assert cap == 90.0
