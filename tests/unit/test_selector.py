from __future__ import annotations

import time

from callosum.backend import HealthStatus, UsageSnapshot
from callosum.codex_quota import CodexQuotaSnapshot
from callosum.fakes import InMemoryFakeBackend
from callosum.routing.cost_estimator import CompositeCostEstimate
from callosum.routing.usage_estimate import Estimate
from callosum.selector import BackendSnapshot, blocking_meters, constraining_meter, select


def _usage(
    *,
    remaining: float | None = 1.0,
    cooldown_until: float | None = None,
    weekly_exhausted: bool = False,
    probed_at: float = 0.0,
) -> UsageSnapshot:
    return UsageSnapshot(
        remaining_fraction=remaining,
        cooldown_until_ts=cooldown_until,
        weekly_exhausted=weekly_exhausted,
        probed_at_ts=probed_at,
    )


def _fake(
    id: str,
    *,
    advertised_models: frozenset[str] | None = None,
    health: HealthStatus | None = None,
    usage: UsageSnapshot | None = None,
    quota: CodexQuotaSnapshot | None = None,
) -> InMemoryFakeBackend:
    backend = InMemoryFakeBackend(
        id=id,
        advertised_models=advertised_models or frozenset({"model-a0d0"}),
        health=health,
        usage=usage,
    )
    if quota is not None:
        backend._fake_quota = quota
    return backend


def _quota(
    *,
    five_hourly: int | None,
    weekly: int | None,
    five_hourly_reset_at: int | None = None,
    weekly_reset_at: int | None = None,
) -> CodexQuotaSnapshot:
    return CodexQuotaSnapshot(
        plan_type=None,
        active_limit=None,
        five_hourly_used_percent=five_hourly,
        weekly_used_percent=weekly,
        five_hourly_window_minutes=None,
        weekly_window_minutes=None,
        five_hourly_reset_at=five_hourly_reset_at,
        weekly_reset_at=weekly_reset_at,
        five_hourly_reset_after_seconds=None,
        weekly_reset_after_seconds=None,
        five_hourly_over_weekly_limit_percent=None,
        credits_balance=None,
        credits_has_credits=None,
        credits_unlimited=None,
        observed_at=0.0,
    )


def _estimate(*, five_hourly: float, weekly: float) -> CompositeCostEstimate:
    return CompositeCostEstimate(
        five_hourly=Estimate(
            point=five_hourly,
            low=five_hourly,
            high=five_hourly,
            unit="five_hourly_used_percent",
            source="test",
            verifiable=True,
        ),
        weekly=Estimate(
            point=weekly,
            low=weekly,
            high=weekly,
            unit="weekly_used_percent",
            source="test",
            verifiable=True,
        ),
    )


async def test_empty_pool_returns_none() -> None:
    assert await select([], model="model-a0d0") is None


async def test_single_healthy_backend_is_chosen() -> None:
    backend = _fake("a")
    assert await select([backend], model="model-a0d0") is backend


async def test_backend_without_model_is_skipped() -> None:
    backend = _fake("a", advertised_models=frozenset({"other-model"}))
    assert await select([backend], model="model-a0d0") is None


async def test_unavailable_backend_is_skipped() -> None:
    backend = _fake("a", health=HealthStatus(available=False, reason="rate_limited"))
    assert await select([backend], model="model-a0d0") is None


async def test_backend_in_cooldown_is_skipped() -> None:
    backend = _fake("a", usage=_usage(cooldown_until=100.0))
    assert await select([backend], model="model-a0d0", now_ts=50.0) is None


async def test_backend_past_cooldown_is_included() -> None:
    backend = _fake("a", usage=_usage(cooldown_until=100.0))
    assert await select([backend], model="model-a0d0", now_ts=200.0) is backend


async def test_weekly_exhausted_is_deprioritized() -> None:
    a = _fake("a", usage=_usage(weekly_exhausted=True))
    b = _fake("b", usage=_usage(weekly_exhausted=False))
    assert await select([a, b], model="model-a0d0") is b


async def test_higher_remaining_fraction_preferred() -> None:
    a = _fake("a", usage=_usage(remaining=0.2))
    b = _fake("b", usage=_usage(remaining=0.8))
    assert await select([a, b], model="model-a0d0") is b


async def test_none_remaining_is_treated_as_half() -> None:
    low = _fake("low", usage=_usage(remaining=0.3))
    unknown = _fake("unknown", usage=_usage(remaining=None))
    assert await select([low, unknown], model="model-a0d0") is unknown


async def test_recency_breaks_ties_when_usage_equal() -> None:
    stale = _fake("stale", usage=_usage(remaining=0.5, probed_at=100.0))
    fresh = _fake("fresh", usage=_usage(remaining=0.5, probed_at=200.0))
    assert await select([stale, fresh], model="model-a0d0") is fresh


async def test_id_is_final_deterministic_tiebreak() -> None:
    a = _fake("a", usage=_usage(remaining=0.5, probed_at=100.0))
    b = _fake("b", usage=_usage(remaining=0.5, probed_at=100.0))
    assert await select([a, b], model="model-a0d0") is a


async def test_all_unhealthy_returns_none() -> None:
    a = _fake("a", health=HealthStatus(available=False, reason="auth_invalid"))
    b = _fake("b", health=HealthStatus(available=False, reason="network"))
    assert await select([a, b], model="model-a0d0") is None


async def test_mixed_pool_picks_best_eligible() -> None:
    pool = [
        _fake("unhealthy", health=HealthStatus(available=False, reason="network")),
        _fake("wrong-model", advertised_models=frozenset({"other"})),
        _fake("exhausted", usage=_usage(weekly_exhausted=True, remaining=1.0)),
        _fake("ok-low", usage=_usage(remaining=0.1)),
        _fake("ok-high", usage=_usage(remaining=0.9)),
    ]
    chosen = await select(pool, model="model-a0d0")
    assert chosen is not None
    assert chosen.id == "ok-high"


async def test_five_hourly_pressure_can_outrank_low_weekly_cost() -> None:
    risky_5h = _fake("risky-5h", quota=_quota(five_hourly=98, weekly=10))
    roomy = _fake("roomy", quota=_quota(five_hourly=50, weekly=50))
    chosen = await select(
        [risky_5h, roomy],
        model="model-a0d0",
        cost_estimate=_estimate(five_hourly=3.0, weekly=0.1),
    )
    assert chosen is roomy


async def test_weekly_pressure_can_outrank_low_five_hourly_cost() -> None:
    risky_weekly = _fake("risky-weekly", quota=_quota(five_hourly=10, weekly=98))
    roomy = _fake("roomy", quota=_quota(five_hourly=50, weekly=50))
    chosen = await select(
        [risky_weekly, roomy],
        model="model-a0d0",
        cost_estimate=_estimate(five_hourly=0.1, weekly=3.0),
    )
    assert chosen is roomy


async def test_insufficient_quota_data_falls_back_to_remaining_fraction() -> None:
    low = _fake("low", usage=_usage(remaining=0.2))
    high = _fake("high", usage=_usage(remaining=0.8))
    chosen = await select(
        [low, high],
        model="model-a0d0",
        cost_estimate=_estimate(five_hourly=5.0, weekly=5.0),
    )
    assert chosen is high


async def test_five_hourly_exhaustion_blocks_backend() -> None:
    exhausted = _fake("exhausted-5h", quota=_quota(five_hourly=100, weekly=1))
    available = _fake("available", quota=_quota(five_hourly=1, weekly=99))
    chosen = await select([exhausted, available], model="model-a0d0")
    assert chosen is available


async def test_diagnostics_expose_blocking_and_constraining_meter() -> None:
    backend = _fake("blocked", quota=_quota(five_hourly=100, weekly=42))
    snapshot = BackendSnapshot(
        backend=backend,
        health=await backend.health(),
        usage=await backend.usage_snapshot(),
        quota=await backend.quota_snapshot(),
    )
    assert blocking_meters(snapshot) == ("five_hourly",)
    assert constraining_meter(snapshot) == "five_hourly"


# ---- reset-aware staleness (overnight-shutdown lockout) -----------------
#
# A snapshot that reads 100% but whose window already reset must NOT block.
# This is the cold-boot failure mode: the host is off all night, the 5h/
# weekly window rolls over while it sleeps, and the persisted snapshot is
# stale until a real request can re-probe upstream. blocking_meters must
# date the percent against the snapshot's own reset_at.

_NOW = int(time.time())
_PAST = _NOW - 3600  # window reset an hour ago
_FUTURE = _NOW + 3600  # window still open for another hour


def _snap(backend: InMemoryFakeBackend, *, quota: CodexQuotaSnapshot, usage: UsageSnapshot | None = None) -> BackendSnapshot:
    return BackendSnapshot(
        backend=backend,
        health=HealthStatus(available=True, reason="ok"),
        usage=usage or _usage(),
        quota=quota,
    )


def test_exhausted_meter_with_passed_reset_does_not_block() -> None:
    snap = _snap(_fake("stale"), quota=_quota(five_hourly=100, weekly=100, five_hourly_reset_at=_PAST, weekly_reset_at=_PAST))
    assert blocking_meters(snap, now_ts=_NOW) == ()


def test_exhausted_meter_with_open_window_still_blocks() -> None:
    snap = _snap(_fake("live"), quota=_quota(five_hourly=100, weekly=1, five_hourly_reset_at=_FUTURE, weekly_reset_at=_FUTURE))
    assert blocking_meters(snap, now_ts=_NOW) == ("five_hourly",)


def test_weekly_exhausted_flag_is_stale_after_weekly_reset() -> None:
    snap = _snap(
        _fake("weekly-stale"),
        quota=_quota(five_hourly=1, weekly=1, weekly_reset_at=_PAST),
        usage=_usage(weekly_exhausted=True),
    )
    assert blocking_meters(snap, now_ts=_NOW) == ()


def test_weekly_exhausted_flag_honored_with_no_quota() -> None:
    # No quota snapshot to date the flag against → honor it (cold-start safe).
    snap = BackendSnapshot(
        backend=_fake("no-quota"),
        health=HealthStatus(available=True, reason="ok"),
        usage=_usage(weekly_exhausted=True),
        quota=None,
    )
    assert blocking_meters(snap, now_ts=_NOW) == ("weekly",)


async def test_select_recovers_backend_after_overnight_window_reset() -> None:
    # The end-to-end morning case: the only backend reads 100% from last
    # night, but its window reset hours ago. It must be selectable again.
    stale = _fake("overnight", quota=_quota(five_hourly=100, weekly=100, five_hourly_reset_at=_PAST, weekly_reset_at=_PAST))
    chosen = await select([stale], model="model-a0d0", now_ts=float(_NOW))
    assert chosen is stale
