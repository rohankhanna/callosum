from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from callosum.backend import Backend, HealthStatus, UsageSnapshot
from callosum.codex_quota import CodexQuotaSnapshot
from callosum.routing.cost_estimator import CompositeCostEstimate

_UNKNOWN_REMAINING = 0.5
_UNKNOWN_PRESSURE = 0.5


@dataclass(frozen=True, slots=True)
class BackendSnapshot:
    backend: Backend
    health: HealthStatus
    usage: UsageSnapshot
    quota: CodexQuotaSnapshot | None = None


def rank(
    snapshots: Sequence[BackendSnapshot],
    *,
    model: str,
    now_ts: float,
    excluded: frozenset[str] | None = None,
    preferred_id: str | None = None,
    cost_estimate: CompositeCostEstimate | None = None,
) -> BackendSnapshot | None:
    """Pick the highest-ranked snapshot serving `model`. Returns None if none viable.

    When `preferred_id` names a viable backend, that one wins over the
    usage-based ranking. This is how sticky sessions pin a request to a
    previously bound backend without giving up the rotation fallback when
    that backend is no longer viable.
    """
    excluded_ids = excluded or frozenset()
    viable = [s for s in snapshots if _is_viable(s, model=model, now_ts=now_ts, excluded=excluded_ids)]
    if not viable:
        return None
    if preferred_id is not None:
        for snapshot in viable:
            if snapshot.backend.id == preferred_id:
                return snapshot
    return min(viable, key=lambda s: _rank_key(s, cost_estimate=cost_estimate))


def _is_viable(
    snapshot: BackendSnapshot,
    *,
    model: str,
    now_ts: float,
    excluded: frozenset[str],
) -> bool:
    if snapshot.backend.id in excluded:
        return False
    if not snapshot.health.available:
        return False
    if blocking_meters(snapshot, now_ts=now_ts):
        return False
    cooldown = snapshot.usage.cooldown_until_ts
    if cooldown is not None and cooldown > now_ts:
        return False
    return model in snapshot.backend.advertised_models


def _window_open(reset_at: float | int | None, now: float) -> bool:
    """True when the quota window a snapshot describes is still current.

    A meter at 100% only blocks routing while its window is open. Once
    `reset_at` passes, the window has provably rolled over and the percent
    in the snapshot is stale. The classic trigger is an overnight host
    shutdown: the 5-hourly/weekly window resets while the machine is off,
    but the persisted snapshot still reads exhausted, so on cold boot the
    proxy locks out a backend whose quota actually reset hours ago. Treat a
    passed reset as an open, un-blocked window so the next real request
    re-probes upstream instead of 503-ing on yesterday's number. A `None`
    reset (cold start, no upstream datum yet) is treated as open and the
    percent honored as-is.
    """
    return reset_at is None or now < float(reset_at)


def blocking_meters(snapshot: BackendSnapshot, *, now_ts: float | None = None) -> tuple[str, ...]:
    now = time.time() if now_ts is None else now_ts
    quota = snapshot.quota
    blocked: list[str] = []
    weekly_window_open = True
    if quota is not None:
        if (
            quota.five_hourly_used_percent is not None
            and quota.five_hourly_used_percent >= 100
            and _window_open(quota.five_hourly_reset_at, now)
        ):
            blocked.append("five_hourly")
        weekly_window_open = _window_open(quota.weekly_reset_at, now)
        if quota.weekly_used_percent is not None and quota.weekly_used_percent >= 100 and weekly_window_open:
            blocked.append("weekly")
    # The sticky weekly_exhausted usage flag is only meaningful while the
    # weekly window it was set in is still open; after weekly_reset_at it is
    # stale (same overnight-shutdown failure mode as the percent meters).
    if snapshot.usage.weekly_exhausted and "weekly" not in blocked and weekly_window_open:
        blocked.append("weekly")
    return tuple(blocked)


def _meter_pressure(used_percent: int | None, estimated_high: float) -> float | None:
    if used_percent is None:
        return None
    remaining = max(0.0, 100.0 - float(used_percent))
    if remaining <= 0.0:
        return float("inf")
    return max(0.0, estimated_high) / remaining


def composite_pressure(
    snapshot: BackendSnapshot,
    *,
    cost_estimate: CompositeCostEstimate | None,
    now_ts: float | None = None,
) -> tuple[float, str | None]:
    if cost_estimate is None or snapshot.quota is None:
        return (_UNKNOWN_PRESSURE, None)
    now = time.time() if now_ts is None else now_ts
    quota = snapshot.quota
    # A meter whose window has reset contributes no pressure: its stale
    # used-percent must not deprioritize a backend that just regained
    # headroom across an overnight shutdown (mirrors blocking_meters).
    five_used = quota.five_hourly_used_percent if _window_open(quota.five_hourly_reset_at, now) else 0
    weekly_used = quota.weekly_used_percent if _window_open(quota.weekly_reset_at, now) else 0
    pressures = [
        (
            _meter_pressure(five_used, cost_estimate.five_hourly.high),
            "five_hourly",
        ),
        (
            _meter_pressure(weekly_used, cost_estimate.weekly.high),
            "weekly",
        ),
    ]
    known = [(pressure, meter) for pressure, meter in pressures if pressure is not None]
    if not known:
        return (_UNKNOWN_PRESSURE, None)
    return max(known, key=lambda item: item[0])


def constraining_meter(snapshot: BackendSnapshot) -> str | None:
    blocked = blocking_meters(snapshot)
    if blocked:
        return blocked[0]
    quota = snapshot.quota
    if quota is None:
        return None
    candidates: list[tuple[float, str]] = []
    if quota.five_hourly_used_percent is not None:
        candidates.append((100.0 - float(quota.five_hourly_used_percent), "five_hourly"))
    if quota.weekly_used_percent is not None:
        candidates.append((100.0 - float(quota.weekly_used_percent), "weekly"))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _rank_key(
    snapshot: BackendSnapshot,
    *,
    cost_estimate: CompositeCostEstimate | None,
) -> tuple[float, int, float, float, str]:
    usage = snapshot.usage
    weekly = 1 if usage.weekly_exhausted else 0
    remaining = usage.remaining_fraction
    if remaining is None:
        remaining = _UNKNOWN_REMAINING
    pressure, _meter = composite_pressure(snapshot, cost_estimate=cost_estimate)
    return (pressure, weekly, -remaining, -usage.probed_at_ts, snapshot.backend.id)


async def select(
    backends: Sequence[Backend],
    *,
    model: str,
    now_ts: float | None = None,
    excluded: frozenset[str] | None = None,
    preferred_id: str | None = None,
    cost_estimate: CompositeCostEstimate | None = None,
) -> Backend | None:
    """Gather snapshots from each backend and pick the best one for `model`."""
    if not backends:
        return None
    resolved_now = time.time() if now_ts is None else now_ts
    snapshots: list[BackendSnapshot] = []
    for backend in backends:
        health = await backend.health()
        usage = await backend.usage_snapshot()
        quota = await backend.quota_snapshot()
        snapshots.append(BackendSnapshot(backend=backend, health=health, usage=usage, quota=quota))
    chosen = rank(
        snapshots,
        model=model,
        now_ts=resolved_now,
        excluded=excluded,
        preferred_id=preferred_id,
        cost_estimate=cost_estimate,
    )
    return chosen.backend if chosen is not None else None
