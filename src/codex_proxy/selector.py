from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from codex_proxy.backend import Backend, HealthStatus, UsageSnapshot

_UNKNOWN_REMAINING = 0.5


@dataclass(frozen=True, slots=True)
class BackendSnapshot:
    backend: Backend
    health: HealthStatus
    usage: UsageSnapshot


def rank(
    snapshots: Sequence[BackendSnapshot],
    *,
    model: str,
    now_ts: float,
    excluded: frozenset[str] | None = None,
) -> BackendSnapshot | None:
    """Pick the highest-ranked snapshot serving `model`. Returns None if none viable."""
    excluded_ids = excluded or frozenset()
    viable = [
        s for s in snapshots if _is_viable(s, model=model, now_ts=now_ts, excluded=excluded_ids)
    ]
    if not viable:
        return None
    return min(viable, key=_rank_key)


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
    cooldown = snapshot.usage.cooldown_until_ts
    if cooldown is not None and cooldown > now_ts:
        return False
    return model in snapshot.backend.advertised_models


def _rank_key(snapshot: BackendSnapshot) -> tuple[int, float, float, str]:
    usage = snapshot.usage
    weekly = 1 if usage.weekly_exhausted else 0
    remaining = usage.remaining_fraction
    if remaining is None:
        remaining = _UNKNOWN_REMAINING
    return (weekly, -remaining, -usage.probed_at_ts, snapshot.backend.id)


async def select(
    backends: Sequence[Backend],
    *,
    model: str,
    now_ts: float | None = None,
    excluded: frozenset[str] | None = None,
) -> Backend | None:
    """Gather snapshots from each backend and pick the best one for `model`."""
    if not backends:
        return None
    resolved_now = time.time() if now_ts is None else now_ts
    snapshots: list[BackendSnapshot] = []
    for backend in backends:
        health = await backend.health()
        usage = await backend.usage_snapshot()
        snapshots.append(BackendSnapshot(backend=backend, health=health, usage=usage))
    chosen = rank(snapshots, model=model, now_ts=resolved_now, excluded=excluded)
    return chosen.backend if chosen is not None else None
