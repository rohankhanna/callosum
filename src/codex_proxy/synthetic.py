"""Background synthetic-request worker for the auto-learning explorer.

Two controllers:

1. **Weekly-exhaustion controller** (primary). For each backend, every tick:
   read the latest CodexQuotaSnapshot, compute "how much weekly% should
   synthetics burn before reset" given projected human burn (last 7d organic
   rate × safety margin), and fire that many synthetics targeted at this
   backend. Honors the invariant that paid-monthly weekly capacity is never
   wasted.

2. **Cold-start fallback** (secondary). Used per-backend when no quota
   snapshot is available yet. Falls back to the original floor + pct + ceiling
   target (per backend, per UTC day). Same hard_ceiling acts as a safety cap.

Synthetics are pinned to the chosen backend via a `forced_backend_id` argument
to dispatch — the natural selector picks "least-loaded by 5h capacity," which
isn't the same thing as "the account closest to weekly reset with leftover
weekly capacity."
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import sqlite3
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import Any

from codex_proxy.backend import Backend
from codex_proxy.codex_quota import CodexQuotaSnapshot
from codex_proxy.config import AutoRouterConfig

logger = logging.getLogger(__name__)


# Short, deterministic, domain-bland prompts. Each call burns a small amount
# of quota; the bland phrasing keeps responses short so output tokens stay low.
_PROMPT_CORPUS: tuple[str, ...] = (
    "Reply with only the word: ok",
    "Reply with only the word: hello",
    "Reply with only a single digit: 1",
    "Reply with only the word: ready",
    "Reply with only the word: ack",
)


def synthetic_body() -> dict[str, Any]:
    """Build a Responses-API request body for one synthetic call."""
    prompt = random.choice(_PROMPT_CORPUS)
    return {
        "model": "auto-learning-synthetic",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "stream": False,
    }


# ---------------------------------------------------------------------------
# Cold-start fallback controller (per-backend, per-day count target)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DailyCounts:
    organic: int
    synthetic: int


def _utc_day_start_ts(now_ts: float) -> float:
    """Return the unix timestamp of 00:00:00 UTC for the day containing now_ts."""
    dt = datetime.fromtimestamp(now_ts, tz=UTC)
    start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.timestamp()


def daily_counts(
    usage_log_path: Path, *, now_ts: float, backend_id: str | None = None
) -> DailyCounts:
    """Count today's organic and synthetic auto-learning requests in the log.

    `backend_id` filters to a single backend when given (for per-account
    cold-start counting); when None, counts globally.

    Returns zeros when the log file doesn't exist yet (fresh deploy).
    """
    if not usage_log_path.exists():
        return DailyCounts(organic=0, synthetic=0)
    day_start = _utc_day_start_ts(now_ts)
    conn = sqlite3.connect(usage_log_path)
    try:
        if backend_id is None:
            rows = conn.execute(
                "SELECT routing_mode, COUNT(*) FROM requests"
                " WHERE ts_start >= ? AND routing_mode IN (?, ?)"
                " GROUP BY routing_mode",
                (day_start, "auto-learning", "auto-learning-synthetic"),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT routing_mode, COUNT(*) FROM requests"
                " WHERE ts_start >= ? AND backend_id = ?"
                "   AND routing_mode IN (?, ?)"
                " GROUP BY routing_mode",
                (day_start, backend_id, "auto-learning", "auto-learning-synthetic"),
            ).fetchall()
    finally:
        conn.close()
    by_mode = dict(rows)
    return DailyCounts(
        organic=int(by_mode.get("auto-learning", 0)),
        synthetic=int(by_mode.get("auto-learning-synthetic", 0)),
    )


def cold_start_target(counts: DailyCounts, cfg: AutoRouterConfig) -> int:
    """Cold-start target: synthetic count we should reach by end of day for one
    backend, given the floor + pct + ceiling fallback bounds.
    """
    floor = max(0, cfg.synthetic_floor_per_day)
    pct_target = ceil(cfg.synthetic_pct_of_organic * counts.organic)
    target = max(floor, pct_target)
    if cfg.synthetic_hard_ceiling_per_day > 0:
        target = min(target, cfg.synthetic_hard_ceiling_per_day)
    return target


def cold_start_fire_count(counts: DailyCounts, cfg: AutoRouterConfig) -> int:
    """How many synthetics to fire on a backend with no quota snapshot yet."""
    if not _cold_start_enabled(cfg):
        return 0
    deficit = cold_start_target(counts, cfg) - counts.synthetic
    return max(0, min(deficit, cfg.max_synthetics_per_tick))


def _cold_start_enabled(cfg: AutoRouterConfig) -> bool:
    return cfg.synthetic_floor_per_day > 0 or cfg.synthetic_pct_of_organic > 0


# ---------------------------------------------------------------------------
# Weekly-exhaustion controller (per-account, primary)
# ---------------------------------------------------------------------------


def organic_burn_rate_pct_per_hour(
    usage_log_path: Path,
    backend_id: str,
    *,
    now_ts: float,
    window_hours: float,
) -> float:
    """Estimate organic weekly%-burn rate on a backend over the last
    `window_hours`. Sums (weekly_used_percent_after − before) across organic
    + pass-through rows with both quota snapshots present and no reset
    crossover.
    """
    if not usage_log_path.exists():
        return 0.0
    if window_hours <= 0:
        return 0.0
    window_start = now_ts - window_hours * 3600.0
    conn = sqlite3.connect(usage_log_path)
    try:
        row = conn.execute(
            "SELECT SUM(weekly_used_percent_after - weekly_used_percent_before)"
            " FROM requests"
            " WHERE backend_id = ?"
            "   AND ts_start >= ?"
            "   AND routing_mode IN ('auto-learning', 'pass-through')"
            "   AND status = 200"
            "   AND quota_reset_crossover = 0"
            "   AND weekly_used_percent_before IS NOT NULL"
            "   AND weekly_used_percent_after IS NOT NULL",
            (backend_id, window_start),
        ).fetchone()
    finally:
        conn.close()
    total = row[0] if row[0] is not None else 0
    return max(0.0, float(total) / window_hours)


def weekly_exhaustion_fire_count(
    quota: CodexQuotaSnapshot,
    *,
    organic_rate_pct_per_hour: float,
    now_ts: float,
    cfg: AutoRouterConfig,
) -> int:
    """How many synthetics to fire on a backend this tick to keep its weekly
    window on track to land at 100% by reset.

    Returns 0 when:
      - quota snapshot is incomplete (no weekly_used_percent or weekly_reset_at)
      - account is past `weekly_target_pct` (close enough; stop)
      - account is past `five_hourly_pause_pct` (rate-limited; pause)
      - reset is imminent or in the past (let it cycle)
      - projected human burn alone will exhaust the weekly window
    """
    if quota.weekly_used_percent is None or quota.weekly_reset_at is None:
        return 0
    if quota.weekly_used_percent >= cfg.weekly_target_pct:
        return 0
    if (
        quota.five_hourly_used_percent is not None
        and quota.five_hourly_used_percent >= cfg.five_hourly_pause_pct
    ):
        return 0
    hours_remaining = max(0.0, (quota.weekly_reset_at - now_ts) / 3600.0)
    if hours_remaining < 0.1:
        return 0
    weekly_remaining_pct = float(cfg.weekly_target_pct) - float(quota.weekly_used_percent)
    if weekly_remaining_pct <= 0:
        return 0
    projected_organic_pct = (
        organic_rate_pct_per_hour * hours_remaining * cfg.prediction_safety_margin
    )
    burnable_pct = max(0.0, weekly_remaining_pct - projected_organic_pct)
    if burnable_pct <= 0:
        return 0
    pct_per_call = max(1e-6, cfg.pct_per_synthetic_estimate)
    synthetics_total = burnable_pct / pct_per_call
    tick_hours = max(1, cfg.synthetic_check_interval_seconds) / 3600.0
    synthetics_this_tick = synthetics_total * tick_hours / hours_remaining
    n = int(round(synthetics_this_tick))
    return max(0, min(cfg.max_synthetics_per_tick, n))


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


SyntheticDispatch = Callable[[dict[str, Any], str], Awaitable[Any]]


class SyntheticTopper:
    """Background asyncio task. Every tick, for each backend:

    1. Try the weekly-exhaustion controller first (uses per-account quota
       snapshot + organic burn rate from log).
    2. If no quota snapshot is available yet, fall back to the cold-start
       per-backend floor + pct + ceiling logic.

    Synthetics fired here are pinned to the backend that earned them via
    `forced_backend_id`.
    """

    def __init__(
        self,
        *,
        cfg: AutoRouterConfig,
        usage_log_path: Path | None,
        backends: Sequence[Backend],
        dispatch: SyntheticDispatch,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._cfg = cfg
        self._usage_log_path = usage_log_path
        self._backends = list(backends)
        self._dispatch = dispatch
        self._clock = clock if clock is not None else _wall_clock
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @property
    def enabled(self) -> bool:
        # The weekly controller has no on/off knob — it kicks in as soon as a
        # quota snapshot exists. The cold-start fallback only activates when
        # floor or pct is set. Either path enables the worker — but both need
        # a usage_log + at least one backend.
        return self._usage_log_path is not None and bool(self._backends)

    def start(self) -> None:
        if not self.enabled:
            logger.info("synthetic worker disabled (no backends or no usage log)")
            return
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="synthetic-topper")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        interval = max(1, self._cfg.synthetic_check_interval_seconds)
        while not self._stop.is_set():
            try:
                await self._tick()
            except Exception:
                logger.exception("synthetic worker tick failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except TimeoutError:
                continue

    async def _tick(self) -> None:
        if self._usage_log_path is None:
            return
        now_ts = self._clock()
        for backend in self._backends:
            n = await self._fire_count_for_backend(backend, now_ts=now_ts)
            for _ in range(n):
                try:
                    await self._dispatch(synthetic_body(), backend.id)
                except Exception:
                    logger.exception("synthetic dispatch failed for %s", backend.id)
                    break  # don't tight-loop on a sick backend

    async def _fire_count_for_backend(self, backend: Backend, *, now_ts: float) -> int:
        snap: Any = await backend.quota_snapshot()
        if snap is not None and self._usage_log_path is not None:
            organic_rate = organic_burn_rate_pct_per_hour(
                self._usage_log_path,
                backend.id,
                now_ts=now_ts,
                window_hours=float(self._cfg.prediction_window_hours),
            )
            return weekly_exhaustion_fire_count(
                snap,
                organic_rate_pct_per_hour=organic_rate,
                now_ts=now_ts,
                cfg=self._cfg,
            )
        if self._usage_log_path is None:
            return 0
        counts = daily_counts(self._usage_log_path, now_ts=now_ts, backend_id=backend.id)
        return cold_start_fire_count(counts, self._cfg)


def _wall_clock() -> float:
    import time

    return time.time()
