"""Background synthetic-request topper for the auto-learning explorer.

Synthetic requests SUPPLEMENT organic auto-learning traffic. Two bounds, both
active simultaneously:

- Floor: minimum synthetics per UTC day so corpus velocity doesn't bottleneck
  on quiet days.
- Pct cap: max growth as a fraction of today's organic volume so the corpus
  isn't biased by synthetic prompts on busy days.

Effective target ≈ min(hard_ceiling, max(floor, ceil(pct * organic))).

The worker re-evaluates this on every tick so config changes / hot reloads
take effect without a restart.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import sqlite3
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True, slots=True)
class DailyCounts:
    organic: int
    synthetic: int


def _utc_day_start_ts(now_ts: float) -> float:
    """Return the unix timestamp of 00:00:00 UTC for the day containing now_ts."""
    dt = datetime.fromtimestamp(now_ts, tz=UTC)
    start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.timestamp()


def daily_counts(usage_log_path: Path, *, now_ts: float) -> DailyCounts:
    """Count today's organic and synthetic auto-learning requests in the log.

    Returns zeros when the log file doesn't exist yet (fresh deploy).
    """
    if not usage_log_path.exists():
        return DailyCounts(organic=0, synthetic=0)
    day_start = _utc_day_start_ts(now_ts)
    conn = sqlite3.connect(usage_log_path)
    try:
        rows = conn.execute(
            "SELECT routing_mode, COUNT(*) FROM requests"
            " WHERE ts_start >= ? AND routing_mode IN (?, ?)"
            " GROUP BY routing_mode",
            (day_start, "auto-learning", "auto-learning-synthetic"),
        ).fetchall()
    finally:
        conn.close()
    by_mode = dict(rows)
    return DailyCounts(
        organic=int(by_mode.get("auto-learning", 0)),
        synthetic=int(by_mode.get("auto-learning-synthetic", 0)),
    )


def synthetic_target_for_today(counts: DailyCounts, cfg: AutoRouterConfig) -> int:
    """Compute today's synthetic target given the floor + pct + ceiling bounds.

    Returns the number of synthetic requests that *should* exist by end of day.
    The worker fires more synthetics when actual count < target.
    """
    floor = max(0, cfg.synthetic_floor_per_day)
    pct_target = ceil(cfg.synthetic_pct_of_organic * counts.organic)
    target = max(floor, pct_target)
    if cfg.synthetic_hard_ceiling_per_day > 0:
        target = min(target, cfg.synthetic_hard_ceiling_per_day)
    return target


def should_fire(counts: DailyCounts, cfg: AutoRouterConfig) -> bool:
    """Decide whether to fire one synthetic request right now."""
    if not _enabled(cfg):
        return False
    return counts.synthetic < synthetic_target_for_today(counts, cfg)


def _enabled(cfg: AutoRouterConfig) -> bool:
    return cfg.synthetic_floor_per_day > 0 or cfg.synthetic_pct_of_organic > 0


class SyntheticTopper:
    """Background asyncio task that periodically fires synthetic auto-learning
    requests through the proxy's own dispatch path.

    Intentionally minimal: one tick = inspect today's counts, decide, maybe fire
    one synthetic. The cap on synthetics-per-tick is 1 so high pct + low
    interval don't burst — the rate is paced naturally by the check interval.
    """

    def __init__(
        self,
        *,
        cfg: AutoRouterConfig,
        usage_log_path: Path | None,
        dispatch: Callable[[dict[str, Any]], Awaitable[Any]],
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._cfg = cfg
        self._usage_log_path = usage_log_path
        self._dispatch = dispatch
        self._clock = clock if clock is not None else _wall_clock
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @property
    def enabled(self) -> bool:
        return _enabled(self._cfg) and self._usage_log_path is not None

    def start(self) -> None:
        if not self.enabled:
            logger.info("synthetic topper disabled (config defaults or no usage log)")
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
                logger.exception("synthetic topper tick failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
            except TimeoutError:
                continue

    async def _tick(self) -> None:
        if self._usage_log_path is None:
            return
        counts = daily_counts(self._usage_log_path, now_ts=self._clock())
        if not should_fire(counts, self._cfg):
            return
        body = synthetic_body()
        try:
            await self._dispatch(body)
        except Exception:
            # Dispatch failures are already logged by the usage_log path; the
            # topper just notes that it tried so we don't tight-loop on them.
            logger.exception("synthetic dispatch failed")


def _wall_clock() -> float:
    import time

    return time.time()
