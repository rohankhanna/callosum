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
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from codex_proxy.backend import Backend
from codex_proxy.codex_quota import CodexQuotaSnapshot
from codex_proxy.config import AutoRouterConfig
from codex_proxy.errors import BackendError

if TYPE_CHECKING:
    from codex_proxy.router import LearnedModelRouter

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

# Codex's Responses API rejects requests without a non-empty `instructions`
# field (400: "Instructions are required"). Mirrors the constant in app.py
# used by the smoke test for the same reason.
_SYNTHETIC_INSTRUCTIONS = "You are a synthetic auto-learning probe. Reply minimally."

# Medium-tier payloads: multi-turn conversations (100-500 tokens)
_MEDIUM_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "instructions": _SYNTHETIC_INSTRUCTIONS,
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Explain the difference between TCP and UDP in two sentences."}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "TCP is connection-oriented and guarantees ordered, reliable delivery via acknowledgments and retransmissions. UDP is connectionless and prioritizes low latency over reliability, with no built-in retransmission or ordering."}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Which would you choose for a live video stream and why?"}]},
        ],
        "max_tokens": 120,
    },
    {
        "instructions": "You are a concise coding tutor.",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Write a Python function that reverses a string."}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "def reverse_string(s: str) -> str:\n    return s[::-1]"}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Now handle None input gracefully."}]},
        ],
        "max_tokens": 100,
    },
    {
        "instructions": _SYNTHETIC_INSTRUCTIONS,
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What does the `yield` keyword do in Python?"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "The `yield` keyword pauses a function and returns a value to the caller, turning the function into a generator that can be resumed on the next call."}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Give a short example showing iteration over a generator."}]},
        ],
        "max_tokens": 150,
    },
    {
        "instructions": "You are a helpful assistant. Be concise.",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What is the CAP theorem?"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "The CAP theorem states that a distributed system can guarantee only two of three properties simultaneously: Consistency (all nodes see the same data), Availability (every request gets a response), and Partition tolerance (the system continues operating despite network splits)."}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Which pair do most modern databases choose and why?"}]},
        ],
        "max_tokens": 150,
    },
    {
        "instructions": _SYNTHETIC_INSTRUCTIONS,
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What are the main differences between SQL and NoSQL databases?"}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "SQL databases use structured schemas with ACID transactions, suited for relational data. NoSQL databases offer flexible schemas and horizontal scalability, suited for unstructured or large-scale data, but typically trade ACID guarantees for performance."}]},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "When would you choose MongoDB over PostgreSQL?"}]},
        ],
        "max_tokens": 120,
    },
)

_LARGE_INSTRUCTIONS = "You are a document analysis assistant. Answer factually based only on the provided text."

# Large-tier payloads: long documents + analysis (1000-3000 tokens)
_LARGE_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "instructions": _LARGE_INSTRUCTIONS,
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "HTTP has evolved significantly across its versions. HTTP/1.1, standardized in 1997, introduced persistent connections via Keep-Alive, chunked transfer encoding, and virtual hosting through the Host header. However, it suffered from head-of-line blocking where a single slow response would delay all subsequent requests on the same connection. Browsers worked around this by opening multiple parallel connections (typically 6 per domain), but this added overhead.\n\nHTTP/2, released in 2015, addressed these limitations with multiplexing — multiple streams over a single TCP connection — eliminating head-of-line blocking at the HTTP layer. It introduced header compression via HPACK, server push (proactively sending resources before the client requests them), and binary framing instead of text. HTTP/2 significantly improved page load times for assets-heavy sites.\n\nHTTP/3 (2022) moved to QUIC, a UDP-based transport built into the protocol itself. QUIC provides independent stream multiplexing so packet loss on one stream doesn't block others, 0-RTT connection resumption for faster reconnects, and built-in encryption. HTTP/3 is particularly beneficial for mobile users and high-latency networks.\n\nQuestion: List the key performance improvement each HTTP version introduced over its predecessor."}]}],
        "max_tokens": 250,
    },
    {
        "instructions": _LARGE_INSTRUCTIONS,
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "The following is a software architecture document:\n\n## System Overview\nThe Order Processing System (OPS) is a microservices-based platform responsible for handling customer orders from submission through fulfillment. It consists of six core services:\n\n1. **API Gateway** — Entry point for all client requests. Handles authentication, rate limiting, and request routing. Stateless; horizontally scalable.\n2. **Order Service** — Creates and manages order lifecycle (pending → confirmed → shipped → delivered). Persists to PostgreSQL. Emits events to Kafka on state transitions.\n3. **Inventory Service** — Tracks stock levels. Consumes Order Service events to reserve items. Uses Redis for low-latency reads and PostgreSQL for persistence.\n4. **Payment Service** — Integrates with Stripe and PayPal. Synchronous for authorization, async for settlement. Retries with exponential backoff on transient failures.\n5. **Notification Service** — Sends emails and SMS on order events. Consumes from Kafka. At-least-once delivery with idempotency keys.\n6. **Fulfillment Service** — Picks, packs, and ships. Integrates with three 3PL warehouse APIs. Handles partial fulfillment when items are split across warehouses.\n\n## Known Failure Modes\n- Kafka lag spikes during peak hours can delay notifications by up to 10 minutes.\n- Inventory double-reservation bug when two orders for the last item arrive within 50ms of each other (race condition in Inventory Service).\n- Payment Service retries can create duplicate charges when Stripe returns ambiguous responses.\n- Fulfillment Service timeout (30s) on slow 3PL API responses causes order to enter an unknown state.\n\nQuestion: What are the main failure modes described in this document?"}]}],
        "max_tokens": 300,
    },
    {
        "instructions": _LARGE_INSTRUCTIONS,
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Distributed consensus is the problem of getting multiple nodes to agree on a single value despite failures. The following describes three common approaches:\n\n**Paxos** (Lamport, 1989) is the foundational algorithm. It operates in two phases: Prepare (a proposer asks acceptors to promise not to accept earlier proposals) and Accept (the proposer sends a value, acceptors commit if they haven't promised to a higher proposal number). Paxos guarantees safety (no two nodes decide differently) but is notoriously difficult to implement correctly. Multi-Paxos extends it for a sequence of decisions, forming the basis for many production systems.\n\n**Raft** (Ongaro & Ousterhout, 2014) was designed to be more understandable than Paxos. It decomposes consensus into leader election, log replication, and safety. A leader is elected per term; all writes go through the leader, which replicates them to followers before committing. Raft guarantees that elected leaders always have the most up-to-date log. It's used in etcd, CockroachDB, and TiKV.\n\n**Viewstamped Replication (VR)** (Liskov, 1988) predates Paxos and uses a primary/backup model with view changes for leader election. It's less widely known but influenced many modern designs.\n\nAll three require a quorum (majority) of nodes to be available. With N nodes, they tolerate ⌊(N-1)/2⌋ failures.\n\nQuestion: What is the key design goal that distinguishes Raft from Paxos, and which production systems use Raft?"}]}],
        "max_tokens": 200,
    },
)


def synthetic_body() -> dict[str, Any]:
    """Build a Responses-API request body for one synthetic call.

    Both `instructions` (non-empty) and `store: False` are required by the
    Codex Responses API — omitting either yields a 400 from upstream.
    """
    prompt = random.choice(_PROMPT_CORPUS)
    return {
        "model": "auto-learning-synthetic",
        "instructions": _SYNTHETIC_INSTRUCTIONS,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "stream": False,
        "store": False,
    }


def _aggressive_synthetic_body() -> dict[str, Any]:
    """Build a diverse synthetic payload for aggressive exhaustion mode.

    Cycles through small/medium/large tiers for dataset variety.
    All tiers include required `store: False` and non-empty `instructions`.
    """
    tier = random.choice(("small", "medium", "large"))
    if tier == "small":
        return synthetic_body()
    template = random.choice(_MEDIUM_TEMPLATES if tier == "medium" else _LARGE_TEMPLATES)
    return {
        **template,
        "model": "auto-learning-synthetic",
        "stream": False,
        "store": False,
    }


def _should_enter_aggressive(snap: Any, cfg: AutoRouterConfig) -> bool:
    """Return True when a backend's weekly quota is high enough to trigger
    aggressive exhaustion mode (bypassing normal pacing accumulator).
    """
    if snap is None:
        return False
    if snap.weekly_used_percent is None:
        return False
    return snap.weekly_used_percent >= cfg.aggressive_exhaustion_pct


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


def weekly_exhaustion_fire_rate(
    quota: CodexQuotaSnapshot,
    *,
    organic_rate_pct_per_hour: float,
    now_ts: float,
    cfg: AutoRouterConfig,
) -> float:
    """Raw fractional rate of synthetics to fire on a backend this tick to
    keep its weekly window on track to land at 100% by reset.

    Returns 0.0 when:
      - quota snapshot is incomplete (no weekly_used_percent or weekly_reset_at)
      - account is past `weekly_target_pct` (close enough; stop)
      - account is past `five_hourly_pause_pct` (rate-limited; pause)
      - reset is imminent or in the past (let it cycle)
      - projected human burn alone will exhaust the weekly window

    The worker uses this float rate via a leaky-bucket accumulator so a
    "0.28 synthetics per tick" target accumulates and fires correctly over
    many ticks instead of rounding to 0 every time.
    """
    if quota.weekly_used_percent is None or quota.weekly_reset_at is None:
        return 0.0
    if quota.weekly_used_percent >= cfg.weekly_target_pct:
        return 0.0
    if (
        quota.five_hourly_used_percent is not None
        and quota.five_hourly_used_percent >= cfg.five_hourly_pause_pct
    ):
        return 0.0
    hours_remaining = max(0.0, (quota.weekly_reset_at - now_ts) / 3600.0)
    if hours_remaining < 0.1:
        return 0.0
    weekly_remaining_pct = float(cfg.weekly_target_pct) - float(quota.weekly_used_percent)
    if weekly_remaining_pct <= 0:
        return 0.0
    projected_organic_pct = (
        organic_rate_pct_per_hour * hours_remaining * cfg.prediction_safety_margin
    )
    burnable_pct = max(0.0, weekly_remaining_pct - projected_organic_pct)
    if burnable_pct <= 0:
        return 0.0
    pct_per_call = max(1e-6, cfg.pct_per_synthetic_estimate)
    synthetics_total = burnable_pct / pct_per_call
    tick_hours = max(1, cfg.synthetic_check_interval_seconds) / 3600.0
    return synthetics_total * tick_hours / hours_remaining


def weekly_exhaustion_fire_count(
    quota: CodexQuotaSnapshot,
    *,
    organic_rate_pct_per_hour: float,
    now_ts: float,
    cfg: AutoRouterConfig,
) -> int:
    """Integer-rounded count for tests and one-shot callers. The worker uses
    `weekly_exhaustion_fire_rate` directly so it can pace via accumulator.
    """
    rate = weekly_exhaustion_fire_rate(
        quota,
        organic_rate_pct_per_hour=organic_rate_pct_per_hour,
        now_ts=now_ts,
        cfg=cfg,
    )
    n = int(round(rate))
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
        cost_router: LearnedModelRouter | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._cfg = cfg
        self._usage_log_path = usage_log_path
        self._backends = list(backends)
        self._dispatch = dispatch
        self._cost_router = cost_router
        self._clock = clock if clock is not None else _wall_clock
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        # Leaky-bucket accumulator per backend: each tick adds the desired
        # fractional rate; whenever the accumulator crosses 1.0, fire that
        # many whole synthetics and decrement. This makes a target like
        # "0.28 synthetics/tick" actually fire 1 synthetic every ~3.5 ticks
        # instead of rounding to 0 every tick forever.
        self._fire_accumulator: dict[str, float] = {}
        # Aggressive exhaustion mode state.
        self._aggressive_consecutive_429s: dict[str, int] = {}
        self._in_aggressive_mode: dict[str, bool] = {}

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
            snap = await backend.quota_snapshot()
            if _should_enter_aggressive(snap, self._cfg):
                await self._aggressive_exhaust_backend(backend, snap, now_ts=now_ts)
            else:
                # Normal pacing — unchanged leaky-bucket accumulator path.
                self._in_aggressive_mode[backend.id] = False
                n = await self._fire_count_for_backend(backend, now_ts=now_ts)
                for _ in range(n):
                    try:
                        await self._dispatch(synthetic_body(), backend.id)
                    except Exception:
                        logger.exception("synthetic dispatch failed for %s", backend.id)
                        break  # don't tight-loop on a sick backend
        # Periodically refit the cost_router cost model
        if (self._cost_router is not None
            and self._cfg.exploiter_refit_interval_seconds > 0
            and (now_ts - self._cost_router._last_fit) >= self._cfg.exploiter_refit_interval_seconds):
            self._cost_router.fit(min_samples_per_cell=self._cfg.exploiter_min_samples_per_cell)

    async def _fire_count_for_backend(self, backend: Backend, *, now_ts: float) -> int:
        snap: Any = await backend.quota_snapshot()
        if snap is not None and self._usage_log_path is not None:
            organic_rate = organic_burn_rate_pct_per_hour(
                self._usage_log_path,
                backend.id,
                now_ts=now_ts,
                window_hours=float(self._cfg.prediction_window_hours),
            )
            # Leaky-bucket pacing: add this tick's fractional rate to the
            # accumulator, fire whatever whole synthetics have accumulated.
            # This is the fix for the "0.28 → round() → 0 forever" bug that
            # used to silently disable the controller in low-rate scenarios.
            rate = weekly_exhaustion_fire_rate(
                snap,
                organic_rate_pct_per_hour=organic_rate,
                now_ts=now_ts,
                cfg=self._cfg,
            )
            self._fire_accumulator[backend.id] = self._fire_accumulator.get(backend.id, 0.0) + rate
            n_whole = int(self._fire_accumulator[backend.id])
            if n_whole > 0:
                # Cap at max_per_tick to avoid bursting; remaining stays in
                # the accumulator and fires next tick.
                fired = min(n_whole, self._cfg.max_synthetics_per_tick)
                self._fire_accumulator[backend.id] -= fired
                return fired
            return 0
        if self._usage_log_path is None:
            return 0
        counts = daily_counts(self._usage_log_path, now_ts=now_ts, backend_id=backend.id)
        return cold_start_fire_count(counts, self._cfg)

    async def _aggressive_exhaust_backend(
        self, backend: Backend, snap: Any, *, now_ts: float
    ) -> None:
        """Fire diverse payloads in a tight loop until confirmed exhaustion."""
        if not self._in_aggressive_mode.get(backend.id, False):
            logger.warning(
                "backend %s: entering aggressive exhaustion mode (weekly=%s%%)",
                backend.id,
                snap.weekly_used_percent,
            )
            self._in_aggressive_mode[backend.id] = True
            self._aggressive_consecutive_429s[backend.id] = 0

        consecutive = self._aggressive_consecutive_429s.get(backend.id, 0)
        confirm_threshold = self._cfg.aggressive_exhaustion_consecutive_429s
        max_burst = self._cfg.aggressive_exhaustion_max_per_burst

        for _ in range(max_burst):
            try:
                await self._dispatch(_aggressive_synthetic_body(), backend.id)
                consecutive = 0
            except Exception as exc:
                is_429 = (isinstance(exc, HTTPException) and exc.status_code == 429) or (
                    isinstance(exc, BackendError)
                    and exc.classification == "rate_limited"
                )
                if is_429:
                    consecutive += 1
                    logger.info(
                        "backend %s: aggressive 429 #%d (confirm at %d)",
                        backend.id,
                        consecutive,
                        confirm_threshold,
                    )
                    if consecutive >= confirm_threshold:
                        await self._handle_confirmed_exhaustion(
                            backend, now_ts=now_ts
                        )
                        self._aggressive_consecutive_429s[backend.id] = consecutive
                        return
                else:
                    logger.warning(
                        "backend %s: aggressive mode non-429 error (%s) — falling back",
                        backend.id,
                        type(exc).__name__,
                    )
                    self._in_aggressive_mode[backend.id] = False
                    self._aggressive_consecutive_429s[backend.id] = 0
                    return

        self._aggressive_consecutive_429s[backend.id] = consecutive

    async def _handle_confirmed_exhaustion(
        self, backend: Backend, *, now_ts: float
    ) -> None:
        """After N consecutive 429s, distinguish weekly-limit vs 5h exhaustion."""
        post_snap = await backend.quota_snapshot()
        five_h = (
            post_snap.five_hourly_used_percent if post_snap is not None else None
        )
        weekly = post_snap.weekly_used_percent if post_snap is not None else None

        if five_h is not None and five_h >= self._cfg.five_hourly_pause_pct:
            logger.warning(
                "backend %s: 5h window exhausted (%d%%) — pausing aggressive mode",
                backend.id,
                five_h,
            )
        else:
            logger.warning(
                "backend %s: WEEKLY QUOTA CONFIRMED EXHAUSTED "
                "(weekly=%s%%, 5h=%s%%)",
                backend.id,
                weekly,
                five_h,
            )
        self._in_aggressive_mode[backend.id] = False


def _wall_clock() -> float:
    import time

    return time.time()
