"""Small TTL memoization helpers.

Some callosum observability surfaces — notably the /status router
sub-reports — aggregate over the multi-GB requests DB on every call. An
external status poller hits /status every ~30s, so a
per-call recompute drives a recurring multi-core burst. These aggregates
drift on traffic timescales but are observability-only, so a short TTL
memo is safe: the first reader after the TTL recomputes, the rest reuse
the last snapshot. TtlCache is the mechanism.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any


class TtlCache:
    """Memoize a single value with a wall-clock TTL.

    get returns the cached value while fresh, else None — the caller
    computes the value and calls set. Owning the compute in the caller
    means a stale read never blocks on a slow producer and a cold start is
    explicit (a None return means "you compute"). Thread-safe via an
    internal lock; clock is injectable for deterministic tests.
    """

    __slots__ = ("_ttl", "_clock", "_lock", "_value", "_ready_at", "_has")

    def __init__(self, ttl_s: float, *, clock: Callable[[], float] = time.time) -> None:
        self._ttl = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._value: Any = None
        self._ready_at = 0.0
        self._has = False

    def get(self) -> Any | None:
        with self._lock:
            if self._has and self._clock() < self._ready_at:
                return self._value
            return None

    def set(self, value: Any) -> None:
        with self._lock:
            self._value = value
            self._ready_at = self._clock() + self._ttl
            self._has = True

    def invalidate(self) -> None:
        with self._lock:
            self._value = None
            self._has = False
            self._ready_at = 0.0

    @property
    def ttl_s(self) -> float:
        return self._ttl
