"""Background scheduler for the thorough capability harness.

Closes the loop: when callosum starts (or when
downloads a new model and ollama picks it up on the next catalog
refresh), the registered cells are walked, and any cell without a
fresh capability profile gets probed across every registered
dimension. Findings persist to disk as JSON, where downstream
consumers (adapter writers, the router itself, operator tooling)
can read them without re-running the harness.

This scheduler is the THOROUGH counterpart to
`routing.probe_scheduler.schedule_background_sweep`:

  * Light probe (routing/probe_scheduler.py): seconds per cell,
    binary supports_tools answer, runs on every startup and gates
    real-time routing decisions.
  * Thorough harness (this module): minutes per cell, multi-
    dimensional findings + adapter hints, runs in background and
    only re-runs when the per-dimension cache TTL expires (default
    one week — see `runner.DEFAULT_DIMENSION_TTL_S`).

Both schedulers can coexist; they probe different things at
different cadences. The light probe is what the router consults
synchronously; the thorough harness produces the artifacts that
adapter authors and  reason about.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from typing import TYPE_CHECKING, Any

from callosum.capability.runner import DEFAULT_DIMENSION_TTL_S, run_dimensions
from callosum.capability.weight_identity import WeightIdentityProvider

if TYPE_CHECKING:
    from callosum.operator_state import OperatorState


logger = logging.getLogger(__name__)


# How often the periodic harness sweeper wakes up to re-check the cell
# grid. 6 hours by default: short enough that a model
# pulls in the middle of the night is probed within hours (not days),
# long enough that re-checks don't spam logs when nothing has changed.
# Steady-state work per sweep is near-zero because the runner's per-
# dimension TTL (1 week) short-circuits cells whose every dimension is
# still fresh.
#
# Override via env (CALLOSUM_HARNESS_SWEEP_INTERVAL_SECONDS) for testing
# or when an operator wants more aggressive re-probing — for example
# right after teaching the harness a new dimension that they want
# back-filled across all cells faster than the default cadence.
def _default_sweep_interval_s() -> float:
    raw = os.environ.get("CALLOSUM_HARNESS_SWEEP_INTERVAL_SECONDS")
    if raw is None:
        return 6 * 3600.0
    try:
        v = float(raw)
    except ValueError:
        logger.warning("ignoring non-numeric CALLOSUM_HARNESS_SWEEP_INTERVAL_SECONDS=%r", raw)
        return 6 * 3600.0
    # Floor to 60s so a typo doesn't pin a GPU forever. No upper cap —
    # operators may genuinely want infrequent sweeps if cell churn is
    # rare in their environment.
    return max(60.0, v)


# Mirror probe_scheduler._LOCAL_BACKEND_KINDS so both schedulers
# agree on which cells warrant probing. Remote (codex_auth_vault)
# cells have curated capability claims and don't need a harness.
_LOCAL_BACKEND_KINDS = frozenset({"litellm_gateway"})


def _cells_to_harness(backends: list[Any]) -> list[tuple[Any, str]]:
    """Return (backend, cell) pairs the harness should consider.

    Deduplicated by (backend_id, cell) so cells advertised by two
    backend instances only get harnessed via one. Whether a given
    cell actually runs is then governed by `run_dimensions`'s TTL
    gate per dimension — we don't second-guess that here.
    """
    seen: set[tuple[str, str]] = set()
    out: list[tuple[Any, str]] = []
    for backend in backends:
        if getattr(backend, "kind", "") not in _LOCAL_BACKEND_KINDS:
            continue
        if not hasattr(backend, "responses"):
            continue
        backend_id = getattr(backend, "id", "?")
        advertised: frozenset[str] = getattr(backend, "advertised_models", frozenset())
        for cell in sorted(advertised):
            key = (backend_id, cell)
            if key in seen:
                continue
            seen.add(key)
            out.append((backend, cell))
    return out


async def _backend_routable_now(backend: Any) -> bool:
    """Mirror of app._routable_backends' per-backend check, scoped to the
    harness's needs: return False when the backend is currently in
    cooldown or weekly-exhausted, so we don't pile probes onto an
    upstream that's already signaled it can't serve. Falls open
    (returns True) on any read failure so a missing usage_snapshot
    doesn't silently disable the harness."""
    snapshot_fn = getattr(backend, "usage_snapshot", None)
    if snapshot_fn is None:
        return True
    try:
        snap = await snapshot_fn()
    except Exception:
        return True
    if getattr(snap, "weekly_exhausted", False):
        return False
    cooldown_until = getattr(snap, "cooldown_until_ts", None)
    return cooldown_until is None or cooldown_until <= time.time()


async def run_harness_sweep(
    *,
    backends: list[Any],
    ttl_s: float = DEFAULT_DIMENSION_TTL_S,
    weight_identity_provider: WeightIdentityProvider | None = None,
) -> int:
    """Run the thorough harness across every local cell. Returns the
    total number of dimensions actually executed (those whose
    cached result was either missing, expired, or non-deterministic
    last time). Cells whose every dimension was already fresh in
    the TTL window contribute zero.

    Harnesses run serially across cells to avoid GPU contention —
    two 31B-class harnesses in parallel would exhaust memory on most hosts.

    Backends in cooldown are skipped on a per-sweep basis: when an
    upstream lane is cold-loading or in transient failure, hammering
    it with probes during the moment it's most fragile contributes
    to the very degradation the harness is supposed to characterize.
    The sweep will pick those cells back up on its next tick once the
    backend stops reporting cooldown. (Observed 2026-06-09: a startup
    sweep against unwarmed responses-proxy lanes contributed to a
    1h+ block on B3.2 verification.)

    `weight_identity_provider` is forwarded to `run_dimensions` so each
    cell's profile gets stamped with stable identity for its underlying
    weights. None disables stamping (older callers, tests).
    """
    todo = _cells_to_harness(backends)
    if not todo:
        return 0
    # Cache per-backend routability so we don't re-snapshot for every
    # cell on the same backend.
    backend_routable: dict[str, bool] = {}
    for backend, _cell in todo:
        bid = getattr(backend, "id", "?")
        if bid in backend_routable:
            continue
        backend_routable[bid] = await _backend_routable_now(backend)
    skipped_cooldown = sum(1 for backend, _ in todo if not backend_routable.get(getattr(backend, "id", "?"), True))
    if skipped_cooldown:
        logger.info("capability harness sweep: %d cell(s) skipped — backend in cooldown", skipped_cooldown)
    logger.info(
        "capability harness sweep: %d cell(s) eligible (per-dim TTL=%.0fs)", len(todo) - skipped_cooldown, ttl_s
    )
    total_dimensions_run = 0
    for backend, cell in todo:
        backend_id = getattr(backend, "id", "?")
        if not backend_routable.get(backend_id, True):
            continue

        async def _call(body: dict[str, Any], _backend: Any = backend) -> dict[str, Any]:
            result: dict[str, Any] = await _backend.responses(body)
            return result

        try:
            fresh = await run_dimensions(
                cell=cell,
                backend_id=backend_id,
                call_responses=_call,
                ttl_s=ttl_s,
                weight_identity_provider=weight_identity_provider,
            )
            total_dimensions_run += len(fresh)
        except Exception:
            # run_dimensions itself is defensive; this is paranoia in
            # case orchestration around it ever raises. One cell's
            # failure must not block the rest of the sweep.
            logger.exception("capability harness: sweep entry %s/%s raised", backend_id, cell)
    logger.info("capability harness sweep: complete (%d dimension run(s) total)", total_dimensions_run)
    return total_dimensions_run


def schedule_background_harness(
    *,
    backends: list[Any],
    operator_state: OperatorState | None = None,
    ttl_s: float = DEFAULT_DIMENSION_TTL_S,
    weight_identity_provider: WeightIdentityProvider | None = None,
) -> asyncio.Task[int]:
    """Fire-and-forget version suitable for callosum's lifespan
    startup. Returns the asyncio Task so callers can join it during
    shutdown if they want; for the typical kick-and-forget pattern,
    the return can be ignored.

    `operator_state` is currently unused — the harness persists
    findings to JSON files in logs/capability_profiles/, not to
    SQLite. The parameter is accepted to keep the call-site
    symmetric with `schedule_background_sweep` so adding harness
    metadata into operator_state later is a one-line change.

    `weight_identity_provider`, when supplied, is forwarded to
    `run_harness_sweep` so profiles get stamped with the underlying-
    weights identity (see capability/weight_identity.py).

    Exception swallowing inside `_runner` is the same pattern as
    the light-probe scheduler: probe sweeps must NEVER crash startup
    and must NEVER block request serving.
    """
    _ = operator_state  # reserved for future use; see docstring

    async def _runner() -> int:
        try:
            return await run_harness_sweep(
                backends=backends, ttl_s=ttl_s, weight_identity_provider=weight_identity_provider
            )
        except Exception:
            logger.exception("capability harness sweep: unexpected failure")
            return 0

    return asyncio.create_task(_runner())


class PeriodicHarnessSweep:
    """Run the capability harness on a recurring cadence.

    Closes the "callosum in the loop on a regular basis" requirement:
    fires the harness once on startup, then re-fires every
    `interval_s` seconds for the lifetime of the proxy. Steady-state
    cost is near zero because `run_dimensions` short-circuits cells
    whose every dimension is still within the per-dimension TTL — so
    the work that actually happens each tick is:

      * cells advertised since the previous tick (new arrivals from
         between sweeps), AND
      * cells whose cached findings have aged past the TTL.

    The existing `_PeriodicSmokeTester` in app.py refreshes each
    backend's `advertised_models` hourly, so newly-pulled local models
    appear in the cell grid without needing a callosum restart. This
    sweeper then picks them up on its next tick.

    Lifecycle mirrors `_PeriodicSmokeTester` and `_PeriodicCooldownProber`
    in app.py: start() spawns the loop task; stop() signals it to exit
    and awaits the task. The class is intentionally not an
    asynccontextmanager so callers can drive start/stop from FastAPI's
    lifespan generator without nesting.
    """

    def __init__(
        self,
        *,
        backends: list[Any],
        interval_s: float | None = None,
        ttl_s: float = DEFAULT_DIMENSION_TTL_S,
        operator_state: OperatorState | None = None,
        weight_identity_provider: WeightIdentityProvider | None = None,
    ) -> None:
        self._backends = backends
        self._interval_s = interval_s if interval_s is not None else _default_sweep_interval_s()
        self._ttl_s = ttl_s
        self._operator_state = operator_state
        self._weight_identity_provider = weight_identity_provider
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        """Disabled when no backends OR a non-positive interval was set
        (operator can set CALLOSUM_HARNESS_SWEEP_INTERVAL_SECONDS=0 to
        turn the periodic sweeper off entirely while keeping the
        one-shot startup sweep). Floor-of-60 in _default_sweep_interval_s
        only applies to env-provided values; programmatic callers
        passing 0 explicitly are honored."""
        return self._interval_s > 0 and bool(self._backends)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="periodic-harness-sweep")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        """Sleep-then-sweep loop. Sleep first because the one-shot
        startup task fired by `schedule_background_harness` already
        ran the initial pass — re-running it back-to-back here would
        just discover everything still TTL-fresh."""
        logger.info(
            "periodic harness sweep: interval=%.0fs (one-shot startup pass "
            "already kicked off; next periodic tick in %.0fs)",
            self._interval_s,
            self._interval_s,
        )
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval_s)
            except TimeoutError:
                pass
            else:
                # stop() called during the sleep — exit cleanly.
                return
            try:
                await run_harness_sweep(
                    backends=self._backends, ttl_s=self._ttl_s, weight_identity_provider=self._weight_identity_provider
                )
            except Exception:
                logger.exception("periodic harness sweep: tick failed")
