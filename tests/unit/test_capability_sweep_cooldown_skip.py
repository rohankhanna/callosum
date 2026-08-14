"""Hermetic regression: the capability harness sweep must skip cells whose
backend is in cooldown or weekly-exhausted (merge-blocking). Pins fix
df2ebd3 — "fix(capability): skip cells whose backend is in cooldown
during harness sweep".

The original live-use bug (observed 2026-06-09): the post-restart harness
sweep iterated EVERY advertised local cell and called backend.responses()
on each. When upstream lanes were still keep-alive-cold (ports 8091/8093),
the sweep's probes piled into those fragile lanes, prolonging cold-load and
producing cascading transient errors that blocked B3.2 verification for >1h.
The sweep was contributing inference load to the very lanes that had signaled
they could not serve.

The fix (capability/scheduler.py) introduced _backend_routable_now —
a per-backend predicate that returns False when usage_snapshot() reports
weekly_exhausted=True OR cooldown_until_ts in the future — and wired
it into run_harness_sweep so cooldown backends are skipped for the whole
sweep tick. It falls OPEN (returns True) on a missing usage_snapshot
interface or a read error, so older/test backends don't silently disable the
harness.

This regression has two layers:

1. **Sweep-level (the faithful merge-blocking signal)** —
   test_sweep_skips_cooldown_backend_but_runs_healthy: a healthy backend
   and a cooldown backend are both eligible (litellm_gateway kind,
   advertised models). run_dimensions is stubbed to a recorder. The sweep
   must call run_dimensions for the healthy backend and NOT for the
   cooldown backend. Reverting the skip (drop the backend_routable cache +
   continue) → run_dimensions is called for BOTH → the assertion that
   the cooldown backend is absent from the recorded calls fails.

2. **Predicate-level** — direct tests of _backend_routable_now pinning
   every branch: weekly_exhausted → False; future cooldown → False; past
   cooldown → True; healthy → True; missing usage_snapshot → True (open);
   usage_snapshot raising → True (open). These isolate the predicate so a
   future refactor that breaks one branch turns the matching test red even if
   the sweep wiring stays correct.

Hermetic: run_dimensions is stubbed (no real dimension probes); the fake
backends are in-process (no network). Runs in the Tier 1 gate's unit suite as
a merge-blocking regression.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import pytest

from callosum.backend import UsageSnapshot
from callosum.capability import scheduler
from callosum.capability.scheduler import _backend_routable_now, run_harness_sweep


@dataclass(slots=True)
class _FakeBackend:
    """Minimal local backend for the sweep.

    kind="litellm_gateway" is the only kind _cells_to_harness admits
    (_LOCAL_BACKEND_KINDS). usage_snapshot is an async method so
    _backend_routable_now can await it; snapshot=None +
    snapshot_raises=True models a backend whose snapshot read fails (the
    predicate must fall open). The missing-interface branch uses
    _NoSnapshotBackend (no usage_snapshot attr at all).
    """

    id: str
    advertised_models: frozenset[str]
    snapshot: UsageSnapshot | None
    kind: str = "litellm_gateway"
    snapshot_raises: bool = False
    responses_called: int = 0

    async def usage_snapshot(self) -> UsageSnapshot:
        if self.snapshot_raises:
            raise RuntimeError("simulated snapshot read failure")
        if self.snapshot is None:
            # Matches an older/test backend with no snapshot interface — but
            # because the attr IS present here, we model "no snapshot" via the
            # _NoSnapshotBackend variant instead. This branch is defensive.
            raise AttributeError("no snapshot")
        return self.snapshot

    async def responses(self, body: dict[str, Any]) -> dict[str, Any]:
        self.responses_called += 1
        return {"output": []}


class _NoSnapshotBackend:
    """A local backend with NO usage_snapshot attribute — the predicate's
    fall-open-on-missing-interface branch."""

    def __init__(self, id: str) -> None:
        self.id = id
        self.kind = "litellm_gateway"
        self.advertised_models = frozenset({"no-snap-cell"})
        self.responses_called = 0

    async def responses(self, body: dict[str, Any]) -> dict[str, Any]:
        self.responses_called += 1
        return {"output": []}


def _stub_dimensions(monkeypatch, recorder: list[str]) -> None:
    """Replace scheduler.run_dimensions with a recorder that captures the
    backend_id of each cell it was asked to probe and returns no fresh
    findings. The sweep only cares that the call happened (or didn't)."""

    async def _record(
        *,
        cell: str,
        backend_id: str | None,
        call_responses: Any,
        ttl_s: float,
        now: float | None = None,
        weight_identity_provider: Any = None,
    ) -> dict[str, Any]:
        recorder.append(backend_id or "?")
        return {}

    monkeypatch.setattr(scheduler, "run_dimensions", _record)


# --------------------------------------------------------------------------- #
# 1. Sweep-level — the faithful merge-blocking signal.                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sweep_skips_cooldown_backend_but_runs_healthy(monkeypatch) -> None:
    """run_harness_sweep must NOT probe a backend whose
    usage_snapshot reports a future cooldown, while still probing a
    healthy backend in the same sweep.

    Reverting the fix (removing the backend_routable cache + the
    if not backend_routable.get(...): continue guard at
    scheduler.py ~L183-186) makes the sweep iterate every eligible cell
    → run_dimensions is called for BOTH backends → the cooldown backend
    appears in probed → the assertion fails.
    """
    healthy = _FakeBackend(
        id="healthy",
        advertised_models=frozenset({"healthy-cell"}),
        snapshot=UsageSnapshot(
            remaining_fraction=1.0,
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=time.time(),
        ),
    )
    cooldown = _FakeBackend(
        id="cooldown",
        advertised_models=frozenset({"cooldown-cell"}),
        snapshot=UsageSnapshot(
            remaining_fraction=0.2,
            cooldown_until_ts=time.time() + 3600.0,  # future → in cooldown
            weekly_exhausted=False,
            probed_at_ts=time.time(),
        ),
    )

    probed: list[str] = []
    _stub_dimensions(monkeypatch, probed)

    ran = await run_harness_sweep(backends=[healthy, cooldown], ttl_s=60.0)

    # The healthy backend's cell was probed.
    assert "healthy" in probed, (
        f"healthy backend was not probed (probed={probed!r}) — the cooldown "
        f"skip is over-skipping eligible backends"
    )
    # The cooldown backend's cell was NOT probed — the core regression.
    assert "cooldown" not in probed, (
        f"cooldown backend was probed (probed={probed!r}) — the sweep is "
        f"hammering a backend that signaled it cannot serve (df2ebd3 regressed)"
    )
    # run_dimensions returns {} → zero fresh dimensions recorded.
    assert ran == 0


@pytest.mark.asyncio
async def test_sweep_skips_weekly_exhausted_backend(monkeypatch) -> None:
    """A backend reporting weekly_exhausted=True must be skipped just like
    a cooldown backend — the predicate treats both as not-routable-now."""
    exhausted = _FakeBackend(
        id="exhausted",
        advertised_models=frozenset({"exhausted-cell"}),
        snapshot=UsageSnapshot(
            remaining_fraction=0.0,
            cooldown_until_ts=None,
            weekly_exhausted=True,
            probed_at_ts=time.time(),
        ),
    )
    probed: list[str] = []
    _stub_dimensions(monkeypatch, probed)
    await run_harness_sweep(backends=[exhausted], ttl_s=60.0)
    assert probed == [], (
        f"weekly-exhausted backend was probed (probed={probed!r}) — "
        f"df2ebd3 regressed for the weekly_exhausted branch"
    )


# --------------------------------------------------------------------------- #
# 2. Predicate-level — _backend_routable_now branch isolation.                #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_routable_now_future_cooldown_is_not_routable() -> None:
    backend = _FakeBackend(
        id="b",
        advertised_models=frozenset({"c"}),
        snapshot=UsageSnapshot(
            remaining_fraction=0.2,
            cooldown_until_ts=time.time() + 3600.0,
            weekly_exhausted=False,
            probed_at_ts=time.time(),
        ),
    )
    assert await _backend_routable_now(backend) is False


@pytest.mark.asyncio
async def test_routable_now_past_cooldown_is_routable() -> None:
    """A cooldown that has already expired must NOT block the sweep — the
    cell is eligible again. Pins the cooldown_until > time.time() guard
    (a reverted >= or removed expiry check would over-skip)."""
    backend = _FakeBackend(
        id="b",
        advertised_models=frozenset({"c"}),
        snapshot=UsageSnapshot(
            remaining_fraction=0.5,
            cooldown_until_ts=time.time() - 3600.0,  # past → expired
            weekly_exhausted=False,
            probed_at_ts=time.time(),
        ),
    )
    assert await _backend_routable_now(backend) is True


@pytest.mark.asyncio
async def test_routable_now_weekly_exhausted_is_not_routable() -> None:
    backend = _FakeBackend(
        id="b",
        advertised_models=frozenset({"c"}),
        snapshot=UsageSnapshot(
            remaining_fraction=0.0,
            cooldown_until_ts=None,
            weekly_exhausted=True,
            probed_at_ts=time.time(),
        ),
    )
    assert await _backend_routable_now(backend) is False


@pytest.mark.asyncio
async def test_routable_now_healthy_is_routable() -> None:
    backend = _FakeBackend(
        id="b",
        advertised_models=frozenset({"c"}),
        snapshot=UsageSnapshot(
            remaining_fraction=1.0,
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=time.time(),
        ),
    )
    assert await _backend_routable_now(backend) is True


@pytest.mark.asyncio
async def test_routable_now_missing_snapshot_interface_falls_open() -> None:
    """A backend with no usage_snapshot attribute (older/test backend)
    must fall OPEN (routable) — the harness must not be silently disabled by
    a missing snapshot interface."""
    backend = _NoSnapshotBackend(id="no-snap")
    assert await _backend_routable_now(backend) is True


@pytest.mark.asyncio
async def test_routable_now_snapshot_read_error_falls_open() -> None:
    """A usage_snapshot() that raises must fall OPEN (routable) so a
    transient snapshot read failure doesn't silently disable the harness."""
    backend = _FakeBackend(
        id="b",
        advertised_models=frozenset({"c"}),
        snapshot=None,
        snapshot_raises=True,
    )
    assert await _backend_routable_now(backend) is True