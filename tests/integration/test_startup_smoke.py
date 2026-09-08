from __future__ import annotations

import logging
from typing import cast

import pytest
from fastapi.testclient import TestClient

from callosum.app import create_app
from callosum.backend import HealthStatus, UsageSnapshot
from callosum.fakes import InMemoryFakeBackend


def _backend(
    *,
    id: str,
    cooldown_until_ts: float | None = None,
    healthy: bool = True,
) -> InMemoryFakeBackend:
    return InMemoryFakeBackend(
        id=id,
        advertised_models=frozenset({"model-a0e7"}),
        health=HealthStatus(available=healthy, reason="ok" if healthy else "down"),
        usage=UsageSnapshot(
            remaining_fraction=0.5,
            cooldown_until_ts=cooldown_until_ts,
            weekly_exhausted=False,
            probed_at_ts=0.0,
        ),
    )


def test_startup_smoke_test_runs_on_each_backend(caplog: pytest.LogCaptureFixture) -> None:
    """When startup_smoke_test=True, the proxy logs one line per backend
    showing OK/SKIPPED/FAILED so the operator sees auth health immediately
    on launch instead of via the first failed user request.
    """
    caplog.set_level(logging.INFO, logger="callosum.startup")
    a = _backend(id="alpha")
    b = _backend(id="beta", cooldown_until_ts=1e12)  # cooldown far in the future
    app = create_app(backends=[a, b], startup_smoke_test=True)

    with TestClient(app):
        pass  # entering the context fires lifespan startup

    msgs = [r.getMessage() for r in caplog.records if r.name == "callosum.startup"]
    text = "\n".join(msgs)
    assert "startup smoke test: probing 2 backend(s)" in text
    # alpha should be probed; beta should be skipped (cooldown).
    assert "[alpha]" in text
    assert "[beta] SKIPPED" in text


def test_startup_smoke_test_can_be_disabled(caplog: pytest.LogCaptureFixture) -> None:
    """When the smoke test is disabled, no smoke-test lines appear (the
    'loaded N backend(s)' roster warning is unrelated and may still fire).
    """
    caplog.set_level(logging.INFO, logger="callosum.startup")
    backend = _backend(id="alpha")
    app = create_app(backends=[backend], startup_smoke_test=False)

    with TestClient(app):
        pass

    msgs = [r.getMessage() for r in caplog.records if r.name == "callosum.startup"]
    assert not any("startup smoke test" in m or "[alpha]" in m for m in msgs), (
        f"expected no smoke-test lines when disabled, got: {msgs}"
    )


def test_startup_smoke_test_no_backends_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    """With zero backends, the smoke test loop must not fire (the roster
    warning may still log "loaded 0 backend(s): (none)" — that's intentional
    and orthogonal).
    """
    caplog.set_level(logging.INFO, logger="callosum.startup")
    app = create_app(backends=[], startup_smoke_test=True)

    with TestClient(app):
        pass

    msgs = cast(
        list[str],
        [r.getMessage() for r in caplog.records if r.name == "callosum.startup"],
    )
    assert not any("startup smoke test" in m for m in msgs), (
        f"empty backend list should not trigger smoke test, got: {msgs}"
    )


# --------- periodic re-runs --------------------------------------------------


import asyncio  # noqa: E402

from callosum.app import _PeriodicSmokeTester  # noqa: E402


async def test_periodic_smoke_tester_fires_after_interval(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Wait for one tick, confirm a 'periodic smoke test cycle' message lands
    in the log, then stop cleanly.
    """
    caplog.set_level(logging.INFO, logger="callosum.startup")
    tester = _PeriodicSmokeTester(backends=[_backend(id="alpha")], interval_s=1)
    tester.start()
    try:
        # Wait long enough for at least one tick to fire and log.
        for _ in range(40):
            await asyncio.sleep(0.1)
            if any(
                "periodic smoke test cycle" in r.getMessage() for r in caplog.records if r.name == "callosum.startup"
            ):
                break
        msgs = [r.getMessage() for r in caplog.records if r.name == "callosum.startup"]
        assert any("periodic smoke test cycle" in m for m in msgs), (
            f"expected a periodic cycle log within 4s, got: {msgs}"
        )
    finally:
        await tester.stop()


async def test_periodic_smoke_tester_disabled_when_interval_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """interval_s=0 means start() is a no-op; no background task is spawned."""
    caplog.set_level(logging.INFO, logger="callosum.startup")
    tester = _PeriodicSmokeTester(backends=[_backend(id="alpha")], interval_s=0)
    tester.start()
    try:
        await asyncio.sleep(0.2)  # give it a chance to misbehave
        msgs = [r.getMessage() for r in caplog.records if r.name == "callosum.startup"]
        assert not any("periodic smoke test cycle" in m for m in msgs), f"expected no cycles when disabled, got: {msgs}"
        assert not tester.enabled
    finally:
        await tester.stop()


async def test_periodic_smoke_tester_stops_cleanly_mid_wait(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A long interval should still let stop() return promptly — the tester's
    sleep is interruptible by the stop event, not a fixed-duration sleep.
    """
    caplog.set_level(logging.INFO, logger="callosum.startup")
    tester = _PeriodicSmokeTester(backends=[_backend(id="alpha")], interval_s=3600)
    tester.start()
    await asyncio.sleep(0.1)  # let the task park on the wait
    # If stop() blocks for the full 3600s, this test would time out.
    await asyncio.wait_for(tester.stop(), timeout=2.0)


# --------- catalog boot resync: give-up -> first-tick hole ()


from callosum.app import _catalog_boot_resync  # noqa: E402


class _FlakyCatalogBackend:
    """Fake backend whose `refresh_advertised_models` raises for the first
    `fail_n` calls (simulating a boot-time dependency that isn't ready yet)
    and then populates `advertised_models`. Used to exercise
    `_catalog_boot_resync`'s fast and slow-tail retry phases without real
    upstream calls or real 15s/60s waits."""

    def __init__(self, *, id: str, fail_n: int, models: frozenset[str]) -> None:
        self.id = id
        self.kind = "codex_auth_vault"
        self._fail_n = fail_n
        self._calls = 0
        self._models = models
        self.advertised_models: frozenset[str] = frozenset()

    async def refresh_advertised_models(self) -> None:
        self._calls += 1
        if self._calls <= self._fail_n:
            raise RuntimeError(f"simulated dependency-not-ready (call {self._calls})")
        self.advertised_models = self._models


def _resync_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "callosum.startup"]


async def test_catalog_boot_resync_populates_during_fast_phase(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Common case: dependency comes up within the fast retry budget. The
    catalog populates and the resync returns during the fast phase."""
    caplog.set_level(logging.WARNING, logger="callosum.startup")
    backend = _FlakyCatalogBackend(id="alpha", fail_n=2, models=frozenset({"model-a0e7"}))
    await _catalog_boot_resync(
        [backend],
        state_store=None,
        attempts=5,
        interval_s=0.01,
        tail_attempts=5,
        tail_interval_s=0.01,
    )
    assert backend.advertised_models == frozenset({"model-a0e7"})
    msgs = _resync_messages(caplog)
    assert any("all catalogs populated after" in m for m in msgs), msgs
    # Did not reach the slow tail.
    assert not any("slow tail" in m for m in msgs), msgs


async def test_catalog_boot_resync_slow_tail_closes_first_tick_hole(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Regression for the give-up -> first-tick hole (): a
    dependency that takes LONGER than the fast budget to come up must still
    self-heal during the slow tail, instead of leaving the catalog empty
    until the hourly smoke tester's first tick. Before the tail phase, the
    resync gave up after the fast budget and the catalog stayed empty for
    up to 1h."""
    caplog.set_level(logging.WARNING, logger="callosum.startup")
    # fail_n=4: fast phase has 3 attempts, so the fast budget is spent while
    # the catalog is still empty; the 4th refresh (first tail attempt) fails
    # too, and the 5th (second tail attempt) populates.
    backend = _FlakyCatalogBackend(id="beta", fail_n=4, models=frozenset({"model-a0e8"}))
    await _catalog_boot_resync(
        [backend],
        state_store=None,
        attempts=3,
        interval_s=0.01,
        tail_attempts=5,
        tail_interval_s=0.01,
    )
    assert backend.advertised_models == frozenset({"model-a0e8"})
    msgs = _resync_messages(caplog)
    # The fast phase exhausted (entered the tail) ...
    assert any("slow tail" in m for m in msgs), msgs
    # ... and the tail populated the catalog rather than giving up.
    assert any("all catalogs populated during slow tail" in m for m in msgs), msgs
    assert not any("gave up" in m for m in msgs), msgs


async def test_catalog_boot_resync_bounded_gives_up_after_both_phases(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The tail is bounded, not infinite: a dependency that never comes up
    exhausts both phases and gives up with an error, handing off to the
    hourly smoke tester. This keeps the resync from retrying forever."""
    caplog.set_level(logging.ERROR, logger="callosum.startup")
    backend = _FlakyCatalogBackend(id="gamma", fail_n=10_000, models=frozenset({"model-a0e9"}))
    await _catalog_boot_resync(
        [backend],
        state_store=None,
        attempts=3,
        interval_s=0.01,
        tail_attempts=4,
        tail_interval_s=0.01,
    )
    assert backend.advertised_models == frozenset()
    msgs = _resync_messages(caplog)
    assert any("gave up after 3 fast + 4 slow attempts" in m for m in msgs), msgs
