"""Tests for callosum.capability.scheduler.PeriodicHarnessSweep.

The periodic sweeper is the difference between "the harness fires once
at startup" and "the harness keeps the cell grid current for the
lifetime of the proxy." These tests verify:

  * the sweeper actually re-fires after its interval elapses
  * stop() cleanly exits the loop without leaking a task
  * .enabled=False scenarios (no backends; zero interval) leave the
    loop dormant rather than spawning a doomed task
  * a sweep that raises does NOT crash the loop — subsequent ticks
    must still run
"""

from __future__ import annotations

import asyncio

import pytest

from callosum.capability import scheduler


class _FakeBackend:
    """Minimal backend stub: advertises one model and exposes a
    responses() coroutine so the harness's cell-walk doesn't filter
    it out."""

    def __init__(self, model: str = "fake-cell") -> None:
        self.kind = "litellm_gateway"
        self.id = f"fake-backend-{model}"
        self.advertised_models = frozenset({model})

    async def responses(self, body: dict) -> dict:
        return {"output": []}


@pytest.mark.asyncio
async def test_periodic_sweeper_ticks_after_interval(monkeypatch) -> None:
    """A short interval should cause `run_harness_sweep` to be invoked
    repeatedly. Counting invocations is the simplest end-to-end signal
    that the loop is alive."""
    calls: list[float] = []

    async def fake_sweep(*, backends, ttl_s):
        calls.append(asyncio.get_event_loop().time())
        return 0

    monkeypatch.setattr(scheduler, "run_harness_sweep", fake_sweep)

    sweeper = scheduler.PeriodicHarnessSweep(
        backends=[_FakeBackend()],
        interval_s=0.05,
    )
    sweeper.start()
    # Sleep long enough for ≥2 ticks (0.05s interval → ≥2 ticks in 0.15s).
    await asyncio.sleep(0.18)
    await sweeper.stop()

    assert len(calls) >= 2, (
        f"expected at least 2 periodic ticks, got {len(calls)}"
    )


@pytest.mark.asyncio
async def test_disabled_when_no_backends(monkeypatch) -> None:
    """An empty backend list means there's nothing to probe — the
    sweeper must not spawn an idle task that loops forever doing
    nothing."""
    called = False

    async def fake_sweep(*, backends, ttl_s):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(scheduler, "run_harness_sweep", fake_sweep)
    sweeper = scheduler.PeriodicHarnessSweep(
        backends=[],
        interval_s=0.01,
    )
    assert sweeper.enabled is False
    sweeper.start()
    await asyncio.sleep(0.05)
    await sweeper.stop()
    assert called is False


@pytest.mark.asyncio
async def test_disabled_when_interval_is_zero(monkeypatch) -> None:
    """Operators may set CALLOSUM_HARNESS_SWEEP_INTERVAL_SECONDS=0 to
    explicitly disable the periodic sweeper while keeping the one-shot
    startup pass. Verify zero interval honors that intent."""
    called = False

    async def fake_sweep(*, backends, ttl_s):
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(scheduler, "run_harness_sweep", fake_sweep)
    sweeper = scheduler.PeriodicHarnessSweep(
        backends=[_FakeBackend()],
        interval_s=0,
    )
    assert sweeper.enabled is False
    sweeper.start()
    await asyncio.sleep(0.05)
    await sweeper.stop()
    assert called is False


@pytest.mark.asyncio
async def test_stop_is_idempotent(monkeypatch) -> None:
    """Calling stop() multiple times (or before start) must not raise.
    FastAPI's lifespan teardown can run finally-blocks in surprising
    orders during cancellation; the sweeper has to tolerate that."""

    async def fake_sweep(*, backends, ttl_s):
        return 0

    monkeypatch.setattr(scheduler, "run_harness_sweep", fake_sweep)
    sweeper = scheduler.PeriodicHarnessSweep(
        backends=[_FakeBackend()],
        interval_s=0.05,
    )
    await sweeper.stop()  # before start
    sweeper.start()
    await sweeper.stop()
    await sweeper.stop()  # second stop


@pytest.mark.asyncio
async def test_sweep_exception_does_not_break_loop(monkeypatch) -> None:
    """If a tick's sweep raises (e.g. transient ollama failure), the
    sweeper must log and continue rather than dying. Verify by
    raising on the first call and counting subsequent successful calls."""
    call_count = 0

    async def flaky_sweep(*, backends, ttl_s):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("simulated transient failure")
        return 0

    monkeypatch.setattr(scheduler, "run_harness_sweep", flaky_sweep)
    sweeper = scheduler.PeriodicHarnessSweep(
        backends=[_FakeBackend()],
        interval_s=0.03,
    )
    sweeper.start()
    await asyncio.sleep(0.15)
    await sweeper.stop()

    # First call raised; subsequent calls must still have happened.
    assert call_count >= 2


def test_default_interval_respects_env_override(monkeypatch) -> None:
    """Verify the env-var override path works end-to-end. Setting the
    env var should make the next PeriodicHarnessSweep() pick up the
    override as its default."""
    monkeypatch.setenv("CALLOSUM_HARNESS_SWEEP_INTERVAL_SECONDS", "120")
    sweeper = scheduler.PeriodicHarnessSweep(backends=[_FakeBackend()])
    assert sweeper._interval_s == 120.0


def test_default_interval_floors_tiny_env_values(monkeypatch) -> None:
    """A typoed env var (e.g. 0.001) must not pin the GPU forever.
    Floor of 60s applies to env-provided values."""
    monkeypatch.setenv("CALLOSUM_HARNESS_SWEEP_INTERVAL_SECONDS", "0.001")
    sweeper = scheduler.PeriodicHarnessSweep(backends=[_FakeBackend()])
    assert sweeper._interval_s == 60.0


def test_default_interval_ignores_non_numeric(monkeypatch) -> None:
    """Bad env var falls back to the 6h default, not a crash. Operator
    typos in env vars are common; the sweeper must degrade gracefully."""
    monkeypatch.setenv("CALLOSUM_HARNESS_SWEEP_INTERVAL_SECONDS", "soon")
    sweeper = scheduler.PeriodicHarnessSweep(backends=[_FakeBackend()])
    assert sweeper._interval_s == 6 * 3600.0
