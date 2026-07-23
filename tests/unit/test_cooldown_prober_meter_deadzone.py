"""Regression tests for the post-single-quota-window dead zone in
`_PeriodicCooldownProber._probe_cooldowned`.

After upstream merged its two quota windows into one weekly window riding
the `x-codex-primary-*` headers, the parsed `weekly_*` fields are permanently
None and `weekly_exhausted` never gets set. A backend blocked ONLY by the 100%
quota meter (expired cooldown, `weekly_exhausted=False`) used to fall into a
dead zone: the prober skipped it (no active cooldown, not weekly-exhausted)
and `_routable_backends` excluded it from real traffic, so nothing refreshed
the stale snapshot and the lock could hold for up to a week. The fix adds a
`blocking_meters` check to the prober's skip condition so the forced probe
fires and self-heals the snapshot.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from callosum import app as app_module
from callosum.app import _PeriodicCooldownProber, _routable_backends
from callosum.backend import HealthStatus, UsageSnapshot
from callosum.codex_quota import CodexQuotaSnapshot


@dataclass
class _MeterBlockedBackend:
    """Minimal Backend whose persisted state is NOT in cooldown and NOT
    weekly-exhausted, but whose cached quota snapshot reports a 100%
    five_hourly meter with a future reset (i.e. blocking_meters is non-empty).

    This is the exact post-single-window dead-zone shape: the only thing
    keeping the backend unroutable is the stale five_hourly meter.
    """

    id: str = "meter-blocked"
    kind: str = "test_stub"
    advertised_models: frozenset[str] = frozenset({"gpt-test"})
    cleared: bool = False
    # Mutable so the fake diagnose probe can "refresh" the snapshot to <100%
    # to simulate the self-heal path.
    quota: CodexQuotaSnapshot = field(
        default_factory=lambda: CodexQuotaSnapshot(
            plan_type=None,
            active_limit=None,
            five_hourly_used_percent=100,
            weekly_used_percent=None,
            five_hourly_window_minutes=None,
            weekly_window_minutes=None,
            five_hourly_reset_at=int(time.time()) + 7 * 86400,
            weekly_reset_at=None,
            five_hourly_reset_after_seconds=None,
            weekly_reset_after_seconds=None,
            five_hourly_over_weekly_limit_percent=None,
            credits_balance=None,
            credits_has_credits=None,
            credits_unlimited=None,
            observed_at=time.time(),
        )
    )

    async def health(self) -> HealthStatus:
        return HealthStatus(available=True, reason="ok")

    async def usage_snapshot(self) -> UsageSnapshot:
        return UsageSnapshot(
            remaining_fraction=None,
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=time.time() - 2 * 86400,
        )

    async def quota_snapshot(self) -> CodexQuotaSnapshot | None:
        return self.quota

    def clear_cooldown(self) -> UsageSnapshot:
        self.cleared = True
        return UsageSnapshot(
            remaining_fraction=None,
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=time.time(),
        )

    async def aclose(self) -> None:
        pass


def test_periodic_prober_reprobes_backend_blocked_only_by_meter(monkeypatch) -> None:
    """A backend with no active cooldown and weekly_exhausted=False but a
    non-empty blocking_meters (100% five_hourly with future reset) MUST be
    re-probed. Pre-fix this backend was skipped forever (the dead zone)."""
    backend = _MeterBlockedBackend(id="meter-blocked")
    calls: list[tuple[str, bool]] = []

    async def fake_diagnose_backend(probed_backend, *, force=False):  # noqa: ANN001 ANN201
        calls.append((probed_backend.id, force))
        return {"ok": True, "skipped": False}

    async def run_once() -> None:
        monkeypatch.setattr(app_module, "_diagnose_backend", fake_diagnose_backend)
        prober = _PeriodicCooldownProber(backends=[backend], interval_s=0.01)
        prober.start()
        await asyncio.sleep(0.05)
        await prober.stop()

    asyncio.run(run_once())

    # The prober did NOT skip: it forced at least one probe. Pre-fix `calls`
    # would be empty (the dead zone). Every recorded call must be a forced
    # probe of this backend — the prober never calls with force=False.
    assert calls, "prober skipped the meter-blocked backend (dead zone regression)"
    assert all(c == ("meter-blocked", True) for c in calls)
    # And the self-heal fired: clear_cooldown was called on the ok path.
    assert backend.cleared is True


def test_periodic_prober_skips_when_no_block_no_cooldown_no_exhaustion(monkeypatch) -> None:
    """Negative control: a fully-healthy backend (no cooldown, no
    weekly_exhausted, no blocking meter) is still skipped by the prober —
    the blocking_meters check must not turn the prober into a constant
    re-probe of every healthy backend."""
    backend = _MeterBlockedBackend(id="healthy")
    # Refresh to a healthy quota (<100%, future reset still open but not used).
    backend.quota = CodexQuotaSnapshot(
        plan_type=None,
        active_limit=None,
        five_hourly_used_percent=42,
        weekly_used_percent=None,
        five_hourly_window_minutes=None,
        weekly_window_minutes=None,
        five_hourly_reset_at=int(time.time()) + 3600,
        weekly_reset_at=None,
        five_hourly_reset_after_seconds=None,
        weekly_reset_after_seconds=None,
        five_hourly_over_weekly_limit_percent=None,
        credits_balance=None,
        credits_has_credits=None,
        credits_unlimited=None,
        observed_at=time.time(),
    )
    calls: list[tuple[str, bool]] = []

    async def fake_diagnose_backend(probed_backend, *, force=False):  # noqa: ANN001 ANN201
        calls.append((probed_backend.id, force))
        return {"ok": True, "skipped": False}

    async def run_once() -> None:
        monkeypatch.setattr(app_module, "_diagnose_backend", fake_diagnose_backend)
        prober = _PeriodicCooldownProber(backends=[backend], interval_s=0.01)
        prober.start()
        await asyncio.sleep(0.05)
        await prober.stop()

    asyncio.run(run_once())

    assert calls == []
    assert backend.cleared is False


def test_routable_backends_excludes_meter_blocked_backend() -> None:
    """`_routable_backends` excludes a backend whose blocking_meters is
    non-empty even when cooldown is expired and weekly_exhausted is False —
    this is the other half of the dead zone and the reason real traffic
    can't self-heal the snapshot. Confirms the prober is the only escape."""
    backend = _MeterBlockedBackend(id="meter-blocked")
    routable = asyncio.run(_routable_backends([backend]))
    assert routable == []

    # And once the quota refreshes to <100%, the backend becomes routable,
    # modelling the post-probe self-heal.
    backend.quota = CodexQuotaSnapshot(
        plan_type=None,
        active_limit=None,
        five_hourly_used_percent=0,
        weekly_used_percent=None,
        five_hourly_window_minutes=None,
        weekly_window_minutes=None,
        five_hourly_reset_at=int(time.time()) + 3600,
        weekly_reset_at=None,
        five_hourly_reset_after_seconds=None,
        weekly_reset_after_seconds=None,
        five_hourly_over_weekly_limit_percent=None,
        credits_balance=None,
        credits_has_credits=None,
        credits_unlimited=None,
        observed_at=time.time(),
    )
    routable = asyncio.run(_routable_backends([backend]))
    assert [b.id for b in routable] == ["meter-blocked"]