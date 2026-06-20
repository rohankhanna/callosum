"""Cold-boot resume invariant.

Regression guard for the every-morning lockout (incident: cold-boot
stale-quota lockout). The host is powered off overnight; the upstream
rate-limit windows reset while it sleeps; on boot callosum reloads last
session's persisted quota snapshot. The invariant: a backend whose persisted
snapshot reads exhausted but whose reset window has already PASSED must be
routable again at boot — callosum must re-sync with reality, not trust the
stale note. A backend whose window is genuinely still OPEN must stay blocked
(so the fix isn't just "ignore quotas").

These tests boot the full app (create_app + TestClient runs the lifespan)
with deliberately-stale state and assert the end-to-end routing outcome,
not just the blocking_meters unit (that is covered in tests/unit/test_selector.py).
The routing mode is forced remote-only so a failure to self-heal surfaces as
a hard 503 rather than being masked by a local fallback.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from callosum.app import create_app
from callosum.backend import HealthStatus, UsageSnapshot
from callosum.cell_grid import DEFAULT_MODELS
from callosum.codex_quota import CodexQuotaSnapshot
from callosum.fakes import InMemoryFakeBackend
from callosum.operator_state import OperatorState


def _quota(
    *,
    five_hourly: int,
    weekly: int,
    five_hourly_reset_at: int,
    weekly_reset_at: int,
) -> CodexQuotaSnapshot:
    return CodexQuotaSnapshot(
        plan_type="plus",
        active_limit="premium",
        five_hourly_used_percent=five_hourly,
        weekly_used_percent=weekly,
        five_hourly_window_minutes=300,
        weekly_window_minutes=10080,
        five_hourly_reset_at=five_hourly_reset_at,
        weekly_reset_at=weekly_reset_at,
        five_hourly_reset_after_seconds=None,
        weekly_reset_after_seconds=None,
        five_hourly_over_weekly_limit_percent=0,
        credits_balance=None,
        credits_has_credits=None,
        credits_unlimited=None,
        observed_at=0.0,
    )


def _remote_backend(*, id: str, quota: CodexQuotaSnapshot, usage: UsageSnapshot) -> InMemoryFakeBackend:
    b = InMemoryFakeBackend(
        id=id,
        advertised_models=frozenset(DEFAULT_MODELS),
        usage=usage,
        health=HealthStatus(available=True, reason="ok"),
    )
    b.kind = "codex_auth_vault"  # remote (not litellm_gateway)
    b._fake_quota = quota
    return b


def test_boot_with_stale_exhausted_quota_self_heals(tmp_path: Path) -> None:
    """Last night's snapshot reads 100% on both meters AND the sticky weekly
    flag is set, but both windows reset an hour ago while the host was off.
    On boot the backend must be routable again — the morning case."""
    now = int(time.time())
    stale = _remote_backend(
        id="remote",
        quota=_quota(
            five_hourly=100,
            weekly=100,
            five_hourly_reset_at=now - 3600,
            weekly_reset_at=now - 3600,
        ),
        usage=UsageSnapshot(
            remaining_fraction=0.0,
            cooldown_until_ts=None,
            weekly_exhausted=True,
            probed_at_ts=0.0,
        ),
    )
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("remote-only")  # no local fallback to mask an unhealed lockout
    try:
        app = create_app(backends=[stale], operator_state=state)
        with TestClient(app) as client:
            resp = client.post("/v1/responses", json={"model": "auto-learning", "input": []})
        assert resp.status_code == 200, resp.text
    finally:
        state.close()


def test_boot_with_open_window_quota_stays_blocked(tmp_path: Path) -> None:
    """Control: a backend genuinely exhausted within an OPEN window (reset in
    the future) must still be blocked at boot — the fix must not blanket-ignore
    quotas."""
    now = int(time.time())
    blocked = _remote_backend(
        id="remote",
        quota=_quota(
            five_hourly=100,
            weekly=1,
            five_hourly_reset_at=now + 3600,
            weekly_reset_at=now + 3600,
        ),
        usage=UsageSnapshot(
            remaining_fraction=0.0,
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=0.0,
        ),
    )
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("remote-only")
    try:
        app = create_app(backends=[blocked], operator_state=state)
        with TestClient(app) as client:
            resp = client.post("/v1/responses", json={"model": "auto-learning", "input": []})
        assert resp.status_code == 503, resp.text
        assert "Retry-After" in resp.headers
    finally:
        state.close()
