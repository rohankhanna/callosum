"""Tests for the verbose 503 detail when all backends are exhausted.

Every downstream tool (Hermes, codex-cli, Cursor) just prints whatever
`detail` field comes back. Without backend status in the body the user has
to separately curl /status to learn what's happening. With it, the error
is self-diagnosing: per-backend cooldown remaining, weekly_exhausted flag,
quota percents, recovery ETA.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from callosum.app import _collect_backend_status, _no_viable
from callosum.backend import HealthStatus, UsageSnapshot


@dataclass
class _FakeQuota:
    five_hourly_used_percent: int | None = 100
    five_hourly_reset_after_seconds: int | None = 1800
    weekly_used_percent: int | None = 44
    weekly_reset_after_seconds: int | None = 500_000
    # reset_at absolute timestamps: None = "no upstream datum", which
    # blocking_meters treats as an open window (percent honored as-is),
    # preserving this test's pre-reset-aware expectations.
    five_hourly_reset_at: int | None = None
    weekly_reset_at: int | None = None


@dataclass
class _FakeBackend:
    id: str
    cooldown_until_ts: float | None
    weekly_exhausted: bool
    advertised_models: frozenset[str] = frozenset({"model-a0e7"})
    kind: str = "test_stub"
    _quota: _FakeQuota | None = None

    async def health(self) -> HealthStatus:
        return HealthStatus(available=True, reason="ok")

    async def usage_snapshot(self) -> UsageSnapshot:
        return UsageSnapshot(
            remaining_fraction=None,
            cooldown_until_ts=self.cooldown_until_ts,
            weekly_exhausted=self.weekly_exhausted,
            probed_at_ts=time.time(),
        )

    async def quota_snapshot(self):  # noqa: ANN201
        return self._quota


def test_collect_backend_status_shape() -> None:
    """_collect_backend_status returns one dict per backend with the keys
    Hermes / codex-cli / Cursor need to print a useful error to the user.
    """
    now = time.time()
    b1 = _FakeBackend(
        id="primary",
        cooldown_until_ts=now + 3000,
        weekly_exhausted=True,
        _quota=_FakeQuota(),
    )
    b2 = _FakeBackend(
        id="secondary",
        cooldown_until_ts=None,
        weekly_exhausted=False,
        _quota=_FakeQuota(weekly_used_percent=34),
    )
    out = asyncio.run(_collect_backend_status([b1, b2]))
    assert len(out) == 2
    p, s = out
    assert p["id"] == "primary"
    assert p["weekly_exhausted"] is True
    assert p["cooldown_in_seconds"] is not None and p["cooldown_in_seconds"] > 0
    assert p["five_hourly_used_percent"] == 100
    assert p["weekly_used_percent"] == 44
    assert "advertised_models" in p

    assert s["id"] == "secondary"
    assert s["weekly_exhausted"] is False
    assert s["cooldown_in_seconds"] is None  # no active cooldown


def test_no_viable_detail_is_dict_when_backend_status_provided() -> None:
    """503 detail switches from a plain string to a structured dict so the
    downstream client can both print it AND parse it programmatically.
    """
    backend_status = [
        {"id": "primary", "cooldown_in_seconds": 2912, "weekly_exhausted": True},
        {"id": "secondary", "cooldown_in_seconds": 2916, "weekly_exhausted": False},
    ]
    exc = _no_viable(
        model="model-a0e7",
        last_error=None,
        backend_status=backend_status,
        recovery_ts=time.time() + 2900,
    )
    assert exc.status_code == 503
    detail = exc.detail
    assert isinstance(detail, dict)
    assert "summary" in detail
    assert detail["model"] == "model-a0e7"
    assert detail["backends"] == backend_status
    assert detail["recovery_in_seconds"] is not None


def test_no_viable_detail_is_string_when_backend_status_omitted() -> None:
    """Backwards-compat: callers that don't pass backend_status still get
    the original short string detail (synthetic path, tests, etc.)."""
    exc = _no_viable(model="model-a0e7", last_error=None)
    assert exc.status_code == 503
    assert isinstance(exc.detail, str)
    assert "model-a0e7" in exc.detail
