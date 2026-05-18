"""Tests for POST /control/clear-cooldown/{backend_id} admin endpoint and
the force-bypass on _diagnose_backend.

Both exist to break out of a stale-snapshot lockout where a persisted
cooldown stops the proxy from probing the backend that would tell it the
cooldown is no longer real.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from fastapi.testclient import TestClient

from codex_proxy.app import _diagnose_backend, create_app
from codex_proxy.backend import HealthStatus, UsageSnapshot
from codex_proxy.errors import BackendError


@dataclass
class _StuckCooldownBackend:
    """Minimal Backend implementing only what the control endpoint and the
    forced-diagnose path read. Always reports an active cooldown so we can
    verify both the skip-by-default and force-bypass behaviors.
    """

    id: str = "stuck"
    kind: str = "test_stub"
    advertised_models: frozenset[str] = frozenset({"gpt-test"})
    cleared: bool = False

    async def health(self) -> HealthStatus:
        return HealthStatus(available=True, reason="ok")

    async def usage_snapshot(self) -> UsageSnapshot:
        # Cooldown in the far future unless cleared.
        return UsageSnapshot(
            remaining_fraction=None,
            cooldown_until_ts=None if self.cleared else time.time() + 7 * 86400,
            weekly_exhausted=not self.cleared,
            probed_at_ts=time.time() - 2 * 86400,
        )

    async def quota_snapshot(self):  # noqa: ANN201 — matches duck-typed contract
        return None

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

    async def responses_stream(self, body, handle):  # noqa: ANN001 ANN201
        # We only need this to be reachable: the force-bypass test asserts that
        # the cooldown guard didn't short-circuit the call, not that the probe
        # succeeded. Raising a controlled BackendError keeps the test sealed
        # from any network interaction while still exercising the post-guard
        # code path.
        raise BackendError(
            status_code=502,
            message="stub backend has no real upstream",
            classification="transient",
        )
        yield  # pragma: no cover — generator protocol


def test_clear_cooldown_endpoint_clears_a_stuck_backend() -> None:
    backend = _StuckCooldownBackend(id="stuck")
    client = TestClient(create_app(backends=[backend]))
    with client:
        resp = client.post("/control/clear-cooldown/stuck")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == "stuck"
    assert body["cleared"] is True
    assert body["snapshot"]["cooldown_until_ts"] is None
    assert body["snapshot"]["weekly_exhausted"] is False
    assert backend.cleared is True


def test_clear_cooldown_endpoint_returns_404_for_unknown_backend() -> None:
    client = TestClient(create_app(backends=[_StuckCooldownBackend(id="stuck")]))
    with client:
        resp = client.post("/control/clear-cooldown/nope")
    assert resp.status_code == 404
    assert "not in pool" in resp.json()["detail"]


def test_diagnose_backend_skips_cooldowned_by_default() -> None:
    """Default behavior: cooldown'd backend is reported as skipped."""
    backend = _StuckCooldownBackend(id="stuck")
    result = asyncio.run(_diagnose_backend(backend))
    assert result["skipped"] is True
    assert result["stage"] == "cooldown"


def test_diagnose_backend_force_bypasses_cooldown_skip() -> None:
    """force=True path: cooldown'd backend is NOT skipped — the probe proceeds.

    We don't expect ok=True here because the stub doesn't implement the
    streaming responses path; we only assert that the cooldown guard didn't
    short-circuit the call.
    """
    backend = _StuckCooldownBackend(id="stuck")
    result = asyncio.run(_diagnose_backend(backend, force=True))
    assert result.get("skipped") is not True
    assert result.get("stage") != "cooldown"
