from __future__ import annotations

import base64
import json
import logging
from typing import cast

import httpx
import pytest
from fastapi.testclient import TestClient

from callosum.app import create_app
from callosum.backend import HealthStatus, UsageSnapshot
from callosum.backends.credential_proxy import CredentialProxyBackend
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


def _credential_proxy_backend_with_quota() -> CredentialProxyBackend:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/standin":
            return httpx.Response(
                200,
                json={
                    "token": "ati_test_token",
                    "token_id": "test_token_id",
                    "account": "primary",
                    "expires_at": 9999999999.0,
                },
            )
        if request.url.path == "/v1/proxy/stream":
            payload = json.loads(request.content.decode("utf-8"))
            body = json.loads(base64.b64decode(payload["body_b64"]).decode("utf-8"))
            assert body["model"] == "model-a0e7"
            return httpx.Response(
                200,
                headers={
                    "content-type": "text/event-stream",
                    "x-credential proxy-upstream-status": "200",
                    "x-codex-plan-type": "plus",
                    "x-codex-active-limit": "premium",
                    "x-codex-primary-used-percent": "1",
                    "x-codex-secondary-used-percent": "10",
                    "x-codex-primary-window-minutes": "300",
                    "x-codex-secondary-window-minutes": "10080",
                    "x-codex-primary-reset-after-seconds": "1000",
                    "x-codex-secondary-reset-after-seconds": "100000",
                    "x-codex-primary-reset-at": "1777000000",
                    "x-codex-secondary-reset-at": "1777500000",
                    "x-codex-primary-over-secondary-limit-percent": "0",
                    "x-codex-credits-has-credits": "False",
                    "x-codex-credits-unlimited": "False",
                },
                content=(
                    b'event: response.created\ndata: {"type":"response.created","id":"r1"}\n\n'
                    b'event: response.completed\ndata: {"type":"response.completed","response":{"id":"r1","model":"model-a0e7","usage":{"input_tokens":4,"output_tokens":1,"total_tokens":5}}}\n\n'
                ),
            )
        raise AssertionError(f"unexpected path: {request.url.path}")

    return CredentialProxyBackend(
        id="primary",
        proxy_url="http://127.0.0.1:7342",
        upstream_url="https://chatgpt.com/backend-api/codex/responses",
        advertised_models=frozenset({"codex-auto-review", "model-a0e7"}),
        custody_account="primary",
        transport=httpx.MockTransport(handler),
    )


def test_startup_smoke_test_refreshes_credential_proxy_quota_cache() -> None:
    backend = _credential_proxy_backend_with_quota()
    app = create_app(backends=[backend], startup_smoke_test=True)

    with TestClient(app):
        pass

    assert backend._last_quota is not None
    assert backend._last_quota.weekly_used_percent == 10
