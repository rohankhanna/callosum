from __future__ import annotations

import logging
from typing import cast

import pytest
from fastapi.testclient import TestClient

from codex_proxy.app import create_app
from codex_proxy.backend import HealthStatus, UsageSnapshot
from codex_proxy.fakes import InMemoryFakeBackend


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
    caplog.set_level(logging.INFO, logger="codex_proxy.startup")
    a = _backend(id="alpha")
    b = _backend(id="beta", cooldown_until_ts=1e12)  # cooldown far in the future
    app = create_app(backends=[a, b], startup_smoke_test=True)

    with TestClient(app):
        pass  # entering the context fires lifespan startup

    msgs = [r.getMessage() for r in caplog.records if r.name == "codex_proxy.startup"]
    text = "\n".join(msgs)
    assert "startup smoke test: probing 2 backend(s)" in text
    # alpha should be probed; beta should be skipped (cooldown).
    assert "[alpha]" in text
    assert "[beta] SKIPPED" in text


def test_startup_smoke_test_can_be_disabled(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="codex_proxy.startup")
    backend = _backend(id="alpha")
    app = create_app(backends=[backend], startup_smoke_test=False)

    with TestClient(app):
        pass

    msgs = [r.getMessage() for r in caplog.records if r.name == "codex_proxy.startup"]
    assert msgs == [], f"expected no smoke-test logs when disabled, got: {msgs}"


def test_startup_smoke_test_no_backends_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="codex_proxy.startup")
    app = create_app(backends=[], startup_smoke_test=True)

    with TestClient(app):
        pass

    msgs = cast(
        list[str],
        [r.getMessage() for r in caplog.records if r.name == "codex_proxy.startup"],
    )
    assert msgs == [], f"empty backend list should not trigger smoke test, got: {msgs}"
