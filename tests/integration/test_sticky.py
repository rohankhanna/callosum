from __future__ import annotations

from fastapi.testclient import TestClient

from callosum.app import create_app
from callosum.backend import UsageSnapshot
from callosum.errors import BackendError
from callosum.fakes import InMemoryFakeBackend


def _usage(remaining: float) -> UsageSnapshot:
    return UsageSnapshot(
        remaining_fraction=remaining,
        cooldown_until_ts=None,
        weekly_exhausted=False,
        probed_at_ts=0.0,
    )


def test_header_opts_session_into_sticky_binding() -> None:
    # Equal ranking initially; alpha wins on id tiebreak.
    alpha = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0d0"}),
        usage=_usage(0.5),
    )
    beta = InMemoryFakeBackend(
        id="beta",
        advertised_models=frozenset({"model-a0d0"}),
        usage=_usage(0.5),
    )
    with TestClient(create_app(backends=[alpha, beta])) as client:
        body = {"model": "model-a0d0", "messages": [{"role": "user", "content": "hi"}]}
        first = client.post("/v1/chat/completions", json=body, headers={"X-Codex-Session-Id": "s1"})
        assert first.status_code == 200
        assert first.json()["id"] == "fake-alpha"

        # After the first bind, make beta the obvious default winner. The
        # session header keeps s1 pinned to alpha.
        beta.set_usage(_usage(0.99))
        alpha.set_usage(_usage(0.1))

        second = client.post("/v1/chat/completions", json=body, headers={"X-Codex-Session-Id": "s1"})
        assert second.status_code == 200
        assert second.json()["id"] == "fake-alpha"


def test_no_header_reselects_fresh_each_request() -> None:
    alpha = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0d0"}),
        usage=_usage(0.5),
    )
    beta = InMemoryFakeBackend(
        id="beta",
        advertised_models=frozenset({"model-a0d0"}),
        usage=_usage(0.5),
    )
    with TestClient(create_app(backends=[alpha, beta])) as client:
        body = {"model": "model-a0d0", "messages": []}
        first = client.post("/v1/chat/completions", json=body)
        assert first.json()["id"] == "fake-alpha"

        # Swap rankings: a request without the header re-selects from scratch.
        beta.set_usage(_usage(0.99))
        alpha.set_usage(_usage(0.1))

        second = client.post("/v1/chat/completions", json=body)
        assert second.json()["id"] == "fake-beta"

        status = client.get("/status").json()
        assert status["sessions"] == {}


def test_sticky_binding_migrates_when_bound_backend_fails() -> None:
    alpha = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0d0"}),
        canned_error=BackendError(
            classification="rate_limited",
            status_code=429,
            message="alpha exhausted",
        ),
    )
    beta = InMemoryFakeBackend(
        id="beta",
        advertised_models=frozenset({"model-a0d0"}),
    )
    with TestClient(create_app(backends=[alpha, beta])) as client:
        body = {"model": "model-a0d0", "messages": []}
        response = client.post("/v1/chat/completions", json=body, headers={"X-Codex-Session-Id": "s1"})
        assert response.status_code == 200
        # alpha failed, rotation landed on beta, and the binding was updated.
        assert response.json()["id"] == "fake-beta"

        status = client.get("/status").json()
        assert status["sessions"] == {"s1": "beta"}


def test_pin_overrides_session_binding() -> None:
    alpha = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0d0"}),
    )
    beta = InMemoryFakeBackend(
        id="beta",
        advertised_models=frozenset({"model-a0d0"}),
    )
    with TestClient(create_app(backends=[alpha, beta])) as client:
        body = {"model": "model-a0d0", "messages": []}
        # Bind s1 to alpha (default tiebreak).
        client.post("/v1/chat/completions", json=body, headers={"X-Codex-Session-Id": "s1"})

        # Pin to beta. Subsequent s1 traffic must go to beta, not its binding.
        client.post("/control/pin", json={"backend_id": "beta"})
        response = client.post("/v1/chat/completions", json=body, headers={"X-Codex-Session-Id": "s1"})
        assert response.json()["id"] == "fake-beta"
