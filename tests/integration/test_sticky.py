from __future__ import annotations

from fastapi.testclient import TestClient

from codex_proxy.app import create_app
from codex_proxy.backend import UsageSnapshot
from codex_proxy.errors import BackendError
from codex_proxy.fakes import InMemoryFakeBackend


def _usage(remaining: float) -> UsageSnapshot:
    return UsageSnapshot(
        remaining_fraction=remaining,
        cooldown_until_ts=None,
        weekly_exhausted=False,
        probed_at_ts=0.0,
    )


def test_sticky_mode_keeps_session_on_first_picked_backend() -> None:
    # Equal ranking initially; alpha wins on id tiebreak.
    alpha = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0f5-mini"}),
        usage=_usage(0.5),
    )
    beta = InMemoryFakeBackend(
        id="beta",
        advertised_models=frozenset({"model-a0f5-mini"}),
        usage=_usage(0.5),
    )
    with TestClient(create_app(backends=[alpha, beta], session_mode="sticky")) as client:
        body = {"model": "model-a0f5-mini", "messages": [{"role": "user", "content": "hi"}]}
        first = client.post("/v1/chat/completions", json=body, headers={"X-Codex-Session-Id": "s1"})
        assert first.status_code == 200
        assert first.json()["id"] == "fake-alpha"

        # After the first bind, make beta the obvious default winner. Sticky
        # should still keep s1 on alpha.
        beta.set_usage(_usage(0.99))
        alpha.set_usage(_usage(0.1))

        second = client.post(
            "/v1/chat/completions", json=body, headers={"X-Codex-Session-Id": "s1"}
        )
        assert second.status_code == 200
        assert second.json()["id"] == "fake-alpha"

        # A different session id is a fresh selection and must reflect the
        # current usage landscape, i.e. pick beta.
        third = client.post("/v1/chat/completions", json=body, headers={"X-Codex-Session-Id": "s2"})
        assert third.status_code == 200
        assert third.json()["id"] == "fake-beta"


def test_sticky_binding_migrates_when_bound_backend_fails() -> None:
    alpha = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0f5-mini"}),
        canned_error=BackendError(
            classification="rate_limited",
            status_code=429,
            message="alpha exhausted",
        ),
    )
    beta = InMemoryFakeBackend(
        id="beta",
        advertised_models=frozenset({"model-a0f5-mini"}),
    )
    with TestClient(create_app(backends=[alpha, beta], session_mode="sticky")) as client:
        body = {"model": "model-a0f5-mini", "messages": []}
        response = client.post(
            "/v1/chat/completions", json=body, headers={"X-Codex-Session-Id": "s1"}
        )
        assert response.status_code == 200
        # alpha failed, rotation landed on beta, and the binding was updated.
        assert response.json()["id"] == "fake-beta"

        status = client.get("/status").json()
        assert status["session_mode"] == "sticky"
        assert status["sessions"] == {"s1": "beta"}


def test_sticky_mode_without_session_header_behaves_like_stateless() -> None:
    alpha = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0f5-mini"}),
        usage=_usage(0.5),
    )
    beta = InMemoryFakeBackend(
        id="beta",
        advertised_models=frozenset({"model-a0f5-mini"}),
        usage=_usage(0.5),
    )
    with TestClient(create_app(backends=[alpha, beta], session_mode="sticky")) as client:
        body = {"model": "model-a0f5-mini", "messages": []}
        first = client.post("/v1/chat/completions", json=body)
        assert first.json()["id"] == "fake-alpha"

        # Swap rankings to prove the second request re-selects afresh.
        beta.set_usage(_usage(0.99))
        alpha.set_usage(_usage(0.1))

        second = client.post("/v1/chat/completions", json=body)
        assert second.json()["id"] == "fake-beta"

        status = client.get("/status").json()
        assert status["sessions"] == {}


def test_stateless_mode_ignores_session_header() -> None:
    alpha = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0f5-mini"}),
        usage=_usage(0.5),
    )
    beta = InMemoryFakeBackend(
        id="beta",
        advertised_models=frozenset({"model-a0f5-mini"}),
        usage=_usage(0.5),
    )
    with TestClient(create_app(backends=[alpha, beta], session_mode="stateless")) as client:
        body = {"model": "model-a0f5-mini", "messages": []}
        first = client.post("/v1/chat/completions", json=body, headers={"X-Codex-Session-Id": "s1"})
        assert first.json()["id"] == "fake-alpha"

        beta.set_usage(_usage(0.99))
        alpha.set_usage(_usage(0.1))

        second = client.post(
            "/v1/chat/completions", json=body, headers={"X-Codex-Session-Id": "s1"}
        )
        assert second.json()["id"] == "fake-beta"

        status = client.get("/status").json()
        assert status["session_mode"] == "stateless"
        assert status["sessions"] == {}


def test_pin_overrides_sticky_binding() -> None:
    alpha = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0f5-mini"}),
    )
    beta = InMemoryFakeBackend(
        id="beta",
        advertised_models=frozenset({"model-a0f5-mini"}),
    )
    with TestClient(create_app(backends=[alpha, beta], session_mode="sticky")) as client:
        body = {"model": "model-a0f5-mini", "messages": []}
        # Bind s1 to alpha (default tiebreak).
        client.post("/v1/chat/completions", json=body, headers={"X-Codex-Session-Id": "s1"})

        # Pin to beta. Subsequent s1 traffic must go to beta, not its binding.
        client.post("/control/pin", json={"backend_id": "beta"})
        response = client.post(
            "/v1/chat/completions", json=body, headers={"X-Codex-Session-Id": "s1"}
        )
        assert response.json()["id"] == "fake-beta"
