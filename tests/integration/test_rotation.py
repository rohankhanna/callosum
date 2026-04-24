from __future__ import annotations

from fastapi.testclient import TestClient

from codex_proxy.app import create_app
from codex_proxy.errors import BackendError
from codex_proxy.fakes import InMemoryFakeBackend


def test_rotation_on_rate_limited_backend() -> None:
    # 'alpha' sorts before 'beta', so the selector picks alpha first (deterministic id tiebreak).
    alpha = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0d0"}),
        canned_error=BackendError(
            classification="rate_limited",
            status_code=429,
            message="rate limited on alpha",
        ),
    )
    beta = InMemoryFakeBackend(
        id="beta",
        advertised_models=frozenset({"model-a0d0"}),
    )
    with TestClient(create_app(backends=[alpha, beta])) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-a0d0", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    assert response.json()["id"] == "fake-beta"


def test_rotation_on_auth_invalid_backend() -> None:
    alpha = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0d0"}),
        canned_error=BackendError(
            classification="auth_invalid",
            status_code=401,
            message="refresh token revoked",
        ),
    )
    beta = InMemoryFakeBackend(
        id="beta",
        advertised_models=frozenset({"model-a0d0"}),
    )
    with TestClient(create_app(backends=[alpha, beta])) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-a0d0", "messages": []},
        )
    assert response.status_code == 200
    assert response.json()["id"] == "fake-beta"


def test_client_error_is_not_retried() -> None:
    alpha = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0d0"}),
        canned_error=BackendError(
            classification="client_error",
            status_code=400,
            message="bad request",
        ),
    )
    beta = InMemoryFakeBackend(
        id="beta",
        advertised_models=frozenset({"model-a0d0"}),
    )
    with TestClient(create_app(backends=[alpha, beta])) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-a0d0", "messages": []},
        )
    assert response.status_code == 400


def test_all_backends_rate_limited_surfaces_429() -> None:
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
        canned_error=BackendError(
            classification="rate_limited",
            status_code=429,
            message="beta exhausted",
        ),
    )
    with TestClient(create_app(backends=[alpha, beta])) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-a0d0", "messages": []},
        )
    assert response.status_code == 429
