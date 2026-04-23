from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from codex_proxy.app import create_app
from codex_proxy.backends.openai_api_key import OpenAIApiKeyBackend
from codex_proxy.errors import BackendError
from codex_proxy.fakes import InMemoryFakeBackend


def test_rotation_on_rate_limited_backend() -> None:
    # 'alpha' sorts before 'beta', so the selector picks alpha first (deterministic id tiebreak).
    alpha = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0f5-mini"}),
        canned_error=BackendError(
            classification="rate_limited",
            status_code=429,
            message="rate limited on alpha",
        ),
    )
    beta = InMemoryFakeBackend(
        id="beta",
        advertised_models=frozenset({"model-a0f5-mini"}),
    )
    with TestClient(create_app(backends=[alpha, beta])) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-a0f5-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    assert response.json()["id"] == "fake-beta"


def test_rotation_on_auth_invalid_backend() -> None:
    alpha = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0f5-mini"}),
        canned_error=BackendError(
            classification="auth_invalid",
            status_code=401,
            message="bad key",
        ),
    )
    beta = InMemoryFakeBackend(
        id="beta",
        advertised_models=frozenset({"model-a0f5-mini"}),
    )
    with TestClient(create_app(backends=[alpha, beta])) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-a0f5-mini", "messages": []},
        )
    assert response.status_code == 200
    assert response.json()["id"] == "fake-beta"


def test_client_error_is_not_retried() -> None:
    alpha = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0f5-mini"}),
        canned_error=BackendError(
            classification="client_error",
            status_code=400,
            message="bad request",
        ),
    )
    beta = InMemoryFakeBackend(
        id="beta",
        advertised_models=frozenset({"model-a0f5-mini"}),
    )
    with TestClient(create_app(backends=[alpha, beta])) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-a0f5-mini", "messages": []},
        )
    assert response.status_code == 400


def test_all_backends_rate_limited_surfaces_429() -> None:
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
        canned_error=BackendError(
            classification="rate_limited",
            status_code=429,
            message="beta exhausted",
        ),
    )
    with TestClient(create_app(backends=[alpha, beta])) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-a0f5-mini", "messages": []},
        )
    assert response.status_code == 429


def test_streaming_rotation_with_real_backend_upstreams() -> None:
    stream_body = b'data: {"id":"c1"}\n\ndata: [DONE]\n\n'

    def alpha_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            headers={"retry-after": "2"},
            json={"error": {"message": "rate limited"}},
        )

    def beta_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=stream_body,
        )

    alpha = OpenAIApiKeyBackend(
        id="alpha",
        api_key="sk-a",
        advertised_models=frozenset({"model-a0f5-mini"}),
        transport=httpx.MockTransport(alpha_handler),
    )
    beta = OpenAIApiKeyBackend(
        id="beta",
        api_key="sk-b",
        advertised_models=frozenset({"model-a0f5-mini"}),
        transport=httpx.MockTransport(beta_handler),
    )
    with (
        TestClient(create_app(backends=[alpha, beta])) as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "model-a0f5-mini",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as response,
    ):
        assert response.status_code == 200
        body = b"".join(response.iter_bytes())
    assert body == stream_body
