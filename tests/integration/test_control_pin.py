from __future__ import annotations

from fastapi.testclient import TestClient

from codex_proxy.app import create_app
from codex_proxy.fakes import InMemoryFakeBackend


def _two_backends() -> list[InMemoryFakeBackend]:
    return [
        InMemoryFakeBackend(id="alpha", advertised_models=frozenset({"model-a0f5-mini"})),
        InMemoryFakeBackend(id="beta", advertised_models=frozenset({"model-a0f5-mini"})),
    ]


def test_pin_forces_routing_to_chosen_backend() -> None:
    with TestClient(create_app(backends=_two_backends())) as client:
        # default: alpha wins on id tiebreak
        default = client.post(
            "/v1/chat/completions",
            json={"model": "model-a0f5-mini", "messages": []},
        )
        assert default.json()["id"] == "fake-alpha"

        pin = client.post("/control/pin", json={"backend_id": "beta"})
        assert pin.status_code == 200
        assert pin.json() == {"pinned": "beta"}

        pinned = client.post(
            "/v1/chat/completions",
            json={"model": "model-a0f5-mini", "messages": []},
        )
        assert pinned.json()["id"] == "fake-beta"


def test_unpin_restores_normal_selection() -> None:
    with TestClient(create_app(backends=_two_backends())) as client:
        client.post("/control/pin", json={"backend_id": "beta"})
        unpin = client.post("/control/unpin")
        assert unpin.status_code == 200
        assert unpin.json() == {"pinned": None}

        after = client.post(
            "/v1/chat/completions",
            json={"model": "model-a0f5-mini", "messages": []},
        )
        assert after.json()["id"] == "fake-alpha"


def test_pin_to_unknown_backend_returns_404() -> None:
    with TestClient(create_app(backends=_two_backends())) as client:
        response = client.post("/control/pin", json={"backend_id": "ghost"})
    assert response.status_code == 404


def test_pin_requires_backend_id_string() -> None:
    with TestClient(create_app(backends=_two_backends())) as client:
        response = client.post("/control/pin", json={})
    assert response.status_code == 400


def test_pin_to_backend_that_cannot_serve_model_returns_503() -> None:
    backends = [
        InMemoryFakeBackend(id="alpha", advertised_models=frozenset({"model-a0f5-mini"})),
        InMemoryFakeBackend(id="beta", advertised_models=frozenset({"other-model"})),
    ]
    with TestClient(create_app(backends=backends)) as client:
        client.post("/control/pin", json={"backend_id": "beta"})
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-a0f5-mini", "messages": []},
        )
    assert response.status_code == 503


def test_status_reports_pin_state() -> None:
    with TestClient(create_app(backends=_two_backends())) as client:
        default = client.get("/status").json()
        assert default["pinned"] is None

        client.post("/control/pin", json={"backend_id": "beta"})
        pinned = client.get("/status").json()
        assert pinned["pinned"] == "beta"
