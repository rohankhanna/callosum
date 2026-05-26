from __future__ import annotations

from fastapi.testclient import TestClient

from callosum.app import create_app
from callosum.fakes import InMemoryFakeBackend


def test_v1_models_returns_union_of_advertised_models() -> None:
    a = InMemoryFakeBackend(id="alpha", advertised_models=frozenset({"model-a0e7", "model-a0c3"}))
    b = InMemoryFakeBackend(id="beta", advertised_models=frozenset({"model-a0e7", "model-a0e6"}))
    with TestClient(create_app(backends=[a, b])) as client:
        response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    ids = sorted(m["id"] for m in body["data"])
    assert ids == ["model-a0e6", "model-a0e7", "model-a0c3"]
    # Each entry has the OpenAI-compatible shape.
    for entry in body["data"]:
        assert entry["object"] == "model"
        assert "created" in entry
        assert "owned_by" in entry


def test_v1_models_lookup_returns_advertised_one() -> None:
    backend = InMemoryFakeBackend(id="alpha", advertised_models=frozenset({"model-a0e7"}))
    with TestClient(create_app(backends=[backend])) as client:
        response = client.get("/v1/models/model-a0e7")
    assert response.status_code == 200
    assert response.json()["id"] == "model-a0e7"


def test_v1_models_lookup_returns_404_for_unknown() -> None:
    backend = InMemoryFakeBackend(id="alpha", advertised_models=frozenset({"model-a0e7"}))
    with TestClient(create_app(backends=[backend])) as client:
        response = client.get("/v1/models/totally-made-up-model")
    assert response.status_code == 404


def test_v1_models_handles_provider_slash_id_format() -> None:
    """Some backend kinds expose model ids that contain '/' (e.g.
    'meta-model-a0g1/model-a0a5'). The path param uses :path so
    slashes survive routing.
    """
    backend = InMemoryFakeBackend(
        id="or",
        advertised_models=frozenset({"meta-model-a0g1/model-a0a5:free"}),
    )
    with TestClient(create_app(backends=[backend])) as client:
        response = client.get("/v1/models/meta-model-a0g1/model-a0a5:free")
    assert response.status_code == 200
    assert response.json()["id"] == "meta-model-a0g1/model-a0a5:free"
