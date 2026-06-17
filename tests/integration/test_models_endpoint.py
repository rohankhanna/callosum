from __future__ import annotations

from fastapi.testclient import TestClient

from callosum.app import create_app
from callosum.fakes import InMemoryFakeBackend


def test_v1_models_returns_canonical_catalog() -> None:
    a = InMemoryFakeBackend(id="alpha", advertised_models=frozenset({"model-a0e7", "model-a0c3"}))
    b = InMemoryFakeBackend(id="beta", advertised_models=frozenset({"model-a0e7", "model-a0e6"}))
    with TestClient(create_app(backends=[a, b])) as client:
        response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    ids = {m["id"] for m in body["data"]}
    # Strategy selectors are always present.
    assert {"callosum:auto", "callosum:local-only", "callosum:remote-only"} <= ids
    # These fakes are remote (kind != litellm_gateway), so each model gets one
    # concrete remote pin per reasoning level (REASONING_LEVELS fallback).
    assert "callosum:remote/model-a0e7:high" in ids
    assert "callosum:remote/model-a0e6:low" in ids
    assert "callosum:remote/model-a0c3:xhigh" in ids
    # No local backend → no local pins.
    assert not any(i.startswith("callosum:local/") for i in ids)
    # Raw passthrough ids remain for back-compat.
    assert {"model-a0e6", "model-a0e7", "model-a0c3"} <= ids
    # Each entry has the OpenAI-compatible shape.
    for entry in body["data"]:
        assert entry["object"] == "model"
        assert "created" in entry
        assert entry["owned_by"] == "callosum"


def test_v1_models_selector_lookup() -> None:
    backend = InMemoryFakeBackend(id="alpha", advertised_models=frozenset({"model-a0e7"}))
    with TestClient(create_app(backends=[backend])) as client:
        # Strategy selector resolves.
        assert client.get("/v1/models/callosum:auto").status_code == 200
        # Valid concrete pin resolves.
        assert client.get("/v1/models/callosum:remote/model-a0e7:high").status_code == 200
        # Pin to an unadvertised model 404s.
        assert client.get("/v1/models/callosum:remote/model-a0g0:high").status_code == 404


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
