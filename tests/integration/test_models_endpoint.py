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


def test_v1_models_advertises_local_effort_variants() -> None:
    """A local backend (kind=litellm_gateway) gets a bare local pin plus a
    callosum:local/<model>:<effort> variant for each non-default reasoning
    level it advertises. Models advertising only ("default",) get the bare pin
    alone. (.)"""

    class _DefaultOnlyLocal(InMemoryFakeBackend):
        @property
        def model_metadata(self):  # type: ignore[override]
            from callosum.cell_grid import ModelMetadata

            return {
                slug: ModelMetadata(
                    slug=slug,
                    supported_in_api=True,
                    visibility="list",
                    priority=100,
                    supported_reasoning_levels=("default",),
                )
                for slug in self.advertised_models
            }

    # model-a0d2 exposes low/medium/high/xhigh (default fake metadata); model-a0g2 only
    # default. Both are local (litellm_gateway).
    effortful = InMemoryFakeBackend(id="local-oss", advertised_models=frozenset({"model-a0d2"}))
    effortful.kind = "litellm_gateway"
    default_only = _DefaultOnlyLocal(id="local-model-a0g3", advertised_models=frozenset({"model-a0g2"}))
    default_only.kind = "litellm_gateway"
    with TestClient(create_app(backends=[effortful, default_only])) as client:
        ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
    # Bare local pins for both.
    assert {"callosum:local/model-a0d2", "callosum:local/model-a0g2"} <= ids
    # Effort variants only for the model that advertises those levels.
    assert "callosum:local/model-a0d2:high" in ids
    assert "callosum:local/model-a0d2:low" in ids
    # default-only model gets NO effort variant, and "default" is never pinned.
    assert not any(i.startswith("callosum:local/model-a0g2:") for i in ids)
    assert "callosum:local/model-a0d2:default" not in ids
    # Single-model lookup resolves a local effort variant.
    with TestClient(create_app(backends=[effortful, default_only])) as client:
        r = client.get("/v1/models/callosum:local/model-a0d2:high")
    assert r.status_code == 200


def test_v1_models_selector_lookup() -> None:
    backend = InMemoryFakeBackend(id="alpha", advertised_models=frozenset({"model-a0e7"}))
    with TestClient(create_app(backends=[backend])) as client:
        # Strategy selector resolves.
        assert client.get("/v1/models/callosum:auto").status_code == 200
        # Valid concrete pin resolves.
        assert client.get("/v1/models/callosum:remote/model-a0e7:high").status_code == 200
        # Pin to an unadvertised model 404s.
        assert client.get("/v1/models/callosum:remote/model-a0g0:high").status_code == 404


def test_v1_models_hides_hidden_raw_ids_but_resolves_explicit_pin() -> None:
    class _HiddenReviewBackend(InMemoryFakeBackend):
        @property
        def model_metadata(self):  # type: ignore[override]
            from callosum.cell_grid import ModelMetadata

            return {
                "model-a0e7": ModelMetadata(
                    slug="model-a0e7",
                    supported_in_api=True,
                    visibility="list",
                    priority=10,
                    supported_reasoning_levels=("low",),
                ),
                "codex-auto-review": ModelMetadata(
                    slug="codex-auto-review",
                    supported_in_api=True,
                    visibility="hide",
                    priority=20,
                    supported_reasoning_levels=("medium",),
                ),
            }

    backend = _HiddenReviewBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0e7", "codex-auto-review"}),
    )
    with TestClient(create_app(backends=[backend])) as client:
        ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
        assert "codex-auto-review" not in ids
        assert "callosum:remote/codex-auto-review:medium" not in ids
        assert client.get("/v1/models/callosum:remote/codex-auto-review:medium").status_code == 200


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
