"""Client-driven `callosum:` selectors at dispatch time.

Pins that a per-request selector overrides the operator routing mode for that
one request only (no global mutation), that concrete pins constrain the served
backend/model, and that an unsatisfiable pin returns a clean 503 (not a 400
retry loop). Companion to test_routing_mode_matrix.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from callosum.app import create_app
from callosum.backend import BackendKind, HealthStatus, UsageSnapshot
from callosum.cell_grid import DEFAULT_MODELS
from callosum.fakes import InMemoryFakeBackend
from callosum.operator_state import OperatorState


def _make_backend(
    *,
    id: str,
    kind: BackendKind,
    weekly_exhausted: bool = False,
) -> InMemoryFakeBackend:
    b = InMemoryFakeBackend(
        id=id,
        advertised_models=frozenset(DEFAULT_MODELS),
        usage=UsageSnapshot(
            remaining_fraction=0.0 if weekly_exhausted else 1.0,
            cooldown_until_ts=None,
            weekly_exhausted=weekly_exhausted,
            probed_at_ts=0.0,
        ),
        health=HealthStatus(available=True, reason="ok"),
    )
    b.kind = kind  # override the default codex_auth_vault
    return b


@pytest.fixture(autouse=True)
def _no_canary(monkeypatch):
    # Keep canary diversion out of the selector tests.
    monkeypatch.setenv("CALLOSUM_CANARY_PERCENT", "0")


def test_remote_only_selector_overrides_operator_default(tmp_path: Path) -> None:
    """Operator default is local-only, but a callosum:remote-only selector
    routes this one request to the remote backend — and leaves the operator
    mode untouched."""
    remote = _make_backend(id="remote", kind="codex_auth_vault")
    local = _make_backend(id="local", kind="litellm_gateway")
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("local-only")
    try:
        app = create_app(backends=[remote, local], operator_state=state)
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "callosum:remote-only", "input": []},
            )
        assert response.status_code == 200, response.text
        assert response.json()["id"] == "resp-remote"
        # No global mutation: operator default is still local-only.
        assert state.get_routing() == "local-only"
    finally:
        state.close()


def test_local_pin_routes_to_local(tmp_path: Path) -> None:
    remote = _make_backend(id="remote", kind="codex_auth_vault")
    local = _make_backend(id="local", kind="litellm_gateway")
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("auto")
    try:
        app = create_app(backends=[remote, local], operator_state=state)
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "callosum:local/model-a0e7", "input": []},
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"] == "resp-local"
        # Upstream sees the resolved real model, never the callosum: id.
        assert body["model"] == "model-a0e7"
    finally:
        state.close()


def test_remote_pin_with_effort_routes_to_remote(tmp_path: Path) -> None:
    remote = _make_backend(id="remote", kind="codex_auth_vault")
    local = _make_backend(id="local", kind="litellm_gateway")
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("auto")
    try:
        app = create_app(backends=[remote, local], operator_state=state)
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "callosum:remote/model-a0e7::high", "input": []},
            )
        assert response.status_code == 200, response.text
        assert response.json()["id"] == "resp-remote"
    finally:
        state.close()


def test_provider_defined_future_effort_routes_by_live_metadata(tmp_path: Path) -> None:
    """The parser accepts new effort names; the live cell grid is authority."""

    class _DynamicEffortRemote(InMemoryFakeBackend):
        @property
        def model_metadata(self):  # type: ignore[override]
            from callosum.cell_grid import ModelMetadata

            return {
                "model-a0d8": ModelMetadata(
                    slug="model-a0d8",
                    supported_in_api=True,
                    visibility="list",
                    priority=1,
                    supported_reasoning_levels=("adaptive-v2",),
                ),
            }

    remote = _DynamicEffortRemote(
        id="remote",
        advertised_models=frozenset({"model-a0d8"}),
        usage=UsageSnapshot(
            remaining_fraction=1.0,
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=0.0,
        ),
        health=HealthStatus(available=True, reason="ok"),
    )
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("auto")
    try:
        with TestClient(create_app(backends=[remote], operator_state=state)) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "callosum:remote/model-a0d8::adaptive-v2", "input": []},
            )
        assert response.status_code == 200, response.text
        assert response.json()["model"] == "model-a0d8"
    finally:
        state.close()


def test_local_pin_with_effort_routes_to_local(tmp_path: Path) -> None:
    """A local pin carrying an effort the model supports routes to local
    (symmetric to the remote-pin-with-effort case). The fake's local metadata
    advertises low/medium/high/xhigh, so :high has a live cell."""
    remote = _make_backend(id="remote", kind="codex_auth_vault")
    local = _make_backend(id="local", kind="litellm_gateway")
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("auto")
    try:
        app = create_app(backends=[remote, local], operator_state=state)
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "callosum:local/model-a0e7::high", "input": []},
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"] == "resp-local"
        assert body["model"] == "model-a0e7"
    finally:
        state.close()


def test_local_pin_with_unsupported_effort_returns_503(tmp_path: Path) -> None:
    """A local pin whose effort the model does NOT advertise parses fine but
    finds no live cell → clean actionable 503, not a 400. (:
    e.g. model-a0g2 advertises only ("default",), so :high has no cell.)"""

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

    remote = _make_backend(id="remote", kind="codex_auth_vault")
    local = _DefaultOnlyLocal(
        id="local",
        advertised_models=frozenset({"model-a0g2"}),
        health=HealthStatus(available=True, reason="ok"),
        usage=UsageSnapshot(
            remaining_fraction=1.0,
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=0.0,
        ),
    )
    local.kind = "litellm_gateway"
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("auto")
    try:
        app = create_app(backends=[remote, local], operator_state=state)
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "callosum:local/model-a0g2::high", "input": []},
            )
        assert response.status_code == 503, response.text
        assert "Retry-After" in response.headers
        detail = response.json()["detail"]
        assert "not available yet" in detail
        assert "model-a0g2" in detail
        assert "high" in detail
    finally:
        state.close()


def test_unsatisfiable_pin_returns_503(tmp_path: Path) -> None:
    """A remote pin while remote quota is exhausted leaves zero routable
    cells — clean 503 + Retry-After, not a 400 loop."""
    remote = _make_backend(id="remote", kind="codex_auth_vault", weekly_exhausted=True)
    local = _make_backend(id="local", kind="litellm_gateway")
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("auto")
    try:
        app = create_app(backends=[remote, local], operator_state=state)
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "callosum:remote/model-a0e7::high", "input": []},
            )
        assert response.status_code == 503, response.text
        assert "Retry-After" in response.headers
    finally:
        state.close()


def test_not_live_lane_returns_actionable_503(tmp_path: Path) -> None:
    """A well-formed pin to a model no backend serves (a declared-but-not-yet-
    live `/model` lane, e.g. an aspirational catalog entry) returns a clean
    503 that names the lane and the missing model — not the generic
    routing-mode exclusion, and not a 400 loop. (Requirement of the
    client-driven routing catalog; .)"""
    remote = _make_backend(id="remote", kind="codex_auth_vault")
    local = _make_backend(id="local", kind="litellm_gateway")
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("auto")
    try:
        app = create_app(backends=[remote, local], operator_state=state)
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "callosum:remote/model-a0b2::high",
                    "input": [],
                },
            )
        assert response.status_code == 503, response.text
        assert "Retry-After" in response.headers
        detail = response.json()["detail"]
        assert "not available yet" in detail
        assert "model-a0b2" in detail
    finally:
        state.close()


def test_hidden_model_excluded_from_auto_but_reachable_by_explicit_pin(tmp_path: Path) -> None:
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
        id="remote",
        advertised_models=frozenset({"model-a0e7", "codex-auto-review"}),
    )
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("auto")
    try:
        app = create_app(backends=[backend], operator_state=state)
        with TestClient(app) as client:
            auto_response = client.post(
                "/v1/responses",
                json={"model": "auto-learning", "input": []},
            )
            explicit_response = client.post(
                "/v1/responses",
                json={
                    "model": "callosum:remote/codex-auto-review::medium",
                    "input": [],
                },
            )
        assert auto_response.status_code == 200, auto_response.text
        assert auto_response.json()["model"] == "model-a0e7"
        assert explicit_response.status_code == 200, explicit_response.text
        assert explicit_response.json()["model"] == "codex-auto-review"
    finally:
        state.close()


def test_hidden_model_explicit_pin_with_unsupported_effort_returns_503(tmp_path: Path) -> None:
    class _HiddenReviewBackend(InMemoryFakeBackend):
        @property
        def model_metadata(self):  # type: ignore[override]
            from callosum.cell_grid import ModelMetadata

            return {
                "codex-auto-review": ModelMetadata(
                    slug="codex-auto-review",
                    supported_in_api=True,
                    visibility="hide",
                    priority=20,
                    supported_reasoning_levels=("medium",),
                ),
            }

    backend = _HiddenReviewBackend(
        id="remote",
        advertised_models=frozenset({"codex-auto-review"}),
    )
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("auto")
    try:
        app = create_app(backends=[backend], operator_state=state)
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={
                    "model": "callosum:remote/codex-auto-review::high",
                    "input": [],
                },
            )
        assert response.status_code == 503, response.text
        detail = response.json()["detail"]
        assert "not available yet" in detail
        assert "codex-auto-review" in detail
        assert "high" in detail
    finally:
        state.close()


def test_invalid_selectors_return_400(tmp_path: Path) -> None:
    remote = _make_backend(id="remote", kind="codex_auth_vault")
    local = _make_backend(id="local", kind="litellm_gateway")
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("auto")
    try:
        app = create_app(backends=[remote, local], operator_state=state)
        with TestClient(app) as client:
            # Effort vocabulary is metadata-driven; only malformed selector
            # structure/policy is rejected by the parser.
            for bad in ("callosum:offline", "callosum:unknown-strategy"):
                r = client.post("/v1/responses", json={"model": bad, "input": []})
                assert r.status_code == 400, f"{bad}: {r.text}"
    finally:
        state.close()
