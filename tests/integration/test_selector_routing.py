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
                json={"model": "callosum:remote/model-a0e7:high", "input": []},
            )
        assert response.status_code == 200, response.text
        assert response.json()["id"] == "resp-remote"
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
                json={"model": "callosum:remote/model-a0e7:high", "input": []},
            )
        assert response.status_code == 503, response.text
        assert "Retry-After" in response.headers
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
            for bad in ("callosum:offline", "callosum:local/model-a0e7:high"):
                r = client.post("/v1/responses", json={"model": bad, "input": []})
                assert r.status_code == 400, f"{bad}: {r.text}"
    finally:
        state.close()
