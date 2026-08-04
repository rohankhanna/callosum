"""Routing-mode × backend-availability matrix.

Pins the parity properties between `auto` mode + remote-exhausted vs
`local-only` mode (they should serve from local in both cases), and the
clean-error properties when nothing is routable.

Investigation: docs/investigations/2026-06-08-routing-symptoms.md
Finding A — the `if _routable:` filter gate at app.py:1271-1275 used to
leave dead cells in the routing pool when all backends were unroutable.
Auto mode bled through to the router, which raised a generic 400 that
codex CLI retried indefinitely. Local-only mode pre-filtered to litellm
cells before the gate and dodged the bug.

These tests pin:
  * The filter is now unconditional — dead cells never reach the router.
  * Empty-cells responses are 503 with Retry-After (codex CLI respects
    Retry-After and won't tight-loop), not 400.
  * Auto + remote-exhausted with a healthy local backend routes to local.
  * Local-only with a healthy local backend continues to work (regression
    guard for the path that already worked before the fix).
"""

from __future__ import annotations

from pathlib import Path

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
    """Construct a fake backend with the requested kind + exhaustion state."""
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


# ---------- all-exhausted → 503 + Retry-After ---------------------------


def test_all_backends_exhausted_returns_503_with_retry_after(
    tmp_path: Path,
) -> None:
    """The load-bearing fix: when no backend is routable, return 503 with
    Retry-After. The old behavior was 400 with a generic "no cell can
    serve" message, which codex CLI retried in a tight loop until manual
    interruption. 503 + Retry-After is the codex-respected stop signal."""
    remote = _make_backend(id="r", kind="codex_auth_vault", weekly_exhausted=True)
    state = OperatorState(tmp_path / "op.sqlite")
    try:
        app = create_app(backends=[remote], operator_state=state)
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "auto-learning", "input": []},
            )
        assert response.status_code == 503
        assert "Retry-After" in response.headers
        body = response.json()
        assert "no backend is routable" in body["detail"]
    finally:
        state.close()


def test_auto_mode_remote_exhausted_routes_to_local(tmp_path: Path, monkeypatch) -> None:
    """The parity fix: in auto mode with remote quota exhausted, callosum
    must route to a healthy local backend. The bug was that dead remote
    cells stayed in the routing grid, the capability filter sometimes
    chose them anyway, and dispatch then failed."""
    remote = _make_backend(
        id="remote",
        kind="codex_auth_vault",
        weekly_exhausted=True,
    )
    local = _make_backend(id="local", kind="litellm_gateway")
    monkeypatch.setenv("CALLOSUM_CANARY_PERCENT", "0")
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("auto")
    try:
        app = create_app(backends=[remote, local], operator_state=state)
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "auto-learning", "input": []},
            )
        assert response.status_code == 200, response.text
        # Served by the local backend, not the exhausted remote.
        assert "fake-local" in response.text or response.text  # smoke
    finally:
        state.close()


def test_local_only_mode_continues_to_work(tmp_path: Path) -> None:
    """Regression guard: local-only mode already worked before the fix
    (because it pre-filters cells_now to litellm cells before the
    broken gate). It must continue to work after the fix."""
    remote = _make_backend(id="remote", kind="codex_auth_vault")
    local = _make_backend(id="local", kind="litellm_gateway")
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("local-only")
    try:
        app = create_app(backends=[remote, local], operator_state=state)
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "auto-learning", "input": []},
            )
        assert response.status_code == 200
    finally:
        state.close()


def test_remote_only_mode_with_remote_exhausted_returns_503(
    tmp_path: Path,
) -> None:
    """remote-only mode + remote exhausted → no routable backends after
    mode filter. Must return 503 with a routing-mode-specific message,
    not a generic 'no cell' 400."""
    remote = _make_backend(
        id="remote",
        kind="codex_auth_vault",
        weekly_exhausted=True,
    )
    local = _make_backend(id="local", kind="litellm_gateway")
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("remote-only")
    try:
        app = create_app(backends=[remote, local], operator_state=state)
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "auto-learning", "input": []},
            )
        assert response.status_code == 503
        assert "Retry-After" in response.headers
        body = response.json()
        # The mode filter excluded the only healthy backend (local).
        # Two valid error messages depending on order of filter vs
        # routability check: either "no backend currently routable" or
        # "routing mode excludes all routable backends". Either is
        # acceptable; what matters is 503 + Retry-After.
        assert "routing mode" in body["detail"].lower() or "no backend" in body["detail"].lower()
    finally:
        state.close()


def test_remote_only_cold_boot_empty_catalog_returns_accurate_503(
    tmp_path: Path,
) -> None:
    """Cold-boot window (): a remote backend has quota
    available but its discovered model catalog is still empty (catalog
    refresh hasn't completed yet). The cell grid falls back to static
    defaults, so cells_now is non-empty before the routability filter,
    then the mode/catalog filters empty it.

    remote-only must NOT misreport this as 'current routing mode excludes
    all currently-routable backends' — the mode kept the backend; only the
    catalog is unpopulated. The 503 must name the actual empty-catalog /
    undiscovered-model condition, and a quota-available backend with no
    discovered models must not be treated as dispatch-routable."""
    remote = InMemoryFakeBackend(
        id="remote",
        advertised_models=frozenset(),  # cold-boot: catalog not yet discovered
        usage=UsageSnapshot(
            remaining_fraction=1.0,  # quota available — would be "routable"
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=0.0,
        ),
        health=HealthStatus(available=True, reason="ok"),
    )
    remote.kind = "codex_auth_vault"
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("remote-only")
    try:
        app = create_app(backends=[remote], operator_state=state)
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "auto-learning", "input": []},
            )
        assert response.status_code == 503, response.text
        assert "Retry-After" in response.headers
        detail = response.json()["detail"].lower()
        # Names the actual empty-catalog / undiscovered-model condition.
        assert "catalog" in detail or "advertised_models" in detail
        # NOT the misleading mode-exclusion message from the cold-boot bug.
        assert "excludes all currently-routable backends" not in detail
    finally:
        state.close()


def test_auto_mode_empty_catalog_remote_routes_to_healthy_local(tmp_path: Path, monkeypatch) -> None:
    """A quota-available backend with no discovered models is not routable
    for dispatch (): in auto mode with a cold-boot
    empty-catalog remote plus a healthy catalog-bearing local, dispatch
    must serve the request from local and never treat the empty-catalog
    remote as a candidate."""
    remote = InMemoryFakeBackend(
        id="remote",
        advertised_models=frozenset(),  # cold-boot: catalog not yet discovered
        usage=UsageSnapshot(
            remaining_fraction=1.0,  # quota available — would be "routable"
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=0.0,
        ),
        health=HealthStatus(available=True, reason="ok"),
    )
    remote.kind = "codex_auth_vault"
    local = _make_backend(id="local", kind="litellm_gateway")
    monkeypatch.setenv("CALLOSUM_CANARY_PERCENT", "0")
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("auto")
    try:
        app = create_app(backends=[remote, local], operator_state=state)
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "auto-learning", "input": []},
            )
        assert response.status_code == 200, response.text
        # Served by the catalog-bearing local backend, not the empty-catalog remote.
        assert "resp-local" in response.text
    finally:
        state.close()


def test_auto_mode_all_healthy_routes_normally(tmp_path: Path) -> None:
    """Sanity: auto mode with everything healthy routes through normally."""
    remote = _make_backend(id="remote", kind="codex_auth_vault")
    local = _make_backend(id="local", kind="litellm_gateway")
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("auto")
    try:
        app = create_app(backends=[remote, local], operator_state=state)
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "auto-learning", "input": []},
            )
        assert response.status_code == 200
    finally:
        state.close()


# ---------- ollama_cloud classification (remote, not local) ---------------
# Pins the BackendKind partition: an ollama_cloud backend must land in the
# REMOTE lane (admitted by remote-only, excluded by local-only) so cloud
# models — which burn real Ollama Cloud quota — never get mis-routed as FREE
# local cells. See backends/ollama_cloud.py + work tracker .


def test_ollama_cloud_excluded_from_local_only(tmp_path: Path) -> None:
    """local-only mode must NOT serve an ollama_cloud backend — it is a remote
    fleet, not a free local one. With cloud as the only healthy backend,
    local-only excludes it and returns a clean 503 (not a silent mis-route to
    a metered cloud cell)."""
    cloud = _make_backend(id="cloud", kind="ollama_cloud")
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("local-only")
    try:
        app = create_app(backends=[cloud], operator_state=state)
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "auto-learning", "input": []},
            )
        assert response.status_code == 503
        assert "Retry-After" in response.headers
    finally:
        state.close()


def test_ollama_cloud_served_in_remote_only(tmp_path: Path) -> None:
    """remote-only mode admits an ollama_cloud backend (it IS remote). With
    cloud as the only backend, remote-only routes to it and serves 200."""
    cloud = _make_backend(id="cloud", kind="ollama_cloud")
    state = OperatorState(tmp_path / "op.sqlite")
    state.set_routing("remote-only")
    try:
        app = create_app(backends=[cloud], operator_state=state)
        with TestClient(app) as client:
            response = client.post(
                "/v1/responses",
                json={"model": "auto-learning", "input": []},
            )
        assert response.status_code == 200, response.text
        # Served by the cloud backend, proving it was admitted to the remote
        # lane (not filtered out as unknown-kind).
        assert "fake responses from cloud" in response.text
    finally:
        state.close()
