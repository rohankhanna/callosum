"""Tests for the Dispatch-facing surface of the cost router:

- POST /control/refit-router triggers cost_router.fit() and reports state.
- GET /status includes a `router` block with v1 + v2 sample counts so
  operators can watch v2 ramp up without poking at SQLite.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from codex_proxy.app import create_app


def test_status_includes_router_block() -> None:
    client = TestClient(create_app())
    with client:
        resp = client.get("/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "router" in body
    r = body["router"]
    # is_ready may be False with no DB; that's fine — the shape must exist.
    assert "is_ready" in r
    assert "last_fit_ts" in r
    assert "last_fit_age_s" in r
    assert "v1_cells" in r and isinstance(r["v1_cells"], list)
    assert "v2_buckets_with_data" in r and isinstance(r["v2_buckets_with_data"], list)


def test_refit_router_endpoint_calls_fit_and_returns_summary() -> None:
    client = TestClient(create_app())
    with client:
        before = client.get("/status").json()["router"]["last_fit_ts"]
        # Wait a hair so the post-refit timestamp is strictly newer.
        time.sleep(0.01)
        resp = client.post("/control/refit-router")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "refit_at_ts" in body
    assert "is_ready" in body
    assert "v1_cell_count" in body
    assert "v2_bucket_count" in body
    # Strict monotonic-or-equal timestamp: a fresh fit must not regress.
    if before is not None:
        assert body["refit_at_ts"] >= before


def test_refit_router_endpoint_repeatable() -> None:
    """Multiple back-to-back refits should each succeed; nothing in
    cost_router.fit() should mutate state in a way that breaks the next call.
    """
    client = TestClient(create_app())
    with client:
        for _ in range(3):
            resp = client.post("/control/refit-router")
            assert resp.status_code == 200, resp.text
