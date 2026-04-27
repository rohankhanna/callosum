from __future__ import annotations

import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from codex_proxy.app import create_app
from codex_proxy.auth import AuthService
from codex_proxy.auth_db import AuthDB
from codex_proxy.auth_vault import AuthVault
from codex_proxy.backend import UsageSnapshot
from codex_proxy.backends.codex_auth_vault import CodexAuthVaultBackend


def _write_auth_json(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "access-test",
                    "refresh_token": "refresh-test",
                    "account_id": "acct-test",
                }
            }
        )
    )


def _healthy_sse() -> bytes:
    """Build a minimal SSE blob with the events the diagnostic looks for."""
    events = [
        ("response.created", {"type": "response.created", "id": "r1"}),
        (
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "ok"},
        ),
        (
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": "r1",
                    "model": "model-a0e7",
                    "usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
                },
            },
        ),
    ]
    parts = [f"event: {n}\ndata: {json.dumps(p)}\n\n" for n, p in events]
    return "".join(parts).encode()


def _healthy_codex_headers() -> dict[str, str]:
    return {
        "content-type": "text/event-stream",
        "x-codex-plan-type": "plus",
        "x-codex-active-limit": "premium",
        "x-codex-primary-used-percent": "1",
        "x-codex-secondary-used-percent": "10",
        "x-codex-primary-window-minutes": "300",
        "x-codex-secondary-window-minutes": "10080",
        "x-codex-primary-reset-after-seconds": "1000",
        "x-codex-secondary-reset-after-seconds": "100000",
        "x-codex-primary-reset-at": "1777000000",
        "x-codex-secondary-reset-at": "1777500000",
        "x-codex-primary-over-secondary-limit-percent": "0",
        "x-codex-credits-has-credits": "False",
        "x-codex-credits-unlimited": "False",
    }


def _make_backend(
    *,
    tmp_path: Path,
    handler: httpx.MockTransport,
    backend_id: str = "primary",
) -> CodexAuthVaultBackend:
    auth_path = tmp_path / f"{backend_id}-auth.json"
    _write_auth_json(auth_path)

    def vault_handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("vault transport should not be reached during diagnose")

    vault = AuthVault(path=auth_path, transport=httpx.MockTransport(vault_handler))
    return CodexAuthVaultBackend(
        id=backend_id,
        vault=vault,
        advertised_models=frozenset({"model-a0e7"}),
        transport=handler,
    )


def test_diagnose_green_path_reports_all_checks_pass(tmp_path: Path) -> None:
    def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers=_healthy_codex_headers(),
            content=_healthy_sse(),
        )

    backend = _make_backend(tmp_path=tmp_path, handler=httpx.MockTransport(upstream))
    with TestClient(create_app(backends=[backend])) as client:
        r = client.get("/diagnose/upstream")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    [bd] = body["backends"]
    assert bd["id"] == "primary"
    assert bd["ok"] is True
    assert bd["failed_checks"] == []
    # Every named check passed.
    assert all(bd["checks"].values())


def test_diagnose_flags_missing_quota_headers(tmp_path: Path) -> None:
    # Upstream returns 200 + a valid SSE stream BUT no x-codex-* headers.
    # That's the signal that the proxy's quota-tracking model is broken.
    def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=_healthy_sse(),
        )

    backend = _make_backend(tmp_path=tmp_path, handler=httpx.MockTransport(upstream))
    with TestClient(create_app(backends=[backend])) as client:
        body = client.get("/diagnose/upstream").json()
    assert body["ok"] is False
    [bd] = body["backends"]
    assert bd["ok"] is False
    assert "quota_headers_present" in bd["failed_checks"]
    assert "five_hourly_used_percent_present" in bd["failed_checks"]


def test_diagnose_flags_missing_response_completed_event(tmp_path: Path) -> None:
    # Headers are healthy but the SSE stream never produces response.completed.
    def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers=_healthy_codex_headers(),
            content=b'event: response.created\ndata: {"type":"response.created"}\n\n',
        )

    backend = _make_backend(tmp_path=tmp_path, handler=httpx.MockTransport(upstream))
    with TestClient(create_app(backends=[backend])) as client:
        body = client.get("/diagnose/upstream").json()
    assert body["ok"] is False
    [bd] = body["backends"]
    assert "response_completed_event_present" in bd["failed_checks"]
    assert "usage_block_present" in bd["failed_checks"]


def test_diagnose_reports_upstream_400_with_classification(tmp_path: Path) -> None:
    # Simulates "Codex changed the model name" — upstream rejects the request.
    def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=400,
            json={"detail": "model not supported"},
        )

    backend = _make_backend(tmp_path=tmp_path, handler=httpx.MockTransport(upstream))
    with TestClient(create_app(backends=[backend])) as client:
        body = client.get("/diagnose/upstream").json()
    assert body["ok"] is False
    [bd] = body["backends"]
    assert bd["ok"] is False
    assert bd["stage"] == "upstream"
    assert bd["status_code"] == 400


def test_diagnose_skips_backend_in_cooldown(tmp_path: Path) -> None:
    # A cooled-down backend gets reported as skipped, not probed. The aggregate
    # `ok` for the response stays True if no other backend failed.
    import time as _time

    def upstream(_: httpx.Request) -> httpx.Response:
        raise AssertionError("cooled-down backend must not be probed")

    backend = _make_backend(tmp_path=tmp_path, handler=httpx.MockTransport(upstream))
    backend._usage = UsageSnapshot(
        remaining_fraction=None,
        cooldown_until_ts=_time.time() + 600,
        weekly_exhausted=False,
        probed_at_ts=_time.time(),
    )
    with TestClient(create_app(backends=[backend])) as client:
        body = client.get("/diagnose/upstream").json()
    assert body["ok"] is True  # nothing probed = nothing failed
    [bd] = body["backends"]
    assert bd["skipped"] is True
    assert bd["stage"] == "cooldown"


def test_diagnose_aggregates_across_multiple_backends(tmp_path: Path) -> None:
    def healthy(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers=_healthy_codex_headers(),
            content=_healthy_sse(),
        )

    def broken(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=500, json={"detail": "boom"})

    a = _make_backend(tmp_path=tmp_path, handler=httpx.MockTransport(healthy), backend_id="alpha")
    b = _make_backend(tmp_path=tmp_path, handler=httpx.MockTransport(broken), backend_id="beta")
    with TestClient(create_app(backends=[a, b])) as client:
        body = client.get("/diagnose/upstream").json()
    assert body["ok"] is False  # one bad apple flips the aggregate
    statuses = {b["id"]: b["ok"] for b in body["backends"]}
    assert statuses == {"alpha": True, "beta": False}


def test_diagnose_requires_bearer_when_auth_enabled(tmp_path: Path) -> None:
    auth_service = AuthService(AuthDB(tmp_path / "auth.sqlite"))

    def upstream(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers=_healthy_codex_headers(),
            content=_healthy_sse(),
        )

    backend = _make_backend(tmp_path=tmp_path, handler=httpx.MockTransport(upstream))
    with TestClient(create_app(backends=[backend], auth_service=auth_service)) as client:
        anon = client.get("/diagnose/upstream")
        assert anon.status_code == 401

        client.post("/auth/register", json={"username": "alice", "password": "x"})
        sess = client.post("/auth/login", json={"username": "alice", "password": "x"}).json()[
            "session_token"
        ]
        key = client.post(
            "/auth/keys",
            json={"label": "diag-cron"},
            headers={"Authorization": f"Bearer {sess}"},
        ).json()["api_key"]
        ok = client.get("/diagnose/upstream", headers={"Authorization": f"Bearer {key}"})
        assert ok.status_code == 200
        assert ok.json()["ok"] is True
