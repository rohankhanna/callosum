"""Tests for the auth middleware's verbose 401 responses.

Auth failures used to be silent (no log) and uninformative (just
{"detail": "api key not found"}). After a key appeared to "invalidate
overnight" and the proxy had no record of which token was rejected, we
added: a logger.warning on every 401 + a response body carrying the
rejected key's prefix (never the secret) and the count of active keys,
so the next occurrence is diagnosable in one glance.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from codex_proxy.app import create_app
from codex_proxy.auth import AuthService
from codex_proxy.auth_db import AuthDB


def _app_with_auth(tmp_path: Path):
    svc = AuthService(AuthDB(tmp_path / "auth.sqlite"))
    user = svc.register(username="op", password="pw")
    issued = svc.create_api_key(user_id=user.id, label="hermes")
    app = create_app(auth_service=svc)
    return app, svc, issued.plaintext


def test_missing_bearer_returns_structured_401(tmp_path: Path) -> None:
    app, svc, _ = _app_with_auth(tmp_path)
    with TestClient(app) as client:
        r = client.get("/v1/models")  # no Authorization header
    assert r.status_code == 401
    body = r.json()
    assert body["reason"] == "missing_bearer"
    svc.db.close()


def test_unknown_key_401_includes_prefix_and_active_count(tmp_path: Path) -> None:
    app, svc, _ = _app_with_auth(tmp_path)
    with TestClient(app) as client:
        r = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer wrongkey12345abcdef"},
        )
    assert r.status_code == 401
    body = r.json()
    assert body["reason"] == "key_not_recognized"
    # Prefix is the first 8 chars of the REJECTED key (not a registered one).
    assert body["rejected_key_prefix"] == "wrongkey"
    # One active key is registered (the 'hermes' key we minted).
    assert body["active_keys_registered"] == 1
    # The secret itself never appears in the response.
    assert "wrongkey12345abcdef" not in r.text
    svc.db.close()


def test_valid_key_passes_middleware(tmp_path: Path) -> None:
    app, svc, plaintext = _app_with_auth(tmp_path)
    with TestClient(app) as client:
        r = client.get(
            "/v1/models", headers={"Authorization": f"Bearer {plaintext}"}
        )
    # 200 (models list) — the point is it's NOT 401; auth passed.
    assert r.status_code == 200
    svc.db.close()
