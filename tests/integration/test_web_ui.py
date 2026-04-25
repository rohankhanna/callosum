from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from codex_proxy.app import create_app
from codex_proxy.auth import AuthService
from codex_proxy.auth_db import AuthDB
from codex_proxy.fakes import InMemoryFakeBackend


def _service(tmp_path: Path) -> AuthService:
    return AuthService(AuthDB(tmp_path / "auth.sqlite"))


def _backend() -> InMemoryFakeBackend:
    return InMemoryFakeBackend(id="primary", advertised_models=frozenset({"model-a0e7"}))


def test_ui_served_when_auth_enabled(tmp_path: Path) -> None:
    auth = _service(tmp_path)
    with TestClient(create_app(backends=[_backend()], auth_service=auth)) as client:
        r = client.get("/ui/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    # Spot-check the page actually carries the JS that talks to /auth/*.
    assert "/auth/login" in body
    assert "/auth/register" in body
    assert "/auth/keys" in body
    assert "/auth/logout" in body
    # And the user-visible affordance exists.
    assert "Mint key" in body or "Mint a new key" in body


def test_ui_root_without_trailing_slash_also_works(tmp_path: Path) -> None:
    auth = _service(tmp_path)
    with TestClient(create_app(backends=[_backend()], auth_service=auth)) as client:
        r = client.get("/ui")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")


def test_ui_404_when_auth_disabled(tmp_path: Path) -> None:
    # In single-operator mode there is nothing for the UI to do — the auth
    # routes don't exist, so the UI is not mounted either.
    with TestClient(create_app(backends=[_backend()], auth_service=None)) as client:
        r = client.get("/ui/")
    assert r.status_code == 404


def test_ui_does_not_require_bearer(tmp_path: Path) -> None:
    # /ui/ has to be reachable BEFORE the user has a key, otherwise nobody
    # could ever bootstrap. Confirm the bearer middleware does not gate it.
    auth = _service(tmp_path)
    with TestClient(create_app(backends=[_backend()], auth_service=auth)) as client:
        # No Authorization header — must still serve the page.
        r = client.get("/ui/")
    assert r.status_code == 200
