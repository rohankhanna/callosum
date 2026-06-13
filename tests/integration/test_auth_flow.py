from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from callosum.app import create_app
from callosum.auth import AuthService
from callosum.auth_db import AuthDB
from callosum.fakes import InMemoryFakeBackend
from callosum.usage_log import UsageLog


def _service(tmp_path: Path) -> AuthService:
    return AuthService(AuthDB(tmp_path / "auth.sqlite"))


def _backend() -> InMemoryFakeBackend:
    return InMemoryFakeBackend(id="primary", advertised_models=frozenset({"model-a0e7"}))


@asynccontextmanager
async def _client(**kwargs: object) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=create_app(**kwargs))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.mark.asyncio
async def test_register_login_create_key_call_v1_logs_attribution(tmp_path: Path) -> None:
    auth_service = _service(tmp_path)
    log = UsageLog(tmp_path / "u.sqlite")
    async with _client(backends=[_backend()], usage_log=log, auth_service=auth_service) as client:
        # 1. Register
        r = await client.post("/auth/register", json={"username": "alice", "password": "x"})
        assert r.status_code == 201
        user_id = r.json()["user_id"]

        # 2. Login -> session token
        r = await client.post("/auth/login", json={"username": "alice", "password": "x"})
        assert r.status_code == 200
        session_token = r.json()["session_token"]

        # 3. Mint API key with the session
        r = await client.post(
            "/auth/keys",
            json={"label": "laptop"},
            headers={"Authorization": f"Bearer {session_token}"},
        )
        assert r.status_code == 201
        key_data = r.json()
        api_key = key_data["api_key"]
        api_key_id = key_data["id"]
        assert key_data["label"] == "laptop"

        # 4. Call /v1/* with the api key
        r = await client.post(
            "/v1/chat/completions",
            json={"model": "model-a0e7", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert r.status_code == 200

    # 5. Usage log should have the user_id + api_key_id columns populated.
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    row = conn.execute("SELECT user_id, api_key_id, status FROM requests").fetchone()
    assert row == (user_id, api_key_id, 200)


@pytest.mark.asyncio
async def test_v1_without_bearer_returns_401(tmp_path: Path) -> None:
    auth_service = _service(tmp_path)
    async with _client(backends=[_backend()], auth_service=auth_service) as client:
        r = await client.post(
            "/v1/chat/completions",
            json={"model": "model-a0e7", "messages": []},
        )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_v1_with_unknown_bearer_returns_401(tmp_path: Path) -> None:
    auth_service = _service(tmp_path)
    async with _client(backends=[_backend()], auth_service=auth_service) as client:
        r = await client.post(
            "/v1/chat/completions",
            json={"model": "model-a0e7", "messages": []},
            headers={"Authorization": "Bearer ghost-key"},
        )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_v1_with_revoked_bearer_returns_401(tmp_path: Path) -> None:
    auth_service = _service(tmp_path)
    async with _client(backends=[_backend()], auth_service=auth_service) as client:
        await client.post("/auth/register", json={"username": "alice", "password": "x"})
        login = (await client.post("/auth/login", json={"username": "alice", "password": "x"})).json()
        session_token = login["session_token"]
        key = (
            await client.post(
                "/auth/keys",
                json={},
                headers={"Authorization": f"Bearer {session_token}"},
            )
        ).json()
        # Revoke and then try to use it.
        await client.delete(
            f"/auth/keys/{key['id']}",
            headers={"Authorization": f"Bearer {session_token}"},
        )
        r = await client.post(
            "/v1/chat/completions",
            json={"model": "model-a0e7", "messages": []},
            headers={"Authorization": f"Bearer {key['api_key']}"},
        )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_no_auth_service_leaves_v1_open(tmp_path: Path) -> None:
    # Regression: with auth_service=None (single-operator mode), /v1/* must
    # not require a bearer.
    async with _client(backends=[_backend()], auth_service=None) as client:
        r = await client.post(
            "/v1/chat/completions",
            json={"model": "model-a0e7", "messages": []},
        )
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_auth_routes_unavailable_when_auth_disabled(tmp_path: Path) -> None:
    async with _client(backends=[_backend()], auth_service=None) as client:
        r = await client.post("/auth/register", json={"username": "x", "password": "y"})
    # No auth service -> /auth/* not installed -> FastAPI returns 405/404.
    assert r.status_code in (404, 405)


@pytest.mark.asyncio
async def test_register_rejects_duplicate_username(tmp_path: Path) -> None:
    auth_service = _service(tmp_path)
    async with _client(backends=[_backend()], auth_service=auth_service) as client:
        r1 = await client.post("/auth/register", json={"username": "alice", "password": "x"})
        assert r1.status_code == 201
        r2 = await client.post("/auth/register", json={"username": "alice", "password": "y"})
        assert r2.status_code == 400


@pytest.mark.asyncio
async def test_login_with_wrong_password_returns_401(tmp_path: Path) -> None:
    auth_service = _service(tmp_path)
    async with _client(backends=[_backend()], auth_service=auth_service) as client:
        await client.post("/auth/register", json={"username": "alice", "password": "right"})
        r = await client.post("/auth/login", json={"username": "alice", "password": "wrong"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_create_key_requires_session_token(tmp_path: Path) -> None:
    auth_service = _service(tmp_path)
    async with _client(backends=[_backend()], auth_service=auth_service) as client:
        r = await client.post("/auth/keys", json={})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_list_keys_returns_only_callers_keys(tmp_path: Path) -> None:
    auth_service = _service(tmp_path)
    async with _client(backends=[_backend()], auth_service=auth_service) as client:
        for username in ("alice", "bob"):
            await client.post("/auth/register", json={"username": username, "password": "x"})
        alice_session = (await client.post("/auth/login", json={"username": "alice", "password": "x"})).json()[
            "session_token"
        ]
        bob_session = (await client.post("/auth/login", json={"username": "bob", "password": "x"})).json()[
            "session_token"
        ]
        await client.post(
            "/auth/keys",
            json={"label": "alice-key-1"},
            headers={"Authorization": f"Bearer {alice_session}"},
        )
        await client.post(
            "/auth/keys",
            json={"label": "alice-key-2"},
            headers={"Authorization": f"Bearer {alice_session}"},
        )
        await client.post(
            "/auth/keys",
            json={"label": "bob-key"},
            headers={"Authorization": f"Bearer {bob_session}"},
        )
        alice_keys = (await client.get("/auth/keys", headers={"Authorization": f"Bearer {alice_session}"})).json()[
            "keys"
        ]
        bob_keys = (await client.get("/auth/keys", headers={"Authorization": f"Bearer {bob_session}"})).json()["keys"]
    assert {k["label"] for k in alice_keys} == {"alice-key-1", "alice-key-2"}
    assert {k["label"] for k in bob_keys} == {"bob-key"}
