from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import httpx
import pytest

from codex_proxy.auth_vault import AuthVault
from codex_proxy.errors import BackendError


def _b64url(data: dict[str, object]) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(data).encode("utf-8")).rstrip(b"=")
    return encoded.decode("ascii")


def _make_jwt(payload: dict[str, object]) -> str:
    header = _b64url({"alg": "none"})
    body = _b64url(payload)
    return f"{header}.{body}.sig"


def _write_auth_json(
    path: Path,
    *,
    access_token: str = "access-1",
    refresh_token: str = "refresh-1",
    id_token: str | None = None,
    account_id: str | None = "acct-1",
) -> None:
    tokens: dict[str, object] = {
        "access_token": access_token,
        "refresh_token": refresh_token,
    }
    if id_token is not None:
        tokens["id_token"] = id_token
    if account_id is not None:
        tokens["account_id"] = account_id
    path.write_text(json.dumps({"tokens": tokens, "last_refresh": "2024-01-01T00:00:00Z"}))


async def test_current_returns_cached_tokens_without_network(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    _write_auth_json(path)

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not hit network when token is not near expiry")

    vault = AuthVault(path=path, transport=httpx.MockTransport(handler))
    try:
        tokens = await vault.current(now=0.0)
        assert tokens.access_token == "access-1"
        assert tokens.refresh_token == "refresh-1"
        assert tokens.account_id == "acct-1"
    finally:
        await vault.aclose()


async def test_current_triggers_refresh_when_access_token_near_expiry(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    expiring_access = _make_jwt({"exp": 1000})
    _write_auth_json(
        path,
        access_token=expiring_access,
        id_token=_make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-jwt"}}),
    )

    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        new_access = _make_jwt({"exp": 2_000_000_000})
        return httpx.Response(
            200,
            json={
                "access_token": new_access,
                "refresh_token": "refresh-2",
                "id_token": _make_jwt(
                    {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-jwt-2"}}
                ),
            },
        )

    vault = AuthVault(
        path=path,
        transport=httpx.MockTransport(handler),
        refresh_url="https://auth.example.com/oauth/token",
    )
    try:
        tokens = await vault.current(now=1001)
        assert tokens.refresh_token == "refresh-2"
        assert tokens.access_token.startswith("eyJhbGciOiAibm9uZSJ9.")
        assert captured["url"] == "https://auth.example.com/oauth/token"
        body = captured["body"]
        assert isinstance(body, dict)
        assert body["grant_type"] == "refresh_token"
        assert body["refresh_token"] == "refresh-1"

        reloaded = json.loads(path.read_text())
        assert reloaded["tokens"]["access_token"] == tokens.access_token
        assert reloaded["tokens"]["refresh_token"] == "refresh-2"
    finally:
        await vault.aclose()


async def test_refresh_rejection_raises_auth_invalid(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    _write_auth_json(path, access_token=_make_jwt({"exp": 100}))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"code": "refresh_token_expired"}})

    vault = AuthVault(path=path, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(BackendError) as excinfo:
            await vault.current(now=time.time())
        assert excinfo.value.classification == "auth_invalid"
    finally:
        await vault.aclose()


async def test_missing_auth_json_raises_auth_invalid(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no network expected")

    vault = AuthVault(path=path, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(BackendError) as excinfo:
            await vault.current()
        assert excinfo.value.classification == "auth_invalid"
    finally:
        await vault.aclose()


async def test_force_refresh_rotates_tokens_on_disk(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    _write_auth_json(path)

    def handler(request: httpx.Request) -> httpx.Response:
        new_access = _make_jwt({"exp": 2_000_000_000})
        return httpx.Response(
            200,
            json={"access_token": new_access, "refresh_token": "refresh-rotated"},
        )

    vault = AuthVault(path=path, transport=httpx.MockTransport(handler))
    try:
        refreshed = await vault.force_refresh()
        assert refreshed.refresh_token == "refresh-rotated"
        disk = json.loads(path.read_text())
        assert disk["tokens"]["refresh_token"] == "refresh-rotated"
    finally:
        await vault.aclose()


async def test_current_reloads_when_disk_mtime_changes(tmp_path: Path) -> None:
    """Operator vault rotation (codex login + cp into vault, file copy, etc.)
    bumps auth.json mtime. Vault must reload from disk on next call without
    requiring a process restart.
    """
    path = tmp_path / "auth.json"
    _write_auth_json(path, access_token="cached-1", refresh_token="cached-1")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no network expected; this test exercises disk reload only")

    vault = AuthVault(path=path, transport=httpx.MockTransport(handler))
    try:
        # First call returns the original tokens.
        first = await vault.current(now=0.0)
        assert first.access_token == "cached-1"

        # Operator rewrites auth.json with new tokens. Bump mtime explicitly so
        # the test isn't sensitive to filesystem mtime granularity.
        _write_auth_json(path, access_token="rotated-2", refresh_token="rotated-2")
        st = path.stat()
        import os

        os.utime(path, (st.st_atime, st.st_mtime + 1.0))

        # Next call should return the rotated tokens — no force_refresh, no
        # network — purely a cache reload triggered by mtime change.
        second = await vault.current(now=0.0)
        assert second.access_token == "rotated-2"
        assert second.refresh_token == "rotated-2"
    finally:
        await vault.aclose()


async def test_account_id_extracted_from_id_token_when_absent_from_tokens_block(
    tmp_path: Path,
) -> None:
    path = tmp_path / "auth.json"
    id_token = _make_jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct-jwt-only"}})
    _write_auth_json(path, id_token=id_token, account_id=None)

    vault = AuthVault(path=path, transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    try:
        tokens = await vault.current(now=0.0)
        assert tokens.account_id == "acct-jwt-only"
    finally:
        await vault.aclose()
