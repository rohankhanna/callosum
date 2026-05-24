from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx

from callosum.errors import BackendError

DEFAULT_REFRESH_URL = "https://auth.openai.com/oauth/token"
DEFAULT_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

REFRESH_URL_ENV = "CODEX_REFRESH_TOKEN_URL_OVERRIDE"
REFRESH_CLIENT_ID_ENV = "CODEX_REFRESH_CLIENT_ID_OVERRIDE"

REFRESH_SAFETY_WINDOW_S = 5 * 60


@dataclass(frozen=True, slots=True)
class AuthTokens:
    access_token: str
    refresh_token: str
    id_token: str | None
    account_id: str | None
    access_token_exp: int | None


class AuthVault:
    """Loads, refreshes, and persists a single account's auth.json.

    Safe for concurrent use: only one refresh runs at a time; cached tokens
    are returned until they are within the safety window of expiring, at which
    point a refresh is triggered.
    """

    def __init__(
        self,
        *,
        path: Path,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        refresh_url: str | None = None,
        client_id: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._path = path
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(transport=transport, timeout=timeout_s)
            self._owns_client = True
        self._refresh_url = refresh_url or os.environ.get(REFRESH_URL_ENV) or DEFAULT_REFRESH_URL
        self._client_id = client_id or os.environ.get(REFRESH_CLIENT_ID_ENV) or DEFAULT_CLIENT_ID
        self._lock = asyncio.Lock()
        self._cached, self._cached_mtime = _read_tokens_with_mtime(path)

    @property
    def path(self) -> Path:
        return self._path

    def peek(self) -> AuthTokens | None:
        """Return cached tokens without refreshing or reading disk."""
        return self._cached

    def _reload_if_changed_on_disk(self) -> None:
        """If auth.json's mtime is newer than what we last read, refresh the
        cache from disk. Lets operator vault rotations (codex login, file copy)
        propagate to a running proxy without a restart.
        """
        try:
            current_mtime = self._path.stat().st_mtime
        except OSError:
            return
        if self._cached_mtime is None or current_mtime > self._cached_mtime:
            tokens, mtime = _read_tokens_with_mtime(self._path)
            if tokens is not None:
                self._cached = tokens
                self._cached_mtime = mtime

    async def current(self, *, now: float | None = None) -> AuthTokens:
        """Return tokens, refreshing if close to expiry."""
        self._reload_if_changed_on_disk()
        tokens = self._cached
        if tokens is None:
            tokens, self._cached_mtime = _read_tokens_with_mtime(self._path)
        if tokens is None:
            raise BackendError(
                classification="auth_invalid",
                status_code=401,
                message=f"auth.json at {self._path} is missing or malformed",
            )
        self._cached = tokens
        ts = now if now is not None else time.time()
        if _should_refresh(tokens, ts):
            tokens = await self._refresh_locked(tokens)
        return tokens

    async def force_refresh(self) -> AuthTokens:
        self._reload_if_changed_on_disk()
        tokens = self._cached
        if tokens is None:
            tokens, self._cached_mtime = _read_tokens_with_mtime(self._path)
        if tokens is None:
            raise BackendError(
                classification="auth_invalid",
                status_code=401,
                message=f"auth.json at {self._path} is missing or malformed",
            )
        return await self._refresh_locked(tokens)

    async def _refresh_locked(self, tokens: AuthTokens) -> AuthTokens:
        async with self._lock:
            current = self._cached
            if current is not None and current is not tokens:
                return current
            refreshed = await self._request_refresh(tokens.refresh_token)
            _write_tokens_to_disk(self._path, refreshed)
            self._cached = refreshed
            try:
                self._cached_mtime = self._path.stat().st_mtime
            except OSError:
                self._cached_mtime = None
            return refreshed

    async def _request_refresh(self, refresh_token: str) -> AuthTokens:
        body = {
            "client_id": self._client_id,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        try:
            response = await self._client.post(
                self._refresh_url,
                json=body,
                headers={"Content-Type": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise BackendError(
                classification="transient",
                message=f"refresh transport error: {exc}",
            ) from exc
        if response.status_code == 401 or response.status_code == 400:
            raise BackendError(
                classification="auth_invalid",
                status_code=response.status_code,
                message=f"refresh rejected: {response.status_code} {response.text[:200]}",
            )
        if response.status_code >= 500:
            raise BackendError(
                classification="transient",
                status_code=response.status_code,
                message=f"refresh upstream error: {response.status_code}",
            )
        if response.status_code >= 400:
            raise BackendError(
                classification="auth_invalid",
                status_code=response.status_code,
                message=f"refresh failed: {response.status_code} {response.text[:200]}",
            )
        payload = cast(dict[str, Any], response.json())
        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise BackendError(
                classification="auth_invalid",
                status_code=response.status_code,
                message="refresh response missing access_token",
            )
        new_refresh = payload.get("refresh_token")
        if not isinstance(new_refresh, str) or not new_refresh:
            new_refresh = refresh_token
        id_token = payload.get("id_token")
        if not isinstance(id_token, str):
            id_token = None
        account_id = _account_id_from_id_token(id_token) or self._cached_account_id()
        exp = _jwt_expiration(access_token)
        return AuthTokens(
            access_token=access_token,
            refresh_token=new_refresh,
            id_token=id_token,
            account_id=account_id,
            access_token_exp=exp,
        )

    def _cached_account_id(self) -> str | None:
        return self._cached.account_id if self._cached is not None else None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _should_refresh(tokens: AuthTokens, now: float) -> bool:
    if tokens.access_token_exp is None:
        return False
    return tokens.access_token_exp - now <= REFRESH_SAFETY_WINDOW_S


def _read_tokens_with_mtime(path: Path) -> tuple[AuthTokens | None, float | None]:
    """Read tokens AND capture mtime atomically — caller uses mtime to decide
    whether the cache needs refresh on subsequent calls.
    """
    try:
        mtime: float | None = path.stat().st_mtime
    except OSError:
        mtime = None
    return _read_tokens_from_disk(path), mtime


def _read_tokens_from_disk(path: Path) -> AuthTokens | None:
    try:
        raw = path.read_text()
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        return None
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    if not isinstance(access_token, str) or not access_token:
        return None
    if not isinstance(refresh_token, str) or not refresh_token:
        return None
    id_token_raw = tokens.get("id_token")
    id_token = id_token_raw if isinstance(id_token_raw, str) else None
    account_id_raw = tokens.get("account_id")
    account_id = account_id_raw if isinstance(account_id_raw, str) else None
    if account_id is None and id_token is not None:
        account_id = _account_id_from_id_token(id_token)
    exp = _jwt_expiration(access_token)
    return AuthTokens(
        access_token=access_token,
        refresh_token=refresh_token,
        id_token=id_token,
        account_id=account_id,
        access_token_exp=exp,
    )


def _write_tokens_to_disk(path: Path, tokens: AuthTokens) -> None:
    existing: dict[str, Any] = {}
    try:
        raw = path.read_text()
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            existing = loaded
    except (OSError, json.JSONDecodeError):
        existing = {}
    tokens_block = existing.get("tokens")
    if not isinstance(tokens_block, dict):
        tokens_block = {}
    tokens_block["access_token"] = tokens.access_token
    tokens_block["refresh_token"] = tokens.refresh_token
    if tokens.id_token is not None:
        tokens_block["id_token"] = tokens.id_token
    if tokens.account_id is not None:
        tokens_block["account_id"] = tokens.account_id
    existing["tokens"] = tokens_block
    existing["last_refresh"] = _format_timestamp(time.time())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(existing, indent=2))
    with contextlib.suppress(OSError):
        os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _format_timestamp(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _jwt_expiration(token: str) -> int | None:
    claims = _decode_jwt_claims(token)
    if claims is None:
        return None
    exp = claims.get("exp")
    if isinstance(exp, int):
        return exp
    if isinstance(exp, float):
        return int(exp)
    return None


def _account_id_from_id_token(id_token: str | None) -> str | None:
    if id_token is None:
        return None
    claims = _decode_jwt_claims(id_token)
    if claims is None:
        return None
    auth_claims = claims.get("https://api.openai.com/auth")
    if isinstance(auth_claims, dict):
        account_id = auth_claims.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    return None


def _decode_jwt_claims(token: str) -> dict[str, Any] | None:
    parts = token.split(".")
    if len(parts) < 2:
        return None
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
        claims = json.loads(decoded)
    except (ValueError, json.JSONDecodeError):
        return None
    if isinstance(claims, dict):
        return cast(dict[str, Any], claims)
    return None
