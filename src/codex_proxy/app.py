from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from codex_proxy import __version__
from codex_proxy.auth import (
    ApiKeyInvalidError,
    AuthService,
    InvalidCredentialsError,
    SessionInvalidError,
)
from codex_proxy.auth_db import ApiKey, Session
from codex_proxy.backend import Backend, CallHandle
from codex_proxy.errors import RETRYABLE, BackendError, ErrorClass
from codex_proxy.selector import select
from codex_proxy.session import SessionRegistry
from codex_proxy.usage_log import UsageLog, UsageLogEntry

# Clients opt into sticky routing by sending this header. When absent, every
# request is a fresh selection. There is no server-side "session mode" knob.
SESSION_HEADER = "x-codex-session-id"

NonstreamCall = Callable[[Backend, dict[str, Any], CallHandle], Awaitable[dict[str, Any]]]
StreamCall = Callable[[Backend, dict[str, Any], CallHandle], AsyncIterator[bytes]]

_EXHAUSTED_STATUS: dict[ErrorClass, int] = {
    "auth_invalid": 502,
    "unknown_model": 400,
    "rate_limited": 429,
    "transient": 502,
    "client_error": 400,
}


class PinState:
    """Process-wide backend pin. Thread-safety not needed under single-loop uvicorn."""

    def __init__(self) -> None:
        self._pinned: str | None = None

    def get(self) -> str | None:
        return self._pinned

    def set(self, backend_id: str) -> None:
        self._pinned = backend_id

    def clear(self) -> None:
        self._pinned = None


def create_app(
    *,
    backends: Sequence[Backend] = (),
    sessions: SessionRegistry | None = None,
    usage_log: UsageLog | None = None,
    auth_service: AuthService | None = None,
) -> FastAPI:
    backends_list: list[Backend] = list(backends)
    pin_state = PinState()
    session_registry = sessions if sessions is not None else SessionRegistry()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            for backend in backends_list:
                await backend.aclose()
            if usage_log is not None:
                usage_log.close()
            if auth_service is not None:
                auth_service.db.close()

    app = FastAPI(title="codex-proxy", version=__version__, lifespan=lifespan)

    # Bearer middleware for /v1/* — only enforced when an auth service is
    # configured. In single-operator mode (no auth db) /v1/* stays open.
    @app.middleware("http")
    async def _enforce_api_key(request: Request, call_next: Callable[[Request], Any]) -> Any:
        if auth_service is None or not request.url.path.startswith("/v1/"):
            return await call_next(request)
        plaintext = _bearer(request)
        if plaintext is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authorization: Bearer <api-key> required"},
            )
        try:
            api_key = auth_service.resolve_api_key(plaintext)
        except ApiKeyInvalidError as exc:
            return JSONResponse(status_code=401, content={"detail": str(exc)})
        request.state.api_key = api_key
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/status")
    async def status() -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for backend in backends_list:
            h = await backend.health()
            u = await backend.usage_snapshot()
            entries.append(
                {
                    "id": backend.id,
                    "kind": backend.kind,
                    "advertised_models": sorted(backend.advertised_models),
                    "health": {
                        "available": h.available,
                        "reason": h.reason,
                        "retry_after_s": h.retry_after_s,
                    },
                    "usage": {
                        "remaining_fraction": u.remaining_fraction,
                        "cooldown_until_ts": u.cooldown_until_ts,
                        "weekly_exhausted": u.weekly_exhausted,
                        "probed_at_ts": u.probed_at_ts,
                    },
                }
            )
        return {
            "backends": entries,
            "pinned": pin_state.get(),
            "sessions": session_registry.snapshot(),
        }

    if auth_service is not None:
        _install_auth_routes(app, auth_service)

    @app.post("/control/pin")
    async def control_pin(body: dict[str, Any]) -> dict[str, str | None]:
        backend_id = body.get("backend_id")
        if not isinstance(backend_id, str):
            raise HTTPException(status_code=400, detail="'backend_id' must be a string")
        if not any(b.id == backend_id for b in backends_list):
            raise HTTPException(status_code=404, detail=f"backend {backend_id!r} not in pool")
        pin_state.set(backend_id)
        return {"pinned": backend_id}

    @app.post("/control/unpin")
    async def control_unpin() -> dict[str, str | None]:
        pin_state.clear()
        return {"pinned": None}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, body: dict[str, Any]) -> Any:
        return await _dispatch_route(
            body,
            request=request,
            route_name="chat_completions",
            backends_list=backends_list,
            pin_state=pin_state,
            session_registry=session_registry,
            usage_log=usage_log,
            nonstream=lambda b, p, h: b.chat_completions(p, h),
            stream=lambda b, p, h: b.chat_completions_stream(p, h),
        )

    @app.post("/v1/responses")
    async def responses(request: Request, body: dict[str, Any]) -> Any:
        return await _dispatch_route(
            body,
            request=request,
            route_name="responses",
            backends_list=backends_list,
            pin_state=pin_state,
            session_registry=session_registry,
            usage_log=usage_log,
            nonstream=lambda b, p, h: b.responses(p, h),
            stream=lambda b, p, h: b.responses_stream(p, h),
        )

    return app


async def _dispatch_route(
    body: dict[str, Any],
    *,
    request: Request,
    route_name: str,
    backends_list: Sequence[Backend],
    pin_state: PinState,
    session_registry: SessionRegistry,
    usage_log: UsageLog | None,
    nonstream: NonstreamCall,
    stream: StreamCall,
) -> Any:
    model = _require_model(body)
    pinned = pin_state.get()
    active = _active_pool(backends_list, pinned)
    session_id = _session_id_from_request(request, pinned=pinned)
    preferred_id = session_registry.get(session_id) if session_id is not None else None
    api_key: ApiKey | None = getattr(request.state, "api_key", None)
    user_id = api_key.user_id if api_key is not None else None
    api_key_id = api_key.id if api_key is not None else None
    if body.get("stream") is True:
        return await _dispatch_stream(
            body,
            model=model,
            route_name=route_name,
            backends_list=active,
            preferred_id=preferred_id,
            session_id=session_id,
            session_registry=session_registry,
            usage_log=usage_log,
            call=stream,
            user_id=user_id,
            api_key_id=api_key_id,
        )
    return await _dispatch_nonstream(
        body,
        model=model,
        route_name=route_name,
        backends_list=active,
        preferred_id=preferred_id,
        session_id=session_id,
        session_registry=session_registry,
        usage_log=usage_log,
        call=nonstream,
        user_id=user_id,
        api_key_id=api_key_id,
    )


def _require_model(body: dict[str, Any]) -> str:
    model = body.get("model")
    if not isinstance(model, str):
        raise HTTPException(status_code=400, detail="'model' must be a string")
    return model


def _active_pool(backends_list: Sequence[Backend], pinned: str | None) -> Sequence[Backend]:
    if pinned is None:
        return backends_list
    return [b for b in backends_list if b.id == pinned]


def _session_id_from_request(request: Request, *, pinned: str | None) -> str | None:
    # A pin is an operator override: it wins over any client-declared session.
    if pinned is not None:
        return None
    raw = request.headers.get(SESSION_HEADER)
    if raw is None:
        return None
    session_id = raw.strip()
    return session_id or None


def _remember_binding(registry: SessionRegistry, session_id: str | None, backend_id: str) -> None:
    if session_id is None:
        return
    registry.set(session_id, backend_id)


async def _dispatch_nonstream(
    body: dict[str, Any],
    *,
    model: str,
    route_name: str,
    backends_list: Sequence[Backend],
    preferred_id: str | None,
    session_id: str | None,
    session_registry: SessionRegistry,
    usage_log: UsageLog | None,
    call: NonstreamCall,
    user_id: int | None = None,
    api_key_id: int | None = None,
) -> dict[str, Any]:
    excluded: set[str] = set()
    last_error: BackendError | None = None
    while True:
        backend = await select(
            backends_list,
            model=model,
            excluded=frozenset(excluded),
            preferred_id=preferred_id,
        )
        if backend is None:
            break
        handle = CallHandle()
        ts_start = time.time()
        try:
            result = await call(backend, body, handle)
        except BackendError as exc:
            ts_end = time.time()
            _log_attempt(
                usage_log,
                body=body,
                model=model,
                route_name=route_name,
                stream=False,
                session_id=session_id,
                backend=backend,
                handle=handle,
                ts_start=ts_start,
                ts_end=ts_end,
                error=exc,
                resp_body=None,
                user_id=user_id,
                api_key_id=api_key_id,
            )
            last_error = exc
            if exc.classification not in RETRYABLE:
                raise _terminal_http(exc) from exc
            excluded.add(backend.id)
            continue
        ts_end = time.time()
        _remember_binding(session_registry, session_id, backend.id)
        _log_attempt(
            usage_log,
            body=body,
            model=model,
            route_name=route_name,
            stream=False,
            session_id=session_id,
            backend=backend,
            handle=handle,
            ts_start=ts_start,
            ts_end=ts_end,
            error=None,
            resp_body=result,
            user_id=user_id,
            api_key_id=api_key_id,
        )
        return result
    raise _no_viable(model=model, last_error=last_error)


async def _dispatch_stream(
    body: dict[str, Any],
    *,
    model: str,
    route_name: str,
    backends_list: Sequence[Backend],
    preferred_id: str | None,
    session_id: str | None,
    session_registry: SessionRegistry,
    usage_log: UsageLog | None,
    call: StreamCall,
    user_id: int | None = None,
    api_key_id: int | None = None,
) -> StreamingResponse:
    excluded: set[str] = set()
    last_error: BackendError | None = None
    while True:
        backend = await select(
            backends_list,
            model=model,
            excluded=frozenset(excluded),
            preferred_id=preferred_id,
        )
        if backend is None:
            break
        handle = CallHandle()
        ts_start = time.time()
        iterator = call(backend, body, handle)
        try:
            first_chunk = await iterator.__anext__()
        except StopAsyncIteration:
            ts_end = time.time()
            _log_attempt(
                usage_log,
                body=body,
                model=model,
                route_name=route_name,
                stream=True,
                session_id=session_id,
                backend=backend,
                handle=handle,
                ts_start=ts_start,
                ts_end=ts_end,
                error=None,
                resp_body=None,
                user_id=user_id,
                api_key_id=api_key_id,
            )
            _remember_binding(session_registry, session_id, backend.id)
            return StreamingResponse(_empty_iter(), media_type="text/event-stream")
        except BackendError as exc:
            ts_end = time.time()
            _log_attempt(
                usage_log,
                body=body,
                model=model,
                route_name=route_name,
                stream=True,
                session_id=session_id,
                backend=backend,
                handle=handle,
                ts_start=ts_start,
                ts_end=ts_end,
                error=exc,
                resp_body=None,
                user_id=user_id,
                api_key_id=api_key_id,
            )
            last_error = exc
            if exc.classification not in RETRYABLE:
                raise _terminal_http(exc) from exc
            excluded.add(backend.id)
            continue
        _remember_binding(session_registry, session_id, backend.id)
        return StreamingResponse(
            _log_on_complete(
                _prepend(first_chunk, iterator),
                usage_log=usage_log,
                body=body,
                model=model,
                route_name=route_name,
                session_id=session_id,
                backend=backend,
                handle=handle,
                ts_start=ts_start,
                user_id=user_id,
                api_key_id=api_key_id,
            ),
            media_type="text/event-stream",
        )
    raise _no_viable(model=model, last_error=last_error)


async def _prepend(first: bytes, rest: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    yield first
    async for chunk in rest:
        yield chunk


async def _empty_iter() -> AsyncIterator[bytes]:
    if False:
        yield b""


async def _log_on_complete(
    source: AsyncIterator[bytes],
    *,
    usage_log: UsageLog | None,
    body: dict[str, Any],
    model: str,
    route_name: str,
    session_id: str | None,
    backend: Backend,
    handle: CallHandle,
    ts_start: float,
    user_id: int | None = None,
    api_key_id: int | None = None,
) -> AsyncIterator[bytes]:
    """Pass-through wrapper that writes the usage log row when the stream ends.

    The backend's `*_stream` methods populate `handle.stream_summary` after the
    final chunk is yielded, so we log on the other side of the async-for loop.
    """
    async for chunk in source:
        yield chunk
    ts_end = time.time()
    _log_attempt(
        usage_log,
        body=body,
        model=model,
        route_name=route_name,
        stream=True,
        session_id=session_id,
        backend=backend,
        handle=handle,
        ts_start=ts_start,
        ts_end=ts_end,
        error=None,
        resp_body=None,  # for streams the raw body lives in handle.stream_summary
        user_id=user_id,
        api_key_id=api_key_id,
    )


def _log_attempt(
    usage_log: UsageLog | None,
    *,
    body: dict[str, Any],
    model: str,
    route_name: str,
    stream: bool,
    session_id: str | None,
    backend: Backend,
    handle: CallHandle,
    ts_start: float,
    ts_end: float,
    error: BackendError | None,
    resp_body: dict[str, Any] | None,
    user_id: int | None = None,
    api_key_id: int | None = None,
) -> None:
    if usage_log is None:
        return
    req_payload = json.dumps(body).encode()
    if stream and handle.stream_summary is not None:
        resp_payload: bytes | None = handle.stream_summary.raw_blob
        response_bytes = handle.stream_summary.total_bytes
        completed = handle.stream_summary.completed_response
        tokens = _extract_tokens(completed.get("usage") if completed else None)
    elif resp_body is not None:
        resp_payload = json.dumps(resp_body).encode()
        response_bytes = len(resp_payload)
        tokens = _extract_tokens(resp_body.get("usage"))
    else:
        resp_payload = None
        response_bytes = None
        tokens = _Tokens(None, None, None, None, None)
    status = error.status_code if error is not None else 200
    classification = error.classification if error is not None else "ok"
    entry = UsageLogEntry(
        ts_start=ts_start,
        ts_end=ts_end,
        route=route_name,
        stream=stream,
        session_id=session_id,
        backend_id=backend.id,
        model=model,
        reasoning_effort=_extract_reasoning_effort(body),
        status=int(status) if status is not None else 0,
        classification=classification,
        request_bytes=len(req_payload),
        response_bytes=response_bytes,
        prompt_tokens=tokens.prompt,
        completion_tokens=tokens.completion,
        total_tokens=tokens.total,
        cached_tokens=tokens.cached,
        reasoning_tokens=tokens.reasoning,
        quota_before=handle.quota_before,
        quota_after=handle.quota_after,
        req_payload=req_payload,
        resp_payload=resp_payload,
        upstream_headers=dict(handle.upstream_headers) if handle.upstream_headers else None,
        user_id=user_id,
        api_key_id=api_key_id,
    )
    usage_log.record(entry)


class _Tokens:
    __slots__ = ("prompt", "completion", "total", "cached", "reasoning")

    def __init__(
        self,
        prompt: int | None,
        completion: int | None,
        total: int | None,
        cached: int | None,
        reasoning: int | None,
    ) -> None:
        self.prompt = prompt
        self.completion = completion
        self.total = total
        self.cached = cached
        self.reasoning = reasoning


def _extract_tokens(usage: Any) -> _Tokens:
    """Map either a Responses-API or chat-completions usage block to a common struct."""
    if not isinstance(usage, dict):
        return _Tokens(None, None, None, None, None)
    prompt = _int(usage.get("input_tokens")) or _int(usage.get("prompt_tokens"))
    completion = _int(usage.get("output_tokens")) or _int(usage.get("completion_tokens"))
    total = _int(usage.get("total_tokens"))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    # Cached input tokens appear under input_tokens_details.cached_tokens on
    # the Responses API; some chat responses expose prompt_tokens_details.
    cached: int | None = None
    details_in = usage.get("input_tokens_details")
    if isinstance(details_in, dict):
        cached = _int(details_in.get("cached_tokens"))
    if cached is None:
        details_p = usage.get("prompt_tokens_details")
        if isinstance(details_p, dict):
            cached = _int(details_p.get("cached_tokens"))
    reasoning: int | None = None
    details_out = usage.get("output_tokens_details")
    if isinstance(details_out, dict):
        reasoning = _int(details_out.get("reasoning_tokens"))
    return _Tokens(prompt, completion, total, cached, reasoning)


def _extract_reasoning_effort(body: dict[str, Any]) -> str | None:
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if isinstance(effort, str):
            return effort
    # Some clients put reasoning_effort at the top level.
    effort_top = body.get("reasoning_effort")
    if isinstance(effort_top, str):
        return effort_top
    return None


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _bearer(request: Request) -> str | None:
    raw = request.headers.get("authorization")
    if raw is None:
        return None
    parts = raw.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def _install_auth_routes(app: FastAPI, auth_service: AuthService) -> None:
    """Add /auth/* routes when an AuthService is configured.

    Endpoints:
    - POST /auth/register {username, password} -> {user_id, username}
    - POST /auth/login    {username, password} -> {session_token, expires_at}
    - POST /auth/logout                         -> 204 (session-bearer)
    - POST /auth/keys     {label?}             -> issued key (plaintext shown ONCE)
    - GET  /auth/keys                          -> list keys for the session's user
    - DELETE /auth/keys/{key_id}               -> revoke
    """

    def _require_session(request: Request) -> Session:
        plaintext = _bearer(request)
        if plaintext is None:
            raise HTTPException(status_code=401, detail="session token required")
        try:
            return auth_service.resolve_session(plaintext)
        except SessionInvalidError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @app.post("/auth/register", status_code=201)
    async def register(body: dict[str, Any]) -> dict[str, Any]:
        username = body.get("username")
        password = body.get("password")
        if not isinstance(username, str) or not isinstance(password, str):
            raise HTTPException(status_code=400, detail="'username' and 'password' must be strings")
        try:
            user = auth_service.register(username=username, password=password)
        except InvalidCredentialsError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"user_id": user.id, "username": user.username}

    @app.post("/auth/login")
    async def login(body: dict[str, Any]) -> dict[str, Any]:
        username = body.get("username")
        password = body.get("password")
        if not isinstance(username, str) or not isinstance(password, str):
            raise HTTPException(status_code=400, detail="'username' and 'password' must be strings")
        try:
            issued = auth_service.login(username=username, password=password)
        except InvalidCredentialsError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return {
            "session_token": issued.plaintext,
            "expires_at": issued.session.expires_at,
        }

    @app.post("/auth/logout", status_code=204)
    async def logout(request: Request) -> None:
        plaintext = _bearer(request)
        if plaintext is not None:
            auth_service.logout(plaintext)

    @app.post("/auth/keys", status_code=201)
    async def create_key(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        session = _require_session(request)
        label = body.get("label") if isinstance(body.get("label"), str) else None
        issued = auth_service.create_api_key(user_id=session.user_id, label=label)
        return {
            "id": issued.api_key.id,
            "api_key": issued.plaintext,  # plaintext shown ONCE
            "prefix": issued.api_key.key_prefix,
            "label": issued.api_key.label,
            "created_at": issued.api_key.created_at,
        }

    @app.get("/auth/keys")
    async def list_keys(request: Request) -> dict[str, Any]:
        session = _require_session(request)
        keys = auth_service.list_api_keys(user_id=session.user_id)
        return {
            "keys": [
                {
                    "id": k.id,
                    "prefix": k.key_prefix,
                    "label": k.label,
                    "created_at": k.created_at,
                    "last_used_at": k.last_used_at,
                    "revoked_at": k.revoked_at,
                }
                for k in keys
            ]
        }

    @app.delete("/auth/keys/{key_id}")
    async def revoke_key(request: Request, key_id: int) -> dict[str, bool]:
        session = _require_session(request)
        revoked = auth_service.revoke_api_key(key_id=key_id, user_id=session.user_id)
        if not revoked:
            raise HTTPException(status_code=404, detail="key not found or already revoked")
        return {"revoked": True}


def _terminal_http(exc: BackendError) -> HTTPException:
    status = exc.status_code or _EXHAUSTED_STATUS.get(exc.classification, 500)
    return HTTPException(status_code=status, detail=exc.message or exc.classification)


def _no_viable(*, model: str, last_error: BackendError | None) -> HTTPException:
    if last_error is None:
        return HTTPException(status_code=503, detail=f"no viable backend for model {model!r}")
    status = _EXHAUSTED_STATUS.get(last_error.classification, 502)
    return HTTPException(
        status_code=status,
        detail=last_error.message or last_error.classification,
    )
