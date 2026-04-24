from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from codex_proxy import __version__
from codex_proxy.backend import Backend
from codex_proxy.errors import RETRYABLE, BackendError, ErrorClass
from codex_proxy.selector import select
from codex_proxy.session import SessionRegistry

# Clients opt into sticky routing by sending this header. When absent, every
# request is a fresh selection. There is no server-side "session mode" knob.
SESSION_HEADER = "x-codex-session-id"

NonstreamCall = Callable[[Backend, dict[str, Any]], Awaitable[dict[str, Any]]]
StreamCall = Callable[[Backend, dict[str, Any]], AsyncIterator[bytes]]

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

    app = FastAPI(title="codex-proxy", version=__version__, lifespan=lifespan)

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
            backends_list=backends_list,
            pin_state=pin_state,
            session_registry=session_registry,
            nonstream=lambda b, payload: b.chat_completions(payload),
            stream=lambda b, payload: b.chat_completions_stream(payload),
        )

    @app.post("/v1/responses")
    async def responses(request: Request, body: dict[str, Any]) -> Any:
        return await _dispatch_route(
            body,
            request=request,
            backends_list=backends_list,
            pin_state=pin_state,
            session_registry=session_registry,
            nonstream=lambda b, payload: b.responses(payload),
            stream=lambda b, payload: b.responses_stream(payload),
        )

    return app


async def _dispatch_route(
    body: dict[str, Any],
    *,
    request: Request,
    backends_list: Sequence[Backend],
    pin_state: PinState,
    session_registry: SessionRegistry,
    nonstream: NonstreamCall,
    stream: StreamCall,
) -> Any:
    model = _require_model(body)
    pinned = pin_state.get()
    active = _active_pool(backends_list, pinned)
    session_id = _session_id_from_request(request, pinned=pinned)
    preferred_id = session_registry.get(session_id) if session_id is not None else None
    if body.get("stream") is True:
        return await _dispatch_stream(
            body,
            model=model,
            backends_list=active,
            preferred_id=preferred_id,
            session_id=session_id,
            session_registry=session_registry,
            call=stream,
        )
    return await _dispatch_nonstream(
        body,
        model=model,
        backends_list=active,
        preferred_id=preferred_id,
        session_id=session_id,
        session_registry=session_registry,
        call=nonstream,
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
    backends_list: Sequence[Backend],
    preferred_id: str | None,
    session_id: str | None,
    session_registry: SessionRegistry,
    call: NonstreamCall,
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
        try:
            result = await call(backend, body)
        except BackendError as exc:
            last_error = exc
            if exc.classification not in RETRYABLE:
                raise _terminal_http(exc) from exc
            excluded.add(backend.id)
            continue
        _remember_binding(session_registry, session_id, backend.id)
        return result
    raise _no_viable(model=model, last_error=last_error)


async def _dispatch_stream(
    body: dict[str, Any],
    *,
    model: str,
    backends_list: Sequence[Backend],
    preferred_id: str | None,
    session_id: str | None,
    session_registry: SessionRegistry,
    call: StreamCall,
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
        iterator = call(backend, body)
        try:
            first_chunk = await iterator.__anext__()
        except StopAsyncIteration:
            _remember_binding(session_registry, session_id, backend.id)
            return StreamingResponse(_empty_iter(), media_type="text/event-stream")
        except BackendError as exc:
            last_error = exc
            if exc.classification not in RETRYABLE:
                raise _terminal_http(exc) from exc
            excluded.add(backend.id)
            continue
        _remember_binding(session_registry, session_id, backend.id)
        return StreamingResponse(
            _prepend(first_chunk, iterator),
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
