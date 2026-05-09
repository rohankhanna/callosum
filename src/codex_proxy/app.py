from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from codex_proxy import __version__
from codex_proxy.auth import (
    ApiKeyInvalidError,
    AuthService,
    InvalidCredentialsError,
    SessionInvalidError,
)
from codex_proxy.auth_db import ApiKey, Session
from codex_proxy.backend import Backend, CallHandle
from codex_proxy.cell_grid import VIRTUAL_MODELS, Cell, build_cells, live_completion_models
from codex_proxy.config import AutoRouterConfig
from codex_proxy.errors import RETRYABLE, BackendError, ErrorClass
from codex_proxy.fallback import FallbackExecutor, should_attempt_fallback
from codex_proxy.router import LearnedModelRouter, ExplorerRouter
from codex_proxy.selector import select
from codex_proxy.session import SessionRegistry
from codex_proxy.synthetic import SyntheticTopper
from codex_proxy.usage_log import UsageLog, UsageLogEntry

logger = logging.getLogger("codex_proxy.startup")


def _utc_timestamp() -> str:
    """Return current time in ISO 8601 UTC format with Z suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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

# Context variable to track the current request's database rowid, set during dispatch
# and read by response handlers to include in X-Proxy-Request-ID header.
_request_id_context: ContextVar[int | None] = ContextVar("request_id", default=None)

# Context variable to signal that this request needs complexity extraction from the response.
# Set to True for auto-learning requests, used by response handlers to extract {{{N}}}.
_extract_complexity_context: ContextVar[bool] = ContextVar("extract_complexity", default=False)

# Context variable to store the extracted complexity class (1, 2, or 3) after response is processed.
_complexity_class_context: ContextVar[int | None] = ContextVar("complexity_class", default=None)

# Complexity classification instruction appended to auto-learning requests.
# The model outputs {{{1}}}, {{{2}}}, or {{{3}}} at the start of its response.
_COMPLEXITY_CLASSIFIER_INSTRUCTION = (
    "Before answering, classify this prompt's complexity as {{{1}}} (simple factual/short), "
    "{{{2}}} (moderate analysis), or {{{3}}} (complex reasoning/long output). "
    "Output ONLY the classification token first, then your answer."
)


def _extract_complexity_class(text: str) -> tuple[int | None, str]:
    """Extract complexity classification token {{{N}}} from response start.

    Returns (complexity_class, cleaned_text) where complexity_class is 1, 2, or 3,
    or None if the marker is not found. cleaned_text has the marker stripped.

    The marker should appear at the very start of the response (after whitespace).
    """
    import re
    match = re.match(r'^\s*\{\{\{([123])\}\}\}', text)
    if match:
        complexity_class = int(match.group(1))
        cleaned = text[match.end():].lstrip()
        return complexity_class, cleaned
    return None, text


def _extract_and_strip_complexity(result: dict[str, Any]) -> tuple[int | None, dict[str, Any]]:
    """Extract complexity class from response dict and strip the marker.

    Handles both response formats:
    1. Chat completions: choices[0].message.content
    2. Responses API: output[0].content[0].text

    Returns (complexity_class, modified_result_dict) where result is updated with cleaned content.
    """
    # Try chat completions format first
    try:
        if result.get("choices") and len(result["choices"]) > 0:
            choice = result["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                content = choice["message"]["content"]
                if isinstance(content, str):
                    complexity_class, cleaned = _extract_complexity_class(content)
                    if complexity_class is not None:
                        result["choices"][0]["message"]["content"] = cleaned
                    return complexity_class, result
    except (KeyError, IndexError, TypeError):
        pass

    # Try Responses API format
    try:
        output = result.get("output")
        if output and isinstance(output, list) and len(output) > 0:
            output_item = output[0]
            content_list = output_item.get("content")
            if content_list and isinstance(content_list, list) and len(content_list) > 0:
                content_item = content_list[0]
                text = content_item.get("text")
                if isinstance(text, str):
                    complexity_class, cleaned = _extract_complexity_class(text)
                    if complexity_class is not None:
                        result["output"][0]["content"][0]["text"] = cleaned
                    return complexity_class, result
    except (KeyError, IndexError, TypeError):
        pass

    logger.warning("Could not extract complexity marker from response (unsupported format)")
    return None, result


def _learned_model_cap_pct(
    startup_timestamp: float | None,
    model_release_timestamp: float | None,
) -> float:
    """Calculate current learned model usage cap percentage.

    Two phases:

    1. **Initial phase (no model release detected)**:
       - Ramp from 1% to 90% over 90 days from proxy startup
       - After 90 days: hold at 90% until new model detected

    2. **After model release detected**:
       - Reset to 75% when new model appears
       - Ramp from 75% to 90% over 30 days
       - After 30 days: hold at 90% until next model release

    Returns the learned model cap as a percentage (0-100).
    """
    now_ts = time.time()

    # Phase 2: New model detected
    if model_release_timestamp is not None and model_release_timestamp <= now_ts:
        days_since_release = (now_ts - model_release_timestamp) / 86400

        if days_since_release >= 30:
            # Fully ramped after 30 days
            return 90.0

        # Linearly ramp from 75% to 90% over 30 days
        cap = 75.0 + (days_since_release / 30) * 15.0
        return cap

    # Phase 1: No model release detected yet (or startup)
    if startup_timestamp is not None and startup_timestamp <= now_ts:
        days_since_startup = (now_ts - startup_timestamp) / 86400

        if days_since_startup >= 90:
            # Fully ramped after 90 days
            return 90.0

        # Linearly ramp from 1% to 90% over 90 days
        cap = 1.0 + (days_since_startup / 90) * 89.0
        return cap

    # Fallback (shouldn't happen): conservative default
    return 1.0


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
    auto_router_config: AutoRouterConfig | None = None,
    startup_smoke_test: bool = False,
    smoke_test_interval_seconds: int = 0,
) -> FastAPI:
    from codex_proxy.state import StateStore

    backends_list: list[Backend] = list(backends)
    pin_state = PinState()
    session_registry = sessions if sessions is not None else SessionRegistry()

    # Extract state_store from the first CodexAuthVaultBackend (for model release tracking)
    state_store: StateStore | None = None
    for b in backends_list:
        if hasattr(b, "_state_store"):
            state_store = b._state_store  # type: ignore
            break

    def _live_cells() -> list[Cell]:
        """Build the auto-learning cell grid from the union of every Codex
        backend's CURRENT advertised_models (excluding OpenRouter's shadow
        set since OpenRouter doesn't participate in the cost-model corpus).

        Filters to chat-completion-shaped ids only (skips codex-auto-review,
        embeddings, audio, etc.) and orders strongest-model-first so any
        ties in coverage round-robin break in favor of the more useful cell.

        Recomputed on every router decision — when CodexAuthVaultBackend's
        hourly catalog refresh discovers a new model, the cell grid follows
        without a proxy restart. When a model is retired upstream, it drops
        out of the grid the next time the router consults it.
        """
        pool: set[str] = set()
        for b in backends_list:
            if b.kind != "codex_auth_vault":
                continue
            pool.update(b.advertised_models)
        models = live_completion_models(frozenset(pool))
        if not models:
            # Cold start — backends haven't populated catalogs yet, fall
            # back to the static defaults so the router can still operate.
            return build_cells()
        return build_cells(models=models)

    explorer = ExplorerRouter(
        usage_log_path=usage_log.path if usage_log is not None else None,
        cells_fn=_live_cells,
    )
    synthetic_explorer = ExplorerRouter(
        usage_log_path=usage_log.path if usage_log is not None else None,
        routing_mode="auto-learning-synthetic",
        cells_fn=_live_cells,
    )
    cost_router = LearnedModelRouter(usage_log_path=usage_log.path if usage_log is not None else None)

    auto_cfg = auto_router_config if auto_router_config is not None else AutoRouterConfig()

    async def _synthetic_dispatch(body: dict[str, Any], backend_id: str) -> Any:
        # The topper calls dispatch directly — no Request, no auth attribution,
        # and pinned to the backend the controller picked (the natural selector
        # ranks by 5h capacity, which isn't the same as "weekly headroom that
        # won't be human-consumed in time"). Routes through the responses path.
        return await _dispatch_internal(
            body,
            route_name="responses",
            backends_list=backends_list,
            pin_state=pin_state,
            session_registry=session_registry,
            usage_log=usage_log,
            explorer=explorer,
            synthetic_explorer=synthetic_explorer,
            cost_router=cost_router,
            nonstream=lambda b, p, h: b.responses(p, h),
            stream=lambda b, p, h: b.responses_stream(p, h),
            session_id=None,
            user_id=None,
            api_key_id=None,
            forced_backend_id=backend_id,
            router_context_safety_margin=auto_cfg.router_context_safety_margin,
            state_store=state_store,
        )

    topper = SyntheticTopper(
        cfg=auto_cfg,
        usage_log_path=usage_log.path if usage_log is not None else None,
        backends=backends_list,
        dispatch=_synthetic_dispatch,
        cost_router=cost_router,
    )
    smoke_tester = _PeriodicSmokeTester(
        backends=backends_list,
        interval_s=smoke_test_interval_seconds,
        state_store=state_store,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Make the loaded-backend roster trivially greppable in the launch log
        # so operators can see at a glance whether OPENROUTER_API_KEY etc. were
        # picked up by the process they actually launched.
        ids = ", ".join(b.id for b in backends_list) if backends_list else "(none)"
        logger.warning("loaded %d backend(s): %s", len(backends_list), ids)

        # Initialize proxy startup timestamp if not already set (for initial 90-day ramp)
        if state_store is not None and state_store.get_proxy_startup_timestamp() is None:
            state_store.set_proxy_startup_timestamp(time.time())
        # Refresh dynamic model lists for backends that support it BEFORE the
        # smoke test runs — that way the smoke test probes models the upstream
        # actually still serves, not stale TOML names. Best-effort: any
        # backend that fails to refresh just keeps using its cold-start set.
        # Also detect when new models appear (model release) and timestamp them.
        for backend in backends_list:
            refresh = getattr(backend, "refresh_advertised_models", None)
            if refresh is not None:
                try:
                    # Capture models before refresh to detect new ones
                    models_before = set(backend.advertised_models)
                    await refresh()
                    models_after = set(backend.advertised_models)
                    # If new models detected, record the release timestamp
                    if models_after > models_before and state_store is not None:
                        new_models = models_after - models_before
                        logger.info(
                            "new models detected for backend %r: %s — recording model release timestamp",
                            backend.id, sorted(new_models)
                        )
                        state_store.set_model_release_timestamp(time.time())
                except Exception:
                    logger.exception("startup model-list refresh failed for %r", backend.id)
        if startup_smoke_test and backends_list:
            await _run_startup_smoke_test(backends_list)
        # Prime the cost_router cost model if data exists
        cost_router.fit(min_samples_per_cell=auto_cfg.exploiter_min_samples_per_cell)
        topper.start()
        smoke_tester.start()
        try:
            yield
        finally:
            await smoke_tester.stop()
            await topper.stop()
            for backend in backends_list:
                await backend.aclose()
            if usage_log is not None:
                usage_log.close()
            if auth_service is not None:
                auth_service.db.close()

    app = FastAPI(title="codex-proxy", version=__version__, lifespan=lifespan)

    # Bearer middleware for /v1/* and /diagnose/* — only enforced when an
    # auth service is configured. In single-operator mode (no auth db) those
    # routes stay open.
    @app.middleware("http")
    async def _enforce_api_key(request: Request, call_next: Callable[[Request], Any]) -> Any:
        if auth_service is None or not _path_requires_api_key(request.url.path):
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
            q = await backend.quota_snapshot()
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
                    "quota": _quota_to_dict(q),
                }
            )
        return {
            "backends": entries,
            "pinned": pin_state.get(),
            "sessions": session_registry.snapshot(),
        }

    if auth_service is not None:
        _install_auth_routes(app, auth_service)
        _install_web_ui(app)

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

    @app.get("/diagnose/upstream")
    async def diagnose_upstream() -> dict[str, Any]:
        """Probe each backend with a tiny real request and verify the upstream
        contract still holds (HTTP 200, x-codex-* headers parse, response.completed
        SSE event with usage block, model name still accepted).

        Intended for a daily cron — run it, alert on any backend's `ok=false`.
        Each invocation makes one real upstream call per backend, costing a
        small number of quota tokens. Cooldown'd backends are reported as
        skipped (the daily run shouldn't kick a backend that's already
        recovering).
        """
        results = []
        for backend in backends_list:
            results.append(await _diagnose_backend(backend))
        all_ok = all(r["ok"] for r in results if not r.get("skipped"))
        return {"ok": all_ok, "backends": results}

    def _live_catalog() -> dict[str, Any]:
        """OpenAI-compatible model list, recomputed on each call from the
        union of every backend's live advertised_models.
        """
        seen: set[str] = set()
        for backend in backends_list:
            for m in backend.advertised_models:
                seen.add(m)
        now_ts = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": now_ts,
                    "owned_by": "codex-proxy",
                }
                for model_id in sorted(seen)
            ],
        }

    @app.get("/v1/models")
    async def list_models_v1() -> dict[str, Any]:
        """OpenAI-standard catalog endpoint."""
        return _live_catalog()

    @app.get("/models")
    async def list_models_root() -> dict[str, Any]:
        """Some clients probe /models without the /v1 prefix."""
        return _live_catalog()

    @app.get("/api/v1/models")
    async def list_models_api_v1() -> dict[str, Any]:
        """Ollama-style /api/v1/models prefix some clients try."""
        return _live_catalog()

    @app.get("/api/tags", include_in_schema=False)
    async def ollama_tags() -> dict[str, Any]:
        """Ollama compatibility probe. Returning an empty `models: []` is
        valid Ollama shape and signals "I'm not Ollama" without 404'ing.
        """
        return {"models": []}

    @app.get("/version", include_in_schema=False)
    @app.get("/api/version", include_in_schema=False)
    async def version_probe() -> dict[str, str]:
        """Version probe — Ollama, model-a0e0, and others all hit this path."""
        return {"version": __version__}

    @app.get("/v1/props", include_in_schema=False)
    @app.get("/props", include_in_schema=False)
    async def llamacpp_props() -> dict[str, Any]:
        """model-a0e0 /props probe. Empty dict is a valid response that
        signals "I don't speak model-a0e0" without 404'ing.
        """
        return {}

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        """Root probe — common reachability check; surface enough that an
        operator hitting the proxy in a browser sees something useful.
        """
        return {
            "service": "codex-proxy",
            "version": __version__,
            "docs": "/docs",
            "status": "/status",
        }

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        """No favicon, but don't 404 — that just clutters the log."""
        return Response(status_code=204)

    @app.get("/v1/models/{model_id:path}")
    async def get_model(model_id: str) -> dict[str, Any]:
        """OpenAI-compatible single-model lookup. Returns 404 when the model
        isn't advertised by any backend.
        """
        for backend in backends_list:
            if model_id in backend.advertised_models:
                return {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "codex-proxy",
                }
        raise HTTPException(status_code=404, detail=f"model {model_id!r} not found")

    @app.post("/v1/feedback")
    async def feedback(body: dict[str, Any]) -> dict[str, str]:
        """Record user feedback (quality label) for a request.

        Expected body: {"request_id": <int>, "rating": <-1|0|1>}
        """
        if usage_log is None:
            raise HTTPException(status_code=503, detail="usage logging disabled")
        request_id = body.get("request_id")
        rating = body.get("rating")
        if not isinstance(request_id, int) or request_id <= 0:
            raise HTTPException(status_code=400, detail="request_id must be a positive integer")
        if rating not in (-1, 0, 1):
            raise HTTPException(status_code=400, detail="rating must be -1, 0, or 1")
        try:
            usage_log.record_quality(request_id, rating, "user")
        except Exception as exc:
            logging.getLogger("codex_proxy.app").warning("feedback record failed: %s", exc)
            raise HTTPException(status_code=400, detail="request_id not found or feedback failed")
        return {"status": "recorded", "request_id": request_id}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, body: dict[str, Any]) -> Any:
        _request_id_context.set(None)  # Reset context for this request
        _extract_complexity_context.set(False)
        _complexity_class_context.set(None)
        result = await _dispatch_route(
            body,
            request=request,
            route_name="chat_completions",
            backends_list=backends_list,
            pin_state=pin_state,
            session_registry=session_registry,
            usage_log=usage_log,
            explorer=explorer,
            synthetic_explorer=synthetic_explorer,
            cost_router=cost_router,
            nonstream=lambda b, p, h: b.chat_completions(p, h),
            stream=lambda b, p, h: b.chat_completions_stream(p, h),
            router_context_safety_margin=auto_cfg.router_context_safety_margin,
            state_store=state_store,
        )
        # Add X-Proxy-Request-ID header if a request was logged
        request_id = _request_id_context.get()
        if request_id is not None:
            headers = {"X-Proxy-Request-ID": str(request_id)}
            if isinstance(result, dict):
                return JSONResponse(result, headers=headers)
            # For streaming (AsyncIterator), wrap with StreamingResponse
            return StreamingResponse(result, media_type="text/event-stream", headers=headers)
        return result

    @app.post("/v1/responses")
    async def responses(request: Request, body: dict[str, Any]) -> Any:
        _request_id_context.set(None)  # Reset context for this request
        _extract_complexity_context.set(False)
        _complexity_class_context.set(None)
        result = await _dispatch_route(
            body,
            request=request,
            route_name="responses",
            backends_list=backends_list,
            pin_state=pin_state,
            session_registry=session_registry,
            usage_log=usage_log,
            explorer=explorer,
            synthetic_explorer=synthetic_explorer,
            cost_router=cost_router,
            nonstream=lambda b, p, h: b.responses(p, h),
            stream=lambda b, p, h: b.responses_stream(p, h),
            router_context_safety_margin=auto_cfg.router_context_safety_margin,
            state_store=state_store,
        )
        # Add X-Proxy-Request-ID header if a request was logged
        request_id = _request_id_context.get()
        if request_id is not None:
            headers = {"X-Proxy-Request-ID": str(request_id)}
            if isinstance(result, dict):
                return JSONResponse(result, headers=headers)
            # For streaming (AsyncIterator), wrap with StreamingResponse
            return StreamingResponse(result, media_type="text/event-stream", headers=headers)
        return result

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
    explorer: ExplorerRouter,
    synthetic_explorer: ExplorerRouter,
    cost_router: LearnedModelRouter,
    nonstream: NonstreamCall,
    stream: StreamCall,
    router_context_safety_margin: int = 8192,
    state_store: Any | None = None,
) -> Any:
    """HTTP entry-point. Pulls session_id + api_key off the Request, then
    hands off to _dispatch_internal for the rewrite + dispatch logic.
    """
    pinned_now = pin_state.get()
    session_id = _session_id_from_request(request, pinned=pinned_now)
    api_key: ApiKey | None = getattr(request.state, "api_key", None)
    user_id = api_key.user_id if api_key is not None else None
    api_key_id = api_key.id if api_key is not None else None
    return await _dispatch_internal(
        body,
        route_name=route_name,
        backends_list=backends_list,
        pin_state=pin_state,
        session_registry=session_registry,
        usage_log=usage_log,
        explorer=explorer,
        synthetic_explorer=synthetic_explorer,
        cost_router=cost_router,
        nonstream=nonstream,
        stream=stream,
        session_id=session_id,
        user_id=user_id,
        api_key_id=api_key_id,
        router_context_safety_margin=router_context_safety_margin,
        state_store=state_store,
    )


async def _dispatch_internal(
    body: dict[str, Any],
    *,
    route_name: str,
    backends_list: Sequence[Backend],
    pin_state: PinState,
    session_registry: SessionRegistry,
    usage_log: UsageLog | None,
    explorer: ExplorerRouter,
    synthetic_explorer: ExplorerRouter,
    cost_router: LearnedModelRouter,
    nonstream: NonstreamCall,
    stream: StreamCall,
    session_id: str | None,
    user_id: int | None,
    api_key_id: int | None,
    forced_backend_id: str | None = None,
    router_context_safety_margin: int = 8192,
    state_store: Any | None = None,
) -> Any:
    """Dispatch core, no Request dependency. Used by the HTTP entry-points and
    by the synthetic-request worker.

    `forced_backend_id` constrains the selector to a single backend (the
    synthetic worker uses this to land synthetics on the account it picked).
    HTTP entry-points never set it.
    """
    requested_model = _require_model(body)
    requested_reasoning = _extract_reasoning_effort(body)
    routing_mode = "pass-through"

    # Look up current session context size for context-safe routing
    session_prompt_tokens: int | None = None
    if session_id is not None and usage_log is not None:
        session_prompt_tokens = usage_log.last_session_prompt_tokens(session_id)

    # Virtual model rewrite. The body is mutated in place so the downstream
    # selector and backend see the resolved (model, reasoning) pair.
    if requested_model == "auto-learning":
        # Per-request learned model routing: occasionally route to cost_router instead of explorer
        # to test the learned model and keep it fresh against changing landscape.
        learned_model_cap = _learned_model_cap_pct(
            startup_timestamp=state_store.get_proxy_startup_timestamp() if state_store is not None else None,
            model_release_timestamp=state_store.get_model_release_timestamp() if state_store is not None else None,
        )
        should_use_learned_model = random.randint(0, 100) < learned_model_cap

        if should_use_learned_model and cost_router._model is not None and cost_router._model.is_ready:
            # Route to cost_router (cost-optimal) for this request
            try:
                decision = cost_router.choose(
                    model_hint=None,
                    session_prompt_tokens=session_prompt_tokens,
                    router_context_safety_margin=router_context_safety_margin,
                )
                body["model"] = decision.cell.model
                body.setdefault("reasoning", {})["effort"] = decision.cell.reasoning_effort
                routing_mode = "auto"
            except LearnedModelRouter.NotTrained:
                # Fallback to explorer if cost_router raises (shouldn't happen with is_ready check)
                decision = explorer.choose(
                    session_prompt_tokens=session_prompt_tokens,
                    router_context_safety_margin=router_context_safety_margin,
                )
                body["model"] = decision.cell.model
                body.setdefault("reasoning", {})["effort"] = decision.cell.reasoning_effort
                routing_mode = "auto-learning"
        else:
            # Route to explorer (data collection) for this request
            decision = explorer.choose(
                session_prompt_tokens=session_prompt_tokens,
                router_context_safety_margin=router_context_safety_margin,
            )
            body["model"] = decision.cell.model
            body.setdefault("reasoning", {})["effort"] = decision.cell.reasoning_effort
            routing_mode = "auto-learning"

        # Inject complexity classification instruction only for exploration requests
        # (not for exploitation routing, which uses learned costs)
        if routing_mode == "auto-learning":
            body.setdefault("instructions", "")
            if body["instructions"]:
                body["instructions"] += "\n\n" + _COMPLEXITY_CLASSIFIER_INSTRUCTION
            else:
                body["instructions"] = _COMPLEXITY_CLASSIFIER_INSTRUCTION
            # Signal response handlers to extract and strip the {{{N}}} marker
            _extract_complexity_context.set(True)
            _complexity_class_context.set(None)
    elif requested_model == "auto-learning-synthetic":
        # Constrain the cell grid to models the forced backend advertises.
        # Without this, choose() may pick a model from the union-of-all-backends
        # that the forced backend doesn't serve, causing _no_viable at dispatch time.
        allowed: frozenset[str] | None = None
        if forced_backend_id is not None:
            fb = next((b for b in backends_list if b.id == forced_backend_id), None)
            if fb is not None:
                allowed = frozenset(fb.advertised_models)
        decision = synthetic_explorer.choose(
            allowed_models=allowed,
            session_prompt_tokens=session_prompt_tokens,
            router_context_safety_margin=router_context_safety_margin,
        )
        body["model"] = decision.cell.model
        body.setdefault("reasoning", {})["effort"] = decision.cell.reasoning_effort
        routing_mode = "auto-learning-synthetic"
    elif requested_model == "auto":
        try:
            decision = cost_router.choose(
                model_hint=None,
                session_prompt_tokens=session_prompt_tokens,
                router_context_safety_margin=router_context_safety_margin,
            )
        except LearnedModelRouter.NotTrained as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        body["model"] = decision.cell.model
        body.setdefault("reasoning", {})["effort"] = decision.cell.reasoning_effort
        routing_mode = "auto"
    model = _require_model(body)
    pinned = pin_state.get()
    active = _active_pool(backends_list, pinned)
    if forced_backend_id is not None:
        active = [b for b in active if b.id == forced_backend_id]
        if not active:
            raise HTTPException(
                status_code=503,
                detail=f"forced backend {forced_backend_id!r} not in active pool",
            )
    preferred_id = session_registry.get(session_id) if session_id is not None else None
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
            requested_model=requested_model,
            requested_reasoning_effort=requested_reasoning,
            routing_mode=routing_mode,
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
        requested_model=requested_model,
        requested_reasoning_effort=requested_reasoning,
        routing_mode=routing_mode,
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
    requested_model: str | None = None,
    requested_reasoning_effort: str | None = None,
    routing_mode: str = "pass-through",
) -> dict[str, Any]:
    excluded: set[str] = set()
    excluded_errors: dict[str, BackendError] = {}
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
                requested_model=requested_model,
                requested_reasoning_effort=requested_reasoning_effort,
                routing_mode=routing_mode,
                prompt_complexity_class=None,
            )
            last_error = exc
            if exc.classification not in RETRYABLE:
                raise _terminal_http(exc) from exc
            excluded.add(backend.id)
            excluded_errors[backend.id] = exc
            continue
        ts_end = time.time()
        _remember_binding(session_registry, session_id, backend.id)

        # Always extract and strip complexity markers from responses
        # (auto-learning requests track the class; regular requests just clean the output)
        complexity_class, result = _extract_and_strip_complexity(result)
        if _extract_complexity_context.get():
            _complexity_class_context.set(complexity_class)

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
            requested_model=requested_model,
            requested_reasoning_effort=requested_reasoning_effort,
            routing_mode=routing_mode,
            prompt_complexity_class=_complexity_class_context.get(),
        )
        return result

    # All primary backends exhausted. Try fallback strategies.
    error_classifications = {
        backend_id: error.classification
        for backend_id, error in excluded_errors.items()
    }

    if should_attempt_fallback(error_classifications):
        fallback = FallbackExecutor()
        # TODO: Implement fallback retry logic here
        # For now, just log that we attempted it
        fallback.record_attempt(
            "considered_fallback",
            "skipped",
            reason="fallback_not_yet_implemented",
        )
        raise _no_viable(
            model=model,
            last_error=last_error,
            excluded_backends=excluded_errors,
            fallback_executor=fallback,
        )

    raise _no_viable(
        model=model,
        last_error=last_error,
        excluded_backends=excluded_errors,
        fallback_executor=None,
    )


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
    requested_model: str | None = None,
    requested_reasoning_effort: str | None = None,
    routing_mode: str = "pass-through",
) -> StreamingResponse:
    excluded: set[str] = set()
    excluded_errors: dict[str, BackendError] = {}
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
                requested_model=requested_model,
                requested_reasoning_effort=requested_reasoning_effort,
                routing_mode=routing_mode,
                prompt_complexity_class=None,
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
                requested_model=requested_model,
                requested_reasoning_effort=requested_reasoning_effort,
                routing_mode=routing_mode,
                prompt_complexity_class=None,
            )
            last_error = exc
            if exc.classification not in RETRYABLE:
                raise _terminal_http(exc) from exc
            excluded.add(backend.id)
            excluded_errors[backend.id] = exc
            continue
        _remember_binding(session_registry, session_id, backend.id)

        # Always extract and strip complexity markers from streaming responses
        stream = _prepend(first_chunk, iterator)
        stream = _extract_complexity_from_stream(stream)

        # Wrap stream with safe error handling for peer disconnections
        stream = _safe_stream(stream, backend_id=backend.id)

        return StreamingResponse(
            _log_on_complete(
                stream,
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
                requested_model=requested_model,
                requested_reasoning_effort=requested_reasoning_effort,
                routing_mode=routing_mode,
            ),
            media_type="text/event-stream",
        )

    # All primary backends exhausted. Try fallback strategies.
    error_classifications = {
        backend_id: error.classification
        for backend_id, error in excluded_errors.items()
    }

    if should_attempt_fallback(error_classifications):
        fallback = FallbackExecutor()
        # TODO: Implement fallback retry logic here
        # For now, just log that we attempted it
        fallback.record_attempt(
            "considered_fallback",
            "skipped",
            reason="fallback_not_yet_implemented",
        )
        raise _no_viable(
            model=model,
            last_error=last_error,
            excluded_backends=excluded_errors,
            fallback_executor=fallback,
        )

    raise _no_viable(
        model=model,
        last_error=last_error,
        excluded_backends=excluded_errors,
        fallback_executor=None,
    )


async def _prepend(first: bytes, rest: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    yield first
    async for chunk in rest:
        yield chunk


async def _safe_stream(
    source: AsyncIterator[bytes], backend_id: str = "unknown"
) -> AsyncIterator[bytes]:
    """Safely stream chunks, gracefully handling peer disconnections.

    When a peer closes connection without completing the response body,
    log the error but don't crash the ASGI app. The client sees the partial
    response (already sent headers are committed).
    """
    try:
        async for chunk in source:
            yield chunk
    except Exception as exc:
        # Peer closed connection or other streaming error
        ts = _utc_timestamp()
        logger.warning(
            f"[{ts}] streaming error from {backend_id}: {type(exc).__name__}: {exc}"
        )
        # Don't re-raise; client already got partial response. Just stop streaming.


async def _extract_complexity_from_stream(
    source: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Extract and strip complexity marker from SSE stream's first content chunk.

    For streaming responses, the {{{N}}} marker appears at the start of the first
    delta message. This function buffers until it finds the marker, extracts it,
    then streams the rest transparently.

    Handles both chat completions format (choices[0].delta.content) and Responses API
    format (response.output_text.delta events with delta field).

    Stores the extracted complexity class in _complexity_class_context.
    Preserves all lines in multi-line chunks; only the marker-containing line is modified.
    """
    complexity_found = False
    async for chunk in source:
        if not complexity_found:
            try:
                chunk_text = chunk.decode("utf-8")
                lines = chunk_text.split("\n")
                new_lines = []
                for line in lines:
                    if not complexity_found and line.startswith("data: "):
                        json_str = line[6:]
                        if json_str.strip() == "[DONE]":
                            complexity_found = True
                            new_lines.append(line)
                            continue
                        try:
                            data = json.loads(json_str)
                            # Try chat completions format first
                            choices = data.get("choices")
                            if choices and len(choices) > 0:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content") or ""
                                if content:
                                    complexity_class, cleaned = _extract_complexity_class(content)
                                    complexity_found = True
                                    if complexity_class is not None:
                                        _complexity_class_context.set(complexity_class)
                                    delta["content"] = cleaned
                                    choices[0]["delta"] = delta
                                    data["choices"] = choices
                                    new_lines.append("data: " + json.dumps(data))
                                    continue
                            # Try Responses API format (response.output_text.delta events)
                            event_type = data.get("type")
                            if event_type == "response.output_text.delta":
                                delta_text = data.get("delta") or ""
                                if delta_text:
                                    complexity_class, cleaned = _extract_complexity_class(delta_text)
                                    complexity_found = True
                                    if complexity_class is not None:
                                        _complexity_class_context.set(complexity_class)
                                    data["delta"] = cleaned
                                    new_lines.append("data: " + json.dumps(data))
                                    continue
                        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                            complexity_found = True  # Stop searching; don't loop on malformed chunks
                    new_lines.append(line)
                chunk = "\n".join(new_lines).encode("utf-8")
            except (UnicodeDecodeError, AttributeError):
                pass
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
    requested_model: str | None = None,
    requested_reasoning_effort: str | None = None,
    routing_mode: str = "pass-through",
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
        requested_model=requested_model,
        requested_reasoning_effort=requested_reasoning_effort,
        routing_mode=routing_mode,
        prompt_complexity_class=_complexity_class_context.get(),
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
    requested_model: str | None = None,
    requested_reasoning_effort: str | None = None,
    routing_mode: str = "pass-through",
    prompt_complexity_class: int | None = None,
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
        requested_model=requested_model,
        requested_reasoning_effort=requested_reasoning_effort,
        routing_mode=routing_mode,
        prompt_complexity_class=prompt_complexity_class,
    )
    request_id = usage_log.record(entry)
    # Store request_id in context for response handlers to access
    _request_id_context.set(request_id)


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


def _quota_to_dict(q: Any) -> dict[str, Any] | None:
    """Serialize a CodexQuotaSnapshot to a JSON-friendly dict, or None if no
    snapshot is available yet. Backend type is loosely typed because only the
    codex_auth_vault backend produces snapshots; everything else returns None.
    """
    if q is None:
        return None
    return {
        "plan_type": q.plan_type,
        "active_limit": q.active_limit,
        "five_hourly_used_percent": q.five_hourly_used_percent,
        "weekly_used_percent": q.weekly_used_percent,
        "five_hourly_window_minutes": q.five_hourly_window_minutes,
        "weekly_window_minutes": q.weekly_window_minutes,
        "five_hourly_reset_at": q.five_hourly_reset_at,
        "weekly_reset_at": q.weekly_reset_at,
        "five_hourly_reset_after_seconds": q.five_hourly_reset_after_seconds,
        "weekly_reset_after_seconds": q.weekly_reset_after_seconds,
        "five_hourly_over_weekly_limit_percent": q.five_hourly_over_weekly_limit_percent,
        "credits_balance": q.credits_balance,
        "credits_has_credits": q.credits_has_credits,
        "credits_unlimited": q.credits_unlimited,
        "observed_at": q.observed_at,
    }


def _bearer(request: Request) -> str | None:
    raw = request.headers.get("authorization")
    if raw is None:
        return None
    parts = raw.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def _path_requires_api_key(path: str) -> bool:
    """Routes the bearer middleware enforces when auth is enabled."""
    return path.startswith("/v1/") or path.startswith("/diagnose/")


# Tiny prompt + instructions the diagnostic uses. Kept short to limit quota
# burn. The Codex Responses API requires `instructions` to be present and
# non-empty; an absent or empty value gets rejected with "Instructions are
# required" upstream.
_DIAGNOSE_PROMPT = "say only: ok"
_DIAGNOSE_INSTRUCTIONS = "You are a smoke-test probe. Reply minimally."


async def _diagnose_backend(backend: Backend) -> dict[str, Any]:
    """Send one minimal streaming request and check upstream contract holds."""
    if not backend.advertised_models:
        return {
            "id": backend.id,
            "ok": False,
            "skipped": False,
            "stage": "config",
            "reason": "backend has no advertised_models",
        }
    usage = await backend.usage_snapshot()
    now = time.time()
    if usage.cooldown_until_ts is not None and usage.cooldown_until_ts > now:
        return {
            "id": backend.id,
            "ok": True,
            "skipped": True,
            "stage": "cooldown",
            "reason": "backend in cooldown; not probed",
            "cooldown_until_ts": usage.cooldown_until_ts,
        }
    # Pick a real model for the probe — skip virtual selectors like
    # `auto-fallback`/`auto-learning`, which would route through the picker
    # rather than exercise the upstream contract directly.
    real_models = sorted(m for m in backend.advertised_models if m not in VIRTUAL_MODELS)
    if not real_models:
        return {
            "id": backend.id,
            "ok": False,
            "skipped": False,
            "stage": "config",
            "reason": "backend advertises only virtual models",
        }
    model = real_models[0]
    handle = CallHandle()
    body = {
        "model": model,
        "instructions": _DIAGNOSE_INSTRUCTIONS,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": _DIAGNOSE_PROMPT}],
            }
        ],
        "stream": True,
        "store": False,
    }
    try:
        async for _chunk in backend.responses_stream(body, handle):
            pass
    except BackendError as exc:
        return {
            "id": backend.id,
            "ok": False,
            "skipped": False,
            "stage": "upstream",
            "classification": exc.classification,
            "status_code": exc.status_code,
            "reason": exc.message or exc.classification,
            "model": model,
        }
    return _evaluate_diagnostic(backend.id, handle, model, kind=backend.kind)


class _PeriodicSmokeTester:
    """Background asyncio task that re-runs the smoke test on a fixed interval
    so operators see live backend state (weekly resets, auth refreshes, model
    catalog churn) without restarting the proxy.

    The startup pass runs synchronously in `lifespan` before connections are
    accepted, so the operator sees current state immediately on launch. This
    class only handles the recurring follow-up ticks. interval_s=0 disables.
    """

    def __init__(
        self, *, backends: Sequence[Backend], interval_s: int, state_store: Any | None = None
    ) -> None:
        self._backends = list(backends)
        self._interval_s = interval_s
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._state_store = state_store

    @property
    def enabled(self) -> bool:
        return self._interval_s > 0 and bool(self._backends)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="periodic-smoke-test")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            # Sleep BEFORE the first periodic tick. The startup pass already
            # ran synchronously; the first re-run should land an interval later.
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval_s)
            except TimeoutError:
                pass
            else:
                # Stop event fired during the wait → exit cleanly.
                return
            try:
                # Refresh dynamic model lists and detect new models
                for backend in self._backends:
                    refresh = getattr(backend, "refresh_advertised_models", None)
                    if refresh is not None:
                        try:
                            models_before = set(backend.advertised_models)
                            await refresh()
                            models_after = set(backend.advertised_models)
                            if models_after > models_before and self._state_store is not None:
                                new_models = models_after - models_before
                                logger.info(
                                    "new models detected in periodic refresh for backend %r: %s",
                                    backend.id, sorted(new_models)
                                )
                                self._state_store.set_model_release_timestamp(time.time())
                        except Exception:
                            logger.exception("periodic model-list refresh failed for %r", backend.id)
                logger.info("periodic smoke test cycle")
                await _run_startup_smoke_test(self._backends)
            except Exception:
                logger.exception("periodic smoke test cycle failed")


async def _run_startup_smoke_test(backends_list: Sequence[Backend]) -> None:
    """Probe each non-cooldown backend with one minimal upstream call so the
    operator sees auth health (or quota exhaustion, or any other backend
    issue) immediately in the launch log. Doesn't fail startup — just logs.

    Reuses _diagnose_backend so the smoke test and the on-demand
    /diagnose/upstream endpoint share contract definitions.
    """
    logger.info("startup smoke test: probing %d backend(s)", len(backends_list))
    results = []
    for backend in backends_list:
        try:
            result = await _diagnose_backend(backend)
        except Exception as exc:
            logger.exception("startup smoke test: backend %r raised", backend.id)
            results.append(
                {"id": backend.id, "ok": False, "stage": "exception", "reason": str(exc)}
            )
            continue
        results.append(result)
        _log_smoke_result(result)
    n_ok = sum(1 for r in results if r.get("ok"))
    n_skipped = sum(1 for r in results if r.get("skipped"))
    n_failed = len(results) - n_ok - n_skipped
    logger.info(
        "startup smoke test: %d ok, %d skipped (cooldown), %d failed",
        n_ok - n_skipped,  # 'ok' counts skipped too in _diagnose_backend's contract
        n_skipped,
        n_failed,
    )


def _log_smoke_result(result: dict[str, Any]) -> None:
    """One human-readable line per backend smoke test result."""
    backend_id = result.get("id", "?")
    if result.get("skipped"):
        logger.info("  [%s] SKIPPED — %s", backend_id, result.get("reason", "in cooldown"))
        return
    if result.get("ok"):
        upstream = result.get("upstream_status")
        logger.info("  [%s] OK — upstream %s", backend_id, upstream)
        return
    stage = result.get("stage", "?")
    classification = result.get("classification")
    status_code = result.get("status_code")
    reason = result.get("reason") or result.get("failed_checks") or "(no reason)"
    bits = [f"stage={stage}"]
    if classification is not None:
        bits.append(f"class={classification}")
    if status_code is not None:
        bits.append(f"status={status_code}")
    bits.append(f"detail={reason}")
    logger.warning("  [%s] FAILED — %s", backend_id, " ".join(bits))


def _evaluate_diagnostic(
    backend_id: str, handle: CallHandle, model: str, *, kind: str = "codex_auth_vault"
) -> dict[str, Any]:
    """Per-backend-kind check over a CallHandle from a diagnostic request.

    `codex_auth_vault` backends require Codex-shape contract guarantees —
    quota headers + Responses-API SSE terminal events. Non-Codex backends
    (e.g. `openrouter_free`) only need to confirm a 2xx came back; they
    don't carry Codex-specific headers and the contract definition differs.
    """
    if kind == "codex_auth_vault":
        return _evaluate_codex_diagnostic(backend_id, handle, model)
    return _evaluate_generic_diagnostic(backend_id, handle, model)


def _evaluate_codex_diagnostic(backend_id: str, handle: CallHandle, model: str) -> dict[str, Any]:
    summary = handle.stream_summary
    completed = summary.completed_response if summary is not None else None
    usage_block = completed.get("usage") if isinstance(completed, dict) else None
    quota = handle.quota_after
    checks = {
        "http_2xx": handle.upstream_status is not None and 200 <= handle.upstream_status < 300,
        "quota_headers_present": quota is not None,
        "five_hourly_used_percent_present": quota is not None
        and quota.five_hourly_used_percent is not None,
        "weekly_used_percent_present": quota is not None and quota.weekly_used_percent is not None,
        "response_completed_event_present": completed is not None,
        "usage_block_present": isinstance(usage_block, dict),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "id": backend_id,
        "ok": not failed,
        "skipped": False,
        "stage": "evaluate",
        "model": model,
        "checks": checks,
        "failed_checks": failed,
        "upstream_status": handle.upstream_status,
    }


def _evaluate_generic_diagnostic(backend_id: str, handle: CallHandle, model: str) -> dict[str, Any]:
    """For non-Codex backends the contract is much weaker — only verify a 2xx
    came back. Stream-summary-style checks are Codex-specific (the OpenRouter
    backend yields synthetic SSE events that don't flow through
    ResponsesStreamCollector, so requiring stream_summary here would be a
    false negative).
    """
    http_2xx = handle.upstream_status is not None and 200 <= handle.upstream_status < 300
    checks = {"http_2xx": http_2xx}
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "id": backend_id,
        "ok": not failed,
        "skipped": False,
        "stage": "evaluate",
        "model": model,
        "checks": checks,
        "failed_checks": failed,
        "upstream_status": handle.upstream_status,
    }


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


_STATIC_DIR = Path(__file__).parent / "static"


def _install_web_ui(app: FastAPI) -> None:
    """Serve a small browser UI at /ui/ for users who don't want to curl
    the /auth/* endpoints by hand. Single-page, vanilla HTML+JS, no build
    step. Calls the same /auth/* JSON endpoints the curl flow uses; the
    browser stores the session token in localStorage. Only mounted when
    auth is enabled — without auth there's nothing to register or log in
    against.
    """
    index = _STATIC_DIR / "index.html"

    @app.get("/ui", include_in_schema=False)
    async def ui_root_redirect() -> FileResponse:
        return FileResponse(index, media_type="text/html")

    @app.get("/ui/", include_in_schema=False)
    async def ui_root() -> FileResponse:
        return FileResponse(index, media_type="text/html")


def _terminal_http(exc: BackendError) -> HTTPException:
    status = exc.status_code or _EXHAUSTED_STATUS.get(exc.classification, 500)
    return HTTPException(status_code=status, detail=exc.message or exc.classification)


def _no_viable(
    *,
    model: str,
    last_error: BackendError | None,
    excluded_backends: dict[str, BackendError] | None = None,
    fallback_executor: FallbackExecutor | None = None,
) -> HTTPException:
    """Log all failed backends and return appropriate error."""
    if excluded_backends:
        # Build error classification map for logging
        error_classifications = {
            backend_id: error.classification
            for backend_id, error in excluded_backends.items()
        }

        if fallback_executor:
            fallback_executor.log_final_exhaustion(model, error_classifications)
        else:
            # Fallback not attempted, log basic info
            failures = []
            for backend_id, error in excluded_backends.items():
                failures.append(f"{backend_id}: {error.classification}")
            ts = _utc_timestamp()
            logger.warning(
                f"[{ts}] all backends exhausted for model {model!r}. "
                f"Failures: {'; '.join(failures)}"
            )

    if last_error is None:
        return HTTPException(status_code=503, detail=f"no viable backend for model {model!r}")
    status = _EXHAUSTED_STATUS.get(last_error.classification, 502)
    return HTTPException(
        status_code=status,
        detail=last_error.message or last_error.classification,
    )
