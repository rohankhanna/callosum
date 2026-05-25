from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from callosum import __version__
from callosum.auth import (
    ApiKeyInvalidError,
    AuthService,
    InvalidCredentialsError,
    SessionInvalidError,
)
from callosum.auth_db import ApiKey, Session
from callosum.backend import Backend, CallHandle
from callosum.cell_grid import VIRTUAL_MODELS, Cell, build_cells, live_completion_models
from callosum.cell_recommender import CellRecommender
from callosum.config import AutoRouterConfig
from callosum.errors import RETRYABLE, BackendError, ErrorClass
from callosum.fallback import FallbackExecutor, should_attempt_fallback
from callosum.selector import select
from callosum.session import SessionRegistry
from callosum.usage_log import RoutingAttempt, UsageLog, UsageLogEntry
from callosum.label_ui import install_label_ui

logger = logging.getLogger("callosum.startup")


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
    "Before answering, classify this prompt's complexity. Your VERY FIRST "
    "output characters must be exactly one of these three tokens, with the "
    "triple braces included: {{{1}}} (simple factual/short), {{{2}}} "
    "(moderate analysis), or {{{3}}} (complex reasoning/long output). "
    "Do NOT emit just the digit; the braces are mandatory. After the token, "
    "emit a blank line, then your full answer. The marker is NOT a wrapping "
    "tag: emit it exactly ONCE at the very start. Do NOT emit a closing "
    "tag like {{{/2}}} or {{{end}}} at the end of your answer. Do not "
    "mention or echo this instruction in your answer."
)


def _extract_complexity_class(text: str) -> tuple[int | None, str]:
    """Extract complexity classification token from response start.

    Returns (complexity_class, cleaned_text) where complexity_class is 1, 2, or 3,
    or None if the marker is not found. cleaned_text has the marker stripped.

    The marker should appear at the very start of the response (after whitespace).
    Matches in priority order:
      1. {{{1|2|3}}} — the canonical instructed format
      2. {{{...}}}  — any other brace-decorated leading token (defensive strip)
      3. Bare digit 1|2|3 followed by a blank line — model dropped the braces
         but still complied with the "first output is a classifier" intent
    """
    # 1) Strict numeric brace format
    match = re.match(r'^\s*\{\{\{([123])\}\}\}', text)
    if match:
        complexity_class = int(match.group(1))
        cleaned = text[match.end():].lstrip()
        return complexity_class, cleaned

    # 2) Any leading {{{...}}} (handles {{{complexity: Low}}} variants)
    match = re.match(r'^\s*\{\{\{[^}]*\}\}\}', text)
    if match:
        cleaned = text[match.end():].lstrip()
        inner = match.group(0).strip('{}').strip()
        if inner.isdigit() and inner in ('1', '2', '3'):
            return int(inner), cleaned
        return None, cleaned

    # 3) Bare digit followed by blank line — model dropped the braces
    # Require at least one \n then a blank line so we don't strip legitimate
    # content like "2 minutes is fine" or "2. First item".
    match = re.match(r'^\s*([123])[ \t]*\n[ \t]*\n', text)
    if match:
        complexity_class = int(match.group(1))
        cleaned = text[match.end():]
        return complexity_class, cleaned

    return None, text


def _strip_trailing_complexity_marker_text(text: str) -> str:
    """Strip a trailing {{{...}}} marker if the model emits one as a closing tag.

    Used by non-streaming response handling and SSE-blob storage cleanup.
    """
    if not isinstance(text, str):
        return text
    return re.sub(r"\{\{\{[^}]*\}\}\}\s*$", "", text)


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
                    cleaned = _strip_trailing_complexity_marker_text(cleaned)
                    if complexity_class is not None or cleaned != content:
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
                    cleaned = _strip_trailing_complexity_marker_text(cleaned)
                    if complexity_class is not None or cleaned != text:
                        result["output"][0]["content"][0]["text"] = cleaned
                    return complexity_class, result
    except (KeyError, IndexError, TypeError):
        pass

    logger.warning("Could not extract complexity marker from response (unsupported format)")
    return None, result


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
    from callosum.state import StateStore

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
        backend's CURRENT advertised_models.

        Prefers per-model metadata from the upstream catalog
        (supported_in_api, visibility, priority, supported_reasoning_levels,
        context_window) when backends provide it. Falls back to the regex-
        and-static-list path for backends that don't expose metadata.

        Recomputed on every router decision — when CodexAuthVaultBackend's
        hourly catalog refresh discovers a new model, the cell grid follows
        without a proxy restart. When a model is retired upstream, it drops
        out of the grid the next time the router consults it.
        """
        from callosum.cell_grid import (
            ModelMetadata,
            build_cells_from_metadata,
        )

        # Merge metadata from every backend that exposes it. Today: Codex
        # auth-vault backends (real upstream metadata) + LiteLLM gateway
        # (synthesizes a default-shape ModelMetadata per local model). When
        # two backends advertise the same slug, keep the record with more
        # populated fields (fewer "Unknown" defaults).
        merged_metadata: dict[str, ModelMetadata] = {}
        for b in backends_list:
            if b.kind not in ("codex_auth_vault", "litellm_gateway"):
                continue
            backend_meta = getattr(b, "model_metadata", None) or {}
            for slug, m in backend_meta.items():
                existing = merged_metadata.get(slug)
                if existing is None:
                    merged_metadata[slug] = m
                else:
                    # Pick the record with more populated fields.
                    new_filled = sum(1 for f in (m.supported_in_api, m.visibility, m.priority, m.context_window) if f is not None)
                    old_filled = sum(1 for f in (existing.supported_in_api, existing.visibility, existing.priority, existing.context_window) if f is not None)
                    if new_filled > old_filled:
                        merged_metadata[slug] = m

        if merged_metadata:
            cells = build_cells_from_metadata(merged_metadata)
            if cells:
                return cells

        # Fallback path: no metadata yet (cold start) or backend doesn't
        # expose it. Use the legacy regex filter + static reasoning-level
        # enumeration so the router still operates.
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

    auto_cfg = auto_router_config if auto_router_config is not None else AutoRouterConfig()

    # Cell recommender: ask the cheapest cell to pick the cell that should
    # handle this prompt. Output IS the routing decision; the model-based
    # routing infra (cost router, explorer, synthetic ticker, complexity
    # heuristic) was removed entirely in favor of this. Only instantiated
    # when at least one backend exists — without a backend, the recommender
    # has nothing to call and dispatch falls back to the configured cheap
    # cell name.
    cell_recommender: CellRecommender | None = None
    if backends_list:
        cheap_cell = Cell(
            model=auto_cfg.cell_recommender_cheap_model,
            reasoning_effort=auto_cfg.cell_recommender_cheap_effort,
            context_window=None,
        )
        cell_recommender = CellRecommender(
            cheap_backend=backends_list[0],
            cheap_cell=cheap_cell,
            cache_max=auto_cfg.cell_recommender_cache_max,
            cache_ttl_seconds=auto_cfg.cell_recommender_cache_ttl_seconds,
            upstream_timeout_s=auto_cfg.cell_recommender_upstream_timeout_s,
        )

    smoke_tester = _PeriodicSmokeTester(
        backends=backends_list,
        interval_s=smoke_test_interval_seconds,
        state_store=state_store,
    )
    cooldown_prober = _PeriodicCooldownProber(
        backends=backends_list,
        interval_s=auto_cfg.cooldown_probe_interval_seconds,
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
        smoke_tester.start()
        cooldown_prober.start()
        try:
            yield
        finally:
            await cooldown_prober.stop()
            await smoke_tester.stop()
            for backend in backends_list:
                await backend.aclose()
            if usage_log is not None:
                usage_log.close()
            if auth_service is not None:
                auth_service.db.close()

    app = FastAPI(title="callosum", version=__version__, lifespan=lifespan)

    # Install quality labeling UI if usage_log is available
    if usage_log is not None:
        install_label_ui(app, usage_log)

    # Bearer middleware for /v1/* and /diagnose/* — only enforced when an
    # auth service is configured. In single-operator mode (no auth db) those
    # routes stay open.
    @app.middleware("http")
    async def _enforce_api_key(request: Request, call_next: Callable[[Request], Any]) -> Any:
        if auth_service is None or not _path_requires_api_key(request.url.path):
            return await call_next(request)
        plaintext = _bearer(request)
        if plaintext is None:
            logger.warning(
                "auth 401: no bearer token on %s %s",
                request.method, request.url.path,
            )
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Authorization: Bearer <api-key> required",
                    "reason": "missing_bearer",
                },
            )
        try:
            api_key = auth_service.resolve_api_key(plaintext)
        except ApiKeyInvalidError as exc:
            # Log + return the rejected key's PREFIX (never the secret) and how
            # many active keys exist, so a downstream tool printing the error
            # — or an operator scanning logs — can immediately tell whether
            # this is a wrong-key vs empty-auth-store situation. Auth failures
            # used to be completely silent, which made post-mortems impossible.
            prefix = plaintext[:8] if len(plaintext) >= 8 else plaintext[:4]
            active = _active_api_key_count(auth_service)
            logger.warning(
                "auth 401: %s on %s %s — rejected key prefix=%r (%d active keys registered)",
                exc, request.method, request.url.path, prefix, active,
            )
            return JSONResponse(
                status_code=401,
                content={
                    "detail": str(exc),
                    "reason": "key_not_recognized",
                    "rejected_key_prefix": prefix,
                    "active_keys_registered": active,
                },
            )
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
        # Recommender state: cache hit rate, upstream-call counts, and
        # which cells the recommender has been picking. Lets the operator
        # see at a glance if the cheap classifier is biased (e.g. always
        # picking itself, or always picking the largest) and whether
        # off-peak comparison sampling has fired.
        recommender_block: dict[str, Any] = {
            "enabled": cell_recommender is not None,
            "cheap_cell": (
                f"{auto_cfg.cell_recommender_cheap_model} "
                f"{auto_cfg.cell_recommender_cheap_effort}"
            ),
            "alternative_classifier_pct": auto_cfg.cell_recommender_alternative_classifier_pct,
            "stats": {},
            "recommendation_counts": {},
            "classifier_call_counts": {},
        }
        if cell_recommender is not None:
            stats = cell_recommender.stats
            calls = max(stats.get("calls", 0), 1)
            recommender_block["stats"] = {
                **stats,
                "cache_hit_rate": round(stats.get("cache_hits", 0) / calls, 3),
                "fallback_rate": round(stats.get("fallback_count", 0) / calls, 3),
                "alternative_rate": round(stats.get("alternative_calls", 0) / calls, 3),
            }
            recommender_block["recommendation_counts"] = (
                cell_recommender.recommendation_counts
            )
            recommender_block["classifier_call_counts"] = (
                cell_recommender.classifier_call_counts
            )
        return {
            "backends": entries,
            "pinned": pin_state.get(),
            "sessions": session_registry.snapshot(),
            "recommender": recommender_block,
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

    @app.post("/control/clear-cooldown/{backend_id}")
    async def control_clear_cooldown(backend_id: str) -> dict[str, Any]:
        """Operator override: clear a stale cooldown on one backend.

        Companion to the periodic cooldown prober (`_PeriodicCooldownProber`)
        for cases where you'd rather not wait an interval for the next probe
        — e.g. you know out of band that the account was just topped up.
        Returns 404 if no backend with that id is configured.
        """
        backend = next((b for b in backends_list if b.id == backend_id), None)
        if backend is None:
            raise HTTPException(
                status_code=404, detail=f"backend {backend_id!r} not in pool"
            )
        clear = getattr(backend, "clear_cooldown", None)
        if clear is None:
            raise HTTPException(
                status_code=400,
                detail=f"backend {backend_id!r} does not support clear_cooldown",
            )
        snap = clear()
        return {
            "id": backend_id,
            "cleared": True,
            "snapshot": {
                "cooldown_until_ts": snap.cooldown_until_ts,
                "weekly_exhausted": snap.weekly_exhausted,
                "probed_at_ts": snap.probed_at_ts,
            },
        }

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
                    "owned_by": "callosum",
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
            "service": "callosum",
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
                    "owned_by": "callosum",
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
            logging.getLogger("callosum.app").warning("feedback record failed: %s", exc)
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
            nonstream=lambda b, p, h: b.chat_completions(p, h),
            stream=lambda b, p, h: b.chat_completions_stream(p, h),
            router_context_safety_margin=auto_cfg.router_context_safety_margin,
            state_store=state_store,
            auto_cfg=auto_cfg,
            cell_recommender=cell_recommender,
            live_cells_fn=_live_cells,
        )
        # Add X-Proxy-Request-ID header if a request was logged
        request_id = _request_id_context.get()
        if request_id is not None:
            headers = {"X-Proxy-Request-ID": str(request_id)}
            if isinstance(result, dict):
                return JSONResponse(result, headers=headers)
            # Result is already a StreamingResponse from _dispatch_stream
            if isinstance(result, StreamingResponse):
                result.headers.update(headers)
                return result
            # Fallback: shouldn't reach here, but wrap just in case
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
            nonstream=lambda b, p, h: b.responses(p, h),
            stream=lambda b, p, h: b.responses_stream(p, h),
            router_context_safety_margin=auto_cfg.router_context_safety_margin,
            state_store=state_store,
            auto_cfg=auto_cfg,
            cell_recommender=cell_recommender,
            live_cells_fn=_live_cells,
        )
        # Add X-Proxy-Request-ID header if a request was logged
        request_id = _request_id_context.get()
        if request_id is not None:
            headers = {"X-Proxy-Request-ID": str(request_id)}
            if isinstance(result, dict):
                return JSONResponse(result, headers=headers)
            # Result is already a StreamingResponse from _dispatch_stream
            if isinstance(result, StreamingResponse):
                result.headers.update(headers)
                return result
            # Fallback: shouldn't reach here, but wrap just in case
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
    nonstream: NonstreamCall,
    stream: StreamCall,
    router_context_safety_margin: int = 8192,
    state_store: Any | None = None,
    auto_cfg: AutoRouterConfig | None = None,
    cell_recommender: CellRecommender | None = None,
    live_cells_fn: Callable[[], list[Cell]] | None = None,
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
        nonstream=nonstream,
        stream=stream,
        session_id=session_id,
        user_id=user_id,
        api_key_id=api_key_id,
        router_context_safety_margin=router_context_safety_margin,
        state_store=state_store,
        auto_cfg=auto_cfg,
        cell_recommender=cell_recommender,
        live_cells_fn=live_cells_fn,
    )


def _maybe_fire_comparison_sampling(
    body: dict[str, Any],
    *,
    cell_recommender: CellRecommender | None,
    backends_list: Sequence[Backend],
    live_cells_fn: Callable[[], list[Cell]],
    compare_pct: float,
    compare_max_weekly_pct: float,
) -> None:
    """Maybe spawn a fire-and-forget comparison-sampling task.

    Off-peak gating: only fires when this request was randomly selected
    (pct) AND the first backend's weekly_used% is below the configured
    ceiling (don't burn quota during peak). Pure observation — fires the
    same prompt at every live cell as a classifier and logs each one's
    recommendation, but never affects the routing decision for the
    actual user request currently in flight.

    Logs are structured: easy to grep + parse for later analysis of
    "did the cheap classifier disagree with bigger ones?".
    """
    if cell_recommender is None:
        return
    if compare_pct <= 0:
        return
    if random.random() >= compare_pct:
        return
    if not backends_list:
        return

    async def _run() -> None:
        # Off-peak gate; check inside the task so we don't block dispatch.
        try:
            quota = await backends_list[0].quota_snapshot()
        except Exception:
            quota = None
        if quota is not None:
            wkly = getattr(quota, "weekly_used_percent", None)
            if isinstance(wkly, (int, float)) and wkly > compare_max_weekly_pct:
                return  # peak hours — skip comparison spam
        try:
            cells = live_cells_fn()
            results = await cell_recommender.fire_comparisons(
                body, allowed_cells=cells, comparison_tiers=cells
            )
            for r in results:
                logger.info(
                    "cell_recommender comparison: tier=%s/%s recommended=%s/%s latency=%.2fs error=%s",
                    r["tier_model"], r["tier_effort"],
                    r["recommended_model"], r["recommended_effort"],
                    r["latency_s"], r["error"],
                )
        except Exception:
            logger.exception("cell_recommender comparison sampling failed")

    asyncio.create_task(_run(), name="cell-recommender-comparison")


async def _dispatch_internal(
    body: dict[str, Any],
    *,
    route_name: str,
    backends_list: Sequence[Backend],
    pin_state: PinState,
    session_registry: SessionRegistry,
    usage_log: UsageLog | None,
    nonstream: NonstreamCall,
    stream: StreamCall,
    session_id: str | None,
    user_id: int | None,
    api_key_id: int | None,
    forced_backend_id: str | None = None,
    router_context_safety_margin: int = 8192,
    state_store: Any | None = None,
    auto_cfg: AutoRouterConfig | None = None,
    cell_recommender: CellRecommender | None = None,
    live_cells_fn: Callable[[], list[Cell]] | None = None,
) -> Any:
    if auto_cfg is None:
        auto_cfg = AutoRouterConfig()
    if live_cells_fn is None:
        live_cells_fn = build_cells
    """Dispatch core, no Request dependency. Used by the HTTP entry-points and
    by the synthetic-request worker.

    `forced_backend_id` constrains the selector to a single backend (the
    synthetic worker uses this to land synthetics on the account it picked).
    HTTP entry-points never set it.
    """
    requested_model = _require_model(body)
    requested_reasoning = _extract_reasoning_effort(body)
    routing_mode = "pass-through"
    # Recommender provenance for the request log. Stays None on pass-through
    # requests; populated below when the recommender fires.
    recommender_classifier_cell: str | None = None
    recommender_raw_output: str | None = None
    recommender_source: str | None = None
    cell_candidates: tuple[Cell, ...] = ()

    # Look up current session context size for context-safe routing
    session_prompt_tokens: int | None = None
    if session_id is not None and usage_log is not None:
        session_prompt_tokens = usage_log.last_session_prompt_tokens(session_id)

    # Virtual model rewrite. Every model=auto / model=auto-learning /
    # model=auto-learning-synthetic request now goes through the cell
    # recommender — the cheap upstream classifier picks which (model,
    # effort) cell should handle the prompt. The two "learning" virtual
    # names are kept as aliases for backward compatibility with clients
    # that still set them, but they no longer drive data collection
    # (model-based routing was abandoned). The body is mutated in place
    # so the downstream selector and backend see the resolved pair.
    if requested_model in ("auto", "auto-learning", "auto-learning-synthetic"):
        fallback_cell = Cell(
            model=auto_cfg.cell_recommender_cheap_model,
            reasoning_effort=auto_cfg.cell_recommender_cheap_effort,
            context_window=None,
        )
        if cell_recommender is not None:
            # Bias mitigation: with a small probability, route THIS request
            # through a non-cheap classifier rather than the default cheap
            # one. Prevents the cheap classifier's biases from being the
            # sole authority on every routing decision. The alternative
            # classifier's result intentionally bypasses the cache so it
            # doesn't poison subsequent cheap-classifier routing.
            cells_now = live_cells_fn()
            classifier_override: Cell | None = None
            alt_pct = auto_cfg.cell_recommender_alternative_classifier_pct
            if alt_pct > 0 and random.random() < alt_pct:
                cheap = (auto_cfg.cell_recommender_cheap_model,
                         auto_cfg.cell_recommender_cheap_effort)
                alternatives = [
                    c for c in cells_now
                    if (c.model, c.reasoning_effort) != cheap
                ]
                if alternatives:
                    classifier_override = random.choice(alternatives)
            rec = await cell_recommender.recommend(
                body,
                allowed_cells=cells_now,
                fallback=fallback_cell,
                classifier_cell=classifier_override,
                local_exploration_pct=auto_cfg.cell_recommender_local_exploration_pct,
            )
            chosen = rec.cell
            # Capture recommender provenance for the request log — training
            # consumers downstream filter on recommender_source IN
            # ('upstream', 'alternative') to get unbiased classifier picks.
            recommender_classifier_cell = (
                f"{rec.classifier_cell.model} {rec.classifier_cell.reasoning_effort}"
                if rec.classifier_cell is not None
                else None
            )
            # Truncate raw output to bound request-log row size; the parser
            # only ever expects a short "model effort" reply, so 500 chars
            # is plenty even for chatty classifier outputs.
            recommender_raw_output = (
                rec.raw_output[:500]
                if isinstance(rec.raw_output, str) and rec.raw_output
                else None
            )
            recommender_source = rec.source
            # Top-N (default 3) candidate cells the dispatch layer will walk
            # if the primary cell's backend pool errors. Recommender returns
            # them ordered: classifier's pick first, rest of the compatible
            # set in grid-priority order.
            cell_candidates: tuple[Cell, ...] = rec.candidates[:MAX_CELL_ATTEMPTS]
        else:
            chosen = fallback_cell
            # No recommender wired up — leave all three columns NULL so
            # the training pipeline can distinguish "no classifier ran" from
            # "classifier ran and fell back".
            recommender_classifier_cell = None
            recommender_raw_output = None
            recommender_source = None
            cell_candidates = ()
        body["model"] = chosen.model
        body.setdefault("reasoning", {})["effort"] = chosen.reasoning_effort
        # Preserve the requested virtual-model name in the log so synthetic vs
        # organic routing (and any future virtual model) can be distinguished
        # downstream. All three names route through the same recommender today,
        # but tracking which alias the client sent has observability value —
        # e.g. for measuring synthetic-tier velocity against organic traffic.
        routing_mode = requested_model
        _maybe_fire_comparison_sampling(
            body,
            cell_recommender=cell_recommender,
            backends_list=backends_list,
            live_cells_fn=live_cells_fn,
            compare_pct=auto_cfg.cell_recommender_compare_pct,
            compare_max_weekly_pct=auto_cfg.cell_recommender_compare_max_weekly_pct,
        )
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
        return await _dispatch_stream_with_cell_retry(
            body,
            candidates=cell_candidates,
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
            recommender_classifier_cell=recommender_classifier_cell,
            recommender_raw_output=recommender_raw_output,
            recommender_source=recommender_source,
        )
    return await _dispatch_nonstream_with_cell_retry(
        body,
        candidates=cell_candidates,
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
        recommender_classifier_cell=recommender_classifier_cell,
        recommender_raw_output=recommender_raw_output,
        recommender_source=recommender_source,
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


# Maximum number of cells the dispatch layer will try before giving up on
# a request. The recommender returns its primary pick at index 0 plus the
# rest of the compatible cell set in priority order; the dispatch layer
# walks them on retryable failures (5xx from a cell's backend pool). Cap
# is intentionally small — three attempts cover the common "primary cell
# is rate-limited" + "next-best cell unhealthy" sequence without blowing
# user-perceived latency. There's no wall-clock cap; latency is bounded
# only by upstream timeouts.
MAX_CELL_ATTEMPTS = 3


async def _dispatch_nonstream_with_cell_retry(
    body: dict[str, Any],
    *,
    candidates: tuple["Cell", ...],
    usage_log: UsageLog | None,
    model: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Walk up to MAX_CELL_ATTEMPTS candidate cells, rerouting on 5xx.

    Empty `candidates` short-circuits to a single _dispatch_nonstream call,
    preserving existing behavior for pass-through requests. Per-cell
    attempt history is persisted to request_routing_attempts via the
    final request_id (success row, or last failure row written by the
    inner dispatch's _log_attempt path).
    """
    cells_to_try = list(candidates[:MAX_CELL_ATTEMPTS])
    if not cells_to_try:
        return await _dispatch_nonstream(
            body, model=model, usage_log=usage_log, **kwargs
        )

    attempts: list[RoutingAttempt] = []
    final_request_id: int | None = None

    for cell_idx, cell in enumerate(cells_to_try):
        body["model"] = cell.model
        body.setdefault("reasoning", {})["effort"] = cell.reasoning_effort
        attempt_start = time.time()
        try:
            result = await _dispatch_nonstream(
                body, model=cell.model, usage_log=usage_log, **kwargs
            )
        except HTTPException as exc:
            attempt_ms = int((time.time() - attempt_start) * 1000)
            final_request_id = _request_id_context.get()
            attempts.append(
                RoutingAttempt(
                    attempt_idx=cell_idx,
                    # Cell may have spanned multiple backends inside the inner
                    # loop; backend_id at the cell level is intentionally NULL.
                    # Consumers wanting that join requests on (request_id, model).
                    backend_id=None,
                    model=cell.model,
                    reasoning_effort=cell.reasoning_effort,
                    status=exc.status_code,
                    classification=(
                        "retried_next_cell"
                        if exc.status_code >= 500
                        and cell_idx + 1 < len(cells_to_try)
                        else "failed"
                    ),
                    latency_ms=attempt_ms,
                    error_message=(str(exc.detail)[:500] if exc.detail else None),
                )
            )
            # 4xx → non-retryable (auth_invalid, malformed request, etc).
            # 5xx + cells remaining → reroute. 5xx + no cells left → propagate.
            if exc.status_code < 500 or cell_idx + 1 >= len(cells_to_try):
                if usage_log is not None and final_request_id is not None:
                    usage_log.record_routing_attempts(final_request_id, attempts)
                raise
            continue
        # Success.
        attempt_ms = int((time.time() - attempt_start) * 1000)
        final_request_id = _request_id_context.get()
        attempts.append(
            RoutingAttempt(
                attempt_idx=cell_idx,
                backend_id=None,
                model=cell.model,
                reasoning_effort=cell.reasoning_effort,
                status=200,
                classification="ok",
                latency_ms=attempt_ms,
            )
        )
        # Persist only multi-attempt histories. Single-attempt requests are the
        # common case and the requests row already tells the whole story —
        # keeping the sibling table to the reroute cohort makes "how often did
        # we have to reroute?" a one-line query.
        if (
            usage_log is not None
            and final_request_id is not None
            and len(attempts) > 1
        ):
            usage_log.record_routing_attempts(final_request_id, attempts)
        return result

    # Unreachable: the loop above either returns on success or raises after
    # the final cell. Kept for type-checker clarity.
    raise RuntimeError(
        "cell-level retry exhausted without raising"
    )  # pragma: no cover


async def _dispatch_stream_with_cell_retry(
    body: dict[str, Any],
    *,
    candidates: tuple["Cell", ...],
    usage_log: UsageLog | None,
    model: str,
    **kwargs: Any,
) -> StreamingResponse:
    """Stream-path mirror of _dispatch_nonstream_with_cell_retry.

    Reroute happens only on pre-first-chunk HTTPException from the inner
    dispatch — once StreamingResponse is committed and bytes start flowing,
    mid-stream failover stays a non-goal (per The Project Documentation). The inner
    dispatch raises HTTPException only before first-chunk; after that, it
    returns StreamingResponse and any backend failure is bubbled in-band.
    """
    cells_to_try = list(candidates[:MAX_CELL_ATTEMPTS])
    if not cells_to_try:
        return await _dispatch_stream(
            body, model=model, usage_log=usage_log, **kwargs
        )

    attempts: list[RoutingAttempt] = []
    final_request_id: int | None = None

    for cell_idx, cell in enumerate(cells_to_try):
        body["model"] = cell.model
        body.setdefault("reasoning", {})["effort"] = cell.reasoning_effort
        attempt_start = time.time()
        try:
            result = await _dispatch_stream(
                body, model=cell.model, usage_log=usage_log, **kwargs
            )
        except HTTPException as exc:
            attempt_ms = int((time.time() - attempt_start) * 1000)
            final_request_id = _request_id_context.get()
            attempts.append(
                RoutingAttempt(
                    attempt_idx=cell_idx,
                    backend_id=None,
                    model=cell.model,
                    reasoning_effort=cell.reasoning_effort,
                    status=exc.status_code,
                    classification=(
                        "retried_next_cell"
                        if exc.status_code >= 500
                        and cell_idx + 1 < len(cells_to_try)
                        else "failed"
                    ),
                    latency_ms=attempt_ms,
                    error_message=(str(exc.detail)[:500] if exc.detail else None),
                )
            )
            if exc.status_code < 500 or cell_idx + 1 >= len(cells_to_try):
                if usage_log is not None and final_request_id is not None:
                    usage_log.record_routing_attempts(final_request_id, attempts)
                raise
            continue
        attempt_ms = int((time.time() - attempt_start) * 1000)
        final_request_id = _request_id_context.get()
        attempts.append(
            RoutingAttempt(
                attempt_idx=cell_idx,
                backend_id=None,
                model=cell.model,
                reasoning_effort=cell.reasoning_effort,
                status=200,
                classification="ok",
                latency_ms=attempt_ms,
            )
        )
        if (
            usage_log is not None
            and final_request_id is not None
            and len(attempts) > 1
        ):
            usage_log.record_routing_attempts(final_request_id, attempts)
        return result

    raise RuntimeError(
        "cell-level stream retry exhausted without raising"
    )  # pragma: no cover


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
    recommender_classifier_cell: str | None = None,
    recommender_raw_output: str | None = None,
    recommender_source: str | None = None,
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
                recommender_classifier_cell=recommender_classifier_cell,
                recommender_raw_output=recommender_raw_output,
                recommender_source=recommender_source,
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
            recommender_classifier_cell=recommender_classifier_cell,
            recommender_raw_output=recommender_raw_output,
            recommender_source=recommender_source,
        )
        return result

    # All primary backends exhausted. Compute recovery timestamp.
    recovery_ts: float | None = None
    for backend in backends_list:
        try:
            usage = await backend.usage_snapshot()
            if usage and usage.cooldown_until_ts:
                if recovery_ts is None or usage.cooldown_until_ts < recovery_ts:
                    recovery_ts = usage.cooldown_until_ts
        except Exception:
            pass  # Skip backends that fail to report usage

    # Try fallback strategies.
    error_classifications = {
        backend_id: error.classification
        for backend_id, error in excluded_errors.items()
    }

    backend_status = await _collect_backend_status(backends_list)
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
            recovery_ts=recovery_ts,
            backend_status=backend_status,
        )

    raise _no_viable(
        model=model,
        last_error=last_error,
        excluded_backends=excluded_errors,
        fallback_executor=None,
        recovery_ts=recovery_ts,
        backend_status=backend_status,
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
    recommender_classifier_cell: str | None = None,
    recommender_raw_output: str | None = None,
    recommender_source: str | None = None,
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
                recommender_classifier_cell=recommender_classifier_cell,
                recommender_raw_output=recommender_raw_output,
                recommender_source=recommender_source,
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
                recommender_classifier_cell=recommender_classifier_cell,
                recommender_raw_output=recommender_raw_output,
                recommender_source=recommender_source,
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
        # Also scrub a trailing {{{...}}} marker if the model emits one as a
        # closing tag (e.g. {{{/2}}} at the very end of the answer).
        stream = _strip_trailing_complexity_marker(stream)
        # Final scrub: full-text .done events (which Hermes / codex-cli
        # often read for the final UI render). The delta filters above
        # never touch these — see _scrub_full_text_events docstring.
        stream = _scrub_full_text_events(stream)

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
                recommender_classifier_cell=recommender_classifier_cell,
                recommender_raw_output=recommender_raw_output,
                recommender_source=recommender_source,
            ),
            media_type="text/event-stream",
        )

    # All primary backends exhausted. Compute recovery timestamp.
    recovery_ts: float | None = None
    for backend in backends_list:
        try:
            usage = await backend.usage_snapshot()
            if usage and usage.cooldown_until_ts:
                if recovery_ts is None or usage.cooldown_until_ts < recovery_ts:
                    recovery_ts = usage.cooldown_until_ts
        except Exception:
            pass  # Skip backends that fail to report usage

    # Try fallback strategies.
    error_classifications = {
        backend_id: error.classification
        for backend_id, error in excluded_errors.items()
    }

    backend_status = await _collect_backend_status(backends_list)
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
            recovery_ts=recovery_ts,
            backend_status=backend_status,
        )

    raise _no_viable(
        model=model,
        last_error=last_error,
        excluded_backends=excluded_errors,
        fallback_executor=None,
        recovery_ts=recovery_ts,
        backend_status=backend_status,
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


# Responses API event types whose `delta` field carries text-like content that
# the model may inadvertently lead with the classifier marker. We strip from
# all of these. We deliberately do NOT include
# `response.function_call_arguments.delta` because that field carries
# tool-call JSON fragments — removing `{` / `}` would corrupt the JSON.
_TEXT_BEARING_DELTA_EVENT_TYPES: frozenset[str] = frozenset({
    "response.output_text.delta",
    "response.reasoning_summary_text.delta",
})

# Final / accumulated-text events. These carry the FULL response text after
# streaming completes, and Hermes / codex-cli often read from them for the
# final display (bypassing the delta stream). Marker scrubbing has to cover
# these too or a model that voluntarily echoes "{{{N}}}\n\n..." at the start
# of its response (because its conversation history is poisoned with prior
# marker-prefixed turns) will leak through to the user even though every
# delta event was stripped clean.
#
# Map: event_type -> JSON path to the string field that holds the full text.
# Path syntax is dot-separated keys, with `[N]` for list indexing.
_FULL_TEXT_EVENT_PATHS: dict[str, str] = {
    "response.output_text.done": "text",
    "response.content_part.done": "part.text",
    "response.output_item.done": "item.content[0].text",
}


def _get_at_path(obj: Any, path: str) -> Any:
    """Read a value out of a nested JSON-like dict/list using a 'a.b[0].c'
    style path. Returns None if any step is missing or wrongly typed.
    Cheap parser; not a full JSONPath impl."""
    cur = obj
    for part in path.split("."):
        # Split off any list indices like 'content[0]'
        while "[" in part and part.endswith("]"):
            head, idx_str = part[: part.index("[")], part[part.index("[") + 1 : -1]
            try:
                idx = int(idx_str)
            except ValueError:
                return None
            if head:
                cur = cur.get(head) if isinstance(cur, dict) else None
            if not isinstance(cur, list) or idx >= len(cur):
                return None
            cur = cur[idx]
            part = ""
        if part:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        if cur is None:
            return None
    return cur


def _set_at_path(obj: Any, path: str, value: Any) -> bool:
    """In-place set a value at the given path. Returns True on success."""
    cur = obj
    parts = path.split(".")
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        while "[" in part and part.endswith("]"):
            head, idx_str = part[: part.index("[")], part[part.index("[") + 1 : -1]
            try:
                idx = int(idx_str)
            except ValueError:
                return False
            if head:
                if not isinstance(cur, dict):
                    return False
                cur = cur.get(head)
            if not isinstance(cur, list) or idx >= len(cur):
                return False
            if last and "[" not in part[part.index("[") + 1 :]:
                cur[idx] = value
                return True
            cur = cur[idx]
            part = ""
        if part:
            if not isinstance(cur, dict):
                return False
            if last:
                cur[part] = value
                return True
            cur = cur.get(part)
            if cur is None:
                return False
    return False


def _scrub_full_text_event(data: dict[str, Any]) -> bool:
    """If this event is a known full-text 'done' event, strip leading +
    trailing {{{...}}} markers from its text field. Returns True if the
    event was mutated.
    """
    et = data.get("type")
    if not isinstance(et, str):
        return False
    path = _FULL_TEXT_EVENT_PATHS.get(et)
    if path is None:
        return False
    text = _get_at_path(data, path)
    if not isinstance(text, str) or not text:
        return False
    # Reuse the existing leading + trailing strippers.
    _, cleaned = _extract_complexity_class(text)
    cleaned = _strip_trailing_complexity_marker_text(cleaned)
    if cleaned == text:
        return False
    _set_at_path(data, path, cleaned)
    return True


def _event_delta_text(event_bytes: bytes) -> str:
    """Concatenated delta text from one SSE event (chat-completions + Responses API).

    Reads both chat-completions choices[0].delta.content and the union of
    text-bearing Responses API delta event types defined above.
    """
    try:
        event_str = event_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    out = ""
    for line in event_str.split("\n"):
        if not line.startswith("data: "):
            continue
        json_str = line[6:]
        if json_str.strip() == "[DONE]":
            continue
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            continue
        try:
            choices = data.get("choices")
            if choices and len(choices) > 0:
                out += choices[0].get("delta", {}).get("content") or ""
                continue
            if data.get("type") in _TEXT_BEARING_DELTA_EVENT_TYPES:
                out += data.get("delta") or ""
        except (AttributeError, KeyError, IndexError, TypeError):
            pass
    return out


def _strip_chars_from_event(event_bytes: bytes, n: int) -> tuple[bytes, int]:
    """Strip up to n characters from delta content fields in this event.

    Walks data: lines in order; for each, removes up to (n - already_stripped)
    leading chars from delta.content (chat-completions) or delta (Responses API).
    Returns (modified_event_bytes, chars_actually_stripped).
    """
    if n <= 0:
        return event_bytes, 0
    try:
        event_str = event_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return event_bytes, 0
    new_lines: list[str] = []
    stripped_total = 0
    for line in event_str.split("\n"):
        if not (line.startswith("data: ") and stripped_total < n):
            new_lines.append(line)
            continue
        json_str = line[6:]
        if json_str.strip() == "[DONE]":
            new_lines.append(line)
            continue
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            new_lines.append(line)
            continue
        modified = False
        try:
            choices = data.get("choices")
            if choices and len(choices) > 0:
                delta = choices[0].get("delta", {})
                content = delta.get("content") or ""
                if content:
                    take = min(n - stripped_total, len(content))
                    delta["content"] = content[take:]
                    stripped_total += take
                    choices[0]["delta"] = delta
                    data["choices"] = choices
                    modified = True
            elif data.get("type") in _TEXT_BEARING_DELTA_EVENT_TYPES:
                delta_text = data.get("delta") or ""
                if delta_text:
                    take = min(n - stripped_total, len(delta_text))
                    data["delta"] = delta_text[take:]
                    stripped_total += take
                    modified = True
        except (AttributeError, KeyError, IndexError, TypeError):
            pass
        new_lines.append("data: " + json.dumps(data) if modified else line)
    return "\n".join(new_lines).encode("utf-8"), stripped_total


# Max delta chars to accumulate while searching for the leading marker.
# Numeric marker is 7 chars ("{{{N}}}"); allow generous slack for whitespace
# or unexpected variants. Once exceeded we give up and flush as-is.
_COMPLEXITY_MARKER_LOOKAHEAD = 32

# Sliding-window size (in delta chars) used by the trailing-marker stripper.
# The longest trailing marker we expect is ~16 chars ({{{/N}}} = 8, or
# {{{complexity: Medium}}} = 23); 32 covers all observed variants with slack.
_COMPLEXITY_TRAILING_WINDOW = 32


def _strip_chars_from_event_end(event_bytes: bytes, n: int) -> tuple[bytes, int]:
    """Strip up to n characters from the END of delta content fields in this event.

    Mirror of _strip_chars_from_event but operating on the tail. Walks data:
    lines in reverse so the LAST line's delta content is shaved first (it
    holds the trailing-most text); excess strip-budget then bleeds into the
    preceding data: line's delta tail.
    """
    if n <= 0:
        return event_bytes, 0
    try:
        event_str = event_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return event_bytes, 0
    lines = event_str.split("\n")
    stripped_total = 0
    for i in range(len(lines) - 1, -1, -1):
        if stripped_total >= n:
            break
        line = lines[i]
        if not line.startswith("data: "):
            continue
        json_str = line[6:]
        if json_str.strip() == "[DONE]":
            continue
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            continue
        modified = False
        try:
            choices = data.get("choices")
            if choices and len(choices) > 0:
                delta = choices[0].get("delta", {})
                content = delta.get("content") or ""
                if content:
                    take = min(n - stripped_total, len(content))
                    delta["content"] = content[: len(content) - take]
                    stripped_total += take
                    choices[0]["delta"] = delta
                    data["choices"] = choices
                    modified = True
            elif data.get("type") in _TEXT_BEARING_DELTA_EVENT_TYPES:
                delta_text = data.get("delta") or ""
                if delta_text:
                    take = min(n - stripped_total, len(delta_text))
                    data["delta"] = delta_text[: len(delta_text) - take]
                    stripped_total += take
                    modified = True
        except (AttributeError, KeyError, IndexError, TypeError):
            pass
        if modified:
            lines[i] = "data: " + json.dumps(data)
    return "\n".join(lines).encode("utf-8"), stripped_total


async def _extract_complexity_from_stream(
    source: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Strip a leading {{{N}}} / {{{...}}} marker from an SSE stream.

    The marker is emitted as the first output of the model when the proxy
    injects the complexity classifier instruction. Tokenizers usually split
    the marker across multiple SSE events (e.g. "{{{", "2", "}}}"), so we
    must buffer events until either the full marker has arrived (then strip
    its bytes across whichever events carry them) or we can prove no marker
    is present (then flush as-is). Once decided, the rest of the stream is
    passed through unchanged.
    """
    buffered_events: list[bytes] = []
    accumulated_text = ""
    decided = False
    leftover = b""

    def _strip_and_flush(chars_to_strip: int) -> list[bytes]:
        nonlocal buffered_events, decided
        out: list[bytes] = []
        for ev in buffered_events:
            if chars_to_strip > 0:
                modified, stripped = _strip_chars_from_event(ev, chars_to_strip)
                chars_to_strip -= stripped
                out.append(modified + b"\n\n")
            else:
                out.append(ev + b"\n\n")
        buffered_events = []
        decided = True
        return out

    def _flush_as_is() -> list[bytes]:
        nonlocal buffered_events, decided
        out = [ev + b"\n\n" for ev in buffered_events]
        buffered_events = []
        decided = True
        return out

    def _decision_output(force: bool = False) -> list[bytes]:
        """Return bytes to yield once a decision is reachable, or [] if still buffering.

        Three accepted marker formats:
          A) {{{N}}} where N ∈ {1,2,3}                — canonical
          B) {{{...}}} (any other braced leading token) — defensive strip
          C) bare digit 1|2|3 followed by a blank line  — model dropped braces

        When force=True (end of stream / [DONE]), commits to a decision
        unconditionally: strip the best match available, otherwise flush as-is.
        """
        # A) strict {{{N}}}
        match = re.match(r"^\s*\{\{\{([123])\}\}\}", accumulated_text)
        if match:
            _complexity_class_context.set(int(match.group(1)))
            return _strip_and_flush(match.end())

        # B) any {{{...}}}
        match = re.match(r"^\s*\{\{\{[^}]*\}\}\}", accumulated_text)
        if match:
            inner = match.group(0).strip().strip("{").strip("}").strip()
            if inner in ("1", "2", "3"):
                _complexity_class_context.set(int(inner))
            return _strip_and_flush(match.end())

        # C) bare digit followed by blank line
        match = re.match(r"^\s*([123])[ \t]*\n[ \t]*\n", accumulated_text)
        if match:
            _complexity_class_context.set(int(match.group(1)))
            return _strip_and_flush(match.end())

        stripped_acc = accumulated_text.lstrip()

        # Decide whether to keep buffering or flush as-is.
        if not stripped_acc:
            return _flush_as_is() if force else []

        first = stripped_acc[0]
        could_be_marker = first == "{" or first in "123"

        # If digit-starting and we have 2+ chars, the next char tells us whether
        # this is a bare-digit marker candidate (followed by whitespace/newline)
        # or legitimate content like "2 minutes" / "2." / "20 things".
        if first in "123" and len(stripped_acc) >= 2:
            nxt = stripped_acc[1]
            if nxt not in (" ", "\t", "\n", "\r"):
                return _flush_as_is()
            # Else: still a bare-digit candidate, keep buffering for the blank line.

        if not could_be_marker:
            return _flush_as_is()

        if force or len(accumulated_text) >= _COMPLEXITY_MARKER_LOOKAHEAD:
            return _flush_as_is()

        return []

    async for chunk in source:
        if decided:
            yield chunk
            continue

        combined = leftover + chunk
        parts = combined.split(b"\n\n")
        leftover = parts[-1]
        events = parts[:-1]

        for idx, ev in enumerate(events):
            if decided:
                yield ev + b"\n\n"
                continue
            buffered_events.append(ev)
            accumulated_text += _event_delta_text(ev)
            force = b"data: [DONE]" in ev
            for out_bytes in _decision_output(force=force):
                yield out_bytes

        if decided and leftover:
            yield leftover
            leftover = b""

    # Stream ended; force a final decision on anything still buffered.
    if buffered_events:
        for out_bytes in _decision_output(force=True):
            yield out_bytes
    if leftover:
        yield leftover


async def _scrub_full_text_events(
    source: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Strip leading / trailing {{{...}}} markers from full-text 'done' events.

    The streaming filters (_extract_complexity_from_stream and
    _strip_trailing_complexity_marker) scrub the .delta event stream only.
    Codex also emits accumulated-text events at the end of each output:
    response.output_text.done, response.content_part.done,
    response.output_item.done — each carries the FULL response text and
    Hermes / codex-cli often read from those for the final display. If a
    marker was in the deltas (because the model voluntarily echoed it from
    its conversation history), it ends up in these too, bypassing both
    delta filters. This pass rewrites those events in-place so the
    consumer never sees the marker.

    Single-pass, no buffering: every event is parsed, the known .done
    event types have their text field scrubbed, then the (possibly
    modified) event is reserialized and yielded.
    """
    leftover = b""
    async for chunk in source:
        combined = leftover + chunk
        parts = combined.split(b"\n\n")
        leftover = parts[-1]
        events = parts[:-1]
        for ev in events:
            try:
                ev_str = ev.decode("utf-8")
            except UnicodeDecodeError:
                yield ev + b"\n\n"
                continue
            lines = ev_str.split("\n")
            modified_any = False
            new_lines: list[str] = []
            for line in lines:
                if not line.startswith("data: "):
                    new_lines.append(line)
                    continue
                json_str = line[6:]
                if json_str.strip() == "[DONE]":
                    new_lines.append(line)
                    continue
                try:
                    data = json.loads(json_str)
                except (json.JSONDecodeError, ValueError):
                    new_lines.append(line)
                    continue
                if _scrub_full_text_event(data):
                    new_lines.append("data: " + json.dumps(data))
                    modified_any = True
                else:
                    new_lines.append(line)
            yield ("\n".join(new_lines)).encode("utf-8") + b"\n\n"
    if leftover:
        yield leftover


async def _strip_trailing_complexity_marker(
    source: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Strip a trailing {{{...}}} marker if the model emits one as a closing tag.

    Maintains a sliding window of the most recent SSE events whose accumulated
    delta text is at least _COMPLEXITY_TRAILING_WINDOW characters. Older
    events are flushed downstream the moment they fall out of the window — so
    streaming latency only increases by ~32 chars worth of buffering. On
    stream end (or [DONE]), checks the buffered tail for a trailing
    {{{...}}} pattern and shaves its bytes off the appropriate event(s)
    before flushing the rest.
    """
    trail: list[tuple[bytes, str]] = []
    trail_chars = 0
    leftover = b""

    def _flush(final: bool) -> list[bytes]:
        nonlocal trail, trail_chars
        if final and trail:
            tail_text = "".join(t for _, t in trail)
            match = re.search(r"\{\{\{[^}]*\}\}\}\s*$", tail_text)
            if match:
                chars_to_strip = len(match.group(0))
                for i in range(len(trail) - 1, -1, -1):
                    if chars_to_strip <= 0:
                        break
                    ev_b, ev_text = trail[i]
                    new_ev, stripped = _strip_chars_from_event_end(ev_b, chars_to_strip)
                    if stripped > 0:
                        ev_text = ev_text[: len(ev_text) - stripped] if stripped <= len(ev_text) else ""
                        trail[i] = (new_ev, ev_text)
                        chars_to_strip -= stripped
        out = [ev + b"\n\n" for ev, _ in trail]
        trail = []
        trail_chars = 0
        return out

    async for chunk in source:
        combined = leftover + chunk
        parts = combined.split(b"\n\n")
        leftover = parts[-1]
        events = parts[:-1]

        for ev in events:
            is_done = b"data: [DONE]" in ev
            delta_text = _event_delta_text(ev)
            trail.append((ev, delta_text))
            trail_chars += len(delta_text)

            if is_done:
                for out_bytes in _flush(final=True):
                    yield out_bytes
                continue

            while (
                len(trail) > 1
                and trail_chars - len(trail[0][1]) >= _COMPLEXITY_TRAILING_WINDOW
            ):
                old_ev, old_text = trail.pop(0)
                trail_chars -= len(old_text)
                yield old_ev + b"\n\n"

    for out_bytes in _flush(final=True):
        yield out_bytes
    if leftover:
        yield leftover


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
    recommender_classifier_cell: str | None = None,
    recommender_raw_output: str | None = None,
    recommender_source: str | None = None,
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
        recommender_classifier_cell=recommender_classifier_cell,
        recommender_raw_output=recommender_raw_output,
        recommender_source=recommender_source,
    )


def _clean_sse_blob(blob: bytes | None) -> bytes | None:
    """Remove complexity markers from SSE blob before storage.

    Parses SSE format, extracts and cleans response content from delta/text fields,
    and reconstructs the blob for database storage. Ensures markers never persist
    in the database even if they leak through the stream cleaning phase.
    """
    if blob is None:
        return None

    try:
        text = blob.decode('utf-8')
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            # Check if this is a data line with JSON content
            if line.startswith('data: '):
                try:
                    json_str = line[6:]  # Strip 'data: '
                    data = json.loads(json_str)

                    # Handle text-bearing Responses API delta events
                    if data.get('type') in _TEXT_BEARING_DELTA_EVENT_TYPES:
                        delta = data.get('delta', '')
                        if isinstance(delta, str) and delta:
                            _, cleaned_delta = _extract_complexity_class(delta)
                            cleaned_delta = _strip_trailing_complexity_marker_text(cleaned_delta)
                            data['delta'] = cleaned_delta

                    # Handle full-text 'done' events (Hermes / codex-cli often
                    # read these for final UI render — must be scrubbed too)
                    elif data.get('type') in _FULL_TEXT_EVENT_PATHS:
                        _scrub_full_text_event(data)

                    # Handle chat completions format (choices[0].delta.content)
                    elif 'choices' in data and len(data.get('choices', [])) > 0:
                        delta = data['choices'][0].get('delta', {})
                        content = delta.get('content', '')
                        if isinstance(content, str) and content:
                            _, cleaned_content = _extract_complexity_class(content)
                            cleaned_content = _strip_trailing_complexity_marker_text(cleaned_content)
                            delta['content'] = cleaned_content
                            data['choices'][0]['delta'] = delta

                    cleaned_lines.append('data: ' + json.dumps(data))
                except (json.JSONDecodeError, KeyError, TypeError):
                    # Not JSON or unexpected format, pass through unchanged
                    cleaned_lines.append(line)
            else:
                # Non-data lines pass through unchanged
                cleaned_lines.append(line)

        cleaned_text = '\n'.join(cleaned_lines)
        return cleaned_text.encode('utf-8')
    except (UnicodeDecodeError, AttributeError):
        # If we can't decode, return original blob
        return blob


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
    recommender_classifier_cell: str | None = None,
    recommender_raw_output: str | None = None,
    recommender_source: str | None = None,
) -> None:
    if usage_log is None:
        return
    req_payload = json.dumps(body).encode()
    if stream and handle.stream_summary is not None:
        resp_payload: bytes | None = handle.stream_summary.raw_blob
        # Clean markers from raw blob before storing in database
        resp_payload = _clean_sse_blob(resp_payload)
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
        client_request=body,
        recommender_classifier_cell=recommender_classifier_cell,
        recommender_raw_output=recommender_raw_output,
        recommender_source=recommender_source,
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


def _active_api_key_count(auth_service: Any) -> int:
    """Best-effort count of active API keys for 401 diagnostics. Returns -1
    if the count can't be obtained — the error path must never raise."""
    try:
        return auth_service.db.count_active_api_keys()
    except Exception:
        return -1


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


async def _diagnose_backend(backend: Backend, *, force: bool = False) -> dict[str, Any]:
    """Send one minimal streaming request and check upstream contract holds.

    When `force=True` the cooldown-skip guard is bypassed: the probe goes
    out even if the persisted snapshot claims the backend is still in
    cooldown. The periodic cooldown prober uses this to break out of a
    stale-snapshot lockout (see CodexAuthVaultBackend.clear_cooldown).
    """
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
    if (
        not force
        and usage.cooldown_until_ts is not None
        and usage.cooldown_until_ts > now
    ):
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
        # Build detailed error result with quota information if available
        result = {
            "id": backend.id,
            "ok": False,
            "skipped": False,
            "stage": "upstream",
            "classification": exc.classification,
            "status_code": exc.status_code,
            "reason": exc.message or exc.classification,
            "model": model,
        }
        # Include quota snapshots for rate-limited errors
        if exc.classification == "rate_limited" and handle.quota_after is not None:
            quota = handle.quota_after
            result["quota_after"] = quota
        return result
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


class _PeriodicCooldownProber:
    """Background task that re-probes cooldown'd backends so the proxy can
    self-heal from stale-snapshot lockouts.

    The dispatcher excludes any backend whose persisted cooldown_until_ts is
    in the future, and every other probe path (startup smoke test, periodic
    smoke test, /diagnose/upstream) also skips cooldown'd backends by design
    — they're meant to respect a real cooldown rather than burn quota on a
    backend that just said no. That defensive behavior turns into a
    chicken-and-egg lockout whenever the persisted snapshot stops matching
    reality (upstream's reported weekly_reset_at was wrong, account was
    topped up out of band, original 429 was transient, etc.): the proxy
    can't learn that headroom returned because nothing inside it is allowed
    to probe.

    This task is the deliberate counter to that: every `interval_s` it
    sends a minimal forced probe to each cooldown'd backend and clears the
    cooldown if the probe comes back clean. Cost is tiny (a handful of
    tokens per backend per cycle); the failure mode it prevents is days-of-
    blocked-traffic stuck on stale state. interval_s=0 disables.
    """

    def __init__(self, *, backends: Sequence[Backend], interval_s: int) -> None:
        self._backends = list(backends)
        self._interval_s = interval_s
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @property
    def enabled(self) -> bool:
        return self._interval_s > 0 and bool(self._backends)

    def start(self) -> None:
        if not self.enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="periodic-cooldown-prober")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval_s)
            except TimeoutError:
                pass
            else:
                return
            for backend in self._backends:
                try:
                    usage = await backend.usage_snapshot()
                except Exception:
                    logger.exception("cooldown prober: usage_snapshot failed for %r", backend.id)
                    continue
                now = time.time()
                if usage.cooldown_until_ts is None or usage.cooldown_until_ts <= now:
                    continue
                try:
                    result = await _diagnose_backend(backend, force=True)
                except Exception:
                    logger.exception("cooldown prober: probe raised for %r", backend.id)
                    continue
                if result.get("ok") and not result.get("skipped"):
                    clear = getattr(backend, "clear_cooldown", None)
                    if clear is not None:
                        try:
                            clear()
                            logger.warning(
                                "cooldown prober: %r probe succeeded; cooldown cleared "
                                "(was until %s)",
                                backend.id, usage.cooldown_until_ts,
                            )
                        except Exception:
                            logger.exception(
                                "cooldown prober: clear_cooldown raised for %r", backend.id
                            )


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
        cooldown_until = result.get("cooldown_until_ts")
        reason = result.get("reason", "in cooldown")
        if cooldown_until is not None:
            from datetime import datetime, timezone
            try:
                reset_time = datetime.fromtimestamp(
                    cooldown_until, tz=timezone.utc
                ).isoformat()
                reason = f"{reason} (reset at {reset_time})"
            except (OverflowError, ValueError, OSError):
                # Bogus/very-large cooldown timestamps (e.g. test sentinels or
                # stuck-clock state) shouldn't crash the smoke-test log line.
                reason = f"{reason} (reset at ts={cooldown_until})"
        logger.info("  [%s] SKIPPED — %s", backend_id, reason)
        return
    if result.get("ok"):
        upstream = result.get("upstream_status")
        logger.info("  [%s] OK — upstream %s", backend_id, upstream)
        return

    stage = result.get("stage", "?")
    classification = result.get("classification")
    status_code = result.get("status_code")
    reason = result.get("reason") or result.get("failed_checks") or "(no reason)"

    # For rate_limited errors, include quota exhaustion details
    quota_msg = ""
    if classification == "rate_limited":
        quota = result.get("quota_after")
        if quota is not None:
            from datetime import datetime, timezone
            exhaustion_info = []

            # 5-hour quota status
            if quota.five_hourly_used_percent is not None:
                pct = quota.five_hourly_used_percent
                exhaustion_info.append(f"5h-window {pct}%")
                if pct >= 99:
                    if quota.five_hourly_reset_at is not None:
                        reset = datetime.fromtimestamp(quota.five_hourly_reset_at, tz=timezone.utc)
                        exhaustion_info.append(f"(resets {reset.isoformat()})")
                    else:
                        exhaustion_info.append("(resets ~5 hours)")

            # Weekly quota status
            if quota.weekly_used_percent is not None:
                pct = quota.weekly_used_percent
                exhaustion_info.append(f"weekly {pct}%")
                if pct >= 99:
                    if quota.weekly_reset_at is not None:
                        reset = datetime.fromtimestamp(quota.weekly_reset_at, tz=timezone.utc)
                        exhaustion_info.append(f"(resets {reset.isoformat()})")
                    else:
                        exhaustion_info.append("(resets ~7 days)")

            if exhaustion_info:
                quota_msg = f" [{' | '.join(exhaustion_info)}]"
        reason = f"{reason}{quota_msg}"

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


async def _collect_backend_status(
    backends_list: Sequence[Backend],
) -> list[dict[str, Any]]:
    """Snapshot each backend's diagnosis-relevant state for an error response.

    Built so a 503 detail can be self-explanatory: every downstream tool
    (Hermes, codex-cli, Cursor) prints the proxy's error verbatim, so the
    proxy is the only place that can pack this context in once.
    """
    out: list[dict[str, Any]] = []
    now = time.time()
    for b in backends_list:
        try:
            usage = await b.usage_snapshot()
        except Exception:
            usage = None
        try:
            quota = await b.quota_snapshot()
        except Exception:
            quota = None
        cd = getattr(usage, "cooldown_until_ts", None)
        cd_in_s = (cd - now) if cd else None
        out.append({
            "id": b.id,
            "kind": getattr(b, "kind", "unknown"),
            "advertised_models": sorted(b.advertised_models),
            "cooldown_until_ts": cd,
            "cooldown_in_seconds": int(cd_in_s) if cd_in_s and cd_in_s > 0 else None,
            "weekly_exhausted": bool(getattr(usage, "weekly_exhausted", False)),
            "five_hourly_used_percent": getattr(quota, "five_hourly_used_percent", None),
            "five_hourly_reset_after_seconds": getattr(quota, "five_hourly_reset_after_seconds", None),
            "weekly_used_percent": getattr(quota, "weekly_used_percent", None),
            "weekly_reset_after_seconds": getattr(quota, "weekly_reset_after_seconds", None),
        })
    return out


def _no_viable(
    *,
    model: str,
    last_error: BackendError | None,
    excluded_backends: dict[str, BackendError] | None = None,
    fallback_executor: FallbackExecutor | None = None,
    recovery_ts: float | None = None,
    backend_status: list[dict[str, Any]] | None = None,
) -> HTTPException:
    """Log all failed backends and return appropriate error with Retry-After header.

    When `backend_status` is provided, the response body's `detail` becomes a
    structured dict including each backend's cooldown + quota state so
    downstream tools printing the error verbatim have enough information
    to diagnose without separately curling /status.
    """
    if excluded_backends:
        # Build error classification map for logging
        error_classifications = {
            backend_id: error.classification
            for backend_id, error in excluded_backends.items()
        }

        if fallback_executor:
            fallback_executor.log_final_exhaustion(model, error_classifications)
        else:
            # Fallback not attempted, log with recovery info if available
            failures = []
            for backend_id, error in excluded_backends.items():
                failures.append(f"{backend_id}: {error.classification}")
            ts = _utc_timestamp()
            msg = f"[{ts}] all backends exhausted for model {model!r}. Failures: {'; '.join(failures)}"
            if recovery_ts:
                recovery_dt = datetime.fromtimestamp(recovery_ts, tz=timezone.utc)
                recovery_s = max(1, int(recovery_ts - time.time()))
                msg += f" | Earliest recovery: {recovery_dt.isoformat()} (in {recovery_s}s)"
            logger.warning(msg)

    def _build_detail(short: str) -> Any:
        if backend_status is None:
            return short
        summary = short
        if recovery_ts:
            recovery_s = max(1, int(recovery_ts - time.time()))
            summary += f" | earliest recovery in {recovery_s}s ({recovery_s/60:.1f}min)"
        return {
            "error": short,
            "summary": summary,
            "model": model,
            "backends": backend_status,
            "recovery_in_seconds": (
                max(1, int(recovery_ts - time.time())) if recovery_ts else None
            ),
        }

    if last_error is None:
        return HTTPException(
            status_code=503,
            detail=_build_detail(f"no viable backend for model {model!r}"),
        )
    status = _EXHAUSTED_STATUS.get(last_error.classification, 502)

    # Build response headers with Retry-After if available
    headers: dict[str, str] = {}
    if recovery_ts and status == 429:
        retry_after_s = max(1, int(recovery_ts - time.time()))
        headers["Retry-After"] = str(retry_after_s)
        recovery_dt = datetime.fromtimestamp(recovery_ts, tz=timezone.utc)
        headers["X-Retry-After-UTC"] = recovery_dt.isoformat()

    return HTTPException(
        status_code=status,
        detail=_build_detail(last_error.message or last_error.classification),
        headers=headers or None,
    )
