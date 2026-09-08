"""Generic Codex gateway backend — talks to a Codex-compatible endpoint
using a plain API key.

This backend forwards requests to an OpenAI-compatible Codex Responses
endpoint using a Bearer token the operator provides. It supports the
/responses API (buffered and streaming), /chat/completions
(translated to/from responses), and model discovery via GET /models.

The operator points base_url at any Codex-compatible endpoint and
provides an API key. The key is sent as Authorization: Bearer <key>
on every request. No OAuth, no token refresh, no proxy concepts — just
an endpoint and a key.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from callosum.cell_grid import ModelMetadata

import httpx

from callosum.backend import BackendKind, CallHandle, HealthStatus, UsageSnapshot
from callosum.backends._http import DEFAULT_COOLDOWN_S, error_from_response
from callosum.backends.codex_auth_vault import (
    DEFAULT_MODELS_REFRESH_S,
    RESPONSES_BETA_HEADER_VALUE,
    _chat_to_responses_request,
    _extract_model_catalog,
    _int_or,
    _require_model,
    _resolve_codex_client_version,
    _responses_to_chat_response,
    _str_or,
)
from callosum.codex_quota import parse_codex_headers
from callosum.errors import BackendError
from callosum.sse_tee import (
    ResponsesStreamCollector,
    assemble_completed_with_text,
    namespaced_tool_names_from_request,
    strip_namespace_stream,
)
from callosum.state import StateStore

DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"


class CodexGatewayBackend:
    """Backend that talks to a Codex-compatible endpoint with an API key.

    Takes a base_url and an api_key. The key is sent as a Bearer
    token on every request. Model discovery, quota parsing, and the
    chat↔responses translation are the same as CodexAuthVaultBackend;
    the only difference is authentication (API key vs OAuth vault).
    """

    kind: BackendKind = "codex_gateway"

    def __init__(
        self,
        *,
        id: str,
        api_key: str,
        advertised_models: frozenset[str] = frozenset(),
        base_url: str = DEFAULT_BASE_URL,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = 60.0,
        state_store: StateStore | None = None,
        models_refresh_s: float = DEFAULT_MODELS_REFRESH_S,
    ) -> None:
        self.id = id
        self._static_advertised_models: frozenset[str] = advertised_models
        self._dynamic_advertised_models: frozenset[str] | None = None
        self._model_context_windows: dict[str, int] = {}
        self._model_metadata: dict[str, ModelMetadata] = {}
        self._models_fetched_at: float = 0.0
        self._models_refresh_s = models_refresh_s
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(
                transport=transport,
                timeout=timeout_s,
            )
            self._owns_client = True
        self._state_store = state_store
        loaded = state_store.load_usage(id) if state_store is not None else None
        self._usage = loaded or UsageSnapshot(
            remaining_fraction=None,
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=time.time(),
        )
        self._last_quota: Any = None
        self._consecutive_transport_failures: int = 0
        self._transport_cooldown_until_ts: float = 0.0
        self._transport_failure_threshold: int = 3
        self._transport_cooldown_seconds: float = 30.0
        self._warm_start_catalog()

    @property
    def advertised_models(self) -> frozenset[str]:
        if self._dynamic_advertised_models is not None:
            return self._dynamic_advertised_models
        return self._static_advertised_models

    @property
    def model_context_windows(self) -> dict[str, int]:
        return self._model_context_windows

    def model_metadata(self):  # type: ignore[no-untyped-def]
        return self._model_metadata

    def _warm_start_catalog(self) -> None:
        if self._state_store is None:
            return
        persisted = self._state_store.load_catalog(self.id)
        if not isinstance(persisted, dict):
            return
        raw_models = persisted.get("advertised_models")
        if not isinstance(raw_models, list):
            return
        slugs = frozenset(m for m in raw_models if isinstance(m, str) and m)
        if not slugs:
            return
        raw_ctx = persisted.get("context_windows")
        ctx_map: dict[str, int] = {}
        if isinstance(raw_ctx, dict):
            for slug, ctx in raw_ctx.items():
                if isinstance(slug, str) and isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0:
                    ctx_map[slug] = ctx
        raw_meta = persisted.get("model_metadata")
        meta_map: dict[str, ModelMetadata] = {}
        if isinstance(raw_meta, dict):
            from callosum.cell_grid import model_metadata_from_dict

            for slug, md_raw in raw_meta.items():
                if not isinstance(slug, str):
                    continue
                md = model_metadata_from_dict(md_raw)
                if md is not None:
                    meta_map[slug] = md
        self._dynamic_advertised_models = slugs
        self._model_context_windows = ctx_map
        self._model_metadata = meta_map
        self._models_fetched_at = 0.0

    def _persist_catalog(self, *, fetched_at: float) -> None:
        if self._state_store is None or self._dynamic_advertised_models is None:
            return
        self._state_store.save_catalog(
            self.id,
            {
                "advertised_models": sorted(self._dynamic_advertised_models),
                "context_windows": self._model_context_windows,
                "model_metadata": {
                    slug: {
                        "slug": m.slug,
                        "supported_in_api": m.supported_in_api,
                        "visibility": m.visibility,
                        "priority": m.priority,
                        "supported_reasoning_levels": list(m.supported_reasoning_levels),
                        "context_window": m.context_window,
                    }
                    for slug, m in self._model_metadata.items()
                },
                "fetched_at": fetched_at,
            },
        )

    def cell_capabilities(self, model: str):  # type: ignore[no-untyped-def]
        from callosum.routing.protocols import CellCapabilities

        ctx = self._model_context_windows.get(model, 128_000)
        return CellCapabilities(
            context_window=ctx,
            modalities=frozenset({"text", "image"}),
            supports_tools=True,
            cost_rank=1,
        )

    async def refresh_advertised_models(self, *, now: float | None = None) -> None:
        ts = now if now is not None else time.time()
        if self._dynamic_advertised_models is not None and ts - self._models_fetched_at < self._models_refresh_s:
            return
        try:
            response = await self._client.get(
                f"{self._base_url}/models",
                params={"client_version": _resolve_codex_client_version()},
                headers=self._build_headers(),
            )
        except httpx.HTTPError:
            return
        if response.status_code != 200:
            detail = ""
            try:
                detail = response.text[:200]
            except Exception:
                detail = "(no body)"
            logging.getLogger("callosum.backend").warning(
                "models discovery failed for backend %r: HTTP %d %s",
                self.id,
                response.status_code,
                detail,
            )
            return
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError):
            return
        models, context_windows, metadata = _extract_model_catalog(payload)
        if models:
            self._dynamic_advertised_models = models
            self._model_context_windows = context_windows
            self._model_metadata = metadata
            self._models_fetched_at = ts
            self._persist_catalog(fetched_at=ts)

    async def health(self) -> HealthStatus:
        return HealthStatus(available=True, reason="ok")

    def clear_cooldown(self) -> UsageSnapshot:
        self._usage = UsageSnapshot(
            remaining_fraction=self._usage.remaining_fraction,
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=time.time(),
        )
        if self._state_store is not None:
            self._state_store.save_usage(self.id, self._usage)
        return self._usage

    def _on_transport_failure(self) -> None:
        self._consecutive_transport_failures += 1
        if self._consecutive_transport_failures >= self._transport_failure_threshold:
            self._transport_cooldown_until_ts = time.time() + self._transport_cooldown_seconds

    def _on_transport_success(self) -> None:
        self._consecutive_transport_failures = 0
        self._transport_cooldown_until_ts = 0.0

    async def usage_snapshot(self) -> UsageSnapshot:
        if self._last_quota is not None and self._last_quota.weekly_used_percent is not None:
            weekly_exhausted = self._last_quota.weekly_used_percent >= 99
        else:
            weekly_exhausted = self._usage.weekly_exhausted
        now = time.time()
        snap_cooldown = self._usage.cooldown_until_ts
        effective_cooldown = snap_cooldown
        if self._transport_cooldown_until_ts > now and (
            effective_cooldown is None or self._transport_cooldown_until_ts > effective_cooldown
        ):
            effective_cooldown = self._transport_cooldown_until_ts
        if weekly_exhausted == self._usage.weekly_exhausted and effective_cooldown == self._usage.cooldown_until_ts:
            return self._usage
        return UsageSnapshot(
            remaining_fraction=self._usage.remaining_fraction,
            cooldown_until_ts=effective_cooldown,
            weekly_exhausted=weekly_exhausted,
            probed_at_ts=self._usage.probed_at_ts,
        )

    async def quota_snapshot(self) -> Any:  # CodexQuotaSnapshot | None
        return self._last_quota

    async def chat_completions(self, body: dict[str, Any], handle: CallHandle | None = None) -> dict[str, Any]:
        requested_model = _require_model(body)
        responses_payload = _chat_to_responses_request(body, stream=False)
        upstream = await self.responses(responses_payload, handle)
        return _responses_to_chat_response(upstream, model=requested_model)

    async def chat_completions_stream(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> AsyncIterator[bytes]:
        completion = await self.chat_completions({**body, "stream": False}, handle)
        completion_id = _str_or(completion.get("id"), "chatcmpl-codex")
        model = _str_or(completion.get("model"), "")
        created = _int_or(completion.get("created"), int(time.time()))
        choices = completion.get("choices")
        if not isinstance(choices, list) or not choices:
            content = ""
            finish_reason = "stop"
        else:
            first = choices[0] if isinstance(choices[0], dict) else {}
            message = first.get("message") if isinstance(first.get("message"), dict) else {}
            raw_content = message.get("content") if isinstance(message, dict) else None
            content = raw_content if isinstance(raw_content, str) else ""
            finish_reason_raw = first.get("finish_reason")
            finish_reason = finish_reason_raw if isinstance(finish_reason_raw, str) else "stop"

        def frame(delta: dict[str, Any], finish: str | None) -> bytes:
            payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish,
                    }
                ],
            }
            return f"data: {json.dumps(payload)}\n\n".encode()

        yield frame({"role": "assistant"}, None)
        if content:
            yield frame({"content": content}, None)
        yield frame({}, finish_reason)
        yield b"data: [DONE]\n\n"

    async def responses(self, body: dict[str, Any], handle: CallHandle | None = None) -> dict[str, Any]:
        if handle is not None:
            handle.quota_before = self._last_quota
        streaming_body = {**body, "stream": True}
        from callosum.backends._http import _strip_unsupported_input_fields

        _strip_unsupported_input_fields(streaming_body)
        headers = self._build_headers(accept_event_stream=True)
        try:
            stream_ctx = self._client.stream(
                "POST",
                f"{self._base_url}/responses",
                json=streaming_body,
                headers=headers,
            )
            async with stream_ctx as response:
                self._apply_response_to_handle(response.headers, response.status_code, handle)
                if response.status_code >= 400:
                    await response.aread()
                    err = error_from_response(response)
                    self._apply_error_to_usage(err)
                    raise err
                namespaced_names = namespaced_tool_names_from_request(body)
                collector = ResponsesStreamCollector(strip_namespace_stream(response.aiter_bytes(), namespaced_names))
                async for _chunk in collector.iter_through():
                    pass
                if handle is not None:
                    handle.stream_summary = collector.summary
                completed = assemble_completed_with_text(collector.summary.raw_blob)
                if completed is None:
                    raise BackendError(
                        classification="transient",
                        message="upstream stream ended without response.completed event",
                    )
                self._on_transport_success()
                return completed
        except httpx.HTTPError as exc:
            self._on_transport_failure()
            raise BackendError(classification="transient", message=str(exc)) from exc

    async def responses_stream(self, body: dict[str, Any], handle: CallHandle | None = None) -> AsyncIterator[bytes]:
        if handle is not None:
            handle.quota_before = self._last_quota
        from callosum.backends._http import _strip_unsupported_input_fields

        _strip_unsupported_input_fields(body)
        headers = self._build_headers(accept_event_stream=True)
        try:
            stream_ctx = self._client.stream(
                "POST",
                f"{self._base_url}/responses",
                json=body,
                headers=headers,
            )
            async with stream_ctx as response:
                self._apply_response_to_handle(response.headers, response.status_code, handle)
                if response.status_code >= 400:
                    await response.aread()
                    err = error_from_response(response)
                    self._apply_error_to_usage(err)
                    raise err
                namespaced_names = namespaced_tool_names_from_request(body)
                collector = ResponsesStreamCollector(strip_namespace_stream(response.aiter_bytes(), namespaced_names))
                async for chunk in collector.iter_through():
                    yield chunk
                if handle is not None:
                    handle.stream_summary = collector.summary
                self._on_transport_success()
                return
        except httpx.HTTPError as exc:
            self._on_transport_failure()
            raise BackendError(classification="transient", message=str(exc)) from exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _build_headers(self, *, accept_event_stream: bool = False) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if accept_event_stream else "application/json",
            "OpenAI-Beta": RESPONSES_BETA_HEADER_VALUE,
            "originator": "codex_cli_rs",
            "version": _resolve_codex_client_version(),
        }

    def _apply_response_to_handle(
        self,
        headers: Any,
        status_code: int,
        handle: CallHandle | None,
    ) -> None:
        snapshot = parse_codex_headers(dict(headers))
        if snapshot is not None:
            self._last_quota = snapshot
        if handle is not None:
            handle.upstream_status = status_code
            handle.upstream_headers = dict(headers)
            handle.quota_after = snapshot

    def _apply_error_to_usage(self, err: BackendError) -> None:
        if err.classification != "rate_limited":
            return
        now = time.time()
        quota = self._last_quota
        if err.retry_after_s:
            cooldown_until = now + err.retry_after_s
        elif quota is not None and quota.weekly_used_percent is not None and quota.weekly_used_percent >= 99:
            cooldown_until = float(quota.weekly_reset_at) if quota.weekly_reset_at else now + 7 * 86400
        elif quota is not None and quota.five_hourly_used_percent is not None and quota.five_hourly_used_percent >= 95:
            cooldown_until = float(quota.five_hourly_reset_at) if quota.five_hourly_reset_at else now + 5 * 3600
        else:
            cooldown_until = now + DEFAULT_COOLDOWN_S
        if quota is not None and quota.weekly_used_percent is not None:
            weekly_exhausted = quota.weekly_used_percent >= 99
        else:
            weekly_exhausted = self._usage.weekly_exhausted
        self._usage = UsageSnapshot(
            remaining_fraction=self._usage.remaining_fraction,
            cooldown_until_ts=cooldown_until,
            weekly_exhausted=weekly_exhausted,
            probed_at_ts=now,
        )
        if self._state_store is not None:
            self._state_store.save_usage(self.id, self._usage)
