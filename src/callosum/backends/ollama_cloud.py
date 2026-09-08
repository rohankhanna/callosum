"""Ollama Cloud backend served directly by ollama.com.

The backend uses an ollama.com API key, sends requests directly to the
upstream service, and keeps per-request usage accounting from the
OpenAI-compatible chat-completions response.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager as AsyncContextManager
from typing import Any, cast

import httpx

from callosum.backend import BackendKind, CallHandle, HealthStatus, UsageSnapshot
from callosum.backends._http import error_from_response, stall_guarded
from callosum.backends._responses_chat import (
    _chat_to_responses_response,
    _responses_to_chat_request,
    _strip_codex_only_fields,
    chat_to_responses_stream,
)
from callosum.cell_grid import ModelMetadata
from callosum.config import LOCAL_STREAM_FIRST_BYTE_TIMEOUT_S, LOCAL_STREAM_IDLE_TIMEOUT_S
from callosum.errors import BackendError
from callosum.routing.protocols import CellCapabilities
from callosum.sse_tee import ResponsesStreamCollector

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "https://ollama.com"
DEFAULT_MODEL_SUFFIX = ""
DEFAULT_CATALOG_REFRESH_S = 60.0
DEFAULT_HEALTH_TIMEOUT_S = 5.0
DEFAULT_CALL_TIMEOUT_S = 300.0
CLOUD_PRIORITY_OFFSET = 1_000


class OllamaCloudBackend:
    """Cloud models served by ollama.com with direct API-key authentication."""

    kind: BackendKind = "ollama_cloud"

    def __init__(
        self,
        *,
        id: str,
        api_key: str,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model_suffix: str = DEFAULT_MODEL_SUFFIX,
        catalog_refresh_s: float = DEFAULT_CATALOG_REFRESH_S,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
    ) -> None:
        self.id = id
        self._api_key = api_key
        self._ollama_url = ollama_url.rstrip("/")
        self._model_suffix = model_suffix
        self._catalog_refresh_s = catalog_refresh_s
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(transport=transport, timeout=timeout_s)
            self._owns_client = True
        self._catalog: tuple[str, ...] = ()
        self._catalog_fetched_at: float = 0.0
        self._healthy: bool = False
        self._last_health_reason: str = "unknown"
        self._capabilities_cache: dict[str, CellCapabilities] = {}

    @property
    def advertised_models(self) -> frozenset[str]:
        return frozenset(self._catalog)

    @property
    def model_metadata(self) -> dict[str, ModelMetadata]:
        """Build model metadata for every discovered cloud model."""
        return {
            slug: ModelMetadata(
                slug=slug,
                supported_in_api=True,
                visibility="list",
                priority=CLOUD_PRIORITY_OFFSET + idx,
                supported_reasoning_levels=("default",),
            )
            for idx, slug in enumerate(self._catalog)
        }

    async def health(self) -> HealthStatus:
        """Check ollama.com directly with the configured API key."""
        if not self._api_key:
            self._healthy = False
            self._last_health_reason = "no-key"
            return HealthStatus(available=False, reason="no-key")
        try:
            status, _headers, _body = await self._direct_request(
                method="GET",
                url=f"{self._ollama_url}/v1/models",
                app_headers={},
                body=b"",
                timeout=DEFAULT_HEALTH_TIMEOUT_S,
            )
        except httpx.HTTPError:
            self._mark_unhealthy()
            return HealthStatus(available=False, reason="network")
        if status == 200:
            self._healthy = True
            self._last_health_reason = "ok"
            return HealthStatus(available=True, reason="ok")
        if status == 401:
            self._healthy = False
            self._last_health_reason = "auth_invalid"
            return HealthStatus(available=False, reason="auth_invalid")
        self._healthy = False
        self._last_health_reason = "unknown"
        return HealthStatus(available=False, reason="unknown")

    async def usage_snapshot(self) -> UsageSnapshot:
        """Report advisory usage for a remote cell without a quota header."""
        now = time.time()
        cooldown_until: float | None = None
        if not self._healthy and self._catalog_fetched_at > 0:
            cooldown_until = now + 30.0
        return UsageSnapshot(
            remaining_fraction=1.0,
            cooldown_until_ts=cooldown_until,
            weekly_exhausted=False,
            probed_at_ts=now,
        )

    async def quota_snapshot(self) -> None:
        return None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _mark_unhealthy(self) -> None:
        self._healthy = False
        self._last_health_reason = "network"

    async def _direct_request(
        self,
        *,
        method: str,
        url: str,
        app_headers: dict[str, str],
        body: bytes,
        timeout: float = DEFAULT_CALL_TIMEOUT_S,  # noqa: ASYNC109 - forwards to httpx transport
    ) -> tuple[int, dict[str, str], bytes]:
        headers = {**app_headers, "Authorization": f"Bearer {self._api_key}"}
        response = await self._client.request(method, url, headers=headers, content=body, timeout=timeout)
        return response.status_code, dict(response.headers), response.content

    def _app_request_headers(self, *, stream: bool) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }

    @contextlib.asynccontextmanager
    async def _direct_stream_cm(self, out_body: dict[str, Any]) -> AsyncIterator[httpx.Response]:
        headers = {**self._app_request_headers(stream=True), "Authorization": f"Bearer {self._api_key}"}
        async with self._client.stream(
            "POST",
            f"{self._ollama_url}/v1/chat/completions",
            headers=headers,
            content=json.dumps(out_body).encode("utf-8"),
        ) as response:
            yield response

    async def chat_completions(self, body: dict[str, Any], handle: CallHandle | None = None) -> dict[str, Any]:
        """Send a non-streaming chat-completions request directly to ollama.com."""
        await self._refresh_catalog_if_stale()
        out_body = _strip_codex_only_fields({**body, "stream": False})
        try:
            upstream_status, upstream_headers, upstream_body = await self._direct_request(
                method="POST",
                url=f"{self._ollama_url}/v1/chat/completions",
                app_headers=self._app_request_headers(stream=False),
                body=json.dumps(out_body).encode("utf-8"),
            )
        except httpx.HTTPError as exc:
            self._mark_unhealthy()
            raise BackendError(classification="transient", message=str(exc)) from exc
        if handle is not None:
            handle.upstream_status = upstream_status
            handle.upstream_headers = upstream_headers
        if upstream_status >= 400:
            synthetic = httpx.Response(upstream_status, headers=upstream_headers, content=upstream_body)
            raise error_from_response(synthetic, status_code=upstream_status)
        return cast(dict[str, Any], json.loads(upstream_body))

    async def chat_completions_stream(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> AsyncIterator[bytes]:
        """Send a streaming chat-completions request directly to ollama.com."""
        await self._refresh_catalog_if_stale()
        out_body = _strip_codex_only_fields({**body, "stream": True})
        headers = {**self._app_request_headers(stream=True), "Authorization": f"Bearer {self._api_key}"}
        try:
            async with self._client.stream(
                "POST",
                f"{self._ollama_url}/v1/chat/completions",
                headers=headers,
                content=json.dumps(out_body).encode("utf-8"),
            ) as response:
                upstream_status = response.status_code
                if handle is not None:
                    handle.upstream_status = upstream_status
                    handle.upstream_headers = dict(response.headers)
                if upstream_status >= 400:
                    await response.aread()
                    raise error_from_response(response, status_code=upstream_status)
                async for chunk in stall_guarded(
                    response.aiter_bytes(),
                    first_item_timeout_s=LOCAL_STREAM_FIRST_BYTE_TIMEOUT_S,
                    idle_timeout_s=LOCAL_STREAM_IDLE_TIMEOUT_S,
                    what=f"ollama-cloud {out_body.get('model', '')}",
                    handle=handle,
                ):
                    yield chunk
        except httpx.HTTPError as exc:
            self._mark_unhealthy()
            raise BackendError(classification="transient", message=str(exc)) from exc

    async def responses(self, body: dict[str, Any], handle: CallHandle | None = None) -> dict[str, Any]:
        """Translate a Responses request to chat-completions and back."""
        chat_body = _responses_to_chat_request(body)
        chat_response = await self.chat_completions(chat_body, handle)
        return _chat_to_responses_response(chat_response)

    async def responses_stream(self, body: dict[str, Any], handle: CallHandle | None = None) -> AsyncIterator[bytes]:
        """Translate a chat-completions stream into a Responses stream."""

        def open_chat_stream(
            out_body: dict[str, Any], _headers: dict[str, str]
        ) -> AsyncContextManager[httpx.Response]:
            return self._direct_stream_cm(out_body)

        collector = ResponsesStreamCollector(
            chat_to_responses_stream(
                client=self._client,
                chat_url=f"{self._ollama_url}/v1/chat/completions",
                body=body,
                handle=handle,
                prep_body=lambda b: _strip_codex_only_fields({**b, "stream": True}),
                headers={},
                first_item_timeout_s=LOCAL_STREAM_FIRST_BYTE_TIMEOUT_S,
                idle_timeout_s=LOCAL_STREAM_IDLE_TIMEOUT_S,
                what_label="ollama-cloud",
                on_success=lambda: None,
                on_transport_error=self._mark_unhealthy,
                open_chat_stream=open_chat_stream,
                upstream_status_of=lambda response: response.status_code,
            )
        )
        try:
            async for chunk in collector.iter_through():
                yield chunk
        finally:
            if handle is not None:
                handle.stream_summary = collector.summary

    def cell_capabilities(self, model: str) -> CellCapabilities:
        """Return cached capabilities or conservative text-only defaults."""
        cached = self._capabilities_cache.get(model)
        if cached is not None:
            return cached
        return CellCapabilities(
            context_window=128_000,
            modalities=frozenset({"text"}),
            supports_tools=False,
            cost_rank=10,
        )

    async def refresh_advertised_models(self, *, now: float | None = None) -> None:
        """Force a catalog refresh and make the result immediately available."""
        self._catalog_fetched_at = 0.0
        await self._refresh_catalog_if_stale(now=now)

    async def _refresh_catalog_if_stale(self, *, now: float | None = None) -> None:
        timestamp = now if now is not None else time.time()
        if self._catalog and timestamp - self._catalog_fetched_at < self._catalog_refresh_s:
            return
        try:
            upstream_status, _upstream_headers, upstream_body = await self._direct_request(
                method="GET",
                url=f"{self._ollama_url}/api/tags",
                app_headers=self._app_request_headers(stream=False),
                body=b"",
                timeout=DEFAULT_HEALTH_TIMEOUT_S,
            )
        except httpx.HTTPError:
            self._mark_unhealthy()
            return
        if upstream_status == 401:
            self._healthy = False
            self._last_health_reason = "auth_invalid"
            return
        if upstream_status != 200:
            self._healthy = False
            self._last_health_reason = "unknown"
            return
        try:
            payload = json.loads(upstream_body)
        except (ValueError, TypeError):
            self._healthy = False
            self._last_health_reason = "unknown"
            return
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            self._healthy = False
            self._last_health_reason = "unknown"
            return
        slugs: list[str] = []
        for entry in models:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if isinstance(name, str) and name and name.endswith(self._model_suffix):
                slugs.append(name)
        self._catalog = tuple(slugs)
        self._catalog_fetched_at = timestamp
        self._healthy = True
        self._last_health_reason = "ok"
        await self._refresh_capabilities()

    async def _refresh_capabilities(self) -> None:
        """Populate capabilities from ollama.com's model detail endpoint."""
        for name in self._catalog:
            try:
                upstream_status, _upstream_headers, upstream_body = await self._direct_request(
                    method="POST",
                    url=f"{self._ollama_url}/api/show",
                    app_headers=self._app_request_headers(stream=False),
                    body=json.dumps({"name": name}).encode("utf-8"),
                    timeout=DEFAULT_HEALTH_TIMEOUT_S,
                )
            except httpx.HTTPError:
                return
            if upstream_status != 200:
                continue
            try:
                info = json.loads(upstream_body)
            except (ValueError, TypeError):
                continue
            capabilities_list = info.get("capabilities") or []
            capabilities_set = {str(item).lower() for item in capabilities_list if isinstance(item, str)}
            modalities: set[str] = {"text"}
            if "vision" in capabilities_set:
                modalities.add("image")
            if "audio" in capabilities_set:
                modalities.add("audio")
            context_window = 128_000
            parameter_count: int | None = None
            model_info = info.get("model_info") or {}
            if isinstance(model_info, dict):
                for key, value in model_info.items():
                    if (
                        isinstance(key, str)
                        and key.endswith("context_length")
                        and isinstance(value, int)
                        and value > 0
                    ):
                        context_window = value
                general_parameter_count = model_info.get("general.parameter_count")
                if isinstance(general_parameter_count, int) and general_parameter_count > 0:
                    parameter_count = general_parameter_count
            self._capabilities_cache[name] = CellCapabilities(
                context_window=context_window,
                modalities=frozenset(modalities),
                supports_tools="tools" in capabilities_set,
                cost_rank=10,
                parameter_count=parameter_count,
            )
