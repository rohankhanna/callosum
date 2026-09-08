"""OpenRouter backend using direct API-key authentication.

The backend talks directly to OpenRouter's OpenAI-compatible chat endpoint
and model catalog. Requests use the configured OpenRouter API key as a bearer
token while retaining Callosum's model filtering, provider-residency controls,
catalog synthesis, and stall-guarded streaming behavior.
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

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CATALOG_REFRESH_S = 300.0
DEFAULT_HEALTH_TIMEOUT_S = 10.0
DEFAULT_CALL_TIMEOUT_S = 300.0
OPENROUTER_PRIORITY_OFFSET = 2_000

# Countries whose data-residency rules should reject an inference provider.
DEFAULT_BLOCKED_COUNTRIES = frozenset({"CN", "RU", "KP"})
# Families the paid ollama_cloud backend already serves. They are dropped from
# the OpenRouter catalog so Callosum does not pay twice for the same models.
DEFAULT_EXCLUDE_FAMILIES = frozenset({"model-a0g3", "model-a0g1", "model-a0d5", "model-a0e2"})
# Cooldown applied once OpenRouter reports insufficient credits.
OPENROUTER_EXHAUSTED_COOLDOWN_S = 3_600.0


class OpenRouterBackend:
    """Models served directly by OpenRouter with an API key.

    The backend keeps its own BackendKind="openrouter" so dispatch treats
    it as a remote backend and can apply OpenRouter-specific catalog and
    provider-preference logic. Metering remains advisory; OpenRouter usage is
    consumed from response bodies, and HTTP 402 sets an exhausted cooldown.
    """

    kind: BackendKind = "openrouter"

    def __init__(
        self,
        *,
        id: str,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        model_filter: str = "all",
        allowlist: frozenset[str] = frozenset(),
        model_prefix: str | None = None,
        exclude_families: frozenset[str] = DEFAULT_EXCLUDE_FAMILIES,
        catalog_refresh_s: float = DEFAULT_CATALOG_REFRESH_S,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
    ) -> None:
        self.id = id
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        if model_filter not in ("all", "allowlist", "prefix"):
            raise ValueError(f"model_filter must be all/allowlist/prefix, got {model_filter!r}")
        self._model_filter = model_filter
        self._allowlist = frozenset(allowlist)
        self._model_prefix = model_prefix
        self._exclude_families = frozenset(exclude_families)
        self._catalog_refresh_s = catalog_refresh_s
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(transport=transport, timeout=timeout_s)
            self._owns_client = True
        self._catalog: tuple[str, ...] = ()
        self._context_windows: dict[str, int] = {}
        self._catalog_fetched_at: float = 0.0
        self._healthy: bool = False
        self._last_health_reason: str = "unknown"
        self._exhausted_until: float = 0.0

    @property
    def advertised_models(self) -> frozenset[str]:
        return frozenset(self._catalog)

    @property
    def model_metadata(self) -> dict[str, ModelMetadata]:
        """Synthesize model metadata for each cataloged OpenRouter model."""
        return {
            slug: ModelMetadata(
                slug=slug,
                supported_in_api=True,
                visibility="list",
                priority=OPENROUTER_PRIORITY_OFFSET + idx,
                supported_reasoning_levels=("default",),
                context_window=self._context_windows.get(slug),
            )
            for idx, slug in enumerate(self._catalog)
        }

    async def health(self) -> HealthStatus:
        await self._refresh_catalog_if_stale()
        if self._healthy:
            return HealthStatus(available=True, reason="ok")
        if self._last_health_reason == "network":
            return HealthStatus(available=False, reason="network")
        if self._last_health_reason == "no-key":
            return HealthStatus(available=False, reason="no-key")
        if self._last_health_reason == "auth_invalid":
            return HealthStatus(available=False, reason="auth_invalid")
        return HealthStatus(available=False, reason="unknown")

    async def usage_snapshot(self) -> UsageSnapshot:
        """Report advisory quota state and short outage cooldowns."""
        now = time.time()
        if now < self._exhausted_until:
            return UsageSnapshot(
                remaining_fraction=0.0,
                cooldown_until_ts=self._exhausted_until,
                weekly_exhausted=True,
                probed_at_ts=now,
            )
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
        timeout: float = DEFAULT_CALL_TIMEOUT_S,  # noqa: ASYNC109 - forwarded to httpx, not asyncio.wait_for
    ) -> tuple[int, dict[str, str], bytes]:
        headers = {**app_headers, "Authorization": f"Bearer {self._api_key}"}
        response = await self._client.request(
            method,
            url,
            headers=headers,
            content=body,
            timeout=timeout,
        )
        return response.status_code, dict(response.headers), response.content

    def _app_request_headers(self, *, stream: bool) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }

    def _apply_model_filter(self, slugs: list[str]) -> list[str]:
        if self._model_filter == "allowlist":
            allow = self._allowlist
            return [slug for slug in slugs if slug in allow]
        if self._model_filter == "prefix":
            prefix = self._model_prefix or ""
            return [slug for slug in slugs if prefix and slug.startswith(prefix)]
        return slugs

    def _apply_family_exclude(self, slugs: list[str]) -> list[str]:
        if not self._exclude_families:
            return slugs
        tokens = self._exclude_families
        return [slug for slug in slugs if not any(token in slug.lower() for token in tokens)]

    @contextlib.asynccontextmanager
    async def _direct_stream_cm(self, out_body: dict[str, Any]) -> AsyncIterator[httpx.Response]:
        headers = {
            **self._app_request_headers(stream=True),
            "Authorization": f"Bearer {self._api_key}",
        }
        async with self._client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=headers,
            content=json.dumps(out_body).encode("utf-8"),
        ) as response:
            yield response

    async def chat_completions(self, body: dict[str, Any], handle: CallHandle | None = None) -> dict[str, Any]:
        """Send a non-streaming chat request directly to OpenRouter."""
        await self._refresh_catalog_if_stale()
        prepped = _strip_codex_only_fields({**body, "stream": False})
        out_body = prepped
        try:
            upstream_status, upstream_headers, upstream_body = await self._direct_request(
                method="POST",
                url=f"{self._base_url}/chat/completions",
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
            if upstream_status == 402:
                self._exhausted_until = time.time() + OPENROUTER_EXHAUSTED_COOLDOWN_S
                raise BackendError(
                    classification="rate_limited",
                    status_code=402,
                    message="openrouter insufficient credits (402); backend cooldown set",
                )
            synthetic = httpx.Response(upstream_status, headers=upstream_headers, content=upstream_body)
            raise error_from_response(synthetic, status_code=upstream_status)
        return cast(dict[str, Any], json.loads(upstream_body))

    async def chat_completions_stream(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> AsyncIterator[bytes]:
        """Yield OpenRouter's chat-completions stream directly."""
        await self._refresh_catalog_if_stale()
        prepped = _strip_codex_only_fields({**body, "stream": True})
        out_body = prepped
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers={
                    **self._app_request_headers(stream=True),
                    "Authorization": f"Bearer {self._api_key}",
                },
                content=json.dumps(out_body).encode("utf-8"),
            ) as response:
                upstream_status = response.status_code
                if handle is not None:
                    handle.upstream_status = upstream_status
                    handle.upstream_headers = dict(response.headers)
                if upstream_status >= 400:
                    if upstream_status == 402:
                        self._exhausted_until = time.time() + OPENROUTER_EXHAUSTED_COOLDOWN_S
                        raise BackendError(
                            classification="rate_limited",
                            status_code=402,
                            message="openrouter insufficient credits (402); backend cooldown set",
                        )
                    await response.aread()
                    raise error_from_response(response, status_code=upstream_status)
                async for chunk in stall_guarded(
                    response.aiter_bytes(),
                    first_item_timeout_s=LOCAL_STREAM_FIRST_BYTE_TIMEOUT_S,
                    idle_timeout_s=LOCAL_STREAM_IDLE_TIMEOUT_S,
                    what=f"openrouter {out_body.get('model', '')}",
                    handle=handle,
                ):
                    yield chunk
        except httpx.HTTPError as exc:
            raise BackendError(classification="transient", message=str(exc)) from exc

    async def responses(self, body: dict[str, Any], handle: CallHandle | None = None) -> dict[str, Any]:
        chat_body = _responses_to_chat_request(body)
        chat_response = await self.chat_completions(chat_body, handle)
        return _chat_to_responses_response(chat_response)

    async def responses_stream(self, body: dict[str, Any], handle: CallHandle | None = None) -> AsyncIterator[bytes]:
        """Translate an OpenRouter chat stream into Responses events."""

        def open_chat_stream(out_body: dict[str, Any], _headers: dict[str, str]) -> AsyncContextManager[httpx.Response]:
            return self._direct_stream_cm(out_body)

        collector = ResponsesStreamCollector(
            chat_to_responses_stream(
                client=self._client,
                chat_url=f"{self._base_url}/chat/completions",
                body=body,
                handle=handle,
                prep_body=lambda chat_body: _strip_codex_only_fields({**chat_body, "stream": True}),
                headers={},
                first_item_timeout_s=LOCAL_STREAM_FIRST_BYTE_TIMEOUT_S,
                idle_timeout_s=LOCAL_STREAM_IDLE_TIMEOUT_S,
                what_label="openrouter",
                on_success=lambda: None,
                on_transport_error=self._mark_unhealthy,
                open_chat_stream=open_chat_stream,
                upstream_status_of=lambda response: response.status_code,
            )
        )
        try:
            async for chunk in collector.iter_through():
                yield chunk
        except BackendError as exc:
            if exc.status_code == 402:
                self._exhausted_until = time.time() + OPENROUTER_EXHAUSTED_COOLDOWN_S
                raise BackendError(
                    classification="rate_limited",
                    status_code=402,
                    message="openrouter insufficient credits (402); backend cooldown set",
                ) from exc
            raise
        finally:
            if handle is not None:
                handle.stream_summary = collector.summary

    def cell_capabilities(self, model: str) -> CellCapabilities:
        return CellCapabilities(
            context_window=self._context_windows.get(model, 128_000),
            modalities=frozenset({"text"}),
            supports_tools=False,
            cost_rank=10,
        )

    async def refresh_advertised_models(self, *, now: float | None = None) -> None:
        """Force a catalog refresh and make the result immediately visible."""
        self._catalog_fetched_at = 0.0
        await self._refresh_catalog_if_stale(now=now)

    async def _refresh_catalog_if_stale(self, *, now: float | None = None) -> None:
        timestamp = now if now is not None else time.time()
        if self._catalog and timestamp - self._catalog_fetched_at < self._catalog_refresh_s:
            return
        if not self._api_key:
            self._healthy = False
            self._last_health_reason = "no-key"
            return
        try:
            models_status, _models_headers, models_body = await self._direct_request(
                method="GET",
                url=f"{self._base_url}/models",
                app_headers=self._app_request_headers(stream=False),
                body=b"",
                timeout=DEFAULT_HEALTH_TIMEOUT_S,
            )
        except httpx.HTTPError:
            self._mark_unhealthy()
            return
        if models_status == 401:
            self._healthy = False
            self._last_health_reason = "auth_invalid"
            return
        if models_status != 200:
            self._healthy = False
            self._last_health_reason = "unknown"
            return
        try:
            models_payload = json.loads(models_body)
        except (ValueError, TypeError):
            self._healthy = False
            self._last_health_reason = "unknown"
            return
        models = models_payload.get("data") if isinstance(models_payload, dict) else None
        if not isinstance(models, list):
            self._healthy = False
            self._last_health_reason = "unknown"
            return

        slugs: list[str] = []
        context_windows: dict[str, int] = {}
        for entry in models:
            if not isinstance(entry, dict):
                continue
            slug = entry.get("id")
            if not isinstance(slug, str) or not slug:
                continue
            slugs.append(slug)
            context_length = entry.get("context_length")
            if isinstance(context_length, int) and context_length > 0:
                context_windows[slug] = context_length
        slugs = self._apply_model_filter(slugs)
        slugs = self._apply_family_exclude(slugs)
        self._catalog = tuple(slugs)
        self._context_windows = context_windows
        self._catalog_fetched_at = timestamp
        self._healthy = True
        self._last_health_reason = "ok"
