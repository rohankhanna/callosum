from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx

from codex_proxy.backend import BackendKind, HealthStatus, UsageSnapshot
from codex_proxy.errors import BackendError, classify_http_status
from codex_proxy.state import StateStore

_DEFAULT_COOLDOWN_S = 60.0


class OpenAIApiKeyBackend:
    """Forwards requests to an OpenAI-compatible HTTP endpoint using a bearer token."""

    kind: BackendKind = "openai_api_key"

    def __init__(
        self,
        *,
        id: str,
        api_key: str,
        advertised_models: frozenset[str],
        base_url: str = "https://api.openai.com/v1",
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = 60.0,
        state_store: StateStore | None = None,
    ) -> None:
        self.id = id
        self.advertised_models = advertised_models
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(transport=transport, timeout=timeout_s)
            self._owns_client = True
        self._state_store = state_store
        loaded = state_store.load_usage(id) if state_store is not None else None
        self._usage = loaded or UsageSnapshot(
            remaining_fraction=None,
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=time.time(),
        )

    async def health(self) -> HealthStatus:
        return HealthStatus(available=True, reason="ok")

    async def usage_snapshot(self) -> UsageSnapshot:
        return self._usage

    async def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.HTTPError as exc:
            raise BackendError(classification="transient", message=str(exc)) from exc
        if response.status_code >= 400:
            err = _error_from_response(response)
            self._apply_error_to_usage(err)
            raise err
        return cast(dict[str, Any], response.json())

    async def chat_completions_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        try:
            stream_ctx = self._client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            async with stream_ctx as response:
                if response.status_code >= 400:
                    await response.aread()
                    err = _error_from_response(response)
                    self._apply_error_to_usage(err)
                    raise err
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.HTTPError as exc:
            raise BackendError(classification="transient", message=str(exc)) from exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _apply_error_to_usage(self, err: BackendError) -> None:
        if err.classification != "rate_limited":
            return
        now = time.time()
        cooldown_until = now + (err.retry_after_s or _DEFAULT_COOLDOWN_S)
        self._usage = UsageSnapshot(
            remaining_fraction=self._usage.remaining_fraction,
            cooldown_until_ts=cooldown_until,
            weekly_exhausted=self._usage.weekly_exhausted,
            probed_at_ts=now,
        )
        if self._state_store is not None:
            self._state_store.save_usage(self.id, self._usage)


def _error_from_response(response: httpx.Response) -> BackendError:
    classification = classify_http_status(response.status_code)
    retry_after = response.headers.get("retry-after")
    retry_after_s: float | None = None
    if retry_after is not None:
        try:
            retry_after_s = float(retry_after)
        except ValueError:
            retry_after_s = None
    return BackendError(
        classification=classification,
        status_code=response.status_code,
        retry_after_s=retry_after_s,
        message=f"upstream {response.status_code}",
    )
