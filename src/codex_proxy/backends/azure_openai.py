from __future__ import annotations

import time
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

import httpx

from codex_proxy.backend import BackendKind, HealthStatus, UsageSnapshot
from codex_proxy.backends._http import DEFAULT_COOLDOWN_S, error_from_response
from codex_proxy.errors import BackendError
from codex_proxy.state import StateStore


class AzureOpenAIBackend:
    """Forwards requests to an Azure OpenAI resource using api-key auth and deployment URLs."""

    kind: BackendKind = "azure_openai"
    responses_supported: bool = False

    def __init__(
        self,
        *,
        id: str,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployments: Mapping[str, str],
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = 60.0,
        state_store: StateStore | None = None,
    ) -> None:
        if not deployments:
            raise ValueError(f"backend {id!r}: deployments mapping cannot be empty")
        self.id = id
        self.advertised_models = frozenset(deployments.keys())
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._api_version = api_version
        self._deployments = dict(deployments)
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
        url = self._url_for(body)
        try:
            response = await self._client.post(url, json=body, headers=self._headers())
        except httpx.HTTPError as exc:
            raise BackendError(classification="transient", message=str(exc)) from exc
        if response.status_code >= 400:
            err = error_from_response(response)
            self._apply_error_to_usage(err)
            raise err
        return cast(dict[str, Any], response.json())

    async def chat_completions_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        url = self._url_for(body)
        try:
            stream_ctx = self._client.stream("POST", url, json=body, headers=self._headers())
            async with stream_ctx as response:
                if response.status_code >= 400:
                    await response.aread()
                    err = error_from_response(response)
                    self._apply_error_to_usage(err)
                    raise err
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.HTTPError as exc:
            raise BackendError(classification="transient", message=str(exc)) from exc

    async def responses(self, body: dict[str, Any]) -> dict[str, Any]:
        raise BackendError(
            classification="unknown_model",
            status_code=404,
            message=f"azure_openai backend {self.id!r} does not serve the Responses API",
        )

    async def responses_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        raise BackendError(
            classification="unknown_model",
            status_code=404,
            message=f"azure_openai backend {self.id!r} does not serve the Responses API",
        )
        if False:
            yield b""  # pragma: no cover

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        return {"api-key": self._api_key}

    def _url_for(self, body: dict[str, Any]) -> str:
        model = body.get("model")
        if not isinstance(model, str):
            raise BackendError(
                classification="client_error",
                status_code=400,
                message="azure_openai requires a string 'model' in the request body",
            )
        deployment = self._deployments.get(model)
        if deployment is None:
            raise BackendError(
                classification="unknown_model",
                status_code=404,
                message=f"model {model!r} has no azure deployment mapping",
            )
        return (
            f"{self._endpoint}/openai/deployments/{deployment}"
            f"/chat/completions?api-version={self._api_version}"
        )

    def _apply_error_to_usage(self, err: BackendError) -> None:
        if err.classification != "rate_limited":
            return
        now = time.time()
        cooldown_until = now + (err.retry_after_s or DEFAULT_COOLDOWN_S)
        self._usage = UsageSnapshot(
            remaining_fraction=self._usage.remaining_fraction,
            cooldown_until_ts=cooldown_until,
            weekly_exhausted=self._usage.weekly_exhausted,
            probed_at_ts=now,
        )
        if self._state_store is not None:
            self._state_store.save_usage(self.id, self._usage)
