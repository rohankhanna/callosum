from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx

from codex_proxy.backend import BackendKind, HealthStatus, UsageSnapshot


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

    async def health(self) -> HealthStatus:
        return HealthStatus(available=True, reason="ok")

    async def usage_snapshot(self) -> UsageSnapshot:
        return UsageSnapshot(
            remaining_fraction=None,
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=time.time(),
        )

    async def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        response = await self._client.post(
            f"{self._base_url}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    async def chat_completions_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        async with self._client.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self._api_key}"},
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                yield chunk

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
