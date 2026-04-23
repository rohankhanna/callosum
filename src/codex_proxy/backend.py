from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol

BackendKind = Literal["openai_api_key", "azure_openai", "codex_auth_vault"]
HealthReason = Literal["ok", "rate_limited", "auth_invalid", "network", "unknown"]


@dataclass(frozen=True, slots=True)
class HealthStatus:
    available: bool
    reason: HealthReason
    retry_after_s: float | None = None


@dataclass(frozen=True, slots=True)
class UsageSnapshot:
    remaining_fraction: float | None
    cooldown_until_ts: float | None
    weekly_exhausted: bool
    probed_at_ts: float


class Backend(Protocol):
    id: str
    kind: BackendKind
    advertised_models: frozenset[str]
    # True when this backend speaks the OpenAI Responses API natively. The
    # /v1/responses route filters the pool by this flag before ranking so it
    # never picks a backend that cannot serve the request.
    responses_supported: bool

    async def health(self) -> HealthStatus: ...

    async def usage_snapshot(self) -> UsageSnapshot: ...

    async def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]: ...

    def chat_completions_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]: ...

    async def responses(self, body: dict[str, Any]) -> dict[str, Any]: ...

    def responses_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]: ...

    async def aclose(self) -> None: ...
