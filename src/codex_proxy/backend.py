from __future__ import annotations

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

    async def health(self) -> HealthStatus: ...

    async def usage_snapshot(self) -> UsageSnapshot: ...

    async def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...
