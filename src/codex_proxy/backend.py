from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from codex_proxy.codex_quota import CodexQuotaSnapshot
from codex_proxy.sse_tee import ResponsesStreamSummary

BackendKind = Literal["codex_auth_vault"]
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


@dataclass(slots=True)
class CallHandle:
    """Per-call scratch space the dispatch layer hands to a backend.

    The backend populates whatever it can observe (upstream status, response
    headers, pre/post quota snapshots, and for streams the final
    `response.completed` summary). Each call gets its own handle so concurrent
    requests don't race — do not reuse a handle across calls.
    """

    upstream_status: int | None = None
    upstream_headers: Mapping[str, str] = field(default_factory=dict)
    quota_before: CodexQuotaSnapshot | None = None
    quota_after: CodexQuotaSnapshot | None = None
    stream_summary: ResponsesStreamSummary | None = None


class Backend(Protocol):
    id: str
    kind: BackendKind
    advertised_models: frozenset[str]

    async def health(self) -> HealthStatus: ...

    async def usage_snapshot(self) -> UsageSnapshot: ...

    async def chat_completions(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> dict[str, Any]: ...

    def chat_completions_stream(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> AsyncIterator[bytes]: ...

    async def responses(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> dict[str, Any]: ...

    def responses_stream(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> AsyncIterator[bytes]: ...

    async def aclose(self) -> None: ...
