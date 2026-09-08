from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from callosum.codex_quota import CodexQuotaSnapshot
from callosum.sse_tee import ResponsesStreamSummary

BackendKind = Literal[
    "codex_auth_vault",
    "codex_gateway",
    "litellm_gateway",
    "ollama_cloud",
    "openrouter",
]
HealthReason = Literal[
    "ok",
    "rate_limited",
    "auth_invalid",
    "network",
    "unknown",
    # Local-lane catalog reasons. `catalog_cli_broken` = the shelled-out
    # local model catalog CLI failed to run (missing, hung, or non-zero exit —
    # e.g. an orphaned pipx venv lost its package). `catalog_empty` = the CLI
    # is healthy but the local model registry lists zero models. Distinct from
    # the opaque "unknown" so /status names the real cause and the operator
    # knows whether to repair the sibling CLI or populate the registry.
    "catalog_cli_broken",
    "catalog_empty",
    # Ollama-cloud / OpenRouter reason for missing or invalid API key.
    # `no-key` = the API key was not provided or is empty. Distinct from
    # "auth_invalid" (the upstream rejected the key — operator needs to
    # rotate or re-provision it) and "network" (upstream transport
    # unreachable) so /status names whether the operator should provide
    # a key vs. rotate it vs. check network.
    "no-key",
]


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
    # Wall-clock (time.time()) timestamp of the first streamed chunk received
    # from upstream, set by the dispatch layer at the first-chunk probe. NULL
    # for non-stream calls and for streams that produced no chunks (e.g.
    # empty StopAsyncIteration, pre-first-chunk timeout/error). Persisted as
    # `requests.ttfb_ms = (first_byte_at - ts_start) * 1000` to feed the
    # per-cell time estimator's TTFB-vs-decode split and
    # data-driven stall-guard tuning. Same clock as
    # ts_start/ts_end so the subtraction is valid.
    first_byte_at: float | None = None
    # Largest inter-chunk idle gap (seconds) observed by the local-lane
    # stall guard, i.e. the longest the upstream made us wait between two
    # consecutive chunks once data was flowing. Set by stall_guarded on
    # local streams only (remote lanes are not stall-guarded). NULL for
    # non-stream calls, remote streams, and streams that ended before a
    # second chunk. Persisted as requests.idle_gap_ms to tune
    # CALLOSUM_LOCAL_STREAM_IDLE_TIMEOUT_S. monotonic clock
    # (a pure duration, not a wall-clock stamp).
    max_idle_gap_s: float | None = None


class Backend(Protocol):
    id: str
    kind: BackendKind

    @property
    def advertised_models(self) -> frozenset[str]:
        """The set of model names this backend will accept. Implementations
        may set it as a plain attribute (Codex backends, fakes — static after
        construction) or compute it dynamically (the LiteLLM gateway backend
        recomputes from each periodic `/v1/models` poll).
        """
        ...

    async def health(self) -> HealthStatus: ...

    async def usage_snapshot(self) -> UsageSnapshot: ...

    async def quota_snapshot(self) -> CodexQuotaSnapshot | None:
        """Most recent CodexQuotaSnapshot observed from upstream response
        headers, or None if this backend has not been called yet (or is a
        kind that does not produce one)."""
        ...

    async def chat_completions(self, body: dict[str, Any], handle: CallHandle | None = None) -> dict[str, Any]: ...

    def chat_completions_stream(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> AsyncIterator[bytes]: ...

    async def responses(self, body: dict[str, Any], handle: CallHandle | None = None) -> dict[str, Any]: ...

    def responses_stream(self, body: dict[str, Any], handle: CallHandle | None = None) -> AsyncIterator[bytes]: ...

    async def aclose(self) -> None: ...
