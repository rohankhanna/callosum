"""OpenRouter backend — routes to models served by OpenRouter's hosted
aggregator through credential proxy's boundary-native proxy custody (the real
OpenRouter API key NEVER enters callosum's process).

This is a conscious REVERSAL of the 2026-05-26 removal (a prior commit) of the
prior openrouter_free backend. That backend was a free-tier
Codex-exhaustion fallback: it held OPENROUTER_API_KEY in env, sat OUTSIDE
the cell grid, shadow-advertised Codex model names and substituted a free
model at call time, faked /v1/responses streaming as two synthetic SSE
events, and used a remaining_fraction=0.001 rank hack to suppress itself.
It was removed because it was "an interim hack no longer the direction; the
non-Codex path is local LLM gateway + LiteLLMGatewayBackend." The reversal is
justified by (a) a new motivation — the operator holds expiring *paid*
OpenRouter credits to spend, not free-tier fallback — and (b) post-removal
architecture changes: ollama_cloud (2026-08-04) re-opened the
"remote non-Codex backend" door, and the **credential proxy stand-in custody seam** now
exists, so the old env-var-key custody pattern is retired for every remote
backend. This backend is rebuilt mirroring ollama_cloud (the direct
credential proxy-custody sibling), not revived from the deleted file — it shares almost
no load-bearing structure with the old design.

Callosum mints a short-TTL openrouter-scoped stand-in token via credential proxy's
existing POST /v1/standin, then POSTs every OpenRouter call (chat,
catalog) through credential proxy's existing POST /v1/proxy (buffered) or
POST /v1/proxy/stream (streaming) with the stand-in as
Authorization: Bearer <stand-in>. credential proxy validates the token, reads the
real OpenRouter API key from its pass store, strips the caller
authorization/chatgpt-account-id, and injects Authorization: Bearer
<real key> on the final hop. The real key therefore NEVER crosses into
callosum's memory — this backend holds only a short-TTL stand-in token
(revocable, auto-expiring, loopback-minted), unifying custody with
credential_proxy and ollama_cloud.

Catalog discovery is **full auto-discovery by default** (all of
OpenRouter's /v1/models), with a configurable filter (all /
allowlist / prefix). An ollama-cloud-overlap **family-exclude**
(default model-a0g3/model-a0g1/model-a0d5/model-a0e2) drops families the paid
ollama_cloud backend already serves, so OpenRouter doesn't get paid for
models the operator already has covered. Combined with ollama_cloud's higher
priority (offset 1000 < 2000), this keeps OpenRouter the conservative-overflow
remote for models NOT on ollama_cloud.


Metering is **honest-advisory**, not header-parsed: OpenRouter returns
usage in the response body (not Codex quota headers), which the dispatch
layer's CallHandle + _extract_tokens consume generically.
usage_snapshot() reports "remote, full, eligible, no signal yet" and
quota_snapshot() returns None — both advisory, separate from per-request
accounting. A 402 (insufficient credits) flips an in-memory exhausted flag +
cooldown so dispatch rotates away; the credits-expiry motivation is why this
backend exists, so exhausted-credits is the one signal worth caching.

This backend is OPTIONAL and **env-gated OFF by default**: it is not
constructed unless CALLOSUM_OPENROUTER_ENABLED=1, so enabling is a no-op
for live routing until the operator turns it on. The credential proxy proxy path is
MANDATORY (not separately gated) when the backend is enabled — if credential proxy is
down, the openrouter scope is not configured, or the pass key entry is
missing/empty, the backend goes unhealthy with a clear "no-key" reason
(the correct failure mode). The four chat methods hit OpenRouter's
OpenAI-compatible /v1/chat/completions endpoint (same path litellm_gateway
and ollama_cloud use, so the shared Responses↔Chat translators in
callosum.backends._responses_chat apply verbatim). The live-routing flip
is operator-gated; until credential proxy ships the openrouter scope (see the
handoff doc) the backend is inert.
"""

from __future__ import annotations

import base64
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

# Direct upstream target: OpenRouter's hosted OpenAI-compatible aggregator.
# Callosum never presents the real OpenRouter API key itself — it POSTs each
# call through credential proxy's `/v1/proxy`(`/stream`) with a short-TTL
# `openrouter`-scoped stand-in token, and credential proxy injects the real key on the
# final hop. See `OpenRouterBackend._proxy_buffered` / `responses_stream`.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CATALOG_REFRESH_S = 300.0
# Remote TLS+WAN round-trip latency for /models + /providers is higher than a
# local daemon; raise so catalog refresh doesn't flap. OpenRouter's catalog
# is large and changes slowly, so a 5-minute refresh is plenty.
DEFAULT_HEALTH_TIMEOUT_S = 10.0
DEFAULT_CALL_TIMEOUT_S = 300.0
# Priority offset for OpenRouter cells in the merged cell grid. Remote Codex
# priorities are small ints (tens); ollama_cloud sits at +1_000 and local at
# +10_000. OpenRouter is a CONSERVATIVE-OVERFLOW remote: behind Codex and
# ollama_cloud (so the operator spends paid Codex quota and the already-paid
# ollama_cloud first), before free local. This same number feeds
# `_remote_catalog_priorities` as the cold-start cost_rank prior: OpenRouter
# models (never measured — no quota headers) land at `prior_floor + offset`
# inside CostRankProvider, so they are conservatively treated as expensive
# remote cells until the operator overrides via `cost_rank_overrides`. See
# routing/cost_model.py.
OPENROUTER_PRIORITY_OFFSET = 2_000

# ---------- credential proxy stand-in custody (loopback-only, lazy+cached) ------------
DEFAULT_CUSTODY_URL = "http://127.0.0.1:7342"
DEFAULT_CUSTODY_ACCOUNT = "primary"
# CRITICAL: the scope MUST be sent explicitly — credential_proxy omits `scope`
# (credential proxy defaults it to `provider-upstream`), but this path needs the
# dedicated `openrouter` scope so credential proxy routes to the OpenRouter key in its
# `pass` store. Short on purpose (re-mint is a cheap loopback POST).
OPENROUTER_SCOPE = "openrouter"
OPENROUTER_STANDIN_TTL_S = 300
# credential proxy surfaces the real upstream status on its streaming proxy via this
# response header (the proxy itself returns HTTP 200 whenever the hop
# succeeded). Mirrors `credential_proxy.CUSTODY_UPSTREAM_STATUS_HEADER` and
# `ollama_cloud.CUSTODY_UPSTREAM_STATUS_HEADER`.
CUSTODY_UPSTREAM_STATUS_HEADER = "x-credential proxy-upstream-status"

# ---------- data-residency defaults -----------------------------------------
# The regime set: countries whose authoritarian governments the operator
# does not want their data reaching. Default covers China, Russia, North
# Korea; extensible via CALLOSUM_OPENROUTER_BLOCKED_COUNTRIES. Enforcement is
# per-request via OpenRouter `provider.ignore` (NOT a model-origin blocklist),
# derived from each provider's `datacenters`/`headquarters` country metadata.
DEFAULT_BLOCKED_COUNTRIES = frozenset({"CN", "RU", "KP"})
# Families the paid ollama_cloud backend already serves — drop these from the
# OpenRouter catalog so the operator doesn't pay OpenRouter for models they
# already have covered. Best-effort family-level (exact-model dedup is
# deferred; the slug namespaces differ, e.g. `model-a0g3/model-a0g3-2.5` vs
# `model-a0f3:cloud`). Extensible via CALLOSUM_OPENROUTER_EXCLUDE_FAMILIES.
DEFAULT_EXCLUDE_FAMILIES = frozenset({"model-a0g3", "model-a0g1", "model-a0d5", "model-a0e2"})
# Cooldown applied once OpenRouter returns 402 (insufficient credits). Long
# because credits are purchased in bulk; re-probing every minute would just
# re-fail. The operator tops up credits and restarts (or the cooldown
# expires) to re-enable. classified as `rate_limited` so dispatch rotates to
# another cell immediately.
OPENROUTER_EXHAUSTED_COOLDOWN_S = 3_600.0


class OpenRouterBackend:
    """Models served by OpenRouter, reached through credential proxy's proxy.

    Boundary-native credential custody: the REAL OpenRouter API key NEVER
    enters this process. Callosum mints a short-TTL openrouter-scoped
    stand-in token via credential proxy's POST /v1/standin (loopback), then POSTs
    each OpenRouter request through credential proxy's POST /v1/proxy (buffered) or
    POST /v1/proxy/stream (streaming) with the stand-in as bearer. credential proxy
    validates the token, reads the real key from its pass store, strips
    caller Authorization/chatgpt-account-id, and injects
    Authorization: Bearer <real key> on the final hop to OpenRouter.

    The backend keeps its own BackendKind="openrouter" (distinct from
    litellm_gateway and ollama_cloud) so dispatch routes it as a
    REMOTE fleet (not local), and so the catalog/dispatch hooks stay scoped to
    OpenRouter's shape. Honest-advisory metering + a 402-exhausted cooldown.
    The backend is env-gated OFF by default (CALLOSUM_OPENROUTER_ENABLED=1
    to enable) and INERT until credential proxy ships the openrouter scope.

    The no-key health reason, under proxy custody, means the stand-in mint
    failed (credential proxy unreachable on /v1/standin or returned
    non-200/non-401) OR credential proxy's /v1/proxy returned 503 (the
    openrouter scope is not wired at credential proxy, or the pass entry is
    missing/empty). It does NOT mean OpenRouter rejected the key (that
    surfaces as auth_invalid via the wrapper's upstream status).
    """

    kind: BackendKind = "openrouter"

    def __init__(
        self,
        *,
        id: str,
        base_url: str = DEFAULT_BASE_URL,
        model_filter: str = "all",
        allowlist: frozenset[str] = frozenset(),
        model_prefix: str | None = None,
        blocked_countries: frozenset[str] = DEFAULT_BLOCKED_COUNTRIES,
        blocked_providers: frozenset[str] = frozenset(),
        allowed_providers: frozenset[str] = frozenset(),
        exclude_families: frozenset[str] = DEFAULT_EXCLUDE_FAMILIES,
        catalog_refresh_s: float = DEFAULT_CATALOG_REFRESH_S,
        custody_url: str = DEFAULT_CUSTODY_URL,
        custody_account: str = DEFAULT_CUSTODY_ACCOUNT,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
    ) -> None:
        self.id = id
        self._base_url = base_url.rstrip("/")
        if model_filter not in ("all", "allowlist", "prefix"):
            raise ValueError(f"model_filter must be all/allowlist/prefix, got {model_filter!r}")
        self._model_filter = model_filter
        self._allowlist = frozenset(allowlist)
        self._model_prefix = model_prefix
        self._blocked_countries = frozenset(blocked_countries)
        self._blocked_providers = frozenset(blocked_providers)
        self._allowed_providers = frozenset(allowed_providers)
        self._exclude_families = frozenset(exclude_families)
        self._catalog_refresh_s = catalog_refresh_s
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(transport=transport, timeout=timeout_s)
            self._owns_client = True
        # Catalog state — populated lazily from /models (through the proxy).
        self._catalog: tuple[str, ...] = ()
        self._context_windows: dict[str, int] = {}
        self._catalog_fetched_at: float = 0.0
        # Blocked-provider deny set derived from /providers at refresh; empty
        # until the first refresh lands. Injected as `provider.ignore` on
        # every outbound chat call.
        self._regime_provider_deny: frozenset[str] = frozenset()
        # Health derived from the most recent /models poll. Until the first
        # poll lands we report unavailable so dispatch doesn't route here
        # before discovery completes.
        self._healthy: bool = False
        self._last_health_reason: str = "unknown"
        # 402 exhausted-credits cooldown (wall-clock). 0.0 = not exhausted.
        self._exhausted_until: float = 0.0
        # credential proxy boundary-native proxy custody. Callosum holds ONLY a
        # short-TTL `openrouter`-scoped stand-in token (never the real key).
        self._custody_url = custody_url.rstrip("/")
        self._custody_account = custody_account
        self._standin_token: str | None = None
        self._standin_expires_at: float = 0.0

    @property
    def advertised_models(self) -> frozenset[str]:
        return frozenset(self._catalog)

    @property
    def model_metadata(self) -> dict[str, ModelMetadata]:
        """Synthesize ModelMetadata for each cataloged OpenRouter model.

        OpenRouter models lack Codex-style metadata fields; populate the cell
        grid by hand so the merge in app.py picks them up without changing the
        cell-grid filter logic:

          * supported_in_api=True, visibility="list" — pass the inclusion
            filter in `live_completion_models_from_metadata`.
          * supported_reasoning_levels=("default",) — one cell per OpenRouter
            model rather than one cell per (model, effort). v1 default-only;
            OpenRouter's per-model `reasoning.supported_efforts` could enrich
            this later.
          * priority=OPENROUTER_PRIORITY_OFFSET + idx — sorts in the remote
            band, after curated Codex and ollama_cloud (1000), before free
            local (10_000). Conservative-overflow placement.
          * context_window — from OpenRouter's `context_length` when present.
        """
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
        # Force a catalog refresh if we're past TTL so dispatch sees
        # OpenRouter's current state rather than a stale snapshot.
        await self._refresh_catalog_if_stale()
        if self._healthy:
            return HealthStatus(available=True, reason="ok")
        # Surface the concrete failure reason so /status names the cause:
        # "network" (credential proxy/OpenRouter unreachable), "no-key" (stand-in mint
        # failed — credential proxy unreachable on /v1/standin or returned non-200/non-401
        # — OR credential proxy /v1/proxy returned 503 because the `openrouter` scope is
        # not wired / the pass entry is missing/empty), "auth_invalid"
        # (OpenRouter rejected the key credential proxy injected), else "unknown".
        # Explicit == branches keep mypy's Literal narrowing sound.
        if self._last_health_reason == "network":
            return HealthStatus(available=False, reason="network")
        if self._last_health_reason == "no-key":
            return HealthStatus(available=False, reason="no-key")
        if self._last_health_reason == "auth_invalid":
            return HealthStatus(available=False, reason="auth_invalid")
        return HealthStatus(available=False, reason="unknown")

    async def usage_snapshot(self) -> UsageSnapshot:
        """Report remaining quota for this OpenRouter cell.

        Honest-advisory: report a remote cell with no quota signal yet.
        OpenRouter returns usage in the response body (consumed generically by
        the dispatch layer), not quota/usage headers, so we cannot report a
        measured remaining fraction from headers. We report
        `remaining_fraction=1.0` ("full, eligible") rather than the local
        free-stub's 0.001 — OpenRouter is NOT free and must compete as a normal
        remote for primary selection, not be suppressed. `weekly_exhausted`
        is False unless a 402 (insufficient credits) has been seen, in which
        case we report exhausted + a cooldown until the cached
        `_exhausted_until` timestamp (the credits-expiry motivation is why this
        backend exists, so exhausted-credits is the one signal worth caching).

        When the last health probe failed (OpenRouter unreachable, or key
        unavailable), report a short cooldown so `_routable_backends` excludes
        us — mirrors litellm_gateway's / ollama_cloud's cold-start-vs-outage
        handling.
        """
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
            # We've polled successfully before; treat current unhealthy state
            # as a transient outage. On cold start (never-polled) keep "always
            # routable" so backends_list isn't empty before the first refresh.
            cooldown_until = now + 30.0
        return UsageSnapshot(
            remaining_fraction=1.0,
            cooldown_until_ts=cooldown_until,
            weekly_exhausted=False,
            probed_at_ts=now,
        )

    async def quota_snapshot(self) -> None:
        # No Codex-style quota headers from OpenRouter; honest-advisory state
        # lives in usage_snapshot(). Per-request usage accumulates through
        # CallHandle (the body `usage` block is parsed by _extract_tokens).
        return None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _mark_unhealthy(self) -> None:
        """Flip health to network-down so usage_snapshot reports a cooldown
        immediately, without waiting for the catalog TTL to drive a refresh.
        Shared by the chat transport-error path and passed as the
        `on_transport_error` hook to chat_to_responses_stream. Mirrors
        ollama_cloud._mark_unhealthy."""
        self._healthy = False
        self._last_health_reason = "network"

    async def _ensure_standin(self) -> str:
        """Return a fresh openrouter-scoped stand-in token, minting if
        needed.

        Callosum holds ONLY this short-TTL stand-in — the real OpenRouter key
        never enters this process (credential proxy injects it on the final hop). Mints
        via credential proxy's POST /v1/standin with an EXPLICIT scope (unlike
        credential_proxy, which omits scope and gets the default
        provider-upstream; the openrouter scope routes credential proxy to the
        OpenRouter key in its pass store). Caches with a 60s pre-expiry
        slop; expires_at is an ABSOLUTE Unix epoch timestamp (credential proxy
        returns issued_at + ttl_seconds), compared against wall-clock
        time.time(). Uses the SAME self._client (no second client).

        Failures: network error → _mark_unhealthy + BackendError(transient);
        credential proxy 401 on mint → invalidate + BackendError(transient); credential proxy
        non-200/non-401 → _healthy=False, _last_health_reason="no-key" +
        BackendError(transient); malformed/no-token → "no-key" + transient.
        """
        if self._standin_token is not None and time.time() < (self._standin_expires_at - 60.0):
            return self._standin_token
        try:
            resp = await self._client.post(
                f"{self._custody_url}/v1/standin",
                json={
                    "account": self._custody_account,
                    "scope": OPENROUTER_SCOPE,
                    "ttl_seconds": OPENROUTER_STANDIN_TTL_S,
                },
            )
        except httpx.HTTPError as exc:
            self._mark_unhealthy()
            raise BackendError(
                classification="transient",
                message=f"openrouter stand-in mint transport error: {exc}",
            ) from exc
        if resp.status_code == 401:
            self._invalidate_standin()
            raise BackendError(
                classification="transient",
                message="openrouter stand-in mint rejected by credential proxy (401)",
            )
        if resp.status_code != 200:
            self._healthy = False
            self._last_health_reason = "no-key"
            raise BackendError(
                classification="transient",
                message=f"openrouter stand-in mint failed (credential proxy HTTP {resp.status_code})",
            )
        try:
            standin_body = resp.json()
        except ValueError as exc:
            self._healthy = False
            self._last_health_reason = "no-key"
            raise BackendError(
                classification="transient",
                message=f"openrouter stand-in mint returned non-JSON: {exc}",
            ) from exc
        token = standin_body.get("token") if isinstance(standin_body, dict) else None
        if not isinstance(token, str) or not token:
            self._healthy = False
            self._last_health_reason = "no-key"
            raise BackendError(
                classification="transient",
                message="openrouter stand-in mint returned no token",
            )
        self._standin_token = token
        expires_raw = standin_body.get("expires_at") if isinstance(standin_body, dict) else None
        try:
            # credential proxy returns expires_at as an ABSOLUTE Unix epoch timestamp
            # (UTC) — issued_at + ttl_seconds. Store it directly and compare
            # against wall-clock time.time() (credential proxy mints with
            # int(time.time()), not monotonic). Fall back to now + TTL.
            expires_at = float(expires_raw) if expires_raw else time.time() + float(OPENROUTER_STANDIN_TTL_S)
        except (TypeError, ValueError):
            expires_at = time.time() + float(OPENROUTER_STANDIN_TTL_S)
        self._standin_expires_at = expires_at
        return self._standin_token

    def _invalidate_standin(self) -> None:
        """Clear the cached stand-in so the next call re-mints. Called on an
        credential proxy-level 401 (the stand-in itself was rejected) — distinct from
        an upstream OpenRouter 401, which leaves the stand-in alone."""
        self._standin_token = None
        self._standin_expires_at = 0.0

    async def _proxy_buffered(
        self,
        *,
        method: str,
        url: str,
        app_headers: dict[str, str],
        body: bytes,
        timeout: float = DEFAULT_CALL_TIMEOUT_S,  # noqa: ASYNC109 - forwards to httpx transport, not asyncio.wait_for
    ) -> tuple[int, dict[str, str], bytes]:
        """POST one OpenRouter call through credential proxy's buffered /v1/proxy.

        Envelope: {"url","method","headers":app_headers,"body_b64"} with
        the stand-in as bearer. credential proxy validates the stand-in, reads the real
        key from pass, strips caller auth headers, injects the real key on
        the final hop, and returns {"status_code","headers","body_b64"}
        wrapping the UPSTREAM response. The CALLER classifies the returned
        upstream status (so an upstream 401 → auth_invalid without
        re-minting the stand-in, which is innocent).

        Re-mints the stand-in ONCE on an credential proxy real-HTTP-401 (the stand-in
        was rejected/expired) and retries. A second credential proxy 401 →
        _mark_unhealthy (reason network) + BackendError(transient)
        — NOT auth_invalid (that is reserved for upstream OpenRouter 401).
        credential proxy 503 → _healthy=False, _last_health_reason="no-key" (the
        openrouter scope is not wired / the pass entry is
        missing/empty). credential proxy other non-200 → _mark_unhealthy + transient.
        """
        envelope = {
            "url": url,
            "method": method,
            "headers": app_headers,
            "body_b64": self._b64(body),
        }
        for attempt in (0, 1):
            token = await self._ensure_standin()
            try:
                response = await self._client.post(
                    f"{self._custody_url}/v1/proxy",
                    json=envelope,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=timeout,
                )
            except httpx.HTTPError as exc:
                # Transport error is not recoverable by re-minting.
                self._mark_unhealthy()
                raise BackendError(
                    classification="transient",
                    message=f"openrouter proxy transport error: {exc}",
                ) from exc
            if response.status_code == 401:
                # credential proxy rejected the stand-in itself. Invalidate, re-mint
                # once (attempt 0 → 1); on the second 401 give up as
                # transient (NOT auth_invalid — that's an upstream signal).
                self._invalidate_standin()
                if attempt == 0:
                    continue
                self._mark_unhealthy()
                raise BackendError(
                    classification="transient",
                    status_code=401,
                    message="openrouter stand-in rejected by credential proxy after re-mint",
                )
            if response.status_code == 503:
                # credential proxy can't serve the scope — not wired / pass entry
                # missing. A provisioning gap, not a network outage.
                self._healthy = False
                self._last_health_reason = "no-key"
                raise BackendError(
                    classification="transient",
                    status_code=503,
                    message="openrouter scope not available at credential proxy (503)",
                )
            if response.status_code != 200:
                # Other credential proxy-level non-200 — a transient outage.
                self._mark_unhealthy()
                raise BackendError(
                    classification="transient",
                    status_code=response.status_code,
                    message=f"openrouter proxy failed (credential proxy HTTP {response.status_code})",
                )
            # credential proxy 200 — unwrap the upstream response. The CALLER
            # classifies the upstream status (upstream 401 → auth_invalid).
            try:
                wrapper = response.json()
            except ValueError as exc:
                self._mark_unhealthy()
                raise BackendError(
                    classification="transient",
                    message=f"openrouter proxy returned non-JSON wrapper: {exc}",
                ) from exc
            if not isinstance(wrapper, dict):
                self._mark_unhealthy()
                raise BackendError(
                    classification="transient",
                    message="openrouter proxy returned a non-object wrapper",
                )
            upstream_status = wrapper.get("status_code", 0)
            if not isinstance(upstream_status, int):
                upstream_status = 0
            raw_headers = wrapper.get("headers") or {}
            upstream_headers = dict(raw_headers) if isinstance(raw_headers, dict) else {}
            upstream_body = self._unb64(wrapper.get("body_b64", "") or "")
            return upstream_status, upstream_headers, upstream_body
        # Unreachable: every loop path either returns or raises. Present so
        # mypy sees the function returns on all control-flow paths.
        raise BackendError(
            classification="transient",
            message="openrouter proxy exhausted re-mint attempts",
        )

    def _openrouter_upstream_status(self, response: httpx.Response) -> int:
        """Disambiguate the credential proxy-level status from the upstream status on a
        streaming /v1/proxy/stream response.

        credential proxy returns its own HTTP 200 on the stream and carries the real
        upstream status in the x-credential proxy-upstream-status header. When the
        header is present → that is the upstream status (an upstream 401
        classifies as auth_invalid). When absent → this is an credential proxy-level
        response (e.g. credential proxy itself returned 401/503); map a non-2xx credential proxy
        status to 502 so it classifies as transient, NOT auth_invalid.
        This keeps the two 401 surfaces distinct: credential proxy-401 → transient,
        upstream-401 → auth_invalid.
        """
        raw = response.headers.get(CUSTODY_UPSTREAM_STATUS_HEADER)
        if raw is not None:
            try:
                return int(raw)
            except ValueError:
                pass
        return response.status_code if response.status_code < 400 else 502

    def _app_request_headers(self, *, stream: bool) -> dict[str, str]:
        """App-level request headers for the OpenRouter hop. NEVER
        Authorization or chatgpt-account-id — credential proxy scrubs those and
        overlays the real key on the final hop. Only content negotiation."""
        return {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
        }

    @staticmethod
    def _b64(data: bytes) -> str:
        return base64.b64encode(data).decode("ascii") if data else ""

    @staticmethod
    def _unb64(data: str) -> bytes:
        return base64.b64decode(data) if data else b""

    def _apply_model_filter(self, slugs: list[str]) -> list[str]:
        """Apply the configured model filter (all / allowlist / prefix) to a
        list of catalog slugs. all is a pass-through; allowlist keeps
        only listed slugs; prefix keeps slugs starting with the prefix."""
        if self._model_filter == "allowlist":
            allow = self._allowlist
            return [s for s in slugs if s in allow]
        if self._model_filter == "prefix":
            prefix = self._model_prefix or ""
            return [s for s in slugs if prefix and s.startswith(prefix)]
        return slugs

    def _apply_family_exclude(self, slugs: list[str]) -> list[str]:
        """Drop any slug whose lowercase form contains a family token the paid
        ollama_cloud backend already serves, so OpenRouter doesn't get paid
        for models the operator already has covered. Best-effort substring
        match against the slug (the namespaces differ from ollama_cloud, so
        exact-model dedup is deferred)."""
        if not self._exclude_families:
            return slugs
        tokens = self._exclude_families
        return [s for s in slugs if not any(tok in s.lower() for tok in tokens)]

    def _derive_regime_provider_deny(self, providers: list[Any]) -> frozenset[str]:
        """Derive the set of OpenRouter inference-provider slugs to deny at
        request time, from the `/providers` payload.

        A provider is denied when its datacenters intersects the
        blocked-countries set OR its headquarters is in the set (HQ
        fallback when datacenters is null/empty — a regime-HQ'd provider
        is suspect even without datacenter metadata; note Alibaba's HQ is
        SG but its datacenters include CN, so the datacenters
        check is what catches it). Providers with null headquarters AND
        null/empty datacenters are NOT denied (no positive evidence they
        are regime; the operator can add them via blocked_providers).

        The country-derived deny set is unioned with explicit
        blocked_providers and then minus allowed_providers (which
        always wins — exempts a regime provider the operator confirms serves
        only from non-regime datacenters).
        """
        blocked = self._blocked_countries
        denied: set[str] = set()
        for entry in providers:
            if not isinstance(entry, dict):
                continue
            slug = entry.get("slug")
            if not isinstance(slug, str) or not slug:
                continue
            hq = entry.get("headquarters")
            dcs = entry.get("datacenters")
            regime = False
            if isinstance(hq, str) and hq in blocked:
                regime = True
            if isinstance(dcs, list):
                for dc in dcs:
                    if isinstance(dc, str) and dc in blocked:
                        regime = True
                        break
            if regime:
                denied.add(slug)
        denied |= set(self._blocked_providers)
        denied -= set(self._allowed_providers)
        return frozenset(denied)

    def _inject_residency_preferences(self, body: dict[str, Any]) -> dict[str, Any]:
        """Inject the blocked-provider deny set as OpenRouter provider.ignore
        on a chat request body. When the deny set is empty, leaves the body
        untouched (no `provider` key). Preserves any caller-supplied
        provider preferences by merging `ignore` into them."""
        if not self._regime_provider_deny:
            return body
        out = dict(body)
        existing = out.get("provider")
        if isinstance(existing, dict):
            merged = dict(existing)
            ignore = set(merged.get("ignore") or [])
            if isinstance(ignore, list):
                ignore = set(ignore)
            ignore |= set(self._regime_provider_deny)
            merged["ignore"] = sorted(ignore)
            out["provider"] = merged
        else:
            out["provider"] = {"ignore": sorted(self._regime_provider_deny)}
        return out

    @contextlib.asynccontextmanager
    async def _proxy_stream_cm(self, out_body: dict[str, Any]) -> AsyncIterator[httpx.Response]:
        """Open a streaming /v1/proxy/stream call to OpenRouter through
        credential proxy, yielding the raw httpx.Response for the generator to parse.

        Mints/refreshes the stand-in BEFORE opening (no in-stream re-mint —
        matches credential_proxy's and ollama_cloud's streaming
        behavior). Builds the proxy envelope with the app-level headers and
        the JSON-serialized chat body. The generator's unchanged SSE state
        machine parses data: lines from credential proxy's verbatim-forwarded
        upstream OpenAI SSE.
        """
        token = await self._ensure_standin()
        envelope = {
            "url": f"{self._base_url}/chat/completions",
            "method": "POST",
            "headers": self._app_request_headers(stream=True),
            "body_b64": self._b64(json.dumps(out_body).encode("utf-8")),
        }
        async with self._client.stream(
            "POST",
            f"{self._custody_url}/v1/proxy/stream",
            json=envelope,
            headers={"Authorization": f"Bearer {token}"},
        ) as response:
            yield response

    async def chat_completions(self, body: dict[str, Any], handle: CallHandle | None = None) -> dict[str, Any]:
        """Non-stream POST to OpenRouter's /v1/chat/completions endpoint,
        reached through credential proxy's buffered /v1/proxy with a short-TTL
        openrouter stand-in (the real key never enters this process).

        Body-prep strips Codex-only fields (reasoning,
        parallel_tool_calls) the aggregator rejects, then injects the
        blocked-provider residency deny set as provider.ignore.
        """
        await self._refresh_catalog_if_stale()
        prepped = _strip_codex_only_fields({**body, "stream": False})
        out_body = self._inject_residency_preferences(prepped)
        upstream_status, upstream_headers, upstream_body = await self._proxy_buffered(
            method="POST",
            url=f"{self._base_url}/chat/completions",
            app_headers=self._app_request_headers(stream=False),
            body=json.dumps(out_body).encode("utf-8"),
        )
        if handle is not None:
            handle.upstream_status = upstream_status
            handle.upstream_headers = upstream_headers
        if upstream_status >= 400:
            # 402 (insufficient credits) — the credits-expiry motivation is
            # why this backend exists, so cache an exhausted cooldown and
            # classify as rate_limited so dispatch rotates to another cell.
            if upstream_status == 402:
                self._exhausted_until = time.time() + OPENROUTER_EXHAUSTED_COOLDOWN_S
                raise BackendError(
                    classification="rate_limited",
                    status_code=402,
                    message="openrouter insufficient credits (402); backend cooldown set",
                )
            # Upstream non-2xx (401/403 ⇒ key rejected by OpenRouter →
            # auth_invalid; the stand-in is innocent — do NOT re-mint). The
            # CALLER classifies via the real upstream status; build a
            # synthetic response so error_from_response reads the body/headers.
            synthetic = httpx.Response(upstream_status, headers=upstream_headers, content=upstream_body)
            raise error_from_response(synthetic, status_code=upstream_status)
        return cast(dict[str, Any], json.loads(upstream_body))

    async def chat_completions_stream(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> AsyncIterator[bytes]:
        """Raw byte passthrough of the chat-completions SSE stream, reached
        through credential proxy's streaming /v1/proxy/stream with a short-TTL
        openrouter stand-in (the real key never enters this process).

        Injects the blocked-provider residency deny set as provider.ignore
        on the outbound body. Chat-native path (rarely used: Codex traffic
        flows through /v1/responses → responses_stream).
        """
        await self._refresh_catalog_if_stale()
        prepped = _strip_codex_only_fields({**body, "stream": True})
        out_body = self._inject_residency_preferences(prepped)
        token = await self._ensure_standin()
        envelope = {
            "url": f"{self._base_url}/chat/completions",
            "method": "POST",
            "headers": self._app_request_headers(stream=True),
            "body_b64": self._b64(json.dumps(out_body).encode("utf-8")),
        }
        try:
            stream_ctx = self._client.stream(
                "POST",
                f"{self._custody_url}/v1/proxy/stream",
                json=envelope,
                headers={"Authorization": f"Bearer {token}"},
            )
            async with stream_ctx as response:
                upstream_status = self._openrouter_upstream_status(response)
                if handle is not None:
                    handle.upstream_status = upstream_status
                    handle.upstream_headers = dict(response.headers)
                # credential proxy-401 → header absent → 502 → transient (NOT
                # auth_invalid); upstream-401 → header 401 → auth_invalid.
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
        """Responses-API surface, translated to/from chat-completions.

        OpenRouter has no native Responses endpoint, so translate the
        Responses-shaped request to chat, call the chat-completions endpoint
        and translate the chat reply
        back to Responses shape. The shared translators in
        `callosum.backends._responses_chat` are the same ones litellm_gateway
        and ollama_cloud use.
        """
        chat_body = _responses_to_chat_request(body)
        chat_response = await self.chat_completions(chat_body, handle)
        return _chat_to_responses_response(chat_response)

    async def responses_stream(self, body: dict[str, Any], handle: CallHandle | None = None) -> AsyncIterator[bytes]:
        """Responses-API SSE stream, translated from chat-completions deltas.

        Codex CLI traffic flows through here (/v1/responses, stream:true).
        The chat→Responses streaming generator lives in the shared
        callosum.backends._responses_chat module (one copy for every
        chat-shaped backend); we inject this backend's coupling points via
        the keyword-only hooks. Under proxy custody the generator opens the
        upstream through this backend's _proxy_stream_cm (which mints the
        stand-in, builds the /v1/proxy/stream envelope, and yields the raw
        credential proxy stream response) instead of a direct client.stream POST,
        and reads the real upstream status via _openrouter_upstream_status
        (the x-credential proxy-upstream-status header) instead of the credential proxy HTTP
        status. The collector tees the generator's output so
        handle.stream_summary carries the response.completed usage block →
        per-request token accounting + the usage log populate automatically.

        **Two 401 surfaces under proxy custody:** a mid-stream upstream 401
        from OpenRouter surfaces via x-credential proxy-upstream-status:401 →
        auth_invalid via error_from_response(status_code=401) (no
        stand-in re-mint — the stand-in is innocent). An credential proxy-level 401
        (stand-in rejected) maps to 502 → transient.
        """

        def open_chat_stream(out_body: dict[str, Any], _headers: dict[str, str]) -> AsyncContextManager[httpx.Response]:
            # `_headers` (the generator's default auth headers) is ignored —
            # credential proxy builds the final-hop headers from the envelope + the
            # real key. The stand-in is minted inside `_proxy_stream_cm`.
            # Inject the residency deny set on the outbound body before the
            # generator's prep_body runs (the generator already strips
            # codex-only fields + sets stream=True; residency is orthogonal).
            return self._proxy_stream_cm(self._inject_residency_preferences(out_body))

        collector = ResponsesStreamCollector(
            chat_to_responses_stream(
                client=self._client,
                chat_url=f"{self._base_url}/chat/completions",
                body=body,
                handle=handle,
                prep_body=lambda b: _strip_codex_only_fields({**b, "stream": True}),
                headers={},  # unused on the proxy path; open_chat_stream ignores it
                first_item_timeout_s=LOCAL_STREAM_FIRST_BYTE_TIMEOUT_S,
                idle_timeout_s=LOCAL_STREAM_IDLE_TIMEOUT_S,
                what_label="openrouter",
                on_success=lambda: None,
                on_transport_error=self._mark_unhealthy,
                open_chat_stream=open_chat_stream,
                upstream_status_of=self._openrouter_upstream_status,
            )
        )
        try:
            async for chunk in collector.iter_through():
                yield chunk
        except BackendError as exc:
            # The shared generator classifies a 402 via `error_from_response`
            # as `transient` (classify_http_status falls 402 through). On the
            # chat paths we intercept 402 directly; here the generator raises
            # first, so catch and convert: set the exhausted cooldown and
            # re-raise as `rate_limited` so dispatch rotates to another cell.
            # Keeps the 402→exhausted+cooldown behavior consistent across all
            # four dispatch paths (the credits-expiry motivation is why this
            # backend exists, so exhausted-credits is the one signal worth
            # caching — even on the main Codex streaming route).
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
        """Return conservative capabilities for `model`. OpenRouter's
        `/models` endpoint exposes per-model modality/context metadata that
        could enrich this, but v1 returns safe text-only/no-tools defaults
        with the discovered context_window when available; the request can
        still dispatch. Cost rank is always 10 (remote default); the
        CostRankProvider overlay in app.py may replace it."""
        return CellCapabilities(
            context_window=self._context_windows.get(model, 128_000),
            modalities=frozenset({"text"}),
            supports_tools=False,
            cost_rank=10,
        )

    async def refresh_advertised_models(self, *, now: float | None = None) -> None:
        """Public refresh entry point — mirrors the other backends' contract
        so the lifespan startup loop can refresh all backends uniformly.

        Forces a catalog re-fetch unconditionally (no TTL gate) so callers can
        rely on advertised_models being current immediately after this
        returns.
        """
        # Clear the fetched-at timestamp so _refresh_catalog_if_stale doesn't
        # short-circuit on a stale-but-young entry.
        self._catalog_fetched_at = 0.0
        await self._refresh_catalog_if_stale(now=now)

    # ---------- internals --------------------------------------------------

    async def _refresh_catalog_if_stale(self, *, now: float | None = None) -> None:
        ts = now if now is not None else time.time()
        if self._catalog and ts - self._catalog_fetched_at < self._catalog_refresh_s:
            return
        # Catalog refresh runs through credential proxy's buffered proxy (the real
        # OpenRouter key never enters this process). Fetch /models and
        # /providers in two proxy hops. This method is called from health(),
        # which MUST NOT raise — map every proxy/upstream failure to a health
        # reason and return. The proxy helper already set `_healthy=False` +
        # a concrete reason (network / no-key) before raising, so a bare
        # `return` suffices.
        models_status, _mh, models_body = await self._proxy_get(
            f"{self._base_url}/models",
            timeout=DEFAULT_HEALTH_TIMEOUT_S,
        )
        # `None` => the proxy itself failed (network / no-key). `_proxy_buffered`
        # already set `_healthy=False` + a concrete reason; preserve it (mirrors
        # ollama_cloud returning on BackendError) instead of clobbering with the
        # opaque "unknown" — a network outage must surface as "network", a
        # missing scope/key as "no-key", NOT "unknown".
        if models_status is None:
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

        # Derive the blocked-provider deny set from /providers (best-effort:
        # a /providers failure does NOT fail catalog refresh — residency
        # enforcement simply stays at the previous/empty deny set until the
        # next refresh succeeds).
        providers: list[Any] = []
        prov_status, _ph, prov_body = await self._proxy_get(
            f"{self._base_url}/providers",
            timeout=DEFAULT_HEALTH_TIMEOUT_S,
        )
        if isinstance(prov_status, int) and prov_status == 200:
            try:
                prov_payload = json.loads(prov_body)
            except (ValueError, TypeError):
                prov_payload = None
            prov_data = prov_payload.get("data") if isinstance(prov_payload, dict) else None
            if isinstance(prov_data, list):
                providers = prov_data
        self._regime_provider_deny = self._derive_regime_provider_deny(providers)

        # Parse model slugs + context_length. OpenRouter's /models shape:
        # {"data":[{"id":"openai/model-a0f5","context_length":128000,...}]}.
        slugs: list[str] = []
        context_windows: dict[str, int] = {}
        for entry in models:
            if not isinstance(entry, dict):
                continue
            slug = entry.get("id")
            if not isinstance(slug, str) or not slug:
                continue
            slugs.append(slug)
            ctx = entry.get("context_length")
            if isinstance(ctx, int) and ctx > 0:
                context_windows[slug] = ctx
        # Apply the model filter (all/allowlist/prefix), then the
        # ollama-overlap family-exclude. Residency is NOT a catalog filter —
        # it's enforced per-request via provider.ignore, so a model served by
        # both regime and non-regime providers stays advertised and routable.
        slugs = self._apply_model_filter(slugs)
        slugs = self._apply_family_exclude(slugs)
        self._catalog = tuple(slugs)
        self._context_windows = context_windows
        self._catalog_fetched_at = ts
        self._healthy = True
        self._last_health_reason = "ok"

    async def _proxy_get(
        self,
        url: str,
        *,
        timeout: float,  # noqa: ASYNC109 - forwards to httpx transport, not asyncio.wait_for
    ) -> tuple[int | None, dict[str, str], bytes]:
        """GET through credential proxy's buffered proxy for catalog refresh, mapping
        proxy failures to a sentinel `(None, {}, b"")` so the caller (which
        runs from health() and MUST NOT raise) can classify by status. A
        `None` status means the proxy itself failed (network/no-key) and the
        proxy helper already set `_healthy=False` + a reason."""
        try:
            upstream_status, upstream_headers, upstream_body = await self._proxy_buffered(
                method="GET",
                url=url,
                app_headers=self._app_request_headers(stream=False),
                body=b"",
                timeout=timeout,
            )
        except BackendError:
            return None, {}, b""
        return upstream_status, upstream_headers, upstream_body