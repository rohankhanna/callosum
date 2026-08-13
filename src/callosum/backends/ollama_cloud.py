"""Ollama Cloud backend — routes to cloud models served by ollama.com through
credential proxy's boundary-native proxy custody (the real ollama.com API key NEVER
enters callosum's process).

Callosum mints a short-TTL `ollama-cloud`-scoped stand-in token via credential proxy's
existing `POST /v1/standin`, then POSTs every ollama.com call (chat, catalog,
capabilities) through credential proxy's existing `POST /v1/proxy` (buffered) or
`POST /v1/proxy/stream` (streaming) with the stand-in as
`Authorization: Bearer <stand-in>`. credential proxy validates the token, reads the
real ollama.com API key from its `pass` store, strips the caller `authorization`
/ `chatgpt-account-id`, and injects `Authorization: Bearer <real key>` on the
final hop. The real key therefore NEVER crosses into callosum's memory — this
backend holds only a short-TTL stand-in token (revocable, auto-expiring,
loopback-minted), which unifies it with `credential_proxy` (callosum already
routes the OpenAI lane through credential proxy's `/v1/proxy` with stand-in tokens).

This retires both prior architectures: (1) the original indirect hop through
the local ollama daemon holding cloud auth under `ollama signin` (a
browser-session cookie), and (2) the unmerged direct+memory-only-key path
where callosum fetched the real key from credential proxy and held it in memory. Under
proxy custody callosum holds NO provider credential at all — real key OR
memory-only key — only the stand-in. credential proxy KEEPS its browser-cookie custody
for menubar metering only (decoupled from this chat path via
`OllamaCloudUsageSource`). The daemon `ollama signin` chat path is RETIRED.

Why a distinct `BackendKind` (not a model-name predicate):
  "Local" is classified across the codebase by `backend.kind ==
  "litellm_gateway"` (~13 sites). A cloud model would otherwise be
  mis-routed as a FREE local cell — but cloud requests burn real Ollama
  Cloud quota. The new `BackendKind = "ollama_cloud"` makes every
  `== "litellm_gateway"` (local-affirmative) site NOT match and every
  `!= "litellm_gateway"` (remote-affirmative) site match, so cloud cells
  land in remote lanes and stay out of local-only/probing without further
  edits. See work tracker .

Metering is **honest-advisory**, not header-parsed: Ollama Cloud exposes no
quota/usage response headers (ollama/ollama #15663, wontfix), so this backend
cannot read remaining-quota the way the Codex backends read `x-codex-*`
headers. `usage_snapshot()` reports "remote, full, eligible, no signal yet" and
`quota_snapshot()` returns None — both are advisory state, SEPARATE from
per-request accounting. Per-request usage accumulates automatically through
the dispatch layer's `CallHandle`: the OpenAI-compatible `/v1/chat/completions`
endpoint returns `usage` in already-parsed shape (`prompt_tokens` /
`completion_tokens` / `total_tokens`), which `_extract_tokens` accepts
alongside the Responses-API shape. No separate metering wiring is needed.
Advisory usage metering via credential proxy's `/v1/ollama/usage` loopback
(`OllamaCloudUsageSource`) is DECOUPLED from the chat path — it still reads
credential proxy's browser-cookie-backed meter, independent of the proxy path.

This backend is OPTIONAL and **env-gated OFF by default**: it is not
constructed unless `CALLOSUM_OLLAMA_CLOUD_ENABLED=1`, so enabling it is a
no-op for live routing until the operator turns it on. The credential proxy proxy path
is MANDATORY (not separately gated) when the backend is enabled — if credential proxy
is down, the `ollama-cloud` scope is not configured, or the `pass` key entry is
missing/empty, the backend goes unhealthy with a clear `"no-key"` reason (the
correct failure mode). The four chat methods hit ollama.com's OpenAI-compatible
`/v1/chat/completions` endpoint (same path litellm_gateway uses, so the shared
Responses↔Chat translators in `callosum.backends._responses_chat` apply
verbatim). The live-routing flip is operator-gated.

**Probe-gated assumption:** docs.ollama.com/cloud documents only the
Ollama-native `/api/chat`; the local daemon additionally serves the
OpenAI-compatible `/v1/chat/completions`. This backend assumes ollama.com
serves the OpenAI-compatible path (Probe 2 confirms before merge). If
ollama.com lacks it, a new Responses↔Ollama-native translator is needed —
deferred to a follow-up slice (see  log 2026-08-07). credential proxy's proxy
contract is shape-agnostic (forwards bytes verbatim, stripping upstream
content-type on streams), so it is unaffected either way.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import math
import time
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager as AsyncContextManager
from dataclasses import dataclass
from datetime import datetime
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

# Direct upstream target: ollama.com (NOT the local daemon). Callosum never
# presents the real ollama.com API key itself — it POSTs each call through
# credential proxy's `/v1/proxy`(`/stream`) with a short-TTL `ollama-cloud`-scoped
# stand-in token, and credential proxy injects the real key on the final hop. See
# `OllamaCloudBackend._proxy_buffered` / `responses_stream` below. This retires
# the prior architecture where callosum proxied through the local ollama daemon
# (127.0.0.1:11434) that held cloud auth under `ollama signin`.
DEFAULT_OLLAMA_URL = "https://ollama.com"
# Cloud catalog name suffix filter. The local daemon tagged cloud-pulled models
# with a `:cloud` suffix (e.g. `model-a0d2:cloud`); against ollama.com directly the
# `/api/tags` catalog may list bare names. Probe 1 ( log 2026-08-07)
# confirms the shape. Empty default accepts ALL catalog names (correct for a
# direct all-cloud host); the operator overrides with
# CALLOSUM_OLLAMA_CLOUD_MODEL_SUFFIX=:cloud if the suffix is still present.
DEFAULT_MODEL_SUFFIX = ""
DEFAULT_CATALOG_REFRESH_S = 60.0
# Remote TLS+WAN round-trip latency for /api/tags + /api/show is higher than the
# local daemon's 2s budget; raise so catalog/capability refresh doesn't flap.
DEFAULT_HEALTH_TIMEOUT_S = 5.0
DEFAULT_CALL_TIMEOUT_S = 300.0
# Priority offset for cloud cells in the merged cell grid. Remote Codex
# priorities are small ints (tens); local cells offset by +10_000 (see
# litellm_gateway.LOCAL_PRIORITY_OFFSET) to sort after Codex. Cloud sits in
# a remote band above curated Codex so the recommender presents curated Codex
# first, cloud as a secondary remote, then free local last. This same number
# feeds `_remote_catalog_priorities` as the cold-start cost_rank prior: cloud
# models (never measured — no quota headers) land at `prior_floor + offset`
# inside CostRankProvider, i.e. cost_rank >= base_rank (10), so they are
# conservatively treated as expensive remote cells until the operator overrides
# via `cost_rank_overrides`. See routing/cost_model.py.
CLOUD_PRIORITY_OFFSET = 1_000

# ---------------------------------------------------------------------
# credential proxy loopback sources (loopback-only, lazy+cached).
#
# Two independent concerns, both served by the credential proxy applet on 127.0.0.1:7342:
#
# 1. Chat-auth via the credential proxy proxy (boundary-native custody). The REAL
#    ollama.com API key NEVER enters callosum. Callosum mints a short-TTL
#    stand-in token for the `ollama-cloud` scope at `POST /v1/standin`, then
#    POSTs each ollama.com call through credential proxy's `POST /v1/proxy` (buffered)
#    or `POST /v1/proxy/stream` (streaming) with the stand-in as bearer.
#    credential proxy validates the token, reads the real key from its `pass` store,
#    and injects `Authorization: Bearer <real key>` on the final hop. If
#    credential proxy is down, the `ollama-cloud` scope is not wired, or the `pass`
#    entry is missing/empty, the backend goes unhealthy with reason `no-key`.
#    See `OllamaCloudBackend._ensure_standin` / `_proxy_buffered` below.
#
# 2. `OllamaCloudUsageSource` (advisory metering, DECOUPLED from chat) — reads
#    `/v1/ollama/usage` using a short-lived stand-in token minted for the
#    `ollama-usage` scope (TTL clamped to credential proxy's MAX_STANDIN_TTL_SECONDS =
#    1800s). Backed by credential proxy's browser-cookie custody (menubar needs the
#    cookie). All failures are swallowed → `None`; usage is advisory only and
#    must never fail routing. Independent of the proxy chat path.
# ---------------------------------------------------------------------
DEFAULT_CUSTODY_URL = "http://127.0.0.1:7342"
DEFAULT_OLLAMA_USAGE_ACCOUNT = "primary"
DEFAULT_STANDIN_TTL_S = 1800
MAX_STANDIN_TTL_S = 1800  # credential proxy's MAX_STANDIN_TTL_SECONDS; clamp requested TTL down to this
OLLAMA_USAGE_SCOPE = "ollama-usage"
DEFAULT_USAGE_CACHE_TTL_S = 30.0
DEFAULT_USAGE_TIMEOUT_S = 5.0
# Stand-in scope + TTL for the ollama-cloud chat-auth path (proxy custody).
# CRITICAL: the scope MUST be sent explicitly — credential_proxy omits `scope`
# (credential proxy defaults it to `provider-upstream`), but the ollama-cloud path needs
# the dedicated `ollama-cloud` scope so credential proxy routes to the right `pass` key.
# 300s is the credential proxy default; short on purpose (re-mint is a cheap loopback
# POST) and clamped under MAX_STANDIN_TTL_S on the credential proxy side regardless.
OLLAMA_CLOUD_SCOPE = "ollama-cloud"
OLLAMA_CLOUD_STANDIN_TTL_S = 300
# credential proxy surfaces the real upstream status on its streaming proxy via this
# response header (the proxy itself returns HTTP 200 whenever the hop
# succeeded). Mirrors `credential_proxy.CUSTODY_UPSTREAM_STATUS_HEADER`.
CUSTODY_UPSTREAM_STATUS_HEADER = "x-credential proxy-upstream-status"


@dataclass(frozen=True, slots=True)
class UsageMeter:
    percent_used: float
    resets_at_ts: float | None


@dataclass(frozen=True, slots=True)
class OllamaUsage:
    plan: str | None
    session: UsageMeter
    weekly: UsageMeter
    fetched_at_ts: float


def _parse_meter(meter: Any) -> UsageMeter:
    """Parse an credential proxy meter dict into a `UsageMeter`.

    credential proxy emits `{"percent": float, "resets_at": "ISO8601 UTC str"}` for a
    populated meter, or `{}` (empty) when the upstream value is unparseable.
    `resets_at` is an ISO8601 UTC string like "2026-06-30T19:00:00Z"; convert
    to epoch via `datetime.fromisoformat(s.replace("Z","+00:00")).timestamp()`
    guarded by try/except → None on failure.

    An empty/missing meter yields `percent_used=NaN` so the live projection's
    finiteness check (`math.isfinite`) skips it cleanly.
    """
    if not isinstance(meter, dict) or not meter:
        return UsageMeter(percent_used=float("nan"), resets_at_ts=None)
    percent = meter.get("percent")
    if not isinstance(percent, (int, float)):
        percent = float("nan")
    resets_raw = meter.get("resets_at")
    resets_ts: float | None = None
    if isinstance(resets_raw, str):
        try:
            resets_ts = datetime.fromisoformat(resets_raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            resets_ts = None
    return UsageMeter(percent_used=float(percent), resets_at_ts=resets_ts)


class OllamaCloudUsageSource:
    """Credential-free, loopback-only, lazy+cached reader of credential proxy's
    `/v1/ollama/usage` endpoint.

    Callosum holds NO credential proxy credential; the credential proxy menubar applet holds
    the real Ollama-cookie custody. This source mints a short-lived stand-in
    token (scope `ollama-usage`, TTL clamped to `MAX_STANDIN_TTL_S`) against
    credential proxy's loopback `/v1/standin` endpoint, then uses that token to GET
    `/v1/ollama/usage`. All failures are swallowed and returned as `None`;
    this source is advisory-only and must never fail routing.

    The HTTP client is built the same way as `OllamaCloudBackend`'s: owned
    when neither `client` nor `transport` is supplied; constructed with the
    given `transport` when `client` is None; borrowed (not owned) when
    `client` is supplied.
    """

    def __init__(
        self,
        *,
        custody_url: str = DEFAULT_CUSTODY_URL,
        account: str = DEFAULT_OLLAMA_USAGE_ACCOUNT,
        standin_ttl_s: int = DEFAULT_STANDIN_TTL_S,
        cache_ttl_s: float = DEFAULT_USAGE_CACHE_TTL_S,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = DEFAULT_USAGE_TIMEOUT_S,
    ) -> None:
        self._custody_url = custody_url.rstrip("/")
        self._account = account
        # Clamp the requested stand-in TTL down to credential proxy's hard ceiling so
        # a misconfigured request doesn't ask credential proxy for more than it will
        # grant.
        self._standin_ttl_s = min(standin_ttl_s, MAX_STANDIN_TTL_S)
        self._cache_ttl_s = cache_ttl_s
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(transport=transport, timeout=timeout_s)
            self._owns_client = True
        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._cached: OllamaUsage | None = None
        self._cached_at: float = 0.0

    async def fetch(self) -> OllamaUsage | None:
        """Return a cached `OllamaUsage` if fresh, else fetch from credential proxy.

        Never raises. On any failure (network, non-200, parse error, token
        mint failure) returns `None`. On a 401 from the usage endpoint the
        stand-in token is invalidated so the next call re-mints, but a
        transient upstream failure (502/503/network) does NOT clear the
        token.
        """
        if self._cached is not None and (time.monotonic() - self._cached_at) < self._cache_ttl_s:
            return self._cached

        # Ensure a fresh stand-in token (60s safety margin before expiry).
        if self._token is None or time.monotonic() >= (self._token_expires_at - 60.0):
            try:
                resp = await self._client.post(
                    f"{self._custody_url}/v1/standin",
                    json={
                        "account": self._account,
                        "scope": OLLAMA_USAGE_SCOPE,
                        "ttl_seconds": self._standin_ttl_s,
                    },
                )
            except httpx.HTTPError:
                return None
            if resp.status_code != 200:
                return None
            try:
                standin_body = resp.json()
            except ValueError:
                return None
            token = standin_body.get("token") if isinstance(standin_body, dict) else None
            if not isinstance(token, str) or not token:
                return None
            self._token = token
            expires_raw = standin_body.get("expires_at") if isinstance(standin_body, dict) else None
            try:
                # credential proxy returns expires_at as an epoch/ts number (relative
                # seconds until expiry). Fall back to monotonic + standin TTL
                # if missing/unparseable. The `or` short-circuit handles a
                # falsy/zero/missing value just like the spec formula.
                expires_at = float(expires_raw) if expires_raw else float(self._standin_ttl_s)
            except (TypeError, ValueError):
                expires_at = float(self._standin_ttl_s)
            self._token_expires_at = time.monotonic() + expires_at

        # Fetch the usage payload with the stand-in token.
        try:
            resp = await self._client.get(
                f"{self._custody_url}/v1/ollama/usage",
                headers={"Authorization": f"Bearer {self._token}"},
            )
        except httpx.HTTPError:
            return None

        if resp.status_code == 401:
            # Token rejected/expired — invalidate so next fetch re-mints.
            self._token = None
            return None
        if resp.status_code != 200:
            # Transient upstream (502/503) — do NOT punish the token.
            return None
        try:
            payload = resp.json()
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None

        plan = payload.get("plan")
        if not isinstance(plan, str):
            plan = None
        session = _parse_meter(payload.get("session"))
        weekly = _parse_meter(payload.get("weekly"))
        usage = OllamaUsage(
            plan=plan,
            session=session,
            weekly=weekly,
            fetched_at_ts=time.time(),
        )
        self._cached = usage
        self._cached_at = time.monotonic()
        return usage

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class OllamaCloudBackend:
    """Cloud models served by ollama.com, reached through credential proxy's proxy.

    Boundary-native credential custody: the REAL ollama.com API key NEVER
    enters this process. Callosum mints a short-TTL ollama-cloud-scoped
    stand-in token via credential proxy's POST /v1/standin (loopback), then POSTs
    each ollama.com request through credential proxy's POST /v1/proxy (buffered) or
    POST /v1/proxy/stream (streaming) with the stand-in as bearer. credential proxy
    validates the token, reads the real key from its pass store, strips
    caller Authorization/chatgpt-account-id, and injects
    Authorization: Bearer <real key> on the final hop to ollama.com. This
    unifies the custody model with the OpenAI lane, which already routes
    through the same /v1/proxy surface via credential_proxy — so
    ollama_cloud stops being a lone outlier holding a real provider key.

    Two prior architectures are RETIRED by this: (1) the local ollama daemon's
    ollama signin browser-session chat-auth (the daemon hop is gone; credential proxy
    injects the key), and (2) an unmerged direct-to-ollama.com variant where
    callosum fetched the real key over loopback and held it in memory — under
    proxy custody callosum holds only a short-TTL stand-in, never the real key.

    The backend keeps its own BackendKind="ollama_cloud" (distinct from
    litellm_gateway) so dispatch routes it as a REMOTE fleet (not local),
    and so the catalog/capabilities/usage hooks stay scoped to ollama.com's
    shape. Honest-advisory metering + the optional live OllamaCloudUsageSource
    are unchanged and DECOUPLED from the proxy chat path. The backend is
    env-gated OFF by default (CALLOSUM_OLLAMA_CLOUD_ENABLED=1 to enable).

    The no-key health reason, under proxy custody, means the stand-in mint
    failed (credential proxy unreachable on /v1/standin or returned non-200/non-401)
    OR credential proxy's /v1/proxy returned 503 (the ollama-cloud scope is not
    wired at credential proxy, or the pass entry is missing/empty). It does NOT mean
    a key fetch failed (there is no key fetch) nor that ollama.com rejected the
    key (that surfaces as auth_invalid via the wrapper's upstream status).

    Probe-gated: this assumes ollama.com serves OpenAI-compatible
    /v1/chat/completions (reusing the shared _responses_chat
    translators). credential proxy's proxy contract is shape-agnostic (it forwards bytes
    verbatim, stripping upstream content-type on streams), so if ollama.com
    serves only the native /api/chat this slice lands inert and a follow-up
    adds a native translator — the credential proxy contract is unaffected either way.
    """

    kind: BackendKind = "ollama_cloud"

    def __init__(
        self,
        *,
        id: str,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model_suffix: str = DEFAULT_MODEL_SUFFIX,
        catalog_refresh_s: float = DEFAULT_CATALOG_REFRESH_S,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
        custody_url: str = DEFAULT_CUSTODY_URL,
        custody_account: str = DEFAULT_OLLAMA_USAGE_ACCOUNT,
        usage_source: OllamaCloudUsageSource | None = None,
        usage_live: bool = False,
    ) -> None:
        self.id = id
        self._ollama_url = ollama_url.rstrip("/")
        self._model_suffix = model_suffix
        self._catalog_refresh_s = catalog_refresh_s
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(transport=transport, timeout=timeout_s)
            self._owns_client = True
        # Catalog state — populated lazily from /api/tags (through the proxy).
        self._catalog: tuple[str, ...] = ()
        self._catalog_fetched_at: float = 0.0
        # Health is derived from the most recent /api/tags poll. Until the
        # first poll lands we report unavailable so dispatch doesn't route
        # here before discovery completes.
        self._healthy: bool = False
        self._last_health_reason: str = "unknown"
        # Real per-model capabilities, populated lazily from ollama's
        # /api/show endpoint during catalog refresh. cell_capabilities()
        # reads this cache synchronously; when discovery hasn't run yet (or
        # /api/show failed for a model), falls back to conservative defaults.
        self._capabilities_cache: dict[str, CellCapabilities] = {}
        # credential proxy boundary-native proxy custody. Callosum holds ONLY a
        # short-TTL `ollama-cloud`-scoped stand-in token (never the real key).
        # Minted lazily by `_ensure_standin`, cached with a 60s pre-expiry slop,
        # and invalidated on an credential proxy-level 401 (stand-in rejected) so the
        # next call re-mints. `expires_at` is a `time.monotonic()` deadline
        # (credential proxy returns relative seconds; see OllamaCloudUsageSource).
        self._custody_url = custody_url.rstrip("/")
        self._custody_account = custody_account
        self._standin_token: str | None = None
        self._standin_expires_at: float = 0.0
        # Optional credential-free credential proxy usage source + live-projection
        # flag. When `usage_live` is True and `usage_source` is set,
        # `usage_snapshot()` prefers a measured session/weekly percent from
        # credential proxy's loopback `/v1/ollama/usage` over the honest-advisory
        # fallback. Both default off so existing construction is unchanged.
        # DECOUPLED from the proxy chat path (it mints its own `ollama-usage`
        # stand-in and reads the browser-cookie-backed meter).
        self._usage_source = usage_source
        self._usage_live = usage_live

    @property
    def advertised_models(self) -> frozenset[str]:
        return frozenset(self._catalog)

    @property
    def model_metadata(self) -> dict[str, ModelMetadata]:
        """Synthesize ModelMetadata for each cataloged cloud model.

        Cloud models lack Codex-style metadata fields; we populate the cell
        grid by hand so the merge in app.py picks them up without changing the
        cell-grid filter logic:

          * supported_in_api=True, visibility="list" — pass the inclusion
            filter in `live_completion_models_from_metadata`.
          * supported_reasoning_levels=("default",) — one cell per cloud
            model rather than one cell per (model, effort).
          * priority=CLOUD_PRIORITY_OFFSET + idx — sorts in the remote band,
            after curated Codex, before free local.
        """
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
        # Force a catalog refresh if we're past TTL so dispatch sees
        # ollama.com's current state rather than a stale snapshot.
        await self._refresh_catalog_if_stale()
        if self._healthy:
            return HealthStatus(available=True, reason="ok")
        # Surface the concrete failure reason so /status names the cause:
        # "network" (credential proxy/ollama.com unreachable), "no-key" (stand-in mint
        # failed — credential proxy unreachable on /v1/standin or returned non-200/non-401
        # — OR credential proxy /v1/proxy returned 503 because the `ollama-cloud` scope is
        # not wired / the pass entry is missing/empty), else "unknown" (non-200
        # catalog response / cold start). Upstream ollama.com 401 surfaces as
        # "auth_invalid", NOT "no-key". Explicit == branches keep mypy's Literal
        # narrowing sound (an `in`-tuple check does not narrow `str` to the
        # HealthReason Literal).
        if self._last_health_reason == "network":
            return HealthStatus(available=False, reason="network")
        if self._last_health_reason == "no-key":
            return HealthStatus(available=False, reason="no-key")
        if self._last_health_reason == "auth_invalid":
            return HealthStatus(available=False, reason="auth_invalid")
        return HealthStatus(available=False, reason="unknown")

    @property
    def usage_source(self) -> OllamaCloudUsageSource | None:
        return self._usage_source

    async def usage_snapshot(self) -> UsageSnapshot:
        """Report remaining quota for this cloud cell.

        When `usage_live` is True and a `usage_source` is wired, prefer a
        measured session/weekly percent from credential proxy's loopback
        `/v1/ollama/usage` endpoint. The credential proxy source is credential-free
        (callosum holds no Ollama-cookie); it mints a short-lived stand-in
        token for the `ollama-usage` scope and reads the usage meter. All
        failures fall through to the honest-advisory block below.

        Honest-advisory fallback: report a remote cell with no quota signal
        yet. Cloud requests burn real Ollama Cloud quota, but ollama.com
        exposes no quota/usage headers (ollama/ollama #15663), so we cannot
        report a measured remaining fraction. We report
        `remaining_fraction=1.0` ("full, eligible") rather than the local
        free-stub's 0.001 — cloud is NOT free and must compete as a normal
        remote for primary selection, not be suppressed.
        `weekly_exhausted=False` because we genuinely don't know exhaustion
        from headers; the dispatch layer's failure-attribution + this
        backend's health handle outages separately.

        When the last health probe failed (ollama.com unreachable, or key
        unavailable), report a short cooldown so `_routable_backends` excludes
        us — mirrors litellm_gateway's cold-start-vs-outage handling.
        """
        now = time.time()

        # Live projection — preferred when wired and enabled.
        if self._usage_live and self._usage_source is not None:
            try:
                payload = await self._usage_source.fetch()
            except Exception:
                logger.exception("ollama-cloud usage source fetch failed")
                payload = None
            if payload is not None and math.isfinite(payload.session.percent_used):
                session_used = payload.session.percent_used
                weekly_used = payload.weekly.percent_used if math.isfinite(payload.weekly.percent_used) else 0.0
                return UsageSnapshot(
                    remaining_fraction=max(0.0, (100.0 - session_used) / 100.0),
                    cooldown_until_ts=None,
                    weekly_exhausted=weekly_used >= 100.0,
                    probed_at_ts=now,
                )
            # else fall through to honest-advisory

        cooldown_until: float | None = None
        if not self._healthy and self._catalog_fetched_at > 0:
            # We've polled successfully before; treat current unhealthy state
            # as a transient outage. On cold start (never-polled) keep
            # "always routable" so backends_list isn't empty before the first
            # refresh runs.
            cooldown_until = now + 30.0
        return UsageSnapshot(
            remaining_fraction=1.0,
            cooldown_until_ts=cooldown_until,
            weekly_exhausted=False,
            probed_at_ts=now,
        )

    async def quota_snapshot(self) -> None:
        # No Codex-style quota headers from ollama; honest-advisory state lives
        # in usage_snapshot() (remaining_fraction=1.0, no cooldown when
        # healthy). Per-request usage accumulates through CallHandle.
        return None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
        if self._usage_source is not None:
            await self._usage_source.aclose()

    def _mark_unhealthy(self) -> None:
        """Flip health to network-down so usage_snapshot reports a cooldown
        immediately, without waiting for the catalog TTL (60s) to drive a
        refresh. Shared by the chat_completions transport-error path and
        passed as the `on_transport_error` hook to chat_to_responses_stream.
        Mirrors litellm_gateway._mark_unhealthy."""
        self._healthy = False
        self._last_health_reason = "network"

    async def _ensure_standin(self) -> str:
        """Return a fresh ollama-cloud-scoped stand-in token, minting if
        needed.

        Callosum holds ONLY this short-TTL stand-in — the real ollama.com key
        never enters this process (credential proxy injects it on the final hop). Mints
        via credential proxy's POST /v1/standin with an EXPLICIT scope (unlike
        credential_proxy, which omits scope and gets the default
        provider-upstream; the ollama-cloud scope routes credential proxy to the
        ollama.com key in its pass store). Caches with a 60s pre-expiry
        slop; expires_at is treated as relative seconds added to
        time.monotonic() (mirroring OllamaCloudUsageSource). Uses the
        SAME self._client (no second client).

        Failures: network error → _mark_unhealthy + BackendError(transient);
        credential proxy 401 on mint → invalidate + BackendError(transient); credential proxy
        non-200/non-401 → _healthy=False, _last_health_reason="no-key" +
        BackendError(transient); malformed/no-token → "no-key" + transient.
        """
        if self._standin_token is not None and time.monotonic() < (self._standin_expires_at - 60.0):
            return self._standin_token
        try:
            resp = await self._client.post(
                f"{self._custody_url}/v1/standin",
                json={
                    "account": self._custody_account,
                    "scope": OLLAMA_CLOUD_SCOPE,
                    "ttl_seconds": OLLAMA_CLOUD_STANDIN_TTL_S,
                },
            )
        except httpx.HTTPError as exc:
            self._mark_unhealthy()
            raise BackendError(
                classification="transient",
                message=f"ollama-cloud stand-in mint transport error: {exc}",
            ) from exc
        if resp.status_code == 401:
            self._invalidate_standin()
            raise BackendError(
                classification="transient",
                message="ollama-cloud stand-in mint rejected by credential proxy (401)",
            )
        if resp.status_code != 200:
            self._healthy = False
            self._last_health_reason = "no-key"
            raise BackendError(
                classification="transient",
                message=f"ollama-cloud stand-in mint failed (credential proxy HTTP {resp.status_code})",
            )
        try:
            standin_body = resp.json()
        except ValueError as exc:
            self._healthy = False
            self._last_health_reason = "no-key"
            raise BackendError(
                classification="transient",
                message=f"ollama-cloud stand-in mint returned non-JSON: {exc}",
            ) from exc
        token = standin_body.get("token") if isinstance(standin_body, dict) else None
        if not isinstance(token, str) or not token:
            self._healthy = False
            self._last_health_reason = "no-key"
            raise BackendError(
                classification="transient",
                message="ollama-cloud stand-in mint returned no token",
            )
        self._standin_token = token
        expires_raw = standin_body.get("expires_at") if isinstance(standin_body, dict) else None
        try:
            # credential proxy returns expires_at as relative seconds until expiry
            # (mirroring OllamaCloudUsageSource's interpretation). Fall back to
            # the requested TTL if missing/unparseable.
            expires_at = float(expires_raw) if expires_raw else float(OLLAMA_CLOUD_STANDIN_TTL_S)
        except (TypeError, ValueError):
            expires_at = float(OLLAMA_CLOUD_STANDIN_TTL_S)
        self._standin_expires_at = time.monotonic() + expires_at
        return self._standin_token

    def _invalidate_standin(self) -> None:
        """Clear the cached stand-in so the next call re-mints. Called on an
        credential proxy-level 401 (the stand-in itself was rejected) — distinct from
        an upstream ollama.com 401, which leaves the stand-in alone."""
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
        """POST one ollama.com call through credential proxy's buffered /v1/proxy.

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
        — NOT auth_invalid (that is reserved for upstream ollama.com 401).
        credential proxy 503 → _healthy=False, _last_health_reason="no-key" (the
        ollama-cloud scope is not wired / the pass entry is
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
                    message=f"ollama-cloud proxy transport error: {exc}",
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
                    message="ollama-cloud stand-in rejected by credential proxy after re-mint",
                )
            if response.status_code == 503:
                # credential proxy can't serve the scope — not wired / pass entry
                # missing. A provisioning gap, not a network outage.
                self._healthy = False
                self._last_health_reason = "no-key"
                raise BackendError(
                    classification="transient",
                    status_code=503,
                    message="ollama-cloud scope not available at credential proxy (503)",
                )
            if response.status_code != 200:
                # Other credential proxy-level non-200 — a transient outage.
                self._mark_unhealthy()
                raise BackendError(
                    classification="transient",
                    status_code=response.status_code,
                    message=f"ollama-cloud proxy failed (credential proxy HTTP {response.status_code})",
                )
            # credential proxy 200 — unwrap the upstream response. The CALLER
            # classifies the upstream status (upstream 401 → auth_invalid).
            try:
                wrapper = response.json()
            except ValueError as exc:
                self._mark_unhealthy()
                raise BackendError(
                    classification="transient",
                    message=f"ollama-cloud proxy returned non-JSON wrapper: {exc}",
                ) from exc
            if not isinstance(wrapper, dict):
                self._mark_unhealthy()
                raise BackendError(
                    classification="transient",
                    message="ollama-cloud proxy returned a non-object wrapper",
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
            message="ollama-cloud proxy exhausted re-mint attempts",
        )

    def _ollama_upstream_status(self, response: httpx.Response) -> int:
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
        """App-level request headers for the ollama.com hop. NEVER
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

    @contextlib.asynccontextmanager
    async def _proxy_stream_cm(self, out_body: dict[str, Any]) -> AsyncIterator[httpx.Response]:
        """Open a streaming /v1/proxy/stream call to ollama.com through
        credential proxy, yielding the raw httpx.Response for the generator to parse.

        Mints/refreshes the stand-in BEFORE opening (no in-stream re-mint —
        matches credential_proxy's streaming behavior). Builds the proxy
        envelope with the app-level headers and the JSON-serialized chat body.
        The generator's unchanged SSE state machine parses data: lines
        from credential proxy's verbatim-forwarded upstream OpenAI SSE.
        """
        token = await self._ensure_standin()
        envelope = {
            "url": f"{self._ollama_url}/v1/chat/completions",
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
        """Non-stream POST to ollama.com's OpenAI-compatible
        /v1/chat/completions endpoint, reached through credential proxy's buffered
        /v1/proxy with a short-TTL ollama-cloud stand-in (the real key
        never enters this process).

        Body-prep strips Codex-only fields (reasoning,
        parallel_tool_calls) that ollama rejects, same as litellm_gateway.
        No inference-param merge this slice (ollama_cloud has no
        _operator_state; see sub-slice 2 plan deferred-item).
        """
        await self._refresh_catalog_if_stale()
        out_body = _strip_codex_only_fields({**body, "stream": False})
        upstream_status, upstream_headers, upstream_body = await self._proxy_buffered(
            method="POST",
            url=f"{self._ollama_url}/v1/chat/completions",
            app_headers=self._app_request_headers(stream=False),
            body=json.dumps(out_body).encode("utf-8"),
        )
        if handle is not None:
            handle.upstream_status = upstream_status
            handle.upstream_headers = upstream_headers
        if upstream_status >= 400:
            # Upstream non-2xx (401 ⇒ key rejected by ollama.com →
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
        ollama-cloud stand-in (the real key never enters this process).

        Chat-native path (rarely used: Codex traffic flows through
        /v1/responses → responses_stream). Token accounting is intentionally
        NOT set on handle.stream_summary here, mirroring litellm_gateway's
        existing NULL-stream_summary gap on its chat-native path — not new
        debt, out of scope for this slice.
        """
        await self._refresh_catalog_if_stale()
        out_body = _strip_codex_only_fields({**body, "stream": True})
        # Mint the stand-in before opening (no in-stream re-mint — matches
        # credential_proxy's streaming behavior). _ensure_standin raises
        # BackendError(transient) on a mint failure (marks us unhealthy with
        # reason "no-key" / "network" as appropriate).
        token = await self._ensure_standin()
        envelope = {
            "url": f"{self._ollama_url}/v1/chat/completions",
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
                upstream_status = self._ollama_upstream_status(response)
                if handle is not None:
                    handle.upstream_status = upstream_status
                    handle.upstream_headers = dict(response.headers)
                # credential proxy-401 → header absent → 502 → transient (NOT
                # auth_invalid); upstream-401 → header 401 → auth_invalid.
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
            raise BackendError(classification="transient", message=str(exc)) from exc

    async def responses(self, body: dict[str, Any], handle: CallHandle | None = None) -> dict[str, Any]:
        """Responses-API surface, translated to/from chat-completions.

        ollama.com has no native Responses endpoint, so translate the
        Responses-shaped request to chat, call the chat-completions endpoint,
        and translate the chat reply back to Responses shape. The shared
        translators in `callosum.backends._responses_chat` are the same ones
        litellm_gateway uses — see that module's docstring.
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
        and reads the real upstream status via _ollama_upstream_status
        (the x-credential proxy-upstream-status header) instead of the credential proxy HTTP
        status. The collector tees the generator's output so
        handle.stream_summary carries the response.completed usage block →
        per-request token accounting + the usage log populate automatically
        (see app.py:_extract_tokens, which accepts the chat usage shape this
        generator emits).

        **Two 401 surfaces under proxy custody:** a mid-stream upstream 401
        from ollama.com surfaces via x-credential proxy-upstream-status:401 →
        auth_invalid via error_from_response(status_code=401) (no
        stand-in re-mint — the stand-in is innocent). An credential proxy-level 401
        (stand-in rejected) maps to 502 → transient. There is no in-stream
        stand-in re-mint (matches credential_proxy's streaming behavior);
        a transient stand-in rejection self-heals on the next request.
        """

        def open_chat_stream(out_body: dict[str, Any], _headers: dict[str, str]) -> AsyncContextManager[httpx.Response]:
            # `_headers` (the generator's default auth headers) is ignored —
            # credential proxy builds the final-hop headers from the envelope + the
            # real key. The stand-in is minted inside `_proxy_stream_cm`.
            return self._proxy_stream_cm(out_body)

        collector = ResponsesStreamCollector(
            chat_to_responses_stream(
                client=self._client,
                chat_url=f"{self._ollama_url}/v1/chat/completions",
                body=body,
                handle=handle,
                prep_body=lambda b: _strip_codex_only_fields({**b, "stream": True}),
                headers={},  # unused on the proxy path; open_chat_stream ignores it
                first_item_timeout_s=LOCAL_STREAM_FIRST_BYTE_TIMEOUT_S,
                idle_timeout_s=LOCAL_STREAM_IDLE_TIMEOUT_S,
                what_label="ollama-cloud",
                on_success=lambda: None,
                on_transport_error=self._mark_unhealthy,
                open_chat_stream=open_chat_stream,
                upstream_status_of=self._ollama_upstream_status,
            )
        )
        try:
            async for chunk in collector.iter_through():
                yield chunk
        finally:
            if handle is not None:
                handle.stream_summary = collector.summary

    def cell_capabilities(self, model: str) -> CellCapabilities:
        """Return real capabilities for `model`, discovered from ollama.

        Looks up the cache populated by `_refresh_capabilities` (called during
        catalog refresh). When the cache hasn't run yet OR /api/show failed,
        falls back to conservative text-only/no-tools defaults so the request
        can still dispatch — usually wrong on offline-but-capable cloud
        models, but better than refusing.

        Cost rank is always 10 (remote default). The CostRankProvider overlay
        in app.py may replace it with a cold-start catalog prior or an
        operator override; see routing/cost_model.py.
        """
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
        # ollama.com key never enters this process). `_proxy_buffered` mints
        # the stand-in, POSTs the envelope, and unwraps the upstream response.
        # This method is called from health(), which MUST NOT raise — map
        # every proxy/upstream failure to a health reason and return. The
        # proxy helper already set `_healthy=False` + a concrete reason
        # (network / no-key) before raising, so a bare `return` suffices.
        try:
            upstream_status, _upstream_headers, upstream_body = await self._proxy_buffered(
                method="GET",
                url=f"{self._ollama_url}/api/tags",
                app_headers=self._app_request_headers(stream=False),
                body=b"",
                timeout=DEFAULT_HEALTH_TIMEOUT_S,
            )
        except BackendError:
            return
        if upstream_status == 401:
            # Upstream ollama.com rejected the key credential proxy injected. The
            # stand-in is fine — do NOT re-mint or invalidate. Operator
            # re-provisions the key in credential proxy's pass; next call works.
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
        # Keep only catalog names matching the configured suffix. The default
        # suffix is "" (accepts ALL names — correct for a direct all-cloud
        # host where every /api/tags entry is a cloud model). The operator
        # overrides with CALLOSUM_OLLAMA_CLOUD_MODEL_SUFFIX=:cloud if ollama.com
        # still tags cloud models with that suffix. See DEFAULT_MODEL_SUFFIX.
        slugs: list[str] = []
        for entry in models:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if isinstance(name, str) and name and name.endswith(self._model_suffix):
                slugs.append(name)
        self._catalog = tuple(slugs)
        self._catalog_fetched_at = ts
        self._healthy = True
        self._last_health_reason = "ok"
        # Best-effort: refresh per-model capabilities so the router's filter
        # sees real tool/vision/context_window values. Failures here don't
        # fail catalog refresh — cell_capabilities falls back to safe defaults
        # when the cache is empty.
        await self._refresh_capabilities()

    async def _refresh_capabilities(self) -> None:
        """Populate `_capabilities_cache` from ollama.com's /api/show endpoint.

        Best-effort. For each cataloged cloud model, POST /api/show with the
        model name and parse the capabilities array + context_length +
        parameter_count. The mapping mirrors litellm_gateway's /api/show
        handling verbatim (cloud models are ollama names directly, so there is
        no litellm→ollama name indirection to resolve first).

        Auth: the request runs through credential proxy's buffered proxy (the real
        ollama.com key never enters this process). Any failure (network, key
        unavailable, malformed response) leaves the cache unchanged and the
        synchronous `cell_capabilities` falls back to conservative defaults.
        Broad exception swallow: this runs on the hot path of the first
        request after each catalog TTL boundary; any defect in /api/show
        parsing must NEVER bubble up and fail user-visible routing.
        """
        for name in self._catalog:
            try:
                upstream_status, _upstream_headers, upstream_body = await self._proxy_buffered(
                    method="POST",
                    url=f"{self._ollama_url}/api/show",
                    app_headers=self._app_request_headers(stream=False),
                    body=json.dumps({"name": name}).encode("utf-8"),
                    timeout=DEFAULT_HEALTH_TIMEOUT_S,
                )
            except BackendError:
                # Stand-in mint / credential proxy transport failure — no point
                # retrying the rest of the loop with the same unavailable
                # stand-in. Leave the cache as-is and bail out.
                return
            if upstream_status == 401:
                # Upstream rejected the key credential proxy injected. Stand-in is
                # fine — do NOT re-mint or invalidate. Skip this model; the
                # catalog-refresh caller already mapped 401 to auth_invalid.
                continue
            if upstream_status != 200:
                continue
            try:
                info = json.loads(upstream_body)
            except (ValueError, TypeError):
                continue
            # ollama's /api/show response shape:
            #   {
            #     "capabilities": ["completion","tools","vision","thinking"],
            #     "model_info": {"general.architecture": "...",
            #                    "<arch>.context_length": <int>, ...}
            #   }
            caps_list = info.get("capabilities") or []
            caps_set = {str(c).lower() for c in caps_list if isinstance(c, str)}
            modalities: set[str] = {"text"}
            if "vision" in caps_set:
                modalities.add("image")
            if "audio" in caps_set:
                modalities.add("audio")
            supports_tools = "tools" in caps_set
            # Context length lives under <architecture>.context_length; we
            # don't know the architecture name a priori. Walk model_info and
            # find any key ending in `.context_length`. Parameter count is
            # exposed at `general.parameter_count` uniformly across
            # architectures — used by the selector as a "more capable"
            # tiebreaker when cost is tied.
            context_window = 128_000  # fallback
            parameter_count: int | None = None
            model_info = info.get("model_info") or {}
            if isinstance(model_info, dict):
                for k, v in model_info.items():
                    if isinstance(k, str) and k.endswith("context_length") and isinstance(v, int) and v > 0:
                        context_window = v
                pc = model_info.get("general.parameter_count")
                if isinstance(pc, int) and pc > 0:
                    parameter_count = pc
            self._capabilities_cache[name] = CellCapabilities(
                context_window=context_window,
                modalities=frozenset(modalities),
                supports_tools=supports_tools,
                cost_rank=10,
                parameter_count=parameter_count,
            )
