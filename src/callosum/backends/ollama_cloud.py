"""Ollama Cloud backend — routes to `*:cloud` models served by the local
ollama daemon under `ollama signin`.

The operator authenticates with `ollama signin`, after which the local ollama
daemon (localhost:11434) holds the Ollama Cloud auth and auto-serves cloud
models. **Callosum holds NO Ollama Cloud credential** — the daemon does. This
backend talks to that daemon's existing OpenAI-compatible + native endpoints,
so no secret ever lives in callosum process state.

Why a distinct `BackendKind` (not a model-name predicate):
  "Local" is classified across the codebase by `backend.kind ==
  "litellm_gateway"` (~13 sites). A cloud model reached through the same
  daemon would otherwise be mis-routed as a FREE local cell — but cloud
  requests burn real Ollama Cloud quota. The new `BackendKind =
  "ollama_cloud"` makes every `== "litellm_gateway"` (local-affirmative) site
  NOT match and every `!= "litellm_gateway"` (remote-affirmative) site match,
  so cloud cells land in remote lanes and stay out of local-only/probing
  without further edits. See work tracker .

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

This backend is OPTIONAL and **env-gated OFF by default**: it is not
constructed unless `CALLOSUM_OLLAMA_CLOUD_ENABLED=1`, so enabling it is a
no-op for live routing until the operator turns it on. Sub-slice 1 covered
catalog enumeration, per-model capabilities, classification, registration, and
health. Sub-slice 2 wires chat dispatch: the four methods below hit the local
ollama daemon's OpenAI-compatible `/v1/chat/completions` endpoint (Approach A —
same path litellm_gateway uses, so the shared Responses↔Chat translators in
`callosum.backends._responses_chat` apply verbatim). The live-routing flip
(config enable + restart) is sub-slice 3, still operator-gated.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import AsyncIterator
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

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
# Ollama tags cloud-pulled models with a `:cloud` suffix (e.g. `model-a0d2:cloud`).
# Only models ending in this suffix are cloud-metered; bare local models stay
# on the LocalModelRegistry / litellm_gateway path. Configurable so the partition is
# tunable without code if the suffix convention changes.
DEFAULT_MODEL_SUFFIX = ":cloud"
DEFAULT_CATALOG_REFRESH_S = 60.0
DEFAULT_HEALTH_TIMEOUT_S = 2.0
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
# credential proxy usage source (credential-free, loopback-only, lazy+cached).
#
# Callosum holds NO credential proxy credential. The `OllamaCloudUsageSource` reads
# the loopback `/v1/ollama/usage` endpoint exposed by the credential proxy menubar
# applet (port 7342) using a short-lived stand-in token minted for the
# `ollama-usage` scope. The token's TTL is clamped to credential proxy's
# MAX_STANDIN_TTL_SECONDS (1800s). All network failures are swallowed and
# surfaced as `None` — usage is advisory only and must never fail routing.
# ---------------------------------------------------------------------
DEFAULT_CUSTODY_URL = "http://127.0.0.1:7342"
DEFAULT_OLLAMA_USAGE_ACCOUNT = "primary"
DEFAULT_STANDIN_TTL_S = 1800
MAX_STANDIN_TTL_S = 1800  # credential proxy's MAX_STANDIN_TTL_SECONDS; clamp requested TTL down to this
OLLAMA_USAGE_SCOPE = "ollama-usage"
DEFAULT_USAGE_CACHE_TTL_S = 30.0
DEFAULT_USAGE_TIMEOUT_S = 5.0


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
    """Cloud models served by the local ollama daemon under `ollama signin`."""

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
        # Catalog state — populated lazily from /api/tags.
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
        # Optional credential-free credential proxy usage source + live-projection
        # flag. When `usage_live` is True and `usage_source` is set,
        # `usage_snapshot()` prefers a measured session/weekly percent from
        # credential proxy's loopback `/v1/ollama/usage` over the honest-advisory
        # fallback. Both default off so existing construction is unchanged.
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
        # Force a catalog refresh if we're past TTL so dispatch sees the
        # daemon's current state rather than a stale snapshot.
        await self._refresh_catalog_if_stale()
        if self._healthy:
            return HealthStatus(available=True, reason="ok")
        return HealthStatus(
            available=False,
            reason="network" if self._last_health_reason == "network" else "unknown",
        )

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
        yet. Cloud requests burn real Ollama Cloud quota, but the daemon
        exposes no quota/usage headers (ollama/ollama #15663), so we cannot
        report a measured remaining fraction. We report
        `remaining_fraction=1.0` ("full, eligible") rather than the local
        free-stub's 0.001 — cloud is NOT free and must compete as a normal
        remote for primary selection, not be suppressed.
        `weekly_exhausted=False` because we genuinely don't know exhaustion
        from headers; the dispatch layer's failure-attribution + this
        backend's health handle outages separately.

        When the last health probe failed (daemon unreachable), report a
        short cooldown so `_routable_backends` excludes us — mirrors
        litellm_gateway's cold-start-vs-outage handling.
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
                weekly_used = (
                    payload.weekly.percent_used
                    if math.isfinite(payload.weekly.percent_used)
                    else 0.0
                )
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

    async def chat_completions(self, body: dict[str, Any], handle: CallHandle | None = None) -> dict[str, Any]:
        """Non-stream POST to the local ollama daemon's OpenAI-compatible
        /v1/chat/completions endpoint.

        The daemon holds the Ollama Cloud auth under `ollama signin`, so this
        backend sends NO `Authorization` header — the daemon injects cloud
        auth upstream. Body-prep strips Codex-only fields (`reasoning`,
        `parallel_tool_calls`) that ollama rejects, same as litellm_gateway.
        No inference-param merge this slice (ollama_cloud has no
        `_operator_state`; see sub-slice 2 plan deferred-item).
        """
        await self._refresh_catalog_if_stale()
        out_body = _strip_codex_only_fields({**body, "stream": False})
        try:
            response = await self._client.post(
                f"{self._ollama_url}/v1/chat/completions",
                json=out_body,
                headers={"Content-Type": "application/json"},
            )
        except httpx.HTTPError as exc:
            # Daemon unreachable — flip health so usage_snapshot reports
            # cooldown immediately, without waiting for the catalog TTL.
            self._mark_unhealthy()
            raise BackendError(classification="transient", message=str(exc)) from exc
        if handle is not None:
            handle.upstream_status = response.status_code
            handle.upstream_headers = dict(response.headers)
        if response.status_code >= 400:
            raise error_from_response(response)
        return cast(dict[str, Any], response.json())

    async def chat_completions_stream(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> AsyncIterator[bytes]:
        """Raw byte passthrough of the chat-completions SSE stream.

        Chat-native path (rarely used: Codex traffic flows through
        /v1/responses → responses_stream). Token accounting is intentionally
        NOT set on `handle.stream_summary` here, mirroring litellm_gateway's
        existing NULL-stream_summary gap on its chat-native path — not new
        debt, out of scope for this slice.
        """
        await self._refresh_catalog_if_stale()
        out_body = _strip_codex_only_fields({**body, "stream": True})
        try:
            stream_ctx = self._client.stream(
                "POST",
                f"{self._ollama_url}/v1/chat/completions",
                json=out_body,
                headers={"Content-Type": "application/json"},
            )
            async with stream_ctx as response:
                if handle is not None:
                    handle.upstream_status = response.status_code
                    handle.upstream_headers = dict(response.headers)
                if response.status_code >= 400:
                    await response.aread()
                    raise error_from_response(response)
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

        The ollama daemon has no native Responses endpoint, so translate the
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
        `callosum.backends._responses_chat` module (one copy for every
        chat-shaped backend); we inject this backend's coupling points via
        the keyword-only hooks: the daemon's /v1/chat/completions URL, the
        no-auth header (the daemon holds cloud auth), the codex-strip body
        prep, and the local-band stall-guard timeouts. The collector tees the
        generator's output so `handle.stream_summary` carries the
        response.completed usage block → per-request token accounting + the
        usage log populate automatically (see app.py:_extract_tokens, which
        accepts the chat usage shape this generator emits).
        """
        collector = ResponsesStreamCollector(
            chat_to_responses_stream(
                client=self._client,
                chat_url=f"{self._ollama_url}/v1/chat/completions",
                body=body,
                handle=handle,
                prep_body=lambda b: _strip_codex_only_fields({**b, "stream": True}),
                headers={"Content-Type": "application/json"},
                first_item_timeout_s=LOCAL_STREAM_FIRST_BYTE_TIMEOUT_S,
                idle_timeout_s=LOCAL_STREAM_IDLE_TIMEOUT_S,
                what_label="ollama-cloud",
                on_success=lambda: None,
                on_transport_error=self._mark_unhealthy,
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
        try:
            response = await self._client.get(
                f"{self._ollama_url}/api/tags",
                timeout=DEFAULT_HEALTH_TIMEOUT_S,
            )
        except httpx.HTTPError:
            # Daemon unreachable. Don't clobber an existing catalog — operator
            # may want last-known-good while the daemon restarts.
            self._healthy = False
            self._last_health_reason = "network"
            return
        if response.status_code != 200:
            self._healthy = False
            self._last_health_reason = "unknown"
            return
        try:
            payload = response.json()
        except ValueError:  # json.JSONDecodeError, but avoid importing json
            self._healthy = False
            self._last_health_reason = "unknown"
            return
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            self._healthy = False
            self._last_health_reason = "unknown"
            return
        # Keep only cloud-suffixed models; bare local models stay on the
        # LocalModelRegistry / litellm_gateway path. The suffix filter is the
        # local/cloud partition — see DEFAULT_MODEL_SUFFIX.
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
        """Populate `_capabilities_cache` from ollama's /api/show endpoint.

        Best-effort. For each cataloged cloud model, POST /api/show with the
        model name and parse the capabilities array + context_length +
        parameter_count. The mapping mirrors litellm_gateway's /api/show
        handling verbatim (cloud models are ollama names directly, so there is
        no litellm→ollama name indirection to resolve first).

        Any failure (daemon unreachable, malformed response) leaves the
        cache unchanged and the synchronous `cell_capabilities` falls back to
        conservative defaults. Broad exception swallow: this runs on the hot
        path of the first request after each catalog TTL boundary; any defect
        in /api/show parsing must NEVER bubble up and fail user-visible
        routing.
        """
        for name in self._catalog:
            try:
                show = await self._client.post(
                    f"{self._ollama_url}/api/show",
                    json={"name": name},
                    timeout=DEFAULT_HEALTH_TIMEOUT_S,
                )
            except Exception:
                continue
            if show.status_code != 200:
                continue
            try:
                info = show.json()
            except Exception:
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