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
headers. Slice 1 ships only the scaffold: `usage_snapshot()` reports
"remote, full, eligible, no signal yet" and `quota_snapshot()` returns None.
Accumulation from response usage counters (`prompt_eval_count` / `eval_count`)
is wired in sub-slice 2 alongside chat dispatch.

This backend is OPTIONAL and **env-gated OFF by default**: it is not
constructed unless `CALLOSUM_OLLAMA_CLOUD_ENABLED=1`, so enabling it is a
no-op for live routing until the operator turns it on. Sub-slice 1 covers
catalog enumeration, per-model capabilities, classification, registration, and
health. Chat dispatch is deferred to sub-slice 2; the dispatch methods below
raise `NotImplementedError` so a misconfigured early enable fails loudly rather
than silently dropping traffic.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from callosum.backend import BackendKind, CallHandle, HealthStatus, UsageSnapshot
from callosum.cell_grid import ModelMetadata
from callosum.routing.protocols import CellCapabilities

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

    async def usage_snapshot(self) -> UsageSnapshot:
        """Honest-advisory: report a remote cell with no quota signal yet.

        Cloud requests burn real Ollama Cloud quota, but the daemon exposes
        no quota/usage headers (ollama/ollama #15663), so we cannot report a
        measured remaining fraction. We report `remaining_fraction=1.0`
        ("full, eligible") rather than the local free-stub's 0.001 — cloud is
        NOT free and must compete as a normal remote for primary selection,
        not be suppressed. `weekly_exhausted=False` because we genuinely don't
        know exhaustion from headers; the dispatch layer's failure-attribution
        + this backend's health handle outages separately.

        When the last health probe failed (daemon unreachable), report a short
        cooldown so `_routable_backends` excludes us — mirrors
        litellm_gateway's cold-start-vs-outage handling.
        """
        now = time.time()
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
        # No Codex-style quota headers from ollama; honest-advisory
        # accumulation from response usage counters is wired in sub-slice 2.
        return None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def chat_completions(self, body: dict[str, Any], handle: CallHandle | None = None) -> dict[str, Any]:
        raise NotImplementedError("Ollama Cloud chat dispatch is wired in sub-slice 2")

    async def chat_completions_stream(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> AsyncIterator[bytes]:
        raise NotImplementedError("Ollama Cloud chat dispatch is wired in sub-slice 2")
        # Unreachable: the raise above fires on first `__anext__`. The yield
        # keeps this an async generator (satisfies the Backend Protocol's
        # AsyncIterator return type) rather than a plain coroutine.
        yield b""  # pragma: no cover

    async def responses(self, body: dict[str, Any], handle: CallHandle | None = None) -> dict[str, Any]:
        raise NotImplementedError("Ollama Cloud responses dispatch is wired in sub-slice 2")

    async def responses_stream(self, body: dict[str, Any], handle: CallHandle | None = None) -> AsyncIterator[bytes]:
        raise NotImplementedError("Ollama Cloud responses dispatch is wired in sub-slice 2")
        yield b""  # pragma: no cover

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