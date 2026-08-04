"""LiteLLM gateway backend — routes to local models behind one
OpenAI-compatible endpoint managed by `local LLM gateway`.

`local LLM gateway` runs LiteLLM as a gateway (default 127.0.0.1:4000) that
abstracts over per-runtime servers (ollama, vllm, model-a0e0, etc.).
Callosum doesn't need N adapters — one Backend talks to the gateway,
and local LLM gateway handles per-runtime detail. Catalog of routable local
models comes from the gateway's /v1/models endpoint, refreshed
periodically so models added/removed via litellm.yaml propagate without
a callosum restart.

This backend is OPTIONAL: callosum continues to work normally when the
gateway is unreachable. Health flips to unavailable on the next poll
after the gateway goes down, dispatch skips this backend, and Codex
backends remain authoritative. When the gateway comes back, the next
catalog refresh re-enables it.

Local models have no concept of:
  * reasoning effort levels (Codex-specific) — we report a single
    "default" effort per local model in `model_metadata`.
  * weekly/5-hourly quotas — `usage_snapshot` always reports
    available, never exhausted, no cooldown.
  * Codex-style quota response headers — `quota_snapshot` returns None.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx

from callosum.backend import BackendKind, CallHandle, HealthStatus, UsageSnapshot
from callosum.backends._http import error_from_response, stall_guarded
from callosum.backends._responses_chat import (  # noqa: F401  (re-exported for tests — see note below)
    _CODEX_ONLY_BODY_KEYS,
    _RESPONSES_ONLY_KEYS,
    _chat_to_responses_response,
    _extract_reasoning_text,
    _extract_text_from_content,
    _responses_to_chat_request,
    _strip_codex_only_fields,
    chat_to_responses_stream,
)
from callosum.cell_grid import ModelMetadata
from callosum.config import LOCAL_STREAM_FIRST_BYTE_TIMEOUT_S, LOCAL_STREAM_IDLE_TIMEOUT_S
from callosum.errors import BackendError
from callosum.operator_state import (
    BACKEND_DEFAULT_INFERENCE_PARAMS,
    OperatorState,
    merge_inference_params,
)
from callosum.routing.protocols import CellCapabilities
from callosum.sse_tee import ResponsesStreamCollector

# The Responses↔Chat translators + chat→Responses streaming generator above
# were extracted to the shared `backends._responses_chat` module once a second
# chat-shaped backend (ollama_cloud) needed the same translation — see that
# module's docstring. We re-import the `_`-prefixed names so this backend's
# internal call sites AND `tests/unit/test_litellm_gateway.py` (which imports
# `_strip_codex_only_fields` etc. from `callosum.backends.litellm_gateway`) keep
# working without a rename sweep. This is a re-export, not a duplicate
# definition; F401 is suppressed because the names are re-exported, not used
# here directly.

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://127.0.0.1:4000"
DEFAULT_CATALOG_REFRESH_S = 60.0  # local model lineup changes via yaml reloads — keep fresh
DEFAULT_HEALTH_TIMEOUT_S = 2.0
# Default HTTP timeout for chat_completions / responses calls. Generous on
# purpose: a 31B-parameter local model can take 60-120s to cold-load from
# disk into VRAM, and the first request after a model swap is the one that
# pays the cost. Operators with fast hardware (or who pre-warm models) can
# lower this via CALLOSUM_LITELLM_TIMEOUT_S; the catalog poll uses
# DEFAULT_HEALTH_TIMEOUT_S separately and is unaffected.
DEFAULT_CALL_TIMEOUT_S = 300.0
# Large streaming tool requests that make a local model stall (accept the
# socket, emit nothing) are caught behaviorally by stall_guarded on the
# streaming read paths below — see callosum.config
# LOCAL_STREAM_FIRST_BYTE_TIMEOUT_S / LOCAL_STREAM_IDLE_TIMEOUT_S. No byte-size
# pre-flight cap; whether a request fits is the chosen model's context window's
# call (soft window-fit in router.py + the at_scale capability gate), not a
# constant.
# Priority offset for local cells in the merged cell grid. Remote Codex
# priorities are small ints (16, 23, etc.); offsetting local by +10_000
# means local cells sort AFTER Codex cells in the recommender's ranking,
# so the cheap-classifier prompt presents Codex first (preserving today's
# behavior) and local as additional options the classifier can pick.
LOCAL_PRIORITY_OFFSET = 10_000


class LiteLLMGatewayBackend:
    """OpenAI-compatible backend talking to a LiteLLM gateway."""

    kind: BackendKind = "litellm_gateway"

    def __init__(
        self,
        *,
        id: str,
        base_url: str = DEFAULT_BASE_URL,
        master_key: str | None = None,
        catalog_refresh_s: float = DEFAULT_CATALOG_REFRESH_S,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
        ollama_url: str = "http://127.0.0.1:11434",
        operator_state: OperatorState | None = None,
    ) -> None:
        self.id = id
        self._base_url = base_url.rstrip("/")
        self._master_key = master_key
        self._catalog_refresh_s = catalog_refresh_s
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(transport=transport, timeout=timeout_s)
            self._owns_client = True
        # Catalog state.
        self._catalog: tuple[str, ...] = ()
        self._catalog_fetched_at: float = 0.0
        # Health is derived from the most recent /v1/models poll. Until the
        # first poll lands we report unavailable so dispatch doesn't try to
        # route here before discovery completes.
        self._healthy: bool = False
        self._last_health_reason: str = "unknown"
        # Real per-model capabilities, populated lazily from ollama's
        # /api/show endpoint after the LiteLLM /model/info call resolves
        # the litellm→ollama name. cell_capabilities() reads from this
        # cache synchronously; refresh happens on each catalog refresh.
        # When discovery hasn't run yet (or the model isn't ollama-backed),
        # fallback to conservative defaults in cell_capabilities itself.
        self._ollama_url = ollama_url.rstrip("/")
        self._capabilities_cache: dict[str, CellCapabilities] = {}
        # Operator-state handle for per-cell inference parameter
        # overrides. None in tests / cold-start; treated as "no
        # overrides" by the merge helper.
        self._operator_state = operator_state

    @property
    def advertised_models(self) -> frozenset[str]:
        return frozenset(self._catalog)

    @property
    def model_metadata(self) -> dict[str, ModelMetadata]:
        """Synthesize ModelMetadata for each cataloged local model.

        Local models lack Codex-style metadata fields. We populate the cell
        grid by hand so the merge in app.py picks them up without changing
        the cell-grid filter logic:

          * supported_in_api=True, visibility="list" — so they pass the
            inclusion filter in `live_completion_models_from_metadata`.
          * supported_reasoning_levels=("default",) — one cell per local
            model rather than one cell per (model, effort).
          * priority=LOCAL_PRIORITY_OFFSET + idx — sorts AFTER Codex cells
            so the recommender's prompt presents remote first by default.
        """
        return {
            slug: ModelMetadata(
                slug=slug,
                supported_in_api=True,
                visibility="list",
                priority=LOCAL_PRIORITY_OFFSET + idx,
                supported_reasoning_levels=("default",),
            )
            for idx, slug in enumerate(self._catalog)
        }

    async def health(self) -> HealthStatus:
        # Force a catalog refresh if we're past TTL, so dispatch sees the
        # gateway's current state rather than a stale snapshot.
        await self._refresh_catalog_if_stale()
        if self._healthy:
            return HealthStatus(available=True, reason="ok")
        return HealthStatus(
            available=False,
            reason="network" if self._last_health_reason == "network" else "unknown",
        )

    async def usage_snapshot(self) -> UsageSnapshot:
        # Local has no quota; under normal conditions the dispatch layer
        # should never skip this backend on cooldown/exhaustion grounds.
        # Tiny non-zero remaining_fraction keeps it eligible without
        # competing with Codex for "primary" status.
        #
        # Exception: when our last health probe failed (gateway / ollama
        # unreachable), report a short cooldown so `_routable_backends`
        # excludes us. Otherwise heuristic fallback picks a local cell
        # whose backend can't actually serve it, and dispatch then
        # silently fails over to Codex with a non-Codex model name.
        # 30s cooldown is short enough to recover quickly once the
        # gateway is back, since the next health probe / call refresh
        # will reset _healthy.
        now = time.time()
        cooldown_until: float | None = None
        if not self._healthy and self._catalog_fetched_at > 0:
            # _catalog_fetched_at > 0 means we've successfully polled at
            # least once before; treat current unhealthy state as a
            # transient outage. On cold start (never-polled) keep the
            # legacy "always routable" behavior so backends_list isn't
            # empty before the first refresh runs.
            cooldown_until = now + 30.0
        return UsageSnapshot(
            remaining_fraction=0.001,
            cooldown_until_ts=cooldown_until,
            weekly_exhausted=False,
            probed_at_ts=now,
        )

    async def quota_snapshot(self) -> None:
        return None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def chat_completions(self, body: dict[str, Any], handle: CallHandle | None = None) -> dict[str, Any]:
        await self._refresh_catalog_if_stale()
        out_body = self._apply_inference_params(_strip_codex_only_fields({**body, "stream": False}))
        # KNOWN LIMITATION (2026-06-09): the LiteLLM gateway's chat-
        # completions hangs indefinitely for certain advertised models
        # whose upstream runtime is a responses-only proxy (model entries
        # with `-responses-proxy` suffix). The gateway's catalog reports
        # them via /v1/models but its chat-completions handler has no
        # working translation for them, so the POST below sits open until
        # the httpx client timeout fires. LocalModelRegistryBackend works around
        # this by translating chat→responses in-process; this backend
        # cannot do the same locally because it doesn't know which gateway
        # path actually serves each model.
        #
        # In the current deployment topology this is dead code:
        # __main__.py registers LiteLLMGatewayBackend ONLY when
        # LocalModelRegistryBackend is unavailable. If a future config re-enables
        # this backend alongside or instead of LocalModelRegistryBackend, the
        # symptom will be chat requests that hang at client timeout for
        # any `-responses-proxy` model. Fix shape (deferred): translate
        # chat→responses in-process when the model name patterns
        # responses-only, or probe each cataloged model at registration
        # and drop the ones whose chat path doesn't respond.
        try:
            response = await self._client.post(
                f"{self._base_url}/v1/chat/completions",
                json=out_body,
                headers=self._build_headers(),
            )
        except httpx.HTTPError as exc:
            # Transport-level failure means the gateway / ollama is
            # unreachable right now. Flip _healthy so usage_snapshot
            # reports cooldown immediately, without waiting for the
            # catalog TTL (60 s) to drive a refresh. Otherwise the
            # heuristic-fallback tier keeps picking a local cell
            # whose backend is dead.
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
        await self._refresh_catalog_if_stale()
        out_body = self._apply_inference_params(_strip_codex_only_fields({**body, "stream": True}))
        try:
            stream_ctx = self._client.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                json=out_body,
                headers=self._build_headers(),
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
                    what=f"local LLM gateway {out_body.get('model', '')}",
                    handle=handle,
                ):
                    yield chunk
        except httpx.HTTPError as exc:
            raise BackendError(classification="transient", message=str(exc)) from exc

    async def responses(self, body: dict[str, Any], handle: CallHandle | None = None) -> dict[str, Any]:
        # LiteLLM is chat-completions-first. Translate to chat, call,
        # translate back. Sufficient for the simple text path; advanced
        # Responses-API features (file inputs, structured outputs, etc.)
        # won't survive this translation.
        chat_body = _responses_to_chat_request(body)
        chat_response = await self.chat_completions(chat_body, handle)
        return _chat_to_responses_response(chat_response)

    async def responses_stream(self, body: dict[str, Any], handle: CallHandle | None = None) -> AsyncIterator[bytes]:
        """True stream-through Responses-API SSE.

        Opens an httpx stream to LiteLLM (which streams from ollama) and
        translates each chat-completions delta into the appropriate
        Responses-API event AS IT ARRIVES. No buffering.

        Why this matters:
          * Cancellation propagates. When the client disconnects,
            asyncio.CancelledError fires here, the `async with stream_ctx`
            block exits, httpx closes the upstream TCP socket, ollama
            sees the connection drop, and the runner can stop
            generating. The previous buffered impl awaited the full
            response before yielding anything, so cancellation reached
            nothing.
          * User sees progress live instead of a 30s+ stall.
          * For thinking models with `think: true`, the user sees
            reasoning content stream too, whether the upstream spells
            it `thinking`, `reasoning_content`, or `reasoning`.

        Translation state we track per choice index:
          * message text accumulation (chat `delta.content` → Responses
            `response.output_text.delta`)
          * reasoning accumulation (chat `delta.{thinking,
            reasoning_content,reasoning}` → Responses
            `response.reasoning_summary_text.delta`)
          * per-tool-call argument accumulation (chat `delta.tool_calls
            [j].function.arguments` chunks → Responses
            `response.function_call_arguments.delta` chunks)

        On finish_reason or [DONE]: emit per-item .done events,
        `response.output_item.done` for each, and `response.completed`
        carrying the assembled output array + usage.

        This method wraps its own translated output stream in a
        ResponsesStreamCollector and assigns the result to
        `handle.stream_summary` so the dispatch layer can extract the
        `response.completed` event's usage block and populate
        prompt_tokens/completion_tokens/total_tokens columns in the
        request log. Without this wrap, token accounting is silently
        NULL for every local-served streamed request. The actual
        translation runs in the shared `chat_to_responses_stream`
        generator (see `callosum.backends._responses_chat`) so this
        method can compose the collector + the generator without
        twisting either's control flow.
        """
        # The chat→Responses translation generator now lives in the shared
        # `backends._responses_chat` module (one copy for every chat-shaped
        # backend). We inject this backend's coupling points via the
        # keyword-only hooks: the gateway base URL + master-key headers, the
        # inference-param-aware body prep, and the local-band stall-guard
        # timeouts. The collector tees the generator's output so the final
        # `handle.stream_summary` has a parseable raw_blob containing the
        # emitted response.completed event (token accounting).
        collector = ResponsesStreamCollector(
            chat_to_responses_stream(
                client=self._client,
                chat_url=f"{self._base_url}/v1/chat/completions",
                body=body,
                handle=handle,
                prep_body=lambda b: self._apply_inference_params(
                    _strip_codex_only_fields({**b, "stream": True})
                ),
                headers=self._build_headers(),
                first_item_timeout_s=LOCAL_STREAM_FIRST_BYTE_TIMEOUT_S,
                idle_timeout_s=LOCAL_STREAM_IDLE_TIMEOUT_S,
                what_label="local LLM gateway",
                on_success=self._on_transport_success_litellm,
                on_transport_error=self._mark_unhealthy,
            )
        )
        try:
            async for chunk in collector.iter_through():
                yield chunk
        finally:
            if handle is not None:
                handle.stream_summary = collector.summary

    def _mark_unhealthy(self) -> None:
        """Flip health to network-down so usage_snapshot reports a cooldown
        immediately, without waiting for the catalog TTL to drive a refresh.
        Shared by the chat_completions / responses_stream transport-error
        paths and passed as the `on_transport_error` hook to
        chat_to_responses_stream."""
        self._healthy = False
        self._last_health_reason = "network"

    def _on_transport_success_litellm(self) -> None:
        """No-op placeholder so the structure mirrors codex_auth_vault's
        _on_transport_success. Currently we only flip _healthy on the
        catalog refresh path; this hook is here for future symmetry."""
        return

    def cell_capabilities(self, model: str) -> CellCapabilities:
        """Return real capabilities for `model`, discovered from ollama.

        Looks up the cache populated by `_refresh_capabilities` (called
        during catalog refresh). When the cache hasn't run yet OR the
        upstream model is non-ollama, falls back to conservative
        text-only/no-tools defaults so the request can still dispatch
        — usually wrong on offline-but-capable local models, but better
        than refusing.

        Cost rank is always 0 (local cells are cheapest by definition).
        """
        cached = self._capabilities_cache.get(model)
        if cached is not None:
            return cached
        return CellCapabilities(
            context_window=128_000,
            modalities=frozenset({"text"}),
            supports_tools=False,
            cost_rank=0,
        )

    async def _refresh_capabilities(self) -> None:
        """Populate `_capabilities_cache` from ollama's /api/show endpoint.

        Best-effort. Steps:
          1. Fetch LiteLLM's /model/info — gives us the litellm-name →
             upstream-model mapping (e.g. "model-a0b0" →
             "ollama/model-a0d7").
          2. For each ollama-backed entry, call ollama's /api/show with
             the upstream name. Parse `capabilities` array (tools,
             vision, audio, etc.) and `context_length`.
          3. Translate to CellCapabilities and store in the cache.

        Any failure (ollama unreachable, /model/info missing, malformed
        response) leaves the cache unchanged and the synchronous
        `cell_capabilities` falls back to conservative defaults. Called
        from `_refresh_catalog_if_stale` after a successful /v1/models
        fetch so capabilities stay in sync with the advertised set.

        Broad exception swallow: this method runs on the hot path of
        the first request after each catalog TTL boundary; any defect
        in /model/info or /api/show parsing must NEVER bubble up and
        fail user-visible routing.
        """
        # Step 1: litellm → ollama mapping.
        try:
            response = await self._client.get(
                f"{self._base_url}/model/info",
                headers=self._build_headers(),
                timeout=DEFAULT_HEALTH_TIMEOUT_S,
            )
        except Exception:
            return
        if response.status_code != 200:
            return
        try:
            payload = response.json()
        except Exception:
            return
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return
        ollama_mapping: dict[str, str] = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            litellm_name = entry.get("model_name")
            params = entry.get("litellm_params") or {}
            model_id = params.get("model", "") if isinstance(params, dict) else ""
            if not isinstance(model_id, str) or not isinstance(litellm_name, str):
                continue
            if model_id.startswith("ollama/"):
                ollama_mapping[litellm_name] = model_id[len("ollama/") :]
        # Step 2: per-model /api/show on ollama directly.
        for litellm_name, ollama_name in ollama_mapping.items():
            try:
                show = await self._client.post(
                    f"{self._ollama_url}/api/show",
                    json={"name": ollama_name},
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
            # Step 3: parse + translate. ollama's response shape:
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
            # Context length lives under <architecture>.context_length;
            # we don't know the architecture name a priori. Walk the
            # model_info dict and find any key ending in `.context_length`.
            # Parameter count is exposed at `general.parameter_count`
            # uniformly across architectures — used by the selector as
            # a "more capable" tiebreaker when cost is tied.
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
            self._capabilities_cache[litellm_name] = CellCapabilities(
                context_window=context_window,
                modalities=frozenset(modalities),
                supports_tools=supports_tools,
                cost_rank=0,
                parameter_count=parameter_count,
            )

    async def refresh_advertised_models(self, *, now: float | None = None) -> None:
        """Public refresh entry point — mirrors codex_auth_vault's contract so
        the lifespan startup loop can refresh all backends uniformly.

        Forces a catalog re-fetch unconditionally (no TTL gate) so callers
        can rely on advertised_models being current immediately after this
        returns. The internal `_refresh_catalog_if_stale` is unchanged for
        hot-path callers that want TTL-cached behavior.
        """
        # Clear the fetched-at timestamp so _refresh_catalog_if_stale
        # doesn't short-circuit on a stale-but-young entry.
        self._catalog_fetched_at = 0.0
        await self._refresh_catalog_if_stale(now=now)

    # ---------- internals --------------------------------------------------

    def _apply_inference_params(self, body: dict[str, Any]) -> dict[str, Any]:
        """Merge backend defaults + operator overrides into `body`.

        Lookup order per `merge_inference_params`:
          1. Client request body (kept untouched unless operator forces).
          2. Operator override for this model (`OperatorState`).
          3. Backend default for this kind (`think: false` for ollama-
             served local cells — keeps thinking-mode models from
             monopolizing inference budget on agentic prompts).

        When `_operator_state` is None (tests / cold start) only the
        backend defaults apply.
        """
        model = body.get("model")
        if not isinstance(model, str) or not model:
            return body
        backend_defaults = BACKEND_DEFAULT_INFERENCE_PARAMS.get(self.kind, {})
        operator_overrides: dict[str, Any] = {}
        operator_force = False
        if self._operator_state is not None:
            operator_overrides, operator_force = self._operator_state.get_inference_overrides(model)
        return merge_inference_params(
            backend_defaults=backend_defaults,
            operator_overrides=operator_overrides,
            operator_force=operator_force,
            client_body=body,
        )

    def _build_headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._master_key:
            h["Authorization"] = f"Bearer {self._master_key}"
        return h

    async def _refresh_catalog_if_stale(self, *, now: float | None = None) -> None:
        ts = now if now is not None else time.time()
        if self._catalog and ts - self._catalog_fetched_at < self._catalog_refresh_s:
            return
        try:
            response = await self._client.get(
                f"{self._base_url}/v1/models",
                headers=self._build_headers(),
                timeout=DEFAULT_HEALTH_TIMEOUT_S,
            )
        except httpx.HTTPError:
            # Gateway unreachable. Don't clobber an existing catalog —
            # operator may want last-known-good while the gateway restarts.
            self._healthy = False
            self._last_health_reason = "network"
            return
        if response.status_code != 200:
            self._healthy = False
            self._last_health_reason = "unknown"
            return
        try:
            payload = response.json()
        except json.JSONDecodeError:
            self._healthy = False
            self._last_health_reason = "unknown"
            return
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            self._healthy = False
            self._last_health_reason = "unknown"
            return
        slugs: list[str] = []
        for entry in data:
            if isinstance(entry, dict):
                slug = entry.get("id")
                if isinstance(slug, str) and slug:
                    slugs.append(slug)
        self._catalog = tuple(slugs)
        self._catalog_fetched_at = ts
        self._healthy = True
        self._last_health_reason = "ok"
        # Best-effort: refresh per-model capabilities from ollama so the
        # router's filter sees real tool/vision/context_window values.
        # Failures here don't fail catalog refresh — cell_capabilities
        # falls back to safe defaults when the cache is empty.
        await self._refresh_capabilities()
