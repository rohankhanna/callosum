"""Local-model Backend that routes directly to each model's endpoint.

Discovery + capability lookup come from the local LLM gateway's CLI (the
operator's single source of truth for what's available locally).
Inference bypasses any centralized gateway — callosum POSTs directly
to each model's runtime endpoint (ollama, vllm, etc.) using whichever
API surface the model advertises (chat-completions or responses).

The result: callosum picks up new models the moment the local LLM gateway
sees them (after the next refresh tick), without ever requiring an
operator to edit a routing yaml. Pulling a model via `ollama pull` or
`the local LLM gateway pull` immediately makes it routable.

Compared to LiteLLMGatewayBackend:
- No litellm.yaml dependency
- One fewer hop on the request hot path
- Per-model endpoint routing (no shared bottleneck)

Compared to talking to ollama directly:
- Supports non-ollama runtimes (vllm, responses_proxy, model-a0e0)
- Capability data from operator-curated registry (the local LLM gateway) rather
  than per-model API probes
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import httpx

from callosum.backend import BackendKind, CallHandle, HealthStatus, UsageSnapshot
from callosum.backends._http import error_from_response, stall_guarded
from callosum.backends._ollama_capabilities import (
    OllamaShowCapabilities,
    fetch_ollama_capabilities,
)
from callosum.cell_grid import ModelMetadata
from callosum.config import LOCAL_STREAM_FIRST_BYTE_TIMEOUT_S, LOCAL_STREAM_IDLE_TIMEOUT_S
from callosum.errors import BackendError
from callosum.local import CapabilityRow, ModelEntry
from callosum.local_model_catalog import CuratedLocalModel, curate_local_models, curated_local_model_ids
from callosum.operator_state import (
    BACKEND_DEFAULT_INFERENCE_PARAMS,
    OperatorState,
    merge_inference_params,
)
from callosum.routing.local_performance import LocalPerformanceModel, build_local_performance_model
from callosum.routing.protocols import CellCapabilities
from callosum.sse_tee import ResponsesStreamCollector, ResponsesStreamSummary
from callosum.usage_log import ModelFitProbe, UsageLog

logger = logging.getLogger(__name__)


DEFAULT_CALL_TIMEOUT_S = 300.0  # Cold-load latency for big models can exceed 60s
LOCAL_PRIORITY_OFFSET = 10_000  # local cells sort after Codex cells in the grid


def _gpu_seconds_per_token(tokens_per_second: float | None) -> float | None:
    """Convert local throughput evidence into a GPU opportunity-cost signal."""
    if tokens_per_second is None or tokens_per_second <= 0:
        return None
    return 1.0 / tokens_per_second


class LocalModelSource(Protocol):
    def models(self, *, force: bool = False) -> list[ModelEntry]: ...

    def capabilities(self, *, force: bool = False) -> dict[str, CapabilityRow]:
        """Per-model hub-canonical capability rows (modalities/supports_tools).

        Returns an empty dict when the source does not emit deeper capability
        info (older hub builds / today) — callers treat absent rows as
        "hub is silent on this model" and fall through to the stopgap or
        conservative defaults. A row present but with `modalities`/`supports_tools`
        left `None` means the hub emitted the row but not those fields.
        """
        ...


class LocalModelRegistryBackend:
    """Backend that uses the local LLM gateway for discovery and routes inference
    directly to each model's runtime endpoint."""

    # reuse existing kind so the cell-grid + routing pipeline don't need to learn a new tag
    kind: BackendKind = "litellm_gateway"

    def __init__(
        self,
        *,
        id: str,
        source: LocalModelSource,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
        operator_state: OperatorState | None = None,
        usage_log_path: Path | None = None,
    ) -> None:
        self.id = id
        self._source = source
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(transport=transport, timeout=timeout_s)
            self._owns_client = True
        self._operator_state = operator_state
        # Full-context fit probe results live in the usage log; the backend
        # reads them on a TTL so the per-request catalog admission path does
        # not hit SQLite. None when no usage log is configured (batch jobs,
        # tests) — the catalog then admits optimistically on fit.
        self._usage_log_path = usage_log_path
        self._usage_log: UsageLog | None = None
        self._probe_results_cache: tuple[float, dict[str, ModelFitProbe]] | None = None
        # Health derives from "did the last source.models() call return
        # any models?" — proxy for "is the the local LLM gateway garage reachable
        # and configured?"
        self._healthy: bool = False
        self._last_health_reason: str = "unknown"
        # Tier-2 (ollama /api/show direct-ask stopgap) capability cache, keyed by
        # model id. Populated asynchronously by _refresh_capabilities on each
        # catalog refresh; cell_capabilities() reads it synchronously. Only
        # ollama runtimes are probed (only ollama exposes /api/show); other
        # runtimes fall to the hub-canonical tier or conservative defaults.
        self._capabilities_cache: dict[str, OllamaShowCapabilities] = {}
        # Stopgap scope, gated per operator decision:
        #   off | modalities (default) | all.
        # Modalities are strictly additive (vision was a hard 400 today -> zero
        # regression) so they ship on by default. Tool accuracy is operator-gated
        # because the runtime tool-probe is one-directional (probe-fail revokes,
        # probe-pass CANNOT grant): a sticky false-negative would exclude a cell
        # from tool routing with no path back in. Read once at construction.
        _stopgap = os.environ.get("CALLOSUM_LOCAL_CAPABILITIES_STOPGAP", "modalities")
        _stopgap = _stopgap.strip().lower()
        self._stopgap_modalities_enabled = _stopgap in {"modalities", "all"}
        self._stopgap_tools_enabled = _stopgap == "all"

    _PROBE_RESULTS_TTL_S = 60.0

    def _probe_results(self) -> Mapping[str, ModelFitProbe]:
        """Latest full-context fit probe per model id, cached on a TTL.

        Returns an empty mapping when no usage log is configured or the read
        fails, so the catalog admits optimistically on fit instead of crashing.
        """
        if self._usage_log_path is None:
            return {}
        now = time.monotonic()
        cache = self._probe_results_cache
        if cache is not None and now - cache[0] < self._PROBE_RESULTS_TTL_S:
            return cache[1]
        try:
            if self._usage_log is None:
                self._usage_log = UsageLog(self._usage_log_path)
            probes = self._usage_log.all_model_fit_probes()
        except OSError:
            self._probe_results_cache = (now, {})
            return {}
        self._probe_results_cache = (now, dict(probes))
        return self._probe_results_cache[1]

    # ---------- catalog ---------------------------------------------------

    @property
    def advertised_models(self) -> frozenset[str]:
        """Set of model_ids the local LLM gateway currently reports, filtered to
        cells this backend can actually serve on the streaming-responses
        path.

        Why the filter exists: callosum's primary inbound traffic
        (Codex CLI) hits /v1/responses with `stream: true`. Chat-only
        cells from the local LLM gateway (api_surfaces == ("chat",), e.g. bare
        `model-a0a9`) can't serve that path from this backend
        — `responses_stream` raises BackendError for them because no
        chat→responses stream translator is wired here. Advertising
        them anyway causes the router to pick them as primary, the
        request to 502 instantly with that error, and dispatch's
        cell-retry to fall through to a `-responses-proxy` sibling
        on the SAME backend that DOES serve responses natively. The
        net effect is one wasted 41-567ms roundtrip plus a noise row
        in the request log per real user request, with zero functional
        gain — the responses-proxy sibling serves the same underlying
        model. Filter the bare cells out at the source so the router
        never picks them in the first place.

        The deferred follow-up is to extract LiteLLMGatewayBackend's
        chat→responses stream translator into a shared helper and
        wire it here too; once that lands, chat-only cells become
        directly serveable and this filter can be dropped.
        """
        return frozenset(m.id for m in self._source.models() if "responses" in m.api_surfaces)

    @property
    def model_metadata(self) -> dict[str, ModelMetadata]:
        """Synthesize ModelMetadata for each local cell so the cell-grid
        merger picks them up alongside Codex models. Sort priority
        offset above the Codex range so local cells appear AFTER
        Codex in the grid's default ordering (the cost-weighted
        selector still picks cheapest, but ties resolve consistently).

        Mirrors `advertised_models` in filtering out chat-only cells —
        see that property's docstring for the rationale.
        """
        out: dict[str, ModelMetadata] = {}
        for idx, m in enumerate(self._source.models()):
            if "responses" not in m.api_surfaces:
                continue
            out[m.id] = ModelMetadata(
                slug=m.id,
                supported_in_api=True,
                visibility="list",
                priority=LOCAL_PRIORITY_OFFSET + idx,
                supported_reasoning_levels=m.supported_reasoning_levels,
                context_window=m.context_window,
            )
        return out

    def curated_local_models(
        self,
        *,
        min_tokens_per_second: float | None = None,
    ) -> list[CuratedLocalModel]:
        """Curated local fleet view for downstream routing policy.

        This is the backend-facing seam that exposes the operator-owned
        admission decision separately from raw discovery. Downstream
        consumers can inspect both admitted ids and rejection reasons
        without re-encoding the catalog policy.
        """
        return curate_local_models(
            self._source.models(),
            min_tokens_per_second=min_tokens_per_second,
            probe_results=self._probe_results(),
        )

    def admitted_local_model_ids(
        self,
        *,
        min_tokens_per_second: float | None = None,
    ) -> frozenset[str]:
        """Convenience view for the routing layer."""
        return frozenset(
            curated_local_model_ids(
                self._source.models(),
                min_tokens_per_second=min_tokens_per_second,
                probe_results=self._probe_results(),
            )
        )

    def local_model_admission_reasons(
        self,
        model: str,
        *,
        min_tokens_per_second: float | None = None,
    ) -> tuple[str, ...]:
        """Return the catalog reasons for one local model.

        Empty tuple means the model is admitted.
        """
        for item in self.curated_local_models(min_tokens_per_second=min_tokens_per_second):
            if item.id == model:
                return item.reasons
        return ("unknown-model",)

    def cell_capabilities(self, model: str) -> CellCapabilities:
        """Per-cell capabilities via a three-tier, per-field precedence.

        Tier 1 — hub-canonical: when the local LLM gateway's `capabilities --json`
            emits a modality/tool field for this model, that field is canonical
            truth (per operator instruction). Read defensively off the source's
            capability row; `None` when the hub is silent on that field (today
            it emits neither, so tier 1 is inert until the sibling hub ships
            the fields — tracked as an upstream handoff, NOT a callosum shim).
        Tier 2 — direct-ask stopgap (gated `CALLOSUM_LOCAL_CAPABILITIES_STOPGAP`
            = off | modalities (default) | all): when the hub won't/can't answer,
            ask the model's runtime directly via ollama `/api/show`. The self-report
            may be inaccurate; the runtime tool-probe stays as the safety net that
            can REVOKE a wrong tool claim (the probe is one-directional — it can
            revoke but cannot grant). Modalities are on by default (strictly
            additive — vision was a hard 400 today); tool accuracy is
            operator-gated (`all`) because a sticky false-negative would exclude
            a cell from tool routing with no path back in.
        Tier 3 — conservative defaults (current values; `supports_tools=True`
            optimistic so non-ollama runtimes with no direct-ask source stay
            routable for tools, probe-revocable).

        Local-performance fields (throughput/quant/admission) ALWAYS come from
        the roster entry, never from tiers 1/2. `context_window`: tier 2 fills
        only when the roster entry is `None` (don't let an ollama self-report
        clobber a measured hub roster value); tier 1 has no context window of
        its own.
        """
        entry = next((m for m in self._source.models() if m.id == model), None)
        if entry is None:
            return CellCapabilities(
                context_window=128_000,
                modalities=frozenset({"text"}),
                supports_tools=False,
                cost_rank=0,
                local_catalog_admitted=False,
                local_admission_reasons=("unknown-model",),
            )
        reasons = self.local_model_admission_reasons(model)

        # Tier 3 baseline (defaults + ALWAYS-PRESERVED local-perf fields).
        modalities: frozenset[str] = frozenset({"text"})
        supports_tools: bool = True  # optimistic; probe-revocable
        context_window: int = entry.context_window or 128_000

        # Tier 2: ollama /api/show self-report (gated per operator decision).
        stopgap = self._capabilities_cache.get(model)
        if stopgap is not None:
            if self._stopgap_modalities_enabled:
                modalities = stopgap.modalities
            if self._stopgap_tools_enabled:
                supports_tools = stopgap.supports_tools
            if entry.context_window is None and stopgap.context_window:
                context_window = stopgap.context_window

        # Tier 1: hub-canonical — PER-FIELD, wins over tier 2 when present. Not
        # gated (canonical truth by operator instruction); only activates when
        # the hub emits the fields, which it doesn't yet (both stay None today).
        hub = self._source.capabilities().get(model)
        if hub is not None:
            if hub.modalities is not None:
                modalities = hub.modalities
            if hub.supports_tools is not None:
                supports_tools = hub.supports_tools

        return CellCapabilities(
            context_window=context_window,
            modalities=modalities,
            supports_tools=supports_tools,
            cost_rank=0,
            local_throughput_tps=entry.estimated_tokens_per_second,
            local_gpu_seconds_per_token=_gpu_seconds_per_token(entry.estimated_tokens_per_second),
            local_quantization=entry.local_quantization,
            local_runnable_on_host=entry.local_runnable_on_host,
            local_status=entry.local_status,
            local_catalog_admitted=not reasons,
            local_admission_reasons=reasons,
        )

    def local_performance_model(self, model: str) -> LocalPerformanceModel | None:
        """Optional per-model local latency surface from the local LLM gateway evidence.

        Returns None when the hub lacks enough structured performance hints,
        letting the request-log time estimator remain the fallback.
        """
        for m in self._source.models():
            if m.id != model:
                continue
            if m.local_pool_bytes is None or m.estimated_tokens_per_second is None:
                return None
            return build_local_performance_model(
                model_id=m.id,
                quantization=m.local_quantization,
                pool_bytes=m.local_pool_bytes,
                free_bytes=m.local_pool_bytes,
                weight_bytes=None,
                kv_bytes_per_token=None,
                activation_bytes=None,
                fit_limit_tokens=m.local_fit_limit_tokens,
                estimated_tokens_per_second=m.estimated_tokens_per_second,
                prefill_ms_per_token=m.local_prefill_ms_per_token or 2.0,
                decode_bandwidth_kappa=m.local_decode_bandwidth_kappa or 1.0,
            )
        return None

    async def refresh_advertised_models(self, *, now: float | None = None) -> None:
        """Force the source to re-fetch. Mirrors codex_auth_vault's
        contract so the lifespan refresh loop drives both."""
        del now  # source has its own TTL
        models = self._source.models(force=True)
        self._healthy = bool(models)
        self._last_health_reason = "ok" if self._healthy else "no models"
        await self._refresh_capabilities()

    async def _refresh_capabilities(self) -> None:
        """Populate `_capabilities_cache` from ollama's /api/show — the tier-2
        direct-ask stopgap. Best-effort, never raises.

        Only ollama runtimes are probed (only ollama exposes /api/show). Non-ollama
        runtimes (vllm, model-a0e0, responses_proxy, gpt_oss) have no callosum-side
        direct-ask source and fall to the hub-canonical tier or conservative
        defaults. Runs concurrently via asyncio.gather so many cells don't block
        the lifespan refresh tick; each call carries its own 2s timeout and
        fetch_ollama_capabilities never raises, so a bad cell is skipped, not
        fatal. Stale entries for removed models are harmless — cell_capabilities
        guards every cache read behind a roster-entry lookup.
        """
        try:
            entries = self._source.models()
        except Exception:
            return
        ollama_entries = [m for m in entries if m.runtime == "ollama" and m.runtime_model]
        if not ollama_entries:
            return
        tasks = [
            (
                m.id,
                asyncio.create_task(
                    fetch_ollama_capabilities(
                        self._client,
                        endpoint=m.endpoint,
                        runtime_model=m.runtime_model,
                    )
                ),
            )
            for m in ollama_entries
        ]
        results = await asyncio.gather(*(t for _, t in tasks), return_exceptions=True)
        for (model_id, _), result in zip(tasks, results, strict=True):
            if isinstance(result, OllamaShowCapabilities):
                self._capabilities_cache[model_id] = result

    async def health(self) -> HealthStatus:
        # Cheap — reads cached state, refreshes only if stale.
        self._source.models()  # primes the cache
        models = self._source.models()
        if models:
            return HealthStatus(available=True, reason="ok")
        # No models. Distinguish "catalog CLI broken/unreachable" (the last
        # fetch failed — missing/broken/timeout) from "catalog empty" (the CLI
        # is healthy but the local model garage lists 0 models) so /status
        # names the real cause instead of an opaque "unknown". The source may
        # be a test stub without last_fetch_reason; default to "unknown" →
        # catalog_empty is not warranted, so fall back to the legacy "unknown".
        reason = getattr(self._source, "last_fetch_reason", "unknown")
        if reason in {"missing", "timeout", "broken"}:
            return HealthStatus(available=False, reason="catalog_cli_broken")
        if reason == "ok":
            # CLI ran clean but listed nothing — the garage is empty.
            return HealthStatus(available=False, reason="catalog_empty")
        return HealthStatus(available=False, reason="unknown")

    async def usage_snapshot(self) -> UsageSnapshot:
        # Local cells have no quota; tiny non-zero remaining_fraction
        # keeps them eligible without competing with Codex for "primary"
        # status (selector prefers larger remaining_fraction).
        return UsageSnapshot(
            remaining_fraction=0.001,
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=time.time(),
        )

    async def quota_snapshot(self) -> None:
        return None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ---------- inference ------------------------------------------------

    def _resolve(self, model_id: str) -> ModelEntry:
        """Look up the ModelEntry for `model_id`. Raises BackendError if
        the model isn't in the current the local LLM gateway registry (could
        have been pruned between the routing decision and dispatch)."""
        for m in self._source.models():
            if m.id == model_id:
                return m
        raise BackendError(
            classification="transient",
            message=f"model {model_id!r} not in local LLM gateway registry",
        )

    def _apply_params(self, body: dict[str, Any]) -> dict[str, Any]:
        """Merge operator-overridden + backend-default inference params
        into the request body. Same precedence as LiteLLMGatewayBackend."""
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

    def _outbound(self, body: dict[str, Any], *, stream: bool) -> tuple[ModelEntry, dict[str, Any]]:
        """Build the request body callosum will send. Resolves the
        the local LLM gateway entry, rewrites `model` from the public id (e.g.
        `model-a0b0`) to the runtime id (e.g. `model-a0d7`)
        that the underlying server expects, applies the operator's
        inference-param overrides, and pins stream."""
        model_id = body.get("model", "")
        entry = self._resolve(str(model_id))
        out = self._apply_params({**body, "stream": stream})
        out["model"] = entry.runtime_model
        # Strip Codex-only request fields that local stacks don't accept.
        # `reasoning.effort` is the main one — Codex uses it; ollama/
        # vllm tend to ignore-or-error.
        out.pop("reasoning", None)
        return entry, out

    async def chat_completions(self, body: dict[str, Any], handle: CallHandle | None = None) -> dict[str, Any]:
        entry, out_body = self._outbound(body, stream=False)
        # Native-responses runtimes (e.g. responses_proxy) advertise
        # api_surfaces=("responses","chat") because the LiteLLM gateway
        # *can* translate chat→responses for them — but the proxy
        # process itself does NOT implement /v1/chat/completions. A
        # blind POST hangs until the client timeout. Mirror the
        # translation pattern responses() already uses for chat-only
        # cells: when the entry advertises "responses", translate the
        # chat body into Responses shape, POST to /v1/responses, and
        # translate the payload back into a chat.completion. See
        # docs/investigations/2026-06-09-b3-multimodal-routing-concurrency.md
        # "Root-cause confirmation" for the upstream-vs-callosum split.
        if "responses" in entry.api_surfaces:
            from callosum.backends.codex_auth_vault import (
                _chat_to_responses_request,
                _responses_to_chat_response,
            )

            responses_body = _chat_to_responses_request(out_body, stream=False)
            # _chat_to_responses_request reads body["model"] for the
            # outgoing payload; out_body already has runtime_model
            # applied by _outbound, so this carries through correctly.
            try:
                response = await self._client.post(
                    f"{entry.endpoint.rstrip('/')}/v1/responses",
                    json=responses_body,
                )
            except httpx.HTTPError as exc:
                self._healthy = False
                self._last_health_reason = "network"
                raise BackendError(classification="transient", message=str(exc)) from exc
            if handle is not None:
                handle.upstream_status = response.status_code
                handle.upstream_headers = dict(response.headers)
            if response.status_code >= 400:
                raise error_from_response(response)
            payload = cast(dict[str, Any], response.json())
            # Preserve the public model id (what the client asked for)
            # in the chat.completion `model` field rather than the
            # runtime alias. Matches LiteLLMGatewayBackend's behavior.
            client_model = str(body.get("model", entry.runtime_model))
            return _responses_to_chat_response(payload, model=client_model)
        try:
            response = await self._client.post(
                f"{entry.endpoint.rstrip('/')}/v1/chat/completions",
                json=out_body,
            )
        except httpx.HTTPError as exc:
            self._healthy = False
            self._last_health_reason = "network"
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
        entry, out_body = self._outbound(body, stream=True)
        if "responses" in entry.api_surfaces:
            from callosum.backends.codex_auth_vault import _chat_to_responses_request

            responses_body = _chat_to_responses_request(out_body, stream=True)
            try:
                async with self._client.stream(
                    "POST",
                    f"{entry.endpoint.rstrip('/')}/v1/responses",
                    json=responses_body,
                ) as response:
                    if handle is not None:
                        handle.upstream_status = response.status_code
                        handle.upstream_headers = dict(response.headers)
                    if response.status_code >= 400:
                        await response.aread()
                        raise error_from_response(response)
                    chunks: list[bytes] = []
                    # Wrap the raw upstream line iterator in stall_guarded so
                    # this local responses-native chat-stream path gets the same
                    # first-byte/idle timeouts + max_idle_gap_s capture as the
                    # chat-native branch below (L491) and responses_stream (L574).
                    # Without this, idle_gap_ms stays NULL for this local
                    # sub-path even though LOCAL_STREAM_IDLE_TIMEOUT_S applies.
                    async for chunk in _responses_sse_to_chat_sse(
                        stall_guarded(
                            response.aiter_lines(),
                            first_item_timeout_s=LOCAL_STREAM_FIRST_BYTE_TIMEOUT_S,
                            idle_timeout_s=LOCAL_STREAM_IDLE_TIMEOUT_S,
                            what=f"local {entry.id}",
                            handle=handle,
                        ),
                        model=str(body.get("model", entry.runtime_model)),
                    ):
                        chunks.append(chunk)
                        yield chunk
                    _record_chat_stream_summary(handle, chunks)
                return
            except httpx.HTTPError as exc:
                self._healthy = False
                self._last_health_reason = "network"
                raise BackendError(classification="transient", message=str(exc)) from exc
        stream_options = out_body.get("stream_options")
        if not isinstance(stream_options, dict):
            stream_options = {}
        stream_options["include_usage"] = True
        out_body["stream_options"] = stream_options
        try:
            async with self._client.stream(
                "POST",
                f"{entry.endpoint.rstrip('/')}/v1/chat/completions",
                json=out_body,
            ) as response:
                if handle is not None:
                    handle.upstream_status = response.status_code
                    handle.upstream_headers = dict(response.headers)
                if response.status_code >= 400:
                    await response.aread()
                    raise error_from_response(response)
                chat_chunks: list[bytes] = []
                async for chunk in stall_guarded(
                    response.aiter_bytes(),
                    first_item_timeout_s=LOCAL_STREAM_FIRST_BYTE_TIMEOUT_S,
                    idle_timeout_s=LOCAL_STREAM_IDLE_TIMEOUT_S,
                    what=f"local {entry.id}",
                    handle=handle,
                ):
                    chat_chunks.append(chunk)
                    yield chunk
                _record_chat_stream_summary(handle, chat_chunks)
        except httpx.HTTPError as exc:
            self._healthy = False
            self._last_health_reason = "network"
            raise BackendError(classification="transient", message=str(exc)) from exc

    async def responses(self, body: dict[str, Any], handle: CallHandle | None = None) -> dict[str, Any]:
        """Responses API surface. When the resolved model advertises
        "responses" natively, POST there; otherwise translate to
        chat-completions and translate back (same path as the legacy
        LiteLLM gateway, but routed directly to the model endpoint)."""
        from callosum.backends._responses_chat import (
            _chat_to_responses_response,
            _responses_to_chat_request,
        )

        entry = self._resolve(str(body.get("model", "")))
        if "responses" in entry.api_surfaces:
            # Native responses route. Apply inference params, swap model
            # id, POST.
            out = self._apply_params({**body, "stream": False})
            out["model"] = entry.runtime_model
            try:
                response = await self._client.post(
                    f"{entry.endpoint.rstrip('/')}/v1/responses",
                    json=out,
                )
            except httpx.HTTPError as exc:
                self._healthy = False
                self._last_health_reason = "network"
                raise BackendError(classification="transient", message=str(exc)) from exc
            if handle is not None:
                handle.upstream_status = response.status_code
                handle.upstream_headers = dict(response.headers)
            if response.status_code >= 400:
                raise error_from_response(response)
            return cast(dict[str, Any], response.json())

        # Chat-only model — translate.
        chat_body = _responses_to_chat_request(body)
        chat_response = await self.chat_completions(chat_body, handle)
        return _chat_to_responses_response(chat_response)

    async def responses_stream(self, body: dict[str, Any], handle: CallHandle | None = None) -> AsyncIterator[bytes]:
        """Streaming responses. Native if the model supports it;
        otherwise we'd need a chat→responses stream translator (defer
        to a later commit — most ollama-served models use chat)."""
        entry = self._resolve(str(body.get("model", "")))
        if "responses" in entry.api_surfaces:
            out = self._apply_params({**body, "stream": True})
            out["model"] = entry.runtime_model
            try:
                async with self._client.stream(
                    "POST",
                    f"{entry.endpoint.rstrip('/')}/v1/responses",
                    json=out,
                ) as response:
                    if handle is not None:
                        handle.upstream_status = response.status_code
                        handle.upstream_headers = dict(response.headers)
                    if response.status_code >= 400:
                        await response.aread()
                        raise error_from_response(response)
                    # Wrap in a tee'ing collector so the bytes flow to
                    # the caller AND get buffered for parse-after-end.
                    # Dispatch reads `handle.stream_summary` to extract
                    # the upstream `response.completed` event's usage
                    # block — without this wrap, prompt_tokens /
                    # completion_tokens / total_tokens columns end up
                    # NULL for every local-served streamed request and
                    # any downstream consumer (shadow-eval, future
                    # bandit) loses cost/length signal entirely.
                    # Mirrors codex_auth_vault.responses_stream:510-514.
                    collector = ResponsesStreamCollector(
                        stall_guarded(
                            response.aiter_bytes(),
                            first_item_timeout_s=LOCAL_STREAM_FIRST_BYTE_TIMEOUT_S,
                            idle_timeout_s=LOCAL_STREAM_IDLE_TIMEOUT_S,
                            what=f"local {entry.id}",
                            handle=handle,
                        )
                    )
                    async for chunk in collector.iter_through():
                        yield chunk
                    if handle is not None:
                        handle.stream_summary = collector.summary
                return
            except httpx.HTTPError as exc:
                self._healthy = False
                self._last_health_reason = "network"
                raise BackendError(classification="transient", message=str(exc)) from exc

        # Chat-only model — fall back to the existing chat→responses
        # stream translator on LiteLLMGatewayBackend. (Defer fancier
        # direct-stream-translation to a follow-up.)
        raise BackendError(
            classification="transient",
            message=(
                f"chat-only model {entry.id!r} streaming via responses path "
                "needs the chat→responses stream translator wired in "
                "(call chat_completions_stream instead, or use the "
                "LiteLLMGatewayBackend translator)."
            ),
        )


async def _responses_sse_to_chat_sse(lines: AsyncIterator[str], *, model: str) -> AsyncIterator[bytes]:
    """Translate Responses-API SSE into OpenAI chat-completions SSE.

    Local responses-proxy cells expose `/v1/responses`, while callers on
    `/v1/chat/completions` expect chat chunk frames. This adapter keeps
    text and function-call argument deltas streaming as they arrive and
    emits a terminal chat chunk plus `[DONE]` when the Responses stream
    completes.
    """

    created = int(time.time())
    completion_id = "chatcmpl-local"
    sent_role = False
    finish_reason = "stop"
    tool_call_names: dict[int, str] = {}
    tool_call_ids: dict[int, str] = {}
    output_index_to_tool_index: dict[int, int] = {}

    def frame(
        delta: dict[str, Any],
        *,
        finish: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> bytes:
        payload: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish,
                }
            ],
        }
        if usage is not None:
            payload["usage"] = usage
        return f"data: {json.dumps(payload)}\n\n".encode()

    def ensure_role() -> bytes | None:
        nonlocal sent_role
        if sent_role:
            return None
        sent_role = True
        return frame({"role": "assistant"})

    data_lines: list[str] = []

    async for line in lines:
        if not line:
            if not data_lines:
                continue
            payload_str = "".join(data_lines)
            data_lines = []
            if payload_str == "[DONE]":
                break
            try:
                event = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            response = event.get("response")
            if isinstance(response, dict):
                if isinstance(response.get("id"), str) and response["id"]:
                    completion_id = response["id"]
                if isinstance(response.get("model"), str) and response["model"]:
                    model = response["model"]
            event_type = event.get("type")
            if event_type == "response.output_item.added":
                item = event.get("item")
                if not isinstance(item, dict) or item.get("type") != "function_call":
                    continue
                output_index = event.get("output_index")
                if not isinstance(output_index, int):
                    continue
                tool_index = len(output_index_to_tool_index)
                output_index_to_tool_index[output_index] = tool_index
                raw_name = item.get("name")
                raw_call_id = item.get("call_id")
                name = raw_name if isinstance(raw_name, str) else ""
                call_id = raw_call_id if isinstance(raw_call_id, str) else ""
                tool_call_names[tool_index] = name
                tool_call_ids[tool_index] = call_id
                role = ensure_role()
                if role is not None:
                    yield role
                yield frame(
                    {
                        "tool_calls": [
                            {
                                "index": tool_index,
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": ""},
                            }
                        ]
                    }
                )
            elif event_type == "response.output_text.delta":
                delta = event.get("delta")
                if not isinstance(delta, str) or not delta:
                    continue
                role = ensure_role()
                if role is not None:
                    yield role
                yield frame({"content": delta})
            elif event_type == "response.function_call_arguments.delta":
                delta = event.get("delta")
                if not isinstance(delta, str):
                    continue
                output_index = event.get("output_index")
                maybe_tool_index = (
                    output_index_to_tool_index.get(output_index) if isinstance(output_index, int) else None
                )
                if maybe_tool_index is None:
                    tool_index = len(output_index_to_tool_index)
                    if isinstance(output_index, int):
                        output_index_to_tool_index[output_index] = tool_index
                else:
                    tool_index = maybe_tool_index
                role = ensure_role()
                if role is not None:
                    yield role
                yield frame(
                    {
                        "tool_calls": [
                            {
                                "index": tool_index,
                                "id": tool_call_ids.get(tool_index, ""),
                                "type": "function",
                                "function": {
                                    "name": tool_call_names.get(tool_index, ""),
                                    "arguments": delta,
                                },
                            }
                        ]
                    }
                )
            elif event_type == "response.completed":
                usage = None
                if isinstance(response, dict):
                    raw_usage = response.get("usage")
                    if isinstance(raw_usage, dict):
                        usage = _responses_usage_to_chat_usage(raw_usage)
                    status = response.get("status")
                    if status == "cancelled":
                        finish_reason = "stop"
                    elif status == "incomplete":
                        finish_reason = "length"
                role = ensure_role()
                if role is not None:
                    yield role
                yield frame({}, finish=finish_reason, usage=usage)
                yield b"data: [DONE]\n\n"
                return
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].strip())

    role = ensure_role()
    if role is not None:
        yield role
    yield frame({}, finish=finish_reason)
    yield b"data: [DONE]\n\n"


def _responses_usage_to_chat_usage(usage: dict[str, Any]) -> dict[str, Any]:
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    total_tokens = usage.get("total_tokens")
    if not isinstance(total_tokens, int):
        total_tokens = (
            input_tokens + output_tokens if isinstance(input_tokens, int) and isinstance(output_tokens, int) else 0
        )
    return {
        "prompt_tokens": input_tokens if isinstance(input_tokens, int) else 0,
        "completion_tokens": output_tokens if isinstance(output_tokens, int) else 0,
        "total_tokens": total_tokens,
    }


def _record_chat_stream_summary(handle: CallHandle | None, chunks: list[bytes]) -> None:
    if handle is None:
        return
    blob = b"".join(chunks)
    usage = _chat_usage_from_sse_blob(blob)
    handle.stream_summary = ResponsesStreamSummary(
        completed_response={"usage": usage} if usage is not None else None,
        total_bytes=len(blob),
        raw_blob=blob,
    )


def _chat_usage_from_sse_blob(blob: bytes) -> dict[str, Any] | None:
    usage: dict[str, Any] | None = None
    for raw_event in blob.split(b"\n\n"):
        for line in raw_event.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            data = line[len(b"data:") :].strip()
            if not data or data == b"[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            raw_usage = payload.get("usage")
            if isinstance(raw_usage, dict):
                usage = raw_usage
    return usage
