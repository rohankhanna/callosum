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
from callosum.backends._http import error_from_response
from callosum.cell_grid import ModelMetadata
from callosum.errors import BackendError
from callosum.operator_state import (
    BACKEND_DEFAULT_INFERENCE_PARAMS,
    OperatorState,
    merge_inference_params,
)
from callosum.routing.protocols import CellCapabilities
from callosum.sse_tee import ResponsesStreamCollector

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
# Codex CLI can send very large streaming tool requests once a session has
# history. Local tool-capable models may accept the socket and then produce no
# useful bytes until the 300s transport timeout. Fail fast before opening the
# upstream stream so local-only mode does not look like a loop.
MAX_LOCAL_TOOL_REQUEST_BYTES = 50_000
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
        _reject_oversized_tool_request(out_body)
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
        await self._refresh_catalog_if_stale()
        out_body = self._apply_inference_params(_strip_codex_only_fields({**body, "stream": True}))
        _reject_oversized_tool_request(out_body)
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
                async for chunk in response.aiter_bytes():
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
            thinking content stream too.

        Translation state we track per choice index:
          * message text accumulation (chat `delta.content` → Responses
            `response.output_text.delta`)
          * thinking accumulation (chat `delta.thinking` → Responses
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
        translation runs in `_responses_stream_inner` so the public
        method can compose the collector + the generator without
        twisting either's control flow.
        """
        # Capture the inner generator's output as it flows so the
        # final `handle.stream_summary` has a parseable raw_blob
        # containing the emitted response.completed event.
        collector = ResponsesStreamCollector(self._responses_stream_inner(body, handle))
        try:
            async for chunk in collector.iter_through():
                yield chunk
        finally:
            if handle is not None:
                handle.stream_summary = collector.summary

    async def _responses_stream_inner(self, body: dict[str, Any], handle: CallHandle | None) -> AsyncIterator[bytes]:
        """The translation generator. Was previously the body of
        `responses_stream` directly; split out so the outer method
        can tee the output through a ResponsesStreamCollector to
        populate stream_summary on the handle. See `responses_stream`
        docstring for the rationale."""
        out_body = self._apply_inference_params(_strip_codex_only_fields({**body, "stream": True}))
        _reject_oversized_tool_request(out_body)
        # If the body arrived in Responses-API shape (has `input` instead
        # of `messages`), translate to Chat Completions shape before
        # forwarding. This is the streaming counterpart of what `responses`
        # already does on the non-stream path. Without this, LiteLLM's
        # Router throws TypeError("missing required argument: 'messages'")
        # against any bare ollama / vLLM endpoint that doesn't have a
        # Responses-API translation wrapper running in front. Idempotent:
        # if the body is already in Chat shape, this branch is skipped.
        if "input" in out_body and "messages" not in out_body:
            out_body = _responses_to_chat_request(out_body)
        # OpenAI Chat Completions streaming omits the `usage` block
        # unless include_usage is explicitly requested. Without it, no
        # SSE chunk carries token counts, the terminal response.completed
        # event ships zeros, and downstream token accounting is NULL for
        # every streamed local-served request. ollama / LiteLLM / vLLM
        # all honor this flag in their OpenAI-compatible mode. We set
        # it unconditionally because every streamed chat completion
        # benefits from it; if the operator already set it in the body,
        # ours is a no-op (same value).
        stream_options = out_body.get("stream_options")
        if not isinstance(stream_options, dict):
            stream_options = {}
        stream_options["include_usage"] = True
        out_body["stream_options"] = stream_options
        seq = 0

        def _emit(event_type: str, payload: dict[str, Any]) -> bytes:
            nonlocal seq
            seq += 1
            ev = {"type": event_type, "sequence_number": seq, **payload}
            return f"event: {event_type}\ndata: {json.dumps(ev)}\n\n".encode()

        # Per-stream accumulation state. Indexed by "output_index" in the
        # Responses-API sense; each output item gets a stable index from
        # the order it FIRST appeared in the stream.
        text_so_far = ""
        thinking_so_far = ""
        tool_calls_state: dict[int, dict[str, Any]] = {}  # idx → {id, name, args, output_index}
        output_items: list[dict[str, Any]] = []  # final assembled output for response.completed
        next_output_index = 0
        reasoning_output_index: int | None = None
        message_output_index: int | None = None
        message_item_id: str | None = None
        reasoning_item_id: str | None = None
        resp_id = "resp-litellm"
        upstream_model = out_body.get("model", "")
        usage: dict[str, Any] | None = None

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
                # Emit response.created early so clients show progress.
                base_response = {
                    "id": resp_id,
                    "object": "response",
                    "model": upstream_model,
                    "status": "in_progress",
                    "output": [],
                }
                yield _emit("response.created", {"response": base_response})
                yield _emit("response.in_progress", {"response": base_response})

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        payload_str = line[5:].strip()
                    else:
                        continue
                    if payload_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(chunk.get("id"), str) and chunk["id"]:
                        resp_id = chunk["id"]
                    if isinstance(chunk.get("model"), str) and chunk["model"]:
                        upstream_model = chunk["model"]
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta") or {}
                    if not isinstance(delta, dict):
                        delta = {}

                    # Thinking deltas (model-a0d5/model-a0g2/r1 thinking mode).
                    thinking_delta = delta.get("thinking")
                    if isinstance(thinking_delta, str) and thinking_delta:
                        if reasoning_output_index is None:
                            reasoning_output_index = next_output_index
                            next_output_index += 1
                            reasoning_item_id = f"rs_{resp_id}_{reasoning_output_index}"
                            yield _emit(
                                "response.output_item.added",
                                {
                                    "output_index": reasoning_output_index,
                                    "item": {
                                        "type": "reasoning",
                                        "id": reasoning_item_id,
                                        "summary": [],
                                    },
                                },
                            )
                        thinking_so_far += thinking_delta
                        yield _emit(
                            "response.reasoning_summary_text.delta",
                            {
                                "item_id": reasoning_item_id,
                                "output_index": reasoning_output_index,
                                "summary_index": 0,
                                "delta": thinking_delta,
                            },
                        )

                    # Tool-call deltas.
                    tool_calls_delta = delta.get("tool_calls") or []
                    if isinstance(tool_calls_delta, list):
                        for tc_delta in tool_calls_delta:
                            if not isinstance(tc_delta, dict):
                                continue
                            tc_idx = tc_delta.get("index", 0)
                            if not isinstance(tc_idx, int):
                                tc_idx = 0
                            state = tool_calls_state.get(tc_idx)
                            if state is None:
                                # First time we see this tool call — emit added.
                                call_id = tc_delta.get("id") or f"fc_{resp_id}_{tc_idx}"
                                fn = tc_delta.get("function") or {}
                                name = fn.get("name", "") if isinstance(fn, dict) else ""
                                out_idx = next_output_index
                                next_output_index += 1
                                state = {
                                    "id": call_id,
                                    "name": name,
                                    "args": "",
                                    "output_index": out_idx,
                                    "item_id": f"fc_{call_id}",
                                }
                                tool_calls_state[tc_idx] = state
                                yield _emit(
                                    "response.output_item.added",
                                    {
                                        "output_index": out_idx,
                                        "item": {
                                            "type": "function_call",
                                            "id": state["item_id"],
                                            "call_id": call_id,
                                            "name": name,
                                            "arguments": "",
                                            "status": "in_progress",
                                        },
                                    },
                                )
                            else:
                                fn = tc_delta.get("function") or {}
                                if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                                    state["name"] = fn["name"] or state["name"]
                            fn = tc_delta.get("function") or {}
                            args_delta = fn.get("arguments") if isinstance(fn, dict) else None
                            if isinstance(args_delta, str) and args_delta:
                                state["args"] += args_delta
                                yield _emit(
                                    "response.function_call_arguments.delta",
                                    {
                                        "item_id": state["item_id"],
                                        "output_index": state["output_index"],
                                        "delta": args_delta,
                                    },
                                )

                    # Message text delta.
                    content_delta = delta.get("content")
                    if isinstance(content_delta, str) and content_delta:
                        if message_output_index is None:
                            message_output_index = next_output_index
                            next_output_index += 1
                            message_item_id = f"msg_{resp_id}_{message_output_index}"
                            yield _emit(
                                "response.output_item.added",
                                {
                                    "output_index": message_output_index,
                                    "item": {
                                        "type": "message",
                                        "id": message_item_id,
                                        "role": "assistant",
                                        "content": [],
                                        "status": "in_progress",
                                    },
                                },
                            )
                        text_so_far += content_delta
                        yield _emit(
                            "response.output_text.delta",
                            {
                                "item_id": message_item_id,
                                "output_index": message_output_index,
                                "content_index": 0,
                                "delta": content_delta,
                            },
                        )

            # Stream ended cleanly.
            # Emit per-item .done events in output order, then
            # response.output_item.done, then response.completed.
            if reasoning_output_index is not None:
                yield _emit(
                    "response.reasoning_summary_text.done",
                    {
                        "item_id": reasoning_item_id,
                        "output_index": reasoning_output_index,
                        "summary_index": 0,
                        "text": thinking_so_far,
                    },
                )
                reasoning_item = {
                    "type": "reasoning",
                    "id": reasoning_item_id,
                    "summary": [{"type": "summary_text", "text": thinking_so_far}],
                }
                output_items.append(reasoning_item)
                yield _emit(
                    "response.output_item.done",
                    {
                        "output_index": reasoning_output_index,
                        "item": reasoning_item,
                    },
                )

            for tc_idx in sorted(tool_calls_state):
                state = tool_calls_state[tc_idx]
                yield _emit(
                    "response.function_call_arguments.done",
                    {
                        "item_id": state["item_id"],
                        "output_index": state["output_index"],
                        "arguments": state["args"],
                    },
                )
                fn_item = {
                    "type": "function_call",
                    "id": state["item_id"],
                    "call_id": state["id"],
                    "name": state["name"],
                    "arguments": state["args"],
                    "status": "completed",
                }
                output_items.append(fn_item)
                yield _emit(
                    "response.output_item.done",
                    {
                        "output_index": state["output_index"],
                        "item": fn_item,
                    },
                )

            if message_output_index is not None:
                yield _emit(
                    "response.output_text.done",
                    {
                        "item_id": message_item_id,
                        "output_index": message_output_index,
                        "content_index": 0,
                        "text": text_so_far,
                    },
                )
                msg_item = {
                    "type": "message",
                    "id": message_item_id,
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text_so_far}],
                    "status": "completed",
                }
                output_items.append(msg_item)
                yield _emit(
                    "response.output_item.done",
                    {
                        "output_index": message_output_index,
                        "item": msg_item,
                    },
                )

            # Codex CLI requires input_tokens; map from chat usage shape.
            chat_usage = usage if isinstance(usage, dict) else {}
            usage_out = {
                "input_tokens": int(chat_usage.get("prompt_tokens", 0) or 0),
                "output_tokens": int(chat_usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(chat_usage.get("total_tokens", 0) or 0),
            }
            full_response = {
                "id": resp_id,
                "object": "response",
                "model": upstream_model,
                "status": "completed",
                "output": output_items,
                "usage": usage_out,
            }
            yield _emit("response.completed", {"response": full_response})
            yield b"data: [DONE]\n\n"
            # Success — clear the offline tracker.
            self._on_transport_success_litellm()
        except httpx.HTTPError as exc:
            # Transport-level error during the stream — flip _healthy
            # so the offline detector kicks in.
            self._healthy = False
            self._last_health_reason = "network"
            raise BackendError(classification="transient", message=str(exc)) from exc

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


# ---------- body hygiene ------------------------------------------------
# Local backends (ollama / vLLM / model-a0e0 behind LiteLLM) don't honor
# every Chat-Completions request field OpenAI advertises. LiteLLM in
# `drop_params: false` mode (the default in local LLM gateway's config) errors
# hard instead of silently dropping — `litellm.UnsupportedParamsError`
# bubbles up as a 400 to the caller. Strip the fields known to trigger
# this BEFORE sending. Currently includes:
#   * `reasoning`: Codex-specific routing hint; no local backend uses it.
#   * `parallel_tool_calls`: standard Chat-Completions field (controls
#     whether the model emits multiple tool calls in one turn) but ollama
#     does not implement it. The model defaults to its native behavior
#     either way; dropping the flag is harmless because no local backend
#     can honor it. Add new keys as more upstream incompatibilities
#     surface in real traffic.
_CODEX_ONLY_BODY_KEYS = ("reasoning", "parallel_tool_calls")


def _reject_oversized_tool_request(body: dict[str, Any]) -> None:
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return
    request_bytes = len(json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    if request_bytes <= MAX_LOCAL_TOOL_REQUEST_BYTES:
        return
    raise BackendError(
        classification="client_error",
        status_code=413,
        message=(
            "local backend request too large for tool-capable local routing "
            f"({request_bytes} bytes > {MAX_LOCAL_TOOL_REQUEST_BYTES}); "
            "use remote-only or reduce conversation context"
        ),
    )


def _strip_codex_only_fields(body: dict[str, Any]) -> dict[str, Any]:
    """Drop request keys local backends refuse from a body destined for
    a local backend.

    Pure-fn returns a new dict; the caller's `body` is untouched. The
    name is historical — the strip list started as Codex-only fields but
    has grown to cover any field that OpenAI Chat Completions advertises
    but local OpenAI-compatible servers reject. See _CODEX_ONLY_BODY_KEYS
    docstring for the current contents and why each is dropped.
    """
    if not any(k in body for k in _CODEX_ONLY_BODY_KEYS):
        return body
    return {k: v for k, v in body.items() if k not in _CODEX_ONLY_BODY_KEYS}


# ---------- /v1/responses ↔ /v1/chat/completions translation -------------
# Translation helpers live here rather than a shared module — they're the
# refactor if a third backend needs the same translation.


# Keys that are part of the Responses-API request envelope and have no
# meaning for /v1/chat/completions. Stripped during translation so the
# Chat backend doesn't reject the body or silently ignore them. Everything
# else (stream, stream_options, temperature, tools, tool_choice, max_tokens,
# top_p, parallel_tool_calls, n, seed, response_format, …) is preserved
# verbatim — keeps the translator small as new request fields are added
# upstream.
_RESPONSES_ONLY_KEYS: frozenset[str] = frozenset(
    {
        "input",
        "instructions",
        "store",
        "include",
        "prompt_cache_key",
        "client_metadata",
        "text",  # Responses-API response_format equivalent; not the same field
        "reasoning",  # Codex-only request hint; redundant since _strip_codex_only_fields
    }
)


def _extract_text_from_content(content: Any) -> str:
    """Pull all text from a Responses-API `content` field.

    The Responses API allows content to be either a plain string OR a list
    of part dicts like `{"type": "input_text", "text": "..."}`. Both shapes
    appear in Codex CLI traffic. Return the concatenation of every text-
    bearing part; non-text parts (images, audio) are silently skipped here
    because the chat-completions translator can't represent them in a
    text-only role.content slot.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "".join(parts)
    return ""


def _responses_to_chat_request(body: dict[str, Any]) -> dict[str, Any]:
    """Translate a Codex /v1/responses request body to /v1/chat/completions shape.

    Handles the item types Codex CLI actually sends:

      * `message` (user / assistant / developer / system): copied to a
        chat message with the same role + extracted text.
      * `function_call`: collected into a pending tool_calls list; consecutive
        function_calls collapse into ONE assistant message with multiple
        tool_calls (matches OpenAI's parallel-tool-call shape so downstream
        models that paid attention to its training know what to do).
      * `function_call_output`: flushes the pending tool_calls group, then
        emits a `role: tool` message with `tool_call_id` matching the call.
      * `custom_tool_call` / `custom_tool_call_output`: same as function_*,
        treated identically since the on-wire shape is the same.
      * `reasoning`: DROPPED. The encrypted_content blobs are OpenAI-server-
        side state with no chat-completions representation. The model loses
        its prior internal CoT but the visible assistant messages still
        carry the conclusions, which is what matters for continuation.
      * `compaction`: preserved as a system note so the model knows prior
        turns were summarized away.

    Preserves every other top-level key from the input body (stream,
    stream_options, temperature, tools, tool_choice, max_tokens, top_p,
    parallel_tool_calls, etc.) so future Responses-API fields that ALSO
    apply to chat-completions don't need a translator change.
    """
    # Start with every non-Responses-only key carried over unchanged. This
    # is the opposite of a key whitelist — we explicitly know what to
    # drop, and pass through everything else. Reduces translator churn as
    # the upstream API grows.
    chat_body: dict[str, Any] = {k: v for k, v in body.items() if k not in _RESPONSES_ONLY_KEYS}
    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    input_block = body.get("input")
    if isinstance(input_block, str):
        messages.append({"role": "user", "content": input_block})
    elif isinstance(input_block, list):
        # Pending group of consecutive function_call items, materialized as
        # a single assistant message with multiple tool_calls when something
        # non-function_call interrupts the run.
        pending_tool_calls: list[dict[str, Any]] = []

        def _flush_pending() -> None:
            if pending_tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": list(pending_tool_calls),
                    }
                )
                pending_tool_calls.clear()

        for item in input_block:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "message":
                _flush_pending()
                role = item.get("role", "user")
                text = _extract_text_from_content(item.get("content"))
                # Always emit even if text is empty — preserves turn
                # structure for models that expect alternation.
                messages.append({"role": role, "content": text})
            elif t in ("function_call", "custom_tool_call"):
                call_id = item.get("call_id") or item.get("id", "")
                args = item.get("arguments")
                if args is None:
                    # custom_tool_call uses `input` instead of `arguments`.
                    args = item.get("input", "")
                if isinstance(args, (dict, list)):
                    args = json.dumps(args)
                elif not isinstance(args, str):
                    args = "{}"
                pending_tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": args,
                        },
                    }
                )
            elif t in ("function_call_output", "custom_tool_call_output"):
                _flush_pending()
                output = item.get("output", "")
                if isinstance(output, (dict, list)):
                    output = json.dumps(output)
                elif not isinstance(output, str):
                    output = str(output)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": output,
                    }
                )
            elif t == "reasoning":
                # Encrypted CoT from prior OpenAI turns; no chat-completions
                # equivalent. Dropping it is correct, not lossy in the sense
                # that matters — visible assistant messages already encode
                # the externally-stated conclusions of whatever the prior
                # reasoning produced.
                continue
            elif t == "compaction":
                summary = _extract_text_from_content(item.get("content")) or (
                    item.get("summary") if isinstance(item.get("summary"), str) else ""
                )
                if summary:
                    _flush_pending()
                    messages.append(
                        {
                            "role": "system",
                            "content": f"[compacted prior turns: {summary}]",
                        }
                    )
            # Unknown types are intentionally dropped. Add a branch here
            # the first time a new type surfaces in real traffic.
        _flush_pending()
    if not messages:
        messages.append({"role": "user", "content": ""})
    chat_body["model"] = body.get("model", "")
    chat_body["messages"] = messages
    return chat_body


def _chat_to_responses_response(chat: dict[str, Any]) -> dict[str, Any]:
    """Inverse of _responses_to_chat_request, on the response side.

    Translates BOTH content text AND tool_calls. Earlier versions only
    extracted message.content and dropped tool_calls on the floor —
    which manifested as Codex CLI receiving a response.completed event
    with empty output and silently displaying nothing. Tool-use prompts
    (Codex sends a tools array on every request, then the model picks
    a tool) require the function_call items in output[].
    """
    choices = chat.get("choices")
    text = ""
    thinking = ""
    tool_calls: list[dict[str, Any]] = []
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        if isinstance(message, dict):
            raw_content = message.get("content")
            if isinstance(raw_content, str):
                text = raw_content
            # Some local models (model-a0e5, model-a0g3-3, model-a0c6) emit a
            # separate `thinking` field with internal chain-of-thought.
            # Preserve it as a reasoning output item — Codex CLI / clients
            # that don't display it just ignore the item, but we never
            # silently drop content the model spent compute generating.
            raw_thinking = message.get("thinking")
            if isinstance(raw_thinking, str) and raw_thinking:
                thinking = raw_thinking
            raw_tool_calls = message.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                for tc in raw_tool_calls:
                    if isinstance(tc, dict):
                        tool_calls.append(tc)
    # Build the Responses-API output list. Reasoning (chain-of-thought
    # from thinking-mode models) comes first if present — matches
    # OpenAI's o1/o3 reasoning-summary convention. function_call items
    # come next so Codex CLI executes them in order. A message item
    # with the assistant's visible text follows. When none of these
    # are present, emit an empty message so output[] is never empty.
    output: list[dict[str, Any]] = []
    if thinking:
        output.append(
            {
                "type": "reasoning",
                "id": f"rs_{chat.get('id', 'reasoning')}",
                "summary": [{"type": "summary_text", "text": thinking}],
            }
        )
    for tc in tool_calls:
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        if not isinstance(fn, dict):
            fn = {}
        # Arguments arrive as a JSON string from ollama-tool-calling
        # convention; some serving stacks return them as dicts instead.
        # The Responses-API spec wants a string.
        args = fn.get("arguments", "{}")
        if isinstance(args, (dict, list)):
            args = json.dumps(args)
        elif not isinstance(args, str):
            args = "{}"
        call_id = tc.get("id") or fn.get("name", "") or "fc-unknown"
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{call_id}",
                "call_id": call_id,
                "name": fn.get("name", ""),
                "arguments": args,
                "status": "completed",
            }
        )
    if text or not tool_calls:
        # Emit the message even when empty if there were no tool calls,
        # so output[] is never an empty list (Codex parsers vary on
        # how strictly they require at least one item).
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        )
    # Translate usage from chat-completions shape (prompt_tokens /
    # completion_tokens) to Responses-API shape (input_tokens /
    # output_tokens). Codex CLI's stream parser hard-fails with
    # "missing field 'input_tokens'" when the response.completed event
    # lacks it, so we ALWAYS emit at least zeros — even when ollama
    # omits usage from its chat-completions reply.
    _maybe_usage = chat.get("usage")
    chat_usage: dict[str, Any] = _maybe_usage if isinstance(_maybe_usage, dict) else {}
    usage: dict[str, Any] = {
        "input_tokens": int(chat_usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(chat_usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(chat_usage.get("total_tokens", 0) or 0),
    }
    # Preserve any cache / reasoning-token sub-fields the upstream
    # included — they're optional in the Responses API but if present
    # they help downstream cost accounting.
    prompt_details = chat_usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        usage["input_tokens_details"] = prompt_details
    completion_details = chat_usage.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        usage["output_tokens_details"] = completion_details
    return {
        "id": chat.get("id", "resp-litellm"),
        "object": "response",
        "model": chat.get("model", ""),
        "status": "completed",
        "output": output,
        "usage": usage,
    }
