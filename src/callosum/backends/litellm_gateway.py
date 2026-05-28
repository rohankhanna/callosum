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

logger = logging.getLogger(__name__)

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

    async def chat_completions(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> dict[str, Any]:
        await self._refresh_catalog_if_stale()
        out_body = self._apply_inference_params(
            _strip_codex_only_fields({**body, "stream": False})
        )
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
        out_body = self._apply_inference_params(
            _strip_codex_only_fields({**body, "stream": True})
        )
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

    async def responses(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> dict[str, Any]:
        # LiteLLM is chat-completions-first. Translate to chat, call,
        # translate back. Sufficient for the simple text path; advanced
        # Responses-API features (file inputs, structured outputs, etc.)
        # won't survive this translation.
        chat_body = _responses_to_chat_request(body)
        chat_response = await self.chat_completions(chat_body, handle)
        return _chat_to_responses_response(chat_response)

    async def responses_stream(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> AsyncIterator[bytes]:
        """Buffered Responses-API SSE: call non-stream upstream, then
        replay as the full sequence of events Codex CLI's parser expects.

        Codex CLI's stream parser doesn't just read response.completed —
        it tracks per-item output via response.output_item.added/done
        plus item-specific events (response.output_text.delta/.done for
        messages, response.function_call_arguments.delta/.done for
        function calls). Emitting ONLY response.created + completed
        leaves the parser with an empty in-memory output list and the
        user sees nothing on screen, with no error because the stream
        IS well-formed at the wire level.

        We synthesize each intermediate event from the buffered full
        payload so a single ollama chat-completions call still produces
        a Codex-CLI-compatible stream.
        """
        full = await self.responses({**body, "stream": False}, handle)
        resp_id = full.get("id", "resp-litellm")
        seq = 0

        def _emit(event_type: str, payload: dict[str, Any]) -> bytes:
            nonlocal seq
            seq += 1
            ev = {"type": event_type, "sequence_number": seq, **payload}
            return f"event: {event_type}\ndata: {json.dumps(ev)}\n\n".encode()

        yield _emit("response.created", {"response": {**full, "status": "in_progress", "output": []}})
        yield _emit("response.in_progress", {"response": {**full, "status": "in_progress", "output": []}})

        for idx, item in enumerate(full.get("output", [])):
            if not isinstance(item, dict):
                continue
            yield _emit("response.output_item.added", {"output_index": idx, "item": item})
            item_type = item.get("type")
            item_id = item.get("id", f"item_{idx}")
            if item_type == "message":
                # Replay text as one delta + one done event.
                content_list = item.get("content") or []
                text = ""
                if isinstance(content_list, list) and content_list:
                    first = content_list[0]
                    if isinstance(first, dict) and isinstance(first.get("text"), str):
                        text = first["text"]
                yield _emit("response.output_text.delta", {
                    "item_id": item_id,
                    "output_index": idx,
                    "content_index": 0,
                    "delta": text,
                })
                yield _emit("response.output_text.done", {
                    "item_id": item_id,
                    "output_index": idx,
                    "content_index": 0,
                    "text": text,
                })
            elif item_type == "function_call":
                args = item.get("arguments", "")
                if not isinstance(args, str):
                    args = json.dumps(args)
                yield _emit("response.function_call_arguments.delta", {
                    "item_id": item_id,
                    "output_index": idx,
                    "delta": args,
                })
                yield _emit("response.function_call_arguments.done", {
                    "item_id": item_id,
                    "output_index": idx,
                    "arguments": args,
                })
            yield _emit("response.output_item.done", {"output_index": idx, "item": item})

        yield _emit("response.completed", {"response": full})
        yield b"data: [DONE]\n\n"

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
                ollama_mapping[litellm_name] = model_id[len("ollama/"):]
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
            operator_overrides, operator_force = (
                self._operator_state.get_inference_overrides(model)
            )
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
# Local backends don't understand Codex-specific request fields. The cell
# recommender writes `reasoning.effort = "<level>"` for whichever cell it
# picks (including local cells, where the synthesized effort is "default").
# LiteLLM may forward unknown fields blindly to ollama/vllm/etc., which can
# reject them or behave unpredictably. Strip the keys the local side
# definitely doesn't take before sending.

_CODEX_ONLY_BODY_KEYS = ("reasoning",)


def _strip_codex_only_fields(body: dict[str, Any]) -> dict[str, Any]:
    """Drop Codex-only request keys from a body destined for a local backend.

    Pure-fn returns a new dict; the caller's `body` is untouched. Add new
    keys to `_CODEX_ONLY_BODY_KEYS` as we discover more that local servers
    refuse — kept conservative (only `reasoning` today) to minimize the
    chance of silently dropping a field a local server actually supports.
    """
    if not any(k in body for k in _CODEX_ONLY_BODY_KEYS):
        return body
    return {k: v for k, v in body.items() if k not in _CODEX_ONLY_BODY_KEYS}


# ---------- /v1/responses ↔ /v1/chat/completions translation -------------
# Translation helpers live here rather than a shared module — they're the
# refactor if a third backend needs the same translation.


def _responses_to_chat_request(body: dict[str, Any]) -> dict[str, Any]:
    """Minimal translation of /v1/responses request body to /v1/chat/completions
    shape. Sufficient for the simple "give me text back" path.
    """
    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    input_block = body.get("input")
    if isinstance(input_block, str):
        messages.append({"role": "user", "content": input_block})
    elif isinstance(input_block, list):
        for item in input_block:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            role = item.get("role", "user")
            content = item.get("content")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and isinstance(p.get("text"), str)
                ]
                text = "".join(parts)
            messages.append({"role": role, "content": text})
    if not messages:
        messages.append({"role": "user", "content": ""})
    chat_body: dict[str, Any] = {
        "model": body.get("model", ""),
        "messages": messages,
    }
    for key in ("temperature", "max_tokens", "top_p", "tools", "tool_choice"):
        if key in body:
            chat_body[key] = body[key]
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
    tool_calls: list[dict[str, Any]] = []
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        if isinstance(message, dict):
            raw_content = message.get("content")
            if isinstance(raw_content, str):
                text = raw_content
            raw_tool_calls = message.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                for tc in raw_tool_calls:
                    if isinstance(tc, dict):
                        tool_calls.append(tc)
    # Build the Responses-API output list. function_call items come
    # first (Codex CLI executes them, then submits results back); a
    # message item with the assistant's text follows. When neither is
    # present, emit an empty message — Codex CLI accepts that as
    # "response complete with no output" rather than failing to parse.
    output: list[dict[str, Any]] = []
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
        output.append({
            "type": "function_call",
            "id": f"fc_{call_id}",
            "call_id": call_id,
            "name": fn.get("name", ""),
            "arguments": args,
            "status": "completed",
        })
    if text or not tool_calls:
        # Emit the message even when empty if there were no tool calls,
        # so output[] is never an empty list (Codex parsers vary on
        # how strictly they require at least one item).
        output.append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        })
    # Translate usage from chat-completions shape (prompt_tokens /
    # completion_tokens) to Responses-API shape (input_tokens /
    # output_tokens). Codex CLI's stream parser hard-fails with
    # "missing field 'input_tokens'" when the response.completed event
    # lacks it, so we ALWAYS emit at least zeros — even when ollama
    # omits usage from its chat-completions reply.
    chat_usage = chat.get("usage") if isinstance(chat.get("usage"), dict) else {}
    usage = {
        "input_tokens": int(chat_usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(chat_usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(chat_usage.get("total_tokens", 0) or 0),
    }
    # Preserve any cache / reasoning-token sub-fields the upstream
    # included — they're optional in the Responses API but if present
    # they help downstream cost accounting.
    if isinstance(chat_usage.get("prompt_tokens_details"), dict):
        usage["input_tokens_details"] = chat_usage["prompt_tokens_details"]
    if isinstance(chat_usage.get("completion_tokens_details"), dict):
        usage["output_tokens_details"] = chat_usage["completion_tokens_details"]
    return {
        "id": chat.get("id", "resp-litellm"),
        "object": "response",
        "model": chat.get("model", ""),
        "status": "completed",
        "output": output,
        "usage": usage,
    }
