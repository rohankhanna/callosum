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
        ollama_unload_url: str = "http://127.0.0.1:11434",
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
        # Direct-ollama escape hatch for VRAM eviction. LiteLLM doesn't
        # forward `keep_alive` to ollama (verified empirically); to
        # unload a model we have to hit ollama itself. URL is the ollama
        # endpoint reachable from this process (typically the host
        # loopback when callosum runs on the host and ollama runs as a
        # native service or via docker -p 11434:11434).
        self._ollama_unload_url = ollama_unload_url.rstrip("/")
        # litellm-name → ollama-name cache, populated lazily via
        # LiteLLM's /model/info. Empty string in the cache means
        # "not ollama-backed" and short-circuits future lookups.
        self._ollama_name_cache: dict[str, str] = {}

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
        out_body = _strip_codex_only_fields({**body, "stream": False})
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
        out_body = _strip_codex_only_fields({**body, "stream": True})
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
        # Buffered translation: call non-stream, synthesize Responses-API SSE
        # (response.created then response.completed with the full payload).
        full = await self.responses({**body, "stream": False}, handle)
        created_event = {"type": "response.created", "id": full.get("id", "resp-litellm")}
        completed_event = {"type": "response.completed", "response": full}
        for ev in (created_event, completed_event):
            yield f"data: {json.dumps(ev)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def unload_model(self, litellm_model_name: str) -> None:
        """Force ollama to evict `litellm_model_name` from GPU memory.

        Empirically, LiteLLM does NOT pass `keep_alive` through to ollama
        — neither as a top-level field nor via `extra_body`. So we
        translate the LiteLLM model name to the ollama-side name (via
        LiteLLM's /model/info catalog) and call ollama directly with
        keep_alive=0. Ollama unloads after the no-op response.

        Used by the dispatch flow to guarantee at-most-one large local
        model resident at a time: when the router (e.g. model-a0c8)
        picks a DIFFERENT local cell (e.g. model-a0c7) for the actual
        request, callosum awaits this method to evict the router BEFORE
        the dispatch loads the chosen cell. Otherwise both co-reside
        until ollama's TTL expires, which on a constrained-VRAM box
        risks OOM.

        Best-effort. Any failure (mapping unknown, ollama unreachable,
        non-ollama upstream, etc.) is swallowed — the caller's user
        request continues. A missed eviction degrades VRAM headroom but
        doesn't break correctness.
        """
        ollama_name = await self._resolve_ollama_name(litellm_model_name)
        if ollama_name is None:
            logger.debug(
                "unload_model(%r): no ollama mapping; skipping",
                litellm_model_name,
            )
            return
        try:
            await self._client.post(
                f"{self._ollama_unload_url}/api/generate",
                json={
                    "model": ollama_name,
                    "prompt": ".",
                    "keep_alive": 0,
                    "options": {"num_predict": 1},
                },
                timeout=10.0,
            )
        except Exception as exc:
            logger.debug(
                "unload_model(%r → %r) failed (%s); VRAM eviction skipped",
                litellm_model_name, ollama_name, type(exc).__name__,
            )

    async def _resolve_ollama_name(self, litellm_model_name: str) -> str | None:
        """Look up the ollama-side model name for a LiteLLM model entry.

        Caches the mapping by querying LiteLLM's `/model/info` once and
        reusing the result; refresh happens when `_catalog_fetched_at`
        resets (the catalog TTL is a reasonable proxy for "config might
        have changed"). Returns None if the entry isn't in the catalog
        or its upstream isn't ollama-based.
        """
        cached = self._ollama_name_cache.get(litellm_model_name)
        if cached is not None:
            return cached if cached else None
        try:
            response = await self._client.get(
                f"{self._base_url}/model/info",
                headers=self._build_headers(),
                timeout=DEFAULT_HEALTH_TIMEOUT_S,
            )
        except httpx.HTTPError:
            return None
        if response.status_code != 200:
            return None
        try:
            payload = response.json()
        except json.JSONDecodeError:
            return None
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            return None
        for entry in data:
            if not isinstance(entry, dict):
                continue
            name = entry.get("model_name")
            params = entry.get("litellm_params") or {}
            model_id = params.get("model", "") if isinstance(params, dict) else ""
            if not isinstance(model_id, str):
                continue
            if model_id.startswith("ollama/"):
                ollama_side = model_id[len("ollama/"):]
                # Cache hit AND miss so we don't re-query every call.
                self._ollama_name_cache[name] = ollama_side
            else:
                # Not ollama-backed; cache empty string so future calls
                # short-circuit instead of re-querying.
                self._ollama_name_cache[name] = ""
        cached = self._ollama_name_cache.get(litellm_model_name, "")
        return cached if cached else None

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
    """Inverse of _responses_to_chat_request, on the response side."""
    choices = chat.get("choices")
    if not isinstance(choices, list) or not choices:
        text = ""
    else:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        raw = message.get("content") if isinstance(message, dict) else None
        text = raw if isinstance(raw, str) else ""
    return {
        "id": chat.get("id", "resp-litellm"),
        "object": "response",
        "model": chat.get("model", ""),
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": chat.get("usage"),
    }
