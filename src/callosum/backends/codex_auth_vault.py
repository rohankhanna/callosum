"""REDUNDANT-CODE: see docs/architecture/redundant_code.md

This backend is no longer exercised by the default production config —
the active path is `callosum.backends.credential_proxy`, which routes
OAuth refresh through the credential proxy service. This module is kept
deliberately as a fallback the operator can activate without code
changes if credential proxy is unreachable: flip a backend's `type` in
config.toml from `credential_proxy` to `codex_auth_vault`, point
`vault_path` at a fresh auth.json, restart.

The tests in `tests/unit/test_codex_auth_vault_backend.py` continue
to run on every CI sweep so the fallback stays working. Behavior
changes that affect this backend require updating those tests, same
as for any active backend.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from callosum.cell_grid import ModelMetadata

import httpx

from callosum.auth_vault import AuthVault, _default_headers
from callosum.backend import BackendKind, CallHandle, HealthStatus, UsageSnapshot
from callosum.backends._http import DEFAULT_COOLDOWN_S, error_from_response
from callosum.codex_quota import parse_codex_headers
from callosum.errors import BackendError
from callosum.sse_tee import (
    ResponsesStreamCollector,
    assemble_completed_with_text,
    namespaced_tool_names_from_request,
    strip_namespace_stream,
)
from callosum.state import StateStore

DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
RESPONSES_BETA_HEADER_VALUE = "responses=v1"

# Codex Responses API requires a non-empty `instructions` field. When a
# client sends a chat-completions body with no system message, we fall
# back to this short placeholder so the translated /responses request
# isn't rejected with "Instructions are required".
_DEFAULT_INSTRUCTIONS = "You are a helpful assistant."

# How long to keep a fetched model catalog before refreshing. The Codex model
# lineup changes monthly (and trending toward weekly per operator), so an
# hourly refresh keeps us current without hammering the upstream.
DEFAULT_MODELS_REFRESH_S = 3600.0

# The /backend-api/codex/models endpoint requires a `client_version` query
# parameter or it 400s. We resolve this dynamically per-call via
# `_resolve_codex_client_version()` so the param tracks the operator's
# installed codex CLI without manual edits when codex updates. Hardcoded
# only as a last-resort fallback when neither env nor the codex version
# file is available.
_DEFAULT_CLIENT_VERSION = "0.125.0"


def _resolve_codex_client_version() -> str:
    """Return the value to use for the `client_version` query param.

    Resolution order (first hit wins):

    1. `CODEX_CLIENT_VERSION` env var — operator override / pinning.
    2. `~/.codex/version.json`'s `latest_version` field — set by the
       codex CLI's auto-update check, so it tracks the local install.
    3. `_DEFAULT_CLIENT_VERSION` — known-working fallback.

    Best-effort: any IO/parse error falls through, never raises.
    """
    override = os.environ.get("CODEX_CLIENT_VERSION")
    if override:
        return override
    try:
        path = Path.home() / ".codex" / "version.json"
        with path.open("r") as f:
            payload = json.load(f)
        version = payload.get("latest_version") if isinstance(payload, dict) else None
        if isinstance(version, str) and version:
            return version
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return _DEFAULT_CLIENT_VERSION


class CodexAuthVaultBackend:
    """Uses an on-disk Codex `auth.json` to hit the ChatGPT backend Responses API.

    Clients talk to this backend via the standard OpenAI chat-completions shape;
    requests are translated into the Responses API format, forwarded with the
    access token, and translated back.
    """

    kind: BackendKind = "codex_auth_vault"

    def __init__(
        self,
        *,
        id: str,
        vault: AuthVault,
        advertised_models: frozenset[str] = frozenset(),
        base_url: str = DEFAULT_BASE_URL,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = 60.0,
        state_store: StateStore | None = None,
        models_refresh_s: float = DEFAULT_MODELS_REFRESH_S,
    ) -> None:
        # `advertised_models` is OPTIONAL — see CredentialProxyBackend
        # for the rationale. Dynamic discovery via
        # refresh_advertised_models is the source of truth; the static
        # set is just a cold-start fallback. Empty is legal.
        self.id = id
        # `_static_advertised_models` is the operator's TOML override / cold-
        # start fallback. `_dynamic_advertised_models` is what we last fetched
        # from upstream's /models endpoint (None until first fetch). The
        # `advertised_models` property prefers dynamic when available.
        self._static_advertised_models: frozenset[str] = advertised_models
        self._dynamic_advertised_models: frozenset[str] | None = None
        self._model_context_windows: dict[str, int] = {}  # slug → context_window tokens
        # Full per-model metadata from the upstream catalog (ModelMetadata
        # records). Empty when we haven't fetched yet OR when the upstream
        # response omitted the fields. Callers fall back to local defaults.
        self._model_metadata: dict[str, ModelMetadata] = {}
        self._models_fetched_at: float = 0.0
        self._models_refresh_s = models_refresh_s
        self._vault = vault
        self._base_url = base_url.rstrip("/")
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(
                transport=transport,
                timeout=timeout_s,
                headers=_default_headers(),
            )
            self._owns_client = True
        self._state_store = state_store
        loaded = state_store.load_usage(id) if state_store is not None else None
        self._usage = loaded or UsageSnapshot(
            remaining_fraction=None,
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=time.time(),
        )
        # Most recent quota snapshot observed from upstream response headers.
        # Becomes `quota_before` on the next call so the logging layer can
        # compute Δquota for a request.
        self._last_quota: Any = None  # CodexQuotaSnapshot | None, loose typed to avoid import cycle noise
        # Offline/network-failure tracking. When responses() or
        # responses_stream() raise transport-layer errors (httpx.HTTPError —
        # ConnectError, ConnectTimeout, NameResolutionError, etc.) we
        # bump a consecutive-failures counter. Once it hits the threshold,
        # usage_snapshot() reports a cooldown until the next successful call
        # resets it. That lets `_filter_cells_to_routable` correctly exclude
        # this backend from the routable set when the operator is offline,
        # so the recommender's heuristic-tier fallback can pick a local cell
        # instead of repeatedly trying a Codex cell whose request will hit
        # NXDOMAIN. Mirrors the LiteLLM gateway's _healthy flag.
        self._consecutive_transport_failures: int = 0
        self._transport_cooldown_until_ts: float = 0.0
        self._transport_failure_threshold: int = 3
        self._transport_cooldown_seconds: float = 30.0
        # Warm-start the model catalog from the last-known-good persisted copy
        # so a cold boot (upstream/auth slow/down) starts with a populated
        # catalog instead of an empty-catalog window. The startup refresh
        # overwrites this on success; on failure we keep serving from it.
        # Parity with CredentialProxyBackend. See  (layer 2).
        self._warm_start_catalog()

    @property
    def advertised_models(self) -> frozenset[str]:
        """Return the most recently fetched upstream model list, falling back
        to the operator-provided static set when we haven't fetched yet (cold
        start) or when the fetch failed.
        """
        if self._dynamic_advertised_models is not None:
            return self._dynamic_advertised_models
        return self._static_advertised_models

    @property
    def model_context_windows(self) -> dict[str, int]:
        """Return the most recently fetched context windows per model. Empty dict
        if not yet fetched or if the upstream API doesn't include context_length.
        """
        return self._model_context_windows

    @property
    def model_metadata(self):  # type: ignore[no-untyped-def]
        """Return the most recently fetched per-model metadata (ModelMetadata
        records keyed by slug). Empty dict if not yet fetched or the upstream
        response was missing the relevant fields. Loose-typed in the
        signature to avoid an import cycle; callers should treat values as
        callosum.cell_grid.ModelMetadata.
        """
        return self._model_metadata

    def _warm_start_catalog(self) -> None:
        """Seed the dynamic catalog from the persisted last-known-good copy.

        Populates `_dynamic_advertised_models` / `_model_context_windows` /
        `_model_metadata` from disk so a cold boot starts with a populated
        catalog (preferred over the static TOML hint) even when the upstream
        /models fetch is slow or failing. `_models_fetched_at` is left at 0.0
        so the startup refresh's freshness gate does NOT skip — the live
        refresh must still run and overwrite this hint on success. On a failed
        refresh the warm-start set is retained (refresh only overwrites when
        it gets models), so routing keeps working off the hint rather than
        going empty.

        The persisted blob is untrusted disk state: validate every field on
        load and fall back to empty (→ static hint) on any shape problem.
        Mirrors CredentialProxyBackend._warm_start_catalog for parity.
        """
        if self._state_store is None:
            return
        persisted = self._state_store.load_catalog(self.id)
        if not isinstance(persisted, dict):
            return
        raw_models = persisted.get("advertised_models")
        if not isinstance(raw_models, list):
            return
        slugs = frozenset(m for m in raw_models if isinstance(m, str) and m)
        if not slugs:
            return
        raw_ctx = persisted.get("context_windows")
        ctx_map: dict[str, int] = {}
        if isinstance(raw_ctx, dict):
            for slug, ctx in raw_ctx.items():
                if isinstance(slug, str) and isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0:
                    ctx_map[slug] = ctx
        raw_meta = persisted.get("model_metadata")
        meta_map: dict[str, ModelMetadata] = {}
        if isinstance(raw_meta, dict):
            from callosum.cell_grid import model_metadata_from_dict

            for slug, md_raw in raw_meta.items():
                if not isinstance(slug, str):
                    continue
                md = model_metadata_from_dict(md_raw)
                if md is not None:
                    meta_map[slug] = md
        self._dynamic_advertised_models = slugs
        self._model_context_windows = ctx_map
        self._model_metadata = meta_map
        # Leave _models_fetched_at=0.0 so the startup refresh is NOT skipped
        # by the freshness gate (ts - 0.0 is always past models_refresh_s).
        self._models_fetched_at = 0.0

    def _persist_catalog(self, *, fetched_at: float) -> None:
        """Write the current dynamic catalog to disk as the next boot's hint.

        Called after a successful refresh so the next cold boot warm-starts
        from this live copy. Overwrites the previous blob so a model retired
        upstream eventually drops from the hint. Mirrors
        CredentialProxyBackend._persist_catalog for parity.
        """
        if self._state_store is None or self._dynamic_advertised_models is None:
            return
        from callosum.cell_grid import model_metadata_to_dict

        self._state_store.save_catalog(
            self.id,
            {
                "advertised_models": sorted(self._dynamic_advertised_models),
                "context_windows": dict(self._model_context_windows),
                "model_metadata": {slug: model_metadata_to_dict(md) for slug, md in self._model_metadata.items()},
                "fetched_at": fetched_at,
            },
        )

    def cell_capabilities(self, model: str):  # type: ignore[no-untyped-def]
        """Return CellCapabilities for `model` — what the learning router
        needs to decide whether this cell can serve a given request.

        Sources from the most recent ModelMetadata when populated; falls
        back to sensible Codex defaults when fields are absent (cold start
        before first catalog poll, or upstream omits the field). The
        Codex Responses API supports tools across all serving models, so
        `supports_tools=True` is the conservative default. Cost rank is
        hardcoded high for remote backends (auto-inferred from kind in
        Phase 1; explicit per-cell ranks can be layered later).

        Returns the type lazily-imported to avoid a circular dependency
        at module load (routing.protocols imports cell_grid; backends
        import routing.protocols only on first call to this method).
        """
        from callosum.routing.protocols import CellCapabilities

        md = self._model_metadata.get(model)
        # Codex's typical max context. Updated only if a model metadata
        # record explicitly says otherwise.
        ctx = md.context_window if md and md.context_window else 256_000
        # Modalities: ModelMetadata.input_modalities is a tuple of strings
        # already normalized to {"text", "image", "audio", ...} by
        # _extract_model_catalog. Default to text-only when absent.
        modalities = frozenset(md.input_modalities) if md and md.input_modalities else frozenset({"text"})
        return CellCapabilities(
            context_window=ctx,
            modalities=modalities,
            supports_tools=True,  # Codex Responses API supports tools.
            # Remote cold-start fallback (> local rank 0). The live ordering
            # is overlaid per-model from MEASURED weekly-quota burn by
            # callosum.routing.cost_model.CostRankProvider; this constant is
            # only used until that provider has data for the model.
            cost_rank=10,
        )

    async def refresh_advertised_models(self, *, now: float | None = None) -> None:
        """Fetch the upstream model catalog for this account and update the
        cached set. Best-effort: silently keeps the current set on any error
        (network down, auth invalid, malformed payload). Skip if the cached
        copy is younger than `models_refresh_s`.
        """
        ts = now if now is not None else time.time()
        if self._dynamic_advertised_models is not None and ts - self._models_fetched_at < self._models_refresh_s:
            return
        try:
            tokens = await self._vault.current()
        except BackendError:
            return
        try:
            response = await self._client.get(
                f"{self._base_url}/models",
                params={"client_version": _resolve_codex_client_version()},
                headers=self._build_headers(tokens.access_token, tokens.account_id),
            )
        except httpx.HTTPError:
            return
        if response.status_code != 200:
            # Log the upstream's complaint so future operators can debug
            # without instrumenting the code. Best-effort still: don't raise.
            detail = ""
            try:
                detail = response.text[:200]
            except Exception:
                detail = "(no body)"
            logging.getLogger("callosum.backend").warning(
                "models discovery failed for backend %r: HTTP %d %s",
                self.id,
                response.status_code,
                detail,
            )
            return
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError):
            return
        models, context_windows, metadata = _extract_model_catalog(payload)
        if models:
            self._dynamic_advertised_models = models
            self._model_context_windows = context_windows
            self._model_metadata = metadata
            self._models_fetched_at = ts
            self._persist_catalog(fetched_at=ts)

    async def health(self) -> HealthStatus:
        return HealthStatus(available=True, reason="ok")

    def clear_cooldown(self) -> UsageSnapshot:
        """Reset cooldown and weekly-exhausted flags to a fresh state.

        The dispatcher excludes any backend whose cooldown_until_ts hasn't
        passed, and the diagnostic / smoke-test paths deliberately skip
        cooldown'd backends. Together those create a chicken-and-egg lockout
        when the persisted cooldown becomes stale (e.g. upstream's
        weekly_reset_at was inaccurate, account quota was topped up out of
        band, or the original 429 was a transient mis-classification).

        Called by the periodic cooldown prober when its probe succeeds, and
        by the /control/clear-cooldown admin endpoint as an explicit operator
        override. Updates the persisted snapshot so the cleared state
        survives proxy restart.
        """
        self._usage = UsageSnapshot(
            remaining_fraction=self._usage.remaining_fraction,
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=time.time(),
        )
        if self._state_store is not None:
            self._state_store.save_usage(self.id, self._usage)
        return self._usage

    def _on_transport_failure(self) -> None:
        """Record an httpx-level transport failure. After threshold consecutive
        failures, set a short cooldown so usage_snapshot reports this backend
        unroutable. Reset by the next successful round-trip in
        `_on_transport_success`. Critical for offline failover: without this,
        an offline operator's Codex backend keeps looking routable to
        `_filter_cells_to_routable`, and the recommender keeps picking it
        even though every request will hit NXDOMAIN.
        """
        self._consecutive_transport_failures += 1
        if self._consecutive_transport_failures >= self._transport_failure_threshold:
            self._transport_cooldown_until_ts = time.time() + self._transport_cooldown_seconds

    def _on_transport_success(self) -> None:
        """Successful HTTP round-trip — clear the offline indicators so this
        backend becomes routable again. Even one successful call is enough
        evidence that the network path is back."""
        self._consecutive_transport_failures = 0
        self._transport_cooldown_until_ts = 0.0

    async def usage_snapshot(self) -> UsageSnapshot:
        # Derive `weekly_exhausted` PURELY from the most recent upstream
        # quota snapshot — no sticky propagation. If upstream's most recent
        # reading says weekly is below the threshold, the flag is False
        # even if a prior 429 had set it True. Otherwise the proxy could
        # demote a backend for a whole week based on one early-in-the-
        # week 429 with a momentarily-high reading.
        #
        # Threshold is 99% rather than 100% because upstream's reported
        # percent is integer-truncated, and any further organic burn
        # while we wait pushes us over the next request's quota threshold.
        #
        # Fallback to the persisted snapshot only when we have no fresh
        # quota reading yet (cold start before the first call returns).
        if self._last_quota is not None and self._last_quota.weekly_used_percent is not None:
            weekly_exhausted = self._last_quota.weekly_used_percent >= 99
        else:
            weekly_exhausted = self._usage.weekly_exhausted
        # Transport-layer offline override: if our recent calls have been
        # failing at the httpx level (DNS, connect refused, etc.), surface
        # that as a cooldown so the routability filter excludes us.
        # Overrides whatever cooldown_until_ts the in-memory snapshot
        # already carries; the larger of the two wins so genuine
        # upstream-imposed cooldowns aren't shortened by transient
        # network blips.
        now = time.time()
        snap_cooldown = self._usage.cooldown_until_ts
        effective_cooldown = snap_cooldown
        if self._transport_cooldown_until_ts > now and (
            effective_cooldown is None or self._transport_cooldown_until_ts > effective_cooldown
        ):
            effective_cooldown = self._transport_cooldown_until_ts
        if weekly_exhausted == self._usage.weekly_exhausted and effective_cooldown == self._usage.cooldown_until_ts:
            return self._usage
        return UsageSnapshot(
            remaining_fraction=self._usage.remaining_fraction,
            cooldown_until_ts=effective_cooldown,
            weekly_exhausted=weekly_exhausted,
            probed_at_ts=self._usage.probed_at_ts,
        )

    async def quota_snapshot(self) -> Any:  # CodexQuotaSnapshot | None
        return self._last_quota

    async def chat_completions(self, body: dict[str, Any], handle: CallHandle | None = None) -> dict[str, Any]:
        requested_model = _require_model(body)
        responses_payload = _chat_to_responses_request(body, stream=False)
        # The chat path is implemented as a translated /responses call; pass
        # the handle through so quota + header observations still land on it.
        upstream = await self.responses(responses_payload, handle)
        return _responses_to_chat_response(upstream, model=requested_model)

    async def chat_completions_stream(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> AsyncIterator[bytes]:
        # Buffered translation: call upstream non-streaming, then synthesise a
        # chat-completions SSE sequence. This keeps the translation simple and
        # avoids parsing upstream Responses-API events. Native streaming is
        # intentionally deferred to the /v1/responses route, which forwards
        # upstream SSE untouched.
        completion = await self.chat_completions({**body, "stream": False}, handle)
        completion_id = _str_or(completion.get("id"), "chatcmpl-codex")
        model = _str_or(completion.get("model"), "")
        created = _int_or(completion.get("created"), int(time.time()))
        choices = completion.get("choices")
        if not isinstance(choices, list) or not choices:
            content = ""
            finish_reason = "stop"
        else:
            first = choices[0] if isinstance(choices[0], dict) else {}
            message = first.get("message") if isinstance(first.get("message"), dict) else {}
            raw_content = message.get("content") if isinstance(message, dict) else None
            content = raw_content if isinstance(raw_content, str) else ""
            finish_reason_raw = first.get("finish_reason")
            finish_reason = finish_reason_raw if isinstance(finish_reason_raw, str) else "stop"

        def frame(delta: dict[str, Any], finish: str | None) -> bytes:
            payload = {
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
            return f"data: {json.dumps(payload)}\n\n".encode()

        yield frame({"role": "assistant"}, None)
        if content:
            yield frame({"content": content}, None)
        yield frame({}, finish_reason)
        yield b"data: [DONE]\n\n"

    async def responses(self, body: dict[str, Any], handle: CallHandle | None = None) -> dict[str, Any]:
        # Codex Responses API now requires `stream: true` for ALL requests
        # (returns 400 "Stream must be set to true" otherwise). For callers
        # that want a non-streaming dict response, we send stream=true to
        # upstream, collect the SSE via ResponsesStreamCollector, and return
        # the buffered response.completed payload — invisible to the caller.
        if handle is not None:
            handle.quota_before = self._last_quota
        streaming_body = {**body, "stream": True}
        # Strip input-item fields the ChatGPT `/codex/responses` endpoint
        # rejects (notably `namespace` on `custom_tool_call` items the Codex
        # CLI emits) — same normalization the active credential_proxy path
        # applies. See callosum.backends.credential_proxy.
        from callosum.backends.credential_proxy import _strip_unsupported_input_fields

        _strip_unsupported_input_fields(streaming_body)
        for attempt in (1, 2):
            tokens = await self._vault.current() if attempt == 1 else await self._vault.force_refresh()
            headers = self._build_headers(tokens.access_token, tokens.account_id, accept_event_stream=True)
            try:
                stream_ctx = self._client.stream(
                    "POST",
                    f"{self._base_url}/responses",
                    json=streaming_body,
                    headers=headers,
                )
                async with stream_ctx as response:
                    if attempt == 1 and response.status_code == 401:
                        await response.aread()
                        continue  # retry with refresh
                    self._apply_response_to_handle(response.headers, response.status_code, handle)
                    if response.status_code >= 400:
                        await response.aread()
                        err = error_from_response(response)
                        self._apply_error_to_usage(err)
                        raise err
                    namespaced_names = namespaced_tool_names_from_request(body)
                    collector = ResponsesStreamCollector(
                        strip_namespace_stream(response.aiter_bytes(), namespaced_names)
                    )
                    async for _chunk in collector.iter_through():
                        pass  # buffer the whole stream
                    if handle is not None:
                        handle.stream_summary = collector.summary
                    # The `response.completed` event ships with `output:[]`
                    # — visible text only lives in the streamed `.delta`/
                    # `.done` events. Assemble the visible text and inject
                    # it into output[] so non-stream callers (e.g. the
                    # cell recommender, internal tests) get a dict whose
                    # output[].content[].text is actually populated.
                    completed = assemble_completed_with_text(collector.summary.raw_blob)
                    if completed is None:
                        raise BackendError(
                            classification="transient",
                            message="upstream stream ended without response.completed event",
                        )
                    # Successful round-trip: reset the offline tracker.
                    self._on_transport_success()
                    return completed
            except httpx.HTTPError as exc:
                self._on_transport_failure()
                raise BackendError(classification="transient", message=str(exc)) from exc
        # Unreachable: loop either returns or raises on attempt 2.
        raise BackendError(classification="auth_invalid", message="auth retry exhausted")

    async def responses_stream(self, body: dict[str, Any], handle: CallHandle | None = None) -> AsyncIterator[bytes]:
        if handle is not None:
            handle.quota_before = self._last_quota
        # Open the stream; on upstream 401 in the response headers, close and
        # restart with refreshed tokens. We can only retry before any chunk
        # has been yielded — once streaming starts, we're committed.
        # Strip upstream-rejected input-item fields (notably `namespace` on
        # `custom_tool_call` items) before forwarding — same normalization as
        # the active credential_proxy path. See callosum.backends.credential_proxy.
        from callosum.backends.credential_proxy import _strip_unsupported_input_fields

        _strip_unsupported_input_fields(body)
        for attempt in (1, 2):
            tokens = await self._vault.current() if attempt == 1 else await self._vault.force_refresh()
            headers = self._build_headers(tokens.access_token, tokens.account_id, accept_event_stream=True)
            try:
                stream_ctx = self._client.stream(
                    "POST",
                    f"{self._base_url}/responses",
                    json=body,
                    headers=headers,
                )
                async with stream_ctx as response:
                    if attempt == 1 and response.status_code == 401:
                        await response.aread()
                        continue  # retry with refresh
                    self._apply_response_to_handle(response.headers, response.status_code, handle)
                    if response.status_code >= 400:
                        await response.aread()
                        err = error_from_response(response)
                        self._apply_error_to_usage(err)
                        raise err
                    namespaced_names = namespaced_tool_names_from_request(body)
                    collector = ResponsesStreamCollector(
                        strip_namespace_stream(response.aiter_bytes(), namespaced_names)
                    )
                    async for chunk in collector.iter_through():
                        yield chunk
                    if handle is not None:
                        handle.stream_summary = collector.summary
                    # Successful round-trip: reset the offline tracker.
                    self._on_transport_success()
                    return
            except httpx.HTTPError as exc:
                self._on_transport_failure()
                raise BackendError(classification="transient", message=str(exc)) from exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
        await self._vault.aclose()

    def _build_headers(
        self,
        access_token: str,
        account_id: str | None,
        *,
        accept_event_stream: bool = False,
    ) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if accept_event_stream else "application/json",
            "OpenAI-Beta": RESPONSES_BETA_HEADER_VALUE,
            "originator": "codex_cli_rs",
            # ChatGPT routes certain models by originator + `version`
            # header. model-a0c4's metadata sets
            # minimal_client_version=0.144.0; without a `version` header
            # ≥ that floor luna 404s "Model not found" while sol/terra
            # serve fine. The official Codex CLI sends this header;
            # mirror it. openai/codex#31967.
            "version": _resolve_codex_client_version(),
        }
        if account_id:
            headers["chatgpt-account-id"] = account_id
        return headers

    def _apply_response_to_handle(
        self,
        headers: Any,
        status_code: int,
        handle: CallHandle | None,
    ) -> None:
        snapshot = parse_codex_headers(dict(headers))
        if snapshot is not None:
            self._last_quota = snapshot
        if handle is not None:
            handle.upstream_status = status_code
            handle.upstream_headers = dict(headers)
            handle.quota_after = snapshot

    def _apply_error_to_usage(self, err: BackendError) -> None:
        if err.classification != "rate_limited":
            return
        now = time.time()
        quota = self._last_quota

        # Determine cooldown duration based on quota state
        if err.retry_after_s:
            cooldown_until = now + err.retry_after_s
        elif quota is not None and quota.weekly_used_percent is not None and quota.weekly_used_percent >= 99:
            # Weekly exhausted — cooldown until weekly reset (usually 44+ hours)
            cooldown_until = float(quota.weekly_reset_at) if quota.weekly_reset_at else now + 7 * 86400
        elif quota is not None and quota.five_hourly_used_percent is not None and quota.five_hourly_used_percent >= 95:
            # 5-hour window exhausted — cooldown until 5h reset
            cooldown_until = float(quota.five_hourly_reset_at) if quota.five_hourly_reset_at else now + 5 * 3600
        else:
            cooldown_until = now + DEFAULT_COOLDOWN_S

        # Determine weekly_exhausted flag — purely from the current quota
        # snapshot, no sticky propagation. If quota is missing, fall back
        # to whatever the persisted snapshot said (cold start). See
        # usage_snapshot() for the rationale: a sticky flag wrongly
        # demotes a backend for the rest of the week based on one
        # momentarily-high reading.
        if quota is not None and quota.weekly_used_percent is not None:
            weekly_exhausted = quota.weekly_used_percent >= 99
        else:
            weekly_exhausted = self._usage.weekly_exhausted

        self._usage = UsageSnapshot(
            remaining_fraction=self._usage.remaining_fraction,
            cooldown_until_ts=cooldown_until,
            weekly_exhausted=weekly_exhausted,
            probed_at_ts=now,
        )
        if self._state_store is not None:
            self._state_store.save_usage(self.id, self._usage)
        _log_cooldown_set(self.id, cooldown_until, quota)


def _extract_model_catalog(
    payload: Any,
) -> tuple[frozenset[str], dict[str, int], dict[str, ModelMetadata]]:
    """Pull model slugs, context windows, and full per-model metadata from an
    upstream `/backend-api/codex/models` response.

    Tolerates two shapes the upstream has shipped over time:
      {"models": [{"slug": ..., "context_window": ..., ...}]}
      {"data":   [{"id":   ..., "context_length": ..., ...}]}

    Returns (frozenset of slugs, dict of slug→context_window, dict of
    slug→ModelMetadata). Every metadata field is defensive — only `slug`
    is required, everything else falls back to None / empty when absent.
    Returns empty results on malformed payload; callers treat empty as
    "fall back to the cold-start static set."
    """
    from callosum.cell_grid import ModelMetadata

    if not isinstance(payload, dict):
        return frozenset(), {}, {}
    items: Any = payload.get("models")
    if not isinstance(items, list):
        items = payload.get("data")
    if not isinstance(items, list):
        return frozenset(), {}, {}
    slugs: set[str] = set()
    context_windows: dict[str, int] = {}
    metadata: dict[str, ModelMetadata] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug:
            slug = item.get("id")
        if not isinstance(slug, str) or not slug:
            continue
        slugs.add(slug)
        # context_window (new shape) or context_length (legacy / OpenAI shape)
        ctx_len = item.get("context_window")
        if not isinstance(ctx_len, int) or ctx_len <= 0:
            ctx_len = item.get("context_length")
        if isinstance(ctx_len, int) and ctx_len > 0:
            context_windows[slug] = ctx_len

        # supported_reasoning_levels: list[{"effort": "low", ...}] → tuple[str, ...]
        levels_raw = item.get("supported_reasoning_levels")
        levels: tuple[str, ...] = ()
        if isinstance(levels_raw, list):
            picked: list[str] = []
            for lvl in levels_raw:
                if isinstance(lvl, dict):
                    eff = lvl.get("effort")
                    if isinstance(eff, str) and eff:
                        picked.append(eff)
                elif isinstance(lvl, str) and lvl:
                    picked.append(lvl)
            levels = tuple(picked)

        modalities_raw = item.get("input_modalities")
        modalities: tuple[str, ...] = ()
        if isinstance(modalities_raw, list):
            modalities = tuple(m for m in modalities_raw if isinstance(m, str) and m)

        # Build the metadata record — every field optional except slug.
        metadata[slug] = ModelMetadata(
            slug=slug,
            display_name=item.get("display_name") if isinstance(item.get("display_name"), str) else None,
            description=item.get("description") if isinstance(item.get("description"), str) else None,
            context_window=context_windows.get(slug),
            supported_in_api=item.get("supported_in_api") if isinstance(item.get("supported_in_api"), bool) else None,
            visibility=item.get("visibility") if isinstance(item.get("visibility"), str) else None,
            priority=item.get("priority") if isinstance(item.get("priority"), int) else None,
            default_reasoning_level=item.get("default_reasoning_level")
            if isinstance(item.get("default_reasoning_level"), str)
            else None,
            supported_reasoning_levels=levels,
            input_modalities=modalities,
        )
    return frozenset(slugs), context_windows, metadata


def _extract_model_slugs(payload: Any) -> frozenset[str]:
    """Pull model slug strings out of an upstream `/backend-api/codex/models`
    response. Deprecated: use _extract_model_catalog instead to get context windows.
    Kept for backward compat. Returns an empty frozenset on anything malformed.
    """
    slugs, _, _ = _extract_model_catalog(payload)
    return slugs


def _require_model(body: dict[str, Any]) -> str:
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise BackendError(
            classification="client_error",
            status_code=400,
            message="codex_auth_vault requires a string 'model' in the request body",
        )
    return model


def _chat_to_responses_request(body: dict[str, Any], *, stream: bool) -> dict[str, Any]:
    messages = body.get("messages")
    input_items: list[dict[str, Any]] = []
    instructions: str | None = None
    if isinstance(messages, list):
        for raw in messages:
            if not isinstance(raw, dict):
                continue
            role = raw.get("role")
            if role == "system":
                content_text = _content_as_text(raw.get("content"))
                instructions = content_text if instructions is None else f"{instructions}\n{content_text}"
                continue
            if role in ("user", "assistant"):
                input_items.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": [
                            {
                                "type": "input_text" if role == "user" else "output_text",
                                "text": _content_as_text(raw.get("content")),
                            }
                        ],
                    }
                )
    payload: dict[str, Any] = {
        "model": body["model"],
        "input": input_items,
        "stream": stream,
        # Codex Responses API requires both `instructions` (non-empty) and
        # `store: False` or it returns 400. Forward whatever the caller
        # provided, then backfill defaults below if absent — clients that
        # send chat-completions bodies (hermes, codex CLI's translator,
        # most OAI-compatible tools) typically omit both.
        "store": False,
    }
    payload["instructions"] = instructions if instructions else _DEFAULT_INSTRUCTIONS
    for key in ("temperature", "top_p", "max_output_tokens", "metadata", "store"):
        if key in body:
            payload[key] = body[key]
    if "max_tokens" in body and "max_output_tokens" not in payload:
        payload["max_output_tokens"] = body["max_tokens"]
    return payload


def _content_as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _responses_to_chat_finish_reason(payload: dict[str, Any]) -> str:
    """Map a Responses-API payload's `status` (+ incomplete_details.reason)
    into the chat-completions `finish_reason` field. Without this mapping
    the chat shell silently reports 'stop' for truncated responses,
    masking budget-eaten generations as natural completions —
    misleading for clients that distinguish 'stop' from 'length'
    (Codex CLI, fitness analyzers, anything counting completed-vs-
    cut-off rates).

    Mapping:
      status == 'completed'                          -> 'stop'
      status == 'incomplete' + max_output_tokens     -> 'length'
      status == 'incomplete' + content_filter        -> 'content_filter'
      status == 'incomplete' + (other or missing)    -> 'length'
      status == 'failed' / anything unexpected       -> 'stop'

    'stop' is the safest unknown-case default because Codex CLI's
    retry behavior on it is benign; mapping unknowns to 'length' would
    cause spurious "truncated" framing in client UIs.
    """
    status = payload.get("status")
    if status == "incomplete":
        details = payload.get("incomplete_details") or {}
        reason = details.get("reason") if isinstance(details, dict) else None
        if reason == "content_filter":
            return "content_filter"
        # max_output_tokens or any other / missing reason: budget truncation
        return "length"
    if status == "completed":
        return "stop"
    return "stop"


def _responses_to_chat_response(payload: dict[str, Any], *, model: str) -> dict[str, Any]:
    text_parts: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            for block in item.get("content", []) if isinstance(item.get("content"), list) else []:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                text = block.get("text")
                if block_type in ("output_text", "text") and isinstance(text, str):
                    text_parts.append(text)
    content = "".join(text_parts)
    completion_id = _str_or(payload.get("id"), f"chatcmpl-{uuid.uuid4().hex}")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
    chat: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion",
        "created": _int_or(payload.get("created_at"), int(time.time())),
        "model": _str_or(payload.get("model"), model),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": _responses_to_chat_finish_reason(payload),
            }
        ],
    }
    if usage is not None:
        chat["usage"] = _normalize_usage(usage)
    return chat


def _normalize_usage(usage: dict[str, Any]) -> dict[str, Any]:
    prompt = usage.get("input_tokens")
    completion = usage.get("output_tokens")
    total = usage.get("total_tokens")
    out: dict[str, Any] = {}
    if isinstance(prompt, int):
        out["prompt_tokens"] = prompt
    if isinstance(completion, int):
        out["completion_tokens"] = completion
    if isinstance(total, int):
        out["total_tokens"] = total
    elif isinstance(prompt, int) and isinstance(completion, int):
        out["total_tokens"] = prompt + completion
    return out


def _str_or(value: Any, default: str) -> str:
    return value if isinstance(value, str) and value else default


def _int_or(value: Any, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _log_cooldown_set(backend_id: str, cooldown_until: float, quota: Any) -> None:
    """Log when a cooldown is set, with details about the quota state.

    Emits a single clear line showing backend ID, reason for cooldown, recovery time, and duration.
    """
    logger = logging.getLogger("callosum.backend")
    now = time.time()
    recovery_seconds = cooldown_until - now
    recovery_dt = datetime.fromtimestamp(cooldown_until, tz=UTC)

    if recovery_seconds < 0:
        duration_str = "immediate"
    elif recovery_seconds < 60:
        duration_str = f"{int(recovery_seconds)}s"
    elif recovery_seconds < 3600:
        duration_str = f"{int(recovery_seconds / 60)}m"
    else:
        hours = int(recovery_seconds / 3600)
        minutes = int((recovery_seconds % 3600) / 60)
        duration_str = f"{hours}h {minutes}m"

    if quota is not None and quota.weekly_used_percent is not None and quota.weekly_used_percent >= 99:
        reason = "weekly 100% exhausted"
    elif quota is not None and quota.five_hourly_used_percent is not None and quota.five_hourly_used_percent >= 95:
        reason = f"5h window {quota.five_hourly_used_percent}% exhausted"
    else:
        reason = "rate limited"

    logger.info(f"[{backend_id}] cooldown set — {reason}, resumes {recovery_dt.isoformat()} ({duration_str})")
