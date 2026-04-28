from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

import httpx

from codex_proxy.auth_vault import AuthVault
from codex_proxy.backend import BackendKind, CallHandle, HealthStatus, UsageSnapshot
from codex_proxy.backends._http import DEFAULT_COOLDOWN_S, error_from_response
from codex_proxy.codex_quota import parse_codex_headers
from codex_proxy.errors import BackendError
from codex_proxy.sse_tee import ResponsesStreamCollector
from codex_proxy.state import StateStore

DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
RESPONSES_BETA_HEADER_VALUE = "responses=v1"

# How long to keep a fetched model catalog before refreshing. The Codex model
# lineup changes monthly (and trending toward weekly per operator), so an
# hourly refresh keeps us current without hammering the upstream.
DEFAULT_MODELS_REFRESH_S = 3600.0


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
        advertised_models: frozenset[str],
        base_url: str = DEFAULT_BASE_URL,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = 60.0,
        state_store: StateStore | None = None,
        models_refresh_s: float = DEFAULT_MODELS_REFRESH_S,
    ) -> None:
        if not advertised_models:
            raise ValueError(f"backend {id!r}: advertised_models cannot be empty")
        self.id = id
        # `_static_advertised_models` is the operator's TOML override / cold-
        # start fallback. `_dynamic_advertised_models` is what we last fetched
        # from upstream's /models endpoint (None until first fetch). The
        # `advertised_models` property prefers dynamic when available.
        self._static_advertised_models: frozenset[str] = advertised_models
        self._dynamic_advertised_models: frozenset[str] | None = None
        self._models_fetched_at: float = 0.0
        self._models_refresh_s = models_refresh_s
        self._vault = vault
        self._base_url = base_url.rstrip("/")
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(transport=transport, timeout=timeout_s)
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
        self._last_quota: Any = (
            None  # CodexQuotaSnapshot | None, loose typed to avoid import cycle noise
        )

    @property
    def advertised_models(self) -> frozenset[str]:
        """Return the most recently fetched upstream model list, falling back
        to the operator-provided static set when we haven't fetched yet (cold
        start) or when the fetch failed.
        """
        if self._dynamic_advertised_models is not None:
            return self._dynamic_advertised_models
        return self._static_advertised_models

    async def refresh_advertised_models(self, *, now: float | None = None) -> None:
        """Fetch the upstream model catalog for this account and update the
        cached set. Best-effort: silently keeps the current set on any error
        (network down, auth invalid, malformed payload). Skip if the cached
        copy is younger than `models_refresh_s`.
        """
        ts = now if now is not None else time.time()
        if (
            self._dynamic_advertised_models is not None
            and ts - self._models_fetched_at < self._models_refresh_s
        ):
            return
        try:
            tokens = await self._vault.current()
        except BackendError:
            return
        try:
            response = await self._client.get(
                f"{self._base_url}/models",
                headers=self._build_headers(tokens.access_token, tokens.account_id),
            )
        except httpx.HTTPError:
            return
        if response.status_code != 200:
            return
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError):
            return
        models = _extract_model_slugs(payload)
        if models:
            self._dynamic_advertised_models = models
            self._models_fetched_at = ts

    async def health(self) -> HealthStatus:
        return HealthStatus(available=True, reason="ok")

    async def usage_snapshot(self) -> UsageSnapshot:
        return self._usage

    async def quota_snapshot(self) -> Any:  # CodexQuotaSnapshot | None
        return self._last_quota

    async def chat_completions(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> dict[str, Any]:
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

    async def responses(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> dict[str, Any]:
        # Body is already in Responses-API shape; forward verbatim.
        # Try once; on upstream 401 (server-side token revocation, not
        # JWT-exp expiry), force-refresh and retry once with new tokens.
        if handle is not None:
            handle.quota_before = self._last_quota
        for attempt in (1, 2):
            tokens = (
                await self._vault.current() if attempt == 1 else await self._vault.force_refresh()
            )
            headers = self._build_headers(tokens.access_token, tokens.account_id)
            try:
                response = await self._client.post(
                    f"{self._base_url}/responses",
                    json=body,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                raise BackendError(classification="transient", message=str(exc)) from exc
            if attempt == 1 and response.status_code == 401:
                # Don't apply this response to the handle/quota — it's a
                # transient artifact of stale credentials, not the call result.
                continue
            self._apply_response_to_handle(response.headers, response.status_code, handle)
            if response.status_code >= 400:
                err = error_from_response(response)
                self._apply_error_to_usage(err)
                raise err
            return cast(dict[str, Any], response.json())
        # Unreachable: loop either returns or raises on attempt 2.
        raise BackendError(classification="auth_invalid", message="auth retry exhausted")

    async def responses_stream(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> AsyncIterator[bytes]:
        if handle is not None:
            handle.quota_before = self._last_quota
        # Open the stream; on upstream 401 in the response headers, close and
        # restart with refreshed tokens. We can only retry before any chunk
        # has been yielded — once streaming starts, we're committed.
        for attempt in (1, 2):
            tokens = (
                await self._vault.current() if attempt == 1 else await self._vault.force_refresh()
            )
            headers = self._build_headers(
                tokens.access_token, tokens.account_id, accept_event_stream=True
            )
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
                    collector = ResponsesStreamCollector(response.aiter_bytes())
                    async for chunk in collector.iter_through():
                        yield chunk
                    if handle is not None:
                        handle.stream_summary = collector.summary
                    return
            except httpx.HTTPError as exc:
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
        cooldown_until = now + (err.retry_after_s or DEFAULT_COOLDOWN_S)
        self._usage = UsageSnapshot(
            remaining_fraction=self._usage.remaining_fraction,
            cooldown_until_ts=cooldown_until,
            weekly_exhausted=self._usage.weekly_exhausted,
            probed_at_ts=now,
        )
        if self._state_store is not None:
            self._state_store.save_usage(self.id, self._usage)


def _extract_model_slugs(payload: Any) -> frozenset[str]:
    """Pull model slug strings out of an upstream `/backend-api/codex/models`
    response. Tolerates both the documented shape {"models": [{"slug": ...}]}
    and the OpenAI-compatible {"data": [{"id": ...}]} shape, since the
    upstream surface has shipped both at different times. Returns an empty
    frozenset on anything malformed — callers treat empty as "fall back to the
    cold-start static set."
    """
    if not isinstance(payload, dict):
        return frozenset()
    items: Any = payload.get("models")
    if not isinstance(items, list):
        items = payload.get("data")
    if not isinstance(items, list):
        return frozenset()
    slugs: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug:
            slug = item.get("id")
        if isinstance(slug, str) and slug:
            slugs.add(slug)
    return frozenset(slugs)


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
                if instructions is None:
                    instructions = content_text
                else:
                    instructions = f"{instructions}\n{content_text}"
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
    }
    if instructions is not None:
        payload["instructions"] = instructions
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
                "finish_reason": "stop",
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
