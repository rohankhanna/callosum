"""Generic credential proxy backend.

Forwards requests to a credential proxy service that handles all credential
management. This backend has no knowledge of the provider, credential types, or
token management — it just routes requests.

The proxy service handles:
- Credential provisioning and refresh
- Token lifecycle management
- Credential injection on requests
- All authentication details
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from codex_proxy.backend import BackendKind, CallHandle, HealthStatus, UsageSnapshot
from codex_proxy.backends._http import DEFAULT_COOLDOWN_S
from codex_proxy.codex_quota import parse_codex_headers
from codex_proxy.errors import BackendError
from codex_proxy.sse_tee import ResponsesStreamCollector
from codex_proxy.state import StateStore

logger = logging.getLogger("codex_proxy.backend")


class CredentialProxyBackend:
    """Generic credential proxy backend.

    Forwards requests to a credential proxy service that handles all credential
    management. This backend only knows how to route requests and has no knowledge
    of how credentials are obtained or managed.

    The service is expected to provide:
    - POST {proxy_url}/v1/proxy - forward non-streaming requests
    - POST {proxy_url}/v1/proxy/stream - forward streaming requests
    """

    kind: BackendKind = "credential_proxy"

    def __init__(
        self,
        *,
        id: str,
        proxy_url: str,
        upstream_url: str,
        advertised_models: frozenset[str],
        custody_account: str | None = None,
        state_store: StateStore | None = None,
    ) -> None:
        if not advertised_models:
            raise ValueError(f"backend {id!r}: advertised_models cannot be empty")
        self.id = id
        self._proxy_url = proxy_url.rstrip("/")
        self._upstream_url = upstream_url.rstrip("/")
        self._advertised_models: frozenset[str] = advertised_models
        # Use provided custody_account or default to f"service-{id}"
        self._custody_account = custody_account or f"service-{id}"
        self._state_store = state_store
        loaded = state_store.load_usage(id) if state_store is not None else None
        self._usage = loaded or UsageSnapshot(
            remaining_fraction=None,
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=time.time(),
        )
        self._last_quota: Any = None
        self._http_client = httpx.AsyncClient(timeout=60.0)
        self._standin_token: str | None = None
        self._standin_expires_at: float | None = None

    @property
    def advertised_models(self) -> frozenset[str]:
        """Return the advertised model set."""
        return self._advertised_models


    async def _ensure_standin_token(self) -> str:
        """Ensure we have a valid stand-in token, provisioning if needed."""
        now = time.time()
        if (
            self._standin_token is not None
            and self._standin_expires_at is not None
            and self._standin_expires_at > now + 60
        ):
            return self._standin_token

        try:
            response = await self._http_client.post(
                f"{self._proxy_url}/v1/standin",
                json={"account": self._custody_account, "ttl_seconds": 1800},
            )
            response.raise_for_status()
            data = response.json()
            self._standin_token = data["token"]
            self._standin_expires_at = data.get("expires_at", now + 1800)
            return self._standin_token
        except Exception as exc:
            raise BackendError(classification="transient", message=f"Failed to provision stand-in token: {exc}") from exc

    async def health(self) -> HealthStatus:
        return HealthStatus(available=True, reason="ok")

    async def usage_snapshot(self) -> UsageSnapshot:
        weekly_exhausted = self._usage.weekly_exhausted
        if (
            self._last_quota is not None
            and self._last_quota.weekly_used_percent is not None
            and self._last_quota.weekly_used_percent >= 99
        ):
            weekly_exhausted = True
        if weekly_exhausted == self._usage.weekly_exhausted:
            return self._usage
        return UsageSnapshot(
            remaining_fraction=self._usage.remaining_fraction,
            cooldown_until_ts=self._usage.cooldown_until_ts,
            weekly_exhausted=weekly_exhausted,
            probed_at_ts=self._usage.probed_at_ts,
        )

    async def quota_snapshot(self) -> Any:
        return self._last_quota

    async def chat_completions(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> dict[str, Any]:
        # Translate chat format to responses format, proxy, translate back
        from codex_proxy.backends.codex_auth_vault import (
            _chat_to_responses_request,
            _responses_to_chat_response,
            _require_model,
        )

        requested_model = _require_model(body)
        responses_payload = _chat_to_responses_request(body, stream=False)
        upstream = await self.responses(responses_payload, handle)
        return _responses_to_chat_response(upstream, model=requested_model)

    async def chat_completions_stream(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> AsyncIterator[bytes]:
        from codex_proxy.backends.codex_auth_vault import (
            _int_or,
            _str_or,
        )

        completion = await self.chat_completions({**body, "stream": False}, handle)
        completion_id = _str_or(completion.get("id"), "chatcmpl-proxy")
        model = _str_or(completion.get("model"), "")
        created = _int_or(completion.get("created"), int(time.time()))
        choices = completion.get("choices", [])
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
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
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
        if handle is not None:
            handle.quota_before = self._last_quota

        try:
            # Ensure we have a valid stand-in token
            standin_token = await self._ensure_standin_token()

            # Ensure body has stream=true for upstream
            upstream_body = {**body, "stream": True}
            request_json = json.dumps(upstream_body).encode("utf-8")

            response = await self._http_client.post(
                f"{self._proxy_url}/v1/proxy",
                headers={"Authorization": f"Bearer {standin_token}"},
                json={
                    "url": self._upstream_url,
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "body_b64": self._b64_encode(request_json),
                },
            )

            response.raise_for_status()
            data = response.json()
            status_code = data.get("status_code", 200)
            response_headers = data.get("headers", {})
            response_body = self._b64_decode(data.get("body_b64", ""))

            self._apply_response_to_handle(response_headers, status_code, handle)
            if status_code >= 400:
                raise self._error_from_body(response_headers, status_code, response_body)

            # Parse streaming body as event stream
            collector = ResponsesStreamCollector(self._iter_bytes(response_body))
            async for _chunk in collector.iter_through():
                pass  # buffer

            if handle is not None:
                handle.stream_summary = collector.summary
            completed = (
                collector.summary.completed_response
                if collector.summary is not None
                else None
            )
            if completed is None:
                raise BackendError(
                    classification="transient",
                    message="upstream stream ended without response.completed event",
                )
            return completed
        except BackendError as err:
            self._apply_error_to_usage(err)
            raise
        except Exception as exc:
            raise BackendError(classification="transient", message=str(exc)) from exc

    async def responses_stream(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> AsyncIterator[bytes]:
        if handle is not None:
            handle.quota_before = self._last_quota

        try:
            # Ensure we have a valid stand-in token
            standin_token = await self._ensure_standin_token()

            request_json = json.dumps(body).encode("utf-8")

            async with self._http_client.stream(
                "POST",
                f"{self._proxy_url}/v1/proxy/stream",
                headers={"Authorization": f"Bearer {standin_token}"},
                json={
                    "url": self._upstream_url,
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "body_b64": self._b64_encode(request_json),
                },
            ) as response:
                response.raise_for_status()

                collector = ResponsesStreamCollector(response.aiter_bytes())
                async for chunk in collector.iter_through():
                    yield chunk
                if handle is not None:
                    handle.stream_summary = collector.summary
        except BackendError as err:
            self._apply_error_to_usage(err)
            raise
        except Exception as exc:
            raise BackendError(classification="transient", message=str(exc)) from exc

    async def aclose(self) -> None:
        await self._http_client.aclose()

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

    def _apply_response_to_handle(
        self, response_headers: dict[str, str], status_code: int, handle: CallHandle | None
    ) -> None:
        if handle is None:
            return
        handle.upstream_status = status_code
        handle.upstream_headers = response_headers
        quota = parse_codex_headers(response_headers)
        if quota is not None:
            self._last_quota = quota
            if handle is not None:
                handle.quota_after = quota

    @staticmethod
    async def _iter_bytes(data: bytes):
        """Async iterator for bytes."""
        if data:
            yield data

    @staticmethod
    def _b64_encode(data: bytes) -> str:
        """Base64 encode bytes."""
        import base64
        return base64.b64encode(data).decode("ascii") if data else ""

    @staticmethod
    def _b64_decode(data: str) -> bytes:
        """Base64 decode string."""
        import base64
        return base64.b64decode(data) if data else b""

    @staticmethod
    def _error_from_body(
        headers: dict[str, str], status_code: int, body: bytes
    ) -> BackendError:
        """Convert response to BackendError."""
        try:
            payload = json.loads(body)
            message = payload.get("message") or payload.get("error", {}).get("message", "")
        except (ValueError, json.JSONDecodeError):
            message = body.decode("utf-8", errors="replace")[:500]

        if status_code == 429:
            return BackendError(classification="rate_limited", message=message or "rate_limited")
        if 400 <= status_code < 500:
            return BackendError(classification="transient", message=f"HTTP {status_code}: {message}")
        return BackendError(classification="transient", message=f"HTTP {status_code}: {message}")
