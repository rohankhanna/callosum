from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from codex_proxy.backend import BackendKind, HealthStatus, UsageSnapshot
from codex_proxy.errors import BackendError


class InMemoryFakeBackend:
    """Deterministic Backend implementation used by tests and local dry-runs."""

    def __init__(
        self,
        *,
        id: str,
        advertised_models: frozenset[str],
        health: HealthStatus | None = None,
        usage: UsageSnapshot | None = None,
        canned_response: dict[str, Any] | None = None,
        canned_stream_chunks: Sequence[bytes] | None = None,
        canned_error: BackendError | None = None,
        canned_responses_response: dict[str, Any] | None = None,
        canned_responses_stream_chunks: Sequence[bytes] | None = None,
    ) -> None:
        self.id = id
        self.kind: BackendKind = "codex_auth_vault"
        self.advertised_models = advertised_models
        self._health = health if health is not None else HealthStatus(available=True, reason="ok")
        self._usage = (
            usage
            if usage is not None
            else UsageSnapshot(
                remaining_fraction=1.0,
                cooldown_until_ts=None,
                weekly_exhausted=False,
                probed_at_ts=0.0,
            )
        )
        self._canned_response = canned_response
        self._canned_stream_chunks = canned_stream_chunks
        self._canned_error = canned_error
        self._canned_responses_response = canned_responses_response
        self._canned_responses_stream_chunks = canned_responses_stream_chunks

    async def health(self) -> HealthStatus:
        return self._health

    async def usage_snapshot(self) -> UsageSnapshot:
        return self._usage

    async def chat_completions(self, body: dict[str, Any]) -> dict[str, Any]:
        if self._canned_error is not None:
            raise self._canned_error
        if self._canned_response is not None:
            return self._canned_response
        model = body.get("model", "unknown")
        return {
            "id": f"fake-{self.id}",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"fake response from {self.id}"},
                    "finish_reason": "stop",
                }
            ],
        }

    async def chat_completions_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        if self._canned_error is not None:
            raise self._canned_error
        chunks = self._canned_stream_chunks
        if chunks is None:
            model = body.get("model", "unknown")
            chunks = (
                f'data: {{"id":"fake-{self.id}","model":"{model}",'
                f'"choices":[{{"delta":{{"content":"fake"}}}}]}}\n\n'.encode(),
                b"data: [DONE]\n\n",
            )
        for chunk in chunks:
            yield chunk

    async def responses(self, body: dict[str, Any]) -> dict[str, Any]:
        if self._canned_error is not None:
            raise self._canned_error
        if self._canned_responses_response is not None:
            return self._canned_responses_response
        model = body.get("model", "unknown")
        return {
            "id": f"resp-{self.id}",
            "object": "response",
            "model": model,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": f"fake responses from {self.id}"}],
                }
            ],
        }

    async def responses_stream(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        if self._canned_error is not None:
            raise self._canned_error
        chunks = self._canned_responses_stream_chunks
        if chunks is None:
            chunks = (
                f'data: {{"type":"response.created","id":"resp-{self.id}"}}\n\n'.encode(),
                b"data: [DONE]\n\n",
            )
        for chunk in chunks:
            yield chunk

    async def aclose(self) -> None:
        return None

    def set_health(self, health: HealthStatus) -> None:
        self._health = health

    def set_usage(self, usage: UsageSnapshot) -> None:
        self._usage = usage
