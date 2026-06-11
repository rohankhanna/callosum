"""Tests for LocalModelRegistryBackend's filtering of chat-only cells.

Background: callosum's primary inbound path is /v1/responses (Codex
CLI's protocol). Chat-only cells on this backend can't serve streaming
responses because no chat→responses stream translator is wired here.
Advertising them anyway causes wasted dispatch retries — the router
picks them as primary, the request 502s instantly, then dispatch
falls through to a `-responses-proxy` sibling on the same backend.
The filter at the source side removes the noise without losing any
functional capability.
"""

from __future__ import annotations

import httpx
import pytest

from callosum.backends.local_direct import (
    MAX_LOCAL_TOOL_REQUEST_BYTES,
    LocalModelRegistryBackend,
)
from callosum.errors import BackendError
from callosum.local import ModelEntry


class _FakeSource:
    """LocalModelRegistrySource stub. The backend only consumes .models(force)
    so a tiny shim suffices."""

    def __init__(self, entries: list[ModelEntry]) -> None:
        self._entries = entries

    def models(self, *, force: bool = False) -> list[ModelEntry]:
        return list(self._entries)


def _entry(model_id: str, surfaces: tuple[str, ...]) -> ModelEntry:
    return ModelEntry(
        id=model_id,
        endpoint="http://127.0.0.1:0",
        runtime="ollama",
        runtime_model=model_id,
        family="test",
        context_window=128_000,
        api_surfaces=surfaces,
        enabled=True,
    )


def test_advertised_models_drops_chat_only_cells() -> None:
    """A chat-only sibling (api_surfaces == ("chat",)) is filtered out
    even when present in the source — the router never picks it, so
    the 502-and-retry waste is avoided."""
    src = _FakeSource([
        _entry("model-a0a9", ("chat",)),
        _entry("model-a0a1", ("responses",)),
    ])
    backend = LocalModelRegistryBackend(id="test", source=src)
    advertised = backend.advertised_models
    assert "model-a0a1" in advertised
    assert "model-a0a9" not in advertised


def test_advertised_models_keeps_cells_with_both_surfaces() -> None:
    """A cell that advertises both surfaces (chat AND responses) is
    serveable on the /v1/responses path and stays advertised."""
    src = _FakeSource([
        _entry("dual-surface", ("chat", "responses")),
    ])
    backend = LocalModelRegistryBackend(id="test", source=src)
    assert "dual-surface" in backend.advertised_models


def test_advertised_models_drops_cells_with_unknown_surfaces() -> None:
    """Defensive: a cell whose api_surfaces doesn't include 'responses'
    (e.g. empty tuple from a malformed registry entry) is filtered out
    rather than silently routed to."""
    src = _FakeSource([
        _entry("malformed", ()),
    ])
    backend = LocalModelRegistryBackend(id="test", source=src)
    assert backend.advertised_models == frozenset()


def test_model_metadata_filtered_same_as_advertised_models() -> None:
    """model_metadata is the cell-grid merger's hook. It must match
    advertised_models in filter behavior — otherwise the grid would
    include cells the router would then refuse to advertise."""
    src = _FakeSource([
        _entry("chat-only", ("chat",)),
        _entry("responses-capable", ("responses",)),
    ])
    backend = LocalModelRegistryBackend(id="test", source=src)
    meta = backend.model_metadata
    assert "responses-capable" in meta
    assert "chat-only" not in meta


async def test_responses_stream_rejects_oversized_tool_request_before_send() -> None:
    src = _FakeSource([_entry("responses-capable", ("responses",))])
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(500)

    backend = LocalModelRegistryBackend(
        id="test",
        source=src,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(BackendError) as exc_info:
        async for _ in backend.responses_stream(
            {
                "model": "responses-capable",
                "input": "x" * MAX_LOCAL_TOOL_REQUEST_BYTES,
                "stream": True,
                "tools": [{"type": "function", "function": {"name": "shell"}}],
            }
        ):
            pass
    assert exc_info.value.classification == "client_error"
    assert exc_info.value.status_code == 413
    assert "/v1/responses" not in calls
    await backend.aclose()
