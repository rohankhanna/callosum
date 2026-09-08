"""Tests for the optional streaming hooks used by chat-shaped backends."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from callosum.backend import CallHandle
from callosum.backends._http import error_from_response
from callosum.backends._responses_chat import chat_to_responses_stream
from callosum.errors import BackendError

CHAT_URL = "https://upstream.test/v1/chat/completions"


def _sse_lines() -> bytes:
    return (
        b'data: {"id":"r1","model":"m","choices":[{"delta":{"content":"hi"}}]}\n\n'
        b'data: {"id":"r1","choices":[{"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
        b"data: [DONE]\n\n"
    )


def _client_serving_sse() -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse_lines(), headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _drain(generator: AsyncIterator[bytes]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async for chunk in generator:
        for event_block in chunk.split(b"\n\n"):
            for line in event_block.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip().decode()
                if payload and payload != "[DONE]":
                    with contextlib.suppress(json.JSONDecodeError):
                        events.append(json.loads(payload))
    return events


def test_error_from_response_status_code_override_classifies_on_override() -> None:
    response = httpx.Response(200, content=b'{"error":{"message":"invalid key"}}')
    error = error_from_response(response, status_code=401)
    assert error.classification == "auth_invalid"
    assert error.status_code == 401
    assert "invalid key" in error.message


def test_error_from_response_without_override_uses_response_status() -> None:
    response = httpx.Response(502, content=b"bad gateway")
    error = error_from_response(response)
    assert error.classification == "transient"
    assert error.status_code == 502


async def test_default_hooks_use_direct_client_stream() -> None:
    client = _client_serving_sse()
    handle = CallHandle()
    generator = chat_to_responses_stream(
        client=client,
        chat_url=CHAT_URL,
        body={"model": "model", "input": "say hi", "stream": True},
        handle=handle,
        prep_body=lambda body: {**body, "stream": True},
        headers={},
        first_item_timeout_s=10.0,
        idle_timeout_s=5.0,
        what_label="test",
        on_success=lambda: None,
        on_transport_error=lambda: None,
    )
    events = await _drain(generator)
    assert events[0]["type"] == "response.created"
    deltas = [event for event in events if event["type"] == "response.output_text.delta"]
    assert "".join(event["delta"] for event in deltas) == "hi"
    completed = [event for event in events if event["type"] == "response.completed"]
    assert len(completed) == 1
    assert handle.upstream_status == 200
    await client.aclose()


async def test_open_chat_stream_hook_replaces_direct_post() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(404)))
    served: dict[str, Any] = {}

    @contextlib.asynccontextmanager
    async def open_chat_stream(out_body: dict[str, Any], _headers: dict[str, str]) -> AsyncIterator[httpx.Response]:
        served["out_body"] = out_body
        yield httpx.Response(200, content=_sse_lines(), headers={"content-type": "text/event-stream"})

    generator = chat_to_responses_stream(
        client=client,
        chat_url=CHAT_URL,
        body={"model": "model", "input": "say hi", "stream": True},
        handle=CallHandle(),
        prep_body=lambda body: {**body, "stream": True},
        headers={},
        first_item_timeout_s=10.0,
        idle_timeout_s=5.0,
        what_label="test",
        on_success=lambda: None,
        on_transport_error=lambda: None,
        open_chat_stream=open_chat_stream,
    )
    events = await _drain(generator)
    assert served["out_body"]["stream"] is True
    assert any(event["type"] == "response.completed" for event in events)
    await client.aclose()


async def test_upstream_status_of_hook_classifies_direct_401() -> None:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(404)))

    @contextlib.asynccontextmanager
    async def open_chat_stream(_out_body: dict[str, Any], _headers: dict[str, str]) -> AsyncIterator[httpx.Response]:
        yield httpx.Response(401, content=b'{"error":{"message":"invalid key"}}')

    handle = CallHandle()
    generator = chat_to_responses_stream(
        client=client,
        chat_url=CHAT_URL,
        body={"model": "model", "input": "say hi", "stream": True},
        handle=handle,
        prep_body=lambda body: {**body, "stream": True},
        headers={},
        first_item_timeout_s=10.0,
        idle_timeout_s=5.0,
        what_label="test",
        on_success=lambda: None,
        on_transport_error=lambda: None,
        open_chat_stream=open_chat_stream,
        upstream_status_of=lambda response: response.status_code,
    )
    with pytest.raises(BackendError) as exc_info:
        await _drain(generator)
    assert exc_info.value.classification == "auth_invalid"
    assert exc_info.value.status_code == 401
    assert handle.upstream_status == 401
    await client.aclose()


async def test_on_transport_error_fires_on_httpx_failure() -> None:
    on_success = MagicMock()
    on_transport_error = MagicMock()

    @contextlib.asynccontextmanager
    async def open_chat_stream(_out_body: dict[str, Any], _headers: dict[str, str]) -> AsyncIterator[httpx.Response]:
        raise httpx.ConnectError("upstream unavailable")
        yield  # pragma: no cover

    generator = chat_to_responses_stream(
        client=httpx.AsyncClient(),
        chat_url=CHAT_URL,
        body={"model": "model", "input": "say hi", "stream": True},
        handle=CallHandle(),
        prep_body=lambda body: {**body, "stream": True},
        headers={},
        first_item_timeout_s=10.0,
        idle_timeout_s=5.0,
        what_label="test",
        on_success=on_success,
        on_transport_error=on_transport_error,
        open_chat_stream=open_chat_stream,
    )
    with pytest.raises(BackendError) as exc_info:
        await _drain(generator)
    assert exc_info.value.classification == "transient"
    on_transport_error.assert_called_once()
    on_success.assert_not_called()
