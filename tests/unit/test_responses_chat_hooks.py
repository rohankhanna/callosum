"""Tests for the two optional hooks on chat_to_responses_stream and the
proxy-custody helpers they compose with.

The shared chat→Responses streaming generator
(callosum.backends._responses_chat.chat_to_responses_stream) is used by two
backends: litellm_gateway (local lane) and ollama_cloud (remote, via
credential proxy's boundary-native proxy). To keep the generator shape-agnostic, its
upstream-POST open and its upstream-status read were extracted into two
keyword-only hooks, both defaulting to None:

  * open_chat_stream(out_body, headers) — replaces the hardcoded
    client.stream("POST", chat_url, json=out_body, headers=headers). The
    ollama-cloud path passes a closure that mints a stand-in, builds the
    /v1/proxy/stream envelope, and yields the raw credential proxy stream response.
  * upstream_status_of(response) — replaces the bare response.status_code
    read. The ollama-cloud path passes OllamaCloudBackend._ollama_upstream_status
    so the real upstream status (from x-credential proxy-upstream-status) classifies
    the error, NOT credential proxy's own HTTP 200.

These tests guard three things:

  1. error_from_response's new status_code override classifies on the
     override while still reading the response body (the proxy-200-wrapping-
     upstream-401 case), and is byte-identical when omitted.
  2. _ollama_upstream_status keeps the two 401 surfaces distinct: header
     present → upstream status (401 ⇒ auth_invalid); header absent → credential proxy
     status, with any credential proxy non-2xx mapped to 502 (⇒ transient, NOT
     auth_invalid).
  3. The generator's default-None path reproduces the original direct-POST
     behavior exactly (litellm regression guard), and the hook path drives the
     open + status read through the injected callables (proxy path).

This file deliberately does NOT re-test the full SSE→Responses state machine
—that is covered by the litellm_gateway + ollama_cloud backend tests. It
tests only the hook wiring + classification disambiguation.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from callosum.backend import CallHandle
from callosum.backends._http import error_from_response
from callosum.backends._responses_chat import chat_to_responses_stream
from callosum.backends.ollama_cloud import (
    CUSTODY_UPSTREAM_STATUS_HEADER,
    OllamaCloudBackend,
)
from callosum.errors import BackendError

CHAT_URL = "https://upstream.test/v1/chat/completions"


def _sse_lines() -> bytes:
    """A minimal OpenAI-compatible chat-completions SSE stream: one content
    delta, a terminal chunk carrying usage, then [DONE]. The generator maps
    prompt_tokens→input_tokens and completion_tokens→
    output_tokens in the terminal response.completed event."""
    return (
        b'data: {"id":"r1","model":"m","choices":[{"delta":{"content":"hi"}}]}\n\n'
        b'data: {"id":"r1","choices":[{"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
        b"data: [DONE]\n\n"
    )


def _client_serving_sse() -> httpx.AsyncClient:
    """An httpx client whose MockTransport serves the canned SSE for any POST
    to CHAT_URL — the shape the default (no-hook) path hits."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse_lines(), headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _drain(gen: AsyncIterator[bytes]) -> list[dict[str, Any]]:
    """Collect emitted Responses-API events (parsed) for assertion. Skips the
    terminal [DONE] sentinel and blank separators."""
    events: list[dict[str, Any]] = []
    async for chunk in gen:
        for ev_block in chunk.split(b"\n\n"):
            for line in ev_block.split(b"\n"):
                if line.startswith(b"data:"):
                    payload = line[len(b"data:"):].strip().decode()
                    if payload and payload != "[DONE]":
                        with contextlib.suppress(json.JSONDecodeError):
                            events.append(json.loads(payload))
    return events


# ---------- error_from_response status_code override ------------------------


def test_error_from_response_status_code_override_classifies_on_override() -> None:
    """The proxy returns its own HTTP 200 wrapping an upstream 401 body. The
    caller passes status_code=401 so the error classifies as
    auth_invalid against the UPSTREAM status, while the message still reads
    the response body (the operator sees WHY ollama.com rejected the key)."""
    response = httpx.Response(200, content=b'{"error":{"message":"invalid api key"}}')
    err = error_from_response(response, status_code=401)
    assert err.classification == "auth_invalid"
    assert err.status_code == 401
    # The body is read from `response`, not the override — the message carries
    # the upstream reason so the operator can distinguish rotate-key from
    # scope-mismatch from quota.
    assert "invalid api key" in err.message


def test_error_from_response_without_override_is_byte_identical() -> None:
    """Omitting status_code reproduces the pre-hook behavior exactly:
    classify on response.status_code and set that as the error's
    status_code. This is the litellm local-lane path (no proxy)."""
    response = httpx.Response(502, content=b"bad gateway")
    err = error_from_response(response)
    assert err.classification == "transient"
    assert err.status_code == 502
    assert "upstream 502" in err.message


def test_error_from_response_502_is_transient_not_auth_invalid() -> None:
    """credential proxy-level non-2xx (the 502 the disambiguator maps an credential proxy 401/503
    to) classifies as transient, NOT auth_invalid — the stand-in rejection is
    a transient routing failure, not a key rejection."""
    response = httpx.Response(502, content=b"credential proxy rejected stand-in")
    err = error_from_response(response, status_code=502)
    assert err.classification == "transient"
    assert err.status_code == 502


# ---------- _ollama_upstream_status disambiguation --------------------------


def _backend_for_status_test() -> OllamaCloudBackend:
    """A backend instance whose transport is never exercised — we test the
    pure status-disambiguation method directly."""
    return OllamaCloudBackend(
        id="ollama-cloud",
        custody_url="http://credential proxy.test",
        transport=httpx.MockTransport(lambda r: httpx.Response(404)),
    )


def test_upstream_status_header_present_returns_upstream_status() -> None:
    """credential proxy HTTP 200 with x-credential proxy-upstream-status: 401 → the upstream
    401, so a key rejection classifies as auth_invalid even though the
    outer response is 200. This is the surface the stand-in is innocent for."""
    backend = _backend_for_status_test()
    response = httpx.Response(
        200,
        content=b'{"error":"auth"}',
        headers={CUSTODY_UPSTREAM_STATUS_HEADER: "401"},
    )
    assert backend._ollama_upstream_status(response) == 401


def test_upstream_status_header_absent_custody_200_returns_200() -> None:
    """A clean credential proxy 200 with no header (e.g. credential proxy forwards the upstream
    status only on streams — buffered uses the wrapper) → the credential proxy status,
    which is 200 → no error."""
    backend = _backend_for_status_test()
    response = httpx.Response(200, content=b"ok")
    assert backend._ollama_upstream_status(response) == 200


def test_upstream_status_header_absent_custody_401_maps_to_502_transient() -> None:
    """credential proxy itself returns 401 (stand-in rejected) with no upstream-status
    header → map to 502 so it classifies as transient, NOT auth_invalid.
    The stand-in is the problem, not the key — the two surfaces stay distinct."""
    backend = _backend_for_status_test()
    response = httpx.Response(401, content=b"stand-in rejected")
    assert backend._ollama_upstream_status(response) == 502


def test_upstream_status_header_absent_custody_503_maps_to_502_transient() -> None:
    """credential proxy 503 (scope not wired / pass entry missing) → 502 → transient.
    NOT auth_invalid (no key was rejected — there is no key to reject)."""
    backend = _backend_for_status_test()
    response = httpx.Response(503, content=b"scope not configured")
    assert backend._ollama_upstream_status(response) == 502


def test_upstream_status_header_non_integer_falls_back_to_custody_status() -> None:
    """A malformed header value (non-integer) is ignored; the method falls
    back to the credential proxy status (200 here → no error). Defensive: credential proxy is a
    sibling and its header shape is not contractually guaranteed forever."""
    backend = _backend_for_status_test()
    response = httpx.Response(
        200, content=b"ok", headers={CUSTODY_UPSTREAM_STATUS_HEADER: "not-a-number"}
    )
    assert backend._ollama_upstream_status(response) == 200


# ---------- generator hooks: default-None regression guard -----------------


async def test_default_none_hooks_use_direct_client_stream() -> None:
    """Regression guard for the litellm local lane: with NEITHER hook passed,
    the generator opens the upstream via client.stream("POST", chat_url, ...)
    and reads response.status_code — exactly the pre-hook behavior. A mock
    client serving SSE on the direct chat URL must produce the same
    response.created / output_text.delta / response.completed sequence the
    backend tests rely on. If a future refactor breaks the default-None branch,
    the local lane silently regresses."""
    client = _client_serving_sse()
    handle = CallHandle()
    gen = chat_to_responses_stream(
        client=client,
        chat_url=CHAT_URL,
        body={"model": "m", "input": "say hi", "stream": True},
        handle=handle,
        prep_body=lambda b: {**b, "stream": True},
        headers={},
        first_item_timeout_s=10.0,
        idle_timeout_s=5.0,
        what_label="test",
        on_success=lambda: None,
        on_transport_error=lambda: None,
    )
    events = await _drain(gen)
    types = [e["type"] for e in events]
    assert types[0] == "response.created"
    deltas = [e for e in events if e["type"] == "response.output_text.delta"]
    assert "".join(e["delta"] for e in deltas) == "hi"
    completed = [e for e in events if e["type"] == "response.completed"]
    assert len(completed) == 1
    assert completed[0]["response"]["usage"]["input_tokens"] == 3
    assert completed[0]["response"]["usage"]["output_tokens"] == 2
    # Default path read the outer status (200) directly.
    assert handle.upstream_status == 200
    await client.aclose()


# ---------- generator hooks: open_chat_stream replaces the direct POST -------


async def test_open_chat_stream_hook_replaces_direct_post() -> None:
    """With open_chat_stream set, the generator opens the upstream through
    the hook and NEVER calls client.stream. The hook yields a synthetic
    response (here standing in for credential proxy's /v1/proxy/stream), so a mock
    client that 404s every direct POST proves the direct path was bypassed."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    served: dict[str, Any] = {}

    @contextlib.asynccontextmanager
    async def open_chat_stream(
        out_body: dict[str, Any], _headers: dict[str, str]
    ) -> AsyncIterator[httpx.Response]:
        served["out_body"] = out_body
        # Synthetic credential proxy stream response: HTTP 200 + SSE bytes.
        yield httpx.Response(
            200, content=_sse_lines(), headers={"content-type": "text/event-stream"}
        )

    gen = chat_to_responses_stream(
        client=client,
        chat_url=CHAT_URL,
        body={"model": "m", "input": "say hi", "stream": True},
        handle=CallHandle(),
        prep_body=lambda b: {**b, "stream": True},
        headers={},
        first_item_timeout_s=10.0,
        idle_timeout_s=5.0,
        what_label="proxy",
        on_success=lambda: None,
        on_transport_error=lambda: None,
        open_chat_stream=open_chat_stream,
    )
    events = await _drain(gen)
    # The hook was used (its prep'd body was captured) and the direct POST was
    # never attempted (the 404 mock never reached).
    assert "out_body" in served
    assert served["out_body"]["stream"] is True
    assert any(e["type"] == "response.completed" for e in events)
    await client.aclose()


# ---------- generator hooks: upstream_status_of classifies proxy 401 --------


async def test_upstream_status_of_hook_classifies_proxy_upstream_401() -> None:
    """The proxy returns HTTP 200 (the hop succeeded) but the UPSTREAM
    ollama.com rejected the key: x-credential proxy-upstream-status: 401. The
    upstream_status_of hook reads the header → 401 → the generator raises
    error_from_response(response, status_code=401) → auth_invalid,
    and handle.upstream_status is 401. The stand-in is innocent (no
    in-stream re-mint here — the hook just reports the status)."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))

    @contextlib.asynccontextmanager
    async def open_chat_stream(
        out_body: dict[str, Any], _headers: dict[str, str]
    ) -> AsyncIterator[httpx.Response]:
        yield httpx.Response(
            200,
            content=b'{"error":{"message":"invalid api key"}}',
            headers={CUSTODY_UPSTREAM_STATUS_HEADER: "401"},
        )

    def upstream_status_of(response: httpx.Response) -> int:
        raw = response.headers.get(CUSTODY_UPSTREAM_STATUS_HEADER)
        return int(raw) if raw is not None else response.status_code

    handle = CallHandle()
    gen = chat_to_responses_stream(
        client=client,
        chat_url=CHAT_URL,
        body={"model": "m", "input": "say hi", "stream": True},
        handle=handle,
        prep_body=lambda b: {**b, "stream": True},
        headers={},
        first_item_timeout_s=10.0,
        idle_timeout_s=5.0,
        what_label="proxy",
        on_success=lambda: None,
        on_transport_error=lambda: None,
        open_chat_stream=open_chat_stream,
        upstream_status_of=upstream_status_of,
    )
    with pytest.raises(BackendError) as exc_info:
        await _drain(gen)
    # Upstream 401 → auth_invalid (the key was rejected by ollama.com).
    assert exc_info.value.classification == "auth_invalid"
    assert exc_info.value.status_code == 401
    # The hook's status landed on the handle for observability.
    assert handle.upstream_status == 401
    await client.aclose()


async def test_upstream_status_of_hook_custody_401_maps_to_transient() -> None:
    """The proxy itself rejects the stand-in: credential proxy HTTP 401, no
    upstream-status header. upstream_status_of returns the credential proxy status
    (401) — but per the disambiguation contract a bare credential proxy-401 should map
    to 502 → transient, NOT auth_invalid. This test pins the EXPECTED
    generator-path behavior by passing an upstream_status_of that already
    applies the 502 mapping (mirroring OllamaCloudBackend._ollama_upstream_status),
    so the generator surfaces a transient error rather than auth_invalid."""
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    backend = OllamaCloudBackend(
        id="ollama-cloud", transport=httpx.MockTransport(lambda r: httpx.Response(404))
    )

    @contextlib.asynccontextmanager
    async def open_chat_stream(
        out_body: dict[str, Any], _headers: dict[str, str]
    ) -> AsyncIterator[httpx.Response]:
        yield httpx.Response(401, content=b"stand-in rejected")

    handle = CallHandle()
    gen = chat_to_responses_stream(
        client=client,
        chat_url=CHAT_URL,
        body={"model": "m", "input": "say hi", "stream": True},
        handle=handle,
        prep_body=lambda b: {**b, "stream": True},
        headers={},
        first_item_timeout_s=10.0,
        idle_timeout_s=5.0,
        what_label="proxy",
        on_success=lambda: None,
        on_transport_error=lambda: None,
        open_chat_stream=open_chat_stream,
        upstream_status_of=backend._ollama_upstream_status,
    )
    with pytest.raises(BackendError) as exc_info:
        await _drain(gen)
    # credential proxy-401 (stand-in rejected) → 502 → transient, NOT auth_invalid.
    assert exc_info.value.classification == "transient"
    assert exc_info.value.status_code == 502
    assert handle.upstream_status == 502
    await client.aclose()
    await backend.aclose()


# ---------- on_transport_error hook fires on httpx transport failure --------


async def test_on_transport_error_fires_on_httpx_failure() -> None:
    """A transport error (httpx.HTTPError) inside the stream triggers the
    on_transport_error hook (the backend marks itself unhealthy) and
    surfaces as a transient BackendError. This is the existing contract the
    hooks preserve; guard it so the hook additions don't change it.

    The health hooks are plain sync callables — backends pass
    self._mark_unhealthy (sync) — so the generator calls them directly, not
    awaited. (on_success/on_transport_error were sync before the hooks
    were added and stay sync; only open_chat_stream/upstream_status_of
    are new.)
    """
    from unittest.mock import MagicMock

    on_success = MagicMock()
    on_transport_error = MagicMock()

    @contextlib.asynccontextmanager
    async def open_chat_stream(
        out_body: dict[str, Any], _headers: dict[str, str]
    ) -> AsyncIterator[httpx.Response]:
        # httpx-level failure opening the stream.
        raise httpx.ConnectError("credential proxy down")
        yield  # pragma: no cover  # unreachable; keeps the generator a CM

    gen = chat_to_responses_stream(
        client=httpx.AsyncClient(),
        chat_url=CHAT_URL,
        body={"model": "m", "input": "hi", "stream": True},
        handle=CallHandle(),
        prep_body=lambda b: {**b, "stream": True},
        headers={},
        first_item_timeout_s=10.0,
        idle_timeout_s=5.0,
        what_label="proxy",
        on_success=on_success,
        on_transport_error=on_transport_error,
        open_chat_stream=open_chat_stream,
    )
    with pytest.raises(BackendError) as exc_info:
        await _drain(gen)
    assert exc_info.value.classification == "transient"
    on_transport_error.assert_called_once()
    on_success.assert_not_called()