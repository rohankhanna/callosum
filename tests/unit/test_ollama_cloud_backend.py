"""Tests for the Ollama Cloud backend (credential proxy boundary-native proxy custody).

The backend NEVER holds the real ollama.com API key. It mints a short-TTL
`ollama-cloud`-scoped stand-in token at credential proxy's `POST /v1/standin` (loopback),
then POSTs every ollama.com call (chat, catalog, capabilities) through
credential proxy's `POST /v1/proxy` (buffered) or `POST /v1/proxy/stream` (streaming)
with the stand-in as `Authorization: Bearer <stand-in>`. credential proxy validates the
token, reads the real key from its `pass` store, strips caller
`authorization`/`chatgpt-account-id`, and injects `Authorization: Bearer
<real key>` on the final hop. The real key therefore never crosses into
callosum's process. This unifies the custody model with `credential_proxy`
(the OpenAI lane already routes through the same `/v1/proxy` surface).

The mock transport routes on the credential proxy path (`/v1/standin`, `/v1/proxy`,
`/v1/proxy/stream`) and unwraps the proxy envelope to recover the ollama.com
target path, so a single `httpx.MockTransport` stands in for the whole
credential proxy boundary + upstream ollama.com.

Contract crux — two distinct 401 surfaces:
  * (a) credential proxy rejects the STAND-IN (real HTTP 401 on /v1/proxy[/stream] or
    /v1/standin) → invalidate, re-mint once, retry; still 401 → transient
    BackendError + reason "network" (NOT auth_invalid).
  * (b) upstream ollama.com rejects the KEY (credential proxy HTTP 200 wrapping
    `status_code:401`, or `x-credential proxy-upstream-status:401` on a stream) →
    auth_invalid BackendError, NO stand-in re-mint, NO cooldown.

Catalog discovery, ModelMetadata synthesis (remote band), /api/show
capabilities, honest-advisory usage_snapshot (NOT the local free-stub),
health, refresh, and the four dispatch methods (chat non-stream + raw stream,
responses non-stream + translated SSE stream) are all exercised through the
envelope. Routing-classification is asserted here at the catalog/metadata
level and in the integration fixtures at the lane level.
"""

from __future__ import annotations

import base64
import contextlib
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from callosum.backend import CallHandle
from callosum.backends.ollama_cloud import (
    CUSTODY_UPSTREAM_STATUS_HEADER,
    CLOUD_PRIORITY_OFFSET,
    OLLAMA_CLOUD_SCOPE,
    OLLAMA_CLOUD_STANDIN_TTL_S,
    OllamaCloudBackend,
)
from callosum.errors import BackendError

credential proxy = "http://credential proxy.test"
# ollama_url defaults to https://ollama.com inside the backend; the envelope
# `url` field therefore carries https://ollama.com/<path>.

# ---------- canned upstream payloads (ollama.com shapes) -------------------


def _tags_payload(*names: str) -> dict[str, Any]:
    return {"models": [{"name": n} for n in names]}


def _show_payload(
    *,
    capabilities: list[str],
    context_length: int,
    parameter_count: int | None = None,
) -> dict[str, Any]:
    model_info: dict[str, Any] = {"gptoss.context_length": context_length}
    if parameter_count is not None:
        model_info["general.parameter_count"] = parameter_count
    return {"capabilities": capabilities, "model_info": model_info}


def _chat_reply(model: str) -> dict[str, Any]:
    """A minimal chat-completions non-stream reply with `usage` in the
    prompt_tokens/completion_tokens shape the OpenAI-compatible
    /v1/chat/completions endpoint returns, which `_extract_tokens` parses
    directly and `_chat_to_responses_response` maps to Responses shape."""
    return {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi there"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


# ---------- envelope helpers (mirror the backend's _b64/_unb64) ---------------


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii") if data else ""


def _unb64(data: str) -> bytes:
    return base64.b64decode(data) if data else b""


def _wrap_upstream(status: int, body: bytes, headers: dict[str, str] | None = None) -> httpx.Response:
    """credential proxy buffered-proxy return shape: HTTP 200 wrapping the upstream
    `{"status_code","headers","body_b64"}`. The CALLER classifies the upstream
    status — so an upstream 401 (status_code:401 in the wrapper) is distinct from
    an credential proxy-level 401 (which would be the outer response.status_code)."""
    return httpx.Response(
        200,
        json={
            "status_code": status,
            "headers": headers or {},
            "body_b64": _b64(body),
        },
    )


def _proxy_handler(
    *,
    tags: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    shows: dict[str, dict[str, Any]] | None = None,
    chat_json: dict[str, Any] | None = None,
    chat_sse: list[bytes] | None = None,
    chat_upstream_status: int = 200,
    chat_error: BaseException | None = None,
    standin_status: int = 200,
    standin_error: BaseException | None = None,
    custody_status: dict[str, int] | None = None,
    stream_upstream_status: int = 200,
) -> tuple[Callable[[httpx.Request], httpx.Response], SimpleNamespace]:
    """Build a MockTransport handler that stands in for the credential proxy boundary.

    Routes on request.url.path:
      * /v1/standin  → mint (200 {"token","expires_at"}), or
        standin_status / standin_error to exercise mint-failure paths.
      * /v1/proxy    → buffered. Unwrap the envelope to recover the ollama.com
        target via httpx.URL(envelope["url"]).path and serve a canned
        upstream payload wrapped in _wrap_upstream. custody_status maps
        a target path → a direct credential proxy HTTP status (401/503), scoped so the
        two-401-surface tests can keep the catalog path healthy while the chat
        path fails. chat_error raises on the chat target (transport error).
      * /v1/proxy/stream → streaming. Returns SSE bytes with the
        x-credential proxy-upstream-status header (stream_upstream_status).

    Returns (handler, rec) where rec records every standin/proxy/
    stream request for re-mint + envelope assertions.
    """
    tags_default = tags if tags is not None else _tags_payload("model-a0d2:cloud")
    shows = shows or {}
    custody_status = custody_status or {}
    rec = SimpleNamespace(standin=[], proxy=[], stream=[])

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/standin":
            rec.standin.append(request)
            if standin_error is not None:
                raise standin_error
            if standin_status != 200:
                return httpx.Response(standin_status)
            return httpx.Response(200, json={"token": "test-standin", "expires_at": 300})
        if path == "/v1/proxy":
            rec.proxy.append(request)
            envelope = json.loads(request.content)
            target = httpx.URL(envelope["url"]).path
            # credential proxy-level HTTP override (401/503) for a specific target — the
            # outer response.status_code, NOT the wrapped upstream status. This
            # is how the two-401-surface tests isolate credential proxy-401 (stand-in
            # rejected) from upstream-401 (key rejected).
            if target in custody_status:
                return httpx.Response(custody_status[target])
            if target == "/api/tags":
                payload = tags_default() if callable(tags_default) else tags_default
                return _wrap_upstream(
                    200, json.dumps(payload).encode(), {"content-type": "application/json"}
                )
            if target == "/api/show":
                name = json.loads(_unb64(envelope["body_b64"])).get("name")
                if name in shows:
                    return _wrap_upstream(
                        200,
                        json.dumps(shows[name]).encode(),
                        {"content-type": "application/json"},
                    )
                return _wrap_upstream(404, b'{"error":"not found"}')
            if target == "/v1/chat/completions":
                if chat_error is not None:
                    raise chat_error
                if chat_upstream_status != 200:
                    return _wrap_upstream(
                        chat_upstream_status,
                        b'{"error":"auth"}',
                        {"content-type": "application/json"},
                    )
                body = json.dumps(chat_json or _chat_reply("model-a0d2:cloud")).encode()
                return _wrap_upstream(200, body, {"content-type": "application/json"})
            return _wrap_upstream(404, b'{"error":"unknown target"}')
        if path == "/v1/proxy/stream":
            rec.stream.append(request)
            sse = (
                b"".join(chat_sse)
                if chat_sse is not None
                else b'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
            )
            return httpx.Response(
                200,
                content=sse,
                headers={
                    "content-type": "text/event-stream",
                    CUSTODY_UPSTREAM_STATUS_HEADER: str(stream_upstream_status),
                },
            )
        return httpx.Response(404)

    return handler, rec


def _backend(handler: Callable[[httpx.Request], httpx.Response], **kw: Any) -> OllamaCloudBackend:
    """Construct the backend pointed at the test credential proxy boundary. ollama_url
    defaults to https://ollama.com inside the backend, so the envelope `url`
    field carries the real ollama.com path the mock unwraps."""
    return OllamaCloudBackend(
        id="ollama-cloud",
        custody_url=credential proxy,
        custody_account="primary",
        transport=httpx.MockTransport(handler),
        **kw,
    )


# ---------- catalog discovery + suffix filter ------------------------------


async def test_advertised_models_empty_before_first_poll() -> None:
    handler, _ = _proxy_handler()
    backend = _backend(handler)
    assert backend.advertised_models == frozenset()
    h = await backend.health()  # forces a refresh through the proxy
    assert h.available is True
    assert backend.advertised_models == frozenset({"model-a0d2:cloud"})
    await backend.aclose()


async def test_default_empty_suffix_accepts_all_catalog_names() -> None:
    """DEFAULT_MODEL_SUFFIX (empty) accepts ALL /api/tags names (every entry
    on a direct all-cloud host is a cloud model). The operator overrides with
    `:cloud` if ollama.com still tags cloud models."""
    handler, _ = _proxy_handler(tags=_tags_payload("model-a0d2", "model-a0g2", "model-a0e4"))
    backend = _backend(handler)
    await backend.health()
    assert backend.advertised_models == frozenset({"model-a0d2", "model-a0g2", "model-a0e4"})
    await backend.aclose()


async def test_suffix_filter_keeps_only_cloud_models() -> None:
    """With the `:cloud` suffix override, only `:cloud`-suffixed models are
    cloud-metered; bare local models stay on the LocalModelRegistry / litellm_gateway
    path and must NOT appear in this backend's catalog."""
    handler, _ = _proxy_handler(
        tags=_tags_payload("model-a0d2:cloud", "model-a0b4", "model-a0f3:cloud", "model-a0d5")
    )
    backend = _backend(handler, model_suffix=":cloud")
    await backend.health()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud", "model-a0f3:cloud"})
    await backend.aclose()


async def test_custom_suffix_filter() -> None:
    handler, _ = _proxy_handler(tags=_tags_payload("model-a0d2-remote", "model-a0e4:cloud"))
    backend = _backend(handler, model_suffix="-remote")
    await backend.health()
    assert backend.advertised_models == frozenset({"model-a0d2-remote"})
    await backend.aclose()


async def test_catalog_picks_up_new_models_on_refresh() -> None:
    poll = {"n": 0}

    def tags_fn() -> dict[str, Any]:
        poll["n"] += 1
        if poll["n"] == 1:
            return _tags_payload("model-a0d2:cloud")
        return _tags_payload("model-a0d2:cloud", "model-a0f3:cloud")

    handler, _ = _proxy_handler(tags=tags_fn)
    backend = _backend(handler, catalog_refresh_s=0)  # always refresh
    await backend.health()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud"})
    await backend.health()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud", "model-a0f3:cloud"})
    await backend.aclose()


# ---------- model_metadata synthesis (remote band) --------------------------


async def test_model_metadata_marks_cloud_models_remote_band() -> None:
    handler, _ = _proxy_handler(tags=_tags_payload("model-a0d2:cloud", "model-a0f3:cloud"))
    backend = _backend(handler)
    await backend.health()
    md = backend.model_metadata
    assert set(md.keys()) == {"model-a0d2:cloud", "model-a0f3:cloud"}
    for _slug, m in md.items():
        assert m.supported_in_api is True
        assert m.visibility == "list"
        assert m.supported_reasoning_levels == ("default",)
        # Remote band: above curated Codex (tens), below free local (10_000).
        assert m.priority is not None and m.priority >= CLOUD_PRIORITY_OFFSET
        assert m.priority < 10_000
    assert md["model-a0d2:cloud"].priority == CLOUD_PRIORITY_OFFSET
    assert md["model-a0f3:cloud"].priority == CLOUD_PRIORITY_OFFSET + 1
    await backend.aclose()


# ---------- /api/show → cell_capabilities -----------------------------------


async def test_cell_capabilities_from_show() -> None:
    handler, _ = _proxy_handler(
        tags=_tags_payload("model-a0d2:cloud"),
        shows={
            "model-a0d2:cloud": _show_payload(
                capabilities=["completion", "tools", "vision"],
                context_length=128_000,
                parameter_count=20_000_000_000,
            )
        },
    )
    backend = _backend(handler)
    await backend.health()  # triggers _refresh_capabilities through the proxy
    caps = backend.cell_capabilities("model-a0d2:cloud")
    assert caps.cost_rank == 10  # remote default
    assert caps.supports_tools is True
    assert "text" in caps.modalities
    assert "image" in caps.modalities
    assert caps.context_window == 128_000
    assert caps.parameter_count == 20_000_000_000
    await backend.aclose()


async def test_cell_capabilities_fallback_when_show_missing() -> None:
    """A model in the catalog whose /api/show 404s falls back to conservative
    text-only/no-tools defaults rather than erroring — the request can still
    dispatch."""
    handler, _ = _proxy_handler(tags=_tags_payload("model-a0d2:cloud"), shows={})
    backend = _backend(handler)
    await backend.health()
    caps = backend.cell_capabilities("model-a0d2:cloud")
    assert caps.cost_rank == 10
    assert caps.supports_tools is False
    assert caps.modalities == frozenset({"text"})
    await backend.aclose()


async def test_cell_capabilities_fallback_before_first_refresh() -> None:
    """Synchronous cell_capabilities called before any catalog refresh
    returns the safe default (no crash, no stale state)."""
    handler, _ = _proxy_handler(tags=_tags_payload("model-a0d2:cloud"))
    backend = _backend(handler)
    caps = backend.cell_capabilities("model-a0d2:cloud")
    assert caps.cost_rank == 10
    assert caps.supports_tools is False
    await backend.aclose()


# ---------- honest-advisory usage_snapshot ----------------------------------


async def test_usage_snapshot_is_honest_not_local_free_stub() -> None:
    """Cloud is NOT free: usage_snapshot reports remaining_fraction=1.0
    ("full, eligible, no signal yet"), NOT the local free-stub's 0.001.
    weekly_exhausted is False (we genuinely don't know from headers), and there
    is no cooldown on a healthy backend."""
    handler, _ = _proxy_handler(tags=_tags_payload("model-a0d2:cloud"))
    backend = _backend(handler)
    await backend.health()
    snap = await backend.usage_snapshot()
    assert snap.remaining_fraction == 1.0
    assert snap.weekly_exhausted is False
    assert snap.cooldown_until_ts is None
    await backend.aclose()


async def test_quota_snapshot_is_none() -> None:
    handler, _ = _proxy_handler(tags=_tags_payload("model-a0d2:cloud"))
    backend = _backend(handler)
    assert await backend.quota_snapshot() is None
    await backend.aclose()


# ---------- health ----------------------------------------------------------


async def test_health_reports_network_when_custody_unreachable() -> None:
    """A transport error reaching credential proxy (on /v1/standin during catalog
    refresh) flips health to "network" — health() MUST NOT raise."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    backend = _backend(handler, catalog_refresh_s=0)
    h = await backend.health()
    assert h.available is False
    assert h.reason == "network"
    await backend.aclose()


async def test_health_reports_no_key_when_standin_mint_fails() -> None:
    """credential proxy reachable but the stand-in mint returns non-200/non-401 (here
    503 — the `ollama-cloud` scope is not wired) → reason "no-key". This is the
    correct failure mode for a provisioning gap, surfaced through health()."""
    handler, _ = _proxy_handler(standin_status=503)
    backend = _backend(handler, catalog_refresh_s=0)
    h = await backend.health()
    assert h.available is False
    assert h.reason == "no-key"
    await backend.aclose()


async def test_health_reports_no_key_when_custody_proxy_503() -> None:
    """The stand-in mints fine, but credential proxy's /v1/proxy returns 503 for the
    catalog call (scope wired but the pass entry holding the real key is
    missing/empty) → reason "no-key" (NOT network). health() MUST NOT raise."""
    handler, _ = _proxy_handler(custody_status={"/api/tags": 503})
    backend = _backend(handler, catalog_refresh_s=0)
    h = await backend.health()
    assert h.available is False
    assert h.reason == "no-key"
    await backend.aclose()


async def test_health_unhealthy_after_poll_sets_cooldown() -> None:
    """Once we've polled successfully, a subsequent outage sets a short
    cooldown so the backend is excluded from routable backends. On cold
    start (never polled) there is no cooldown so the fleet isn't empty."""
    state = {"up": True}
    tags = _tags_payload("model-a0d2:cloud")

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if not state["up"]:
            raise httpx.ConnectError("down")
        if path == "/v1/standin":
            return httpx.Response(200, json={"token": "t", "expires_at": 300})
        if path == "/v1/proxy":
            return _wrap_upstream(200, json.dumps(tags).encode(), {"content-type": "application/json"})
        return httpx.Response(404)

    backend = _backend(handler, catalog_refresh_s=0)
    await backend.health()  # first poll succeeds
    snap = await backend.usage_snapshot()
    assert snap.cooldown_until_ts is None  # healthy, no cooldown
    state["up"] = False
    await backend.health()  # outage
    snap = await backend.usage_snapshot()
    assert snap.cooldown_until_ts is not None  # transient outage → cooldown
    await backend.aclose()


# ---------- refresh_advertised_models ---------------------------------------


async def test_refresh_advertised_models_forces_refetch() -> None:
    """refresh_advertised_models bypasses the TTL gate so the lifespan loop
    can refresh all backends uniformly and callers see a current catalog
    immediately after it returns."""
    poll = {"n": 0}

    def tags_fn() -> dict[str, Any]:
        poll["n"] += 1
        if poll["n"] == 1:
            return _tags_payload("model-a0d2:cloud")
        return _tags_payload("model-a0d2:cloud", "model-a0f3:cloud")

    handler, _ = _proxy_handler(tags=tags_fn)
    backend = _backend(handler, catalog_refresh_s=3600)  # long TTL
    await backend.health()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud"})
    # Within TTL, a plain health() won't re-fetch; refresh_advertised_models must.
    await backend.refresh_advertised_models()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud", "model-a0f3:cloud"})
    await backend.aclose()


# ---------- proxy envelope + stand-in contract ------------------------------


async def test_standin_mint_sends_explicit_ollama_cloud_scope() -> None:
    """CRITICAL regression guard: the stand-in mint MUST send
    `scope="ollama-cloud"` explicitly (credential_proxy omits scope and gets
    the default `provider-upstream`). Forgetting the scope → credential proxy routes to
    the wrong pass key → 401 scope mismatch."""
    handler, rec = _proxy_handler(tags=_tags_payload("model-a0d2:cloud"))
    backend = _backend(handler)
    await backend.health()
    assert len(rec.standin) == 1
    mint_body = json.loads(rec.standin[0].content)
    assert mint_body["scope"] == OLLAMA_CLOUD_SCOPE
    assert mint_body["account"] == "primary"
    assert mint_body["ttl_seconds"] == OLLAMA_CLOUD_STANDIN_TTL_S
    assert mint_body["ttl_seconds"] <= 1800  # credential proxy MAX_STANDIN_TTL_SECONDS
    await backend.aclose()


async def test_buffered_envelope_shape_and_no_caller_auth_headers() -> None:
    """The buffered /v1/proxy envelope has exactly {url,method,headers,
    body_b64}; the inner headers carry ONLY content negotiation
    (Content-Type/Accept) — NEVER `authorization` or `chatgpt-account-id`
    (credential proxy scrubs those and overlays the real key on the final hop). The
    stand-in rides the OUTER request to credential proxy as a Bearer, not the envelope."""
    handler, rec = _proxy_handler(tags=_tags_payload("model-a0d2:cloud"))
    backend = _backend(handler)
    await backend.health()
    # rec.proxy holds the catalog GET; assert its envelope shape.
    assert len(rec.proxy) >= 1
    catalog_req = rec.proxy[0]
    envelope = json.loads(catalog_req.content)
    assert set(envelope.keys()) == {"url", "method", "headers", "body_b64"}
    assert envelope["url"] == "https://ollama.com/api/tags"
    assert envelope["method"] == "GET"
    inner = envelope["headers"]
    assert "authorization" not in inner
    assert "chatgpt-account-id" not in inner
    # GET carries no body.
    assert envelope["body_b64"] == ""
    # The stand-in is the OUTER Authorization to credential proxy, NOT in the envelope.
    assert catalog_req.headers.get("authorization") == "Bearer test-standin"
    await backend.aclose()


async def test_b64_unb64_roundtrip() -> None:
    assert OllamaCloudBackend._b64(b"") == ""
    assert OllamaCloudBackend._unb64("") == b""
    payload = b'{"model":"model-a0d2:cloud","stream":false}'
    assert OllamaCloudBackend._unb64(OllamaCloudBackend._b64(payload)) == payload


# ---------- dispatch (chat + responses, stream + non-stream) ----------------


async def test_chat_completions_nonstream() -> None:
    """Non-stream chat dispatch POSTs the codex-stripped body to ollama.com's
    /v1/chat/completions through credential proxy's buffered /v1/proxy; the envelope
    body_b64 decodes to the chat JSON with Codex-only fields stripped and
    stream forced False. upstream_status lands on the handle."""
    handler, rec = _proxy_handler()
    backend = _backend(handler)
    handle = CallHandle()
    reply = await backend.chat_completions(
        {
            "model": "model-a0d2:cloud",
            "reasoning": {"effort": "high"},
            "parallel_tool_calls": True,
        },
        handle,
    )
    assert reply["choices"][0]["message"]["content"] == "hi there"
    assert handle.upstream_status == 200
    # Exactly one chat /v1/proxy call (plus the catalog /api/tags call).
    chat_proxy = [r for r in rec.proxy if httpx.URL(json.loads(r.content)["url"]).path == "/v1/chat/completions"]
    assert len(chat_proxy) == 1
    env = json.loads(chat_proxy[0].content)
    assert env["url"] == "https://ollama.com/v1/chat/completions"
    assert env["method"] == "POST"
    sent = json.loads(_unb64(env["body_b64"]))
    assert sent["model"] == "model-a0d2:cloud"
    assert sent["stream"] is False
    assert "reasoning" not in sent
    assert "parallel_tool_calls" not in sent
    # Inner envelope headers: content negotiation only.
    assert env["headers"].get("Content-Type") == "application/json"
    assert "authorization" not in env["headers"]
    assert "chatgpt-account-id" not in env["headers"]
    await backend.aclose()


async def test_chat_completions_upstream_401_no_re_mint() -> None:
    """Upstream ollama.com 401 (credential proxy HTTP 200 wrapping status_code:401) →
    auth_invalid BackendError with status_code=401. The stand-in is INNOCENT —
    do NOT re-mint (standin call count stays at 1: the catalog-refresh mint).
    NO cooldown: the operator re-provisions the pass key; next call works."""
    handler, rec = _proxy_handler(chat_upstream_status=401)
    backend = _backend(handler)
    await backend.health()  # catalog refresh mints once (standin call 1)
    mint_count_after_health = len(rec.standin)
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        await backend.chat_completions({"model": "model-a0d2:cloud"}, handle)
    assert exc_info.value.classification == "auth_invalid"
    assert exc_info.value.status_code == 401
    # No stand-in re-mint on an UPSTREAM 401.
    assert len(rec.standin) == mint_count_after_health
    # Still healthy (catalog succeeded) → no cooldown.
    snap = await backend.usage_snapshot()
    assert snap.cooldown_until_ts is None
    await backend.aclose()


async def test_chat_completions_custody_401_re_mints_then_transient() -> None:
    """credential proxy rejects the STAND-IN (real HTTP 401 on /v1/proxy for the chat
    target) → invalidate, re-mint ONCE, retry; still 401 → transient
    BackendError (NOT auth_invalid) + reason "network" + cooldown. The catalog
    path stays healthy (no credential proxy override on /api/tags)."""
    handler, rec = _proxy_handler(custody_status={"/v1/chat/completions": 401})
    backend = _backend(handler)
    await backend.health()  # catalog mints (standin 1)
    with pytest.raises(BackendError) as exc_info:
        await backend.chat_completions({"model": "model-a0d2:cloud"})
    # credential proxy-401 → transient, NOT auth_invalid (that's the upstream surface).
    assert exc_info.value.classification == "transient"
    assert exc_info.value.status_code == 401
    # Re-mint happened exactly once (catalog mint + one retry mint).
    assert len(rec.standin) == 2
    # mark_unhealthy → network reason → cooldown.
    snap = await backend.usage_snapshot()
    assert snap.cooldown_until_ts is not None
    await backend.aclose()


async def test_chat_completions_custody_503_reports_no_key() -> None:
    """credential proxy /v1/proxy 503 for the chat call (scope wired but the pass entry
    is missing/empty) → transient BackendError + reason "no-key" (a
    provisioning gap, NOT a network outage)."""
    handler, rec = _proxy_handler(custody_status={"/v1/chat/completions": 503})
    backend = _backend(handler)
    await backend.health()
    with pytest.raises(BackendError) as exc_info:
        await backend.chat_completions({"model": "model-a0d2:cloud"})
    assert exc_info.value.classification == "transient"
    assert backend._last_health_reason == "no-key"
    await backend.aclose()


async def test_responses_nonstream() -> None:
    """The Responses path translates request→chat, dispatches through the
    buffered proxy, then translates the chat reply back to Responses shape:
    usage mapped from prompt_tokens/completion_tokens to
    input_tokens/output_tokens, and a message item carrying the assistant
    text in output[]."""
    handler, _ = _proxy_handler()
    backend = _backend(handler)
    resp = await backend.responses({"model": "model-a0d2:cloud", "input": "say hi"})
    assert resp["object"] == "response"
    assert resp["status"] == "completed"
    assert resp["usage"]["input_tokens"] == 5
    assert resp["usage"]["output_tokens"] == 2
    msg = next(o for o in resp["output"] if o["type"] == "message")
    assert msg["content"][0]["text"] == "hi there"
    await backend.aclose()


async def test_responses_stream_emits_responses_sse() -> None:
    """responses_stream accepts a Responses-shape body (`input`, `stream:True`)
    and emits Responses-API SSE translated from the chat-completions stream the
    proxy forwards: response.created first, incremental output_text.delta
    events, then response.completed carrying mapped usage. The
    ResponsesStreamCollector tees the completed usage onto
    handle.stream_summary so per-request metering populates automatically."""
    chunks = [
        b'data: {"id":"x","model":"model-a0d2:cloud","choices":[{"delta":{"content":"Hel"}}]}\n\n',
        b'data: {"id":"x","choices":[{"delta":{"content":"lo"}}]}\n\n',
        b'data: {"id":"x","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n',
        b"data: [DONE]\n\n",
    ]
    handler, _ = _proxy_handler(chat_sse=chunks)
    backend = _backend(handler)
    handle = CallHandle()
    events: list[dict[str, Any]] = []
    async for raw in backend.responses_stream(
        {"model": "model-a0d2:cloud", "input": "say hi", "stream": True}, handle
    ):
        for ev_chunk in raw.split(b"\n\n"):
            for line in ev_chunk.split(b"\n"):
                if line.startswith(b"data:"):
                    s = line[5:].strip().decode()
                    if s and s != "[DONE]":
                        with contextlib.suppress(json.JSONDecodeError):
                            events.append(json.loads(s))
    types = [e["type"] for e in events]
    assert types[0] == "response.created"
    text_deltas = [e for e in events if e["type"] == "response.output_text.delta"]
    assert "".join(e["delta"] for e in text_deltas) == "Hello"
    completed = [e for e in events if e["type"] == "response.completed"]
    assert len(completed) == 1
    assert completed[0]["response"]["usage"]["input_tokens"] == 3
    assert completed[0]["response"]["usage"]["output_tokens"] == 2
    # The collector teed the completed usage onto the handle → metering.
    assert handle.stream_summary is not None
    assert handle.stream_summary.completed_response is not None
    assert handle.stream_summary.completed_response["usage"]["input_tokens"] == 3
    await backend.aclose()


async def test_responses_stream_upstream_401_is_auth_invalid() -> None:
    """A mid-stream upstream 401 (credential proxy HTTP 200, `x-credential proxy-upstream-status:
    401` on the stream) → auth_invalid via error_from_response(status_code=401).
    handle.upstream_status == 401. The stand-in is innocent (no in-stream
    re-mint — matches credential_proxy's streaming behavior)."""
    chunks = [
        b'data: {"id":"x","choices":[{"delta":{"content":"Hel"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    handler, rec = _proxy_handler(chat_sse=chunks, stream_upstream_status=401)
    backend = _backend(handler)
    await backend.health()
    mints_after_health = len(rec.standin)
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        # Drain the generator so the upstream-401 raise surfaces.
        async for _ in backend.responses_stream(
            {"model": "model-a0d2:cloud", "input": "say hi", "stream": True}, handle
        ):
            pass
    assert exc_info.value.classification == "auth_invalid"
    assert handle.upstream_status == 401
    # No stand-in re-mint on an UPSTREAM 401.
    assert len(rec.standin) == mints_after_health
    await backend.aclose()


async def test_chat_completions_stream_passthrough() -> None:
    """chat_completions_stream is a raw byte passthrough — chat-completions
    SSE bytes flow through unchanged (no Responses translation). By design it
    does NOT set handle.stream_summary (mirrors litellm_gateway's chat-native
    gap; codex traffic uses /v1/responses → responses_stream, which does)."""
    chunks = [
        b'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    handler, _ = _proxy_handler(chat_sse=chunks)
    backend = _backend(handler)
    handle = CallHandle()
    out = b"".join([c async for c in backend.chat_completions_stream({"model": "model-a0d2:cloud"}, handle)])
    assert b'data: {"id":"x"' in out
    assert b"[DONE]" in out
    assert handle.upstream_status == 200
    assert handle.stream_summary is None
    await backend.aclose()


async def test_chat_completions_stream_upstream_401_is_auth_invalid() -> None:
    """Raw chat stream upstream 401 (x-credential proxy-upstream-status:401) →
    auth_invalid; handle.upstream_status == 401."""
    chunks = [b'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n"]
    handler, _ = _proxy_handler(chat_sse=chunks, stream_upstream_status=401)
    backend = _backend(handler)
    await backend.health()
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        async for _ in backend.chat_completions_stream({"model": "model-a0d2:cloud"}, handle):
            pass
    assert exc_info.value.classification == "auth_invalid"
    assert handle.upstream_status == 401
    await backend.aclose()


async def test_transport_error_marks_unhealthy() -> None:
    """A transport error on the chat /v1/proxy call (scoped to the chat target
    so the catalog /api/tags poll still succeeds) flips the backend unhealthy
    so usage_snapshot reports a cooldown immediately, and surfaces as a
    transient BackendError so the dispatch layer retries a different backend."""
    handler, _ = _proxy_handler(chat_error=httpx.ConnectError("cloud down"))
    backend = _backend(handler)
    await backend.health()  # first /api/tags poll through the proxy → healthy
    snap = await backend.usage_snapshot()
    assert snap.cooldown_until_ts is None
    with pytest.raises(BackendError):
        await backend.chat_completions({"model": "model-a0d2:cloud"})
    snap = await backend.usage_snapshot()
    assert snap.cooldown_until_ts is not None  # transient outage → cooldown
    await backend.aclose()


# ---------- default-OFF registration ----------------------------------------


def test_backend_absent_from_build_when_env_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CALLOSUM_OLLAMA_CLOUD_ENABLED unset → build_runtime_backends does NOT
    append the backend. Default-OFF is a no-op for live routing."""
    from callosum.__main__ import build_runtime_backends
    from callosum.config import Config
    from callosum.operator_state import OperatorState

    monkeypatch.delenv("CALLOSUM_OLLAMA_CLOUD_ENABLED", raising=False)
    # Ensure other optional backends don't add noise to the assertion.
    monkeypatch.setenv("CALLOSUM_LOCAL_DISABLED", "1")
    monkeypatch.setenv("CALLOSUM_LITELLM_GATEWAY_ENABLED", "0")
    cfg = Config()
    backends = build_runtime_backends(cfg, operator_state=OperatorState(tmp_path / "op.sqlite"))
    assert all(getattr(b, "id", None) != "ollama-cloud" for b in backends)


def test_backend_present_when_env_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from callosum.__main__ import build_runtime_backends
    from callosum.backends.ollama_cloud import OllamaCloudBackend
    from callosum.config import Config
    from callosum.operator_state import OperatorState

    monkeypatch.setenv("CALLOSUM_OLLAMA_CLOUD_ENABLED", "1")
    monkeypatch.setenv("CALLOSUM_LOCAL_DISABLED", "1")
    monkeypatch.setenv("CALLOSUM_LITELLM_GATEWAY_ENABLED", "0")
    cfg = Config()
    backends = build_runtime_backends(cfg, operator_state=OperatorState(tmp_path / "op.sqlite"))
    ids = [getattr(b, "id", None) for b in backends]
    assert "ollama-cloud" in ids
    cloud = next(b for b in backends if getattr(b, "id", None) == "ollama-cloud")
    assert cloud.kind == "ollama_cloud"
    # Proxy custody: no key-source attribute exists. The backend holds only a
    # short-TTL stand-in, minted lazily.
    assert isinstance(cloud, OllamaCloudBackend)
    assert not hasattr(cloud, "key_source")