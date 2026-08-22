"""Tests for the OpenRouter backend (credential proxy boundary-native proxy custody).

The backend NEVER holds the real OpenRouter API key. It mints a short-TTL
`openrouter`-scoped stand-in token at credential proxy's `POST /v1/standin` (loopback),
then POSTs every OpenRouter call (chat, catalog) through credential proxy's
`POST /v1/proxy` (buffered) or `POST /v1/proxy/stream` (streaming) with the
stand-in as `Authorization: Bearer <stand-in>`. credential proxy validates the token,
reads the real key from its `pass` store, strips caller auth headers, and
injects `Authorization: Bearer <real key>` on the final hop. The real key
therefore never crosses into callosum's process.

The mock transport routes on the credential proxy path (`/v1/standin`, `/v1/proxy`,
`/v1/proxy/stream`) and unwraps the proxy envelope to recover the OpenRouter
target path, so a single `httpx.MockTransport` stands in for the whole
credential proxy boundary + upstream OpenRouter.

Contract crux — two distinct 401 surfaces:
  * (a) credential proxy rejects the STAND-IN (real HTTP 401 on /v1/proxy[/stream] or
    /v1/standin) → invalidate, re-mint once, retry; still 401 → transient
    BackendError + reason "network" (NOT auth_invalid).
  * (b) upstream OpenRouter rejects the KEY (credential proxy HTTP 200 wrapping
    `status_code:401`, or `x-credential proxy-upstream-status:401` on a stream) →
    auth_invalid BackendError, NO stand-in re-mint.

OpenRouter-specific coverage vs the ollama_cloud tests: full auto-discovery
catalog from `/v1/models` (OpenAI shape), the ollama-overlap family-exclude,
the `/v1/providers`-derived regime-provider deny set (datacenters OR
headquarters), per-request `provider.ignore` residency injection, the 402
(insufficient credits) → rate_limited + exhausted-cooldown path, and the
allowlist/prefix model-filter modes.
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
from callosum.backends.openrouter import (
    CUSTODY_UPSTREAM_STATUS_HEADER,
    OPENROUTER_PRIORITY_OFFSET,
    OPENROUTER_SCOPE,
    OPENROUTER_STANDIN_TTL_S,
    OpenRouterBackend,
)
from callosum.errors import BackendError

credential proxy = "http://credential proxy.test"
# OpenRouter base URL default inside the backend is
# https://openrouter.ai/api/v1; the envelope `url` field therefore carries
# https://openrouter.ai/api/v1/<path>. Far-future ABSOLUTE Unix epoch
# (~year 2286) keeps the stand-in fresh across multiple _ensure_standin
# calls within a test (no spurious re-mint).
_FAR_FUTURE_EXPIRES_AT = 9_999_999_999


# ---------- canned upstream payloads (OpenRouter shapes) --------------------


def _models_payload(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"data": list(entries)}


def _model(slug: str, context_length: int | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"id": slug, "name": slug}
    if context_length is not None:
        entry["context_length"] = context_length
    return entry


# Default mixed catalog: some families ollama_cloud overlaps (model-a0g1/model-a0g3/
# model-a0d5/model-a0e2) and some it doesn't (openai/anthropic/google non-model-a0d5).
_DEFAULT_MODELS = _models_payload(
    _model("openai/model-a0f5", 128_000),
    _model("anthropic/model-a0aa", 200_000),
    _model("meta-model-a0g1/model-a0c2", 128_000),
    _model("model-a0g3/model-a0g3-2.5-72b", 131_072),
    _model("model-a0e2/model-a0c6", 65_536),
    _model("google/model-a0d5", 8_192),
)


# Default providers: alibaba (SG HQ, CN datacenter → caught by datacenters),
# z-ai (CN HQ → caught by headquarters), deepinfra (US, US datacenter → ok),
# nebius (NL HQ, null datacenters → ok), unknown-co (null/null → not denied).
_DEFAULT_PROVIDERS = {
    "data": [
        {"slug": "alibaba", "headquarters": "SG", "datacenters": ["SG", "CN"]},
        {"slug": "z-ai", "headquarters": "CN", "datacenters": ["CN"]},
        {"slug": "deepinfra", "headquarters": "US", "datacenters": ["US"]},
        {"slug": "nebius", "headquarters": "NL", "datacenters": None},
        {"slug": "unknown-co", "headquarters": None, "datacenters": None},
    ]
}


def _chat_reply(model: str) -> dict[str, Any]:
    """Minimal chat-completions non-stream reply with `usage` in the
    prompt_tokens/completion_tokens shape `_extract_tokens` parses and
    `_chat_to_responses_response` maps to Responses shape."""
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
    `{"status_code","headers","body_b64"}`. The CALLER classifies the
    upstream status — an upstream 401 (status_code:401 in the wrapper) is
    distinct from an credential proxy-level 401 (the outer response.status_code)."""
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
    models: dict[str, Any] | None = None,
    providers: dict[str, Any] | None = None,
    chat_json: dict[str, Any] | None = None,
    chat_sse: list[bytes] | None = None,
    chat_upstream_status: int = 200,
    chat_error: BaseException | None = None,
    standin_status: int = 200,
    standin_error: BaseException | None = None,
    custody_status: dict[str, int] | None = None,
    stream_upstream_status: int = 200,
    providers_status: int = 200,
    models_status: int = 200,
) -> tuple[Callable[[httpx.Request], httpx.Response], SimpleNamespace]:
    """Build a MockTransport handler that stands in for the credential proxy boundary.

    Routes on request.url.path:
      * /v1/standin  → mint (200 {"token","expires_at"}), or
        standin_status / standin_error to exercise mint-failure paths.
      * /v1/proxy    → buffered. Unwrap the envelope to recover the
        OpenRouter target via httpx.URL(envelope["url"]).path and serve a
        canned upstream payload wrapped in _wrap_upstream.
        custody_status maps a target path → a direct credential proxy HTTP status
        (401/503), scoped so the two-401-surface tests can keep the catalog
        path healthy while the chat path fails. chat_error raises on the
        chat target (transport error). models_status/providers_status
        exercise catalog-refresh failure paths.
      * /v1/proxy/stream → streaming. Returns SSE bytes with the
        x-credential proxy-upstream-status header (stream_upstream_status).

    Returns (handler, rec) where rec records every
    standin/proxy/stream request for re-mint + envelope assertions.
    """
    models_default = models if models is not None else _DEFAULT_MODELS
    providers_default = providers if providers is not None else _DEFAULT_PROVIDERS
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
            return httpx.Response(200, json={"token": "test-standin", "expires_at": _FAR_FUTURE_EXPIRES_AT})
        if path == "/v1/proxy":
            rec.proxy.append(request)
            envelope = json.loads(request.content)
            # OpenRouter's base URL carries an `/api/v1` prefix, so the
            # envelope url's path is `/api/v1/<endpoint>` (NOT `/models`).
            # Match by path SUFFIX so the handler is robust to the base path
            # and so tests key `custody_status` by the bare endpoint suffix.
            target = httpx.URL(envelope["url"]).path
            # credential proxy-level HTTP override (401/503) for a specific target — the
            # outer response.status_code, NOT the wrapped upstream status.
            for suffix, code in custody_status.items():
                if target.endswith(suffix):
                    return httpx.Response(code)
            if target.endswith("/models"):
                if models_status != 200:
                    return _wrap_upstream(models_status, b'{"error":"upstream"}')
                return _wrap_upstream(200, json.dumps(models_default).encode(), {"content-type": "application/json"})
            if target.endswith("/providers"):
                if providers_status != 200:
                    return _wrap_upstream(providers_status, b'{"error":"upstream"}')
                return _wrap_upstream(200, json.dumps(providers_default).encode(), {"content-type": "application/json"})
            if target.endswith("/chat/completions"):
                if chat_error is not None:
                    raise chat_error
                if chat_upstream_status != 200:
                    return _wrap_upstream(
                        chat_upstream_status,
                        b'{"error":"auth"}',
                        {"content-type": "application/json"},
                    )
                body = json.dumps(chat_json or _chat_reply("openai/model-a0f5")).encode()
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


def _backend(handler: Callable[[httpx.Request], httpx.Response], **kw: Any) -> OpenRouterBackend:
    """Construct the backend pointed at the test credential proxy boundary. base_url
    defaults to https://openrouter.ai/api/v1 inside the backend, so the
    envelope `url` field carries the real OpenRouter path the mock unwraps."""
    return OpenRouterBackend(
        id="openrouter",
        custody_url=credential proxy,
        custody_account="primary",
        transport=httpx.MockTransport(handler),
        **kw,
    )


# ---------- catalog discovery + family-exclude + filter ---------------------


async def test_advertised_models_empty_before_first_poll() -> None:
    handler, _ = _proxy_handler()
    backend = _backend(handler)
    assert backend.advertised_models == frozenset()
    h = await backend.health()  # forces a refresh through the proxy
    assert h.available is True
    assert backend.advertised_models == frozenset(
        {"openai/model-a0f5", "anthropic/model-a0aa"}
    )
    await backend.aclose()


async def test_family_exclude_drops_ollama_cloud_overlap() -> None:
    """Default family-exclude (model-a0g3/model-a0g1/model-a0d5/model-a0e2) drops families the
    paid ollama_cloud backend already serves, so OpenRouter doesn't get paid
    for them. Keeps openai/anthropic (no overlap)."""
    handler, _ = _proxy_handler()
    backend = _backend(handler)
    await backend.health()
    assert backend.advertised_models == frozenset(
        {"openai/model-a0f5", "anthropic/model-a0aa"}
    )
    assert "meta-model-a0g1/model-a0c2" not in backend.advertised_models
    assert "model-a0g3/model-a0g3-2.5-72b" not in backend.advertised_models
    assert "model-a0e2/model-a0c6" not in backend.advertised_models
    assert "google/model-a0d5" not in backend.advertised_models
    await backend.aclose()


async def test_family_exclude_disabled_keeps_all() -> None:
    handler, _ = _proxy_handler()
    backend = _backend(handler, exclude_families=frozenset())
    await backend.health()
    assert backend.advertised_models == frozenset(
        {
            "openai/model-a0f5",
            "anthropic/model-a0aa",
            "meta-model-a0g1/model-a0c2",
            "model-a0g3/model-a0g3-2.5-72b",
            "model-a0e2/model-a0c6",
            "google/model-a0d5",
        }
    )
    await backend.aclose()


async def test_allowlist_filter() -> None:
    handler, _ = _proxy_handler()
    backend = _backend(
        handler,
        model_filter="allowlist",
        allowlist=frozenset({"openai/model-a0f5", "model-a0g3/model-a0g3-2.5-72b"}),
    )
    await backend.health()
    # allowlist applies before family-exclude; model-a0g3 is in the allowlist but
    # then dropped by family-exclude → only model-a0f5 survives.
    assert backend.advertised_models == frozenset({"openai/model-a0f5"})
    await backend.aclose()


async def test_prefix_filter() -> None:
    handler, _ = _proxy_handler()
    backend = _backend(handler, model_filter="prefix", model_prefix="anthropic/")
    await backend.health()
    assert backend.advertised_models == frozenset({"anthropic/model-a0aa"})
    await backend.aclose()


async def test_invalid_model_filter_raises() -> None:
    handler, _ = _proxy_handler()
    with pytest.raises(ValueError):
        _backend(handler, model_filter="bogus")


# ---------- model_metadata synthesis (remote band) --------------------------


async def test_model_metadata_remote_band_and_context_window() -> None:
    handler, _ = _proxy_handler()
    backend = _backend(handler)
    await backend.health()
    md = backend.model_metadata
    assert set(md.keys()) == {"openai/model-a0f5", "anthropic/model-a0aa"}
    for _slug, m in md.items():
        assert m.supported_in_api is True
        assert m.visibility == "list"
        assert m.supported_reasoning_levels == ("default",)
        # Conservative-overflow band: above Codex (tens) + ollama_cloud (1000),
        # below free local (10_000).
        assert m.priority is not None and m.priority >= OPENROUTER_PRIORITY_OFFSET
        assert m.priority < 10_000
    # Index-stable ordering: first-kept slug gets the base offset.
    assert md["openai/model-a0f5"].priority == OPENROUTER_PRIORITY_OFFSET
    assert md["anthropic/model-a0aa"].priority == OPENROUTER_PRIORITY_OFFSET + 1
    # context_window flows from /models context_length.
    assert md["openai/model-a0f5"].context_window == 128_000
    assert md["anthropic/model-a0aa"].context_window == 200_000
    await backend.aclose()


# ---------- regime-provider deny derivation ---------------------------------


async def test_regime_deny_derivation_default() -> None:
    """Default CN/RU/KP: alibaba denied (datacenters include CN), z-ai denied
    (headquarters CN), deepinfra/nebius/unknown-co NOT denied."""
    handler, _ = _proxy_handler()
    backend = _backend(handler)
    await backend.health()
    assert backend._regime_provider_deny == frozenset({"alibaba", "z-ai"})
    await backend.aclose()


async def test_regime_deny_allowed_providers_exempts() -> None:
    """allowed_providers always wins — exempts a regime provider the operator
    confirms serves only from non-regime datacenters."""
    handler, _ = _proxy_handler()
    backend = _backend(handler, allowed_providers=frozenset({"z-ai"}))
    await backend.health()
    assert backend._regime_provider_deny == frozenset({"alibaba"})
    await backend.aclose()


async def test_regime_deny_blocked_providers_adds() -> None:
    """blocked_providers adds extra provider slugs beyond the country-derived
    set (e.g. a provider with null metadata the operator judges regime)."""
    handler, _ = _proxy_handler()
    backend = _backend(handler, blocked_providers=frozenset({"nebius"}))
    await backend.health()
    assert backend._regime_provider_deny == frozenset({"alibaba", "z-ai", "nebius"})
    await backend.aclose()


async def test_regime_deny_custom_blocked_countries() -> None:
    """blocked_countries is extensible: with only US, deepinfra (US
    datacenter) is denied while the CN providers are NOT (CN not in set)."""
    handler, _ = _proxy_handler()
    backend = _backend(handler, blocked_countries=frozenset({"US"}))
    await backend.health()
    assert backend._regime_provider_deny == frozenset({"deepinfra"})
    await backend.aclose()


async def test_regime_deny_null_metadata_not_denied() -> None:
    """A provider with null headquarters AND null/empty datacenters has no
    positive regime evidence → NOT denied (operator can add via
    blocked_providers)."""
    handler, _ = _proxy_handler()
    backend = _backend(handler)
    await backend.health()
    assert "unknown-co" not in backend._regime_provider_deny
    await backend.aclose()


async def test_regime_deny_providers_failure_leaves_previous_set() -> None:
    """A /providers failure does NOT fail catalog refresh — residency
    enforcement simply stays empty (first refresh) until the next refresh
    succeeds. /models still populates the catalog."""
    handler, _ = _proxy_handler(providers_status=500)
    backend = _backend(handler)
    await backend.health()
    assert backend.advertised_models == frozenset(
        {"openai/model-a0f5", "anthropic/model-a0aa"}
    )
    assert backend._regime_provider_deny == frozenset()
    await backend.aclose()


async def test_alibaba_caught_by_datacenters_not_headquarters() -> None:
    """Alibaba's headquarters is SG (not in CN/RU/KP), but its datacenters
    include CN. The datacenters check is what catches it — a headquarters-only
    filter would wrongly miss it. This pins the datacenters-OR-headquarters
    rule."""
    handler, _ = _proxy_handler()
    backend = _backend(handler)
    await backend.health()
    assert "alibaba" in backend._regime_provider_deny
    await backend.aclose()


# ---------- request-time provider.ignore residency injection ---------------


async def test_chat_injects_provider_ignore_with_deny_set() -> None:
    """Outbound chat body carries provider.ignore with the derived deny set;
    the request still sends (residency enforced per-request, NOT at catalog)
    so OpenRouter serves it via a non-regime provider."""
    handler, rec = _proxy_handler()
    backend = _backend(handler)
    await backend.health()
    await backend.chat_completions({"model": "openai/model-a0f5", "messages": [{"role": "user", "content": "hi"}]})
    chat_proxy = [r for r in rec.proxy if httpx.URL(json.loads(r.content)["url"]).path.endswith("/chat/completions")]
    assert len(chat_proxy) == 1
    sent = json.loads(_unb64(json.loads(chat_proxy[0].content)["body_b64"]))
    assert sent["provider"] == {"ignore": ["alibaba", "z-ai"]}
    assert sent["stream"] is False
    await backend.aclose()


async def test_chat_no_provider_key_when_deny_set_empty() -> None:
    """When the deny set is empty (no regime providers discovered), the
    outbound body has NO `provider` key — don't send an empty ignore list."""
    handler, rec = _proxy_handler(providers_status=500)
    backend = _backend(handler)
    await backend.health()
    await backend.chat_completions({"model": "openai/model-a0f5", "messages": [{"role": "user", "content": "hi"}]})
    chat_proxy = [r for r in rec.proxy if httpx.URL(json.loads(r.content)["url"]).path.endswith("/chat/completions")]
    sent = json.loads(_unb64(json.loads(chat_proxy[0].content)["body_b64"]))
    assert "provider" not in sent
    await backend.aclose()


async def test_chat_preserves_caller_provider_preferences() -> None:
    """A caller-supplied `provider` preference is merged with the residency
    ignore list, not overwritten."""
    handler, rec = _proxy_handler()
    backend = _backend(handler)
    await backend.health()
    await backend.chat_completions(
        {
            "model": "openai/model-a0f5",
            "messages": [{"role": "user", "content": "hi"}],
            "provider": {"order": ["openai"], "ignore": ["someother"]},
        }
    )
    chat_proxy = [r for r in rec.proxy if httpx.URL(json.loads(r.content)["url"]).path.endswith("/chat/completions")]
    sent = json.loads(_unb64(json.loads(chat_proxy[0].content)["body_b64"]))
    assert sent["provider"]["order"] == ["openai"]
    assert set(sent["provider"]["ignore"]) == {"alibaba", "z-ai", "someother"}
    await backend.aclose()


async def test_responses_path_also_injects_residency() -> None:
    """The Responses path (translated to chat) injects provider.ignore too —
    residency enforcement is not bypassed by the Responses→chat translation."""
    handler, rec = _proxy_handler()
    backend = _backend(handler)
    await backend.health()
    await backend.responses({"model": "openai/model-a0f5", "input": "say hi"})
    chat_proxy = [r for r in rec.proxy if httpx.URL(json.loads(r.content)["url"]).path.endswith("/chat/completions")]
    sent = json.loads(_unb64(json.loads(chat_proxy[0].content)["body_b64"]))
    assert sent["provider"] == {"ignore": ["alibaba", "z-ai"]}
    await backend.aclose()


# ---------- honest-advisory usage_snapshot + 402 exhausted -----------------


async def test_usage_snapshot_is_honest_not_local_free_stub() -> None:
    handler, _ = _proxy_handler()
    backend = _backend(handler)
    await backend.health()
    snap = await backend.usage_snapshot()
    assert snap.remaining_fraction == 1.0
    assert snap.weekly_exhausted is False
    assert snap.cooldown_until_ts is None
    await backend.aclose()


async def test_quota_snapshot_is_none() -> None:
    handler, _ = _proxy_handler()
    backend = _backend(handler)
    assert await backend.quota_snapshot() is None
    await backend.aclose()


async def test_402_sets_exhausted_cooldown_and_rate_limited() -> None:
    """402 (insufficient credits) → rate_limited BackendError + an in-memory
    exhausted cooldown so usage_snapshot reports exhausted and dispatch
    rotates to another cell. The credits-expiry motivation is why this backend
    exists, so exhausted-credits is the one signal worth caching."""
    handler, _ = _proxy_handler(chat_upstream_status=402)
    backend = _backend(handler)
    await backend.health()
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        await backend.chat_completions({"model": "openai/model-a0f5", "messages": [{"role": "user", "content": "hi"}]}, handle)
    assert exc_info.value.classification == "rate_limited"
    assert exc_info.value.status_code == 402
    snap = await backend.usage_snapshot()
    assert snap.weekly_exhausted is True
    assert snap.remaining_fraction == 0.0
    assert snap.cooldown_until_ts is not None
    await backend.aclose()


async def test_402_stream_sets_exhausted_cooldown() -> None:
    chunks = [b'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n"]
    handler, _ = _proxy_handler(chat_sse=chunks, stream_upstream_status=402)
    backend = _backend(handler)
    await backend.health()
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        async for _ in backend.chat_completions_stream(
            {"model": "openai/model-a0f5", "messages": [{"role": "user", "content": "hi"}]}, handle
        ):
            pass
    assert exc_info.value.classification == "rate_limited"
    snap = await backend.usage_snapshot()
    assert snap.weekly_exhausted is True
    await backend.aclose()


async def test_402_responses_stream_sets_exhausted_cooldown() -> None:
    """The main Codex route (responses_stream) MUST also flip the exhausted
    cooldown on 402 — the credits-expiry motivation is why this backend
    exists, so exhausted-credits is the one signal worth caching on every
    dispatch path. The shared generator classifies 402 as transient; the
    backend catches it and re-raises rate_limited."""
    chunks = [b'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n"]
    handler, _ = _proxy_handler(chat_sse=chunks, stream_upstream_status=402)
    backend = _backend(handler)
    await backend.health()
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        async for _ in backend.responses_stream({"model": "openai/model-a0f5", "input": "say hi", "stream": True}, handle):
            pass
    assert exc_info.value.classification == "rate_limited"
    assert exc_info.value.status_code == 402
    snap = await backend.usage_snapshot()
    assert snap.weekly_exhausted is True
    assert snap.cooldown_until_ts is not None
    await backend.aclose()


# ---------- health ----------------------------------------------------------


async def test_health_reports_network_when_custody_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    backend = _backend(handler, catalog_refresh_s=0)
    h = await backend.health()
    assert h.available is False
    assert h.reason == "network"
    await backend.aclose()


async def test_health_reports_no_key_when_standin_mint_fails() -> None:
    handler, _ = _proxy_handler(standin_status=503)
    backend = _backend(handler, catalog_refresh_s=0)
    h = await backend.health()
    assert h.available is False
    assert h.reason == "no-key"
    await backend.aclose()


async def test_health_reports_no_key_when_custody_proxy_503() -> None:
    handler, _ = _proxy_handler(custody_status={"/models": 503})
    backend = _backend(handler, catalog_refresh_s=0)
    h = await backend.health()
    assert h.available is False
    assert h.reason == "no-key"
    await backend.aclose()


async def test_health_reports_auth_invalid_when_upstream_401() -> None:
    """Upstream OpenRouter 401 on the catalog /models call → auth_invalid
    (the stand-in is innocent; OpenRouter rejected the key credential proxy injected)."""
    handler, _ = _proxy_handler(models_status=401)
    backend = _backend(handler, catalog_refresh_s=0)
    h = await backend.health()
    assert h.available is False
    assert h.reason == "auth_invalid"
    await backend.aclose()


async def test_health_unhealthy_after_poll_sets_cooldown() -> None:
    state = {"up": True}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if not state["up"]:
            raise httpx.ConnectError("down")
        if path == "/v1/standin":
            return httpx.Response(200, json={"token": "t", "expires_at": _FAR_FUTURE_EXPIRES_AT})
        if path == "/v1/proxy":
            envelope = json.loads(request.content)
            target = httpx.URL(envelope["url"]).path
            if target.endswith("/models"):
                return _wrap_upstream(200, json.dumps(_DEFAULT_MODELS).encode(), {"content-type": "application/json"})
            if target.endswith("/providers"):
                return _wrap_upstream(200, json.dumps(_DEFAULT_PROVIDERS).encode(), {"content-type": "application/json"})
        return httpx.Response(404)

    backend = _backend(handler, catalog_refresh_s=0)
    await backend.health()  # first poll succeeds
    snap = await backend.usage_snapshot()
    assert snap.cooldown_until_ts is None
    state["up"] = False
    await backend.health()  # outage
    snap = await backend.usage_snapshot()
    assert snap.cooldown_until_ts is not None
    await backend.aclose()


# ---------- refresh_advertised_models ---------------------------------------


async def test_refresh_advertised_models_forces_refetch() -> None:
    poll = {"n": 0}

    def models_fn() -> dict[str, Any]:
        poll["n"] += 1
        if poll["n"] == 1:
            return _models_payload(_model("openai/model-a0f5", 128_000))
        return _models_payload(_model("openai/model-a0f5", 128_000), _model("anthropic/model-a0aa", 200_000))

    captured = {"models": _DEFAULT_MODELS}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/standin":
            return httpx.Response(200, json={"token": "t", "expires_at": _FAR_FUTURE_EXPIRES_AT})
        if path == "/v1/proxy":
            envelope = json.loads(request.content)
            target = httpx.URL(envelope["url"]).path
            if target.endswith("/models"):
                payload = models_fn()
                captured["models"] = payload
                return _wrap_upstream(200, json.dumps(payload).encode(), {"content-type": "application/json"})
            if target.endswith("/providers"):
                return _wrap_upstream(200, json.dumps(_DEFAULT_PROVIDERS).encode(), {"content-type": "application/json"})
        return httpx.Response(404)

    backend = _backend(handler, catalog_refresh_s=3600)
    await backend.health()
    assert backend.advertised_models == frozenset({"openai/model-a0f5"})
    await backend.refresh_advertised_models()
    assert backend.advertised_models == frozenset({"openai/model-a0f5", "anthropic/model-a0aa"})
    await backend.aclose()


# ---------- proxy envelope + stand-in contract ------------------------------


async def test_standin_mint_sends_explicit_openrouter_scope() -> None:
    """CRITICAL regression guard: the stand-in mint MUST send
    `scope="openrouter"` explicitly (credential_proxy omits scope and gets the
    default `provider-upstream`). Forgetting the scope → credential proxy routes to the
    wrong pass key → 401 scope mismatch."""
    handler, rec = _proxy_handler()
    backend = _backend(handler)
    await backend.health()
    assert len(rec.standin) == 1
    mint_body = json.loads(rec.standin[0].content)
    assert mint_body["scope"] == OPENROUTER_SCOPE
    assert mint_body["account"] == "primary"
    assert mint_body["ttl_seconds"] == OPENROUTER_STANDIN_TTL_S
    await backend.aclose()


async def test_standin_expires_at_absolute_epoch_far_future_is_reused() -> None:
    """credential proxy returns expires_at as an ABSOLUTE Unix epoch. A far-future
    epoch keeps the token fresh: two _ensure_standin calls mint ONCE."""
    handler, rec = _proxy_handler()
    backend = _backend(handler)
    t1 = await backend._ensure_standin()
    t2 = await backend._ensure_standin()
    assert t1 == t2 == "test-standin"
    assert len(rec.standin) == 1  # reused, not re-minted
    await backend.aclose()


async def test_buffered_envelope_shape_and_no_caller_auth_headers() -> None:
    """The buffered /v1/proxy envelope has exactly {url,method,headers,
    body_b64}; the inner headers carry ONLY content negotiation — NEVER
    `authorization` or `chatgpt-account-id` (credential proxy scrubs those and overlays
    the real key). The stand-in rides the OUTER request to credential proxy as a Bearer."""
    handler, rec = _proxy_handler()
    backend = _backend(handler)
    await backend.health()
    assert len(rec.proxy) >= 1
    catalog_req = rec.proxy[0]  # the /models GET
    envelope = json.loads(catalog_req.content)
    assert set(envelope.keys()) == {"url", "method", "headers", "body_b64"}
    assert envelope["url"] == "https://openrouter.ai/api/v1/models"
    assert envelope["method"] == "GET"
    inner = envelope["headers"]
    assert "authorization" not in inner
    assert "chatgpt-account-id" not in inner
    assert envelope["body_b64"] == ""  # GET carries no body
    assert catalog_req.headers.get("authorization") == "Bearer test-standin"
    await backend.aclose()


async def test_b64_unb64_roundtrip() -> None:
    assert OpenRouterBackend._b64(b"") == ""
    assert OpenRouterBackend._unb64("") == b""
    payload = b'{"model":"openai/model-a0f5","stream":false}'
    assert OpenRouterBackend._unb64(OpenRouterBackend._b64(payload)) == payload


# ---------- dispatch (chat + responses, stream + non-stream) ----------------


async def test_chat_completions_nonstream() -> None:
    handler, rec = _proxy_handler()
    backend = _backend(handler)
    handle = CallHandle()
    reply = await backend.chat_completions(
        {
            "model": "openai/model-a0f5",
            "reasoning": {"effort": "high"},
            "parallel_tool_calls": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        handle,
    )
    assert reply["choices"][0]["message"]["content"] == "hi there"
    assert handle.upstream_status == 200
    chat_proxy = [r for r in rec.proxy if httpx.URL(json.loads(r.content)["url"]).path.endswith("/chat/completions")]
    assert len(chat_proxy) == 1
    env = json.loads(chat_proxy[0].content)
    assert env["url"] == "https://openrouter.ai/api/v1/chat/completions"
    sent = json.loads(_unb64(env["body_b64"]))
    assert sent["model"] == "openai/model-a0f5"
    assert sent["stream"] is False
    assert "reasoning" not in sent
    assert "parallel_tool_calls" not in sent
    await backend.aclose()


async def test_chat_completions_upstream_401_no_re_mint() -> None:
    """Upstream OpenRouter 401 → auth_invalid; the stand-in is INNOCENT — no
    re-mint. The catalog path stays healthy (no cooldown)."""
    handler, rec = _proxy_handler(chat_upstream_status=401)
    backend = _backend(handler)
    await backend.health()  # catalog mints once (standin 1)
    mint_after_health = len(rec.standin)
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        await backend.chat_completions({"model": "openai/model-a0f5", "messages": [{"role": "user", "content": "hi"}]}, handle)
    assert exc_info.value.classification == "auth_invalid"
    assert exc_info.value.status_code == 401
    assert len(rec.standin) == mint_after_health  # no re-mint
    snap = await backend.usage_snapshot()
    assert snap.cooldown_until_ts is None
    await backend.aclose()


async def test_chat_completions_custody_401_re_mints_then_transient() -> None:
    """credential proxy rejects the STAND-IN (real HTTP 401 on /v1/proxy for the chat
    target) → invalidate, re-mint ONCE, retry; still 401 → transient
    BackendError (NOT auth_invalid) + reason "network" + cooldown. The catalog
    path stays healthy."""
    handler, rec = _proxy_handler(custody_status={"/chat/completions": 401})
    backend = _backend(handler)
    await backend.health()  # catalog mints (standin 1)
    with pytest.raises(BackendError) as exc_info:
        await backend.chat_completions({"model": "openai/model-a0f5", "messages": [{"role": "user", "content": "hi"}]})
    assert exc_info.value.classification == "transient"
    assert exc_info.value.status_code == 401
    assert len(rec.standin) == 2  # catalog mint + one retry mint
    snap = await backend.usage_snapshot()
    assert snap.cooldown_until_ts is not None
    await backend.aclose()


async def test_chat_completions_custody_503_reports_no_key() -> None:
    handler, rec = _proxy_handler(custody_status={"/chat/completions": 503})
    backend = _backend(handler)
    await backend.health()
    with pytest.raises(BackendError) as exc_info:
        await backend.chat_completions({"model": "openai/model-a0f5", "messages": [{"role": "user", "content": "hi"}]})
    assert exc_info.value.classification == "transient"
    assert backend._last_health_reason == "no-key"
    await backend.aclose()


async def test_chat_completions_429_is_rate_limited() -> None:
    handler, _ = _proxy_handler(chat_upstream_status=429)
    backend = _backend(handler)
    await backend.health()
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        await backend.chat_completions({"model": "openai/model-a0f5", "messages": [{"role": "user", "content": "hi"}]}, handle)
    assert exc_info.value.classification == "rate_limited"
    assert exc_info.value.status_code == 429
    await backend.aclose()


async def test_chat_completions_5xx_is_transient() -> None:
    handler, _ = _proxy_handler(chat_upstream_status=503)
    backend = _backend(handler)
    await backend.health()
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        await backend.chat_completions({"model": "openai/model-a0f5", "messages": [{"role": "user", "content": "hi"}]}, handle)
    assert exc_info.value.classification == "transient"
    await backend.aclose()


async def test_responses_nonstream() -> None:
    handler, _ = _proxy_handler()
    backend = _backend(handler)
    resp = await backend.responses({"model": "openai/model-a0f5", "input": "say hi"})
    assert resp["object"] == "response"
    assert resp["status"] == "completed"
    assert resp["usage"]["input_tokens"] == 5
    assert resp["usage"]["output_tokens"] == 2
    msg = next(o for o in resp["output"] if o["type"] == "message")
    assert msg["content"][0]["text"] == "hi there"
    await backend.aclose()


async def test_responses_stream_emits_responses_sse() -> None:
    chunks = [
        b'data: {"id":"x","model":"openai/model-a0f5","choices":[{"delta":{"content":"Hel"}}]}\n\n',
        b'data: {"id":"x","choices":[{"delta":{"content":"lo"}}]}\n\n',
        b'data: {"id":"x","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n',
        b"data: [DONE]\n\n",
    ]
    handler, rec = _proxy_handler(chat_sse=chunks)
    backend = _backend(handler)
    await backend.health()  # populate the regime deny set before streaming
    handle = CallHandle()
    events: list[dict[str, Any]] = []
    async for raw in backend.responses_stream({"model": "openai/model-a0f5", "input": "say hi", "stream": True}, handle):
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
    assert handle.stream_summary is not None
    assert handle.stream_summary.completed_response is not None
    # Residency was injected on the streamed chat body too.
    stream_env = json.loads(rec.stream[0].content)
    sent = json.loads(_unb64(stream_env["body_b64"]))
    assert sent["provider"] == {"ignore": ["alibaba", "z-ai"]}
    await backend.aclose()


async def test_responses_stream_upstream_401_is_auth_invalid() -> None:
    chunks = [b'data: {"id":"x","choices":[{"delta":{"content":"Hel"}}]}\n\n', b"data: [DONE]\n\n"]
    handler, rec = _proxy_handler(chat_sse=chunks, stream_upstream_status=401)
    backend = _backend(handler)
    await backend.health()
    mints_after_health = len(rec.standin)
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        async for _ in backend.responses_stream({"model": "openai/model-a0f5", "input": "say hi", "stream": True}, handle):
            pass
    assert exc_info.value.classification == "auth_invalid"
    assert handle.upstream_status == 401
    assert len(rec.standin) == mints_after_health  # no re-mint
    await backend.aclose()


async def test_chat_completions_stream_passthrough() -> None:
    chunks = [b'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n"]
    handler, _ = _proxy_handler(chat_sse=chunks)
    backend = _backend(handler)
    handle = CallHandle()
    out = b"".join(
        [c async for c in backend.chat_completions_stream({"model": "openai/model-a0f5", "messages": [{"role": "user", "content": "hi"}]}, handle)]
    )
    assert b'data: {"id":"x"' in out
    assert b"[DONE]" in out
    assert handle.upstream_status == 200
    assert handle.stream_summary is None
    await backend.aclose()


async def test_transport_error_marks_unhealthy() -> None:
    handler, _ = _proxy_handler(chat_error=httpx.ConnectError("openrouter down"))
    backend = _backend(handler)
    await backend.health()  # first /models poll → healthy
    snap = await backend.usage_snapshot()
    assert snap.cooldown_until_ts is None
    with pytest.raises(BackendError):
        await backend.chat_completions({"model": "openai/model-a0f5", "messages": [{"role": "user", "content": "hi"}]})
    snap = await backend.usage_snapshot()
    assert snap.cooldown_until_ts is not None
    await backend.aclose()


# ---------- default-OFF registration ----------------------------------------


def test_backend_absent_from_build_when_env_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CALLOSUM_OPENROUTER_ENABLED unset → build_runtime_backends does NOT
    append the backend. Default-OFF is a no-op for live routing."""
    from callosum.__main__ import build_runtime_backends
    from callosum.config import Config
    from callosum.operator_state import OperatorState

    monkeypatch.delenv("CALLOSUM_OPENROUTER_ENABLED", raising=False)
    monkeypatch.setenv("CALLOSUM_LOCAL_DISABLED", "1")
    monkeypatch.setenv("CALLOSUM_LITELLM_GATEWAY_ENABLED", "0")
    monkeypatch.delenv("CALLOSUM_OLLAMA_CLOUD_ENABLED", raising=False)
    cfg = Config()
    backends = build_runtime_backends(cfg, operator_state=OperatorState(tmp_path / "op.sqlite"))
    assert all(getattr(b, "id", None) != "openrouter" for b in backends)


def test_backend_present_when_env_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from callosum.__main__ import build_runtime_backends
    from callosum.backends.openrouter import OpenRouterBackend
    from callosum.config import Config
    from callosum.operator_state import OperatorState

    monkeypatch.setenv("CALLOSUM_OPENROUTER_ENABLED", "1")
    monkeypatch.setenv("CALLOSUM_LOCAL_DISABLED", "1")
    monkeypatch.setenv("CALLOSUM_LITELLM_GATEWAY_ENABLED", "0")
    monkeypatch.delenv("CALLOSUM_OLLAMA_CLOUD_ENABLED", raising=False)
    cfg = Config()
    backends = build_runtime_backends(cfg, operator_state=OperatorState(tmp_path / "op.sqlite"))
    ids = [getattr(b, "id", None) for b in backends]
    assert "openrouter" in ids
    or_backend = next(b for b in backends if getattr(b, "id", None) == "openrouter")
    assert or_backend.kind == "openrouter"
    assert isinstance(or_backend, OpenRouterBackend)
    # Proxy custody: no key attribute exists. The backend holds only a
    # short-TTL stand-in, minted lazily.
    assert not hasattr(or_backend, "key_source")
    assert not hasattr(or_backend, "api_key")


def test_env_blocked_countries_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CALLOSUM_OPENROUTER_BLOCKED_COUNTRIES overrides the default regime set."""
    import asyncio

    from callosum.__main__ import build_runtime_backends
    from callosum.config import Config
    from callosum.operator_state import OperatorState

    monkeypatch.setenv("CALLOSUM_OPENROUTER_ENABLED", "1")
    monkeypatch.setenv("CALLOSUM_LOCAL_DISABLED", "1")
    monkeypatch.setenv("CALLOSUM_LITELLM_GATEWAY_ENABLED", "0")
    monkeypatch.delenv("CALLOSUM_OLLAMA_CLOUD_ENABLED", raising=False)
    monkeypatch.setenv("CALLOSUM_OPENROUTER_BLOCKED_COUNTRIES", "US")
    cfg = Config()
    backends = build_runtime_backends(cfg, operator_state=OperatorState(tmp_path / "op.sqlite"))
    or_backend = next(b for b in backends if getattr(b, "id", None) == "openrouter")
    assert or_backend._blocked_countries == frozenset({"US"})
    # aclose is async; run it so the owned client closes cleanly.
    asyncio.run(or_backend.aclose())


def test_env_exclude_families_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from callosum.__main__ import build_runtime_backends
    from callosum.config import Config
    from callosum.operator_state import OperatorState

    monkeypatch.setenv("CALLOSUM_OPENROUTER_ENABLED", "1")
    monkeypatch.setenv("CALLOSUM_LOCAL_DISABLED", "1")
    monkeypatch.setenv("CALLOSUM_LITELLM_GATEWAY_ENABLED", "0")
    monkeypatch.delenv("CALLOSUM_OLLAMA_CLOUD_ENABLED", raising=False)
    monkeypatch.setenv("CALLOSUM_OPENROUTER_EXCLUDE_FAMILIES", "model-a0g3,model-a0g1")
    cfg = Config()
    backends = build_runtime_backends(cfg, operator_state=OperatorState(tmp_path / "op.sqlite"))
    or_backend = next(b for b in backends if getattr(b, "id", None) == "openrouter")
    assert or_backend._exclude_families == frozenset({"model-a0g3", "model-a0g1"})
    asyncio.run(or_backend.aclose())