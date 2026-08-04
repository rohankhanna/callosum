"""Tests for the Ollama Cloud backend (sub-slice 1: catalog + classification).

The backend talks to the local ollama daemon (localhost:11434) that holds the
Ollama Cloud auth under `ollama signin`. Callosum holds NO credential. These
tests cover the sub-slice-1 surface only — catalog discovery from `/api/tags`
(with the `:cloud` suffix filter that partitions local vs. cloud),
`ModelMetadata` synthesis (so cloud models join the cell grid in the REMOTE
band), per-model capabilities from `/api/show`, honest-advisory `usage_snapshot`
(NOT the local free-stub), health, refresh, and that the dispatch stubs raise
`NotImplementedError` (chat dispatch is wired in sub-slice 2).

Routing-classification is asserted here at the catalog/metadata level
(remote-band priority, remote `cost_rank`) and in the integration fixtures
(`test_routing_mode_matrix`, `test_selector_routing`) at the lane level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from callosum.backends.ollama_cloud import (
    CLOUD_PRIORITY_OFFSET,
    OllamaCloudBackend,
)


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


def _daemon_handler(
    tags: dict[str, Any],
    shows: dict[str, dict[str, Any]] | None = None,
) -> Any:
    """Route /api/tags and /api/show to canned payloads. `shows` maps model
    name → /api/show response; a model absent from `shows` gets a 404 so the
    capabilities fallback path is exercised."""
    shows = shows or {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=tags)
        if request.url.path == "/api/show":
            try:
                body = request.read()
                import json

                name = json.loads(body).get("name")
            except Exception:
                return httpx.Response(400)
            if name in shows:
                return httpx.Response(200, json=shows[name])
            return httpx.Response(404)
        return httpx.Response(404)

    return handler


# ---------- catalog discovery + suffix filter ------------------------------


async def test_advertised_models_empty_before_first_poll() -> None:
    backend = OllamaCloudBackend(
        id="ollama-cloud",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_tags_payload("model-a0d2:cloud"))),
    )
    assert backend.advertised_models == frozenset()
    h = await backend.health()  # forces a refresh
    assert h.available is True
    assert backend.advertised_models == frozenset({"model-a0d2:cloud"})
    await backend.aclose()


async def test_suffix_filter_keeps_only_cloud_models() -> None:
    """The local/cloud partition: only `:cloud`-suffixed models are cloud-
    metered; bare local models stay on the LocalModelRegistry / litellm_gateway path
    and must NOT appear in this backend's catalog."""
    backend = OllamaCloudBackend(
        id="ollama-cloud",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                json=_tags_payload("model-a0d2:cloud", "model-a0b4", "model-a0f3:cloud", "model-a0d5"),
            )
        ),
    )
    await backend.health()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud", "model-a0f3:cloud"})
    await backend.aclose()


async def test_custom_suffix_filter() -> None:
    backend = OllamaCloudBackend(
        id="ollama-cloud",
        model_suffix="-remote",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200, json=_tags_payload("model-a0d2-remote", "model-a0e4:cloud")
            )
        ),
    )
    await backend.health()
    assert backend.advertised_models == frozenset({"model-a0d2-remote"})
    await backend.aclose()


async def test_catalog_picks_up_new_models_on_refresh() -> None:
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(200, json=_tags_payload("model-a0d2:cloud"))
        return httpx.Response(200, json=_tags_payload("model-a0d2:cloud", "model-a0f3:cloud"))

    backend = OllamaCloudBackend(
        id="ollama-cloud",
        catalog_refresh_s=0,  # always refresh
        transport=httpx.MockTransport(handler),
    )
    await backend.health()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud"})
    await backend.health()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud", "model-a0f3:cloud"})
    await backend.aclose()


# ---------- model_metadata synthesis (remote band) --------------------------


async def test_model_metadata_marks_cloud_models_remote_band() -> None:
    backend = OllamaCloudBackend(
        id="ollama-cloud",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=_tags_payload("model-a0d2:cloud", "model-a0f3:cloud"))
        ),
    )
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
    # Ordering preserved: first cataloged model gets the base offset.
    assert md["model-a0d2:cloud"].priority == CLOUD_PRIORITY_OFFSET
    assert md["model-a0f3:cloud"].priority == CLOUD_PRIORITY_OFFSET + 1
    await backend.aclose()


# ---------- /api/show → cell_capabilities -----------------------------------


async def test_cell_capabilities_from_show() -> None:
    backend = OllamaCloudBackend(
        id="ollama-cloud",
        transport=httpx.MockTransport(
            _daemon_handler(
                _tags_payload("model-a0d2:cloud"),
                shows={
                    "model-a0d2:cloud": _show_payload(
                        capabilities=["completion", "tools", "vision"],
                        context_length=128_000,
                        parameter_count=20_000_000_000,
                    )
                },
            )
        ),
    )
    await backend.health()  # triggers _refresh_capabilities
    caps = backend.cell_capabilities("model-a0d2:cloud")
    assert caps.cost_rank == 10  # remote default
    assert caps.supports_tools is True
    assert "text" in caps.modalities
    assert "image" in caps.modalities
    assert caps.context_window == 128_000
    assert caps.parameter_count == 20_000_000_000
    await backend.aclose()


async def test_cell_capabilities_fallback_when_show_missing() -> None:
    """A model in the catalog whose /api/show 404s (or hasn't been probed yet)
    falls back to conservative text-only/no-tools defaults rather than
    erroring — the request can still dispatch."""
    backend = OllamaCloudBackend(
        id="ollama-cloud",
        transport=httpx.MockTransport(
            _daemon_handler(_tags_payload("model-a0d2:cloud"), shows={})
        ),
    )
    await backend.health()
    caps = backend.cell_capabilities("model-a0d2:cloud")
    assert caps.cost_rank == 10
    assert caps.supports_tools is False
    assert caps.modalities == frozenset({"text"})
    await backend.aclose()


async def test_cell_capabilities_fallback_before_first_refresh() -> None:
    """Synchronous cell_capabilities called before any catalog refresh
    returns the safe default (no crash, no stale state)."""
    backend = OllamaCloudBackend(
        id="ollama-cloud",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_tags_payload("model-a0d2:cloud"))),
    )
    caps = backend.cell_capabilities("model-a0d2:cloud")
    assert caps.cost_rank == 10
    assert caps.supports_tools is False
    await backend.aclose()


# ---------- honest-advisory usage_snapshot ----------------------------------


async def test_usage_snapshot_is_honest_not_local_free_stub() -> None:
    """Cloud is NOT free: usage_snapshot reports remaining_fraction=1.0
    ("full, eligible, no signal yet"), NOT the local free-stub's 0.001 that
    suppresses primary selection. weekly_exhausted is False (we genuinely
    don't know from headers), and there is no cooldown on a healthy daemon."""
    backend = OllamaCloudBackend(
        id="ollama-cloud",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_tags_payload("model-a0d2:cloud"))),
    )
    await backend.health()
    snap = await backend.usage_snapshot()
    assert snap.remaining_fraction == 1.0
    assert snap.weekly_exhausted is False
    assert snap.cooldown_until_ts is None
    await backend.aclose()


async def test_quota_snapshot_is_none() -> None:
    backend = OllamaCloudBackend(
        id="ollama-cloud",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_tags_payload("model-a0d2:cloud"))),
    )
    assert await backend.quota_snapshot() is None
    await backend.aclose()


# ---------- health ----------------------------------------------------------


async def test_health_reports_network_when_daemon_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    backend = OllamaCloudBackend(
        id="ollama-cloud",
        catalog_refresh_s=0,
        transport=httpx.MockTransport(handler),
    )
    h = await backend.health()
    assert h.available is False
    assert h.reason == "network"
    await backend.aclose()


async def test_health_unhealthy_after_poll_sets_cooldown() -> None:
    """Once we've polled successfully, a subsequent outage sets a short
    cooldown so the backend is excluded from routable backends. On cold
    start (never polled) there is no cooldown so the fleet isn't empty."""
    state = {"up": True}
    tags = _tags_payload("model-a0d2:cloud")

    def handler(request: httpx.Request) -> httpx.Response:
        if not state["up"]:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json=tags)

    backend = OllamaCloudBackend(
        id="ollama-cloud",
        catalog_refresh_s=0,
        transport=httpx.MockTransport(handler),
    )
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
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(200, json=_tags_payload("model-a0d2:cloud"))
        return httpx.Response(200, json=_tags_payload("model-a0d2:cloud", "model-a0f3:cloud"))

    backend = OllamaCloudBackend(
        id="ollama-cloud",
        catalog_refresh_s=3600,  # long TTL so normal access won't refresh
        transport=httpx.MockTransport(handler),
    )
    await backend.health()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud"})
    # Within TTL, a plain health() won't re-fetch; refresh_advertised_models must.
    await backend.refresh_advertised_models()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud", "model-a0f3:cloud"})
    await backend.aclose()


# ---------- dispatch stubs (deferred to sub-slice 2) ------------------------


async def test_dispatch_stubs_raise_not_implemented() -> None:
    backend = OllamaCloudBackend(
        id="ollama-cloud",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_tags_payload("model-a0d2:cloud"))),
    )
    with pytest.raises(NotImplementedError):
        await backend.chat_completions({"model": "model-a0d2:cloud"})
    with pytest.raises(NotImplementedError):
        await backend.responses({"model": "model-a0d2:cloud"})
    # Stream generators raise on first iteration.
    with pytest.raises(NotImplementedError):
        async for _ in backend.chat_completions_stream({"model": "model-a0d2:cloud"}):
            pass
    with pytest.raises(NotImplementedError):
        async for _ in backend.responses_stream({"model": "model-a0d2:cloud"}):
            pass
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