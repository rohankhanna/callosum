"""Tests for CellRecommender — upstream-classifier-driven router cell pick.

Covers:
- Parse valid model+effort output from chatty/clean upstream responses.
- Fall back when upstream returns garbage / times out / errors.
- Cache hits dedupe identical prompts.
- Cache TTL expires.
- /status.recommender shape.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

from codex_proxy.app import create_app
from codex_proxy.backend import CallHandle, HealthStatus, UsageSnapshot
from codex_proxy.cell_grid import Cell
from codex_proxy.cell_recommender import (
    CellRecommender,
    _parse_cell_from_output,
)
from codex_proxy.errors import BackendError


CELLS = [
    Cell(model="model-a0c3", reasoning_effort="low", context_window=None),
    Cell(model="model-a0c3", reasoning_effort="medium", context_window=None),
    Cell(model="model-a0e7", reasoning_effort="high", context_window=None),
]


def test_parse_exact_match() -> None:
    assert _parse_cell_from_output("model-a0c3 medium", CELLS) == CELLS[1]


def test_parse_case_insensitive() -> None:
    assert _parse_cell_from_output("MODEL-A0C3 MEDIUM", CELLS) == CELLS[1]


def test_parse_chatty_output_substring_match() -> None:
    """Model didn't follow instructions and explained itself — still recover."""
    txt = "I recommend model-a0e7 high because the prompt requires complex reasoning."
    assert _parse_cell_from_output(txt, CELLS) == CELLS[2]


def test_parse_unknown_cell_returns_none() -> None:
    assert _parse_cell_from_output("claude-opus high", CELLS) is None


def test_parse_empty_returns_none() -> None:
    assert _parse_cell_from_output("", CELLS) is None
    assert _parse_cell_from_output("   ", CELLS) is None


# ---------- async recommender behavior ----------


@dataclass
class _FakeBackend:
    """Minimal Backend stub returning a canned Responses-API result."""
    id: str = "fake"
    kind: str = "fake"
    advertised_models = frozenset({"model-a0c3"})
    canned_text: str | None = "model-a0e7 high"
    raise_error: bool = False

    async def health(self) -> HealthStatus:
        return HealthStatus(available=True, reason="ok")

    async def usage_snapshot(self) -> UsageSnapshot:
        return UsageSnapshot(
            remaining_fraction=None, cooldown_until_ts=None,
            weekly_exhausted=False, probed_at_ts=time.time(),
        )

    async def quota_snapshot(self) -> None:
        return None

    async def responses(self, body: dict[str, Any], handle: CallHandle) -> dict[str, Any]:
        if self.raise_error:
            raise BackendError(classification="transient", message="boom")
        text = self.canned_text or ""
        return {"output": [{"content": [{"text": text}]}]}

    async def aclose(self) -> None:
        pass


def _make_recommender(backend: _FakeBackend, **kwargs) -> CellRecommender:
    return CellRecommender(
        cheap_backend=backend,
        cheap_cell=CELLS[0],
        upstream_timeout_s=2.0,
        **kwargs,
    )


def test_recommend_upstream_path_picks_returned_cell() -> None:
    backend = _FakeBackend(canned_text="model-a0e7 high")
    rec = _make_recommender(backend)
    body = {"messages": [{"role": "user", "content": "explain quicksort"}]}
    out = asyncio.run(rec.recommend(body, allowed_cells=CELLS, fallback=CELLS[0]))
    assert out.cell == CELLS[2]
    assert out.source == "upstream"


def test_recommend_falls_back_when_upstream_raises() -> None:
    backend = _FakeBackend(raise_error=True)
    rec = _make_recommender(backend)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    out = asyncio.run(rec.recommend(body, allowed_cells=CELLS, fallback=CELLS[0]))
    assert out.source == "fallback"
    assert out.cell == CELLS[0]


def test_recommend_falls_back_on_unparseable_output() -> None:
    backend = _FakeBackend(canned_text="claude-opus high")
    rec = _make_recommender(backend)
    body = {"messages": [{"role": "user", "content": "hi"}]}
    out = asyncio.run(rec.recommend(body, allowed_cells=CELLS, fallback=CELLS[1]))
    assert out.source == "fallback"
    assert out.cell == CELLS[1]


def test_recommend_cache_hits_dedup_identical_prompts() -> None:
    backend = _FakeBackend(canned_text="model-a0e7 high")
    rec = _make_recommender(backend)
    body = {"messages": [{"role": "user", "content": "the same prompt"}]}

    async def _both() -> tuple:
        a = await rec.recommend(body, allowed_cells=CELLS, fallback=CELLS[0])
        b = await rec.recommend(body, allowed_cells=CELLS, fallback=CELLS[0])
        return a, b

    a, b = asyncio.run(_both())
    assert a.source == "upstream"
    assert b.source == "cache"
    assert a.cell == b.cell == CELLS[2]
    assert rec.stats["upstream_calls"] == 1
    assert rec.stats["cache_hits"] == 1


def test_recommend_cache_ttl_expires() -> None:
    backend = _FakeBackend(canned_text="model-a0e7 high")
    # TTL of 0 ⇒ every lookup is expired ⇒ no cache hits.
    rec = _make_recommender(backend, cache_ttl_seconds=0)
    body = {"messages": [{"role": "user", "content": "same"}]}

    async def _both() -> tuple:
        a = await rec.recommend(body, allowed_cells=CELLS, fallback=CELLS[0])
        b = await rec.recommend(body, allowed_cells=CELLS, fallback=CELLS[0])
        return a, b

    a, b = asyncio.run(_both())
    assert a.source == "upstream"
    assert b.source == "upstream"  # TTL expired immediately
    assert rec.stats["upstream_calls"] == 2


def test_status_recommender_block_shape_when_disabled() -> None:
    """No backends → recommender disabled but /status still renders the block."""
    client = TestClient(create_app())
    with client:
        body = client.get("/status").json()
    assert "recommender" in body
    r = body["recommender"]
    assert r["enabled"] is False
    assert "cheap_cell" in r
    assert "stats" in r
    assert "recommendation_counts" in r
