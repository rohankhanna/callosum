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
    assert "classifier_call_counts" in r
    assert "alternative_classifier_pct" in r


# ---------- bias mitigation: alternative-classifier path ----------


def test_alternative_classifier_path_uses_override_not_cheap() -> None:
    """When classifier_cell is passed, recommend() calls that cell, not cheap_cell."""
    backend = _FakeBackend(canned_text="model-a0e7 high")
    rec = _make_recommender(backend)
    body = {"messages": [{"role": "user", "content": "an alternative-classifier probe"}]}
    alt = CELLS[2]  # model-a0e7 high — a non-cheap classifier
    out = asyncio.run(
        rec.recommend(
            body, allowed_cells=CELLS, fallback=CELLS[0], classifier_cell=alt
        )
    )
    assert out.source == "alternative"
    assert rec.stats["alternative_calls"] == 1
    # Classifier-call tracking attributes the call to the override cell.
    counts = rec.classifier_call_counts
    assert counts.get(f"{alt.model} {alt.reasoning_effort}", 0) == 1


def test_alternative_classifier_bypasses_cache_on_read() -> None:
    """A prior cheap-classifier cache entry must NOT short-circuit an
    alternative-classifier call — the whole point is to get a second opinion.
    """
    backend = _FakeBackend(canned_text="model-a0c3 medium")
    rec = _make_recommender(backend)
    body = {"messages": [{"role": "user", "content": "same prompt"}]}
    async def _go() -> tuple:
        # Warm cache via cheap classifier.
        first = await rec.recommend(body, allowed_cells=CELLS, fallback=CELLS[0])
        # Same prompt, but with alternative classifier — must NOT hit cache.
        second = await rec.recommend(
            body, allowed_cells=CELLS, fallback=CELLS[0], classifier_cell=CELLS[2]
        )
        return first, second
    first, second = asyncio.run(_go())
    assert first.source == "upstream"
    # Even though the cheap-classifier cache has an entry, the alternative
    # path didn't return source="cache".
    assert second.source == "alternative"
    assert rec.stats["upstream_calls"] == 2  # both paths called upstream


def test_cheap_cell_resolves_dynamically_when_configured_one_disappears() -> None:
    """Regression: if the configured cheap classifier is no longer in the
    live cell grid (model renamed, retired upstream, etc.), the recommender
    must NOT keep trying to call the dead name. It picks the weakest model
    in the current grid + lowest effort instead. Keeps routing alive when
    the model lineup churns underneath us.
    """
    # Build a recommender whose configured cheap cell is NOT in `allowed_cells`.
    # The recommender should resolve to a cell that IS in the list.
    backend = _FakeBackend(canned_text="model-a0e7 high")
    rec = CellRecommender(
        cheap_backend=backend,
        cheap_cell=Cell(
            model="gpt-RETIRED-mini",
            reasoning_effort="low",
            context_window=None,
        ),
        upstream_timeout_s=2.0,
    )
    # Allowed cells: no "gpt-RETIRED-mini" present. Weakest model in CELLS
    # (by model_strength_key) is model-a0c3 → that's what gets picked.
    resolved = rec._resolve_cheap_cell(CELLS)
    assert resolved.model == "model-a0c3"
    assert resolved.reasoning_effort == "low"  # lowest available effort for that model


def test_cheap_cell_resolution_prefers_configured_when_present() -> None:
    """When the configured cheap_cell IS in the grid, use it unchanged —
    auto-resolution is a fallback for model churn, not a steady-state
    override of operator config.
    """
    backend = _FakeBackend(canned_text="model-a0e7 high")
    rec = _make_recommender(backend)  # cheap_cell defaults to CELLS[0] (model-a0c3 low)
    resolved = rec._resolve_cheap_cell(CELLS)
    assert resolved == CELLS[0]


def test_alternative_classifier_does_not_pollute_cache() -> None:
    """An alternative-classifier result must NOT be written to cache —
    otherwise the next cheap-classifier request for the same prompt would
    incorrectly read back the alternative's answer."""
    backend = _FakeBackend(canned_text="model-a0e7 high")  # the alternative recommends 5.4 high
    rec = _make_recommender(backend)
    body = {"messages": [{"role": "user", "content": "test prompt"}]}
    async def _go() -> tuple:
        # First call via alternative classifier.
        alt_result = await rec.recommend(
            body, allowed_cells=CELLS, fallback=CELLS[0], classifier_cell=CELLS[2]
        )
        # Now via cheap classifier — should be a fresh upstream call, NOT a
        # cache hit returning the alternative classifier's answer.
        cheap_result = await rec.recommend(body, allowed_cells=CELLS, fallback=CELLS[0])
        return alt_result, cheap_result
    alt_result, cheap_result = asyncio.run(_go())
    assert alt_result.source == "alternative"
    assert cheap_result.source == "upstream"  # NOT cache
    assert rec.stats["upstream_calls"] == 2
