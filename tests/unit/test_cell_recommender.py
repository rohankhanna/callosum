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

from callosum.app import create_app
from callosum.backend import CallHandle, HealthStatus, UsageSnapshot
from callosum.cell_grid import Cell
from callosum.cell_recommender import (
    _CLASSIFIER_HEAD_CHARS,
    _CLASSIFIER_TAIL_CHARS,
    _CLASSIFIER_TRUNCATION_MARKER,
    CellRecommender,
    _approx_input_tokens,
    _filter_cells_by_context,
    _parse_cell_from_output,
    _truncate_for_classifier,
)
from callosum.errors import BackendError


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


def test_recommendation_carries_classifier_cell_and_raw_output_on_upstream() -> None:
    """Provenance fields on Recommendation feed the request log so a future
    training pipeline can filter to live upstream calls + know which
    classifier produced each label."""
    backend = _FakeBackend(canned_text="model-a0e7 high")
    rec = _make_recommender(backend)
    body = {"messages": [{"role": "user", "content": "test"}]}
    out = asyncio.run(rec.recommend(body, allowed_cells=CELLS, fallback=CELLS[0]))
    assert out.source == "upstream"
    assert out.classifier_cell == CELLS[0]  # the default cheap_cell
    assert out.raw_output == "model-a0e7 high"


def test_recommendation_carries_alt_classifier_on_alternative_path() -> None:
    """When the alternative-classifier path fires, classifier_cell on the
    result must be the alternative cell (not the cheap default), so the
    log row attributes the decision correctly."""
    backend = _FakeBackend(canned_text="model-a0e7 high")
    rec = _make_recommender(backend)
    body = {"messages": [{"role": "user", "content": "alt probe"}]}
    out = asyncio.run(
        rec.recommend(
            body, allowed_cells=CELLS, fallback=CELLS[0], classifier_cell=CELLS[2]
        )
    )
    assert out.source == "alternative"
    assert out.classifier_cell == CELLS[2]
    assert out.raw_output == "model-a0e7 high"


def test_recommendation_carries_raw_output_even_on_fallback_when_parse_fails() -> None:
    """If the classifier responded with text we couldn't parse (unknown
    cell), we still record what it said — diagnoses why fallback fired."""
    backend = _FakeBackend(canned_text="claude-opus high")
    rec = _make_recommender(backend)
    body = {"messages": [{"role": "user", "content": "x"}]}
    out = asyncio.run(rec.recommend(body, allowed_cells=CELLS, fallback=CELLS[0]))
    assert out.source == "fallback"
    assert out.raw_output == "claude-opus high"
    # classifier_cell still records who responded.
    assert out.classifier_cell == CELLS[0]


def test_recommendation_has_no_raw_output_on_cache_hit() -> None:
    """Cache hits intentionally lack provenance — the cached decision came
    from a prior upstream row whose log row already recorded the raw output.
    A training pipeline filtering to recommender_source = 'upstream' won't
    see cache rows at all, so cache hits leaving raw_output=None keeps the
    columns truthful: NULL means 'this row has no classifier output of its
    own', not 'we forgot to record it'."""
    backend = _FakeBackend(canned_text="model-a0e7 high")
    rec = _make_recommender(backend)
    body = {"messages": [{"role": "user", "content": "same"}]}

    async def _twice() -> tuple:
        a = await rec.recommend(body, allowed_cells=CELLS, fallback=CELLS[0])
        b = await rec.recommend(body, allowed_cells=CELLS, fallback=CELLS[0])
        return a, b

    a, b = asyncio.run(_twice())
    assert a.source == "upstream"
    assert a.raw_output == "model-a0e7 high"
    assert b.source == "cache"
    assert b.raw_output is None
    assert b.classifier_cell is None


# ---------- classifier-side prompt truncation -------------------------------


def test_short_prompt_passes_through_classifier_truncation_unchanged() -> None:
    """Prompts that already fit head+tail+marker are not modified."""
    text = "explain quicksort in five sentences"
    assert _truncate_for_classifier(text) == text


def test_long_prompt_is_truncated_to_head_plus_tail() -> None:
    """A 50k-char prompt should shrink to ~head+tail with the marker between.

    The classifier's job is routing-decision-from-prompt-gist, which the
    last few k of text captures. Sending the whole thing causes 5-30s
    classifier latency on small models with no decision-quality benefit.
    """
    head = "S" * 5000  # head signal
    middle = "M" * 40_000  # noise we don't need
    tail = "T" * 5000  # the latest user turn
    out = _truncate_for_classifier(head + middle + tail)
    assert len(out) < len(head + middle + tail) // 2
    assert out.startswith("S")  # head preserved
    assert out.endswith("T")    # tail preserved
    assert _CLASSIFIER_TRUNCATION_MARKER in out
    # Middle ("M"-only) is dropped.
    assert "MMMMMMMMM" not in out


def test_truncation_bounds_match_constants() -> None:
    """Sanity: the head and tail counts in the output match the module
    constants, so we'd notice if a refactor swapped them.
    """
    head_char, tail_char = "H", "T"
    head = head_char * 10_000
    tail = tail_char * 10_000
    out = _truncate_for_classifier(head + tail)
    # Output should contain exactly _CLASSIFIER_HEAD_CHARS of head_char
    # at the front, then the marker, then _CLASSIFIER_TAIL_CHARS of tail_char.
    assert out.startswith(head_char * _CLASSIFIER_HEAD_CHARS)
    assert out.endswith(tail_char * _CLASSIFIER_TAIL_CHARS)


def test_recommender_sends_truncated_prompt_to_classifier() -> None:
    """End-to-end: a huge prompt arrives at recommend(); the body the
    classifier actually receives must be the truncated version, not the
    full text. Otherwise the classifier still times out on huge inputs."""
    seen_text: list[str] = []

    @dataclass
    class _CaptureBackend(_FakeBackend):
        async def responses(self, body, handle):
            content = body.get("input", [{}])[0].get("content", [{}])
            text = content[0].get("text", "") if content else ""
            seen_text.append(text)
            return await super().responses(body, handle)

    backend = _CaptureBackend(canned_text="model-a0e7 high")
    rec = _make_recommender(backend)
    huge = "x" * 200_000
    body = {"messages": [{"role": "user", "content": huge}]}
    asyncio.run(rec.recommend(body, allowed_cells=CELLS, fallback=CELLS[0]))
    assert seen_text, "classifier should have been called"
    # What the classifier actually saw is much shorter than the input.
    sent_len = len(seen_text[0])
    assert sent_len < 10_000
    assert sent_len < len(huge)


# ---------- local-cell exploration (#4) ------------------------------------


def test_local_exploration_zero_pct_never_explores() -> None:
    """Default exploration_pct=0 means the classifier path is always used.
    Backward-compatible with all existing behavior."""
    backend = _FakeBackend(canned_text="model-a0e7 high")
    rec = _make_recommender(backend)
    body = {"messages": [{"role": "user", "content": "x"}]}
    out = asyncio.run(
        rec.recommend(
            body, allowed_cells=CELLS, fallback=CELLS[0],
            local_exploration_pct=0.0,
        )
    )
    assert out.source == "upstream"


def test_local_exploration_pct_1_always_picks_local() -> None:
    """With pct=1.0, every call should go to a local cell (effort=default).
    Classifier is bypassed entirely — no upstream call recorded."""
    backend = _FakeBackend(canned_text="model-a0e7 high")  # would normally win
    rec = _make_recommender(backend)
    local_cell = Cell(
        model="model-a0a9",
        reasoning_effort="default",
        context_window=262_144,
    )
    grid = [*CELLS, local_cell]
    body = {"messages": [{"role": "user", "content": "x"}]}
    out = asyncio.run(
        rec.recommend(
            body, allowed_cells=grid, fallback=CELLS[0],
            local_exploration_pct=1.0,
        )
    )
    assert out.cell == local_cell
    assert out.source == "local_exploration"
    # Classifier was bypassed — no upstream call, no raw_output.
    assert out.raw_output is None
    assert out.classifier_cell is None
    # And no upstream_calls happened in this path.
    assert rec.stats.get("local_exploration_calls", 0) == 1
    assert rec.stats.get("upstream_calls", 0) == 0


def test_local_exploration_no_local_cells_falls_through_to_classifier() -> None:
    """If there are no local cells in the allowed set, exploration can't
    fire — the path falls through to the normal classifier flow."""
    backend = _FakeBackend(canned_text="model-a0e7 high")
    rec = _make_recommender(backend)
    body = {"messages": [{"role": "user", "content": "x"}]}
    # CELLS has no local cells (all Codex with low/medium/high effort).
    out = asyncio.run(
        rec.recommend(
            body, allowed_cells=CELLS, fallback=CELLS[0],
            local_exploration_pct=1.0,  # would explore IF locals existed
        )
    )
    assert out.source == "upstream"
    assert out.cell == CELLS[2]  # the classifier's pick


def test_local_exploration_does_not_pollute_cache() -> None:
    """An exploration pick must not be cached, so subsequent non-exploration
    calls for the same prompt still go through the classifier (which may
    produce a different decision). Mirrors the alternative-classifier
    rotation's cache-bypass semantics."""
    backend = _FakeBackend(canned_text="model-a0e7 high")
    rec = _make_recommender(backend)
    local_cell = Cell(
        model="model-a0a9", reasoning_effort="default",
        context_window=262_144,
    )
    grid = [*CELLS, local_cell]
    body = {"messages": [{"role": "user", "content": "the same prompt"}]}

    async def _two_calls() -> tuple:
        # First call: exploration fires.
        a = await rec.recommend(
            body, allowed_cells=grid, fallback=CELLS[0],
            local_exploration_pct=1.0,
        )
        # Second call: no exploration. Should hit the classifier (not the
        # cache), because exploration didn't write to the cache.
        b = await rec.recommend(
            body, allowed_cells=grid, fallback=CELLS[0],
            local_exploration_pct=0.0,
        )
        return a, b

    a, b = asyncio.run(_two_calls())
    assert a.source == "local_exploration"
    assert b.source == "upstream"  # NOT cache — exploration didn't poison it
    assert b.cell == CELLS[2]


# ---------- compact cell-list format ---------------------------------------


def test_compact_format_groups_efforts_under_one_line_per_model() -> None:
    """Per-permutation enumeration is wasteful — multi-effort Codex models
    should collapse onto one line each with the efforts pipe-joined.
    Classifier parsing is unchanged (it matches the reply, not the list)."""
    from callosum.cell_recommender import _format_cell_list_compact

    cells = [
        Cell(model="model-a0e7", reasoning_effort="low", context_window=128_000),
        Cell(model="model-a0e7", reasoning_effort="medium", context_window=128_000),
        Cell(model="model-a0e7", reasoning_effort="high", context_window=128_000),
        Cell(model="model-a0e7", reasoning_effort="xhigh", context_window=128_000),
    ]
    out = _format_cell_list_compact(cells)
    # One line, all four efforts, context shown once.
    assert out.count("\n") == 0
    assert "model-a0e7" in out
    assert "low|medium|high|xhigh" in out
    assert "128K" in out


def test_compact_format_keeps_single_line_for_local_cells() -> None:
    """Local cells typically have only `default` effort. Format should not
    pretend there's a pipe choice when there isn't."""
    from callosum.cell_recommender import _format_cell_list_compact

    cells = [
        Cell(model="model-a0a9", reasoning_effort="default", context_window=262_144),
    ]
    out = _format_cell_list_compact(cells)
    assert "model-a0a9  default" in out
    assert "|" not in out
    assert "262K" in out


def test_compact_format_omits_context_when_unknown() -> None:
    """context_window=None means we don't know — show the model without
    fabricating a number."""
    from callosum.cell_recommender import _format_cell_list_compact

    cells = [
        Cell(model="mystery-model", reasoning_effort="default", context_window=None),
    ]
    out = _format_cell_list_compact(cells)
    assert "mystery-model  default" in out
    assert "K" not in out  # no fabricated context


def test_compact_format_preserves_model_order_from_input() -> None:
    """Cells flow in priority order (Codex first, locals last). The
    compact format must preserve that order so the operator's mental
    model of 'recommended cells' isn't scrambled."""
    from callosum.cell_recommender import _format_cell_list_compact

    cells = [
        Cell(model="model-a0e8", reasoning_effort="low", context_window=256_000),
        Cell(model="model-a0e8", reasoning_effort="high", context_window=256_000),
        Cell(model="model-a0a9", reasoning_effort="default", context_window=262_144),
    ]
    lines = _format_cell_list_compact(cells).splitlines()
    assert lines[0].startswith("- model-a0e8")
    assert lines[1].startswith("- model-a0a9")


# ---------- compatibility filter (4b.2) ------------------------------------


def test_approx_input_tokens_grows_with_prompt_length() -> None:
    short = _approx_input_tokens("hi")
    long = _approx_input_tokens("x" * 100_000)
    assert long > short
    # Floor: even an empty/short prompt gets a minimum estimate so the
    # filter doesn't accidentally accept zero-context cells.
    assert _approx_input_tokens("") >= 256


def test_filter_drops_cells_too_small_for_prompt() -> None:
    """A 50k-token prompt must filter out a 4k-window cell, keep a 128k cell."""
    small = Cell(model="local-tiny", reasoning_effort="default", context_window=4096)
    big = Cell(model="model-a0e7", reasoning_effort="high", context_window=128_000)
    out = _filter_cells_by_context([small, big], estimated_input_tokens=50_000)
    assert big in out
    assert small not in out


def test_filter_keeps_cells_with_unknown_context_window() -> None:
    """context_window=None means 'we don't know' — treat as compatible
    rather than silently exclude. Many local models report no window."""
    unknown = Cell(model="local-unknown", reasoning_effort="default", context_window=None)
    out = _filter_cells_by_context([unknown], estimated_input_tokens=200_000)
    assert out == [unknown]


def test_recommender_filters_cells_before_asking_classifier() -> None:
    """The classifier prompt must only enumerate cells the prompt fits in,
    so it can't pick a cell that would 4xx on context length.
    """
    seen_allowed_cells: list[list[Cell]] = []

    @dataclass
    class _CaptureBackend(_FakeBackend):
        async def responses(self, body, handle):
            # _build_recommender_body puts the available cell list in
            # `instructions` (Responses-API shape, not chat-completions).
            # The compact format groups by model — checking by model name
            # alone is sufficient to detect whether a cell was offered.
            instructions = body.get("instructions") or ""
            visible = [
                c for c in big_grid
                if f"- {c.model} " in instructions
            ]
            seen_allowed_cells.append(visible)
            return await super().responses(body, handle)

    small = Cell(model="local-tiny", reasoning_effort="default", context_window=4096)
    big = Cell(
        model="model-a0c3", reasoning_effort="low", context_window=128_000
    )
    big_grid = [small, big]
    backend = _CaptureBackend(canned_text="model-a0c3 low")
    rec = _make_recommender(backend)
    # 100k-char prompt → ~50k tokens → too big for small cell, fits big cell.
    body = {"messages": [{"role": "user", "content": "z" * 100_000}]}
    out = asyncio.run(rec.recommend(body, allowed_cells=big_grid, fallback=big))
    assert out.cell == big
    assert seen_allowed_cells, "classifier should have been called"
    # Confirm: the small cell never appeared in the classifier's prompt.
    for visible in seen_allowed_cells:
        assert small not in visible
        assert big in visible


def test_recommendation_candidates_put_primary_first_then_compat_set() -> None:
    """The dispatch layer walks `candidates` on retryable failure. The
    classifier's pick must be at index 0; the rest of the compatible
    cell set follows in its grid-priority order (deduped against
    primary). No duplication, no surprise reordering.
    """
    backend = _FakeBackend(canned_text="model-a0e7 high")
    rec = _make_recommender(backend)
    body = {"messages": [{"role": "user", "content": "p"}]}
    out = asyncio.run(rec.recommend(body, allowed_cells=CELLS, fallback=CELLS[0]))
    assert out.candidates[0] == CELLS[2]  # the classifier's pick (model-a0e7 high)
    # Everything else in the grid (excluding the primary), in order.
    expected_rest = [c for c in CELLS if c != CELLS[2]]
    assert list(out.candidates[1:]) == expected_rest


def test_recommender_falls_back_to_full_grid_when_filter_empties() -> None:
    """If every known-context cell is too small, the recommender uses the
    full grid as a last resort rather than refusing the request."""
    tiny_a = Cell(model="a", reasoning_effort="default", context_window=4096)
    tiny_b = Cell(model="b", reasoning_effort="default", context_window=4096)
    backend = _FakeBackend(canned_text="a default")
    rec = _make_recommender(backend)
    # 1M-char prompt — neither cell fits, but we still get a recommendation
    # (rather than a refusal) so the user request isn't blocked at the router.
    body = {"messages": [{"role": "user", "content": "x" * 1_000_000}]}
    out = asyncio.run(
        rec.recommend(body, allowed_cells=[tiny_a, tiny_b], fallback=tiny_a)
    )
    assert out.cell == tiny_a
    assert out.source == "upstream"


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


# ---------- local-router path ----------------------------------------------


LOCAL_CELL = Cell(
    model="model-a0d5-3-31b-ollama",
    reasoning_effort="default",
    context_window=262_144,
)
ROUTER_GRID = [*CELLS, LOCAL_CELL]


def _make_router_recommender(
    *,
    cheap_backend: _FakeBackend,
    router_backend: _FakeBackend,
    router_cell: Cell = LOCAL_CELL,
    router_timeout_s: float = 2.0,
    router_circuit_threshold: int = 3,
    router_circuit_cooldown_s: float = 60.0,
) -> CellRecommender:
    return CellRecommender(
        cheap_backend=cheap_backend,
        cheap_cell=CELLS[0],
        upstream_timeout_s=2.0,
        router_backend=router_backend,
        router_cell=router_cell,
        router_timeout_s=router_timeout_s,
        router_circuit_threshold=router_circuit_threshold,
        router_circuit_cooldown_s=router_circuit_cooldown_s,
    )


def test_router_success_picks_returned_cell_and_caches() -> None:
    """Router's text response decides the cell. Source is 'router' (not
    'upstream') so the request log can distinguish local-router decisions
    from legacy remote-classifier ones. Second identical prompt hits cache.
    """
    cheap = _FakeBackend(canned_text="should-not-be-called")
    router = _FakeBackend(canned_text="model-a0e7 high")
    rec = _make_router_recommender(cheap_backend=cheap, router_backend=router)
    body = {"messages": [{"role": "user", "content": "explain quicksort"}]}

    async def _two() -> tuple:
        a = await rec.recommend(body, allowed_cells=ROUTER_GRID, fallback=CELLS[0])
        b = await rec.recommend(body, allowed_cells=ROUTER_GRID, fallback=CELLS[0])
        return a, b

    a, b = asyncio.run(_two())
    assert a.source == "router"
    assert a.cell == CELLS[2]
    assert a.classifier_cell == LOCAL_CELL
    assert a.raw_output == "model-a0e7 high"
    assert b.source == "cache"
    assert b.cell == CELLS[2]
    assert rec.stats["router_calls"] == 1
    assert rec.stats["cache_hits"] == 1


def test_router_failure_walks_heuristic_preference() -> None:
    """When the router raises, walk fallback_preference and pick the first
    routable match. Never calls _ask_upstream — the whole point of opting
    into a local router is to escape the remote classifier's selection bias.
    """
    cheap = _FakeBackend(canned_text="should-not-be-called")
    router = _FakeBackend(raise_error=True)
    rec = _make_router_recommender(cheap_backend=cheap, router_backend=router)
    body = {"messages": [{"role": "user", "content": "x"}]}
    out = asyncio.run(
        rec.recommend(
            body,
            allowed_cells=ROUTER_GRID,
            fallback=CELLS[0],
            fallback_preference=["model-a0d5-3-31b-ollama default", "model-a0e7 high"],
        )
    )
    assert out.source == "heuristic"
    assert out.cell == LOCAL_CELL  # first preference matched
    assert rec.stats["heuristic_hits"] == 1
    assert rec.stats["upstream_calls"] == 0  # legacy path never ran


def test_router_failure_with_no_heuristic_match_falls_to_last_resort() -> None:
    """If no preference entry matches the compatible cell set, the existing
    `fallback` cell is returned as the last resort. Still NOT the legacy
    remote classifier."""
    cheap = _FakeBackend(canned_text="should-not-be-called")
    router = _FakeBackend(raise_error=True)
    rec = _make_router_recommender(cheap_backend=cheap, router_backend=router)
    body = {"messages": [{"role": "user", "content": "x"}]}
    out = asyncio.run(
        rec.recommend(
            body,
            allowed_cells=ROUTER_GRID,
            fallback=CELLS[1],  # explicit last-resort cell
            fallback_preference=["nonexistent-model effort"],
        )
    )
    assert out.source == "fallback"
    assert out.cell == CELLS[1]
    assert rec.stats["upstream_calls"] == 0


def test_router_circuit_opens_after_threshold_consecutive_failures() -> None:
    """3 (default) consecutive router failures → circuit opens. While open,
    the router is NOT called, the next request goes straight to heuristic /
    last-resort. router_calls stops incrementing during the cooldown."""
    cheap = _FakeBackend(canned_text="should-not-be-called")
    router = _FakeBackend(raise_error=True)
    rec = _make_router_recommender(
        cheap_backend=cheap,
        router_backend=router,
        router_circuit_threshold=3,
        router_circuit_cooldown_s=60.0,
    )

    async def _four_calls() -> None:
        for i in range(4):
            await rec.recommend(
                {"messages": [{"role": "user", "content": f"req {i}"}]},
                allowed_cells=ROUTER_GRID,
                fallback=CELLS[0],
                fallback_preference=["model-a0d5-3-31b-ollama default"],
            )

    asyncio.run(_four_calls())
    # 3 calls attempted the router (and failed). Call #4 saw the circuit
    # open and skipped the router entirely.
    assert rec.stats["router_calls"] == 3
    assert rec.stats["router_failures"] == 3
    assert rec.stats["router_circuit_opens"] == 1
    assert rec._circuit_is_open() is True


def test_router_circuit_resets_on_success() -> None:
    """Mixed failure/success sequence: consecutive failures count resets
    to 0 on each success, so the circuit only opens on a real run of
    failures, not on cumulative-over-lifetime failures."""
    cheap = _FakeBackend(canned_text="x")
    # Toggle the router between raise and answer across calls.
    @dataclass
    class _ToggleBackend(_FakeBackend):
        calls: int = 0
        def _next_outcome(self) -> bool:
            self.calls += 1
            return self.calls % 2 == 0  # even calls succeed, odd ones raise
        async def responses(self, body, handle):
            if not self._next_outcome():
                raise BackendError(classification="transient", message="boom")
            return {"output": [{"content": [{"text": "model-a0e7 high"}]}]}

    router = _ToggleBackend()
    rec = _make_router_recommender(
        cheap_backend=cheap,
        router_backend=router,
        router_circuit_threshold=3,
    )

    async def _seq() -> None:
        for i in range(6):
            await rec.recommend(
                {"messages": [{"role": "user", "content": f"r{i}"}]},
                allowed_cells=ROUTER_GRID,
                fallback=CELLS[0],
                fallback_preference=["model-a0d5-3-31b-ollama default"],
            )

    asyncio.run(_seq())
    # 6 router calls, 3 failures interleaved with 3 successes. Counter never
    # reached the threshold of 3 consecutive, so circuit stays closed.
    assert rec.stats["router_calls"] == 6
    assert rec.stats["router_failures"] == 3
    assert rec.stats["router_circuit_opens"] == 0
    assert rec._circuit_is_open() is False


def test_router_classifier_override_falls_through_to_legacy_path() -> None:
    """If an explicit classifier_cell is passed AND router is configured,
    honor the override and run the legacy path (alternative-classifier
    rotation). This keeps the orthogonality: opt-in router doesn't ban
    explicit overrides from callers who really want them."""
    cheap = _FakeBackend(canned_text="model-a0c3 medium")
    router = _FakeBackend(canned_text="should-not-be-called")
    rec = _make_router_recommender(cheap_backend=cheap, router_backend=router)
    body = {"messages": [{"role": "user", "content": "test"}]}
    out = asyncio.run(
        rec.recommend(
            body,
            allowed_cells=ROUTER_GRID,
            fallback=CELLS[0],
            classifier_cell=CELLS[2],  # explicit override
        )
    )
    assert out.source == "alternative"
    assert rec.stats["router_calls"] == 0  # router NEVER consulted
    assert rec.stats["upstream_calls"] == 1


def test_router_unconfigured_preserves_legacy_behavior() -> None:
    """Router knobs default to None — recommender behaves exactly as before
    this change. Regression guard."""
    backend = _FakeBackend(canned_text="model-a0e7 high")
    rec = _make_recommender(backend)  # no router_backend kwarg
    body = {"messages": [{"role": "user", "content": "x"}]}
    out = asyncio.run(rec.recommend(body, allowed_cells=CELLS, fallback=CELLS[0]))
    assert out.source == "upstream"
    assert rec.stats["router_calls"] == 0


def test_router_disables_local_exploration_path() -> None:
    """When router is configured the local-exploration random walk is
    suppressed — the router itself can pick local cells freely and we
    don't want compounded randomness muddying the signal."""
    cheap = _FakeBackend(canned_text="should-not-be-called")
    router = _FakeBackend(canned_text="model-a0d5-3-31b-ollama default")
    rec = _make_router_recommender(cheap_backend=cheap, router_backend=router)
    body = {"messages": [{"role": "user", "content": "x"}]}
    out = asyncio.run(
        rec.recommend(
            body,
            allowed_cells=ROUTER_GRID,
            fallback=CELLS[0],
            local_exploration_pct=1.0,  # would normally always explore
        )
    )
    # local_exploration was suppressed; the router made a real decision.
    assert out.source == "router"
    assert out.cell == LOCAL_CELL
    assert rec.stats.get("local_exploration_calls", 0) == 0


def test_heuristic_skips_local_when_prompt_above_ceiling() -> None:
    """When the router fails AND the prompt is above the configured token
    ceiling, the heuristic walks past local cells in fallback_preference
    and picks the first remote one. Stops a 30K-token reasoning task from
    being dumped on a local model just because local is listed first."""
    cheap = _FakeBackend(canned_text="should-not-be-called")
    router = _FakeBackend(raise_error=True)
    rec = _make_router_recommender(cheap_backend=cheap, router_backend=router)
    # Prompt about 60K chars → est_tokens ~ 20K, well above the 8K ceiling.
    big_prompt = "x" * 60_000
    body = {"messages": [{"role": "user", "content": big_prompt}]}
    out = asyncio.run(
        rec.recommend(
            body,
            allowed_cells=ROUTER_GRID,
            fallback=CELLS[0],
            fallback_preference=[
                "model-a0d5-3-31b-ollama default",  # local — should be skipped
                "model-a0e7 high",                # remote — should be picked
            ],
            heuristic_local_complexity_token_ceiling=8000,
        )
    )
    assert out.source == "heuristic"
    assert out.cell == CELLS[2]  # model-a0e7 high


def test_heuristic_keeps_local_when_prompt_below_ceiling() -> None:
    """Below the ceiling, the heuristic walks the preference list as-is —
    local-first stays local-first for small/cheap prompts."""
    cheap = _FakeBackend(canned_text="should-not-be-called")
    router = _FakeBackend(raise_error=True)
    rec = _make_router_recommender(cheap_backend=cheap, router_backend=router)
    body = {"messages": [{"role": "user", "content": "tiny prompt"}]}
    out = asyncio.run(
        rec.recommend(
            body,
            allowed_cells=ROUTER_GRID,
            fallback=CELLS[0],
            fallback_preference=[
                "model-a0d5-3-31b-ollama default",
                "model-a0e7 high",
            ],
            heuristic_local_complexity_token_ceiling=8000,
        )
    )
    assert out.source == "heuristic"
    assert out.cell == LOCAL_CELL  # small prompt, local pick stands


def test_heuristic_does_not_skip_local_when_no_remote_available() -> None:
    """Critical offline-failover invariant: even on a complex prompt, if no
    remote cell is routable (offline, all codex backends filtered out by
    `_filter_cells_to_routable`), the heuristic must NOT skip local —
    otherwise the request dies on a backend that can't be reached. Better
    to send a complex prompt to model-a0d5 (imperfect answer) than to model-a0f8
    (no answer at all)."""
    cheap = _FakeBackend(canned_text="should-not-be-called")
    router = _FakeBackend(raise_error=True)
    rec = _make_router_recommender(cheap_backend=cheap, router_backend=router)
    # Big prompt → would normally trigger skip_local.
    big_prompt = "x" * 60_000
    body = {"messages": [{"role": "user", "content": big_prompt}]}
    # compatible_cells contains ONLY local cells — simulates the offline
    # scenario where remote codex cells have been filtered out by the
    # routability filter upstream of recommend().
    local_only_grid = [LOCAL_CELL]
    out = asyncio.run(
        rec.recommend(
            body,
            allowed_cells=local_only_grid,
            fallback=LOCAL_CELL,
            fallback_preference=[
                "model-a0d5-3-31b-ollama default",  # local — must NOT be skipped offline
                "model-a0e7 high",                # remote — but not in grid
            ],
            heuristic_local_complexity_token_ceiling=8000,
        )
    )
    # Local picked despite being a complex prompt; offline failover preserved.
    assert out.source == "heuristic"
    assert out.cell == LOCAL_CELL


def test_heuristic_ceiling_zero_disables_escalation() -> None:
    """ceiling=0 is the disabled sentinel — local cells are never skipped
    regardless of prompt size. Backward compatibility with operators who
    haven't opted into the size-aware escalation."""
    cheap = _FakeBackend(canned_text="should-not-be-called")
    router = _FakeBackend(raise_error=True)
    rec = _make_router_recommender(cheap_backend=cheap, router_backend=router)
    body = {"messages": [{"role": "user", "content": "x" * 60_000}]}
    out = asyncio.run(
        rec.recommend(
            body,
            allowed_cells=ROUTER_GRID,
            fallback=CELLS[0],
            fallback_preference=[
                "model-a0d5-3-31b-ollama default",
                "model-a0e7 high",
            ],
            heuristic_local_complexity_token_ceiling=0,
        )
    )
    # Big prompt but ceiling disabled → local still picked.
    assert out.source == "heuristic"
    assert out.cell == LOCAL_CELL


def test_heuristic_preference_is_case_insensitive() -> None:
    """Operators writing config shouldn't have to match the exact case of
    the cell's model/effort. 'MODEL-A0D5-3-31B-Ollama DEFAULT' matches just as
    well as the lowercase form."""
    cheap = _FakeBackend(canned_text="should-not-be-called")
    router = _FakeBackend(raise_error=True)
    rec = _make_router_recommender(cheap_backend=cheap, router_backend=router)
    body = {"messages": [{"role": "user", "content": "x"}]}
    out = asyncio.run(
        rec.recommend(
            body,
            allowed_cells=ROUTER_GRID,
            fallback=CELLS[0],
            fallback_preference=["MODEL-A0D5-3-31B-OLLAMA DEFAULT"],
        )
    )
    assert out.source == "heuristic"
    assert out.cell == LOCAL_CELL
