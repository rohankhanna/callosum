from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Used only when a backend omits per-model reasoning metadata. This is a
# compatibility fallback, never the set of values a selector or catalog entry
# is allowed to use: live provider facts always win when present.
FALLBACK_REASONING_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh")

# Cold-start fallback model list. Used by `build_cells()` when no explicit
# models are passed. With dynamic model discovery (CodexAuthVaultBackend
# fetches /backend-api/codex/models per account), the runtime cell grid
# is built from `live_completion_models()` instead — DEFAULT_MODELS is
# only a safety net for tests and cold-start paths where no live data
# is available yet.
DEFAULT_MODELS: tuple[str, ...] = (
    "model-a0e7",
    "model-a0c3",
    "model-a0b8",
    "model-a0e6",
)

# Virtual model names a client can pick to opt into router behavior. Both route
# through the same cell recommender; the names are preserved in the request
# log's `routing_mode` column for observability.
# - "auto":          primary name (recommender-driven)
# - "auto-learning": backward-compat alias (recommender-driven)
# (The "auto-learning-synthetic" background-topper tier was removed —
# .)
VIRTUAL_MODELS: frozenset[str] = frozenset({"auto-learning", "auto"})

# Known context window limits per model. Used as a fallback when the API
# response doesn't include context_length. This map is volatile — models churn
# monthly toward weekly, so treat it as a safety net rather than truth.
_KNOWN_CONTEXT_WINDOWS: dict[str, int] = {
    "model-a0e8": 256_000,
    "model-a0e7": 128_000,
    "model-a0c3": 128_000,
    "model-a0b8": 128_000,
    "model-a0e6": 128_000,
}


@dataclass(frozen=True, slots=True)
class Cell:
    """One (model, reasoning_effort) pair the explorer router can target."""

    model: str
    reasoning_effort: str
    context_window: int | None = None  # Model's max context length in tokens, None = unknown

    def as_tuple(self) -> tuple[str, str]:
        return (self.model, self.reasoning_effort)


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Per-model metadata extracted from /backend-api/codex/models.

    Every field is defensive: only `slug` is required, everything else
    falls back to None / empty / sensible defaults when the API response
    doesn't include it. The proxy's cell-grid logic prefers these fields
    over its own hardcoded constants whenever they're populated.
    """

    slug: str
    display_name: str | None = None
    description: str | None = None
    context_window: int | None = None
    supported_in_api: bool | None = None
    visibility: str | None = None  # 'list' | 'hide' | etc.
    priority: int | None = None  # lower == stronger; higher == weaker
    default_reasoning_level: str | None = None
    # Empty tuple if the API didn't tell us; callers fall back to a default.
    supported_reasoning_levels: tuple[str, ...] = ()
    input_modalities: tuple[str, ...] = ()


def model_metadata_to_dict(md: ModelMetadata) -> dict[str, Any]:
    """Serialize a ModelMetadata record to a JSON-safe dict for disk persistence.

    Round-trips with `model_metadata_from_dict`. Tuples become lists (JSON has
    no tuple type); `model_metadata_from_dict` restores them. Used by the Codex
    backends to warm-start their catalog from the last-known-good persisted copy
    so a cold boot does not start with an empty model catalog.
    """
    return {
        "slug": md.slug,
        "display_name": md.display_name,
        "description": md.description,
        "context_window": md.context_window,
        "supported_in_api": md.supported_in_api,
        "visibility": md.visibility,
        "priority": md.priority,
        "default_reasoning_level": md.default_reasoning_level,
        "supported_reasoning_levels": list(md.supported_reasoning_levels),
        "input_modalities": list(md.input_modalities),
    }


def _opt_str(d: dict[str, Any], k: str) -> str | None:
    v = d.get(k)
    return v if isinstance(v, str) and v else None


def _opt_int(d: dict[str, Any], k: str) -> int | None:
    v = d.get(k)
    # bool is a subclass of int; exclude it so a JSON `true` doesn't masquerade
    # as priority=1. Mirrors the defensive parsing in _extract_model_catalog.
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _opt_bool(d: dict[str, Any], k: str) -> bool | None:
    v = d.get(k)
    return v if isinstance(v, bool) else None


def _opt_str_tuple(d: dict[str, Any], k: str) -> tuple[str, ...]:
    v = d.get(k)
    if not isinstance(v, list):
        return ()
    return tuple(x for x in v if isinstance(x, str) and x)


def model_metadata_from_dict(d: Any) -> ModelMetadata | None:
    """Reconstruct a ModelMetadata record from a persisted dict.

    Returns None on any shape problem (non-dict, missing/invalid slug, bad
    field types) so a corrupted or tampered on-disk blob never crashes startup
    — callers fall back to the empty/static catalog instead. The persisted
    catalog is untrusted disk state; validate, don't trust.
    """
    if not isinstance(d, dict):
        return None
    slug = d.get("slug")
    if not isinstance(slug, str) or not slug:
        return None
    return ModelMetadata(
        slug=slug,
        display_name=_opt_str(d, "display_name"),
        description=_opt_str(d, "description"),
        context_window=_opt_int(d, "context_window"),
        supported_in_api=_opt_bool(d, "supported_in_api"),
        visibility=_opt_str(d, "visibility"),
        priority=_opt_int(d, "priority"),
        default_reasoning_level=_opt_str(d, "default_reasoning_level"),
        supported_reasoning_levels=_opt_str_tuple(d, "supported_reasoning_levels"),
        input_modalities=_opt_str_tuple(d, "input_modalities"),
    )


@dataclass(frozen=True, slots=True)
class CellCoverage:
    """Sample counts per cell, used to pick the next variation target."""

    counts: dict[Cell, int]

    def least_sampled(self, cells: list[Cell]) -> Cell:
        """Return the cell with the fewest samples among `cells`. Ties
        broken by `cells` order (so the caller's preferred ordering wins).
        """
        return min(cells, key=lambda c: (self.counts.get(c, 0), cells.index(c)))

    def total_samples(self) -> int:
        return sum(self.counts.values())

    def cells_below(self, target: int, cells: list[Cell]) -> list[Cell]:
        return [c for c in cells if self.counts.get(c, 0) < target]


def build_cells(
    models: tuple[str, ...] = DEFAULT_MODELS,
    reasoning_levels: tuple[str, ...] = FALLBACK_REASONING_LEVELS,
    context_windows: dict[str, int] | None = None,
) -> list[Cell]:
    """Cross product of (models, reasoning_levels). Order is models-major then
    reasoning-major, so iteration is predictable for round-robin scheduling.

    If `context_windows` is provided, each Cell is stamped with the model's
    known context window. Otherwise defaults to _KNOWN_CONTEXT_WINDOWS map.
    """
    if context_windows is None:
        context_windows = _KNOWN_CONTEXT_WINDOWS
    return [
        Cell(model=m, reasoning_effort=r, context_window=context_windows.get(m))
        for m in models
        for r in reasoning_levels
    ]


# model-XXXX (4-character alphanumeric anonymous identifier). Excludes review models,
# embeddings, audio, etc. — anything that doesn't match this shape is treated
# as a special-purpose model and kept out of the auto-learning cell grid.
_COMPLETION_MODEL_RE = re.compile(r"^model-([a-z0-9]{4})$")

# No suffix ordering needed: each model has a unique anonymous id.
# Completion models are sorted alphabetically by their model-XXXX slug.


def is_completion_model(slug: str) -> bool:
    """True if the model id looks like a general-purpose chat completion
    target (model-XXXX, a 4-character alphanumeric identifier). Excludes
    codex-auto-review,
    embeddings, audio models, and anything else that doesn't match.
    """
    return _COMPLETION_MODEL_RE.match(slug) is not None


def model_strength_key(slug: str) -> tuple[int, int, int, int, str]:
    """Return a sort key where SMALLER tuple == STRONGER model.

    Use as `sorted(models, key=model_strength_key)` to get strongest first.
    Non-completion models sort to the end via a leading sentinel.
    """
    m = _COMPLETION_MODEL_RE.match(slug)
    if m is None:
        # Non-completion: push to the end of any sort. Tiebreak by name so
        # the ordering is deterministic.
        return (1, 0, 0, 0, slug)
    return (0, 0, 0, 0, slug)


def live_completion_models(model_pool: frozenset[str] | set[str]) -> tuple[str, ...]:
    """Filter a backend's advertised_models to just completion-style ids and
    return them sorted strongest-first. Used by app.py to build a cell grid
    that adapts to the current upstream catalog without a hardcoded list.

    This is the legacy regex-only path — works without per-model metadata
    from the API. See `live_completion_models_from_metadata` for the
    API-driven version which uses `supported_in_api` + `visibility` +
    `priority` instead of the regex and a name-shape sort.
    """
    completion_only = [m for m in model_pool if is_completion_model(m)]
    completion_only.sort(key=model_strength_key)
    return tuple(completion_only)


def live_completion_models_from_metadata(
    metadata: dict[str, ModelMetadata],
    *,
    include_hidden: bool = False,
) -> tuple[str, ...]:
    """Filter and rank models using upstream-provided metadata when present.

    Filter rules (each applied only when the field is populated):
      * `supported_in_api == True`     — model must be callable via this surface.
      * `visibility == 'list'`         — model is meant to be user-routable.
                                          Hidden models are excluded unless
                                          `include_hidden=True`, which is
                                          reserved for explicit selector pins.

    Rank: ascending `priority` (lower == stronger per upstream's convention).
    Models missing `priority` sort to the end via a high sentinel.

    Falls back to a name-shape filter (`is_completion_model`) only for
    slugs whose metadata lacks `supported_in_api` — defensive against
    older or unexpectedly stripped API responses.
    """
    out: list[tuple[int, str, str]] = []
    for slug, m in metadata.items():
        if m.supported_in_api is False:
            continue
        if not include_hidden and m.visibility is not None and m.visibility != "list":
            continue
        # If neither supported_in_api nor visibility was given, fall back
        # to the name-shape filter so we don't accidentally route to
        # embeddings / audio / review-style models.
        if m.supported_in_api is None and m.visibility is None and not is_completion_model(slug):
            continue
        # Sort key: priority asc (lower=stronger), with high sentinel for missing.
        prio = m.priority if m.priority is not None else 10_000
        out.append((prio, slug, slug))
    out.sort()
    return tuple(slug for _, _, slug in out)


def reasoning_levels_for(
    slug: str,
    metadata: dict[str, ModelMetadata],
    fallback: tuple[str, ...] = FALLBACK_REASONING_LEVELS,
) -> tuple[str, ...]:
    """Return the reasoning effort levels supported by this model.

    Prefers `metadata[slug].supported_reasoning_levels` when populated.
    Falls back to FALLBACK_REASONING_LEVELS only when the API didn't
    include the field (old responses and cold-start compatibility paths).
    """
    m = metadata.get(slug) if metadata else None
    if m is not None and m.supported_reasoning_levels:
        return m.supported_reasoning_levels
    return fallback


def build_cells_from_metadata(
    metadata: dict[str, ModelMetadata],
    *,
    include_hidden: bool = False,
) -> list[Cell]:
    """Build the cell grid using per-model `supported_reasoning_levels` from
    upstream when available; falls back to FALLBACK_REASONING_LEVELS for
    any model whose metadata is missing or empty.

    Filtering matches `live_completion_models_from_metadata` so the cell
    grid and the model list stay consistent. Hidden models are included only
    for explicit selector pins, never for automatic/free routing.
    """
    completion_slugs = live_completion_models_from_metadata(
        metadata,
        include_hidden=include_hidden,
    )
    cells: list[Cell] = []
    for slug in completion_slugs:
        m = metadata.get(slug)
        ctx = m.context_window if m is not None else None
        levels = reasoning_levels_for(slug, metadata)
        for r in levels:
            cells.append(Cell(model=slug, reasoning_effort=r, context_window=ctx))
    return cells


def coverage_from_db(
    usage_log_path: Path,
    cells: list[Cell],
    *,
    routing_mode: str = "auto-learning",
) -> CellCoverage:
    """Query usage_log for sample counts per cell where routing_mode matches.

    Only counts successful requests (status = 200) — failed requests don't help
    fit a cost model. Cells with zero samples are present in the dict with value 0.

    `routing_mode` selects which logged tier to count; 'auto-learning' (organic)
    is the default. (The minimum-coverage-quota enforcer will extend this to
    count coverage across a lane's organic traffic — .)
    """
    counts: dict[Cell, int] = dict.fromkeys(cells, 0)
    if not usage_log_path.exists():
        return CellCoverage(counts=counts)
    conn = sqlite3.connect(usage_log_path)
    try:
        rows = conn.execute(
            "SELECT model, reasoning_effort, COUNT(*)"
            " FROM requests"
            " WHERE routing_mode = ? AND status = 200"
            " GROUP BY model, reasoning_effort",
            (routing_mode,),
        ).fetchall()
    finally:
        conn.close()
    cell_lookup = {c.as_tuple(): c for c in cells}
    for model, reasoning, count in rows:
        cell = cell_lookup.get((model, reasoning))
        if cell is not None:
            counts[cell] = count
    return CellCoverage(counts=counts)


def cell_sample_counts(
    usage_log_path: Path,
    cells: list[Cell],
    *,
    window_seconds: int,
    now: float | None = None,
) -> CellCoverage:
    """Per-cell successful-served counts over a recent time window.

    Counts every status-200 request served by each (model, reasoning_effort)
    cell within window_seconds — regardless of which virtual-model name or
    routing mode produced it — so the minimum-coverage-quota enforcer can
    compute a cell's share of recent traffic. Lane scope is implicit in cells: pass
    the lane's candidate cells and only those are counted. Cells with no rows
    are present with value 0. (Contrast coverage_from_db, which counts a
    single routing_mode tier over all time.) See .
    """
    counts: dict[Cell, int] = dict.fromkeys(cells, 0)
    if not usage_log_path.exists():
        return CellCoverage(counts=counts)
    cutoff = (time.time() if now is None else now) - window_seconds
    conn = sqlite3.connect(usage_log_path)
    try:
        rows = conn.execute(
            "SELECT model, reasoning_effort, COUNT(*)"
            " FROM requests"
            " WHERE status = 200 AND ts_start >= ?"
            " GROUP BY model, reasoning_effort",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()
    cell_lookup = {c.as_tuple(): c for c in cells}
    for model, reasoning, count in rows:
        cell = cell_lookup.get((model, reasoning))
        if cell is not None:
            counts[cell] = count
    return CellCoverage(counts=counts)


def recent_quota_cooldown_cells(
    usage_log_path: Path,
    cells: list[Cell],
    *,
    window_seconds: int,
    now: float | None = None,
) -> frozenset[Cell]:
    """Cells currently in post-timeout coverage cooldown.

    A cell is cooling when, within window_seconds, its most recent
    forced-coverage (effective_routing_mode='min_coverage_quota') attempt
    FAILED (status != 200 — a stall-guard timeout logs status=0; a
    transient upstream failure logs a non-200) AND no successful
    (status=200) row for the cell has arrived since. The cooldown re-arms
    only on a real completed sample, because the coverage floor's purpose
    is coverage and coverage requires a *completed* sample, not a timeout
    (work tracker ````).

    This is the backstop to the feasibility filter in routing/feasibility:
    even a cold cell the estimator could not rule out (cold-cell grace) gets at
    most one failed forced turn before the quota stops re-targeting it, so the
    doom loop breaks regardless of whether the latency prior was trustworthy.
    The success check spans ALL routing modes — an organic completed sample on
    a small turn re-arms the cell too, since any completion proves the cell can
    finish *something*. Both the failure and success scans are bounded to the
    cooldown window: a failure older than the window no longer cools, and a
    windowed comparison of the two most-recent timestamps is exactly the
    "which happened last" test (a success outside the window is, by
    definition, older than any in-window failure).

    Returns the subset of cells currently cooling. Cells never forced
    (no min_coverage_quota row) are never cooling.
    """
    if not cells or not usage_log_path.exists():
        return frozenset()
    cutoff = (time.time() if now is None else now) - window_seconds
    conn = sqlite3.connect(usage_log_path)
    try:
        fail_rows = conn.execute(
            "SELECT model, reasoning_effort, MAX(ts_start)"
            " FROM requests"
            " WHERE effective_routing_mode = 'min_coverage_quota'"
            "   AND status != 200 AND ts_start >= ?"
            " GROUP BY model, reasoning_effort",
            (cutoff,),
        ).fetchall()
        ok_rows = conn.execute(
            "SELECT model, reasoning_effort, MAX(ts_start)"
            " FROM requests"
            " WHERE status = 200 AND ts_start >= ?"
            " GROUP BY model, reasoning_effort",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()
    cell_lookup = {c.as_tuple(): c for c in cells}
    last_ok: dict[tuple[str, str], float] = {}
    for model, effort, ts in ok_rows:
        if model is None or ts is None:
            continue
        last_ok[(model, effort or "")] = float(ts)
    cooling: set[Cell] = set()
    for model, effort, fail_ts in fail_rows:
        if model is None or fail_ts is None:
            continue
        cell = cell_lookup.get((model, effort or ""))
        if cell is None:
            continue
        if float(fail_ts) > last_ok.get((model, effort or ""), float("-inf")):
            cooling.add(cell)
    return frozenset(cooling)
