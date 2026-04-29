from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

REASONING_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh")

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

# Virtual model names a client can pick to opt into router behavior.
# - "auto-learning":           organic round-robin explorer
# - "auto-learning-synthetic": synthetic background topper (separate coverage tier)
# - "auto":                    cost-optimal exploiter (returns 503 NotTrained)
VIRTUAL_MODELS: frozenset[str] = frozenset({"auto-learning", "auto-learning-synthetic", "auto"})


@dataclass(frozen=True, slots=True)
class Cell:
    """One (model, reasoning_effort) pair the explorer router can target."""

    model: str
    reasoning_effort: str

    def as_tuple(self) -> tuple[str, str]:
        return (self.model, self.reasoning_effort)


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
    reasoning_levels: tuple[str, ...] = REASONING_LEVELS,
) -> list[Cell]:
    """Cross product of (models, reasoning_levels). Order is models-major then
    reasoning-major, so iteration is predictable for round-robin scheduling.
    """
    return [Cell(model=m, reasoning_effort=r) for m in models for r in reasoning_levels]


# `gpt-X.Y` with optional `-mini` or `-codex` suffix. Excludes review models,
# embeddings, audio, etc. — anything that doesn't match this shape is treated
# as a special-purpose model and kept out of the auto-learning cell grid.
_COMPLETION_MODEL_RE = re.compile(r"^gpt-(\d+)\.(\d+)(?:-(mini|codex))?$")

# Suffix ordering: full-model variants beat code-tuned beat smaller. Lower
# rank wins (used as a tiebreak after major.minor desc).
_SUFFIX_RANK: dict[str, int] = {"": 0, "codex": 1, "mini": 2}


def is_completion_model(slug: str) -> bool:
    """True if the model id looks like a general-purpose chat completion
    target (gpt-X.Y, gpt-X.Y-mini, gpt-X.Y-codex). Excludes codex-auto-review,
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
    major = int(m.group(1))
    minor = int(m.group(2))
    suffix = m.group(3) or ""
    return (0, -major, -minor, _SUFFIX_RANK.get(suffix, 9), slug)


def live_completion_models(model_pool: frozenset[str] | set[str]) -> tuple[str, ...]:
    """Filter a backend's advertised_models to just completion-style ids and
    return them sorted strongest-first. Used by app.py to build a cell grid
    that adapts to the current upstream catalog without a hardcoded list.
    """
    completion_only = [m for m in model_pool if is_completion_model(m)]
    completion_only.sort(key=model_strength_key)
    return tuple(completion_only)


def coverage_from_db(
    usage_log_path: Path,
    cells: list[Cell],
    *,
    routing_mode: str = "auto-learning",
) -> CellCoverage:
    """Query usage_log for sample counts per cell where routing_mode matches.

    Only counts successful requests (status = 200) — failed requests don't help
    fit a cost model. Cells with zero samples are present in the dict with value 0.

    `routing_mode` selects which tier to count: 'auto-learning' (organic) is the
    default; 'auto-learning-synthetic' counts the background-topper tier
    independently so the two tiers don't double-count each other.
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
