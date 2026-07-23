"""Protocols and dataclasses for the learning-router pipeline.

Everything the Router orchestrator depends on flows through these
interfaces. Concrete implementations are chosen at startup from config —
swap an embedding model, predictor, or selector without touching the
orchestrator.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from callosum.cell_grid import Cell

# ---------- request-side facts (immutable per request) ----------------------


@dataclass(frozen=True, slots=True)
class PromptFeatures:
    """What the router knows about an incoming request.

    `text` is the concatenated user-visible text from the request body
    (system + user + tool messages). `tokens` is an estimate; `modalities`
    enumerates what kinds of content the body contains (`{"text"}`
    minimally; add `"image"`, `"audio"`, `"video"` when those parts
    appear). `needs_tools` flags a non-empty `tools` field. `embedding`
    is the serialized vector from whichever EmbeddingProvider is wired
    in — None when the no-op provider is active or no provider was set.
    """

    text: str
    tokens: int
    modalities: frozenset[str]
    needs_tools: bool
    embedding: bytes | None = None


# ---------- cell-side facts (looked up from Cell + backend metadata) -------


@dataclass(frozen=True, slots=True)
class CellCapabilities:
    """What a cell can serve.

    `context_window` is the cell's max input+output token capacity.
    `modalities` is the set of content kinds the cell accepts.
    `supports_tools` is whether the cell honors tool-use requests.
    `cost_rank` is an integer ordering (0 = cheapest); operator-supplied
    in later phases, auto-inferred in Phase 1 (local backends rank 0,
    remote rank 10).
    `parameter_count` is the model's total parameter count when known —
    used by the selector as a tiebreaker after cost when no learned
    predictor distinguishes cells. Strictly informational: cost still
    wins; bigger only wins when costs tie. Sourced from ollama's
    `general.parameter_count` for local cells; None for cells whose
    backend doesn't surface this.
    Local-model metadata gives downstream routing stages and diagnostics one
    stable place to read local runtime evidence. Local cells stay nearly-free
    relative to remote quota cost, but `local_gpu_seconds_per_token` carries
    the non-zero GPU opportunity cost inside the admitted local fleet.
    `local_catalog_admitted` is the curated local-fleet routing signal:
    True means the local model passed the repo-owned admission rules, False
    means it is local but rejected, and None means the cell is not from that
    catalog surface.
    """

    context_window: int
    modalities: frozenset[str]
    supports_tools: bool
    cost_rank: int
    parameter_count: int | None = None
    local_throughput_tps: float | None = None
    local_gpu_seconds_per_token: float | None = None
    local_quantization: str | None = None
    local_runnable_on_host: bool | None = None
    local_status: str | None = None
    local_catalog_admitted: bool | None = None
    local_admission_reasons: tuple[str, ...] = ()


# ---------- learning-side facts (used by the predictor) --------------------


@dataclass(frozen=True, slots=True)
class LabeledRow:
    """One training row for the QualityPredictor.

    `prompt_embedding` is the serialized vector for the prompt. `cell_used`
    is the canonical "model effort" string for the cell that handled it.
    `outcome` is in [-1, 1]: -1 bad, 0 neutral / unknown, 1 good. The
    predictor decides how to interpret intermediate values.
    """

    request_id: int
    prompt_embedding: bytes
    cell_used: str
    outcome: float


# ---------- pipeline output -----------------------------------------------


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """The router's output for one request."""

    cell: Cell
    features: PromptFeatures
    # Map of cell-key ("model effort") to predicted satisfaction probability,
    # populated only for cells that passed the capability filter. Useful for
    # logging "what did the predictor think" alongside the eventual outcome.
    predictions: dict[str, float] = field(default_factory=dict)
    # Ordered candidate list for dispatch-level cell-retry. Index 0 is
    # always `cell` (the primary pick); remaining entries are the rest
    # of the compatible set ordered the same way the selector ordered
    # them. The dispatch layer walks this on 5xx / backend errors.
    candidates: tuple[Cell, ...] = ()
    # Optional per-cell p50 latency estimates in milliseconds for the same
    # post-filter candidate set. Populated only when the time estimator is
    # wired and used as a conservative selector tie-break.
    time_estimates_ms: dict[str, float] = field(default_factory=dict)
    # Provenance: which predictor produced the decision. Lets the request
    # log distinguish "uniform-prior cold-start" from "k-NN with N labels"
    # etc., so downstream consumers know how to weight a given decision.
    predictor_id: str = ""


# ---------- Protocols ------------------------------------------------------


class EmbeddingProvider(Protocol):
    """Encodes prompt text into a serialized embedding vector.

    Async because real providers run on GPU (BGE, etc.) and a no-op
    implementation can satisfy the contract by returning None. Returning
    None is a valid "no embedding available" signal — predictors that
    require an embedding fall back to their prior when they see one.
    """

    @property
    def dim(self) -> int:
        """Vector dimensionality. 0 for the no-op provider."""
        ...

    @property
    def id(self) -> str:
        """Stable identifier (e.g. 'bge-large-en-v1.5', 'noop'). Used for
        provenance logging so future analysis knows what produced an
        embedding."""
        ...

    async def embed(self, text: str) -> bytes | None:
        """Return the serialized embedding for one prompt. None means
        no embedding is available for this provider (no-op)."""
        ...


class QualityPredictor(Protocol):
    """Predicts P(this cell satisfies this prompt) for each candidate cell.

    Operates on PromptFeatures plus the surviving candidate set after the
    capability filter. Returns probabilities in [0, 1] keyed by cell. A
    well-calibrated predictor returns 0.5 when it has no opinion (cold
    start, no neighbors, etc.) so the cost-weighted selector falls back
    to cost ordering automatically.
    """

    @property
    def id(self) -> str:
        """Stable identifier (e.g. 'uniform', 'knn', 'gbm-v1')."""
        ...

    def predict(self, features: PromptFeatures, candidates: list[Cell]) -> dict[Cell, float]: ...

    def reload(self, labeled: Iterable[LabeledRow]) -> None:
        """Refresh internal state from a labeled-row iterable. For k-NN
        this rebuilds the index; for a trained classifier this is a no-op
        between training runs. Called by callosum startup and after the
        training Dispatch job emits a new checkpoint."""
        ...


class CellSelector(Protocol):
    """Picks ONE cell from the predicted/scored candidate set.

    The default impl picks the cheapest cell whose prediction crosses
    the binary-classifier decision boundary (0.5); see
    `selector/cost_weighted.py`.
    """

    @property
    def id(self) -> str: ...

    def select(
        self,
        predictions: dict[Cell, float],
        capabilities: dict[Cell, CellCapabilities],
        *,
        time_estimates_ms: dict[Cell, float] | None = None,
    ) -> Cell: ...


class OutcomeLabeler(Protocol):
    """Produces a quality label for one served request.

    Phase 4-5 plugs concrete implementations:
      * ImplicitLabeler — status==200 → +1, error → -1.
      * HumanLabeler — reads the quality_score column set by the label UI.
      * LLMJudgeLabeler — periodically samples local-served responses
        and grades them with a stronger remote model.
    """

    @property
    def id(self) -> str: ...

    async def label(self, request_id: int) -> float | None:
        """Return a label in [-1, 1] or None if no label is available
        for this row yet (Human/LLM-judge frequently return None on
        first call; the row gets re-checked later)."""
        ...
