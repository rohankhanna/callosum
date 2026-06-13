"""Tests for KNNPredictor — k-NN over labeled prompt embeddings."""

from __future__ import annotations

import numpy as np

from callosum.cell_grid import Cell
from callosum.routing.predictor.knn import KNNPredictor
from callosum.routing.protocols import LabeledRow, PromptFeatures


def _emb(values) -> bytes:
    """Serialize a list/array of floats as a normalized float32 vector."""
    v = np.asarray(values, dtype=np.float32)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v.tobytes()


def _features_with_embedding(embedding_bytes: bytes | None) -> PromptFeatures:
    return PromptFeatures(
        text="any",
        tokens=10,
        modalities=frozenset({"text"}),
        needs_tools=False,
        embedding=embedding_bytes,
    )


CELL_LOCAL = Cell(model="local", reasoning_effort="default")
CELL_REMOTE = Cell(model="remote", reasoning_effort="medium")


def test_cold_start_returns_uniform_prior_for_every_candidate() -> None:
    """With no labeled rows, the predictor has nothing to learn from →
    returns 0.5 for every candidate. The selector falls back to cost
    ordering → local-first. This IS the intended cold-start behavior."""
    p = KNNPredictor()
    f = _features_with_embedding(_emb([1.0, 0.0, 0.0]))
    out = p.predict(f, [CELL_LOCAL, CELL_REMOTE])
    assert out == {CELL_LOCAL: 0.5, CELL_REMOTE: 0.5}


def test_predictor_returns_uniform_when_features_has_no_embedding() -> None:
    """No embedding available (noop provider, embedding errored, etc.) →
    fall back to uniform prior so the rest of the pipeline still runs."""
    p = KNNPredictor()
    f = _features_with_embedding(None)
    out = p.predict(f, [CELL_LOCAL])
    assert out == {CELL_LOCAL: 0.5}


def test_predictor_learns_from_labeled_rows() -> None:
    """After reload() with rows showing 'this kind of prompt succeeded on
    local', the predictor should boost P(local) for a query embedding
    near those labeled examples."""
    p = KNNPredictor(k=3)
    # 3 labeled rows, all with embedding ~ [1,0,0], all CELL_LOCAL succeeded.
    rows = [
        LabeledRow(
            request_id=1,
            prompt_embedding=_emb([1.0, 0.01, 0.0]),
            cell_used="local default",
            outcome=1.0,  # +1 = success
        ),
        LabeledRow(
            request_id=2,
            prompt_embedding=_emb([1.0, 0.0, 0.01]),
            cell_used="local default",
            outcome=1.0,
        ),
        LabeledRow(
            request_id=3,
            prompt_embedding=_emb([0.99, 0.0, 0.0]),
            cell_used="local default",
            outcome=1.0,
        ),
    ]
    p.reload(rows)
    f = _features_with_embedding(_emb([1.0, 0.0, 0.0]))
    out = p.predict(f, [CELL_LOCAL, CELL_REMOTE])
    # Local has 3 neighbors with outcome=+1 → P(satisfy) = (1+1)/2 = 1.0
    assert out[CELL_LOCAL] == 1.0
    # Remote has no labeled neighbors → falls back to 0.5 prior.
    assert out[CELL_REMOTE] == 0.5


def test_predictor_maps_negative_outcomes_to_low_probability() -> None:
    """Outcome=-1 means 'this cell failed on this kind of prompt' → the
    predictor should map that to a low P(satisfy) so the selector
    routes away from it."""
    p = KNNPredictor(k=3)
    rows = [
        LabeledRow(
            request_id=1,
            prompt_embedding=_emb([1, 0, 0]),
            cell_used="local default",
            outcome=-1.0,
        ),
        LabeledRow(
            request_id=2,
            prompt_embedding=_emb([1, 0.01, 0]),
            cell_used="local default",
            outcome=-1.0,
        ),
        LabeledRow(
            request_id=3,
            prompt_embedding=_emb([0.99, 0, 0]),
            cell_used="local default",
            outcome=-1.0,
        ),
    ]
    p.reload(rows)
    out = p.predict(_features_with_embedding(_emb([1, 0, 0])), [CELL_LOCAL])
    # P = (-1 + 1) / 2 = 0.0 → the selector drops this cell below the
    # 0.5 decision boundary.
    assert out[CELL_LOCAL] == 0.0


def test_predictor_handles_dimensionality_mismatch_gracefully() -> None:
    """If the index was built with one embedding model and the query
    comes from a different one (different dim), we can't compare
    them — fall back to uniform rather than producing garbage."""
    p = KNNPredictor(k=3)
    p.reload(
        [
            LabeledRow(
                request_id=1,
                prompt_embedding=_emb([1, 0, 0, 0, 0]),  # 5-dim
                cell_used="local default",
                outcome=1.0,
            ),
        ]
    )
    # Query is 3-dim — mismatch.
    out = p.predict(_features_with_embedding(_emb([1, 0, 0])), [CELL_LOCAL])
    assert out[CELL_LOCAL] == 0.5


def test_reload_with_empty_iterable_resets_to_cold_start() -> None:
    """reload([]) should clear any previous state — useful for tests
    and for the case where all labeled rows get filtered out (e.g.
    after a model-version migration)."""
    p = KNNPredictor()
    p.reload(
        [
            LabeledRow(
                request_id=1,
                prompt_embedding=_emb([1, 0, 0]),
                cell_used="local default",
                outcome=1.0,
            ),
        ]
    )
    # Verify it learned.
    out_warm = p.predict(_features_with_embedding(_emb([1, 0, 0])), [CELL_LOCAL])
    assert out_warm[CELL_LOCAL] == 1.0
    # Now reset.
    p.reload([])
    out_cold = p.predict(_features_with_embedding(_emb([1, 0, 0])), [CELL_LOCAL])
    assert out_cold[CELL_LOCAL] == 0.5


def test_predictor_id_is_stable() -> None:
    assert KNNPredictor().id == "knn"
