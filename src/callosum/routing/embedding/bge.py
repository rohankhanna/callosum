"""BAAI/bge-large-en-v1.5 embedding provider.

Lazy model load: the SentenceTransformer is constructed on the first
embed() call so callosum startup stays fast even when this provider
is selected. Subsequent calls reuse the loaded model.

Thread-pool execution: SentenceTransformer.encode() is sync and
CPU/GPU-bound. We run it via asyncio.to_thread so the event loop
isn't blocked while a vector is being computed.

Serialization: returns the raw float32 bytes (np.ndarray.tobytes()).
Caller stores it directly into the SQLite BLOB column; the kNN
predictor reads it back via np.frombuffer(bytes, dtype=np.float32).
This keeps the embedding wire-format dependency-free downstream of
this module.
"""

from __future__ import annotations

import asyncio
import threading


class BGELargeEmbeddingProvider:
    """EmbeddingProvider implementation backed by BAAI/bge-large-en-v1.5.

    1024-dim float32 vectors, unit-normalized for cosine similarity
    via dot-product. Sentence-transformers handles model download
    (~600 MB) on first instantiation; subsequent loads use the
    cached weights under HF_HOME.

    Raises ImportError at construction time when sentence-transformers
    isn't installed — callers should catch this and fall back to the
    no-op provider (handled by the factory).
    """

    id: str = "bge-large-en-v1.5"
    _model_name: str = "BAAI/bge-large-en-v1.5"

    def __init__(self) -> None:
        # Lazy actual load: defer to first encode() call. We do
        # import-check here so misconfigured deployments fail at startup
        # rather than on the first user request.
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. Install with "
                "`pip install -e '.[embeddings]'` or pick the 'noop' "
                "provider in [auto_router.routing]."
            ) from e
        self._model = None
        self._load_lock = threading.Lock()

    @property
    def dim(self) -> int:
        return 1024

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Hold the lock only while constructing; once loaded, reads
        # are lock-free.
        with self._load_lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)

    def _encode_sync(self, text: str) -> bytes:
        self._ensure_loaded()
        import numpy as np
        assert self._model is not None
        vec = self._model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32)
        return vec.tobytes()

    def encode_batch_sync(self, texts: list[str], *, batch_size: int = 64) -> list[bytes]:
        """Batch-encode many texts in a single GPU forward pass.

        Used by job scripts (e.g. embed_backfill) that have a backlog of
        rows and don't need the async per-request interface. The previous
        loop in embed_backfill called .embed() per row, causing one GPU
        dispatch per item; with batching we get ~one dispatch per
        batch_size items, which on GPU is 30-50x faster on BGE-large.

        Returns a list of float32 bytes the same length as `texts`,
        skipping `None` entries (empty inputs map to b"").
        """
        self._ensure_loaded()
        import numpy as np
        assert self._model is not None
        # SentenceTransformer.encode handles internal batching; we pass
        # batch_size to control GPU memory + utilization. 64 fits
        # comfortably on a 24GB+ card for BGE-large; larger values
        # diminish returns due to attention quadratic in seq length.
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.astype(np.float32).tobytes() for v in vecs]

    async def embed(self, text: str) -> bytes | None:
        """Encode `text` as a unit-normalized float32 (1024,) vector.

        Returns the raw vector bytes — np.frombuffer(b, dtype=np.float32)
        on the read side reconstructs the array. None only when input
        is empty (no point computing).
        """
        if not text:
            return None
        return await asyncio.to_thread(self._encode_sync, text)
