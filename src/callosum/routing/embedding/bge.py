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
                # FP16 on CUDA: BGE-large at FP32 is compute-bound on
                # the forward pass; on modern GPUs (Ampere, Hopper,
                # Blackwell) FP16 inference runs 2-3x faster with
                # negligible quality loss for embedding tasks. Falls
                # back to FP32 on CPU or older GPUs.
                import torch
                use_fp16 = torch.cuda.is_available()
                kwargs: dict = {}
                if use_fp16:
                    # model_kwargs flows into the underlying transformers
                    # AutoModel constructor; torch_dtype=float16 keeps the
                    # weights AND activations in half precision.
                    kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
                    kwargs["device"] = "cuda"
                self._model = SentenceTransformer(self._model_name, **kwargs)

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

    # BGE-large's max_seq_length is 512 tokens ≈ ~2000 English chars.
    # We pre-truncate input strings to a safety multiple of that ceiling
    # because sentence-transformers tokenizes the WHOLE input string
    # before truncating to max_seq_length — so feeding it 1MB of text
    # means tokenizing 1MB even though only the first ~2KB will ever
    # be looked at by the model. Tokenization is CPU-serial and scales
    # roughly linearly with input length; a single 1MB row can hang
    # the job for minutes / OOM the process. 8000 chars is comfortably
    # above the 512-token truncation boundary AND lets the model
    # see slightly different prefixes if the operator's tokenizer
    # produces fewer tokens per char than typical English.
    _MAX_INPUT_CHARS_FOR_ENCODE = 8000

    def encode_batch_sync(self, texts: list[str], *, batch_size: int = 64) -> list[bytes]:
        """Batch-encode many texts in a single GPU forward pass.

        Used by job scripts (e.g. embed_backfill) that have a backlog of
        rows and don't need the async per-request interface. The previous
        loop in embed_backfill called .embed() per row, causing one GPU
        dispatch per item; with batching we get ~one dispatch per
        batch_size items, which on GPU is significantly faster on
        BGE-large.

        Returns a list of float32 bytes the same length as `texts`.
        Inputs are pre-truncated to _MAX_INPUT_CHARS_FOR_ENCODE so
        absurdly-long prompts (e.g. accumulated Codex conversation
        contexts at 1MB+) don't stall the tokenizer.
        """
        self._ensure_loaded()
        import numpy as np
        assert self._model is not None
        # Pre-truncate before handing to encode — see comment on
        # _MAX_INPUT_CHARS_FOR_ENCODE for why. Slicing a Python str is
        # O(1) so this adds ~zero overhead, but avoids feeding the
        # tokenizer multi-MB inputs that produce identical embeddings
        # to a ~2KB prefix after BGE's internal truncation.
        truncated = [
            t[: self._MAX_INPUT_CHARS_FOR_ENCODE] if t else t
            for t in texts
        ]
        vecs = self._model.encode(
            truncated,
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
