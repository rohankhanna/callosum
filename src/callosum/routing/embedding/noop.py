"""No-op embedding provider — returns None, dim=0.

Used during cold start (no embedding model wired in), during tests, and
as a sanity-check fallback when the real provider fails to load. Lets
the rest of the pipeline run end-to-end with zero ML dependencies.
"""

from __future__ import annotations


class NoopEmbeddingProvider:
    """EmbeddingProvider impl that emits no embeddings."""

    id: str = "noop"

    @property
    def dim(self) -> int:
        return 0

    async def embed(self, text: str) -> bytes | None:
        return None
