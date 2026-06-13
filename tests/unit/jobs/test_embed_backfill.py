"""Tests for the embed_backfill Dispatch job.

The job has a real BGE dependency; tests stub it via monkeypatch so we
can verify the batch / cursor / checkpoint logic without loading the
actual model.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from callosum.jobs import embed_backfill


@pytest.fixture
def db(tmp_path: Path) -> Path:
    """Build a minimal SQLite that looks like the request log schema —
    just the columns the backfill touches."""
    p = tmp_path / "requests.sqlite"
    conn = sqlite3.connect(str(p))
    conn.execute(
        "CREATE TABLE requests (id INTEGER PRIMARY KEY AUTOINCREMENT, prompt_text TEXT, prompt_embedding BLOB)"
    )
    conn.executemany(
        "INSERT INTO requests (prompt_text, prompt_embedding) VALUES (?, ?)",
        [
            ("first prompt", None),
            ("second prompt", None),
            (None, None),  # no prompt — skipped
            ("fourth prompt", b"already-embedded"),  # already done
            ("fifth prompt", None),
        ],
    )
    conn.commit()
    conn.close()
    return p


@pytest.fixture
def stub_provider(monkeypatch):
    """Replace BGELargeEmbeddingProvider with a deterministic fake — each
    text gets a 4-dim vector based on its length, so we can assert.

    Exposes both async `embed` and batched `encode_batch_sync` so the
    fixture works whether the job calls the per-row or batched path.
    """

    class _FakeProvider:
        @property
        def dim(self) -> int:
            return 4

        def _vec(self, text: str) -> bytes:
            v = np.array([len(text), 1.0, 2.0, 3.0], dtype=np.float32)
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            return v.tobytes()

        async def embed(self, text: str) -> bytes | None:
            if not text:
                return None
            return self._vec(text)

        def encode_batch_sync(self, texts: list[str], *, batch_size: int = 64) -> list[bytes]:
            return [self._vec(t) for t in texts]

    monkeypatch.setattr(
        "callosum.routing.embedding.bge.BGELargeEmbeddingProvider",
        _FakeProvider,
    )
    return _FakeProvider


def test_backfill_populates_embeddings_for_rows_with_prompt_text(db: Path, stub_provider, tmp_path: Path) -> None:
    """Rows 1, 2, 5 should get embedded; row 3 (no prompt_text) and row 4
    (already has embedding) are left alone."""
    ckpt = tmp_path / "ck.json"
    rc = embed_backfill.run(
        db_path=db,
        checkpoint_path=ckpt,
        batch_size=64,
        max_rows=None,
    )
    assert rc == 0
    conn = sqlite3.connect(str(db))
    rows = conn.execute("SELECT id, prompt_embedding IS NOT NULL FROM requests ORDER BY id").fetchall()
    conn.close()
    assert rows == [(1, 1), (2, 1), (3, 0), (4, 1), (5, 1)]


def test_backfill_writes_checkpoint_after_processing(db: Path, stub_provider, tmp_path: Path) -> None:
    """After successful completion the checkpoint records the max id
    processed — a resumed run starts AFTER it (no work redone)."""
    ckpt = tmp_path / "ck.json"
    embed_backfill.run(
        db_path=db,
        checkpoint_path=ckpt,
        batch_size=64,
        max_rows=None,
    )
    assert ckpt.exists()
    data = json.loads(ckpt.read_text())
    assert data["cursor"] == 5  # last id processed


def test_backfill_resumes_from_checkpoint(db: Path, stub_provider, tmp_path: Path) -> None:
    """Pre-seeded checkpoint at cursor=2 → backfill only processes
    rows with id > 2."""
    ckpt = tmp_path / "ck.json"
    ckpt.write_text(json.dumps({"cursor": 2}))
    # Clear any pre-existing embeddings on rows 1-2 so we can prove they
    # were not touched. (They weren't anyway in the fixture, but be
    # explicit.)
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE requests SET prompt_embedding = NULL WHERE id IN (1, 2)")
    conn.commit()
    conn.close()
    embed_backfill.run(
        db_path=db,
        checkpoint_path=ckpt,
        batch_size=64,
        max_rows=None,
    )
    conn = sqlite3.connect(str(db))
    rows = conn.execute("SELECT id, prompt_embedding IS NOT NULL FROM requests ORDER BY id").fetchall()
    conn.close()
    # 1, 2 still NULL because cursor said skip them.
    assert rows == [(1, 0), (2, 0), (3, 0), (4, 1), (5, 1)]


def test_backfill_respects_max_rows(db: Path, stub_provider, tmp_path: Path) -> None:
    """When --max-rows is given, the job stops after the batch that
    crosses it. batch_size=1 makes the cap exact — useful for Dispatch
    jobs that should run for a bounded time slice before yielding the
    GPU. (Larger batch_size may overshoot by up to batch_size-1.)"""
    ckpt = tmp_path / "ck.json"
    embed_backfill.run(
        db_path=db,
        checkpoint_path=ckpt,
        batch_size=1,
        max_rows=2,
    )
    conn = sqlite3.connect(str(db))
    embedded = conn.execute(
        "SELECT COUNT(*) FROM requests WHERE prompt_embedding IS NOT NULL AND prompt_text IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    # 2 newly embedded (rows 1, 2) + 1 already embedded (row 4) = 3.
    # Row 5 is left unembedded because we hit max_rows after row 2.
    assert embedded == 3
    # Row 5 specifically should still be NULL.
    conn = sqlite3.connect(str(db))
    row5_emb = conn.execute("SELECT prompt_embedding FROM requests WHERE id = 5").fetchone()[0]
    conn.close()
    assert row5_emb is None


def test_backfill_missing_db_returns_error(tmp_path: Path, stub_provider) -> None:
    rc = embed_backfill.run(
        db_path=tmp_path / "does-not-exist.sqlite",
        checkpoint_path=tmp_path / "ck.json",
        batch_size=8,
        max_rows=None,
    )
    assert rc == 1
