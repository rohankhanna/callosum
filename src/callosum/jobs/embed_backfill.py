"""Backfill prompt embeddings for request-log rows that lack them.

Runnable as a Dispatch job:

    python -m callosum.jobs.embed_backfill \
        --db-path /home/<user>/.local/state/callosum/requests.sqlite \
        --checkpoint-path /home/<user>/.local/state/callosum/embed_backfill.ckpt \
        --batch-size 64 \
        [--max-rows 10000]

Selects rows where `prompt_text IS NOT NULL AND prompt_embedding IS NULL`,
in ascending id order. Within each batch:

    1. Read N rows starting at the cursor.
    2. Compute embeddings (sync, GPU-bound).
    3. Write them back in a single transaction.
    4. Advance the cursor to the highest processed id and persist to
       the checkpoint file atomically.

SIGTERM handling: the loop checks a flag after every batch. On signal,
the job finishes the current batch (so no in-flight embedding is lost),
persists the checkpoint, and exits 0. Dispatch reschedules the job
when GPU goes idle again; it resumes from the checkpoint.

Best-effort: any row that fails to embed is logged and skipped on the
next pass (cursor still advances). Without this, one malformed row
would block the whole backfill forever.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import sys
import time
from pathlib import Path

# Module-level state for the SIGTERM handler. Set to True when we see
# SIGTERM; the batch loop polls this and exits between batches.
_should_exit = False


def _on_sigterm(signum, frame) -> None:  # type: ignore[no-untyped-def]
    """SIGTERM handler — set a flag; the loop honors it between batches."""
    global _should_exit
    _should_exit = True


def _load_checkpoint(path: Path) -> int:
    """Return the last processed request_id, or 0 if no checkpoint exists."""
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text())
        return int(data.get("cursor", 0))
    except Exception:
        return 0


def _save_checkpoint(path: Path, cursor: int) -> None:
    """Atomic write so a SIGKILL mid-write doesn't corrupt the file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"cursor": cursor, "saved_at": time.time()}))
    os.replace(tmp, path)


def _fetch_batch(conn: sqlite3.Connection, *, since_id: int, limit: int) -> list[tuple[int, str]]:
    """Return (request_id, prompt_text) pairs for rows needing embeddings."""
    rows = conn.execute(
        "SELECT id, prompt_text FROM requests "
        "WHERE id > ? AND prompt_text IS NOT NULL AND prompt_embedding IS NULL "
        "ORDER BY id ASC LIMIT ?",
        (since_id, limit),
    ).fetchall()
    return [(int(r[0]), str(r[1])) for r in rows if r[1]]


def _write_embeddings(conn: sqlite3.Connection, items: list[tuple[int, bytes]]) -> None:
    """Write embeddings for `items = [(request_id, embedding_bytes), ...]`
    in a single transaction. Skipped rows (failed embeddings) don't appear
    in `items`; they remain prompt_embedding=NULL and get retried on the
    next backfill run."""
    if not items:
        return
    with conn:
        conn.executemany(
            "UPDATE requests SET prompt_embedding = ? WHERE id = ?",
            [(emb, rid) for rid, emb in items],
        )


def run(
    *,
    db_path: Path,
    checkpoint_path: Path,
    batch_size: int,
    max_rows: int | None,
) -> int:
    """Main loop. Returns process exit code (0=success/empty, 1=fatal error)."""
    if not db_path.exists():
        print(f"db not found: {db_path}", file=sys.stderr)
        return 1
    # Import the embedding provider lazily — both because it's a heavy
    # import (torch + huggingface) and so unit tests can stub it.
    from callosum.routing.embedding.bge import BGELargeEmbeddingProvider

    provider = BGELargeEmbeddingProvider()

    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA journal_mode = WAL")
    cursor = _load_checkpoint(checkpoint_path)
    processed_total = 0
    start_ts = time.time()
    print(f"embed_backfill: resuming from cursor={cursor}", file=sys.stderr)

    try:
        while True:
            if _should_exit:
                print("embed_backfill: SIGTERM received; exiting cleanly", file=sys.stderr)
                break
            if max_rows is not None and processed_total >= max_rows:
                print(f"embed_backfill: reached --max-rows={max_rows}", file=sys.stderr)
                break
            rows = _fetch_batch(conn, since_id=cursor, limit=batch_size)
            if not rows:
                print("embed_backfill: no more rows to embed", file=sys.stderr)
                break
            texts = [text for _, text in rows]
            ids = [rid for rid, _ in rows]
            try:
                # Single batched GPU dispatch for the whole batch. Previous
                # versions called provider.embed() per row, producing one
                # GPU dispatch per text — wasted parallelism and ~30-50x
                # slower than the natural batch encode.
                embeddings = provider.encode_batch_sync(texts, batch_size=batch_size)
            except Exception as exc:
                print(
                    f"embed_backfill: batch starting at row {ids[0]} failed "
                    f"({type(exc).__name__}: {exc}); falling back to per-row",
                    file=sys.stderr,
                )
                # Defensive fallback: if the whole batch fails (OOM,
                # bad input mid-batch), retry one row at a time so a
                # single malformed row can't block the whole job.
                embeddings = []
                for text in texts:
                    try:
                        embeddings.append(provider.encode_batch_sync([text], batch_size=1)[0])
                    except Exception:
                        embeddings.append(b"")
            items = [(rid, emb) for rid, emb in zip(ids, embeddings, strict=True) if emb]
            _write_embeddings(conn, items)
            cursor = rows[-1][0]
            _save_checkpoint(checkpoint_path, cursor)
            processed_total += len(rows)
            elapsed = time.time() - start_ts
            rate = processed_total / elapsed if elapsed > 0 else 0
            print(
                f"embed_backfill: cursor={cursor} processed={processed_total} rate={rate:.1f}/s elapsed={elapsed:.1f}s",
                file=sys.stderr,
            )
    finally:
        conn.close()
    return 0


def main() -> int:
    """CLI entry. Returns exit code."""
    signal.signal(signal.SIGTERM, _on_sigterm)
    parser = argparse.ArgumentParser(prog="callosum.jobs.embed_backfill")
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Defaults to <db-path>.embed_backfill.ckpt",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap total rows processed this run (omit for unlimited).",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint_path
    if checkpoint is None:
        checkpoint = args.db_path.with_suffix(args.db_path.suffix + ".embed_backfill.ckpt")
    return run(
        db_path=args.db_path,
        checkpoint_path=checkpoint,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    sys.exit(main())
