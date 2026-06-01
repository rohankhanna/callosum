"""Apply failure-detection labelers to every request row that lacks a
quality_score.

Runs as a Dispatch job:

    python -m callosum.jobs.apply_failure_labels \
        --db-path /home/<user>/.local/state/callosum/requests.sqlite \
        --checkpoint-path /home/<user>/.local/state/callosum/apply_failure_labels.ckpt \
        --batch-size 200 \
        [--max-rows 10000]

For each unlabeled row with captured req/resp payloads, the labeler
runs the failure rules; if any fires, quality_score is set to -1.
Rows where no rule fires stay unlabeled — the kNN predictor treats
unlabeled rows as "no opinion."

Checkpointable (SIGTERM safe) and resumable, same pattern as
`embed_backfill`.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import sys
import time
import zlib
from pathlib import Path
from typing import Any

from callosum.routing.labeler.failures import score_row

_should_exit = False


def _on_sigterm(signum, frame) -> None:  # type: ignore[no-untyped-def]
    global _should_exit
    _should_exit = True


def _load_checkpoint(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text()).get("cursor", 0))
    except Exception:
        return 0


def _save_checkpoint(path: Path, cursor: int) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"cursor": cursor, "saved_at": time.time()}))
    os.replace(tmp, path)


def _decompress(blob: bytes | None) -> dict[str, Any] | None:
    """Decompress a zlib blob and decode JSON. Returns None on any
    failure — defensive, since we're walking historical rows that may
    have any number of legacy shapes."""
    if not blob:
        return None
    try:
        raw = zlib.decompress(blob)
        result = json.loads(raw)
        return result if isinstance(result, dict) else None
    except Exception:
        return None


def _fetch_batch(
    conn: sqlite3.Connection, *, since_id: int, limit: int
) -> list[tuple[int, dict[str, Any], bytes | None, bytes | None]]:
    """Return [(request_id, request_row_dict, req_blob, resp_blob), ...]
    for rows that need labeling: quality_score IS NULL AND
    response_bytes IS NOT NULL (we need a captured response to score)."""
    rows = conn.execute(
        """
        SELECT r.id, r.completion_tokens, r.status, r.classification,
               r.model, r.reasoning_effort,
               rb.req_payload, rb.resp_payload
        FROM requests r
        LEFT JOIN request_bodies rb ON rb.request_id = r.id
        WHERE r.id > ?
          AND r.quality_score IS NULL
          AND rb.resp_payload IS NOT NULL
        ORDER BY r.id ASC
        LIMIT ?
        """,
        (since_id, limit),
    ).fetchall()
    out: list[tuple[int, dict[str, Any], bytes | None, bytes | None]] = []
    for r in rows:
        out.append((
            int(r[0]),
            {
                "completion_tokens": r[1],
                "status": r[2],
                "classification": r[3],
                "model": r[4],
                "reasoning_effort": r[5],
            },
            r[6],
            r[7],
        ))
    return out


def _write_labels(
    conn: sqlite3.Connection, items: list[tuple[int, int]]
) -> None:
    """Write quality_score for rows that scored -1. Uses
    quality_label_method='implicit_failure_v1' as provenance so future
    label sources can be distinguished in the corpus."""
    if not items:
        return
    with conn:
        conn.executemany(
            "UPDATE requests SET quality_score = ?, "
            "quality_label_method = 'implicit_failure_v1' "
            "WHERE id = ?",
            [(score, rid) for rid, score in items],
        )


def run(
    *,
    db_path: Path,
    checkpoint_path: Path,
    batch_size: int,
    max_rows: int | None,
) -> int:
    if not db_path.exists():
        print(f"db not found: {db_path}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA journal_mode = WAL")
    cursor = _load_checkpoint(checkpoint_path)
    processed_total = 0
    labeled_total = 0
    start = time.time()
    print(f"apply_failure_labels: resuming cursor={cursor}", file=sys.stderr)
    try:
        while True:
            if _should_exit:
                print("apply_failure_labels: SIGTERM; exiting", file=sys.stderr)
                break
            if max_rows is not None and processed_total >= max_rows:
                print("apply_failure_labels: reached --max-rows", file=sys.stderr)
                break
            rows = _fetch_batch(conn, since_id=cursor, limit=batch_size)
            if not rows:
                print("apply_failure_labels: no more rows to label", file=sys.stderr)
                break
            to_write: list[tuple[int, int]] = []
            for rid, request_row, req_blob, resp_blob in rows:
                request_body = _decompress(req_blob)
                response_body = _decompress(resp_blob)
                score = score_row(
                    request_row=request_row,
                    response_body=response_body,
                    request_body=request_body,
                )
                if score is not None and score < 0:
                    to_write.append((rid, score))
            _write_labels(conn, to_write)
            cursor = rows[-1][0]
            _save_checkpoint(checkpoint_path, cursor)
            processed_total += len(rows)
            labeled_total += len(to_write)
            elapsed = time.time() - start
            print(
                f"apply_failure_labels: cursor={cursor} "
                f"processed={processed_total} labeled={labeled_total} "
                f"elapsed={elapsed:.1f}s",
                file=sys.stderr,
            )
    finally:
        conn.close()
    return 0


def main() -> int:
    signal.signal(signal.SIGTERM, _on_sigterm)
    parser = argparse.ArgumentParser(prog="callosum.jobs.apply_failure_labels")
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    ckpt = args.checkpoint_path
    if ckpt is None:
        ckpt = args.db_path.with_suffix(args.db_path.suffix + ".apply_failure_labels.ckpt")
    return run(
        db_path=args.db_path,
        checkpoint_path=ckpt,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    sys.exit(main())
