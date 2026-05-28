"""Recover prompt_text + response_text from previously-captured payloads.

For rows where:
  - prompt_text IS NULL but request_bodies.req_payload IS NOT NULL
  - response_text IS NULL but request_bodies.resp_payload IS NOT NULL

…decompress the payload and re-extract using the current dual-shape
extractor. Writes the recovered text back to the requests table.

Background: the original _extract_prompt_text in usage_log only handled
Chat Completions (`messages`) and missed the Codex Responses API
(`input` + `instructions`). Every /v1/responses request — i.e. every
Codex CLI session — logged NULL prompt_text from 2026-05-12 through
the fix in this branch. The req_payload was still captured, so we
can recover.

Runnable as a Dispatch job (or directly):

    callosum-recover-prompt-text \
        --db-path ~/.local/state/callosum/requests.sqlite \
        --batch-size 500
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

from callosum.usage_log import _extract_prompt_text, _extract_response_text

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


def _fetch_batch(
    conn: sqlite3.Connection, *, since_id: int, limit: int
) -> list[tuple[int, bytes | None, bytes | None]]:
    """Rows that have at least one missing text field AND a captured
    payload that could supply it."""
    return [
        (int(rid), req, resp)
        for rid, req, resp in conn.execute(
            """
            SELECT r.id, rb.req_payload, rb.resp_payload
            FROM requests r
            LEFT JOIN request_bodies rb ON rb.request_id = r.id
            WHERE r.id > ?
              AND (
                (r.prompt_text   IS NULL AND rb.req_payload  IS NOT NULL)
                OR (r.response_text IS NULL AND rb.resp_payload IS NOT NULL)
              )
            ORDER BY r.id ASC
            LIMIT ?
            """,
            (since_id, limit),
        ).fetchall()
    ]


def _decompress_dict(blob: bytes | None) -> dict | None:
    if not blob:
        return None
    try:
        raw = zlib.decompress(blob)
        out = json.loads(raw)
        return out if isinstance(out, dict) else None
    except Exception:
        return None


def _decompress_bytes(blob: bytes | None) -> bytes | None:
    if not blob:
        return None
    try:
        return zlib.decompress(blob)
    except Exception:
        return None


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
    prompt_recovered = 0
    response_recovered = 0
    start = time.time()
    print(f"recover_prompt_text: resuming cursor={cursor}", file=sys.stderr)
    try:
        while True:
            if _should_exit:
                print("recover_prompt_text: SIGTERM; exiting", file=sys.stderr)
                break
            if max_rows is not None and processed_total >= max_rows:
                print("recover_prompt_text: reached --max-rows", file=sys.stderr)
                break
            rows = _fetch_batch(conn, since_id=cursor, limit=batch_size)
            if not rows:
                print("recover_prompt_text: no more rows to recover", file=sys.stderr)
                break
            updates: list[tuple[str | None, str | None, int]] = []
            for rid, req_blob, resp_blob in rows:
                req_dict = _decompress_dict(req_blob)
                resp_raw = _decompress_bytes(resp_blob)
                prompt = _extract_prompt_text(req_dict)
                response = _extract_response_text(resp_raw)
                if prompt is not None:
                    prompt_recovered += 1
                if response is not None:
                    response_recovered += 1
                if prompt is not None or response is not None:
                    updates.append((prompt, response, rid))
            if updates:
                with conn:
                    # Use COALESCE so we only overwrite columns where
                    # we have a non-null recovered value — never blow
                    # away whatever's already there.
                    conn.executemany(
                        "UPDATE requests SET "
                        "  prompt_text   = COALESCE(?, prompt_text), "
                        "  response_text = COALESCE(?, response_text) "
                        "WHERE id = ?",
                        updates,
                    )
            cursor = rows[-1][0]
            _save_checkpoint(checkpoint_path, cursor)
            processed_total += len(rows)
            elapsed = time.time() - start
            print(
                f"recover_prompt_text: cursor={cursor} "
                f"processed={processed_total} "
                f"prompts={prompt_recovered} responses={response_recovered} "
                f"elapsed={elapsed:.1f}s",
                file=sys.stderr,
            )
    finally:
        conn.close()
    return 0


def main() -> int:
    signal.signal(signal.SIGTERM, _on_sigterm)
    parser = argparse.ArgumentParser(prog="callosum.jobs.recover_prompt_text")
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    ckpt = args.checkpoint_path or args.db_path.with_suffix(
        args.db_path.suffix + ".recover_prompt_text.ckpt"
    )
    return run(
        db_path=args.db_path,
        checkpoint_path=ckpt,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    sys.exit(main())
