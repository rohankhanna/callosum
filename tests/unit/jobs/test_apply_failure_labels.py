"""Hermetic unit tests for the pure helpers in apply_failure_labels.

The job walks historical requests rows and applies failure labelers, so
its helpers are deliberately defensive: _decompress must return None
on any legacy/corrupt blob shape, and the checkpoint helpers must tolerate
missing/corrupt state and resume. These tests pin that defensive contract
with no sqlite, no network, no app — only zlib/json and tmp_path.

The sqlite-backed _fetch_batch / _write_labels / run paths are
integration-shaped (they need a UsageLog-built db plus score_row)
and are out of scope for this pure-logic pin.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

from callosum.jobs.apply_failure_labels import (
    _decompress,
    _load_checkpoint,
    _save_checkpoint,
)

# ---------- _decompress ---------------------------------------------------


def test_decompress_none_returns_none() -> None:
    assert _decompress(None) is None


def test_decompress_empty_bytes_returns_none() -> None:
    # empty blob is falsy -> short-circuits to None (no zlib call)
    assert _decompress(b"") is None


def test_decompress_valid_dict_round_trips() -> None:
    payload = {"model": "model-a0e7", "status": 200, "nested": {"a": [1, 2]}}
    blob = zlib.compress(json.dumps(payload).encode())
    assert _decompress(blob) == payload


def test_decompress_non_dict_json_returns_none() -> None:
    # a valid zlib+JSON payload that is a list (not a dict) must be rejected
    blob = zlib.compress(json.dumps([1, 2, 3]).encode())
    assert _decompress(blob) is None


def test_decompress_scalar_json_returns_none() -> None:
    blob = zlib.compress(b"5")
    assert _decompress(blob) is None


def test_decompress_string_json_returns_none() -> None:
    blob = zlib.compress(json.dumps("just a string").encode())
    assert _decompress(blob) is None


def test_decompress_corrupt_zlib_returns_none() -> None:
    # bytes that are not a valid zlib stream
    assert _decompress(b"not a zlib stream at all") is None


def test_decompress_valid_zlib_invalid_json_returns_none() -> None:
    blob = zlib.compress(b"not json {")
    assert _decompress(blob) is None


def test_decompress_truncated_zlib_returns_none() -> None:
    payload = zlib.compress(json.dumps({"k": "v"}).encode())
    # truncate the stream so decompression fails
    assert _decompress(payload[:-4]) is None


# ---------- _load_checkpoint ----------------------------------------------


def test_load_checkpoint_missing_file_returns_zero(tmp_path: Path) -> None:
    assert _load_checkpoint(tmp_path / "nope.ckpt") == 0


def test_load_checkpoint_valid_cursor(tmp_path: Path) -> None:
    ckpt = tmp_path / "c.ckpt"
    ckpt.write_text(json.dumps({"cursor": 42, "saved_at": 1.0}))
    assert _load_checkpoint(ckpt) == 42


def test_load_checkpoint_corrupt_json_returns_zero(tmp_path: Path) -> None:
    ckpt = tmp_path / "c.ckpt"
    ckpt.write_text("not json {")
    assert _load_checkpoint(ckpt) == 0


def test_load_checkpoint_missing_cursor_key_returns_zero(tmp_path: Path) -> None:
    ckpt = tmp_path / "c.ckpt"
    ckpt.write_text(json.dumps({"saved_at": 1.0}))
    assert _load_checkpoint(ckpt) == 0


def test_load_checkpoint_non_numeric_cursor_returns_zero(tmp_path: Path) -> None:
    ckpt = tmp_path / "c.ckpt"
    ckpt.write_text(json.dumps({"cursor": "abc"}))
    # int("abc") raises -> swallowed -> 0
    assert _load_checkpoint(ckpt) == 0


def test_load_checkpoint_numeric_string_cursor_coerces(tmp_path: Path) -> None:
    ckpt = tmp_path / "c.ckpt"
    ckpt.write_text(json.dumps({"cursor": "42"}))
    # int("42") succeeds -> 42 (defensive: tolerate stringified ints)
    assert _load_checkpoint(ckpt) == 42


# ---------- _save_checkpoint ---------------------------------------------


def test_save_checkpoint_round_trips(tmp_path: Path) -> None:
    ckpt = tmp_path / "c.ckpt"
    _save_checkpoint(ckpt, 99)
    assert _load_checkpoint(ckpt) == 99


def test_save_checkpoint_leaves_no_tmp_file(tmp_path: Path) -> None:
    # the atomic write uses a .tmp sibling + os.replace; the .tmp must not
    # linger after a successful save.
    ckpt = tmp_path / "c.ckpt"
    _save_checkpoint(ckpt, 7)
    assert not (tmp_path / "c.ckpt.tmp").exists()
    assert ckpt.exists()


def test_save_checkpoint_overwrites_existing(tmp_path: Path) -> None:
    ckpt = tmp_path / "c.ckpt"
    _save_checkpoint(ckpt, 10)
    _save_checkpoint(ckpt, 20)
    assert _load_checkpoint(ckpt) == 20


def test_save_checkpoint_written_payload_has_cursor_key(tmp_path: Path) -> None:
    ckpt = tmp_path / "c.ckpt"
    _save_checkpoint(ckpt, 33)
    data = json.loads(ckpt.read_text())
    assert data["cursor"] == 33
    assert "saved_at" in data
