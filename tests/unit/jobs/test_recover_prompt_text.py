"""Hermetic unit tests for the pure helpers in recover_prompt_text.

The job re-extracts prompt_text/response_text from previously-captured
zlib payloads for rows that logged NULL text (the original extractor
missed the Codex Responses API shape). Its helpers are deliberately
defensive: _decompress_dict / _decompress_bytes return None
on any legacy/corrupt blob, and the checkpoint helpers tolerate
missing/corrupt state. These tests pin that contract with no sqlite, no
network, no app — only zlib/json and tmp_path.

_decompress_bytes is the distinct helper here (vs the sibling
apply_failure_labels job): it returns the RAW decompressed bytes
without JSON-parsing, so it must round-trip arbitrary binary, not just
text. The sqlite-backed _fetch_batch / run / main paths are
integration-shaped and out of scope for this pure-logic pin.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

from callosum.jobs.recover_prompt_text import (
    _decompress_bytes,
    _decompress_dict,
    _load_checkpoint,
    _save_checkpoint,
)

# ---------- _decompress_dict ---------------------------------------------


def test_decompress_dict_none_returns_none() -> None:
    assert _decompress_dict(None) is None


def test_decompress_dict_empty_bytes_returns_none() -> None:
    assert _decompress_dict(b"") is None


def test_decompress_dict_valid_dict_round_trips() -> None:
    payload = {"input": "hi", "instructions": "do thing", "n": [1, 2]}
    blob = zlib.compress(json.dumps(payload).encode())
    assert _decompress_dict(blob) == payload


def test_decompress_dict_non_dict_json_returns_none() -> None:
    blob = zlib.compress(json.dumps([1, 2, 3]).encode())
    assert _decompress_dict(blob) is None


def test_decompress_dict_corrupt_zlib_returns_none() -> None:
    assert _decompress_dict(b"not a zlib stream") is None


def test_decompress_dict_valid_zlib_invalid_json_returns_none() -> None:
    blob = zlib.compress(b"not json {")
    assert _decompress_dict(blob) is None


def test_decompress_dict_truncated_zlib_returns_none() -> None:
    blob = zlib.compress(json.dumps({"k": "v"}).encode())
    assert _decompress_dict(blob[:-4]) is None


# ---------- _decompress_bytes (raw binary round-trip) --------------------


def test_decompress_bytes_none_returns_none() -> None:
    assert _decompress_bytes(None) is None


def test_decompress_bytes_empty_returns_none() -> None:
    assert _decompress_bytes(b"") is None


def test_decompress_bytes_round_trips_text() -> None:
    raw = b'{"input": "hi"}'
    assert _decompress_bytes(zlib.compress(raw)) == raw


def test_decompress_bytes_round_trips_arbitrary_binary() -> None:
    # the distinct contract: raw bytes are returned WITHOUT json-parsing,
    # so non-text / non-utf8 binary must survive intact.
    raw = bytes(range(256))
    assert _decompress_bytes(zlib.compress(raw)) == raw


def test_decompress_bytes_corrupt_zlib_returns_none() -> None:
    assert _decompress_bytes(b"not zlib") is None


def test_decompress_bytes_truncated_returns_none() -> None:
    blob = zlib.compress(b"some payload bytes")
    assert _decompress_bytes(blob[:-3]) is None


# ---------- _load_checkpoint ---------------------------------------------


def test_load_checkpoint_missing_file_returns_zero(tmp_path: Path) -> None:
    assert _load_checkpoint(tmp_path / "nope.ckpt") == 0


def test_load_checkpoint_valid_cursor(tmp_path: Path) -> None:
    ckpt = tmp_path / "c.ckpt"
    ckpt.write_text(json.dumps({"cursor": 77, "saved_at": 1.0}))
    assert _load_checkpoint(ckpt) == 77


def test_load_checkpoint_corrupt_json_returns_zero(tmp_path: Path) -> None:
    (tmp_path / "c.ckpt").write_text("not json {")
    assert _load_checkpoint(tmp_path / "c.ckpt") == 0


def test_load_checkpoint_missing_cursor_key_returns_zero(tmp_path: Path) -> None:
    (tmp_path / "c.ckpt").write_text(json.dumps({"saved_at": 1.0}))
    assert _load_checkpoint(tmp_path / "c.ckpt") == 0


def test_load_checkpoint_non_numeric_cursor_returns_zero(tmp_path: Path) -> None:
    (tmp_path / "c.ckpt").write_text(json.dumps({"cursor": "abc"}))
    assert _load_checkpoint(tmp_path / "c.ckpt") == 0


def test_load_checkpoint_numeric_string_cursor_coerces(tmp_path: Path) -> None:
    (tmp_path / "c.ckpt").write_text(json.dumps({"cursor": "77"}))
    assert _load_checkpoint(tmp_path / "c.ckpt") == 77


# ---------- _save_checkpoint ---------------------------------------------


def test_save_checkpoint_round_trips(tmp_path: Path) -> None:
    ckpt = tmp_path / "c.ckpt"
    _save_checkpoint(ckpt, 55)
    assert _load_checkpoint(ckpt) == 55


def test_save_checkpoint_leaves_no_tmp_file(tmp_path: Path) -> None:
    ckpt = tmp_path / "c.ckpt"
    _save_checkpoint(ckpt, 3)
    assert not (tmp_path / "c.ckpt.tmp").exists()
    assert ckpt.exists()


def test_save_checkpoint_overwrites_existing(tmp_path: Path) -> None:
    ckpt = tmp_path / "c.ckpt"
    _save_checkpoint(ckpt, 10)
    _save_checkpoint(ckpt, 20)
    assert _load_checkpoint(ckpt) == 20


def test_save_checkpoint_payload_has_cursor_and_saved_at(tmp_path: Path) -> None:
    ckpt = tmp_path / "c.ckpt"
    _save_checkpoint(ckpt, 9)
    data = json.loads(ckpt.read_text())
    assert data["cursor"] == 9
    assert "saved_at" in data
