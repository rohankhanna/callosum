"""Hermetic unit tests for the model-probe failure-backoff state machine.

callosum.model_probe keeps a two-tier failure backoff for the periodic
model-fit probe: a (content_hash, lane_hash) combo that keeps failing is
retried up to confirmations times, then marked confirmed_broken and
skipped until the model artifact OR the lane changes -- which resets the
counter. These tests pin that state machine plus its JSON persistence
helpers and the lane-hash fingerprint, with no GPU, no network, no app,
no subprocess (_stack_versions is monkeypatched for determinism).

The probe/parse/preflight/runner paths are covered by
tests/unit/jobs/test_model_probe.py; this file covers ONLY the
backoff state machine, which that file does not touch.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import callosum.model_probe as mp

_CONFIRM = 3


def _state(path: Path) -> dict[str, dict[str, object]]:
    return mp._read_backoff_state(path)


# ---------- _backoff_state_path ------------------------------------------


def test_backoff_state_path_colocated_with_usage_log() -> None:
    log = SimpleNamespace(path=str(Path("/tmp/x/requests.sqlite")))
    assert mp._backoff_state_path(log) == Path("/tmp/x/model_probe_backoff.json")


def test_backoff_state_path_none_when_no_path_attr() -> None:
    # back-compat for mocks/tests: no .path -> backoff silently disabled
    assert mp._backoff_state_path(SimpleNamespace()) is None


# ---------- _read_backoff_state -------------------------------------------


def test_read_backoff_state_missing_file_returns_empty(tmp_path: Path) -> None:
    assert mp._read_backoff_state(tmp_path / "nope.json") == {}


def test_read_backoff_state_valid_dict(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"m1": {"consecutive_failures": 2}}))
    assert mp._read_backoff_state(p) == {"m1": {"consecutive_failures": 2}}


def test_read_backoff_state_corrupt_json_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    p.write_text("not json {")
    assert mp._read_backoff_state(p) == {}


def test_read_backoff_state_non_dict_json_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    p.write_text(json.dumps([1, 2, 3]))
    assert mp._read_backoff_state(p) == {}


# ---------- _write_backoff_state ------------------------------------------


def test_write_backoff_state_round_trips(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    mp._write_backoff_state(p, {"m1": {"consecutive_failures": 5}})
    assert mp._read_backoff_state(p) == {"m1": {"consecutive_failures": 5}}


def test_write_backoff_state_leaves_no_tmp(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    mp._write_backoff_state(p, {"m1": {}})
    assert not (tmp_path / "b.json.tmp").exists()
    assert p.exists()


def test_write_backoff_state_overwrites(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    mp._write_backoff_state(p, {"m1": {"consecutive_failures": 1}})
    mp._write_backoff_state(p, {"m1": {"consecutive_failures": 9}})
    assert mp._read_backoff_state(p) == {"m1": {"consecutive_failures": 9}}


# ---------- _record_backoff_failure: guards -------------------------------


def test_record_failure_noop_when_path_none(tmp_path: Path) -> None:
    mp._record_backoff_failure(None, "m1", "h", "l", "boom", 1.0, _CONFIRM)
    # nothing to assert beyond no-raise; ensure no file materializes anywhere
    assert not (tmp_path / "b.json").exists()


def test_record_failure_noop_when_empty_content_hash(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    mp._record_backoff_failure(p, "m1", "", "l", "boom", 1.0, _CONFIRM)
    assert not p.exists()


# ---------- _record_backoff_failure: the counter state machine -------------


def _record(
    p: Path, *, model: str = "m1", ch: str = "ch1", lh: str = "lh1", reason: str = "boom", ts: float = 1.0
) -> None:
    mp._record_backoff_failure(p, model, ch, lh, reason, ts, _CONFIRM)


def test_record_failure_first_failure_counts_one_not_confirmed(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    _record(p, ts=10.0)
    entry = _state(p)["m1"]
    assert entry["consecutive_failures"] == 1
    assert entry["confirmed_broken"] is False
    assert entry["content_hash"] == "ch1"
    assert entry["lane_hash"] == "lh1"
    assert entry["last_failed_at"] == 10.0
    assert entry["last_reason"] == "boom"


def test_record_failure_same_combo_increments(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    _record(p)
    _record(p, ts=2.0)
    entry = _state(p)["m1"]
    assert entry["consecutive_failures"] == 2
    assert entry["confirmed_broken"] is False


def test_record_failure_reaches_confirmations_marks_confirmed_broken(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    _record(p)
    _record(p)
    _record(p)
    assert _state(p)["m1"]["consecutive_failures"] == 3
    assert _state(p)["m1"]["confirmed_broken"] is True


def test_record_failure_changed_content_hash_resets_counter(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    _record(p)
    _record(p)
    # new artifact -> fresh evaluation, even though prior was at 2
    _record(p, ch="ch2")
    entry = _state(p)["m1"]
    assert entry["consecutive_failures"] == 1
    assert entry["confirmed_broken"] is False
    assert entry["content_hash"] == "ch2"


def test_record_failure_changed_lane_hash_resets_even_if_confirmed(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    for _ in range(_CONFIRM):
        _record(p)
    assert _state(p)["m1"]["confirmed_broken"] is True
    # vLLM/stack upgrade changes the lane -> re-probe from 1
    _record(p, lh="lh2")
    entry = _state(p)["m1"]
    assert entry["consecutive_failures"] == 1
    assert entry["confirmed_broken"] is False
    assert entry["lane_hash"] == "lh2"


def test_record_failure_truncates_long_reason(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    _record(p, reason="x" * 400)
    assert _state(p)["m1"]["last_reason"] == "x" * 300


def test_record_failure_confirmations_one_confirms_immediately(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    mp._record_backoff_failure(p, "m1", "ch1", "lh1", "boom", 1.0, confirmations=1)
    entry = _state(p)["m1"]
    assert entry["consecutive_failures"] == 1
    assert entry["confirmed_broken"] is True


def test_record_failure_per_model_isolation(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    _record(p, model="m1")
    _record(p, model="m2")
    st = _state(p)
    assert st["m1"]["consecutive_failures"] == 1
    assert st["m2"]["consecutive_failures"] == 1


# ---------- _clear_backoff ------------------------------------------------


def test_clear_backoff_noop_when_path_none() -> None:
    # must not raise
    mp._clear_backoff(None, "m1")


def test_clear_backoff_removes_model_keeps_others(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    _record(p, model="m1")
    _record(p, model="m2")
    mp._clear_backoff(p, "m1")
    st = _state(p)
    assert "m1" not in st
    assert "m2" in st


def test_clear_backoff_absent_model_does_not_write(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    _record(p, model="m1")
    snap = p.read_text()
    mp._clear_backoff(p, "not-here")
    # no entry to drop -> no rewrite
    assert p.read_text() == snap


# ---------- _lane_hash ----------------------------------------------------


def test_lane_hash_same_config_same_stack_is_stable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CALLOSUM_LOCAL_LLM_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(mp, "_stack_versions", lambda cmd: "vllm 0.6.3 transformers 4.45.0")
    (tmp_path / "m1").mkdir()
    (tmp_path / "m1" / "launch.yaml").write_text("cmd: serve\n")
    h1 = mp._lane_hash("m1", ["local-llm"])
    h2 = mp._lane_hash("m1", ["local-llm"])
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_lane_hash_changes_when_serving_config_changes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CALLOSUM_LOCAL_LLM_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(mp, "_stack_versions", lambda cmd: "stack-v1")
    d = tmp_path / "m1"
    d.mkdir()
    (d / "launch.yaml").write_text("cmd: serve\n")
    before = mp._lane_hash("m1", ["local-llm"])
    (d / "launch.yaml").write_text("cmd: serve --changed\n")
    after = mp._lane_hash("m1", ["local-llm"])
    assert before != after


def test_lane_hash_changes_when_stack_version_changes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CALLOSUM_LOCAL_LLM_MODELS_DIR", str(tmp_path))
    (tmp_path / "m1").mkdir()
    (tmp_path / "m1" / "launch.yaml").write_text("cmd: serve\n")
    monkeypatch.setattr(mp, "_stack_versions", lambda cmd: "vllm 0.6.3")
    before = mp._lane_hash("m1", ["local-llm"])
    monkeypatch.setattr(mp, "_stack_versions", lambda cmd: "vllm 0.7.0")
    after = mp._lane_hash("m1", ["local-llm"])
    assert before != after


def test_lane_hash_missing_model_dir_still_returns_hex(monkeypatch, tmp_path: Path) -> None:
    # all config parts empty + stack "" -> deterministic, non-empty hash
    monkeypatch.setenv("CALLOSUM_LOCAL_LLM_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(mp, "_stack_versions", lambda cmd: "")
    h = mp._lane_hash("never-probed", ["local-llm"])
    assert len(h) == 64
    # two missing-model dirs hash the same (identical empty inputs)
    assert h == mp._lane_hash("also-missing", ["local-llm"])
