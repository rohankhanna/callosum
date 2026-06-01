"""Tests for OperatorState SQLite + the merge_inference_params helper."""

from __future__ import annotations

from pathlib import Path

from callosum.operator_state import (
    OperatorState,
    merge_inference_params,
)

# ---------- merge_inference_params ----------


def test_merge_uses_backend_default_when_neither_operator_nor_client_set() -> None:
    out = merge_inference_params(
        backend_defaults={"think": False},
        operator_overrides={},
        operator_force=False,
        client_body={"model": "m", "messages": []},
    )
    assert out["think"] is False
    assert out["model"] == "m"


def test_merge_client_value_wins_over_backend_default() -> None:
    """If the client explicitly set `think=true`, the operator hasn't
    overridden anything, and the backend default is `think=false`,
    the client wins (default doesn't overwrite explicit values)."""
    out = merge_inference_params(
        backend_defaults={"think": False},
        operator_overrides={},
        operator_force=False,
        client_body={"model": "m", "think": True},
    )
    assert out["think"] is True


def test_merge_operator_override_wins_over_backend_default() -> None:
    out = merge_inference_params(
        backend_defaults={"think": False},
        operator_overrides={"think": True},
        operator_force=False,
        client_body={"model": "m"},
    )
    assert out["think"] is True


def test_merge_client_wins_over_non_forced_operator_override() -> None:
    """force=False is the (a) semantics — operator value is a DEFAULT
    that yields to whatever the client explicitly set."""
    out = merge_inference_params(
        backend_defaults={},
        operator_overrides={"temperature": 0.0},
        operator_force=False,
        client_body={"model": "m", "temperature": 0.9},
    )
    assert out["temperature"] == 0.9


def test_merge_forced_operator_override_wins_over_client() -> None:
    """force=True is the (b) semantics — operator value is authoritative
    and overwrites whatever the client sent. Used when an operator
    explicitly wants to clamp behavior for a misbehaving model."""
    out = merge_inference_params(
        backend_defaults={},
        operator_overrides={"temperature": 0.0},
        operator_force=True,
        client_body={"model": "m", "temperature": 0.9},
    )
    assert out["temperature"] == 0.0


def test_merge_doesnt_touch_unrelated_keys() -> None:
    """Body fields that aren't inference params (model, messages, tools)
    pass through untouched."""
    out = merge_inference_params(
        backend_defaults={"think": False},
        operator_overrides={"temperature": 0.5},
        operator_force=False,
        client_body={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function"}],
        },
    )
    assert out["messages"] == [{"role": "user", "content": "hi"}]
    assert out["tools"] == [{"type": "function"}]
    assert out["think"] is False
    assert out["temperature"] == 0.5


# ---------- OperatorState SQLite ----------


def test_get_returns_empty_dict_when_no_override_exists(tmp_path: Path) -> None:
    state = OperatorState(tmp_path / "op.sqlite")
    try:
        params, force = state.get_inference_overrides("never-set-model")
        assert params == {}
        assert force is False
    finally:
        state.close()


def test_set_and_get_roundtrip(tmp_path: Path) -> None:
    state = OperatorState(tmp_path / "op.sqlite")
    try:
        state.set_inference_overrides(
            "model-a0b0",
            {"think": False, "temperature": 0.0},
            force=True,
        )
        params, force = state.get_inference_overrides("model-a0b0")
        assert params == {"think": False, "temperature": 0.0}
        assert force is True
    finally:
        state.close()


def test_set_upserts_on_repeated_call(tmp_path: Path) -> None:
    """Calling set twice updates the existing row, doesn't insert
    duplicates."""
    state = OperatorState(tmp_path / "op.sqlite")
    try:
        state.set_inference_overrides("m", {"think": False})
        state.set_inference_overrides("m", {"think": True, "temperature": 0.7})
        params, force = state.get_inference_overrides("m")
        assert params == {"think": True, "temperature": 0.7}
        assert force is False
        # Only one row.
        listed = state.list_inference_overrides()
        assert len(listed) == 1
    finally:
        state.close()


def test_clear_removes_override(tmp_path: Path) -> None:
    state = OperatorState(tmp_path / "op.sqlite")
    try:
        state.set_inference_overrides("m", {"think": False})
        state.clear_inference_overrides("m")
        params, force = state.get_inference_overrides("m")
        assert params == {}
        assert force is False
    finally:
        state.close()


def test_state_survives_process_restart(tmp_path: Path) -> None:
    """SQLite persistence — reopening the same path should see the rows
    a prior session wrote."""
    p = tmp_path / "op.sqlite"
    state = OperatorState(p)
    state.set_inference_overrides("m", {"think": False}, force=True)
    state.close()
    # Reopen.
    state2 = OperatorState(p)
    try:
        params, force = state2.get_inference_overrides("m")
        assert params == {"think": False}
        assert force is True
    finally:
        state2.close()


def test_list_returns_all_rows(tmp_path: Path) -> None:
    state = OperatorState(tmp_path / "op.sqlite")
    try:
        state.set_inference_overrides("a", {"think": False})
        state.set_inference_overrides("b", {"temperature": 0.0}, force=True)
        rows = state.list_inference_overrides()
        models = [m for m, _, _ in rows]
        assert sorted(models) == ["a", "b"]
        # `b` was set with force=True.
        b_row = [r for r in rows if r[0] == "b"][0]
        assert b_row[2] is True
    finally:
        state.close()
