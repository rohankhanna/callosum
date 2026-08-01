"""Unit tests for `callosum.gap_triage.classify_gap`.

The gap classifier is the heart of the P3 auto-dev rewrite: it turns a
failing capability-probe finding into an ADR section-5 action. These tests
pin both the structured-field path (a finding carrying `suggested_action`)
and the backward-compat fallback path (old-shape findings classified from
the free-prose `adapter_hint`).
"""

from __future__ import annotations

from callosum.gap_triage import (
    NO_FEASIBLE_ADAPTER_HINT_SUBSTRINGS,
    classify_gap,
)
from callosum.substrate_contract import ContractAction


def _finding(
    *,
    dimension: str = "tool_call_shape",
    status: str = "fail",
    adapter_hint: str | None = "parse tool-call JSON from message text",
    suggested_action: ContractAction | None = None,
    gap_class: str | None = None,
    upstream_owner: str | None = None,
    close_condition: str | None = None,
) -> dict:
    f: dict = {
        "dimension": dimension,
        "status": status,
        "summary": "x",
        "adapter_hint": adapter_hint,
    }
    if suggested_action is not None:
        f["suggested_action"] = suggested_action.value
    if gap_class is not None:
        f["gap_class"] = gap_class
    if upstream_owner is not None:
        f["upstream_owner"] = upstream_owner
    if close_condition is not None:
        f["close_condition"] = close_condition
    return f


# --- structured-field path -------------------------------------------------

def test_structured_author_temporary_adapter_is_returned() -> None:
    cls = classify_gap(
        _finding(
            suggested_action=ContractAction.AUTHOR_TEMPORARY_ADAPTER,
            gap_class="B",
            upstream_owner="substrate X",
            close_condition="substrate fronts the surface",
        )
    )
    assert cls is not None
    assert cls.action is ContractAction.AUTHOR_TEMPORARY_ADAPTER
    assert cls.dimension == "tool_call_shape"
    assert cls.upstream_owner == "substrate X"
    assert cls.close_condition == "substrate fronts the surface"


def test_structured_quarantine_cell_is_returned() -> None:
    cls = classify_gap(_finding(suggested_action=ContractAction.QUARANTINE_CELL))
    assert cls is not None
    assert cls.action is ContractAction.QUARANTINE_CELL
    assert "quarantine" in cls.reason


def test_structured_route_native_is_returned() -> None:
    cls = classify_gap(_finding(suggested_action=ContractAction.ROUTE_NATIVE))
    assert cls is not None
    assert cls.action is ContractAction.ROUTE_NATIVE


def test_structured_remove_shim_is_returned() -> None:
    cls = classify_gap(_finding(suggested_action=ContractAction.REMOVE_SHIM))
    assert cls is not None
    assert cls.action is ContractAction.REMOVE_SHIM


def test_structured_verify_fix_is_returned() -> None:
    cls = classify_gap(_finding(suggested_action=ContractAction.VERIFY_FIX))
    assert cls is not None
    assert cls.action is ContractAction.VERIFY_FIX


def test_structured_file_upstream_gap_is_not_actionable() -> None:
    # file_upstream_gap is metadata on the step-2 adapter, not a loop
    # action; classify_gap does not produce it from a single finding.
    assert (
        classify_gap(_finding(suggested_action=ContractAction.FILE_UPSTREAM_GAP))
        is None
    )


def test_structured_residual_translate_is_not_actionable() -> None:
    # residual_translate needs registry/substrate state (P4/P5).
    assert (
        classify_gap(_finding(suggested_action=ContractAction.RESIDUAL_TRANSLATE))
        is None
    )


def test_unknown_suggested_action_falls_back_to_hint() -> None:
    # A corrupt/unknown action string should not crash; fall back to hint.
    cls = classify_gap(
        {"dimension": "tool_call_shape", "status": "fail", "adapter_hint": "fix it", "suggested_action": "bogus_action"}
    )
    assert cls is not None
    assert cls.action is ContractAction.AUTHOR_TEMPORARY_ADAPTER


# --- backward-compat fallback path (old-shape findings) -------------------

def test_old_shape_non_sentinel_hint_authors_temporary_adapter() -> None:
    cls = classify_gap(
        {"dimension": "tool_call_shape", "status": "fail", "adapter_hint": "parse tool-call JSON from message text"}
    )
    assert cls is not None
    assert cls.action is ContractAction.AUTHOR_TEMPORARY_ADAPTER


def test_old_shape_sentinel_hint_quarantines() -> None:
    for sentinel in NO_FEASIBLE_ADAPTER_HINT_SUBSTRINGS:
        cls = classify_gap(
            {"dimension": "tool_call_shape", "status": "fail", "adapter_hint": f"{sentinel} here"}
        )
        assert cls is not None
        assert cls.action is ContractAction.QUARANTINE_CELL, sentinel


def test_old_shape_no_hint_is_not_actionable() -> None:
    assert classify_gap({"dimension": "tool_call_shape", "status": "fail", "adapter_hint": None}) is None
    assert classify_gap({"dimension": "tool_call_shape", "status": "fail", "adapter_hint": "   "}) is None
    assert classify_gap({"dimension": "tool_call_shape", "status": "fail"}) is None


# --- non-fail statuses are never actionable --------------------------------

def test_pass_error_skipped_are_not_actionable() -> None:
    for status in ("pass", "error", "skipped"):
        assert (
            classify_gap(
                {"dimension": "tool_call_shape", "status": status, "adapter_hint": "no feasible adapter"}
            )
            is None
        ), status


def test_missing_dimension_is_handled() -> None:
    cls = classify_gap({"status": "fail", "adapter_hint": "fix it"})
    assert cls is not None
    assert cls.action is ContractAction.AUTHOR_TEMPORARY_ADAPTER
    assert cls.dimension is None