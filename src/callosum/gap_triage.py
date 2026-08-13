"""Capability-gap triage: classify a failing dimension finding into an
ADR section-5 action.

This is the P3 auto-dev rewrite's core classifier. The dev-loop used to treat
any failing capability-probe adapter_hint as an instruction to author a
transform under src/callosum/transforms/. ADR
docs/adr/2026-07-28-substrate-compatibility-contract.md section 5 replaces
that reflex with a priority order:

    1. route_native           — substrate handles the surface; write nothing.
    2. author_temporary_adapter — Class B model-specific quirk; author +
       label TEMPORARY-DEBT + record upstream owner + close condition.
    3. remove_shim / verify_fix — substrate now fronts the surface;
       delete the callosum-side adapter + re-probe.
    4. quarantine_cell         — no feasible adapter; drop the cell for
       the violated surface.

This module lives outside substrate_contract.py so that module stays pure
(its docstring promises it imports nothing from the routing pipeline).
classify_gap is the bridge: it consumes the serialized dimension-finding
dict that flows through PerceptionSnapshot.profiles[cell]["findings"][dim]
and returns a ContractClassification. The conformance/capability classifier
classify_contract handles surface-contract gaps (conformance invariants +
capability fields); it stays dormant until P4 lands real conformance data.
classify_gap is the first live classifier the dev-loop consumes.

Backward compatibility: a finding without the structured suggested_action
field (an older persisted profile, or a hand-built test fixture) is classified
by fallback inference from the free-prose adapter_hint — the same logic
the pre-filter used before P3. This seam keeps every existing pre-filter test
green without editing their finding dicts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from callosum.substrate_contract import ContractAction, ContractClassification

__all__ = ["classify_gap", "NO_FEASIBLE_ADAPTER_HINT_SUBSTRINGS"]


# Substrings inside adapter_hint that mean "no adapter can make this cell
# work" — the "no feasible adapter" sentinel. Moved here from
# dev_loop.pre_filter._NO_ACTION_HINT_SUBSTRINGS so the gap classifier owns
# the vocabulary. Case-insensitive substring match.
NO_FEASIBLE_ADAPTER_HINT_SUBSTRINGS: tuple[str, ...] = (
    "no feasible adapter",
    "no adapter feasible",
)


def _coerce_action(value: Any) -> ContractAction | None:
    """Accept a ContractAction enum or its string value; None otherwise."""
    if isinstance(value, ContractAction):
        return value
    if isinstance(value, str) and value:
        try:
            return ContractAction(value)
        except ValueError:
            return None
    return None


def classify_gap(finding: Mapping[str, Any]) -> ContractClassification | None:
    """Classify a dimension finding into an ADR section-5 action.

    Returns None when there is nothing for the dev-loop to act on:
    non-fail findings (pass/error/skipped), or a fail finding with no
    actionable signal (no structured suggested_action and no usable
    adapter_hint).

    Decision order:

    1. status != "fail" -> None. Only an informative failure opens
       the gap-triage path.
    2. If the finding carries a structured suggested_action, use it
       (the dimension probe already classified the gap deterministically).
    3. Otherwise, fall back to inferring from the free-prose adapter_hint
       (the pre-P3 behavior, preserved for old profiles + test fixtures):
       a sentinel substring -> QUARANTINE_CELL; any other non-empty hint
       -> AUTHOR_TEMPORARY_ADAPTER; absent/empty hint -> None.

    The returned ContractClassification carries the dimension plus the
    TEMPORARY-DEBT metadata (upstream_owner / close_condition) from the
    finding, so the dev-loop's prompt can hand the agent the labeling
    requirements without re-reading the finding.
    """
    if finding.get("status") != "fail":
        return None

    dimension = finding.get("dimension")
    dimension_str = str(dimension) if isinstance(dimension, str) and dimension else ""

    action = _coerce_action(finding.get("suggested_action"))
    if action is None:
        # Fallback inference from the free-prose adapter_hint (old-shape
        # findings). This is the backward-compat seam.
        hint = finding.get("adapter_hint")
        hint_str = hint if isinstance(hint, str) else ""
        if not hint_str.strip():
            return None
        if any(s in hint_str.lower() for s in NO_FEASIBLE_ADAPTER_HINT_SUBSTRINGS):
            action = ContractAction.QUARANTINE_CELL
        else:
            action = ContractAction.AUTHOR_TEMPORARY_ADAPTER

    upstream_owner = finding.get("upstream_owner")
    close_condition = finding.get("close_condition")
    upstream_owner_str = upstream_owner if isinstance(upstream_owner, str) and upstream_owner else None
    close_condition_str = (
        close_condition if isinstance(close_condition, str) and close_condition else None
    )

    if action is ContractAction.QUARANTINE_CELL:
        reason = (
            f"no feasible adapter for {dimension_str or 'this dimension'} — "
            "quarantine the cell for the violated surface (record, do not "
            "author code)"
        )
    elif action is ContractAction.AUTHOR_TEMPORARY_ADAPTER:
        reason = (
            f"Class B gap on {dimension_str or 'this dimension'} — author a "
            "temporary adapter, label it TEMPORARY-DEBT, and record the "
            "upstream owner + close condition"
        )
    elif action is ContractAction.ROUTE_NATIVE:
        reason = (
            f"Class A gap on {dimension_str or 'this dimension'} — the "
            "substrate owns this; route native and write nothing"
        )
    elif action in (ContractAction.REMOVE_SHIM, ContractAction.VERIFY_FIX):
        reason = (
            f"substrate now fronts {dimension_str or 'this dimension'} — "
            f"delete the callosum-side adapter ({action.value}) and re-probe"
        )
    else:
        # FILE_UPSTREAM_GAP / RESIDUAL_TRANSLATE are not produced by this
        # classifier from a single dimension finding; they need substrate /
        # registry state (P4/P5). Treat as non-actionable here.
        return None

    return ContractClassification(
        action=action,
        reason=reason,
        dimension=dimension_str or None,
        upstream_owner=upstream_owner_str,
        close_condition=close_condition_str,
    )