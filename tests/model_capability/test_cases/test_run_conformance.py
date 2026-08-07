"""Live baseline test for the substrate-contract conformance pipeline.

P5-prep. Mirrors test_run_all_dimensions.py: pytest invokes
build_contract_profile (the producer for the dormant
classify_contract pipeline) with call_responses wired to POST through
/admin/cell-call (router-bypassing, admin-token-gated). The test is
OBSERVATIONAL — it records a current conformance baseline per cell to
logs/contract_profiles/<cell>.json that P5 (shim retirement) consumes;
a cell that violates an invariant is information, not a test failure.

Scope choice: the builder is called with advertised_surfaces={RESPONSES}
and capability_fields=REQUIRED_CAPABILITY_FIELDS to ISOLATE the conformance
signal. The capability-field advertisement gap is already known
(substrate_contract.py:85-88) and would otherwise force
FILE_UPSTREAM_GAP and mask the conformance verdict. A combined profile that
sources real capability fields from the catalog is a P4-handoff concern.

Run:

  uv run pytest -m model_probe tests/model_capability/test_cases/test_run_conformance.py
  uv run pytest -m model_probe tests/model_capability/test_cases/test_run_conformance.py --probe-cell=NAME
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any

import httpx
import pytest

from callosum.capability.conformance import (
    build_contract_profile,
    save_contract_profile,
)
from callosum.substrate_contract import (
    REQUIRED_CAPABILITY_FIELDS,
    ConformanceInvariant,
    Surface,
    classify_contract,
)

_log = logging.getLogger(__name__)


@pytest.mark.model_probe
def test_run_conformance_per_cell(
    callosum_client: httpx.Client,
    cells_to_probe: list[str],
) -> None:
    """For every advertised local cell, build its contract profile and record
    the conformance baseline. Passes as long as the builder completes — the
    per-cell classify_contract verdict is logged, not asserted."""
    for cell in cells_to_probe:

        async def _call_responses(body: dict[str, Any], _cell: str = cell) -> dict[str, Any]:
            """Closure over the httpx client and the current cell. Posts to
            /admin/cell-call (which bypasses the router) so each probe lands
            on the intended cell deterministically."""
            r = await asyncio.to_thread(
                partial(
                    callosum_client.post,
                    "/admin/cell-call",
                    json={"model": _cell, "body": body},
                )
            )
            r.raise_for_status()
            outcome = r.json()
            if outcome.get("status") != "ok":
                raise RuntimeError(f"cell-call failed: {outcome.get('error')}")
            response = outcome.get("response")
            if not isinstance(response, dict):
                raise RuntimeError("cell-call returned no response body")
            return response

        profile = asyncio.run(
            build_contract_profile(
                cell_id=cell,
                # Isolate the conformance signal: assume the responses surface is
                # advertised and capability fields are complete so the verdict
                # reflects invariant pass/fail, not the known capability gap.
                advertised_surfaces=frozenset({Surface.RESPONSES}),
                capability_fields=frozenset(REQUIRED_CAPABILITY_FIELDS),
                call_responses=_call_responses,
            )
        )

        path = save_contract_profile(profile)
        verdict = classify_contract(profile, Surface.RESPONSES)
        conf = profile.conformance.get(Surface.RESPONSES)
        violated = ", ".join(inv.value for inv in conf.violations()) if conf else "n/a"
        _log.info(
            "conformance baseline %s -> %s (violations: %s) saved to %s",
            cell,
            verdict.action.value,
            violated,
            path,
        )
        # Sanity: the always-applicable invariants must never be left None on a
        # probed surface (the None-trap guard). If they are, the builder is
        # broken, not the substrate — that IS a test failure.
        if conf is not None:
            for inv in (
                ConformanceInvariant.USAGE_INPUT_TOKENS_PRESENT,
                ConformanceInvariant.OUTPUT_NEVER_EMPTY,
                ConformanceInvariant.FINISH_REASON_MAPPING,
            ):
                assert conf.invariant_results[inv] is not None, (
                    f"{cell}: always-applicable invariant {inv.value} was left None"
                )
