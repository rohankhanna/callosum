"""Acceptance tests for the substrate compatibility contract classifier.

These encode the P2 contract classification rules
(`docs/adr/2026-07-28-substrate-compatibility-contract.md`): for a
(cell, requested-surface) pair, the classifier returns the action callosum
should take -- route_native / file_upstream_gap / quarantine_cell /
residual_translate. The classifier is pure; these tests pin its decision
order so the P3 auto-dev rewrite (which consumes these actions) and the P4
upstream handoff (which closes the gaps) have a stable contract to execute
against.
"""

from __future__ import annotations

from callosum.substrate_contract import (
    CellContractProfile,
    ContractAction,
    Surface,
    SurfaceConformance,
    classify_contract,
)
from callosum.substrate_contract import (
    ConformanceInvariant as Inv,
)

ALL_INVARIANTS_PASS = {inv: True for inv in Inv}


def _conf(surface: Surface, **overrides: bool) -> SurfaceConformance:
    results = dict(ALL_INVARIANTS_PASS)
    results.update(overrides)
    return SurfaceConformance(surface=surface, invariant_results=results)


def _profile(
    *,
    cell_id: str = "cell",
    advertised: frozenset[Surface] = frozenset({Surface.RESPONSES}),
    conformance: dict[Surface, SurfaceConformance] | None = None,
    capability_fields: frozenset[str] | None = None,
    residual: bool = False,
) -> CellContractProfile:
    from callosum.substrate_contract import REQUIRED_CAPABILITY_FIELDS

    return CellContractProfile(
        cell_id=cell_id,
        advertised_surfaces=advertised,
        conformance=conformance or {},
        capability_fields=capability_fields if capability_fields is not None else frozenset(REQUIRED_CAPABILITY_FIELDS),
        residual_translation=residual,
    )


def test_conformant_advertised_surface_routes_native():
    profile = _profile(
        advertised=frozenset({Surface.RESPONSES}),
        conformance={Surface.RESPONSES: _conf(Surface.RESPONSES)},
    )
    verdict = classify_contract(profile, Surface.RESPONSES)
    assert verdict.action is ContractAction.ROUTE_NATIVE


def test_missing_surface_without_residual_quarantines():
    profile = _profile(advertised=frozenset({Surface.CHAT}))
    verdict = classify_contract(profile, Surface.RESPONSES)
    assert verdict.action is ContractAction.QUARANTINE_CELL
    assert "not advertised" in verdict.reason


def test_missing_surface_with_residual_translates_as_temporary_debt():
    profile = _profile(
        advertised=frozenset({Surface.CHAT}),
        residual=True,
    )
    verdict = classify_contract(profile, Surface.RESPONSES)
    assert verdict.action is ContractAction.RESIDUAL_TRANSLATE
    assert "TEMPORARY-DEBT" in verdict.reason


def test_incomplete_capability_advertisement_files_upstream_gap():
    # local LLM gateway today: context_window + quantization +
    # supported_reasoning_levels + api_surfaces, but NOT supports_tools /
    # modalities / throughput / verified_chat_translation.
    profile = _profile(
        advertised=frozenset({Surface.RESPONSES}),
        conformance={Surface.RESPONSES: _conf(Surface.RESPONSES)},
        capability_fields=frozenset(
            {
                "context_window",
                "quantization",
                "supported_reasoning_levels",
                "api_surfaces",
            }
        ),
    )
    verdict = classify_contract(profile, Surface.RESPONSES)
    assert verdict.action is ContractAction.FILE_UPSTREAM_GAP
    assert "supports_tools" in verdict.missing_capability_fields
    assert "modalities" in verdict.missing_capability_fields
    assert "throughput" in verdict.missing_capability_fields
    assert "verified_chat_translation" in verdict.missing_capability_fields


def test_invariant_violation_quarantines_with_violations_recorded():
    profile = _profile(
        advertised=frozenset({Surface.RESPONSES}),
        conformance={
            Surface.RESPONSES: _conf(
                Surface.RESPONSES,
                **{Inv.USAGE_INPUT_TOKENS_PRESENT.value: False},
            )
        },
    )
    verdict = classify_contract(profile, Surface.RESPONSES)
    assert verdict.action is ContractAction.QUARANTINE_CELL
    assert Inv.USAGE_INPUT_TOKENS_PRESENT in verdict.violated_invariants


def test_advertised_but_unprobed_quarantines():
    profile = _profile(
        advertised=frozenset({Surface.RESPONSES}),
        conformance={},  # advertised but no conformance probe
    )
    verdict = classify_contract(profile, Surface.RESPONSES)
    assert verdict.action is ContractAction.QUARANTINE_CELL
    assert "not probed" in verdict.reason


def test_capability_gap_takes_precedence_over_conformance_check():
    # A surface may be conformant but still require an upstream gap filing
    # if the capability matrix is incomplete (ADR section 3).
    profile = _profile(
        advertised=frozenset({Surface.RESPONSES}),
        conformance={Surface.RESPONSES: _conf(Surface.RESPONSES)},
        capability_fields=frozenset({"context_window"}),
    )
    verdict = classify_contract(profile, Surface.RESPONSES)
    assert verdict.action is ContractAction.FILE_UPSTREAM_GAP


def test_chat_surface_routes_native_when_advertised_and_conformant():
    profile = _profile(
        advertised=frozenset({Surface.CHAT}),
        conformance={Surface.CHAT: _conf(Surface.CHAT)},
    )
    verdict = classify_contract(profile, Surface.CHAT)
    assert verdict.action is ContractAction.ROUTE_NATIVE


def test_dual_surface_cell_routes_native_on_each_advertised_surface():
    # gpt_oss_hf advertises both chat + responses.
    profile = _profile(
        advertised=frozenset({Surface.CHAT, Surface.RESPONSES}),
        conformance={
            Surface.CHAT: _conf(Surface.CHAT),
            Surface.RESPONSES: _conf(Surface.RESPONSES),
        },
    )
    assert classify_contract(profile, Surface.CHAT).action is ContractAction.ROUTE_NATIVE
    assert classify_contract(profile, Surface.RESPONSES).action is ContractAction.ROUTE_NATIVE


def test_surface_conformance_is_conformant_only_when_all_probed_pass():
    ok = _conf(Surface.RESPONSES)
    assert ok.is_conformant() is True
    assert ok.violations() == ()
    bad = _conf(Surface.RESPONSES, **{Inv.FINISH_REASON_MAPPING.value: False})
    assert bad.is_conformant() is False
    assert Inv.FINISH_REASON_MAPPING in bad.violations()
