"""Substrate-owned compatibility contract for local model cells.

Defines the contract a runtime substrate (local LLM gateway responses-proxy,
LiteLLM, ollama, vLLM, model-a0e0 gateways) must satisfy so callosum can
**consume the advertised native surface** instead of hand-translating
chat<->responses. This is the P2 deliverable of the shim-reduction program
(`docs/adr/2026-07-28-substrate-compatibility-contract.md`).

The classifier is pure: it takes a cell's advertised surfaces + probed
conformance + capability fields and returns the action callosum should take.
It performs no I/O and imports nothing from the routing pipeline, so it is
safe to exercise from unit tests and from the P3 auto-dev rewrite's
quarantine/route-native/verify-fix actions.

The seven conformance invariants are encoded as ConformanceInvariant
members. A surface is contract-conformant iff every invariant passes. See
the ADR for the substrate-by-substrate grounding (local LLM gateway meets most
on the responses path; LiteLLM 1.82.6 meets them on /v1/responses).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "Surface",
    "ConformanceInvariant",
    "SurfaceConformance",
    "CellContractProfile",
    "ContractAction",
    "ContractClassification",
    "classify_contract",
]


class Surface(StrEnum):
    """A wire surface a cell may advertise."""

    CHAT = "chat"
    RESPONSES = "responses"


# The seven callosum-enforced invariants (P1 inventory section A4).
# Ordered to match the ADR; the order is not semantically load-bearing.
class ConformanceInvariant(StrEnum):
    USAGE_INPUT_TOKENS_PRESENT = "usage_input_tokens_present"
    OUTPUT_NEVER_EMPTY = "output_never_empty"
    ORDERING_REASONING_THEN_FUNCTION_CALL_THEN_MESSAGE = (
        "ordering_reasoning_then_function_call_then_message"
    )
    PARALLEL_TOOL_CALL_COLLAPSE = "parallel_tool_call_collapse"
    FINISH_REASON_MAPPING = "finish_reason_mapping"
    REASONING_ALIAS_UNIFICATION = "reasoning_alias_unification"
    TOOL_CALL_AT_SCALE_PROBE_THROUGH_SUBSTRATE = (
        "tool_call_at_scale_probe_through_substrate"
    )


@dataclass(frozen=True)
class SurfaceConformance:
    """Probed conformance of one advertised surface.

    invariant_results maps each conformance invariant to True (pass),
    False (violation), or None (not probed / not applicable). A surface
    is conformant iff every probed invariant is True and no invariant is
    False.
    """

    surface: Surface
    invariant_results: Mapping[ConformanceInvariant, bool | None]

    def is_conformant(self) -> bool:
        return all(v is True for v in self.invariant_results.values())

    def violations(self) -> tuple[ConformanceInvariant, ...]:
        return tuple(
            inv
            for inv, v in self.invariant_results.items()
            if v is False
        )


# Capability fields the substrate must advertise per model
# (ADR section 3). local LLM gateway emits context_window + quantization +
# supported_reasoning_levels today; tools/modalities/throughput/
# verified_chat_translation are gaps -> P4 handoff.
REQUIRED_CAPABILITY_FIELDS: frozenset[str] = frozenset(
    {
        "context_window",
        "quantization",
        "supported_reasoning_levels",
        "supports_tools",
        "modalities",
        "throughput",
        "api_surfaces",
        "verified_chat_translation",
    }
)


@dataclass(frozen=True)
class CellContractProfile:
    """A cell's contract profile: what surfaces it advertises, how each
    conforms, and which capability fields the substrate exposes.

    residual_translation is True for cells where no substrate owns the
    requested surface and callosum must keep its own translator
    (the inband_reasoning KEEP-with-release-condition path).
    """

    cell_id: str
    advertised_surfaces: frozenset[Surface]
    conformance: Mapping[Surface, SurfaceConformance] = field(default_factory=dict)
    capability_fields: frozenset[str] = field(default_factory=frozenset)
    residual_translation: bool = False

    def advertises(self, surface: Surface) -> bool:
        return surface in self.advertised_surfaces


class ContractAction(StrEnum):
    """What callosum does for a (cell, requested-surface) pair."""

    ROUTE_NATIVE = "route_native"
    FILE_UPSTREAM_GAP = "file_upstream_gap"
    QUARANTINE_CELL = "quarantine_cell"
    VERIFY_FIX = "verify_fix"
    RESIDUAL_TRANSLATE = "residual_translate"
    REMOVE_SHIM = "remove_shim"


@dataclass(frozen=True)
class ContractClassification:
    """The classifier's verdict for one (cell, requested-surface) pair."""

    action: ContractAction
    surface: Surface
    reason: str
    violated_invariants: tuple[ConformanceInvariant, ...] = ()
    missing_capability_fields: tuple[str, ...] = ()


def classify_contract(
    profile: CellContractProfile, requested: Surface
) -> ContractClassification:
    """Classify a cell against the substrate contract for requested.

    Decision order (ADR section 5):

    1. Surface not advertised and no residual translator -> quarantine_cell.
    2. Surface not advertised but a residual translator exists ->
       residual_translate (TEMPORARY-DEBT, close condition = substrate
       fronts the surface).
    3. Surface advertised but capability advertisement is incomplete ->
       file_upstream_gap (capability-matrix gap, P4 handoff).
    4. Surface advertised but >=1 invariant violated -> quarantine_cell
       with the violated invariants recorded; the upstream gap is filed
       separately.
    5. Surface advertised and conformant -> route_native.

    verify_fix / remove_shim are terminal transitions the P3 loop
    drives *after* a re-probe or a substrate-fronting change; they are not
    produced by this pure classifier from a single snapshot.
    """
    if not profile.advertises(requested):
        if profile.residual_translation:
            return ContractClassification(
                action=ContractAction.RESIDUAL_TRANSLATE,
                surface=requested,
                reason=(
                    "no substrate owns this surface for this cell; "
                    "callosum residual translator is TEMPORARY-DEBT "
                    "with close condition: substrate fronts the surface"
                ),
            )
        return ContractClassification(
            action=ContractAction.QUARANTINE_CELL,
            surface=requested,
            reason=f"surface {requested.value} not advertised and no residual translator",
        )

    missing = tuple(
        sorted(REQUIRED_CAPABILITY_FIELDS - profile.capability_fields)
    )
    if missing:
        return ContractClassification(
            action=ContractAction.FILE_UPSTREAM_GAP,
            surface=requested,
            reason=(
                "substrate capability advertisement incomplete; "
                "missing fields must be exposed in `local-llm capabilities --json`"
            ),
            missing_capability_fields=missing,
        )

    conf = profile.conformance.get(requested)
    if conf is None:
        return ContractClassification(
            action=ContractAction.QUARANTINE_CELL,
            surface=requested,
            reason=f"surface {requested.value} advertised but not probed for conformance",
        )
    violations = conf.violations()
    if violations:
        return ContractClassification(
            action=ContractAction.QUARANTINE_CELL,
            surface=requested,
            reason=(
                f"surface {requested.value} advertised but "
                f"{len(violations)} conformance invariant(s) violated"
            ),
            violated_invariants=violations,
        )

    return ContractClassification(
        action=ContractAction.ROUTE_NATIVE,
        surface=requested,
        reason=f"surface {requested.value} advertised and contract-conformant",
    )