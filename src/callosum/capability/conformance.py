"""Substrate-contract conformance probes + CellContractProfile builder.

This is the P5-prep producer for the dormant contract pipeline: it runs the
seven ConformanceInvariant checks against a cell's advertised responses
surface, assembles a SurfaceConformance per surface, and packages the
result into the CellContractProfile that
callosum.substrate_contract.classify_contract consumes. Until this module
landed the whole CellContractProfile / SurfaceConformance /
classify_contract pipeline was defined and unit-pinned but had no producer
(gap_triage.py keeps classify_contract dormant "until P4 lands real
conformance data"); P5-prep is that producer.

Purity / boundary: this module imports stdlib + callosum.substrate_contract
(already pure) + the existing pure response classifiers
capability/dimensions/_shape_utils.classify_response and
capability/dimensions/reasoning_channel.classify_reasoning_channel +
capability.request_shapes. It imports NOTHING from the routing pipeline.
File I/O (save_contract_profile / load_contract_profile) lives here
rather than in substrate_contract.py so that module stays I/O-free.

Live-routing safety: build_contract_profile is never called from the
routing path, the scheduler, or app.py — only from tests (stubbed or via
the admin-token-gated /admin/cell-call endpoint). It is dormant-until-
invoked, exactly like classify_contract today. Wiring it into the periodic
scheduler and wiring classify_contract / route_native into
_dispatch_route is a follow-on gated on the P4 substrate landing.

The seven checks are pure functions over a response dict returning True
(pass), False (violation), or None (not applicable / could not
determine). The None-vs-False distinction is load-bearing:
classify_contract quarantines only on an explicit False (it uses
SurfaceConformance.violations()); a surface whose invariants are all
None would *falsely* route_native. The builder therefore never leaves
an always-applicable invariant (usage / output-never-empty / finish-reason) as
None — on a transport error those become False (see
_ALWAYS_APPLICABLE and _run_probe).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from callosum.capability.dimensions._shape_utils import classify_response
from callosum.capability.dimensions.reasoning_channel import (
    classify_reasoning_channel,
)
from callosum.capability.request_shapes import (
    parallel_tool_call_probe_body,
    reasoning_channel_probe_body,
    tool_call_simple_body,
    tool_call_with_context_body,
)
from callosum.substrate_contract import (
    REQUIRED_CAPABILITY_FIELDS,
    CellContractProfile,
    ConformanceInvariant,
    Surface,
    SurfaceConformance,
)

_log = logging.getLogger(__name__)

# Only the responses surface is probed in P5-prep — it is the route_native
# target (callosum wants to consume the native responses surface, not
# hand-translate chat). A cell advertising chat only is left without a
# conformance entry, so classify_contract quarantines it via the
# "advertised but not probed" branch (the conservative correct behavior until
# a chat-shaped transport is wired in a follow-on).
PROBEABLE_SURFACES: frozenset[Surface] = frozenset({Surface.RESPONSES})

# Invariants that must get a concrete True/False whenever a surface is
# probed — never left None. They are checkable from any responses payload
# that came back at all. On a transport error the builder sets these to
# False so the surface can never be all-None (which would falsely
# route_native — see the module docstring).
_ALWAYS_APPLICABLE: frozenset[ConformanceInvariant] = frozenset(
    {
        ConformanceInvariant.USAGE_INPUT_TOKENS_PRESENT,
        ConformanceInvariant.OUTPUT_NEVER_EMPTY,
        ConformanceInvariant.FINISH_REASON_MAPPING,
    }
)

# 80K chars ≈ 22K tokens — matches the tool_call_at_scale dimension's
# _PROMPT_CHARS so the at-scale conformance probe stresses the same
# context-sensitive failure mode.
_AT_SCALE_PROMPT_CHARS = 80_000

# Recognized Responses status values (the finish-reason analog on the
# responses surface). Unknown/missing status is a conformance violation.
_RESPONSES_STATUSES = frozenset({"completed", "incomplete"})

CallResponsesFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


# --------------------------------------------------------------------
# Pure invariant checks. Each takes a response dict (or None) and returns
# True / False / None. They reuse the existing pure classifiers rather than
# re-walking the response, so a shape change in one place fixes both.
# --------------------------------------------------------------------


def check_usage_input_tokens_present(
    response: dict[str, Any] | None,
) -> bool | None:
    """Invariant 1: response.usage.input_tokens is present and a positive
    int. Always applicable when a response was obtained — a responses payload
    without it makes Codex hard-fail, so its absence is a hard violation, not
    unknown."""
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return False
    tokens = usage.get("input_tokens")
    return isinstance(tokens, int) and tokens > 0


def check_output_never_empty(response: dict[str, Any] | None) -> bool | None:
    """Invariant 2: response.output is a non-empty list. Always applicable
    when a response was obtained — an empty output is a hard violation."""
    if not isinstance(response, dict):
        return None
    output = response.get("output")
    return isinstance(output, list) and len(output) > 0


# Canonical ordering rank for invariant 3. An output item type not in this map
# (e.g. reasoning_encrypted) is ignored for ordering purposes.
_ORDERING_RANK: dict[str, int] = {
    "reasoning": 0,
    "function_call": 1,
    "message": 2,
}


def check_ordering_reasoning_then_function_call_then_message(
    response: dict[str, Any] | None,
) -> bool | None:
    """Invariant 3: output items appear in reasoning → function_call → message
    order. classify_response.output_item_types is the ordered list of item
    types (_shape_utils.py), so this just checks the present types are a
    non-decreasing subsequence of the canonical order. Absent types are
    skipped — a response with only a message vacuously passes. Returns None
    when there is no responses payload to inspect."""
    if not isinstance(response, dict):
        return None
    cls = classify_response(response)
    ranks = [
        _ORDERING_RANK[t]
        for t in cls.output_item_types
        if t in _ORDERING_RANK
    ]
    return ranks == sorted(ranks)


def check_parallel_tool_call_collapse(
    response: dict[str, Any] | None,
) -> bool | None:
    """Invariant 4: parallel tool calls collapse into one output (multiple
    function_call items), not split. Pass (True) when >=2 function_call
    items are present in one output[]. None when the model chose not to
    parallelize (a model choice is not a substrate violation — we asked for
    parallel, the model did one at a time). Never False: this is a
    positive-evidence invariant; a split-across-messages shape would show up
    as fewer function_call items here and is the model's call, not the
    substrate's."""
    if not isinstance(response, dict):
        return None
    cls = classify_response(response)
    if cls.function_calls_count >= 2:
        return True
    return None


def check_finish_reason_mapping(response: dict[str, Any] | None) -> bool | None:
    """Invariant 5: a recognized finish-reason analog is present. On the
    responses surface status is the finish-reason analog. completed is
    a pass; incomplete passes when incomplete_details.reason is a
    recognized non-empty value. Always applicable when a response was
    obtained."""
    if not isinstance(response, dict):
        return None
    status = response.get("status")
    if status not in _RESPONSES_STATUSES:
        return False
    if status == "completed":
        return True
    # status == "incomplete"
    details = response.get("incomplete_details")
    reason = details.get("reason") if isinstance(details, Mapping) else None
    return isinstance(reason, str) and bool(reason)


def check_reasoning_alias_unification(
    response: dict[str, Any] | None,
) -> bool | None:
    """Invariant 6: reasoning aliases are unified (the substrate lifts CoT into
    a clean reasoning item, not in-band message tags). Reuses
    classify_reasoning_channel. native → pass; inband_tags →
    violation (the substrate did not unify); none → N/A (non-thinking turn);
    unknown → N/A (inconclusive — fail safe, mirrors the dimension's own
    error mapping)."""
    if not isinstance(response, dict):
        return None
    cls = classify_reasoning_channel(response)
    if cls.channel == "native":
        return True
    if cls.channel == "inband_tags":
        return False
    return None  # "none" or "unknown"


def check_tool_call_at_scale_probe_through_substrate(
    response: dict[str, Any] | None,
) -> bool | None:
    """Invariant 7: structured tool calls still work at realistic context size
    through the substrate. Pass (True) iff a structured function_call is
    emitted AND there is no tool-call-shaped JSON leaked into message text —
    the exact condition the tool_call_at_scale dimension uses. False
    otherwise (no structured call at scale, or a text-JSON leak). Returns
    None only when there is no response to inspect; the builder skips this
    check entirely (leaving the invariant None) when the small tool-call
    probe did not pass, mirroring tool_call_at_scale.py:46-55."""
    if not isinstance(response, dict):
        return None
    cls = classify_response(response)
    return cls.has_structured_call and not cls.text_json_leak_examples


# Registry: invariant -> its check. The builder knows which probe body drives
# each check (the check itself only inspects the response).
CHECKS: Mapping[
    ConformanceInvariant, Callable[[dict[str, Any] | None], bool | None]
] = {
    ConformanceInvariant.USAGE_INPUT_TOKENS_PRESENT: check_usage_input_tokens_present,
    ConformanceInvariant.OUTPUT_NEVER_EMPTY: check_output_never_empty,
    ConformanceInvariant.ORDERING_REASONING_THEN_FUNCTION_CALL_THEN_MESSAGE: (
        check_ordering_reasoning_then_function_call_then_message
    ),
    ConformanceInvariant.PARALLEL_TOOL_CALL_COLLAPSE: check_parallel_tool_call_collapse,
    ConformanceInvariant.FINISH_REASON_MAPPING: check_finish_reason_mapping,
    ConformanceInvariant.REASONING_ALIAS_UNIFICATION: check_reasoning_alias_unification,
    ConformanceInvariant.TOOL_CALL_AT_SCALE_PROBE_THROUGH_SUBSTRATE: (
        check_tool_call_at_scale_probe_through_substrate
    ),
}


def _small_tool_call_passed(response: dict[str, Any] | None) -> bool:
    """Did the small tool-call probe produce a clean structured call (the
    at-scale gate signal)? Mirrors tool_call_at_scale.py:79's pass
    condition."""
    if not isinstance(response, dict):
        return False
    cls = classify_response(response)
    return cls.has_structured_call and not cls.text_json_leak_examples


async def _run_probe(
    call_responses: CallResponsesFn,
    body: dict[str, Any],
    cell_id: str,
    checks: tuple[tuple[ConformanceInvariant, Callable[[dict[str, Any] | None], bool | None]], ...],
    results: dict[ConformanceInvariant, bool | None],
) -> dict[str, Any] | None:
    """Issue one probe request and run its designated checks. On a transport
    error, always-applicable invariants (see _ALWAYS_APPLICABLE) become
    False (never None — the None-trap guard); the rest stay None.
    Returns the response (or None on transport error) so the caller can derive
    the at-scale gate signal."""
    body = dict(body)
    body["model"] = cell_id
    try:
        response = await call_responses(body)
    except Exception as exc:  # noqa: BLE001 — transport glitch, fail safe
        _log.debug(
            "conformance probe transport error for %s: %s: %s",
            cell_id,
            type(exc).__name__,
            exc,
        )
        for inv, _fn in checks:
            if inv in _ALWAYS_APPLICABLE:
                results[inv] = False
            # non-always-applicable invariants stay None (genuinely unknown)
        return None
    for inv, fn in checks:
        results[inv] = fn(response)
    return response if isinstance(response, dict) else None


async def build_contract_profile(
    *,
    cell_id: str,
    advertised_surfaces: frozenset[Surface],
    capability_fields: frozenset[str],
    call_responses: CallResponsesFn,
    residual_translation: bool = False,
) -> CellContractProfile:
    """Probe a cell's advertised surfaces and assemble its contract profile.

    For each probeable advertised surface (only Surface.RESPONSES in
    P5-prep) the builder issues four probe requests through call_responses
    — a small tool-call probe (drives invariants 1/2/5), a reasoning probe
    (3/6), a parallel tool-call probe (4), and an at-scale tool-call probe (7,
    short-circuited unless the small probe passed). It assembles a
    SurfaceConformance per probed surface and returns the
    CellContractProfile ready for classify_contract.

    Surfaces that are advertised but not probeable get no conformance entry —
    classify_contract then quarantines them via the "advertised but not
    probed" branch, which is the correct conservative behavior. The builder
    never emits an all-None SurfaceConformance (that would falsely
    route_native); always-applicable invariants become False on
    transport error.

    The caller supplies advertised_surfaces / capability_fields so the
    builder stays free of catalog coupling and unit-testable with stubs (see
    contract_inputs_from_catalog for the catalog adapter).
    """
    conformance: dict[Surface, SurfaceConformance] = {}
    for surface in advertised_surfaces & PROBEABLE_SURFACES:
        results: dict[ConformanceInvariant, bool | None] = {
            inv: None for inv in ConformanceInvariant
        }
        # Probe A — small tool-call: invariants 1, 2, 5 (+ the at-scale gate).
        resp_small = await _run_probe(
            call_responses,
            tool_call_simple_body(),
            cell_id,
            (
                (ConformanceInvariant.USAGE_INPUT_TOKENS_PRESENT, check_usage_input_tokens_present),
                (ConformanceInvariant.OUTPUT_NEVER_EMPTY, check_output_never_empty),
                (ConformanceInvariant.FINISH_REASON_MAPPING, check_finish_reason_mapping),
            ),
            results,
        )
        # Probe B — reasoning: invariants 3, 6.
        await _run_probe(
            call_responses,
            reasoning_channel_probe_body(),
            cell_id,
            (
                (
                    ConformanceInvariant.ORDERING_REASONING_THEN_FUNCTION_CALL_THEN_MESSAGE,
                    check_ordering_reasoning_then_function_call_then_message,
                ),
                (ConformanceInvariant.REASONING_ALIAS_UNIFICATION, check_reasoning_alias_unification),
            ),
            results,
        )
        # Probe C — parallel tool-call: invariant 4.
        await _run_probe(
            call_responses,
            parallel_tool_call_probe_body(),
            cell_id,
            (
                (ConformanceInvariant.PARALLEL_TOOL_CALL_COLLAPSE, check_parallel_tool_call_collapse),
            ),
            results,
        )
        # Probe D — at-scale tool-call: invariant 7, gated on the small probe.
        if _small_tool_call_passed(resp_small):
            await _run_probe(
                call_responses,
                tool_call_with_context_body(target_user_chars=_AT_SCALE_PROMPT_CHARS),
                cell_id,
                (
                    (
                        ConformanceInvariant.TOOL_CALL_AT_SCALE_PROBE_THROUGH_SUBSTRATE,
                        check_tool_call_at_scale_probe_through_substrate,
                    ),
                ),
                results,
            )
        # else: invariant 7 stays None (small probe failed — skip at-scale,
        # mirroring tool_call_at_scale.py:46-55; don't burn GPU on a cell that
        # can't tool-call at all).
        conformance[surface] = SurfaceConformance(
            surface=surface,
            invariant_results=results,
        )
    return CellContractProfile(
        cell_id=cell_id,
        advertised_surfaces=advertised_surfaces,
        conformance=conformance,
        capability_fields=capability_fields,
        residual_translation=residual_translation,
    )


# --------------------------------------------------------------------
# Catalog adapter — maps a local LLM gateway catalog entry + capability row to the
# (advertised_surfaces, capability_fields, residual_translation) tuple the
# builder consumes. Pure; used only by the live baseline test. Today several
# required capability fields are unadvertised (tools/modalities/throughput/
# verified_chat_translation — substrate_contract.py:85-88), so the resulting
# profile files an upstream gap via classify_contract (the expected P5-prep
# signal). The builder does NOT depend on this adapter.
# --------------------------------------------------------------------


def contract_inputs_from_catalog(
    entry: Any,
    cap_row: Any | None = None,
) -> tuple[frozenset[Surface], frozenset[str], bool]:
    """Map a catalog entry (with an api_surfaces attribute/iterable of
    lowercase surface names) and an optional capability cap_row to the
    builder's inputs. Unknown surface names are dropped. residual_translation
    defaults to False — a later adapter can set it from the transforms
    registry."""
    advertised: frozenset[Surface] = frozenset()
    surfaces = getattr(entry, "api_surfaces", None)
    if surfaces is None:
        try:
            surfaces = entry.get("api_surfaces") if isinstance(entry, Mapping) else None
        except AttributeError:
            surfaces = None
    if surfaces:
        for s in surfaces:
            try:
                advertised = advertised | {Surface(str(s))}
            except ValueError:
                continue
    # Capability fields the substrate is known to advertise today. The
    # required set is REQUIRED_CAPABILITY_FIELDS; anything missing here will
    # be reported by classify_contract as a FILE_UPSTREAM_GAP. Note
    # supports_tools/modalities are now defensively consumed (tier-1 hub row
    # + tier-2 ollama /api/show stopgap) when present, so they register here
    # the moment the hub emits them; the gap is the hub EMISSION, not the
    # callosum parse.
    present: set[str] = set()
    for obj in (entry, cap_row):
        if obj is None:
            continue
        attrs: Mapping[str, Any] = obj if isinstance(obj, Mapping) else {
            a: getattr(obj, a, None) for a in dir(obj) if not a.startswith("_")
        }
        for field_name in REQUIRED_CAPABILITY_FIELDS:
            if field_name in attrs and attrs[field_name] is not None:
                present.add(field_name)
    return advertised, frozenset(present), False


# --------------------------------------------------------------------
# Persistence — file I/O lives here, not in substrate_contract.py (purity).
# Mirrors capability/profile.py's atomic tmp+replace pattern.
# --------------------------------------------------------------------

DEFAULT_CONTRACT_DIR = (
    Path(__file__).resolve().parents[3] / "logs" / "contract_profiles"
)


def contract_profile_path(cell_id: str, base: Path | None = None) -> Path:
    """Map a cell_id to its on-disk contract-profile path. Slash-replaced like
    profile_path (capability/profile.py:214) so model IDs are valid
    filenames."""
    safe = cell_id.replace("/", "_")
    directory = base if base is not None else DEFAULT_CONTRACT_DIR
    return directory / f"{safe}.json"


def save_contract_profile(
    profile: CellContractProfile,
    *,
    base: Path | None = None,
) -> Path:
    """Atomically persist the contract profile to disk. Returns the written
    path. Mirrors save_profile (capability/profile.py:245)."""
    path = contract_profile_path(profile.cell_id, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(profile.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


def load_contract_profile(
    cell_id: str,
    *,
    base: Path | None = None,
) -> CellContractProfile | None:
    """Load the contract profile for cell_id, or None when none exists
    or it is corrupt. Unlike load_profile (which returns a fresh empty
    profile) this returns None — a contract profile is only meaningful if
    it was built by probing, so "missing" means "not yet probed", not "empty
    but valid"."""
    path = contract_profile_path(cell_id, base)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return CellContractProfile.from_dict(data)
    except (KeyError, ValueError):
        return None