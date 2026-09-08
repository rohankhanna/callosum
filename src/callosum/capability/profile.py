"""CapabilityProfile: per-model, per-dimension findings.

The capability-test harness runs codex-shape requests against each
candidate cell through callosum and records a structured profile of
how the cell behaves. The profile is the artifact future adapter-
design work (or admission decisions in ) consume —
"does this model emit structured tool_calls under realistic context?"
isn't a yes/no question, it's a multi-dimensional fingerprint of where
the model is competent and where it isn't.

Profiles persist to JSON at logs/capability_profiles/<model_id>.json.
Each test case overwrites its own dimension on the profile, leaving
other dimensions untouched. The file is a flat structured dict that
both humans (when reviewing) and other agents (when synthesizing
adapter recommendations) can consume directly.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from callosum.capability.weight_identity import WeightIdentity
from callosum.substrate_contract import ContractAction

# The repo root's logs directory. Profiles live alongside the prior-
# art gate logs since both are durable operator state about what the
# system has learned about models. Path resolution walks up from this
# file (src/callosum/capability/profile.py) to repo root, then down
# to logs/capability_profiles/.
DEFAULT_PROFILE_DIR = Path(__file__).resolve().parents[3] / "logs" / "capability_profiles"


FindingStatus = Literal["pass", "fail", "error", "skipped"]


@dataclass
class DimensionFinding:
    """One dimension of capability evaluation for one model.

    `status`:
      * "pass"    — the cell exhibits the behavior this dimension tests
      * "fail"    — the cell does not exhibit the behavior (the
                    informative case; this is where adapter hints come
                    from)
      * "error"   — could not complete the test (network, timeout, etc)
      * "skipped" — test was not run for this cell (e.g. not applicable)

    `evidence` carries raw signal — sample response excerpts, latency
    bounds, observed token counts — so future readers can audit the
    finding without re-running the probe.

    `adapter_hint` is the most important field: when `status == "fail"`,
    it describes in concrete terms what an adapter would need to do
    to use this model reliably. This is what 
    admission flow (or a human writing an adapter) consumes.

    `gap_class` / `suggested_action` / `upstream_owner` / `close_condition`
    are the **structured gap report** the P3 dev-loop consumes (ADR
    `docs/adr/2026-07-28-substrate-compatibility-contract.md` section 5).
    A failing dimension populates them deterministically so the dev-loop
    classifies the gap into the section-5 priority order instead of
    automation agently authoring a transform off the free-prose `adapter_hint`.
    `gap_class` is "A" (generic protocol the substrate owns) or "B"
    (model-specific quirk no substrate owns); `suggested_action` is the
    `ContractAction` the loop should take; `upstream_owner` +
    `close_condition` are the TEMPORARY-DEBT metadata on a Class-B
    adapter. All four default to None; older profiles and `pass`/`error`/
    `skipped` findings leave them unset.
    """

    dimension: str
    status: FindingStatus
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    adapter_hint: str | None = None
    latency_ms: int | None = None
    # Structured gap report (ADR section 5). Optional + additive; old
    # profiles without these fields load with None defaults.
    gap_class: Literal["A", "B"] | None = None
    suggested_action: ContractAction | None = None
    upstream_owner: str | None = None
    close_condition: str | None = None


@dataclass
class CapabilityProfile:
    """All accumulated findings for one model. Persists as JSON.

    Profiles are append-only by dimension — running a single dimension
    test updates that dimension's finding and leaves the rest in place.
    `last_updated` tracks the most recent finding's timestamp so the
    consumer knows how fresh the profile is.

    `harness_version` lets future readers know which test-suite version
    produced the findings. Bump when test semantics meaningfully change
    so cached profiles can be regenerated.
    """

    model_id: str
    backend_id: str | None = None
    harness_version: int = 1
    last_updated: float = field(default_factory=lambda: time.time())
    findings: dict[str, DimensionFinding] = field(default_factory=dict)
    # Stable identity for the underlying weights this cell serves.
    # Populated by a `WeightIdentityProvider` at harness time. None
    # for cells whose weight identity is unknowable from any
    # configured provider — older profiles persisted before the
    # weight-identity feature landed also load as None and remain
    # routable until the next harness sweep stamps them.
    weight_identity: WeightIdentity | None = None

    def upsert(self, finding: DimensionFinding) -> None:
        """Write a finding for one dimension, overwriting any previous
        result for the same dimension. Bumps `last_updated`."""
        self.findings[finding.dimension] = finding
        self.last_updated = time.time()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapabilityProfile:
        # Findings stored as nested dicts; rehydrate into dataclass
        # instances. Defensive: skip malformed entries rather than
        # raising — partial profiles are still useful.
        raw_findings = data.get("findings", {})
        findings: dict[str, DimensionFinding] = {}
        if isinstance(raw_findings, dict):
            for dim, payload in raw_findings.items():
                if not isinstance(payload, dict):
                    continue
                try:
                    findings[dim] = DimensionFinding(
                        dimension=str(payload.get("dimension", dim)),
                        status=payload.get("status", "error"),
                        summary=str(payload.get("summary", "")),
                        evidence=dict(payload.get("evidence", {})) if isinstance(payload.get("evidence"), dict) else {},
                        adapter_hint=payload.get("adapter_hint"),
                        latency_ms=payload.get("latency_ms"),
                        gap_class=_parse_gap_class(payload.get("gap_class")),
                        suggested_action=_parse_suggested_action(payload.get("suggested_action")),
                        upstream_owner=_parse_optional_str(payload.get("upstream_owner")),
                        close_condition=_parse_optional_str(payload.get("close_condition")),
                    )
                except (TypeError, ValueError):
                    continue
        # Weight identity is a nested dict; rehydrate or skip on
        # malformed input. Older profiles persisted before this field
        # existed will simply have no entry; new sweeps backfill it.
        weight_identity: WeightIdentity | None = None
        raw_wi = data.get("weight_identity")
        if isinstance(raw_wi, dict):
            try:
                source = raw_wi.get("source")
                runtime = raw_wi.get("runtime")
                if isinstance(source, str) and isinstance(runtime, str):
                    weight_identity = WeightIdentity(
                        source=source,
                        runtime=runtime,
                        quantization=raw_wi.get("quantization")
                        if isinstance(raw_wi.get("quantization"), str)
                        else None,
                        family=raw_wi.get("family") if isinstance(raw_wi.get("family"), str) else None,
                    )
            except (TypeError, ValueError):
                weight_identity = None
        return cls(
            model_id=str(data.get("model_id", "")),
            backend_id=data.get("backend_id"),
            harness_version=int(data.get("harness_version", 1)),
            last_updated=float(data.get("last_updated", time.time())),
            findings=findings,
            weight_identity=weight_identity,
        )


def _parse_optional_str(value: Any) -> str | None:
    """Coerce a JSON value to str | None for the optional gap fields.
    Defensive: anything non-str becomes None so a corrupt field never
    breaks profile loading."""
    return value if isinstance(value, str) and value else None


def _parse_gap_class(value: Any) -> Literal["A", "B"] | None:
    """Coerce the persisted gap_class to its Literal or None. Unknown
    values (including old profiles without the field) load as None."""
    return value if value in ("A", "B") else None


def _parse_suggested_action(value: Any) -> ContractAction | None:
    """Coerce the persisted suggested_action string to a ContractAction.
    Unknown/old values load as None rather than raising — the gap
    classifier falls back to inferring from adapter_hint when this is
    absent (the P3 backward-compat seam)."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return ContractAction(value)
    except ValueError:
        return None


def profile_path(model_id: str, profile_dir: Path | None = None) -> Path:
    """Map a model_id to its on-disk profile path. Model IDs are
    treated as opaque strings but we slash-replace to keep them
    valid filenames on POSIX. Other special chars are left alone —
    we control the input space."""
    safe = model_id.replace("/", "_")
    base = profile_dir if profile_dir is not None else DEFAULT_PROFILE_DIR
    return base / f"{safe}.json"


def load_profile(model_id: str, *, profile_dir: Path | None = None) -> CapabilityProfile:
    """Load the profile for `model_id` from disk, or return a fresh
    empty profile when none exists yet. Never raises on missing files
    — first-probe-ever is the normal case."""
    path = profile_path(model_id, profile_dir)
    if not path.exists():
        return CapabilityProfile(model_id=model_id)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        # Corrupt file → start fresh. The next persist replaces it.
        return CapabilityProfile(model_id=model_id)
    if not isinstance(data, dict):
        return CapabilityProfile(model_id=model_id)
    return CapabilityProfile.from_dict(data)


def save_profile(profile: CapabilityProfile, *, profile_dir: Path | None = None) -> Path:
    """Atomically persist the profile to disk. Returns the written path."""
    path = profile_path(profile.model_id, profile_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True))
    tmp.replace(path)
    return path
