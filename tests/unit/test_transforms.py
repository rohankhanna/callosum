"""Tests for the callosum.transforms substrate.

The substrate is the package the dev loop will eventually write into,
so its contract has to be precise:

  * Empty registry is a no-op (zero behavior change when shipped).
  * Registration order determines request-transform order.
  * Response transforms apply in reverse (middleware-style).
  * One buggy transform cannot 5xx the request.
  * applies_to(ctx) filters per-cell.
  * Duplicate names are a configuration error.
  * The TransformContext is frozen (transforms can't accidentally
    mutate each other's view).

These tests are what protect the dev loop's invariants — when it
ships an auto-generated transform, the substrate's guarantees are
what keep the rest of the pipeline safe.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from callosum.capability.profile import (
    CapabilityProfile,
    DimensionFinding,
)
from callosum.capability.weight_identity import WeightIdentity
from callosum.cell_grid import Cell
from callosum.transforms import (
    Transform,
    TransformContext,
    TransformRegistry,
    build_default_registry,
)
from callosum.transforms.protocol import TransformBase

# ---------- helpers --------------------------------------------------------


def _ctx(model: str = "test-cell") -> TransformContext:
    return TransformContext(
        cell=Cell(model=model, reasoning_effort="default"),
        weight_identity=None,
        capability_profile=None,
    )


class _RecordingTransform(TransformBase):
    """Test fixture: records every request/response body it sees so
    tests can assert the order in which transforms fired."""

    def __init__(
        self,
        name: str,
        *,
        applies: bool = True,
        request_suffix: str = "",
        response_suffix: str = "",
    ) -> None:
        self._name = name
        self._applies = applies
        self._req_suffix = request_suffix
        self._resp_suffix = response_suffix
        self.request_calls: list[dict[str, Any]] = []
        self.response_calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    def applies_to(self, ctx: TransformContext) -> bool:
        return self._applies

    def transform_request(self, body: dict[str, Any], ctx: TransformContext) -> dict[str, Any]:
        self.request_calls.append(dict(body))
        # Append a tag so we can verify ordering on the body.
        new = dict(body)
        new["tags"] = body.get("tags", "") + self._req_suffix
        return new

    def transform_response(self, body: dict[str, Any], ctx: TransformContext) -> dict[str, Any]:
        self.response_calls.append(dict(body))
        new = dict(body)
        new["tags"] = body.get("tags", "") + self._resp_suffix
        return new


# ---------- empty registry -------------------------------------------------


def test_empty_registry_is_request_noop() -> None:
    """Shipping the substrate with an empty registry must not change
    request bodies. This is the contract that lets Phase 1-style
    shipping introduce zero behavior change."""
    reg = TransformRegistry()
    body = {"model": "x", "messages": []}
    assert reg.apply_request(body, _ctx()) == body


def test_empty_registry_is_response_noop() -> None:
    """Same for responses."""
    reg = TransformRegistry()
    body = {"output": []}
    assert reg.apply_response(body, _ctx()) == body


def test_build_default_registry_starts_empty() -> None:
    """The default registry shipped at startup carries no transforms.
    If we ever change this, the change should be deliberate — tests
    catch the moment a default-registered transform shows up."""
    reg = build_default_registry()
    assert len(reg) == 0
    assert reg.names() == []


# ---------- registration ---------------------------------------------------


def test_registration_appends_in_order() -> None:
    reg = TransformRegistry()
    a = _RecordingTransform("a")
    b = _RecordingTransform("b")
    reg.register(a)
    reg.register(b)
    assert reg.names() == ["a", "b"]
    assert len(reg) == 2


def test_duplicate_name_raises() -> None:
    """Two transforms with the same name make the registry's behavior
    ambiguous — which one wins on a tie? Reject at registration time
    so the operator finds out at startup, not at 3am during traffic."""
    reg = TransformRegistry()
    reg.register(_RecordingTransform("a"))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_RecordingTransform("a"))


# ---------- application order ----------------------------------------------


def test_request_transforms_apply_in_registration_order() -> None:
    """Each registered transform tags the body with its suffix. The
    final body's tags should be the concatenation in registration
    order — proving the order is deterministic and matches what an
    operator would expect."""
    reg = TransformRegistry()
    reg.register(_RecordingTransform("first", request_suffix="A"))
    reg.register(_RecordingTransform("second", request_suffix="B"))
    reg.register(_RecordingTransform("third", request_suffix="C"))
    body = reg.apply_request({"tags": ""}, _ctx())
    assert body["tags"] == "ABC"


def test_response_transforms_apply_in_reverse_order() -> None:
    """Response transforms run in REVERSE registration order so a
    transform that wrapped on request gets to unwrap on response.
    Verify by tagging both sides and checking the response tag is
    the reverse of the request tag."""
    reg = TransformRegistry()
    reg.register(_RecordingTransform("outer", response_suffix="A"))
    reg.register(_RecordingTransform("middle", response_suffix="B"))
    reg.register(_RecordingTransform("inner", response_suffix="C"))
    resp = reg.apply_response({"tags": ""}, _ctx())
    # Outer registered first → applies last on response.
    assert resp["tags"] == "CBA"


# ---------- applies_to gating ----------------------------------------------


def test_non_applicable_transforms_are_skipped() -> None:
    """A transform whose applies_to returns False must not be called
    for transform_request or transform_response. This is what lets
    transforms target specific cells without affecting others."""
    applicable = _RecordingTransform("yes", applies=True)
    not_applicable = _RecordingTransform("no", applies=False)
    reg = TransformRegistry()
    reg.register(applicable)
    reg.register(not_applicable)
    reg.apply_request({"tags": ""}, _ctx())
    reg.apply_response({"tags": ""}, _ctx())
    assert len(applicable.request_calls) == 1
    assert len(applicable.response_calls) == 1
    assert len(not_applicable.request_calls) == 0
    assert len(not_applicable.response_calls) == 0


def test_applies_to_can_filter_by_cell_model() -> None:
    """Realistic applicability check: transform fires only for a
    specific cell name. The dev-loop-written transforms will use
    this pattern (or weight_identity-based grouping) to target."""

    class CellSpecific(TransformBase):
        @property
        def name(self) -> str:
            return "cell-specific"

        def applies_to(self, ctx: TransformContext) -> bool:
            return ctx.cell.model == "target-cell"

        def transform_request(self, body: dict[str, Any], ctx: TransformContext) -> dict[str, Any]:
            return {**body, "fired": True}

    reg = TransformRegistry()
    reg.register(CellSpecific())
    fired = reg.apply_request({}, _ctx(model="target-cell"))
    notfired = reg.apply_request({}, _ctx(model="other-cell"))
    assert fired.get("fired") is True
    assert "fired" not in notfired


# ---------- error isolation ------------------------------------------------


def test_transform_raising_in_applies_to_is_skipped() -> None:
    """A buggy applies_to that raises must not crash routing. The
    transform is treated as not applicable and other transforms
    continue to fire."""

    class Bad(TransformBase):
        @property
        def name(self) -> str:
            return "bad"

        def applies_to(self, ctx: TransformContext) -> bool:
            raise RuntimeError("simulated bug")

    good = _RecordingTransform("good")
    reg = TransformRegistry()
    reg.register(Bad())
    reg.register(good)
    reg.apply_request({"tags": ""}, _ctx())
    # The good transform still fired — the bad one didn't crash the
    # pipeline.
    assert len(good.request_calls) == 1


def test_transform_raising_in_transform_request_is_skipped() -> None:
    """A buggy transform_request that raises must not 5xx the call.
    The body the next transform sees is the body BEFORE the bad
    transform — the bad transform's contribution is dropped."""

    class Bad(TransformBase):
        @property
        def name(self) -> str:
            return "bad"

        def applies_to(self, ctx: TransformContext) -> bool:
            return True

        def transform_request(self, body: dict[str, Any], ctx: TransformContext) -> dict[str, Any]:
            raise RuntimeError("simulated bug")

    good_after = _RecordingTransform("good", request_suffix="G")
    reg = TransformRegistry()
    reg.register(Bad())
    reg.register(good_after)
    result = reg.apply_request({"tags": "X"}, _ctx())
    # The "good" transform received the body unchanged by the bad
    # one (tags="X"), and tagged it with G.
    assert result["tags"] == "XG"


def test_transform_raising_in_transform_response_is_skipped() -> None:
    """Same isolation property for the response side."""

    class Bad(TransformBase):
        @property
        def name(self) -> str:
            return "bad"

        def applies_to(self, ctx: TransformContext) -> bool:
            return True

        def transform_response(self, body: dict[str, Any], ctx: TransformContext) -> dict[str, Any]:
            raise RuntimeError("simulated bug")

    good_first = _RecordingTransform("good", response_suffix="G")
    reg = TransformRegistry()
    reg.register(good_first)
    reg.register(Bad())
    # Response order is reverse: Bad runs first (skipped), then good.
    result = reg.apply_response({"tags": "X"}, _ctx())
    assert result["tags"] == "XG"


# ---------- context immutability -------------------------------------------


def test_transform_context_is_frozen() -> None:
    """The dataclass is frozen=True. A transform that tries to mutate
    its context view raises FrozenInstanceError rather than silently
    affecting other transforms."""
    ctx = _ctx()
    with pytest.raises(FrozenInstanceError):
        ctx.cell = Cell(model="changed", reasoning_effort="default")  # type: ignore[misc]


def test_context_carries_weight_identity_and_profile() -> None:
    """Realistic context: the transform sees the cell's weight
    identity and capability profile, both of which the dev loop's
    auto-generated transforms will inspect to decide what to do."""
    profile = CapabilityProfile(model_id="x")
    profile.upsert(
        DimensionFinding(
            dimension="tool_call_at_scale",
            status="fail",
            summary="text-as-JSON at 80K chars",
        )
    )
    profile.weight_identity = WeightIdentity(
        source="model-a0d7",
        runtime="ollama",
    )
    ctx = TransformContext(
        cell=Cell(model="x", reasoning_effort="default"),
        weight_identity=profile.weight_identity,
        capability_profile=profile,
    )
    assert ctx.weight_identity is not None
    assert ctx.weight_identity.source == "model-a0d7"
    assert ctx.capability_profile is not None
    assert "tool_call_at_scale" in ctx.capability_profile.findings


# ---------- realistic finding-driven applicability -------------------------


def test_transform_targets_failing_at_scale_cells() -> None:
    """The pattern the dev loop will use: transform's applies_to
    inspects the capability profile for a specific failing dimension.
    This is what makes transforms self-describing about WHEN they
    fire — no operator has to maintain a list of cell names."""

    class AtScaleAdapter(TransformBase):
        @property
        def name(self) -> str:
            return "at-scale-adapter"

        def applies_to(self, ctx: TransformContext) -> bool:
            if ctx.capability_profile is None:
                return False
            f = ctx.capability_profile.findings.get("tool_call_at_scale")
            return f is not None and f.status == "fail"

    # Cell with a failing at-scale finding → applies.
    failing = CapabilityProfile(model_id="failing")
    failing.upsert(
        DimensionFinding(
            dimension="tool_call_at_scale",
            status="fail",
            summary="",
        )
    )
    ctx_failing = TransformContext(
        cell=Cell(model="failing", reasoning_effort="default"),
        weight_identity=None,
        capability_profile=failing,
    )
    # Cell with a passing at-scale finding → does not apply.
    passing = CapabilityProfile(model_id="passing")
    passing.upsert(
        DimensionFinding(
            dimension="tool_call_at_scale",
            status="pass",
            summary="",
        )
    )
    ctx_passing = TransformContext(
        cell=Cell(model="passing", reasoning_effort="default"),
        weight_identity=None,
        capability_profile=passing,
    )
    # Cell with no profile → does not apply (no info, no action).
    ctx_unprobed = TransformContext(
        cell=Cell(model="unprobed", reasoning_effort="default"),
        weight_identity=None,
        capability_profile=None,
    )

    t = AtScaleAdapter()
    assert t.applies_to(ctx_failing) is True
    assert t.applies_to(ctx_passing) is False
    assert t.applies_to(ctx_unprobed) is False


# ---------- protocol conformance -------------------------------------------


def test_transform_base_satisfies_protocol() -> None:
    """A TransformBase subclass should structurally satisfy the
    Transform protocol. If anyone changes the protocol signatures
    without updating the base, this test catches the drift."""

    class _MinimalSubclass(TransformBase):
        @property
        def name(self) -> str:
            return "minimal"

    instance: Transform = _MinimalSubclass()  # noqa: F841 — type check is the assertion
