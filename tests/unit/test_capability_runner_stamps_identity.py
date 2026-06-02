"""Verify run_dimensions stamps weight_identity on the profile.

The runner takes an optional `weight_identity_provider`. When supplied,
the profile saved to disk after the first dimension probe must carry
the identity returned by the provider. This is the contract that lets
downstream consumers (research_runner, operator dashboards, future routing
gates) detect "two cells share weights — divergent findings = transport
bug, not model bug."

Also verifies the asymmetry: a provider returning None must NOT clobber
an identity that's already in the profile, because a transient
provider outage shouldn't silently regress grouping info.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from callosum.capability import runner
from callosum.capability.profile import (
    CapabilityProfile,
    DimensionFinding,
    load_profile,
    save_profile,
)
from callosum.capability.weight_identity import (
    NullWeightIdentityProvider,
    WeightIdentity,
)


@pytest.fixture(autouse=True)
def _redirect_profile_dir(monkeypatch, tmp_path: Path):
    """Redirect DEFAULT_PROFILE_DIR to a tmpdir so the runner's loads
    AND the test's direct load_profile calls both resolve to the same
    isolated location. Without this, the test could read a production
    profile written by a real running callosum instance and assertions
    would silently see stale field values."""
    monkeypatch.setattr(
        "callosum.capability.profile.DEFAULT_PROFILE_DIR", tmp_path,
    )
    yield


class _StubProvider:
    """Minimal provider that returns a fixed identity for one model and
    None for everything else. Lets tests assert "the runner asked us
    and stamped what we said" without coupling to any concrete
    provider's behavior."""

    id = "stub"

    def __init__(self, mapping: dict[str, WeightIdentity]) -> None:
        self._mapping = mapping

    def identify(self, model_id: str) -> WeightIdentity | None:
        return self._mapping.get(model_id)


def _make_call_responses(response: dict):
    """Build a no-op call_responses closure that ignores the body and
    returns a fixed response. The runner needs ONE successful probe to
    save the profile to disk; the dimension content doesn't matter for
    these tests."""

    async def _call(body: dict) -> dict:
        return response

    return _call


def test_runner_stamps_identity_from_provider() -> None:
    """The contract test: provider returns an identity, the saved
    profile carries it. Without this, the whole 'callosum knows which
    cells share weights' story falls apart at the persistence layer."""
    identity = WeightIdentity(
        source="model-a0d6", runtime="ollama",
        quantization="ollama-q4_k_m", family="model-a0e5",
    )
    provider = _StubProvider({"model-a0a9": identity})

    # Use a non-empty response so at least one dimension produces a
    # result and triggers profile.save().
    fake_response = {
        "output": [
            {"type": "function_call", "name": "exec_command",
             "call_id": "1", "arguments": "{}"}
        ]
    }
    asyncio.run(
        runner.run_dimensions(
            cell="model-a0a9",
            backend_id="test-backend",
            call_responses=_make_call_responses(fake_response),
            weight_identity_provider=provider,
        )
    )
    profile = load_profile("model-a0a9")
    assert profile.weight_identity == identity


def test_runner_persists_identity_in_json_roundtrip() -> None:
    """The identity must survive a JSON round-trip — that's what makes
    research_runner (a separate process) able to consume it. If from_dict
    drops the field, our pipeline is broken regardless of how well the
    in-memory stamping works."""
    identity = WeightIdentity(
        source="model-a0d7", runtime="responses_proxy",
        quantization="ollama-q4_k_m", family="model-a0e5",
    )
    provider = _StubProvider({"model-a0a3": identity})
    fake_response = {
        "output": [
            {"type": "function_call", "name": "exec_command",
             "call_id": "1", "arguments": "{}"}
        ]
    }
    asyncio.run(
        runner.run_dimensions(
            cell="model-a0a3",
            backend_id="test-backend",
            call_responses=_make_call_responses(fake_response),
            weight_identity_provider=provider,
        )
    )
    # Re-load from disk — this exercises from_dict, not the in-memory
    # object we just stamped.
    fresh = load_profile("model-a0a3")
    assert fresh.weight_identity == identity


def test_runner_does_not_clobber_existing_identity_on_none() -> None:
    """A provider that returns None (transient CLI outage, model not
    yet in catalog) must NOT erase a previously-persisted identity.
    Otherwise a flaky provider would silently degrade the system: the
    profile would lose its weight_identity field across sweeps."""
    # Seed the profile with a known identity.
    profile = CapabilityProfile(model_id="seeded-cell")
    profile.weight_identity = WeightIdentity(
        source="known-source", runtime="ollama",
    )
    profile.upsert(
        DimensionFinding(
            dimension="tool_call_shape",
            status="pass",
            summary="seeded for test",
        )
    )
    save_profile(profile)

    # Now run with a provider that knows nothing about this cell.
    fake_response = {
        "output": [
            {"type": "function_call", "name": "exec_command",
             "call_id": "1", "arguments": "{}"}
        ]
    }
    asyncio.run(
        runner.run_dimensions(
            cell="seeded-cell",
            backend_id="test-backend",
            call_responses=_make_call_responses(fake_response),
            ttl_s=0,  # force re-probe so the dimension loop actually runs
            weight_identity_provider=NullWeightIdentityProvider(),
        )
    )
    after = load_profile("seeded-cell")
    # Identity preserved.
    assert after.weight_identity is not None
    assert after.weight_identity.source == "known-source"


def test_runner_omits_identity_when_no_provider() -> None:
    """When the caller passes no provider, the profile's identity stays
    whatever it was (initially None for a fresh cell). Verifies the
    runner doesn't construct a default provider on its own — the
    caller's choice is respected."""
    fake_response = {
        "output": [
            {"type": "function_call", "name": "exec_command",
             "call_id": "1", "arguments": "{}"}
        ]
    }
    asyncio.run(
        runner.run_dimensions(
            cell="unprovided-cell",
            backend_id="test-backend",
            call_responses=_make_call_responses(fake_response),
        )
    )
    profile = load_profile("unprovided-cell")
    assert profile.weight_identity is None
