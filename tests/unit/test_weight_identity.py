"""Tests for callosum.capability.weight_identity.

Two themes:

  * Concrete providers (`LocalLlmCliWeightIdentityProvider`,
    `HeuristicWeightIdentityProvider`, `NullWeightIdentityProvider`)
    each return the right answer for their input shape and degrade
    gracefully on bad input.

  * The `CompositeWeightIdentityProvider` glues them together: first
    non-None wins, disagreements are logged but don't crash, an empty
    composite is legal but warns, and a provider that raises doesn't
    break the rest.

The composite is the contract callers depend on. The concrete
providers are individually replaceable as long as the composite still
returns a sensible value.
"""

from __future__ import annotations

import json
import logging
import subprocess
from unittest.mock import patch

from callosum.capability.weight_identity import (
    CompositeWeightIdentityProvider,
    HeuristicWeightIdentityProvider,
    LocalLlmCliWeightIdentityProvider,
    NullWeightIdentityProvider,
    WeightIdentity,
    build_default_provider,
)

# ---------- LocalLlmCliWeightIdentityProvider ------------------------------


def _fake_cli_payload() -> str:
    """Realistic local-llm `models local --json` output for two cells
    that share weights (one ollama, one responses-proxy over ollama)
    plus an unrelated vllm cell."""
    return json.dumps(
        {
            "action": "models",
            "contract_version": 1,
            "entries": [
                {
                    "model": {
                        "id": "test-26b-ollama",
                        "source": "test",
                        "runtime": "ollama",
                        "quantization": "ollama-q4_k_m",
                        "family": "test",
                    }
                },
                {
                    "model": {
                        "id": "test-26b-ollama-responses-proxy",
                        "source": "test",
                        "runtime": "responses_proxy",
                        "quantization": "ollama-q4_k_m",
                        "family": "test",
                    }
                },
                {
                    "model": {
                        "id": "model-a0a8-responses-proxy-q4_k_m",
                        "source": "/some/path/test-model-q4_k_m.gguf",
                        "runtime": "responses_proxy",
                        "quantization": "q4_k_m",
                        "family": "model-a0b7",
                    }
                },
            ],
        }
    )


def test_cli_provider_returns_identity_for_known_model() -> None:
    """Happy path: the CLI returns JSON, the provider parses it, and
    the cell's identity matches the parsed entry."""
    p = LocalLlmCliWeightIdentityProvider()
    with patch.object(subprocess, "run") as run:
        run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=_fake_cli_payload(), stderr="")
        identity = p.identify("test-26b-ollama")
    assert identity == WeightIdentity(
        source="test",
        runtime="ollama",
        quantization="ollama-q4_k_m",
        family="test",
    )


def test_cli_provider_groups_cells_sharing_weights() -> None:
    """The whole reason the abstraction exists: two cells with
    different transports but the same underlying weights must report
    matching `source` fields. Without this, downstream consumers
    can't tell `weights match, transport differs`."""
    p = LocalLlmCliWeightIdentityProvider()
    with patch.object(subprocess, "run") as run:
        run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=_fake_cli_payload(), stderr="")
        a = p.identify("test-26b-ollama")
        b = p.identify("test-26b-ollama-responses-proxy")
    assert a is not None and b is not None
    assert a.source == b.source == "test"
    assert a.runtime == "ollama"
    assert b.runtime == "responses_proxy"


def test_cli_provider_returns_none_for_unknown_model() -> None:
    """Unknown model is not an error — it's a cooperative None that
    lets the composite fall through to the next provider."""
    p = LocalLlmCliWeightIdentityProvider()
    with patch.object(subprocess, "run") as run:
        run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=_fake_cli_payload(), stderr="")
        assert p.identify("not-in-catalog") is None


def test_cli_provider_handles_missing_binary() -> None:
    """A FileNotFoundError on the subprocess call (CLI not installed)
    must degrade to None, not crash routing. In production this is
    the case that hands off to the heuristic fallback in the
    composite."""
    p = LocalLlmCliWeightIdentityProvider(binary="nonexistent-binary")
    with patch.object(subprocess, "run", side_effect=FileNotFoundError):
        assert p.identify("anything") is None


def test_cli_provider_handles_nonzero_exit() -> None:
    """CLI returning non-zero is treated as no-knowledge, not as an
    exception. Matches the cooperative-None pattern."""
    p = LocalLlmCliWeightIdentityProvider()
    with patch.object(subprocess, "run") as run:
        run.return_value = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="catalog corrupt")
        assert p.identify("anything") is None


def test_cli_provider_handles_malformed_json() -> None:
    """If the CLI's output ever drifts from the expected schema, the
    provider must not raise. Stamp the cache time so we don't retry
    every call, log the failure, and return None."""
    p = LocalLlmCliWeightIdentityProvider()
    with patch.object(subprocess, "run") as run:
        run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="{ not json", stderr="")
        assert p.identify("anything") is None


def test_cli_provider_caches_catalog_within_ttl() -> None:
    """One CLI call should serve many identify() lookups within the
    TTL window. The shell-out is the expensive part; caching makes
    the provider cheap enough to call on every routing decision."""
    p = LocalLlmCliWeightIdentityProvider(cache_ttl_s=600)
    with patch.object(subprocess, "run") as run:
        run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout=_fake_cli_payload(), stderr="")
        for _ in range(50):
            p.identify("test-26b-ollama")
        assert run.call_count == 1


# ---------- HeuristicWeightIdentityProvider --------------------------------


def test_heuristic_strips_responses_proxy_suffix() -> None:
    """The longest-suffix-first ordering matters: `-ollama-responses-
    proxy` must match before `-ollama` so we don't lose the proxy info.
    The user's specific case (model-a0d5 cells differing only by proxy vs.
    direct) is the canonical example."""
    p = HeuristicWeightIdentityProvider()
    id_proxy = p.identify("test-26b-ollama-responses-proxy")
    id_direct = p.identify("test-26b-ollama")
    assert id_proxy is not None and id_direct is not None
    assert id_proxy.source == id_direct.source == "test-26b"
    assert id_proxy.runtime == "ollama_responses_proxy"
    assert id_direct.runtime == "ollama"


def test_heuristic_returns_none_when_no_suffix_matches() -> None:
    """A bare model_id with no recognized transport suffix can't be
    derived from naming alone — return None so the composite falls
    through (rather than asserting an identity we don't have)."""
    p = HeuristicWeightIdentityProvider()
    assert p.identify("totally-bare-model") is None


def test_heuristic_derives_family_label() -> None:
    """The family label drops the size tag. This is informational
    grouping for operators reading profiles — `model-a0c7` and
    `model-a0c8` both report family=`model-a0e5` so they show up
    in the same family bucket."""
    p = HeuristicWeightIdentityProvider()
    ident = p.identify("test-26b-ollama")
    assert ident is not None
    assert ident.family == "test"


# ---------- NullWeightIdentityProvider -------------------------------------


def test_null_provider_returns_none() -> None:
    """Sanity. Null exists so a composite always has a final entry
    that's guaranteed to fall through cleanly."""
    p = NullWeightIdentityProvider()
    assert p.identify("anything") is None


# ---------- Composite ------------------------------------------------------


def test_composite_falls_through_to_next_on_none() -> None:
    """First provider returns None; composite must consult the
    second. This is the central abstraction's value: as long as ONE
    provider knows, the lookup succeeds."""
    null_first = NullWeightIdentityProvider()
    heuristic_second = HeuristicWeightIdentityProvider()
    composite = CompositeWeightIdentityProvider(providers=[null_first, heuristic_second])
    ident = composite.identify("test-26b-ollama")
    assert ident is not None
    assert ident.source == "test-26b"


def test_composite_first_non_none_wins() -> None:
    """Priority is by list order. A precise provider placed first
    must beat a heuristic provider placed second, even when both
    have an answer."""

    class FakePrecise:
        id = "fake-precise"

        def identify(self, model_id):
            if model_id == "test-26b-ollama":
                return WeightIdentity(
                    source="precise-test",
                    runtime="ollama",
                )
            return None

    composite = CompositeWeightIdentityProvider(providers=[FakePrecise(), HeuristicWeightIdentityProvider()])
    ident = composite.identify("test-26b-ollama")
    assert ident is not None
    assert ident.source == "precise-test"


def test_composite_logs_disagreement_but_uses_first_answer(
    caplog,
) -> None:
    """When two providers both have an answer but they differ, the
    composite must keep using the first (priority) one AND log the
    disagreement so an operator notices the mis-configuration."""

    class FirstSays:
        id = "first"

        def identify(self, model_id):
            return WeightIdentity(source="first-source", runtime="ollama")

    class SecondSays:
        id = "second"

        def identify(self, model_id):
            return WeightIdentity(source="second-source", runtime="ollama")

    composite = CompositeWeightIdentityProvider(providers=[FirstSays(), SecondSays()])
    with caplog.at_level(logging.WARNING):
        ident = composite.identify("anything")
    assert ident is not None
    assert ident.source == "first-source"
    assert any("disagreement" in r.message for r in caplog.records)
    assert composite.stats()["disagreements"] == 1


def test_composite_survives_provider_that_raises() -> None:
    """If a provider raises (bug, runtime error, dependency missing),
    the composite must continue with the rest. One bad concretion
    cannot take down the whole abstraction."""

    class BrokenProvider:
        id = "broken"

        def identify(self, model_id):
            raise RuntimeError("simulated provider failure")

    composite = CompositeWeightIdentityProvider(providers=[BrokenProvider(), HeuristicWeightIdentityProvider()])
    ident = composite.identify("test-26b-ollama")
    assert ident is not None
    assert ident.source == "test-26b"


def test_composite_with_empty_providers_returns_none(caplog) -> None:
    """An empty composite is legal (so constructing one mid-init
    doesn't need a None-check) but a WARNING tells the operator the
    wire-up is missing."""
    with caplog.at_level(logging.WARNING):
        composite = CompositeWeightIdentityProvider(providers=[])
    assert composite.identify("anything") is None
    assert any("constructed with NO providers" in r.message for r in caplog.records)


def test_composite_stats_are_tracked() -> None:
    """The composite tracks per-provider hits, miss counts, and
    disagreement counts. The stats() dict is what a `/status`-style
    endpoint would expose, so its shape is part of the contract."""

    class ProviderA:
        id = "A"

        def identify(self, model_id):
            return WeightIdentity(source="x", runtime="r") if model_id == "known" else None

    composite = CompositeWeightIdentityProvider(providers=[ProviderA(), NullWeightIdentityProvider()])
    composite.identify("known")
    composite.identify("known")
    composite.identify("unknown")
    stats = composite.stats()
    assert stats["hits_per_provider"] == {"A": 2}
    assert stats["misses"] == 1
    assert stats["disagreements"] == 0


def test_build_default_provider_orders_concretions() -> None:
    """The default composite's order matters: CLI first (precise),
    heuristic second (fallback), null third (backstop). Verify by
    inspecting the internal list — this is the wiring contract that
    callers rely on."""
    composite = build_default_provider()
    types = [type(p).__name__ for p in composite._providers]
    assert types == [
        "LocalLlmCliWeightIdentityProvider",
        "HeuristicWeightIdentityProvider",
        "NullWeightIdentityProvider",
    ]
