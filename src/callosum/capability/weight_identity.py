"""Weight-identity abstraction for cells.

A callosum "cell" is a (transport + model) pair, not a model alone.
Two cells can route to identical model weights through different
transports — e.g. `model-a0a9` (direct ollama Chat
Completions) and `model-a0a1` (Codex
Responses API → translation layer → ollama). When their capability
findings diverge, the divergence is information about the *transport*,
not the model. We need a way to say "these cells share weights."

This module is the abstraction layer for that lookup. It deliberately
depends only on the Protocol below; concrete sources of weight
identity are plug-replaceable.

Design principles:

  * The router and the harness depend on `WeightIdentityProvider`,
    NOT on any concrete provider. Adding a new source (a different
    CLI, a manifest file, an HTTP endpoint, a static map) is one new
    class plus one constructor-argument change at the wiring site.

  * Multiple concrete providers can be active at once via
    `CompositeWeightIdentityProvider`. The composite queries them in
    priority order and falls through to the next when an earlier one
    returns None. This is the "diversification-based redundancy"
    pattern: as long as ONE provider knows about a model, the lookup
    succeeds.

  * Providers cooperate by returning None for unknown models rather
    than raising — `None` means "I don't have an answer," distinct
    from a value that asserts "no weight family exists." The
    composite uses None as the fall-through signal.

  * Providers should be cheap-to-call. Concrete providers cache
    internally when their underlying source is expensive (shelling
    out to a CLI, an HTTP request). The protocol does not specify
    caching policy; that's a provider-implementation detail.

Current concrete providers:

  * `LocalLlmCliWeightIdentityProvider` — shells out to `local-llm
    models local --json` (the unified front-end CLI over ollama /
    vllm / responses-proxy / etc.). Canonical when available.
  * `HeuristicWeightIdentityProvider` — naming-pattern fallback when
    the CLI is unavailable or the model isn't registered with it.
    Strips known transport suffixes (`-ollama`, `-responses-proxy`,
    `-vllm`, etc.) to derive a weight-family stem.
  * `NullWeightIdentityProvider` — always returns None. For tests
    and for environments where weight identity is genuinely unknowable.

Add concrete providers liberally. The composite tolerates as many as
you give it.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WeightIdentity:
    """Stable identity for the underlying weights a cell serves.

    Two cells whose `source` is equal share weights regardless of
    transport. `source` is the load-bearing field for equality; the
    other fields are informational and may be missing depending on
    which concrete provider produced the identity.

    `runtime` describes the transport layer the cell uses
    (e.g. "ollama", "vllm", "responses_proxy"). Two cells with the
    same `source` but different `runtime` are the "same weights,
    different pipeline" case — the case the abstraction exists to
    expose.

    `quantization` and `family` are informational. `family` is a
    human-readable group label like "model-a0e5" or "model-a0b7";
    `source` is the precise identifier (an ollama tag like
    "model-a0d6", or a filesystem path to a GGUF, or a HuggingFace
    model id).
    """

    source: str
    runtime: str
    quantization: str | None = None
    family: str | None = None


class WeightIdentityProvider(Protocol):
    """Maps a callosum cell's model_id to its WeightIdentity.

    Implementations should:
      * Return None when they don't recognize the model (NOT raise).
      * Be cheap-to-call; cache internally if the underlying source
        is expensive.
      * Be safe to call from multiple threads concurrently.
    """

    @property
    def id(self) -> str:
        """Stable identifier for provenance logging."""
        ...

    def identify(self, model_id: str) -> WeightIdentity | None:
        """Return the weight identity for `model_id`, or None if
        unknown to this provider."""
        ...


# ---------- concretion 1: local-llm CLI -----------------------------------


# Known transport-suffix patterns. The CLI's `id` field has the full
# cell identifier; we also accept the bare model_id for tools that
# strip suffixes before consulting the provider.
_TRANSPORT_SUFFIXES = (
    "-ollama-responses-proxy",
    "-ollama",
    "-responses-proxy",
    "-vllm",
    "-q4_k_m",
    "-q5_k_m",
    "-q8_0",
    "-fp16",
    "-bf16",
)


class LocalLlmCliWeightIdentityProvider:
    """Queries `local-llm models local --json` for weight identity.

    The local LLM gateway CLI is the unified front-end over multiple local
    backends (ollama, vllm, responses-proxy, etc.) — it has one
    catalog with structured `model.source`, `model.runtime`,
    `model.family`, and `model.quantization` per entry. This is the
    canonical source for weight identity when available.

    Caches the parsed catalog for `cache_ttl_s` seconds (default 5
    minutes). Subsequent calls within the TTL window do not shell out;
    after the TTL expires, the next call re-runs the CLI. The TTL is
    short relative to the harness's 6h cadence, so a model adaptive-
    research_runner adds will appear in the cache within minutes.

    Failure modes (CLI missing, CLI exits non-zero, output not valid
    JSON, model not in catalog) all map to `identify(...) → None`.
    Composing this provider with a `HeuristicWeightIdentityProvider`
    behind it ensures degraded environments still get an answer.
    """

    id = "local-llm-cli"

    def __init__(
        self,
        *,
        binary: str = "local-llm",
        cache_ttl_s: float = 300.0,
        timeout_s: float = 15.0,
    ) -> None:
        self._binary = binary
        self._cache_ttl_s = cache_ttl_s
        self._timeout_s = timeout_s
        self._catalog: dict[str, WeightIdentity] = {}
        self._catalog_at: float = 0.0
        self._lock = threading.Lock()

    def _refresh_if_stale(self) -> None:
        now = time.time()
        with self._lock:
            if self._catalog and (now - self._catalog_at) < self._cache_ttl_s:
                return
        try:
            proc = subprocess.run(
                [self._binary, "models", "local", "--json"],
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.warning(
                "%s: failed to invoke local-llm CLI: %s", self.id, exc,
            )
            with self._lock:
                # Stamp time so we don't retry on every call within
                # the TTL window. A persistent-failure environment
                # falls through to other providers via composite.
                self._catalog_at = now
            return
        if proc.returncode != 0:
            logger.warning(
                "%s: local-llm CLI exited %d; stderr=%s",
                self.id, proc.returncode, proc.stderr[:300],
            )
            with self._lock:
                self._catalog_at = now
            return
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            logger.warning(
                "%s: local-llm output not valid JSON: %s", self.id, exc,
            )
            with self._lock:
                self._catalog_at = now
            return
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            logger.warning(
                "%s: local-llm output lacks 'entries' list", self.id,
            )
            with self._lock:
                self._catalog_at = now
            return
        new_catalog: dict[str, WeightIdentity] = {}
        for e in entries:
            if not isinstance(e, dict):
                continue
            m = e.get("model")
            if not isinstance(m, dict):
                continue
            model_id = m.get("id")
            source = m.get("source")
            runtime = m.get("runtime")
            if not isinstance(model_id, str) or not isinstance(source, str):
                continue
            if not isinstance(runtime, str):
                runtime = "unknown"
            new_catalog[model_id] = WeightIdentity(
                source=source,
                runtime=runtime,
                quantization=m.get("quantization")
                if isinstance(m.get("quantization"), str) else None,
                family=m.get("family")
                if isinstance(m.get("family"), str) else None,
            )
        with self._lock:
            self._catalog = new_catalog
            self._catalog_at = now

    def identify(self, model_id: str) -> WeightIdentity | None:
        self._refresh_if_stale()
        with self._lock:
            return self._catalog.get(model_id)


# ---------- concretion 2: heuristic fallback ------------------------------


_SIZE_TAG_PATTERN = re.compile(r"\d+[bB](?:-[a-z0-9]+)?")


class HeuristicWeightIdentityProvider:
    """Naming-pattern fallback for weight identity.

    Strips known transport suffixes from `model_id` to derive a
    weight-family stem. The result is heuristic and best-effort — it
    cannot distinguish quantization variants or detect cases where
    two differently-named cells secretly share the same artifact —
    but it correctly groups the common pattern of `X-ollama` and
    `X-ollama-responses-proxy` etc. as sharing weights.

    Use this provider BEHIND the CLI provider in a composite. When the
    CLI provider knows about a model, its answer is precise and wins.
    When the CLI is unavailable (degraded environment) or doesn't
    know the model (just-added, not yet in the registry), this
    provider provides a usable approximate answer rather than nothing.

    The `runtime` field of the returned identity is inferred from
    the stripped suffix (e.g. stripping `-responses-proxy` →
    `runtime="responses_proxy"`); unknown suffixes map to
    `runtime="unknown"`.
    """

    id = "heuristic-suffix-strip"

    def identify(self, model_id: str) -> WeightIdentity | None:
        runtime = "unknown"
        stem = model_id
        # Order matters: longer suffixes first so we don't strip
        # `-ollama` off `-ollama-responses-proxy` and lose info.
        for suffix in _TRANSPORT_SUFFIXES:
            if model_id.endswith(suffix):
                runtime = suffix.lstrip("-").replace("-", "_")
                stem = model_id[: -len(suffix)]
                break
        if stem == model_id:
            # No known suffix matched. We have no way to derive a
            # different weight family from a cell that already looks
            # bare — return None and let other providers try.
            return None
        # Family is the stem with size tag stripped, if recognizable.
        family_match = _SIZE_TAG_PATTERN.search(stem)
        family = (
            stem[: family_match.start()].rstrip("-")
            if family_match else stem
        )
        return WeightIdentity(
            source=stem,
            runtime=runtime,
            quantization=None,
            family=family or None,
        )


# ---------- concretion 3: null --------------------------------------------


class NullWeightIdentityProvider:
    """Always returns None. For tests and for environments where
    weight identity is genuinely unknowable (no CLI, naming pattern
    isn't recognizable). Including this in a composite is harmless;
    it falls through to whatever's next."""

    id = "null"

    def identify(self, model_id: str) -> WeightIdentity | None:
        return None


# ---------- composite -----------------------------------------------------


@dataclass(slots=True)
class _CompositeStats:
    """Per-composite-instance counters for observability. Not thread-
    safe; if you need exact totals from many threads at once, add a
    lock or replace with atomic counters. For typical use (logged at
    sweep time, not at request time), the existing approximation is
    fine."""

    hits: dict[str, int] = field(default_factory=dict)
    misses: int = 0
    disagreements: int = 0


class CompositeWeightIdentityProvider:
    """Tries each provider in priority order. First non-None wins.

    Also detects DISAGREEMENT: when two providers both have an answer
    but their `source` fields differ. The first-priority answer is
    still used, but the disagreement is logged at WARNING level so an
    operator notices when their concrete providers contradict each
    other. This is the safeguard against "two CLIs say two different
    things and one of them is wrong."

    The whole point of this class is that callers depend ONLY on
    `WeightIdentityProvider` (the protocol) and the composite. They
    never name a concrete provider. Adding a new concrete provider
    (e.g. a manifest-file reader, a different CLI) is one new class
    plus one item in the composite's constructor — no callers change.
    """

    id = "composite"

    def __init__(self, providers: list[WeightIdentityProvider]) -> None:
        if not providers:
            # An empty composite is legal but useless; warn loudly so
            # an operator notices a missing wire-up at startup rather
            # than wondering why every weight_family is None forever.
            logger.warning(
                "composite weight-identity provider constructed with NO "
                "providers; identify() will always return None"
            )
        self._providers = providers
        self._stats = _CompositeStats()
        self._lock = threading.Lock()

    def identify(self, model_id: str) -> WeightIdentity | None:
        first_answer: WeightIdentity | None = None
        first_provider_id: str | None = None
        for p in self._providers:
            try:
                ans = p.identify(model_id)
            except Exception:
                logger.exception(
                    "composite weight-identity: provider %r raised on %r",
                    p.id, model_id,
                )
                continue
            if ans is None:
                continue
            if first_answer is None:
                first_answer = ans
                first_provider_id = p.id
                continue
            if ans.source != first_answer.source:
                logger.warning(
                    "weight-identity disagreement for %r: %s says source=%r, "
                    "%s says source=%r (using %s)",
                    model_id, first_provider_id, first_answer.source,
                    p.id, ans.source, first_provider_id,
                )
                with self._lock:
                    self._stats.disagreements += 1
        with self._lock:
            if first_answer is None:
                self._stats.misses += 1
            elif first_provider_id is not None:
                self._stats.hits[first_provider_id] = (
                    self._stats.hits.get(first_provider_id, 0) + 1
                )
        return first_answer

    def stats(self) -> dict[str, Any]:
        """Snapshot of hit counts per provider plus disagreement and
        miss totals. Returned as a plain dict so it serializes
        cleanly for status endpoints."""
        with self._lock:
            return {
                "hits_per_provider": dict(self._stats.hits),
                "misses": self._stats.misses,
                "disagreements": self._stats.disagreements,
            }


def build_default_provider() -> CompositeWeightIdentityProvider:
    """Construct the standard composite used by callosum at startup:
    CLI first, heuristic fallback, null backstop. Centralized here so
    test fixtures, ad-hoc scripts, and the app share one definition.
    Adding a new provider system-wide is one edit to this function."""
    return CompositeWeightIdentityProvider(
        providers=[
            LocalLlmCliWeightIdentityProvider(),
            HeuristicWeightIdentityProvider(),
            NullWeightIdentityProvider(),
        ]
    )
