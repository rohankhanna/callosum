"""Probe runner: orchestrates a sequence of dimensions for one cell.

Single entry point both the pytest harness and callosum's background
scheduler call. Keeps the orchestration logic in one place so the
two invocation paths can't drift apart over time.

Flow per cell:
  1. Load (or create) the cell's CapabilityProfile from disk.
  2. For each dimension in DIMENSIONS:
       a. Skip if the cell has a fresh-enough result already (TTL gate)
       b. Run the dimension probe with the per-call closure
       c. Upsert the resulting DimensionFinding into the profile
       d. Persist the profile to disk after each dimension so a
          partial run (interrupted by shutdown / cancellation /
          process kill) doesn't lose completed dimensions.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from callosum.capability.dimensions import DIMENSIONS
from callosum.capability.profile import (
    DimensionFinding,
    load_profile,
    save_profile,
)
from callosum.capability.weight_identity import WeightIdentityProvider

logger = logging.getLogger(__name__)

# Default TTL for a dimension's cached result before it's considered
# stale enough to re-run. The thorough harness is minutes per cell so
# we don't want to re-probe every restart. Operators can force a
# re-probe by deleting the profile JSON or by passing ttl_s=0.
DEFAULT_DIMENSION_TTL_S: float = 7 * 24 * 3600.0  # one week


CallResponsesFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


async def run_dimensions(
    *,
    cell: str,
    backend_id: str | None,
    call_responses: CallResponsesFn,
    ttl_s: float = DEFAULT_DIMENSION_TTL_S,
    now: float | None = None,
    weight_identity_provider: WeightIdentityProvider | None = None,
) -> dict[str, DimensionFinding]:
    """Run every registered dimension probe against `cell`. Returns the
    set of findings that were actually produced this invocation (skips
    fresh-cached dimensions; those stay in the profile but aren't in
    the returned dict).

    `call_responses` is the abstraction over how a probe body actually
    reaches the cell. Pytest passes a function that posts to
    /admin/cell-call. The background scheduler passes a function that
    invokes the backend's responses() method directly. The runner
    doesn't care.

    `weight_identity_provider`, when supplied, is consulted once per
    sweep to stamp `profile.weight_identity`. This lets a downstream
    consumer detect "two cells share weights but produced divergent
    findings" — divergence is then information about the transport,
    not the model. The provider is a protocol so callers can mix in
    any concrete provider list. None disables stamping; any
    previously-persisted identity remains in place.
    """
    now = now or time.time()
    profile = load_profile(cell)
    if backend_id is not None:
        profile.backend_id = backend_id
    # Stamp identity BEFORE the dimension loop so even a partial-
    # results profile (interrupted sweep) carries the cross-cell
    # grouping signal. A provider that returns None or raises leaves
    # any previously-persisted identity in place — we never clobber a
    # known identity to None on a transient provider outage.
    identity_changed = False
    if weight_identity_provider is not None:
        try:
            new_identity = weight_identity_provider.identify(cell)
        except Exception:
            logger.exception(
                "probe %s: weight-identity provider raised; leaving existing identity in place",
                cell,
            )
            new_identity = None
        if new_identity is not None and new_identity != profile.weight_identity:
            profile.weight_identity = new_identity
            identity_changed = True

    fresh_results: dict[str, DimensionFinding] = {}
    for name, probe_fn in DIMENSIONS:
        existing = profile.findings.get(name)
        if existing is not None and (now - profile.last_updated) < ttl_s and existing.status in ("pass", "fail"):
            # Skip — we have a fresh deterministic result. "error" and
            # "skipped" don't count as fresh; those should re-run so
            # transient failures don't stick forever.
            logger.debug(
                "probe %s/%s: skipping (cached %s within TTL)",
                cell,
                name,
                existing.status,
            )
            continue
        try:
            finding = await probe_fn(cell, call_responses, profile)
        except Exception as exc:
            # The probes themselves are defensive but a programming
            # bug in a probe must not crash the runner — partial
            # results from earlier dimensions stay persisted.
            logger.exception(
                "probe %s/%s: dimension probe raised",
                cell,
                name,
            )
            finding = DimensionFinding(
                dimension=name,
                status="error",
                summary=(f"probe function raised: {type(exc).__name__}: {exc}"),
                evidence={"exception": f"{type(exc).__name__}: {exc}"},
            )
        profile.upsert(finding)
        save_profile(profile)
        fresh_results[name] = finding
        logger.info(
            "probe %s/%s: %s — %s",
            cell,
            name,
            finding.status,
            finding.summary,
        )
    # If every dimension was TTL-fresh, the dimension loop did no
    # work and didn't persist. But a freshly-stamped identity still
    # needs to land on disk; otherwise the next process restart would
    # see the cached profile without the identity it had in-memory.
    if identity_changed and not fresh_results:
        save_profile(profile)
    return fresh_results
