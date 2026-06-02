"""Background scheduler for the thorough capability harness.

Closes the loop: when callosum starts (or when 
downloads a new model and ollama picks it up on the next catalog
refresh), the registered cells are walked, and any cell without a
fresh capability profile gets probed across every registered
dimension. Findings persist to disk as JSON, where downstream
consumers (adapter writers, the router itself, operator tooling)
can read them without re-running the harness.

This scheduler is the THOROUGH counterpart to
`routing.probe_scheduler.schedule_background_sweep`:

  * Light probe (routing/probe_scheduler.py): seconds per cell,
    binary supports_tools answer, runs on every startup and gates
    real-time routing decisions.
  * Thorough harness (this module): minutes per cell, multi-
    dimensional findings + adapter hints, runs in background and
    only re-runs when the per-dimension cache TTL expires (default
    one week — see `runner.DEFAULT_DIMENSION_TTL_S`).

Both schedulers can coexist; they probe different things at
different cadences. The light probe is what the router consults
synchronously; the thorough harness produces the artifacts that
adapter authors and  reason about.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from callosum.capability.runner import (
    DEFAULT_DIMENSION_TTL_S,
    run_dimensions,
)

if TYPE_CHECKING:
    from callosum.operator_state import OperatorState


logger = logging.getLogger(__name__)


# Mirror probe_scheduler._LOCAL_BACKEND_KINDS so both schedulers
# agree on which cells warrant probing. Remote (codex_auth_vault)
# cells have curated capability claims and don't need a harness.
_LOCAL_BACKEND_KINDS = frozenset({"litellm_gateway"})


def _cells_to_harness(backends: list[Any]) -> list[tuple[Any, str]]:
    """Return (backend, cell) pairs the harness should consider.

    Deduplicated by (backend_id, cell) so cells advertised by two
    backend instances only get harnessed via one. Whether a given
    cell actually runs is then governed by `run_dimensions`'s TTL
    gate per dimension — we don't second-guess that here.
    """
    seen: set[tuple[str, str]] = set()
    out: list[tuple[Any, str]] = []
    for backend in backends:
        if getattr(backend, "kind", "") not in _LOCAL_BACKEND_KINDS:
            continue
        if not hasattr(backend, "responses"):
            continue
        backend_id = getattr(backend, "id", "?")
        advertised: frozenset[str] = getattr(
            backend, "advertised_models", frozenset()
        )
        for cell in sorted(advertised):
            key = (backend_id, cell)
            if key in seen:
                continue
            seen.add(key)
            out.append((backend, cell))
    return out


async def run_harness_sweep(
    *,
    backends: list[Any],
    ttl_s: float = DEFAULT_DIMENSION_TTL_S,
) -> int:
    """Run the thorough harness across every local cell. Returns the
    total number of dimensions actually executed (those whose
    cached result was either missing, expired, or non-deterministic
    last time). Cells whose every dimension was already fresh in
    the TTL window contribute zero.

    Harnesses run serially across cells to avoid GPU contention —
    two 31B-class harnesses in parallel would OOM most workstations.
    """
    todo = _cells_to_harness(backends)
    if not todo:
        return 0
    logger.info(
        "capability harness sweep: %d cell(s) eligible (per-dim TTL=%.0fs)",
        len(todo),
        ttl_s,
    )
    total_dimensions_run = 0
    for backend, cell in todo:
        backend_id = getattr(backend, "id", "?")

        async def _call(
            body: dict[str, Any], _backend: Any = backend
        ) -> dict[str, Any]:
            result: dict[str, Any] = await _backend.responses(body)
            return result

        try:
            fresh = await run_dimensions(
                cell=cell,
                backend_id=backend_id,
                call_responses=_call,
                ttl_s=ttl_s,
            )
            total_dimensions_run += len(fresh)
        except Exception:
            # run_dimensions itself is defensive; this is paranoia in
            # case orchestration around it ever raises. One cell's
            # failure must not block the rest of the sweep.
            logger.exception(
                "capability harness: sweep entry %s/%s raised",
                backend_id, cell,
            )
    logger.info(
        "capability harness sweep: complete (%d dimension run(s) total)",
        total_dimensions_run,
    )
    return total_dimensions_run


def schedule_background_harness(
    *,
    backends: list[Any],
    operator_state: OperatorState | None = None,
    ttl_s: float = DEFAULT_DIMENSION_TTL_S,
) -> asyncio.Task[int]:
    """Fire-and-forget version suitable for callosum's lifespan
    startup. Returns the asyncio Task so callers can join it during
    shutdown if they want; for the typical kick-and-forget pattern,
    the return can be ignored.

    `operator_state` is currently unused — the harness persists
    findings to JSON files in logs/capability_profiles/, not to
    SQLite. The parameter is accepted to keep the call-site
    symmetric with `schedule_background_sweep` so adding harness
    metadata into operator_state later is a one-line change.

    Exception swallowing inside `_runner` is the same pattern as
    the light-probe scheduler: probe sweeps must NEVER crash startup
    and must NEVER block request serving.
    """
    _ = operator_state  # reserved for future use; see docstring

    async def _runner() -> int:
        try:
            return await run_harness_sweep(backends=backends, ttl_s=ttl_s)
        except Exception:
            logger.exception("capability harness sweep: unexpected failure")
            return 0

    return asyncio.create_task(_runner())
