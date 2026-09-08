"""Auto-probe scheduler.

Wires `routing/probe.py` into automatic cell registration so the
operator never has to manually run `callosum-ctl probe-tools` to
discover which local models actually emit OpenAI-shaped tool calls.
Closes the whack-a-mole gap where new models entered the pool, got
routed tool-using traffic, and emitted JSON-as-text to the user
before anyone noticed.

Design:

  * On callosum startup (and on operator-driven re-probe), iterate
    every advertised cell across the loaded backends.
  * For cells served by LOCAL backends (where the supports_tools
    flag is uncertain because ollama's catalog is overly permissive),
    run the probe in the background. Remote cells (Codex) are
    skipped — their capability claims are trustworthy.
  * Persist each probe outcome in operator_state.sqlite. `_capabilities_of`
    (in app.py) consults the cache to OVERRIDE the backend's claimed
    supports_tools when the probe disagrees.
  * Re-probe cells whose cached result is older than `probe_ttl_s`
    (default 24h). Skip cells whose result is fresh enough.
  * Run probes serially per cell to avoid VRAM contention on the
    local GPU — two 31B-class probes in parallel would OOM most
    workstations.

This module is the orchestrator. The actual probing logic stays in
`routing/probe.py` so it remains independently testable.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from callosum.routing.probe import probe_supports_tools

if TYPE_CHECKING:
    from callosum.operator_state import OperatorState


logger = logging.getLogger(__name__)


# Cells served by these backend kinds get probed automatically. Remote
# cells (codex_auth_vault) are skipped — their capability claims come
# from a curated upstream catalog and don't lie about tool support.
# Local cells (litellm_gateway kind, which covers both
# LocalModelRegistryBackend and LiteLLMGatewayBackend) probe because their
# upstream catalogs (ollama /api/show) are overly permissive about
# tool support and the model's emitted shape is the real ground truth.
_LOCAL_BACKEND_KINDS = frozenset({"litellm_gateway"})

# How long a probe result stays trusted before re-probing. 24 hours
# balances "catch newly-installed model upgrades" against "don't burn
# 30s-per-cell every startup." Operator can force a re-probe by
# running `callosum-ctl probe-tools` (operator-driven path; the
# scheduler treats those results identically to its own).
DEFAULT_PROBE_TTL_S: float = 24 * 3600.0


async def _probe_one_cell(*, backend: Any, model: str, operator_state: OperatorState) -> None:
    """Probe one cell and persist the result. Defensive — never raises;
    a probe that errors out is recorded as supports_tools=False with
    the error captured for the operator to inspect later.

    The error-capture mechanism uses a closure-local mutable list
    because `probe_supports_tools` (in routing/probe.py) has its own
    try/except that swallows transport errors before they reach us.
    The closure here captures the error message at the point of failure
    so it survives that internal swallow; without this, every
    transport-error probe would record supports_tools=False with no
    explanation, leaving the operator no way to distinguish "model
    works but emits wrong shape" from "couldn't reach the model at all."
    """
    backend_id = getattr(backend, "id", "?")
    t0 = time.time()
    captured_error: list[str | None] = [None]
    supports = False
    try:

        async def _call(probe_body: dict[str, Any]) -> dict[str, Any]:
            try:
                result: dict[str, Any] = await backend.responses(probe_body)
                return result
            except Exception as exc:
                captured_error[0] = f"{type(exc).__name__}: {exc}"
                raise

        supports = await probe_supports_tools(model=model, call_responses=_call)
    except Exception as exc:
        # Reached only when the probe machinery itself errors (not the
        # call inside it — that's captured above). Belt-and-suspenders.
        if captured_error[0] is None:
            captured_error[0] = f"{type(exc).__name__}: {exc}"
    error_msg = captured_error[0]
    latency_ms = int((time.time() - t0) * 1000)
    try:
        operator_state.set_probe_result(
            backend_id, model, supports_tools=supports, error=error_msg, latency_ms=latency_ms
        )
        logger.info(
            "probe %s/%s: supports_tools=%s latency=%dms%s",
            backend_id,
            model,
            supports,
            latency_ms,
            f" error={error_msg}" if error_msg else "",
        )
    except Exception:
        # Even the persistence path is defensive — losing one probe
        # result is far better than crashing the proxy startup loop.
        logger.exception("probe %s/%s: failed to persist result", backend_id, model)


def _cells_needing_probe(
    *, backends: list[Any], operator_state: OperatorState, probe_ttl_s: float, now: float
) -> list[tuple[Any, str]]:
    """Return (backend, model) pairs that should be probed right now.

    A cell needs probing if:
      * the cell's backend kind is in `_LOCAL_BACKEND_KINDS`, AND
      * the backend exposes `.responses(body)` (non-stream path the
        probe needs), AND
      * no cached probe result exists yet, OR the cached result is
        older than `probe_ttl_s`.

    Cells are de-duplicated by (backend_id, model) so when two backend
    instances both advertise the same cell name we only probe via one.
    """
    seen: set[tuple[str, str]] = set()
    out: list[tuple[Any, str]] = []
    for backend in backends:
        if getattr(backend, "kind", "") not in _LOCAL_BACKEND_KINDS:
            continue
        if not hasattr(backend, "responses"):
            continue
        backend_id = getattr(backend, "id", "?")
        advertised: frozenset[str] = getattr(backend, "advertised_models", frozenset())
        for model in sorted(advertised):
            key = (backend_id, model)
            if key in seen:
                continue
            seen.add(key)
            cached = operator_state.get_probe_result(backend_id, model)
            if cached is not None:
                _supports, probed_at = cached
                if now - probed_at < probe_ttl_s:
                    continue
            out.append((backend, model))
    return out


async def run_probe_sweep(
    *, backends: list[Any], operator_state: OperatorState, probe_ttl_s: float = DEFAULT_PROBE_TTL_S
) -> int:
    """Probe every uncached / stale local cell and persist the
    results. Returns the number of probes actually executed (excludes
    cells whose cached result is still fresh).

    Probes run serially to avoid VRAM contention on the local GPU.
    Each probe persists its own result immediately, so a partial sweep
    (e.g. interrupted by shutdown) doesn't lose the cells that did
    complete.
    """
    todo = _cells_needing_probe(
        backends=backends, operator_state=operator_state, probe_ttl_s=probe_ttl_s, now=time.time()
    )
    if not todo:
        return 0
    logger.info("probe sweep: %d cell(s) need probing (TTL=%.0fs)", len(todo), probe_ttl_s)
    for backend, model in todo:
        await _probe_one_cell(backend=backend, model=model, operator_state=operator_state)
    return len(todo)


def schedule_background_sweep(
    *, backends: list[Any], operator_state: OperatorState, probe_ttl_s: float = DEFAULT_PROBE_TTL_S
) -> asyncio.Task[int]:
    """Fire-and-forget version of `run_probe_sweep` suitable for
    callosum's startup. Returns the asyncio Task so callers can join
    it during shutdown if desired; for the typical "kick it off and
    let it run" pattern, the return value can be ignored.

    Wraps the sweep in an exception handler that logs but never
    propagates — probe failures must not crash startup, and probe
    progress must not block request serving.
    """

    async def _runner() -> int:
        try:
            return await run_probe_sweep(backends=backends, operator_state=operator_state, probe_ttl_s=probe_ttl_s)
        except Exception:
            logger.exception("probe sweep: unexpected failure")
            return 0

    return asyncio.create_task(_runner())


def supports_tools_override(*, operator_state: OperatorState, backend_id: str, model: str) -> bool | None:
    """Return the probe-derived override for a cell's supports_tools
    flag, or None when no override applies.

    Returns:
      * False when the cell has a cached probe result that failed
        (the cell does NOT emit structured tool_calls end-to-end).
      * None when there's no cached result, OR when the cached
        result PASSED (in which case the caller should trust the
        backend's own claim).

    The asymmetry is deliberate: a probe pass doesn't reach UP to
    grant tool support to a backend that says it lacks tools; a
    probe fail reaches DOWN to deny tool support a backend wrongly
    claims. We only override in the safer direction.
    """
    cached = operator_state.get_probe_result(backend_id, model)
    if cached is None:
        return None
    supports, _probed_at = cached
    return False if not supports else None
