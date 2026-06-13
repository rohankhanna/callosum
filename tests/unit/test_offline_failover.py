"""Phase 4d: offline / Codex-out failover tests.

When Codex is exhausted (weekly_used >= 99%), in cooldown, or fully
unreachable, callosum should keep serving on local backends with no
operator action required. The cell grid filtering at the recommender
call site is the foundation — cells whose only serving backends are
unroutable get dropped from the classifier's view, so the recommender
can only pick something dispatch can actually reach.

These tests cover the cell-filter helper in isolation. The end-to-end
behavior (recommender + dispatch under simulated Codex outage) is
exercised through the integration test that drives a full request.
"""

from __future__ import annotations

import asyncio
import time

from callosum.app import _filter_cells_to_routable, _routable_backends
from callosum.backend import HealthStatus, UsageSnapshot
from callosum.cell_grid import Cell
from callosum.fakes import InMemoryFakeBackend


def _fake_backend(
    *,
    id: str,
    models: list[str],
    weekly_exhausted: bool = False,
    cooldown_until_ts: float | None = None,
) -> InMemoryFakeBackend:
    return InMemoryFakeBackend(
        id=id,
        advertised_models=frozenset(models),
        health=HealthStatus(available=True, reason="ok"),
        usage=UsageSnapshot(
            remaining_fraction=1.0,
            cooldown_until_ts=cooldown_until_ts,
            weekly_exhausted=weekly_exhausted,
            probed_at_ts=0.0,
        ),
    )


# ---------- _routable_backends -----------------------------------------------


def test_routable_excludes_weekly_exhausted() -> None:
    """Codex weekly-exhausted state must take a backend out of rotation —
    the whole point of the failover."""
    healthy = _fake_backend(id="local", models=["model-a0a9"])
    exhausted = _fake_backend(id="codex-primary", models=["model-a0e7"], weekly_exhausted=True)
    routable = asyncio.run(_routable_backends([healthy, exhausted]))
    assert [b.id for b in routable] == ["local"]


def test_routable_excludes_active_cooldown() -> None:
    """A backend with cooldown_until_ts in the future is unroutable
    until the cooldown expires."""
    healthy = _fake_backend(id="local", models=["model-a0a9"])
    future = time.time() + 600  # cooldown ends in 10 minutes
    cooling = _fake_backend(id="codex-secondary", models=["model-a0e7"], cooldown_until_ts=future)
    routable = asyncio.run(_routable_backends([healthy, cooling]))
    assert [b.id for b in routable] == ["local"]


def test_routable_includes_expired_cooldown() -> None:
    """A cooldown_until_ts in the past doesn't exclude the backend —
    it's recovered, the snapshot just hasn't been refreshed."""
    past = time.time() - 60
    recovered = _fake_backend(id="codex-primary", models=["model-a0e7"], cooldown_until_ts=past)
    routable = asyncio.run(_routable_backends([recovered]))
    assert [b.id for b in routable] == ["codex-primary"]


def test_routable_handles_unreadable_backend() -> None:
    """If a backend's usage_snapshot raises, treat as unroutable —
    'we can't tell' is safer than 'assume healthy'."""

    class _BrokenBackend(InMemoryFakeBackend):
        async def usage_snapshot(self):
            raise RuntimeError("backend state unreadable")

    broken = _BrokenBackend(id="broken", advertised_models=frozenset({"model-a0e7"}))
    healthy = _fake_backend(id="ok", models=["model-a0a9"])
    routable = asyncio.run(_routable_backends([broken, healthy]))
    assert [b.id for b in routable] == ["ok"]


# ---------- _filter_cells_to_routable ----------------------------------------


def test_filter_keeps_cells_when_at_least_one_backend_serves() -> None:
    """A cell whose model is advertised by ANY routable backend stays in
    the grid. Multi-backend redundancy (Codex primary + secondary serving
    the same model) means if either is healthy, the cells survive."""
    cells = [
        Cell(model="model-a0e7", reasoning_effort="high", context_window=128_000),
        Cell(model="model-a0a9", reasoning_effort="default", context_window=262_144),
    ]
    primary = _fake_backend(id="primary", models=["model-a0e7"])
    out = _filter_cells_to_routable(cells, [primary])
    # model-a0e7 stays (primary serves it); model-a0d5 stays out (no backend serves it).
    assert out == [cells[0]]


def test_filter_drops_codex_cells_when_all_codex_unroutable() -> None:
    """The core failover behavior: Codex exhausted → all Codex cells
    disappear from the grid → classifier can only pick local."""
    codex_cells = [
        Cell(model="model-a0e7", reasoning_effort=e, context_window=128_000) for e in ("low", "medium", "high", "xhigh")
    ]
    local_cells = [
        Cell(model="model-a0a9", reasoning_effort="default", context_window=262_144),
        Cell(model="model-a0a8", reasoning_effort="default", context_window=32_768),
    ]
    all_cells = codex_cells + local_cells
    # Only the local backend is routable (Codex backends omitted from list).
    local_backend = _fake_backend(
        id="local LLM gateway",
        models=["model-a0a9", "model-a0a8"],
    )
    out = _filter_cells_to_routable(all_cells, [local_backend])
    assert out == local_cells


def test_filter_returns_empty_when_no_routable_backends() -> None:
    """All backends down → no cells survive. Caller is expected to
    fall back to the unfiltered grid as a last resort rather than
    refuse the request."""
    cells = [Cell(model="model-a0e7", reasoning_effort="high", context_window=128_000)]
    assert _filter_cells_to_routable(cells, []) == []


# NOTE: The legacy "recommender skips upstream when its cheap_backend is
# unroutable" tests were removed when the LLM-based router was deleted.
# Equivalent coverage in the new world: when a backend marks itself
# unroutable via usage_snapshot, _filter_cells_to_routable drops its
# cells from the candidate set (verified by tests above), so the Router's
# capability filter never sees them. No "skip the classifier call" code
# path exists in the new architecture because there IS no per-request
# classifier call to skip.
