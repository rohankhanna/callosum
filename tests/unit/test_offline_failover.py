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
    exhausted = _fake_backend(
        id="codex-primary", models=["model-a0e7"], weekly_exhausted=True
    )
    routable = asyncio.run(_routable_backends([healthy, exhausted]))
    assert [b.id for b in routable] == ["local"]


def test_routable_excludes_active_cooldown() -> None:
    """A backend with cooldown_until_ts in the future is unroutable
    until the cooldown expires."""
    healthy = _fake_backend(id="local", models=["model-a0a9"])
    future = time.time() + 600  # cooldown ends in 10 minutes
    cooling = _fake_backend(
        id="codex-secondary", models=["model-a0e7"], cooldown_until_ts=future
    )
    routable = asyncio.run(_routable_backends([healthy, cooling]))
    assert [b.id for b in routable] == ["local"]


def test_routable_includes_expired_cooldown() -> None:
    """A cooldown_until_ts in the past doesn't exclude the backend —
    it's recovered, the snapshot just hasn't been refreshed."""
    past = time.time() - 60
    recovered = _fake_backend(
        id="codex-primary", models=["model-a0e7"], cooldown_until_ts=past
    )
    routable = asyncio.run(_routable_backends([recovered]))
    assert [b.id for b in routable] == ["codex-primary"]


def test_routable_handles_unreadable_backend() -> None:
    """If a backend's usage_snapshot raises, treat as unroutable —
    'we can't tell' is safer than 'assume healthy'."""

    class _BrokenBackend(InMemoryFakeBackend):
        async def usage_snapshot(self):
            raise RuntimeError("backend state unreadable")

    broken = _BrokenBackend(
        id="broken", advertised_models=frozenset({"model-a0e7"})
    )
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
        Cell(model="model-a0e7", reasoning_effort=e, context_window=128_000)
        for e in ("low", "medium", "high", "xhigh")
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


# ---------- recommender skips classifier when its backend is unroutable -----


def test_recommender_skips_upstream_call_when_cheap_backend_exhausted() -> None:
    """Phase 4d-v2: when the classifier's backend is weekly_exhausted,
    the recommender shouldn't even attempt the upstream call — it would
    just timeout or 429, costing latency on the first post-outage
    request. Falls back immediately."""
    from dataclasses import dataclass

    from callosum.backend import CallHandle, HealthStatus, UsageSnapshot
    from callosum.cell_recommender import CellRecommender
    from typing import Any

    @dataclass
    class _ExhaustedBackend:
        id: str = "primary"
        kind: str = "codex_auth_vault"
        advertised_models = frozenset({"model-a0c3"})
        call_count: int = 0

        async def health(self) -> HealthStatus:
            return HealthStatus(available=False, reason="rate_limited")

        async def usage_snapshot(self) -> UsageSnapshot:
            return UsageSnapshot(
                remaining_fraction=0.0,
                cooldown_until_ts=None,
                weekly_exhausted=True,
                probed_at_ts=0.0,
            )

        async def quota_snapshot(self) -> None:
            return None

        async def responses(
            self, body: dict[str, Any], handle: CallHandle
        ) -> dict[str, Any]:
            self.call_count += 1
            raise RuntimeError("would 429 if actually called")

        async def aclose(self) -> None:
            pass

    backend = _ExhaustedBackend()
    cheap = Cell(model="model-a0c3", reasoning_effort="low", context_window=None)
    rec = CellRecommender(cheap_backend=backend, cheap_cell=cheap, upstream_timeout_s=2.0)
    cells = [
        cheap,
        Cell(model="model-a0a9", reasoning_effort="default",
             context_window=262_144),
    ]
    body = {"messages": [{"role": "user", "content": "what is 2+2?"}]}
    out = asyncio.run(rec.recommend(body, allowed_cells=cells, fallback=cheap))
    assert out.source == "fallback"
    # Crucially: no upstream call was made.
    assert backend.call_count == 0
    assert rec.stats.get("upstream_skipped_unroutable", 0) == 1
    assert rec.stats.get("upstream_calls", 0) == 0


def test_recommender_skips_upstream_when_cheap_backend_on_cooldown() -> None:
    """Same skip behavior for cooldown_until_ts in the future."""
    from dataclasses import dataclass

    from callosum.backend import CallHandle, HealthStatus, UsageSnapshot
    from callosum.cell_recommender import CellRecommender
    from typing import Any

    @dataclass
    class _CoolingBackend:
        id: str = "primary"
        kind: str = "codex_auth_vault"
        advertised_models = frozenset({"model-a0c3"})
        call_count: int = 0

        async def health(self) -> HealthStatus:
            return HealthStatus(available=False, reason="rate_limited")

        async def usage_snapshot(self) -> UsageSnapshot:
            return UsageSnapshot(
                remaining_fraction=0.5,
                cooldown_until_ts=time.time() + 600,  # 10 min from now
                weekly_exhausted=False,
                probed_at_ts=0.0,
            )

        async def quota_snapshot(self) -> None:
            return None

        async def responses(self, body, handle):
            self.call_count += 1
            raise RuntimeError("would hang")

        async def aclose(self) -> None:
            pass

    backend = _CoolingBackend()
    cheap = Cell(model="model-a0c3", reasoning_effort="low", context_window=None)
    rec = CellRecommender(cheap_backend=backend, cheap_cell=cheap, upstream_timeout_s=2.0)
    out = asyncio.run(rec.recommend(
        {"messages": [{"role": "user", "content": "x"}]},
        allowed_cells=[cheap], fallback=cheap,
    ))
    assert out.source == "fallback"
    assert backend.call_count == 0
