from __future__ import annotations

from codex_proxy.backend import HealthStatus, UsageSnapshot
from codex_proxy.fakes import InMemoryFakeBackend
from codex_proxy.selector import select


def _usage(
    *,
    remaining: float | None = 1.0,
    cooldown_until: float | None = None,
    weekly_exhausted: bool = False,
    probed_at: float = 0.0,
) -> UsageSnapshot:
    return UsageSnapshot(
        remaining_fraction=remaining,
        cooldown_until_ts=cooldown_until,
        weekly_exhausted=weekly_exhausted,
        probed_at_ts=probed_at,
    )


def _fake(
    id: str,
    *,
    advertised_models: frozenset[str] | None = None,
    health: HealthStatus | None = None,
    usage: UsageSnapshot | None = None,
) -> InMemoryFakeBackend:
    return InMemoryFakeBackend(
        id=id,
        advertised_models=advertised_models or frozenset({"model-a0d0"}),
        health=health,
        usage=usage,
    )


async def test_empty_pool_returns_none() -> None:
    assert await select([], model="model-a0d0") is None


async def test_single_healthy_backend_is_chosen() -> None:
    backend = _fake("a")
    assert await select([backend], model="model-a0d0") is backend


async def test_backend_without_model_is_skipped() -> None:
    backend = _fake("a", advertised_models=frozenset({"other-model"}))
    assert await select([backend], model="model-a0d0") is None


async def test_unavailable_backend_is_skipped() -> None:
    backend = _fake("a", health=HealthStatus(available=False, reason="rate_limited"))
    assert await select([backend], model="model-a0d0") is None


async def test_backend_in_cooldown_is_skipped() -> None:
    backend = _fake("a", usage=_usage(cooldown_until=100.0))
    assert await select([backend], model="model-a0d0", now_ts=50.0) is None


async def test_backend_past_cooldown_is_included() -> None:
    backend = _fake("a", usage=_usage(cooldown_until=100.0))
    assert await select([backend], model="model-a0d0", now_ts=200.0) is backend


async def test_weekly_exhausted_is_deprioritized() -> None:
    a = _fake("a", usage=_usage(weekly_exhausted=True))
    b = _fake("b", usage=_usage(weekly_exhausted=False))
    assert await select([a, b], model="model-a0d0") is b


async def test_higher_remaining_fraction_preferred() -> None:
    a = _fake("a", usage=_usage(remaining=0.2))
    b = _fake("b", usage=_usage(remaining=0.8))
    assert await select([a, b], model="model-a0d0") is b


async def test_none_remaining_is_treated_as_half() -> None:
    low = _fake("low", usage=_usage(remaining=0.3))
    unknown = _fake("unknown", usage=_usage(remaining=None))
    assert await select([low, unknown], model="model-a0d0") is unknown


async def test_recency_breaks_ties_when_usage_equal() -> None:
    stale = _fake("stale", usage=_usage(remaining=0.5, probed_at=100.0))
    fresh = _fake("fresh", usage=_usage(remaining=0.5, probed_at=200.0))
    assert await select([stale, fresh], model="model-a0d0") is fresh


async def test_id_is_final_deterministic_tiebreak() -> None:
    a = _fake("a", usage=_usage(remaining=0.5, probed_at=100.0))
    b = _fake("b", usage=_usage(remaining=0.5, probed_at=100.0))
    assert await select([a, b], model="model-a0d0") is a


async def test_all_unhealthy_returns_none() -> None:
    a = _fake("a", health=HealthStatus(available=False, reason="auth_invalid"))
    b = _fake("b", health=HealthStatus(available=False, reason="network"))
    assert await select([a, b], model="model-a0d0") is None


async def test_mixed_pool_picks_best_eligible() -> None:
    pool = [
        _fake("unhealthy", health=HealthStatus(available=False, reason="network")),
        _fake("wrong-model", advertised_models=frozenset({"other"})),
        _fake("exhausted", usage=_usage(weekly_exhausted=True, remaining=1.0)),
        _fake("ok-low", usage=_usage(remaining=0.1)),
        _fake("ok-high", usage=_usage(remaining=0.9)),
    ]
    chosen = await select(pool, model="model-a0d0")
    assert chosen is not None
    assert chosen.id == "ok-high"
