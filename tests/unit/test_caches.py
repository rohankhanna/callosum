from __future__ import annotations

from callosum.caches import TtlCache


def test_ttl_cache_serves_within_ttl() -> None:
    t = [0.0]
    cache = TtlCache(10.0, clock=lambda: t[0])

    assert cache.get() is None  # cold

    cache.set("v1")
    t[0] = 5.0
    assert cache.get() == "v1"
    t[0] = 9.9
    assert cache.get() == "v1"


def test_ttl_cache_expires_at_and_after_ttl() -> None:
    t = [0.0]
    cache = TtlCache(10.0, clock=lambda: t[0])

    cache.set("v1")
    t[0] = 10.0  # exactly at the TTL boundary -> expired (strict <)
    assert cache.get() is None
    t[0] = 15.0
    assert cache.get() is None


def test_ttl_cache_invalidate_clears_until_refilled() -> None:
    t = [0.0]
    cache = TtlCache(10.0, clock=lambda: t[0])

    cache.set("v1")
    assert cache.get() == "v1"

    cache.invalidate()
    assert cache.get() is None

    t[0] = 3.0
    cache.set("v2")
    t[0] = 5.0
    assert cache.get() == "v2"


def test_ttl_cache_overwrite_renews_ready_at() -> None:
    t = [0.0]
    cache = TtlCache(10.0, clock=lambda: t[0])

    cache.set("v1")  # ready_at = 10
    t[0] = 3.0
    cache.set("v2")  # ready_at renewed to 13
    t[0] = 12.0  # would be expired under v1's ready_at, fresh under v2's
    assert cache.get() == "v2"


def test_ttl_cache_ttl_property() -> None:
    assert TtlCache(42.0).ttl_s == 42.0
