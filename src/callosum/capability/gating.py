"""Routing-time consumers of capability profile findings.

The capability harness produces findings on disk; the router consumes
them at request time to gate cells out of routing decisions they would
fail. This module is the thin reader-and-decision layer between the two.

Currently exposed gates:

  * `at_scale_tool_call_fails(model_id)` — True when the cell's persisted
    profile contains a `tool_call_at_scale` finding with status="fail".
    The router uses this together with the request's prompt size to
    exclude the cell from tool-using requests whose context exceeds a
    threshold below which the cell is still trusted to emit structured
    tool_calls.

Caches profiles in-memory keyed by mtime so a fresh harness run is
picked up on the next routing decision without proxy restart. The
cache is best-effort: a stat() failure (deleted file, permission error)
falls back to "no profile" rather than raising — routing decisions
should not error out because the harness hasn't run yet.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from callosum.capability.profile import CapabilityProfile, load_profile, profile_path

# Default threshold for "this request is large enough to count as
# at-scale." 50K chars sits well above the small-turn case (a few
# thousand chars of system + user + tool messages) and well below the
# 80K-char at-scale probe size, so we exclude failing cells from the
# region where they're observed to break, without losing the small-
# context utility we know they retain.
#
# Chars (not tokens) because the router's PromptFeatures.tokens is a
# chars/3 heuristic that's wrong by ~30% on Codex CLI corpora — the
# raw text length is the more reliable signal at this granularity.
DEFAULT_AT_SCALE_CHARS_THRESHOLD: int = 50_000


@dataclass
class _CachedProfile:
    """Cache entry. Stores both the profile and the mtime that produced
    it so an mtime change invalidates."""

    profile: CapabilityProfile
    mtime_ns: int


_cache: dict[str, _CachedProfile] = {}
_cache_lock = threading.Lock()


def _profile_for(model_id: str) -> CapabilityProfile | None:
    """Return the cached profile for `model_id`, reloading from disk if
    the file's mtime has changed. None when no profile exists yet."""
    path = profile_path(model_id)
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        # No profile file (or no permissions). The harness hasn't run
        # for this cell yet — gate should be inert.
        with _cache_lock:
            _cache.pop(model_id, None)
        return None
    with _cache_lock:
        entry = _cache.get(model_id)
        if entry is not None and entry.mtime_ns == mtime_ns:
            return entry.profile
    # Pass the parent dir explicitly so the loader stays consistent with
    # whatever profile_path() resolved to. Without this, load_profile
    # would default to DEFAULT_PROFILE_DIR — fine in production, but
    # tests that redirect profile_path() to a tmpdir would read from
    # the wrong location.
    profile = load_profile(model_id, profile_dir=path.parent)
    with _cache_lock:
        _cache[model_id] = _CachedProfile(profile=profile, mtime_ns=mtime_ns)
    return profile


def at_scale_tool_call_fails(model_id: str) -> bool:
    """True iff the cell has a `tool_call_at_scale` finding with
    status="fail" on disk. False otherwise (no profile, no at-scale
    finding, or status in {pass, error, skipped}).

    The asymmetry matters: only "fail" gates. An "error" finding means
    the probe couldn't complete — we don't know how the cell behaves
    at scale, and routing should not punish that. A "skipped" finding
    means the small probe didn't pass either (the harness skips the
    expensive at-scale probe when the cheap one already failed); those
    cells are gated out by other capability checks already.
    """
    profile = _profile_for(model_id)
    if profile is None:
        return False
    finding = profile.findings.get("tool_call_at_scale")
    if finding is None:
        return False
    return finding.status == "fail"


def clear_cache() -> None:
    """Drop the in-memory cache. Test-only; production relies on
    mtime invalidation."""
    with _cache_lock:
        _cache.clear()
