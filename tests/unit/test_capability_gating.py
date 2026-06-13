"""Tests for callosum.capability.gating — disk → routing decision.

Exercises the reader the router consumes to decide whether to gate a
cell out of large-context tool requests. The reader caches profiles in
memory keyed by mtime; these tests verify the cache is correct (hits
when nothing has changed, invalidates when the file changes) and that
the four finding statuses (pass/fail/error/skipped) map to the right
gate decision.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from callosum.capability import gating
from callosum.capability.profile import (
    CapabilityProfile,
    DimensionFinding,
    save_profile,
)


@pytest.fixture(autouse=True)
def _reset_cache_and_redirect(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Point the gating reader at a tmp profile dir and drop its
    in-memory cache before each test. Without this every test would
    pollute the next via the module-global cache."""
    monkeypatch.setattr(
        gating,
        "profile_path",
        lambda model_id, profile_dir=None: tmp_path / f"{model_id}.json",
    )
    gating.clear_cache()
    yield
    gating.clear_cache()


def _write_finding(tmp_path: Path, model_id: str, status: str) -> None:
    """Helper: persist a profile with one tool_call_at_scale finding."""
    profile = CapabilityProfile(model_id=model_id)
    profile.upsert(
        DimensionFinding(
            dimension="tool_call_at_scale",
            status=status,  # type: ignore[arg-type]
            summary=f"test fixture: status={status}",
        )
    )
    save_profile(profile, profile_dir=tmp_path)


def test_no_profile_means_gate_inert(tmp_path: Path) -> None:
    """A cell with no profile on disk is unprobed; the gate must not
    punish it. This is the steady-state for any newly-arrived cell
    before the background harness has run."""
    assert gating.at_scale_tool_call_fails("never-probed") is False


def test_fail_finding_triggers_gate(tmp_path: Path) -> None:
    _write_finding(tmp_path, "broken-at-scale", "fail")
    assert gating.at_scale_tool_call_fails("broken-at-scale") is True


def test_pass_finding_does_not_trigger_gate(tmp_path: Path) -> None:
    _write_finding(tmp_path, "works-at-scale", "pass")
    assert gating.at_scale_tool_call_fails("works-at-scale") is False


def test_error_finding_does_not_trigger_gate(tmp_path: Path) -> None:
    """An `error` finding means the probe couldn't complete — we have
    no information. Routing must not punish "unknown" the same way it
    punishes "known broken"; otherwise transient probe failures (e.g.
    ollama temporarily down) would silently exclude cells from routing
    until the next harness run."""
    _write_finding(tmp_path, "probe-erred", "error")
    assert gating.at_scale_tool_call_fails("probe-erred") is False


def test_skipped_finding_does_not_trigger_gate(tmp_path: Path) -> None:
    """`skipped` means the cheaper small probe failed first, so we
    never ran the at-scale probe. Those cells are already excluded by
    the supports_tools light-probe gate; the at-scale gate stays inert
    rather than double-punishing them."""
    _write_finding(tmp_path, "small-probe-failed", "skipped")
    assert gating.at_scale_tool_call_fails("small-probe-failed") is False


def test_mtime_invalidation_picks_up_fresh_finding(tmp_path: Path) -> None:
    """Write a passing finding, read it (caches), then overwrite with a
    failing finding. The next read must reflect the new state — this is
    what lets a fresh harness sweep change routing behavior without
    requiring a proxy restart."""
    model = "evolving-cell"
    _write_finding(tmp_path, model, "pass")
    assert gating.at_scale_tool_call_fails(model) is False

    # Bump mtime explicitly. Some filesystems only update mtime at
    # second granularity, and the two writes can happen within the
    # same nanosecond on fast storage; force a fresh mtime so the
    # invalidation path is deterministically exercised.
    _write_finding(tmp_path, model, "fail")
    later = time.time() + 1
    os.utime(tmp_path / f"{model}.json", (later, later))

    assert gating.at_scale_tool_call_fails(model) is True


def test_corrupt_profile_does_not_raise(tmp_path: Path) -> None:
    """A malformed JSON file on disk must not crash routing. The
    reader's underlying load_profile is defensive (returns an empty
    profile on parse error); confirm the gate decision falls through
    to the safe "no finding" branch."""
    path = tmp_path / "corrupt.json"
    path.write_text("{ not valid json")
    assert gating.at_scale_tool_call_fails("corrupt") is False


def test_finding_without_at_scale_dimension(tmp_path: Path) -> None:
    """A profile that has other dimensions but not tool_call_at_scale
    must not trigger the gate — the finding's absence isn't a failure."""
    profile = CapabilityProfile(model_id="partial-profile")
    profile.upsert(
        DimensionFinding(
            dimension="tool_call_shape",
            status="pass",
            summary="small probe passed; at-scale not yet run",
        )
    )
    save_profile(profile, profile_dir=tmp_path)
    assert gating.at_scale_tool_call_fails("partial-profile") is False


def test_cache_is_used_when_mtime_unchanged(tmp_path: Path, monkeypatch) -> None:
    """Reading the same model twice without disk change should hit the
    in-memory cache. Verify by counting how many times load_profile is
    called — second read must not re-read from disk."""
    _write_finding(tmp_path, "cached", "fail")
    calls: list[str] = []
    real_load = gating.load_profile

    def counting_load(model_id: str, *, profile_dir=None):
        calls.append(model_id)
        return real_load(model_id, profile_dir=profile_dir)

    monkeypatch.setattr(gating, "load_profile", counting_load)

    assert gating.at_scale_tool_call_fails("cached") is True
    assert gating.at_scale_tool_call_fails("cached") is True
    # First read populates the cache; second uses it. Exactly one disk
    # load total.
    assert len(calls) == 1


def test_deleted_profile_evicts_cache_entry(tmp_path: Path) -> None:
    """After a probe result is read into the cache, deleting the
    underlying file should evict — otherwise stale findings would
    persist after an operator clears the profile dir."""
    _write_finding(tmp_path, "transient", "fail")
    assert gating.at_scale_tool_call_fails("transient") is True

    (tmp_path / "transient.json").unlink()
    assert gating.at_scale_tool_call_fails("transient") is False


def test_default_threshold_is_above_small_turns_below_probe_size() -> None:
    """The threshold must sit in the gap between typical small turns
    (a few thousand chars) and the at-scale probe size (80K chars).
    Catches accidental future changes that would either over-gate
    small requests or under-gate the region where breakage is observed."""
    assert gating.DEFAULT_AT_SCALE_CHARS_THRESHOLD > 5_000
    assert gating.DEFAULT_AT_SCALE_CHARS_THRESHOLD < 80_000
