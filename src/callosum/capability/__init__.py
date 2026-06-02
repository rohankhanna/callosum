"""Model-capability evaluation library.

callosum's runtime — not just pytest — drives the harness. This
package is the callable surface that both code paths (the background
auto-trigger in app.py and the pytest model_probe suite) call into.

Layout:

  profile.py       — CapabilityProfile dataclass + JSON persistence
  request_shapes.py — Codex-CLI-shape request bodies for probes
  runner.py        — runs a sequence of dimension probes against a cell
  dimensions/      — one module per capability dimension (tool_call_shape,
                     tool_call_at_scale, ...). Each exports a `probe()`
                     coroutine returning a DimensionFinding.

Why the library lives under src/callosum/ rather than tests/:
  * callosum's lifespan startup needs to import it (background scheduler
    fires the harness on new-cell detection)
  * keeping the harness pytest-only would mean operators have to invoke
    it manually forever — the "callosum in the loop, not the operator"
    principle requires the library be callable from the proxy itself
  * tests/ retains a thin wrapper that calls into this library so the
    pytest-driven workflow still works for ad-hoc operator runs
"""

from callosum.capability.profile import (
    CapabilityProfile,
    DimensionFinding,
    FindingStatus,
    load_profile,
    profile_path,
    save_profile,
)

__all__ = [
    "CapabilityProfile",
    "DimensionFinding",
    "FindingStatus",
    "load_profile",
    "profile_path",
    "save_profile",
]
