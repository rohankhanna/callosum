"""Compat shim — the harness library lives in src/callosum/capability/.

Older test code under tests/model_capability/ imports profile pieces
from this module. Re-export from the canonical location so we don't
have to rewrite every import site. New code should import directly
from `callosum.capability`.
"""

from __future__ import annotations

from callosum.capability.profile import (  # noqa: F401
    CapabilityProfile,
    DimensionFinding,
    FindingStatus,
    load_profile,
    profile_path,
    save_profile,
)
