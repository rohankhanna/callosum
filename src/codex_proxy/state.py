from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from codex_proxy.backend import UsageSnapshot


class StateStore:
    """Per-backend on-disk cache for UsageSnapshot so cooldowns survive restarts.

    Also tracks model release cycle and proxy startup for adaptive learned model scheduling.
    """

    def __init__(self, base_dir: Path) -> None:
        self._usage_dir = base_dir / "usage"
        self._timing_path = base_dir / "timing.json"

    def load_usage(self, backend_id: str) -> UsageSnapshot | None:
        path = self._usage_dir / f"{backend_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return UsageSnapshot(
                remaining_fraction=data.get("remaining_fraction"),
                cooldown_until_ts=data.get("cooldown_until_ts"),
                weekly_exhausted=bool(data.get("weekly_exhausted", False)),
                probed_at_ts=float(data.get("probed_at_ts", 0.0)),
            )
        except (TypeError, ValueError):
            return None

    def save_usage(self, backend_id: str, snapshot: UsageSnapshot) -> None:
        self._usage_dir.mkdir(parents=True, exist_ok=True)
        path = self._usage_dir / f"{backend_id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(snapshot)))
        os.replace(tmp, path)

    def get_proxy_startup_timestamp(self) -> float | None:
        """Return the timestamp when the proxy started (first run or restart).

        Used to calculate days-since-startup for the initial learned model ramp.
        """
        if not self._timing_path.exists():
            return None
        try:
            data = json.loads(self._timing_path.read_text())
            return float(data.get("startup_timestamp"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def set_proxy_startup_timestamp(self, ts: float) -> None:
        """Record the proxy startup timestamp (called once on lifespan startup).

        Persists across restarts to track the initial 90-day ramp.
        """
        self._timing_path.parent.mkdir(parents=True, exist_ok=True)
        # Read existing data to preserve model_release_timestamp
        existing = {}
        if self._timing_path.exists():
            try:
                existing = json.loads(self._timing_path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        existing["startup_timestamp"] = ts
        tmp = self._timing_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(existing))
        os.replace(tmp, self._timing_path)

    def get_model_release_timestamp(self) -> float | None:
        """Return the timestamp (seconds since epoch) of the last detected model release.

        Used to calculate days-since-release for model-specific learned model ramp.
        """
        if not self._timing_path.exists():
            return None
        try:
            data = json.loads(self._timing_path.read_text())
            return float(data.get("model_release_timestamp"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def set_model_release_timestamp(self, ts: float) -> None:
        """Record a new model release detection.

        Called when advertised_models changes. Resets the learned model ramp.
        """
        self._timing_path.parent.mkdir(parents=True, exist_ok=True)
        # Read existing data to preserve startup_timestamp
        existing = {}
        if self._timing_path.exists():
            try:
                existing = json.loads(self._timing_path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        existing["model_release_timestamp"] = ts
        tmp = self._timing_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(existing))
        os.replace(tmp, self._timing_path)
