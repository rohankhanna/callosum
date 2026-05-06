from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from codex_proxy.backend import UsageSnapshot


class StateStore:
    """Per-backend on-disk cache for UsageSnapshot so cooldowns survive restarts.

    Also tracks model release cycle for adaptive exploration scheduling.
    """

    def __init__(self, base_dir: Path) -> None:
        self._usage_dir = base_dir / "usage"
        self._model_release_path = base_dir / "model_release.json"

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

    def get_model_release_timestamp(self) -> float | None:
        """Return the timestamp (seconds since epoch) of the last detected model release.

        Used to calculate days-since-release for adaptive exploration scheduling.
        """
        if not self._model_release_path.exists():
            return None
        try:
            data = json.loads(self._model_release_path.read_text())
            return float(data.get("release_timestamp"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None

    def set_model_release_timestamp(self, ts: float) -> None:
        """Record a new model release detection.

        Called when advertised_models changes.
        """
        self._model_release_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._model_release_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"release_timestamp": ts}))
        os.replace(tmp, self._model_release_path)
