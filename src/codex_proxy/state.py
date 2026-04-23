from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from codex_proxy.backend import UsageSnapshot


class StateStore:
    """Per-backend on-disk cache for UsageSnapshot so cooldowns survive restarts."""

    def __init__(self, base_dir: Path) -> None:
        self._usage_dir = base_dir / "usage"

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
