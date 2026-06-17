from __future__ import annotations

import contextlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from callosum.backend import UsageSnapshot
from callosum.codex_quota import CodexQuotaSnapshot


class StateStore:
    """Per-backend on-disk cache for UsageSnapshot so cooldowns survive restarts.

    Also tracks model release cycle and proxy startup for adaptive learned model scheduling.
    """

    def __init__(self, base_dir: Path) -> None:
        self._usage_dir = base_dir / "usage"
        self._quota_dir = base_dir / "quota"
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

    def load_quota(self, backend_id: str) -> CodexQuotaSnapshot | None:
        """Reload the last-known Codex quota snapshot so /status survives restarts.

        The live value lives in-memory on the backend and is only refreshed by a
        routed/probed Codex response. Persisting it here means an operator polling
        /status sees the last-known quota even after a restart with no traffic.
        """
        path = self._quota_dir / f"{backend_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return CodexQuotaSnapshot(
                plan_type=data.get("plan_type"),
                active_limit=data.get("active_limit"),
                five_hourly_used_percent=data.get("five_hourly_used_percent"),
                weekly_used_percent=data.get("weekly_used_percent"),
                five_hourly_window_minutes=data.get("five_hourly_window_minutes"),
                weekly_window_minutes=data.get("weekly_window_minutes"),
                five_hourly_reset_at=data.get("five_hourly_reset_at"),
                weekly_reset_at=data.get("weekly_reset_at"),
                five_hourly_reset_after_seconds=data.get("five_hourly_reset_after_seconds"),
                weekly_reset_after_seconds=data.get("weekly_reset_after_seconds"),
                five_hourly_over_weekly_limit_percent=data.get("five_hourly_over_weekly_limit_percent"),
                credits_balance=data.get("credits_balance"),
                credits_has_credits=data.get("credits_has_credits"),
                credits_unlimited=data.get("credits_unlimited"),
                observed_at=float(data.get("observed_at", 0.0)),
            )
        except (TypeError, ValueError):
            return None

    def save_quota(self, backend_id: str, snapshot: CodexQuotaSnapshot) -> None:
        self._quota_dir.mkdir(parents=True, exist_ok=True)
        path = self._quota_dir / f"{backend_id}.json"
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
            with contextlib.suppress(OSError, json.JSONDecodeError):
                existing = json.loads(self._timing_path.read_text())
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
            with contextlib.suppress(OSError, json.JSONDecodeError):
                existing = json.loads(self._timing_path.read_text())
        existing["model_release_timestamp"] = ts
        tmp = self._timing_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(existing))
        os.replace(tmp, self._timing_path)
