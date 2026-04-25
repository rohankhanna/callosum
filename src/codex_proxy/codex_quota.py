from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CodexQuotaSnapshot:
    """Per-account quota state extracted from upstream `/responses` headers.

    Every response from the ChatGPT backend includes `x-codex-*` headers that
    report the account's 5-hour ("primary") and weekly ("secondary") usage
    percentages, reset times, and plan metadata. This dataclass captures a
    single such snapshot.

    Differences between two snapshots on the same backend describe the cost,
    in Codex-quota-percent, of whatever requests happened between them.
    """

    plan_type: str | None
    active_limit: str | None
    primary_used_percent: int | None
    secondary_used_percent: int | None
    primary_window_minutes: int | None
    secondary_window_minutes: int | None
    primary_reset_at: int | None
    secondary_reset_at: int | None
    primary_reset_after_seconds: int | None
    secondary_reset_after_seconds: int | None
    primary_over_secondary_limit_percent: int | None
    credits_balance: str | None
    credits_has_credits: bool | None
    credits_unlimited: bool | None
    observed_at: float


def parse_codex_headers(headers: Mapping[str, str]) -> CodexQuotaSnapshot | None:
    """Build a snapshot from an upstream response's headers.

    Returns None when no Codex headers are present (i.e. the response did not
    come from the ChatGPT backend). When at least one `x-codex-*` field is
    present, a snapshot is returned with nulls for any fields that were absent
    or malformed — the caller decides how to handle partial data.
    """
    lowered = {k.lower(): v for k, v in headers.items()}
    if not any(k.startswith("x-codex-") for k in lowered):
        return None
    return CodexQuotaSnapshot(
        plan_type=_str(lowered, "x-codex-plan-type"),
        active_limit=_str(lowered, "x-codex-active-limit"),
        primary_used_percent=_int(lowered, "x-codex-primary-used-percent"),
        secondary_used_percent=_int(lowered, "x-codex-secondary-used-percent"),
        primary_window_minutes=_int(lowered, "x-codex-primary-window-minutes"),
        secondary_window_minutes=_int(lowered, "x-codex-secondary-window-minutes"),
        primary_reset_at=_int(lowered, "x-codex-primary-reset-at"),
        secondary_reset_at=_int(lowered, "x-codex-secondary-reset-at"),
        primary_reset_after_seconds=_int(lowered, "x-codex-primary-reset-after-seconds"),
        secondary_reset_after_seconds=_int(lowered, "x-codex-secondary-reset-after-seconds"),
        primary_over_secondary_limit_percent=_int(
            lowered, "x-codex-primary-over-secondary-limit-percent"
        ),
        credits_balance=_str(lowered, "x-codex-credits-balance"),
        credits_has_credits=_bool(lowered, "x-codex-credits-has-credits"),
        credits_unlimited=_bool(lowered, "x-codex-credits-unlimited"),
        observed_at=time.time(),
    )


def _str(h: Mapping[str, str], key: str) -> str | None:
    value = h.get(key)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _int(h: Mapping[str, str], key: str) -> int | None:
    raw = _str(h, key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _bool(h: Mapping[str, str], key: str) -> bool | None:
    raw = _str(h, key)
    if raw is None:
        return None
    if raw.lower() in ("true", "1", "yes"):
        return True
    if raw.lower() in ("false", "0", "no"):
        return False
    return None
