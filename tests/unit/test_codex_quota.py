from __future__ import annotations

from codex_proxy.codex_quota import parse_codex_headers


def test_returns_none_when_no_codex_headers_present() -> None:
    assert parse_codex_headers({"content-type": "application/json"}) is None


def test_parses_full_header_set_from_real_response() -> None:
    headers = {
        "x-codex-active-limit": "premium",
        "x-codex-plan-type": "plus",
        "x-codex-primary-used-percent": "1",
        "x-codex-secondary-used-percent": "53",
        "x-codex-primary-window-minutes": "300",
        "x-codex-secondary-window-minutes": "10080",
        "x-codex-primary-over-secondary-limit-percent": "0",
        "x-codex-primary-reset-after-seconds": "6664",
        "x-codex-secondary-reset-after-seconds": "389495",
        "x-codex-primary-reset-at": "1777057648",
        "x-codex-secondary-reset-at": "1777440478",
        "x-codex-credits-balance": "",
        "x-codex-credits-has-credits": "False",
        "x-codex-credits-unlimited": "False",
    }
    snap = parse_codex_headers(headers)
    assert snap is not None
    assert snap.plan_type == "plus"
    assert snap.active_limit == "premium"
    assert snap.five_hourly_used_percent == 1
    assert snap.weekly_used_percent == 53
    assert snap.five_hourly_window_minutes == 300
    assert snap.weekly_window_minutes == 10080
    assert snap.five_hourly_reset_at == 1777057648
    assert snap.weekly_reset_at == 1777440478
    assert snap.five_hourly_over_weekly_limit_percent == 0
    assert snap.credits_has_credits is False
    assert snap.credits_unlimited is False
    assert snap.credits_balance is None  # empty string becomes None
    assert snap.observed_at > 0


def test_partial_headers_fill_in_nulls() -> None:
    snap = parse_codex_headers({"x-codex-primary-used-percent": "42"})
    assert snap is not None
    assert snap.five_hourly_used_percent == 42
    assert snap.weekly_used_percent is None
    assert snap.plan_type is None


def test_malformed_integers_become_null() -> None:
    snap = parse_codex_headers({"x-codex-primary-used-percent": "not-a-number"})
    assert snap is not None
    assert snap.five_hourly_used_percent is None


def test_case_insensitive_header_names() -> None:
    snap = parse_codex_headers({"X-Codex-Primary-Used-Percent": "7"})
    assert snap is not None
    assert snap.five_hourly_used_percent == 7


def test_bool_parses_true_false_and_numeric() -> None:
    snap = parse_codex_headers(
        {
            "x-codex-credits-has-credits": "true",
            "x-codex-credits-unlimited": "1",
        }
    )
    assert snap is not None
    assert snap.credits_has_credits is True
    assert snap.credits_unlimited is True
