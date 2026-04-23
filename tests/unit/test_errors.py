from __future__ import annotations

import pytest

from codex_proxy.errors import RETRYABLE, BackendError, classify_http_status


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, "client_error"),
        (401, "auth_invalid"),
        (403, "auth_invalid"),
        (404, "unknown_model"),
        (429, "rate_limited"),
        (500, "transient"),
        (502, "transient"),
        (503, "transient"),
        (504, "transient"),
        (418, "transient"),
    ],
)
def test_classify_http_status(status: int, expected: str) -> None:
    assert classify_http_status(status) == expected


def test_retryable_set_excludes_client_error() -> None:
    assert "client_error" not in RETRYABLE
    assert "auth_invalid" in RETRYABLE
    assert "unknown_model" in RETRYABLE
    assert "rate_limited" in RETRYABLE
    assert "transient" in RETRYABLE


def test_backend_error_carries_metadata() -> None:
    err = BackendError(
        classification="rate_limited",
        status_code=429,
        retry_after_s=12.5,
        message="slow down",
    )
    assert err.classification == "rate_limited"
    assert err.status_code == 429
    assert err.retry_after_s == 12.5
    assert str(err) == "slow down"
