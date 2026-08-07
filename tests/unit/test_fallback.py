"""Unit tests for the pure fallback decision logic in callosum.fallback.

fallback.py contains zero external dependencies and no I/O beyond
logging — its decision functions are pure mappings over
error_classifications and an available-model set. These tests pin
each decision boundary directly so a regression in the fallback policy
is caught without standing up the full dispatch path.
"""

from __future__ import annotations

import logging

import pytest

from callosum.fallback import (
    FallbackExecutor,
    should_attempt_fallback,
)

# ---------------------------------------------------------------------------
# should_retry_with_backoff
# ---------------------------------------------------------------------------


def test_should_retry_with_backoff_true_for_transient_classification() -> None:
    ex = FallbackExecutor()
    assert ex.should_retry_with_backoff({"b1": "transient"}) is True


@pytest.mark.parametrize("classification", ["rate_limited", "transient", "timeout"])
def test_should_retry_with_backoff_true_for_each_transient_kind(
    classification: str,
) -> None:
    ex = FallbackExecutor()
    assert ex.should_retry_with_backoff({"b1": classification}) is True


def test_should_retry_with_backoff_true_when_only_some_are_transient() -> None:
    ex = FallbackExecutor()
    assert ex.should_retry_with_backoff({"b1": "auth_invalid", "b2": "rate_limited"}) is True


def test_should_retry_with_backoff_false_when_none_transient() -> None:
    ex = FallbackExecutor()
    assert ex.should_retry_with_backoff({"b1": "auth_invalid", "b2": "unknown_model"}) is False


def test_should_retry_with_backoff_false_when_empty() -> None:
    ex = FallbackExecutor()
    assert ex.should_retry_with_backoff({}) is False


# ---------------------------------------------------------------------------
# should_try_fallback_model
# ---------------------------------------------------------------------------


def test_should_try_fallback_model_returns_next_in_chain_when_available() -> None:
    ex = FallbackExecutor()
    available = frozenset({"model-a0e8", "model-a0f8", "model-a0f7", "model-a0c9"})
    assert ex.should_try_fallback_model("model-a0e8", available) == "model-a0f8"
    assert ex.should_try_fallback_model("model-a0f8", available) == "model-a0f7"
    assert ex.should_try_fallback_model("model-a0f7", available) == "model-a0c9"


def test_should_try_fallback_model_returns_none_when_fallback_not_available() -> None:
    ex = FallbackExecutor()
    # model-a0f8 is the chain target for model-a0e8 but is absent from the pool.
    assert ex.should_try_fallback_model("model-a0e8", frozenset({"model-a0f7"})) is None


def test_should_try_fallback_model_returns_none_for_unknown_model() -> None:
    ex = FallbackExecutor()
    assert ex.should_try_fallback_model("model-a0c1", frozenset({"model-a0f8"})) is None


def test_should_try_fallback_model_returns_none_at_chain_floor() -> None:
    ex = FallbackExecutor()
    # model-a0c9 is the terminal node — no further fallback defined.
    assert ex.should_try_fallback_model("model-a0c9", frozenset({"model-a0c9", "model-a0f7"})) is None


# ---------------------------------------------------------------------------
# should_try_different_backend
# ---------------------------------------------------------------------------


def test_should_try_different_backend_false_with_one_backend() -> None:
    ex = FallbackExecutor()
    assert ex.should_try_different_backend({"b1": "stream_ended"}) is False


def test_should_try_different_backend_false_when_empty() -> None:
    ex = FallbackExecutor()
    assert ex.should_try_different_backend({}) is False


def test_should_try_different_backend_true_for_backend_specific_errors() -> None:
    ex = FallbackExecutor()
    assert ex.should_try_different_backend({"b1": "transient", "b2": "stream_ended"}) is True


@pytest.mark.parametrize("classification", ["stream_ended", "upstream_error", "auth_invalid"])
def test_should_try_different_backend_true_for_each_backend_specific_kind(
    classification: str,
) -> None:
    ex = FallbackExecutor()
    assert ex.should_try_different_backend({"b1": "transient", "b2": classification}) is True


def test_should_try_different_backend_false_when_many_but_none_backend_specific() -> None:
    ex = FallbackExecutor()
    assert ex.should_try_different_backend({"b1": "transient", "b2": "rate_limited", "b3": "timeout"}) is False


# ---------------------------------------------------------------------------
# should_attempt_fallback (module-level)
# ---------------------------------------------------------------------------


def test_should_attempt_fallback_true_when_any_backend_is_recoverable() -> None:
    assert should_attempt_fallback({"b1": "auth_invalid", "b2": "transient"}) is True


def test_should_attempt_fallback_false_when_all_unknown_model() -> None:
    assert should_attempt_fallback({"b1": "unknown_model", "b2": "unknown_model"}) is False


def test_should_attempt_fallback_false_when_all_auth_invalid() -> None:
    assert should_attempt_fallback({"b1": "auth_invalid", "b2": "auth_invalid"}) is False


def test_should_attempt_fallback_false_when_mixed_permanent_only() -> None:
    # Both permanent-unavailability kinds together still short-circuit.
    assert should_attempt_fallback({"b1": "unknown_model", "b2": "auth_invalid"}) is False


def test_should_attempt_fallback_false_when_empty() -> None:
    # `all([])` is vacuously True, so "all permanent" holds and fallback
    # is NOT attempted when there are no classifications to recover from.
    assert should_attempt_fallback({}) is False


# ---------------------------------------------------------------------------
# record_attempt / log_final_exhaustion (state + logging side effects)
# ---------------------------------------------------------------------------


def test_record_attempt_appends_with_duration_and_reason(caplog: pytest.LogCaptureFixture) -> None:
    ex = FallbackExecutor()
    with caplog.at_level(logging.INFO, logger="callosum.fallback"):
        ex.record_attempt(
            "retry_with_backoff",
            "success",
            reason="rate_limit_recovered",
            alt_model="model-a0f8",
            alt_backend="b2",
        )
    assert len(ex.attempts) == 1
    attempt = ex.attempts[0]
    assert attempt.strategy == "retry_with_backoff"
    assert attempt.result == "success"
    assert attempt.reason == "rate_limit_recovered"
    assert attempt.alt_model == "model-a0f8"
    assert attempt.alt_backend == "b2"
    assert attempt.duration_ms is not None and attempt.duration_ms >= 0.0
    # INFO log line carries the strategy + result + reason.
    assert any("retry_with_backoff" in rec.message and "success" in rec.message for rec in caplog.records)


def test_record_attempt_defaults_reason_to_no_reason_in_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ex = FallbackExecutor()
    with caplog.at_level(logging.INFO, logger="callosum.fallback"):
        ex.record_attempt("fallback_to_cheaper_model", "failed")
    assert ex.attempts[0].reason is None
    assert any("no reason" in rec.message for rec in caplog.records)


def test_log_final_exhaustion_emits_error_with_strategies_and_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ex = FallbackExecutor()
    ex.record_attempt("retry_with_backoff", "failed", reason="transient_error")
    ex.record_attempt("fallback_to_cheaper_model", "failed", reason="no_model")
    with caplog.at_level(logging.ERROR, logger="callosum.fallback"):
        ex.log_final_exhaustion("model-a0e8", {"b1": "stream_ended", "b2": "auth_invalid"})
    errors = [rec for rec in caplog.records if rec.levelno == logging.ERROR]
    assert errors, "expected an ERROR log for final exhaustion"
    msg = errors[-1].message
    assert "model-a0e8" in msg
    assert "2 strategies" in msg
    assert "retry_with_backoff" in msg
    assert "fallback_to_cheaper_model" in msg
    assert "b1: stream_ended" in msg
    assert "b2: auth_invalid" in msg
