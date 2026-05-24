"""Intelligent fallback strategies for when primary backends are exhausted.

When all primary routing options fail, the fallback system tries increasingly
intelligent alternatives before returning an error to the user. Each attempt is
logged with data collection to enable continuous improvement via feedback.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("callosum.fallback")


def _utc_timestamp() -> str:
    """Return current time in ISO 8601 UTC format with Z suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class FallbackAttempt:
    """Record of a single fallback strategy attempt."""

    strategy: str  # e.g., "retry_with_backoff", "fallback_to_cheaper_model"
    timestamp: float
    result: str  # "success", "failed", "retried"
    reason: str | None = None  # e.g., "rate_limit_recovered", "transient_error"
    alt_model: str | None = None  # If tried alternative model
    alt_backend: str | None = None  # If tried alternative backend
    duration_ms: float | None = None


class FallbackExecutor:
    """Executes intelligent fallback strategies when primary routing fails.

    Strategies are tried in priority order. Only after all strategies are exhausted
    does the system return an error to the user. Each attempt is logged for
    feedback-driven continuous improvement.
    """

    def __init__(self) -> None:
        self.attempts: list[FallbackAttempt] = []
        self.start_time = time.time()

    def record_attempt(
        self,
        strategy: str,
        result: str,
        reason: str | None = None,
        alt_model: str | None = None,
        alt_backend: str | None = None,
    ) -> None:
        """Record a fallback attempt for analysis and learning."""
        now = time.time()
        duration_ms = (now - self.start_time) * 1000
        attempt = FallbackAttempt(
            strategy=strategy,
            timestamp=now,
            result=result,
            reason=reason,
            alt_model=alt_model,
            alt_backend=alt_backend,
            duration_ms=duration_ms,
        )
        self.attempts.append(attempt)
        logger.info(
            f"[{_utc_timestamp()}] "
            f"fallback strategy '{strategy}': {result} "
            f"({reason or 'no reason'}) in {duration_ms:.0f}ms"
        )

    def should_retry_with_backoff(
        self, error_classifications: dict[str, str]
    ) -> bool:
        """Decide if we should retry with exponential backoff.

        Useful for transient errors like rate limits that may recover.
        """
        # If majority of failures are transient (rate_limited, timeout, transient),
        # backoff is worth trying
        transient_count = sum(
            1
            for classification in error_classifications.values()
            if classification in {"rate_limited", "transient", "timeout"}
        )
        return transient_count > 0 and len(error_classifications) > 0

    def should_try_fallback_model(
        self, model: str, available_models: frozenset[str]
    ) -> str | None:
        """Choose a fallback model if primary is exhausted.

        Strategy: try next-cheaper/next-slower model on the same backend.
        Returns the fallback model name, or None if no fallback available.
        """
        # Model hierarchy: model-a0e8 (best/most expensive) → model-a0f8 → model-a0f7 (cheaper)
        fallback_chain = {
            "model-a0e8": "model-a0f8",
            "model-a0f8": "model-a0f7",
            "model-a0f7": "model-a0c9",
        }
        fallback = fallback_chain.get(model)
        if fallback and fallback in available_models:
            return fallback
        return None

    def should_try_different_backend(
        self, error_classifications: dict[str, str]
    ) -> bool:
        """Decide if we should try a completely different backend.

        Useful when one backend is having persistent issues but others might work.
        """
        # If only one backend is available, no point trying different one
        if len(error_classifications) <= 1:
            return False
        # If errors are backend-specific (stream_ended, upstream_error), rotate backends
        backend_specific = sum(
            1
            for classification in error_classifications.values()
            if classification in {"stream_ended", "upstream_error", "auth_invalid"}
        )
        return backend_specific > 0

    def log_final_exhaustion(
        self, model: str, error_classifications: dict[str, str]
    ) -> None:
        """Log when all strategies are exhausted."""
        ts = _utc_timestamp()
        total_ms = (time.time() - self.start_time) * 1000
        strategies_tried = [a.strategy for a in self.attempts]

        logger.error(
            f"[{ts}] fallback exhausted for model '{model}'. "
            f"Tried {len(strategies_tried)} strategies in {total_ms:.0f}ms: "
            f"{', '.join(strategies_tried)}. "
            f"Backend failures: {'; '.join(f'{b}: {c}' for b, c in error_classifications.items())}"
        )


def should_attempt_fallback(
    error_classifications: dict[str, str],
) -> bool:
    """Quick check: is fallback strategy worth attempting?

    Returns False only if errors suggest permanent unavailability.
    """
    # If all backends return "unknown_model" or "auth_invalid", don't fallback
    if all(
        c in {"unknown_model", "auth_invalid"} for c in error_classifications.values()
    ):
        return False
    return True
