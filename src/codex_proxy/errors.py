from __future__ import annotations

from typing import Literal

ErrorClass = Literal[
    "client_error",
    "auth_invalid",
    "unknown_model",
    "rate_limited",
    "transient",
]

RETRYABLE: frozenset[ErrorClass] = frozenset(
    {"auth_invalid", "unknown_model", "rate_limited", "transient"}
)


class BackendError(Exception):
    """Raised by Backend implementations to signal a classified upstream failure."""

    def __init__(
        self,
        *,
        classification: ErrorClass,
        status_code: int | None = None,
        retry_after_s: float | None = None,
        message: str = "",
    ) -> None:
        super().__init__(message or classification)
        self.classification: ErrorClass = classification
        self.status_code = status_code
        self.retry_after_s = retry_after_s
        self.message = message


def classify_http_status(status: int) -> ErrorClass:
    if status == 400:
        return "client_error"
    if status in (401, 403):
        return "auth_invalid"
    if status == 404:
        return "unknown_model"
    if status == 429:
        return "rate_limited"
    if 500 <= status < 600:
        return "transient"
    return "transient"
