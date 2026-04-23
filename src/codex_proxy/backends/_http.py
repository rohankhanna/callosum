from __future__ import annotations

import httpx

from codex_proxy.errors import BackendError, classify_http_status

DEFAULT_COOLDOWN_S = 60.0


def error_from_response(response: httpx.Response) -> BackendError:
    """Build a classified BackendError from an upstream non-2xx response."""
    classification = classify_http_status(response.status_code)
    retry_after = response.headers.get("retry-after")
    retry_after_s: float | None = None
    if retry_after is not None:
        try:
            retry_after_s = float(retry_after)
        except ValueError:
            retry_after_s = None
    return BackendError(
        classification=classification,
        status_code=response.status_code,
        retry_after_s=retry_after_s,
        message=f"upstream {response.status_code}",
    )
