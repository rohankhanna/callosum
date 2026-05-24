from __future__ import annotations

import httpx

from callosum.errors import BackendError, classify_http_status

DEFAULT_COOLDOWN_S = 60.0


def error_from_response(response: httpx.Response) -> BackendError:
    """Build a classified BackendError from an upstream non-2xx response.

    Includes a truncated copy of the upstream response body in the message
    so operators can distinguish "model retired" from "rate limited" from
    "auth invalid" etc. without having to enable verbose logging.
    """
    classification = classify_http_status(response.status_code)
    retry_after = response.headers.get("retry-after")
    retry_after_s: float | None = None
    if retry_after is not None:
        try:
            retry_after_s = float(retry_after)
        except ValueError:
            retry_after_s = None
    detail = _extract_response_detail(response)
    message = f"upstream {response.status_code}"
    if detail:
        message = f"{message}: {detail}"
    return BackendError(
        classification=classification,
        status_code=response.status_code,
        retry_after_s=retry_after_s,
        message=message,
    )


def _extract_response_detail(response: httpx.Response) -> str:
    """Pull a human-readable message out of an upstream error body.

    Tries JSON first (most upstreams return `{"error": {"message": "..."}}`
    or a flat `{"message": "..."}`), falls back to the raw text. Capped at
    400 chars to keep log lines manageable.
    """
    try:
        body = response.text
    except Exception:
        return ""
    if not body:
        return ""
    try:
        import json

        parsed = json.loads(body)
    except (ValueError, TypeError):
        return body[:400]
    if isinstance(parsed, dict):
        # Common shapes: {error: {message: "..."}}, {error: "..."}, {message: "..."}
        err = parsed.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str) and msg:
                return msg[:400]
            code = err.get("code")
            if isinstance(code, str) and code:
                return code[:400]
        if isinstance(err, str) and err:
            return err[:400]
        msg = parsed.get("message")
        if isinstance(msg, str) and msg:
            return msg[:400]
        detail = parsed.get("detail")
        if isinstance(detail, str) and detail:
            return detail[:400]
    return body[:400]
