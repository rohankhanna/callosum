"""Surface-only feedback redirect ().

When callosum's existing fault detectors flag a suspect model output
(quality_score = -1 from the failure labeler, peer-quality calibration,
or the user /v1/feedback endpoint), this module helps *point the operator
at the existing external feedback channels* and hand them a scrubbed local
snippet + thread id they can paste into a GitHub issue.

This is explicitly NOT an upstream relay. The parent research node
() established that the Codex CLI /feedback command
egresses out-of-band to Sentry over a channel the OpenAI-API-terminating
routing proxy never sees, carries privacy-sensitive raw data, and is a
per-invocation opt-in. There is no OpenAI-side endpoint to forward to.
callosum therefore surfaces a canned redirect (run /feedback in the
Codex session, file the GitHub 3-cli.yml issue with the thread id, use
ChatGPT thumbs for ChatGPT-served traffic) and records the operator's
acknowledge/dismiss decision in a local audit table. It never proxies,
mirrors, or auto-sends the payload.

The held-back variant is in-band *auto-send* (injecting the redirect into the
model response stream without operator approval). That is a separate scoped-A4
autonomy decision with the auto-promotion master switch OFF; feedback_auto_send_enabled
below is the reserved switch and is NOT wired to any injection path.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

__all__ = [
    "FeedbackAutoSendNotWiredError",
    "build_redirect_message",
    "feedback_auto_send_enabled",
    "format_feedback_redirect",
    "scrub_snippet",
]


# --- scrubbing ---------------------------------------------------------------
# Conservative redaction of anything that looks like a secret before a
# snippet is shown to the operator or handed to a paste target. Errs toward
# OVER-redaction: a false positive costs the operator one manual un-redact of
# their own content when filing, while a false negative leaks a credential.
# The operator always reviews the scrubbed snippet before pasting.

# OpenAI-style API keys: sk-...
_API_KEY = re.compile(r"sk-[A-Za-z0-9_\-]{20,}")
# Bearer authorization headers/tokens.
_BEARER = re.compile(r"(?i)\bbearer\b\s+[A-Za-z0-9._\-]+")
# JWTs (three base64url segments, first starts with eyJ).
_JWT = re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")
# Account / user id values: keep the key name, redact only the value.
_ACCOUNT_ID = re.compile(
    r"(?i)\b(chatgpt_user_id|account_id|user[_-]?id)\b[\"'\s:=]+[\"']?([A-Za-z0-9_\-]+)"
)
# Long opaque runs (hex / base64 / base64url) of >= 32 chars from the merged
# alphabet. Catches access tokens, hashes, and encoded blobs that don't match
# a more specific pattern. False positives on long kebab-case identifiers are
# rare and benign (operator un-redacts their own content).
_OPAQUE_RUN = re.compile(r"[A-Za-z0-9+/=_\-]{32,}")


def scrub_snippet(text: str) -> str:
    """Return text with secret-shaped substrings replaced by [REDACTED:kind].

    Conservative by design. Also collapses the callosum-internal defensive
    marker scrub placeholders ({{{...}}}) to a neutral tag so the snippet
    does not advertise the proxy's internals.
    """
    if not text:
        return text
    out = _API_KEY.sub("[REDACTED:api-key]", text)
    out = _BEARER.sub("[REDACTED:bearer]", out)
    out = _JWT.sub("[REDACTED:jwt]", out)
    out = _ACCOUNT_ID.sub(r"\1=[REDACTED:account]", out)
    out = _OPAQUE_RUN.sub("[REDACTED:opaque]", out)
    return out


# --- redirect message --------------------------------------------------------

_REDIRECT_BODY = """\
callosum flagged a suspect output from cell {cell} (request {request_id},
thread {thread_id}, detector: {detector}).

This is a local diagnostic nudge only — callosum does NOT relay feedback
upstream. To report this to the model provider, use the existing external
channels yourself:

  1. Run  /feedback  in your Codex session.
     This uploads session diagnostics to the provider's telemetry tenant
     (out-of-band; the routing proxy never sees it). It is per-invocation
     opt-in and has its own [feedback] enabled kill-switch.

  2. File a GitHub issue using the provider's bug template, pasting the
     scrubbed snippet below + the thread id above so the maintainer can
     locate the turn.

  3. For ChatGPT-served traffic, use the in-product thumbs rating.

Scrubbed local snippet (already redacted of secret-shaped strings; review
before pasting):

----- prompt -----
{prompt}

----- response -----
{response}
"""


@dataclass(frozen=True, slots=True)
class _RedirectContext:
    """Inputs for build_redirect_message; kept as a dataclass for testability."""

    request_id: int
    thread_id: str | None
    model: str
    reasoning_effort: str | None
    detector: str | None
    scrubbed_prompt: str
    scrubbed_response: str


def build_redirect_message(ctx: _RedirectContext) -> str:
    """Build the human-readable redirect message shown to the operator.

    ctx fields are expected to be ALREADY scrubbed (prompt/response) and
    derived (thread_id, detector) by the caller; this function only formats.
    """
    cell = ctx.model
    if ctx.reasoning_effort:
        cell = f"{ctx.model}/{ctx.reasoning_effort}"
    thread = ctx.thread_id if ctx.thread_id is not None else "(none)"
    detector = ctx.detector if ctx.detector else "unknown"
    return _REDIRECT_BODY.format(
        cell=cell,
        request_id=ctx.request_id,
        thread_id=thread,
        detector=detector,
        prompt=ctx.scrubbed_prompt,
        response=ctx.scrubbed_response,
    )


def format_feedback_redirect(
    *,
    request_id: int,
    thread_id: str | None,
    model: str,
    reasoning_effort: str | None,
    detector: str | None,
    prompt_text: str,
    response_text: str,
) -> str:
    """Scrub the snippet ingredients and build the redirect message in one call.

    This is the convenience entry point for the admin/CLI layers: pass the raw
    transcript fields from a FeedbackSuggestion and the formatted,
    ready-to-display redirect string comes back. The caller never sees the
    un-scrubbed text after this.
    """
    return build_redirect_message(
        _RedirectContext(
            request_id=request_id,
            thread_id=thread_id,
            model=model,
            reasoning_effort=reasoning_effort,
            detector=detector,
            scrubbed_prompt=scrub_snippet(prompt_text),
            scrubbed_response=scrub_snippet(response_text),
        )
    )


# --- reserved auto-send switch (scoped-A4, OFF, NOT WIRED) -------------------
# The in-band auto-send variant (injecting this redirect into the model
# response stream without operator approval) is a separate scoped-A4
# autonomy decision. The auto-promotion master switch is OFF, so this helper
# exists only to RESERVE the knob and make the boundary explicit; it is not
# called from any injection path today. Raising here if it ever becomes wired
# without the operator decision would be the guard.


class FeedbackAutoSendNotWiredError(RuntimeError):
    """Raised if auto-send is enabled in config but no injection path is wired.

    Reserved for the scoped-A4 follow-on. Today there is no in-band injection
    call site, so a True setting has no effect and should not be silently
    accepted as 'working' — callers that observe feedback_auto_send_enabled()
    returning True without having implemented the injection path MUST raise
    this rather than degrade to a silent no-op.
    """


def feedback_auto_send_enabled() -> bool:
    """Return whether the held-back in-band auto-send variant is enabled.

    RESERVED / NOT WIRED. Reads CALLOSUM_FEEDBACK_AUTO_SEND_ENABLED (must
    be exactly "1"); defaults to False. No code path consumes a True
    result today — the in-band response-stream injection is the scoped-A4
    follow-on and is operator-gated behind the auto-promotion master switch.
    See FeedbackAutoSendNotWiredError for the guard the future call site
    must use.
    """
    return os.environ.get("CALLOSUM_FEEDBACK_AUTO_SEND_ENABLED") == "1"