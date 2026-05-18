"""Lightweight prompt-complexity classifier used by the v2 cost router.

Returns 1 / 2 / 3 matching the same scale the model's classifier instruction
emits, so v2 routing buckets line up with the training data we've been
collecting.

This v1 of the classifier is a structural heuristic: count the rough size
of the input payload (characters across the message tree) and bucket. It is
intentionally cheap (~microseconds, no network call, no model call) so it
can run on the dispatch hot path. When the auto-learning corpus grows
enough that a learned classifier outperforms the heuristic, swap this
function out — the only contract callers depend on is "give me 1/2/3".
"""

from __future__ import annotations

from typing import Any

# Rough character → token ratio for English text. Used for the
# token-equivalent thresholds below so buckets stay sensible even when
# upstream tokenization changes slightly.
_CHARS_PER_TOKEN = 4

# Bucket cutoffs in characters. ~500 tok and ~5_000 tok approximate
# the rough "short question / paragraph-sized prompt / long-context"
# split that matches how the model labels its own complexity classes.
_SIMPLE_MAX_CHARS = 500 * _CHARS_PER_TOKEN
_MODERATE_MAX_CHARS = 5_000 * _CHARS_PER_TOKEN


def _walk_text(node: Any) -> int:
    """Return the total length of all string text in this nested structure.

    Handles both OpenAI shapes the proxy translates between:
      * chat completions: messages[].content is str OR list[{type, text}]
      * responses API:    input[].content[].text
    Plus the top-level `instructions` field on /v1/responses.
    """
    if node is None:
        return 0
    if isinstance(node, str):
        return len(node)
    if isinstance(node, list):
        return sum(_walk_text(item) for item in node)
    if isinstance(node, dict):
        # Prefer narrow fields when present, otherwise walk everything.
        if "text" in node and isinstance(node["text"], str):
            return len(node["text"])
        if "content" in node:
            return _walk_text(node["content"])
        return sum(_walk_text(v) for v in node.values())
    return 0


def classify_prompt_complexity(
    body: dict[str, Any] | None,
    *,
    session_prompt_tokens: int | None = None,
) -> int:
    """Return 1 (simple), 2 (moderate), or 3 (complex) for the request body.

    If `session_prompt_tokens` is provided (the proxy already tracks this
    per-session from the last request's upstream usage block), we use it as
    a more accurate signal of accumulated context size. Otherwise we fall
    back to a character-based scan of the request body.

    Defaults to 2 (moderate) on any structural surprise so we never bias
    routing toward the cheapest cell on something we can't size.
    """
    if session_prompt_tokens is not None and session_prompt_tokens > 0:
        if session_prompt_tokens < 500:
            return 1
        if session_prompt_tokens < 5_000:
            return 2
        return 3

    if not isinstance(body, dict):
        return 2

    char_count = 0
    for key in ("input", "messages", "instructions"):
        if key in body:
            char_count += _walk_text(body[key])

    if char_count == 0:
        return 2

    if char_count <= _SIMPLE_MAX_CHARS:
        return 1
    if char_count <= _MODERATE_MAX_CHARS:
        return 2
    return 3
