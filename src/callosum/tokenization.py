"""Exact-ish token counting for the peer-quality audit overhead.

The peer-quality audit injects a hidden instruction + provenance tags into the
outbound request. Those tokens are real input the model bills, and Codex derives
its "Context X% left" meter from the response's usage.input_tokens (verified
against codex-rs core/src/client.rs). To keep the audit invisible to that
meter we subtract its token cost from the Codex-facing usage — so we need the
cost in *tokens*, not the chars/3 heuristic the router uses elsewhere.

For OpenAI GPT models we use their own tokenizer (tiktoken), which matches
the provider's input_tokens count. For anything else (local models, or if
tiktoken lacks an encoding) we fall back to a rounded-up chars/3 estimate.
The count is always biased to **round up** so the subtraction never *under*-
counts the audit — the meter can then never show audit tokens, only ever a
hair more context-free than reality (sub-resolution on a 256K window).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

_CHARS_PER_TOKEN = 3
# Default encoding for the MODEL-A0F8 family when a model id is not in tiktoken's
# registry yet (newer models ship before the library knows them). o200k_base is
# the MODEL-A0G5/5-era base encoding.
_DEFAULT_GPT_ENCODING = "o200k_base"


@lru_cache(maxsize=16)
def _encoder(model: str) -> Any:
    """Return a cached tiktoken encoder for model, or None if unavailable.

    Only attempted for GPT-style ids; local model slugs return None and use the
    fallback estimate (their windows/meters differ and tiktoken does not apply).
    """
    if not model.startswith("gpt-"):
        return None
    try:
        import tiktoken
    except Exception:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        try:
            return tiktoken.get_encoding(_DEFAULT_GPT_ENCODING)
        except Exception:
            return None


def _fallback_tokens(text: str) -> int:
    # Round UP so we never under-count the audit.
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


def count_tokens(text: str, *, model: str) -> int:
    """Token count of text for model.

    Exact via tiktoken for GPT models; rounded-up chars/3 otherwise.
    """
    if not text:
        return 0
    enc = _encoder(model)
    if enc is None:
        return _fallback_tokens(text)
    try:
        return int(len(enc.encode(text)))
    except Exception:
        return _fallback_tokens(text)
