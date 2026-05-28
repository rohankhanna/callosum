"""Extract a PromptFeatures record from an incoming request body.

Pure (synchronous) for the deterministic facts (tokens, modalities,
needs_tools); the optional embedding step is async because real providers
run on GPU. Both OpenAI Chat Completions shape and Codex Responses shape
are handled.
"""

from __future__ import annotations

from typing import Any

from callosum.routing.protocols import EmbeddingProvider, PromptFeatures


# Conversion factor for the rough token estimate. Matches the existing
# `_approx_input_tokens` in the legacy recommender — ~3 chars/token gives
# a ~33% safety margin over real Codex tokenization. Kept here instead of
# imported so the routing module has no dependency on the soon-to-be-
# deleted cell_recommender.
_CHARS_PER_TOKEN = 3


def _walk_text(node: Any) -> list[str]:
    """Pull every user-visible text string out of a nested message body.

    Handles chat-completions (messages[].content as str or list-of-parts
    with `text`), Codex Responses (input[].content[].text), and top-level
    `instructions`. Returns a list of strings; caller joins.
    """
    if node is None:
        return []
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        out: list[str] = []
        for x in node:
            out.extend(_walk_text(x))
        return out
    if isinstance(node, dict):
        if isinstance(node.get("text"), str):
            return [node["text"]]
        if "content" in node:
            return _walk_text(node["content"])
        out: list[str] = []
        for v in node.values():
            out.extend(_walk_text(v))
        return out
    return []


def _extract_text(body: dict[str, Any]) -> str:
    """Concatenate every text-shaped string from the user-visible fields."""
    parts: list[str] = []
    for key in ("input", "messages", "instructions"):
        if key in body:
            parts.extend(_walk_text(body[key]))
    return "\n".join(p for p in parts if p)


def _approx_tokens(text: str) -> int:
    """Rough chars/3 estimate (overestimates real Codex token count by
    ~33%; leaves margin so capability filtering errs toward larger
    cells when uncertain)."""
    return max(256, len(text) // _CHARS_PER_TOKEN)


def _detect_modalities(body: dict[str, Any]) -> frozenset[str]:
    """Walk the body looking for non-text content-part `type` markers.

    OpenAI-style multimodal messages have parts like
    `{"type": "image_url", ...}` or `{"type": "input_image", ...}`. Codex
    Responses uses `input_image`, `input_audio`, etc. We collect every
    non-text type tag and normalize to {"text", "image", "audio", "video"}.
    """
    modalities: set[str] = {"text"}

    def _walk(node: Any) -> None:
        if isinstance(node, list):
            for x in node:
                _walk(x)
        elif isinstance(node, dict):
            t = node.get("type")
            if isinstance(t, str):
                if "image" in t.lower():
                    modalities.add("image")
                elif "audio" in t.lower():
                    modalities.add("audio")
                elif "video" in t.lower():
                    modalities.add("video")
            for v in node.values():
                _walk(v)

    for key in ("input", "messages"):
        if key in body:
            _walk(body[key])
    return frozenset(modalities)


def _detect_tools(body: dict[str, Any]) -> bool:
    """A non-empty `tools` field at the top level means the caller expects
    tool-use. Some clients send `functions` (older OpenAI shape) too; we
    accept either."""
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        return True
    functions = body.get("functions")
    if isinstance(functions, list) and functions:
        return True
    return False


async def extract_features(
    body: dict[str, Any],
    embedding_provider: EmbeddingProvider,
) -> PromptFeatures:
    """Build a PromptFeatures record from a request body.

    The embedding step is the only async part; deterministic facts are
    extracted synchronously. When the embedding provider returns None
    (no-op or cold-start), `features.embedding` is None and downstream
    predictors fall back to their priors.
    """
    text = _extract_text(body)
    tokens = _approx_tokens(text)
    modalities = _detect_modalities(body)
    needs_tools = _detect_tools(body)
    embedding = await embedding_provider.embed(text) if text else None
    return PromptFeatures(
        text=text,
        tokens=tokens,
        modalities=modalities,
        needs_tools=needs_tools,
        embedding=embedding,
    )
