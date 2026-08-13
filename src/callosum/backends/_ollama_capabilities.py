"""Shared ollama /api/show capability parse + fetch.

Extracted so the litellm_gateway and local_direct backends share one parser.
Returns primitives (not CellCapabilities) so each backend merges with its own
surrounding fields. Never raises — all failures return None.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Hoisted here (was in litellm_gateway) to avoid a circular import: litellm_gateway
# imports this constant back from this module.
DEFAULT_HEALTH_TIMEOUT_S = 2.0


@dataclass(frozen=True, slots=True)
class OllamaShowCapabilities:
    """Parsed ollama /api/show capability self-report (primitives, not CellCapabilities)."""

    modalities: frozenset[str]  # always includes "text"
    supports_tools: bool
    context_window: int  # from model_info *.context_length, else 128_000
    parameter_count: int | None  # general.parameter_count when present


def parse_ollama_show(info: dict[str, Any]) -> OllamaShowCapabilities | None:
    """Pure-sync parser for an ollama /api/show response body.

    Returns None on malformed input (missing/empty capabilities). Mirrors the
    parse previously inlined in litellm_gateway._refresh_capabilities:
      - `capabilities` list -> a lowercase set;
      - modalities = {"text"} + ("image" if "vision" in set) + ("audio" if "audio" in set);
      - supports_tools = "tools" in set;
      - model_info walk for the first `*.context_length` (fallback 128_000);
      - general.parameter_count when present (else None).
    """
    caps_list = info.get("capabilities")
    if not isinstance(caps_list, list):
        return None
    caps_set = {str(c).lower() for c in caps_list if isinstance(c, str)}
    modalities: set[str] = {"text"}
    if "vision" in caps_set:
        modalities.add("image")
    if "audio" in caps_set:
        modalities.add("audio")
    supports_tools = "tools" in caps_set
    # Context length lives under <architecture>.context_length; we don't know
    # the architecture name a priori. Walk the model_info dict and find any key
    # ending in `.context_length`. Parameter count is exposed at
    # `general.parameter_count` uniformly across architectures.
    context_window = 128_000  # fallback
    parameter_count: int | None = None
    model_info = info.get("model_info") or {}
    if isinstance(model_info, dict):
        for k, v in model_info.items():
            if isinstance(k, str) and k.endswith("context_length") and isinstance(v, int) and v > 0:
                context_window = v
        pc = model_info.get("general.parameter_count")
        if isinstance(pc, int) and pc > 0:
            parameter_count = pc
    return OllamaShowCapabilities(
        modalities=frozenset(modalities),
        supports_tools=supports_tools,
        context_window=context_window,
        parameter_count=parameter_count,
    )


async def fetch_ollama_capabilities(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    runtime_model: str,
    timeout_s: float = DEFAULT_HEALTH_TIMEOUT_S,
) -> OllamaShowCapabilities | None:
    """POST {endpoint}/api/show with body {"name": runtime_model}.

    endpoint is caller-supplied (NOT hardcoded) — callers pass their own ollama
    base URL (litellm_gateway passes its daemon URL; local_direct passes the
    per-cell entry endpoint). Never raises: returns None on any transport, HTTP,
    or parse failure. Matches the "must NEVER bubble up" contract the original
    code had (broad except, per-model continue).
    """
    try:
        resp = await client.post(
            f"{endpoint.rstrip('/')}/api/show",
            json={"name": runtime_model},
            timeout=timeout_s,
        )
    except httpx.HTTPError as exc:
        logger.debug("ollama /api/show transport failure for %s: %s", runtime_model, exc)
        return None
    if resp.status_code != 200:
        logger.debug("ollama /api/show non-200 for %s: %s", runtime_model, resp.status_code)
        return None
    try:
        body = resp.json()
    except ValueError as exc:
        logger.debug("ollama /api/show bad json for %s: %s", runtime_model, exc)
        return None
    if not isinstance(body, dict):
        return None
    return parse_ollama_show(body)
