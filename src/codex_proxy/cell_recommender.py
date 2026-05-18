"""Upstream-driven cell recommender for the cost-optimal router.

Instead of training a local classifier (data-sparse — only ~300 unique
prompts in the corpus after dedup), we ask the cheapest available cell
to pick the right model + reasoning-effort cell for an incoming prompt.
The recommender's output IS the routing decision — no intermediate
complexity bucket, no lookup table.

Hash-cache by the user's prompt text so repeated prompts (the audit
found 17,236 copies of the Hermes persona prompt alone) pay the
upstream classifier call exactly once.

This module is intentionally narrow: it computes a recommendation, it
does not dispatch the actual user request. Callers wire the chosen
cell into the request body and route as normal.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from codex_proxy.backend import Backend, CallHandle
from codex_proxy.cell_grid import Cell
from codex_proxy.errors import BackendError

logger = logging.getLogger(__name__)


_RECOMMENDER_INSTRUCTION = (
    "You are a routing classifier. Read the user prompt below and choose "
    "EXACTLY one model + reasoning-effort cell from the available list to "
    "handle it. Pick the cheapest cell that can answer the prompt well — "
    "do not always pick the largest or the smallest. "
    "Output ONLY the chosen cell as 'model effort' (e.g. 'model-a0c3 medium'). "
    "No explanation, no quotes, no other text."
)


@dataclass(frozen=True, slots=True)
class Recommendation:
    """Result of CellRecommender.recommend()."""

    cell: Cell
    source: str  # "upstream" | "cache" | "fallback"
    cache_key: str | None
    latency_s: float


def _walk_text(node: Any) -> str:
    """Concatenate every text-shaped string in this nested request body.

    Handles both chat-completions (messages[].content as str or list of
    parts with `text`) and Responses API (input[].content[].text) plus the
    top-level `instructions` field.
    """
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "\n".join(_walk_text(x) for x in node)
    if isinstance(node, dict):
        if isinstance(node.get("text"), str):
            return node["text"]
        if "content" in node:
            return _walk_text(node["content"])
        return "\n".join(_walk_text(v) for v in node.values())
    return ""


def _extract_user_text(body: dict[str, Any]) -> str:
    """Pull every user-visible text fragment from the request body."""
    parts: list[str] = []
    for key in ("input", "messages", "instructions"):
        if key in body:
            parts.append(_walk_text(body[key]))
    return "\n".join(p for p in parts if p)


def _cache_key(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8", errors="replace")).hexdigest()


def _parse_cell_from_output(
    output: str, allowed_cells: list[Cell]
) -> Cell | None:
    """Match the recommender's free-text output against the allowed cell list.

    Tries exact `model effort` match first (case-insensitive), then a
    substring sweep so a slightly chatty model output ("I recommend
    model-a0c3 medium because...") still resolves. Returns None if
    no allowed cell can be matched, which sends the caller to its
    fallback path.
    """
    if not isinstance(output, str) or not output.strip():
        return None
    text = output.strip().lower()
    # Try exact line-shaped match first.
    for c in allowed_cells:
        candidate = f"{c.model} {c.reasoning_effort}".lower()
        if text == candidate or text.startswith(candidate + "\n"):
            return c
    # Substring fallback for chatty outputs.
    for c in allowed_cells:
        candidate = f"{c.model} {c.reasoning_effort}".lower()
        if candidate in text:
            return c
    return None


class CellRecommender:
    """Recommend a Cell for a request by asking a cheap upstream model.

    Thread-safety: the in-memory cache and stats counters are touched only
    from the asyncio event loop; the existing proxy is single-loop uvicorn
    so no explicit locking is needed. If that ever changes, wrap the
    mutations in an asyncio.Lock — the call surface stays the same.
    """

    def __init__(
        self,
        *,
        cheap_backend: Backend,
        cheap_cell: Cell,
        cache_max: int = 4096,
        cache_ttl_seconds: int = 3600,
        upstream_timeout_s: float = 5.0,
    ) -> None:
        self._cheap_backend = cheap_backend
        self._cheap_cell = cheap_cell
        self._cache: OrderedDict[str, tuple[Cell, float]] = OrderedDict()
        self._cache_max = cache_max
        self._cache_ttl_seconds = cache_ttl_seconds
        self._upstream_timeout_s = upstream_timeout_s
        self._stats: dict[str, int] = {
            "calls": 0,
            "cache_hits": 0,
            "upstream_calls": 0,
            "upstream_failures": 0,
            "fallback_count": 0,
        }
        self._recommendation_counts: dict[str, int] = {}

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    @property
    def recommendation_counts(self) -> dict[str, int]:
        return dict(self._recommendation_counts)

    def _cache_get(self, key: str) -> Cell | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        cell, expires_at = entry
        if expires_at < time.time():
            # Expired; drop and treat as miss.
            self._cache.pop(key, None)
            return None
        # Refresh LRU position.
        self._cache.move_to_end(key)
        return cell

    def _cache_put(self, key: str, cell: Cell) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (cell, time.time() + self._cache_ttl_seconds)
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

    def _build_recommender_body(
        self, prompt_text: str, allowed_cells: list[Cell]
    ) -> dict[str, Any]:
        cell_list = "\n".join(
            f"- {c.model} {c.reasoning_effort}" for c in allowed_cells
        )
        instructions = (
            _RECOMMENDER_INSTRUCTION
            + "\n\nAvailable cells:\n"
            + cell_list
        )
        return {
            "model": self._cheap_cell.model,
            "reasoning": {"effort": self._cheap_cell.reasoning_effort},
            "instructions": instructions,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt_text}],
                }
            ],
            "stream": False,
            "store": False,
        }

    async def _ask_upstream(
        self, prompt_text: str, allowed_cells: list[Cell]
    ) -> Cell | None:
        """One non-streaming responses call to the cheap cell; parse cell name."""
        body = self._build_recommender_body(prompt_text, allowed_cells)
        try:
            result = await asyncio.wait_for(
                self._cheap_backend.responses(body, CallHandle()),
                timeout=self._upstream_timeout_s,
            )
        except (BackendError, asyncio.TimeoutError, Exception) as exc:
            self._stats["upstream_failures"] += 1
            logger.warning(
                "cell_recommender: upstream call failed (%s); falling back",
                type(exc).__name__,
            )
            return None

        self._stats["upstream_calls"] += 1
        # Parse model output from the Responses API shape.
        try:
            output = result.get("output") or []
            for item in output:
                for c in item.get("content") or []:
                    text = c.get("text")
                    if isinstance(text, str) and text.strip():
                        return _parse_cell_from_output(text, allowed_cells)
        except (AttributeError, KeyError, IndexError, TypeError):
            pass
        return None

    async def recommend(
        self,
        body: dict[str, Any],
        *,
        allowed_cells: list[Cell],
        fallback: Cell,
    ) -> Recommendation:
        """Return a Cell recommendation for this request body.

        Priority:
          1. Cache hit (no upstream call).
          2. Upstream classifier call to cheap_cell.
          3. Fallback Cell on any failure (timeout, garbage output, unrecognized cell).
        """
        self._stats["calls"] += 1
        t0 = time.time()
        prompt_text = _extract_user_text(body)
        if not prompt_text:
            self._stats["fallback_count"] += 1
            self._record_recommendation(fallback)
            return Recommendation(
                cell=fallback,
                source="fallback",
                cache_key=None,
                latency_s=time.time() - t0,
            )

        key = _cache_key(prompt_text)
        cached = self._cache_get(key)
        if cached is not None:
            self._stats["cache_hits"] += 1
            self._record_recommendation(cached)
            return Recommendation(
                cell=cached,
                source="cache",
                cache_key=key,
                latency_s=time.time() - t0,
            )

        chosen = await self._ask_upstream(prompt_text, allowed_cells)
        if chosen is None:
            self._stats["fallback_count"] += 1
            self._record_recommendation(fallback)
            return Recommendation(
                cell=fallback,
                source="fallback",
                cache_key=key,
                latency_s=time.time() - t0,
            )

        self._cache_put(key, chosen)
        self._record_recommendation(chosen)
        return Recommendation(
            cell=chosen,
            source="upstream",
            cache_key=key,
            latency_s=time.time() - t0,
        )

    def _record_recommendation(self, cell: Cell) -> None:
        k = f"{cell.model} {cell.reasoning_effort}"
        self._recommendation_counts[k] = self._recommendation_counts.get(k, 0) + 1

    async def fire_comparisons(
        self,
        body: dict[str, Any],
        *,
        allowed_cells: list[Cell],
        comparison_tiers: list[Cell],
    ) -> list[dict[str, Any]]:
        """Fire the same prompt at each comparison_tier in parallel.

        Pure observation — never used to make the actual routing decision.
        Returns one dict per tier: {tier_model, tier_effort, recommended_model,
        recommended_effort, latency_s, error}. Caller is expected to
        fire-and-forget (asyncio.create_task) and log/store results;
        comparison sampling must never block the user response.
        """
        prompt_text = _extract_user_text(body)
        if not prompt_text or not comparison_tiers:
            return []
        self._stats["comparison_runs"] = self._stats.get("comparison_runs", 0) + 1

        async def _one(tier: Cell) -> dict[str, Any]:
            tier_body = self._build_recommender_body(prompt_text, allowed_cells)
            tier_body["model"] = tier.model
            tier_body["reasoning"] = {"effort": tier.reasoning_effort}
            t0 = time.time()
            try:
                result = await asyncio.wait_for(
                    self._cheap_backend.responses(tier_body, CallHandle()),
                    timeout=self._upstream_timeout_s,
                )
            except (BackendError, asyncio.TimeoutError, Exception) as exc:
                return {
                    "tier_model": tier.model,
                    "tier_effort": tier.reasoning_effort,
                    "recommended_model": None,
                    "recommended_effort": None,
                    "latency_s": time.time() - t0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            chosen: Cell | None = None
            try:
                output = result.get("output") or []
                for item in output:
                    for c in item.get("content") or []:
                        text = c.get("text")
                        if isinstance(text, str) and text.strip():
                            chosen = _parse_cell_from_output(text, allowed_cells)
                            break
                    if chosen is not None:
                        break
            except (AttributeError, KeyError, IndexError, TypeError):
                pass
            return {
                "tier_model": tier.model,
                "tier_effort": tier.reasoning_effort,
                "recommended_model": chosen.model if chosen else None,
                "recommended_effort": chosen.reasoning_effort if chosen else None,
                "latency_s": time.time() - t0,
                "error": None if chosen else "unparseable_output",
            }

        return await asyncio.gather(*[_one(tier) for tier in comparison_tiers])
