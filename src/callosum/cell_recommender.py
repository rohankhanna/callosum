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
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from callosum.backend import Backend, CallHandle
from callosum.cell_grid import Cell, model_strength_key
from callosum.errors import BackendError

logger = logging.getLogger(__name__)


_RECOMMENDER_INSTRUCTION_PREFIX = (
    "You are a routing classifier. Read the user prompt below and choose "
    "EXACTLY one model + reasoning-effort cell from the available list to "
    "handle it.\n\n"
    "Cost asymmetry: remote cells consume the operator's weekly Codex "
    "quota (finite, refills weekly); local cells are free (hosted on this "
    "machine, no token cost). Prefer the smallest cell that can answer "
    "well — pick a local cell when it can handle the task; escalate to a "
    "more capable remote cell only when the task genuinely needs it "
    "(complex reasoning, long context, specialized capabilities).\n\n"
    "Output ONLY the chosen cell as 'model effort' "
)


@dataclass(frozen=True, slots=True)
class Recommendation:
    """Result of CellRecommender.recommend().

    `classifier_cell` and `raw_output` are populated only when this
    decision came from a live upstream classifier call (source =
    'upstream' or 'alternative'). Cache hits and fallbacks leave them
    None — there's no per-decision classifier output to record, and we
    don't want training data to confuse 'classifier said X' with
    'we reused a prior decision'.

    `candidates` is an ordered tuple of cells the dispatch layer may try
    if the primary `cell` fails with a retryable error. Index 0 is
    always `cell`; remaining entries are the rest of the compatible cell
    set in grid-priority order (Codex first by upstream priority, local
    after — same order build_cells_from_metadata produces). Streams use
    only index 0 (mid-stream failover stays a non-goal); non-streaming
    requests walk candidates up to dispatch's cell-retry cap.
    """

    cell: Cell
    source: str  # "upstream" | "cache" | "fallback" | "alternative" | "fallback_alt"
    cache_key: str | None
    latency_s: float
    classifier_cell: Cell | None = None
    raw_output: str | None = None
    candidates: tuple[Cell, ...] = ()


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


# Maximum prompt text size to send to the classifier. Hermes-shaped requests
# carry 30-40k-token personas plus full conversation history; sending all of
# that to a small classifier (a) chews enormous input-side latency on the
# classifier call, often busting the 5s default timeout, and (b) is wasteful
# — the routing decision is dominated by the LATEST user turn, which lives
# at the tail of the text. Keep a head sample so a system-message signal
# isn't lost, plus a larger tail.
_CLASSIFIER_HEAD_CHARS = 1024
_CLASSIFIER_TAIL_CHARS = 3072
_CLASSIFIER_TRUNCATION_MARKER = "\n…[truncated for classifier]…\n"


def _truncate_for_classifier(prompt_text: str) -> str:
    """Trim a long prompt down to head + tail samples for the classifier.

    Short prompts pass through unchanged. Long prompts get the first
    _CLASSIFIER_HEAD_CHARS + last _CLASSIFIER_TAIL_CHARS, joined with a
    marker. Cache key is computed on the FULL prompt elsewhere so two
    different long prompts whose head/tail happen to align don't collide.
    """
    head_n = _CLASSIFIER_HEAD_CHARS
    tail_n = _CLASSIFIER_TAIL_CHARS
    if len(prompt_text) <= head_n + tail_n + len(_CLASSIFIER_TRUNCATION_MARKER):
        return prompt_text
    return (
        prompt_text[:head_n]
        + _CLASSIFIER_TRUNCATION_MARKER
        + prompt_text[-tail_n:]
    )


# Conservative buffer above the input-token estimate. Leaves room for the
# completion, the system prompt, and the recommender prompt overhead. Big
# enough that small-context cells (e.g. 4096-window local models) are
# filtered out for non-trivial prompts; small enough not to push every
# request to the largest cell.
_CONTEXT_HEADROOM_TOKENS = 4096


def _approx_input_tokens(prompt_text: str) -> int:
    """Rough chars/2 estimate. Overshoots vs real tokenization — that's
    deliberate, since the filter should err toward larger-context cells
    (better to send a 4k-prompt to a 128k model than to a 4k model that
    will reject it).
    """
    return max(256, len(prompt_text) // 2)


def _candidates_ordered(primary: Cell, pool: list[Cell]) -> tuple[Cell, ...]:
    """Build the ordered candidates tuple for cell-level retry.

    Index 0 is the primary pick. Remaining entries are the rest of `pool`
    in its existing order — typically the cell grid's priority ranking
    from build_cells_from_metadata, so the dispatch loop's "next-best"
    cell is the next-strongest compatible model. `primary` is deduped
    out if it appears in `pool`.
    """
    rest = tuple(c for c in pool if c != primary)
    return (primary,) + rest


# Effort ordering for display in the compact cell list — matches the
# canonical low→xhigh progression with "default" (the local-cell sentinel)
# sorted to the end since it carries no relative ranking.
_EFFORT_DISPLAY_ORDER = ("low", "medium", "high", "xhigh", "default")


def _infer_runtime_kind(cell: Cell) -> str:
    """Heuristic: cells whose only effort level is `default` are local;
    everything else is remote (Codex). Used to label cells in the
    classifier prompt so the model has factual info about cost
    asymmetry (remote consumes quota, local is free). Will be replaced
    by an explicit field on Cell once Phase 4d threads backend
    metadata through.
    """
    return "local" if cell.reasoning_effort == "default" else "remote"


def _format_cell_list_compact(cells: list[Cell]) -> str:
    """Group cells by model and emit one line per model with all available
    efforts, runtime kind, and the model's context window. Compresses
    the per-permutation enumeration (~32 lines for a typical mixed grid)
    down to one line per distinct model (~12 lines), cutting classifier-
    prompt tokens roughly in half on every routing decision.

    The cell-list format is purely for the classifier's *view*; the parser
    still substring-matches `<model> <effort>` in the classifier's reply
    against allowed_cells, so the response shape is unchanged.

    Format per model:
        - <model>  <effort>|<effort>|...  (<runtime>, <ctx>K)
    or for single-effort models (typical for local cells):
        - <model>  <effort>  (<runtime>, <ctx>K)
    """
    by_model: dict[str, list[Cell]] = {}
    model_order: list[str] = []
    for c in cells:
        if c.model not in by_model:
            model_order.append(c.model)
        by_model.setdefault(c.model, []).append(c)

    def _effort_rank(effort: str) -> int:
        try:
            return _EFFORT_DISPLAY_ORDER.index(effort)
        except ValueError:
            return len(_EFFORT_DISPLAY_ORDER)

    lines: list[str] = []
    for model in model_order:
        ms = by_model[model]
        efforts = sorted({c.reasoning_effort for c in ms}, key=_effort_rank)
        ctx = next((c.context_window for c in ms if c.context_window), None)
        kind = _infer_runtime_kind(ms[0])
        effort_str = "|".join(efforts)
        meta_parts = [kind]
        if ctx:
            meta_parts.append(f"{ctx // 1000}K")
        meta_str = ", ".join(meta_parts)
        lines.append(f"- {model}  {effort_str}  ({meta_str})")
    return "\n".join(lines)


def _filter_cells_by_context(
    cells: list[Cell], estimated_input_tokens: int
) -> list[Cell]:
    """Drop cells whose known context_window can't fit input + headroom.

    Cells with `context_window=None` are kept — unknown context is treated
    as compatible rather than excluded. This is intentional: many local
    models report no window via /v1/models, and excluding them entirely
    would defeat the point of having local cells in the grid.

    If every cell with a known window is too small, callers should treat
    an empty return as "filter found nothing usable" and fall back to the
    full grid — better to ask a backend that may reject than to refuse
    the user request outright.
    """
    threshold = estimated_input_tokens + _CONTEXT_HEADROOM_TOKENS
    return [
        c for c in cells
        if c.context_window is None or c.context_window >= threshold
    ]


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
        upstream_timeout_s: float = 30.0,
    ) -> None:
        self._cheap_backend = cheap_backend
        self._configured_cheap_cell = cheap_cell
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
            "alternative_calls": 0,
        }
        self._recommendation_counts: dict[str, int] = {}
        # Per-classifier-cell call counts: how many times each cell was used
        # AS the classifier (cheap path + alternative path combined). Lets us
        # see if bias mitigation actually rotated through alternatives at the
        # configured rate.
        self._classifier_call_counts: dict[str, int] = {}

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    @property
    def recommendation_counts(self) -> dict[str, int]:
        return dict(self._recommendation_counts)

    @property
    def classifier_call_counts(self) -> dict[str, int]:
        return dict(self._classifier_call_counts)

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

    def _resolve_cheap_cell(self, allowed_cells: list[Cell]) -> Cell:
        """Pick the cell that should serve as the classifier for this call.

        Preference order (so the router keeps working when OpenAI's model
        names churn underneath us):
          1. The configured cheap cell if it's still in `allowed_cells`.
          2. The weakest model (by model_strength_key, "weakest last") at
             the lowest reasoning effort actually present for that model.
          3. The first allowed cell as a last resort.
          4. The configured cheap cell unchanged when nothing's available
             (degenerate / cold-start; upstream call will then fail
             gracefully and the dispatch path falls back).
        """
        configured_tuple = (
            self._configured_cheap_cell.model,
            self._configured_cheap_cell.reasoning_effort,
        )
        for c in allowed_cells:
            if (c.model, c.reasoning_effort) == configured_tuple:
                return c
        if not allowed_cells:
            return self._configured_cheap_cell
        # Configured cheap cell isn't in the live grid (model renamed,
        # retired, or replaced upstream). Pick a weakest-available cell.
        weakest_model = sorted(
            {c.model for c in allowed_cells},
            key=model_strength_key,
            reverse=True,  # model_strength_key returns SMALLER==STRONGER, so reverse for weakest first
        )[0]
        same_model = [c for c in allowed_cells if c.model == weakest_model]
        # Stable preference for lower effort first, but tolerate unknown effort labels.
        _EFFORT_ORDER = {"low": 0, "medium": 1, "high": 2, "xhigh": 3}
        same_model.sort(
            key=lambda c: _EFFORT_ORDER.get(c.reasoning_effort, 99)
        )
        return same_model[0]

    def _build_recommender_body(
        self,
        prompt_text: str,
        allowed_cells: list[Cell],
        *,
        classifier_cell: Cell | None = None,
    ) -> dict[str, Any]:
        """Construct the classifier request body.

        Defaults to whatever _resolve_cheap_cell picks from the current
        allowed_cells; callers can override classifier_cell to fire the same
        prompt at a different classifier (bias mitigation + comparison
        sampling). The instruction's example cell name is filled in from
        the current cell list so the model never sees a stale or invented
        name as guidance.
        """
        cl = classifier_cell if classifier_cell is not None else self._resolve_cheap_cell(allowed_cells)
        cell_list = _format_cell_list_compact(allowed_cells)
        example_cell = allowed_cells[0] if allowed_cells else self._configured_cheap_cell
        instructions = (
            _RECOMMENDER_INSTRUCTION_PREFIX
            + f"(e.g. '{example_cell.model} {example_cell.reasoning_effort}'). "
            + "No explanation, no quotes, no other text."
            + "\n\nAvailable cells:\n"
            + cell_list
        )
        return {
            "model": cl.model,
            "reasoning": {"effort": cl.reasoning_effort},
            "instructions": instructions,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": _truncate_for_classifier(prompt_text),
                        }
                    ],
                }
            ],
            "stream": False,
            "store": False,
        }

    async def _ask_upstream(
        self,
        prompt_text: str,
        allowed_cells: list[Cell],
        *,
        classifier_cell: Cell | None = None,
    ) -> tuple[Cell | None, str | None]:
        """One non-streaming responses call to a classifier cell; parse cell name.

        Returns (parsed_cell_or_None, raw_classifier_text_or_None). The raw
        text is returned even when parsing fails so the caller (and the
        request log via Recommendation.raw_output) can record what the
        classifier actually said — useful for training data and for
        debugging why a fallback fired.

        Defaults to self._configured_cheap_cell; pass classifier_cell to
        override (bias mitigation path).
        """
        # Phase 4d: pre-check classifier backend's routability. If the
        # configured cheap_backend is weekly-exhausted or in cooldown, the
        # upstream call would either 429 or hang on a now-defunct route;
        # either way the recommender ends up falling back. Skip the call
        # entirely so the user doesn't pay the timeout latency on the
        # first request after an outage. Reads cached state — no I/O.
        # Skipped when an alternative classifier is in play (the caller
        # specified a different backend cell on purpose).
        if classifier_cell is None:
            try:
                _u = await self._cheap_backend.usage_snapshot()
                if _u.weekly_exhausted or (
                    _u.cooldown_until_ts is not None
                    and _u.cooldown_until_ts > time.time()
                ):
                    self._stats["upstream_skipped_unroutable"] = (
                        self._stats.get("upstream_skipped_unroutable", 0) + 1
                    )
                    return (None, None)
            except Exception:
                # If we can't even read the snapshot, fall through to the
                # call — the existing exception handler below catches the
                # actual upstream error.
                pass

        body = self._build_recommender_body(
            prompt_text, allowed_cells, classifier_cell=classifier_cell
        )
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
            return (None, None)

        self._stats["upstream_calls"] += 1
        # Pull the literal text out of the Responses API shape.
        raw_text: str | None = None
        try:
            output = result.get("output") or []
            for item in output:
                for c in item.get("content") or []:
                    text = c.get("text")
                    if isinstance(text, str) and text.strip():
                        raw_text = text
                        break
                if raw_text is not None:
                    break
        except (AttributeError, KeyError, IndexError, TypeError):
            pass
        if raw_text is None:
            return (None, None)
        return (_parse_cell_from_output(raw_text, allowed_cells), raw_text)

    async def recommend(
        self,
        body: dict[str, Any],
        *,
        allowed_cells: list[Cell],
        fallback: Cell,
        classifier_cell: Cell | None = None,
        local_exploration_pct: float = 0.0,
    ) -> Recommendation:
        """Return a Cell recommendation for this request body.

        Priority (normal path, classifier_cell=None):
          1. Cache hit (no upstream call).
          2. Upstream classifier call to cheap_cell.
          3. Fallback Cell on any failure (timeout, garbage, unrecognized cell).

        Bias-mitigation path (classifier_cell=<non-cheap Cell>):
          * Cache is skipped on read AND write — the alternative classifier
            gives a one-off second opinion; we don't poison the cheap-
            classifier-cached routing decisions with another classifier's
            answers, and a cache hit from a prior cheap-classifier call is
            also bypassed so this request really exercises the alternative.
          * Stats track per-classifier call counts so /status surfaces how
            often each classifier was used and what it recommended.
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

        # Compatibility filter: drop cells that can't fit this prompt's input
        # + a headroom buffer. The classifier sees only the compatible set,
        # so it can't pick a cell that would 4xx for context-length reasons.
        # If the filter empties the set (huge prompt vs every known window
        # too small), fall back to the full grid and let the backend report
        # the real error — better than silently refusing the request.
        est_tokens = _approx_input_tokens(prompt_text)
        compatible_cells = _filter_cells_by_context(allowed_cells, est_tokens)
        if not compatible_cells:
            logger.warning(
                "cell_recommender: no cells fit ~%d input tokens; using full "
                "grid as a last resort.",
                est_tokens,
            )
            compatible_cells = allowed_cells

        # Local-cell exploration. Independent of the classifier — with
        # `local_exploration_pct` probability, skip the classifier entirely
        # and route to a random eligible local cell. Generates training
        # data on local-cell outcomes that the cheap remote classifier
        # would otherwise never produce (it prefers familiar Codex names).
        # Does not write to the prompt-text cache so the cheap classifier's
        # cache stays clean for non-exploration paths. Only fires on the
        # non-alternative-classifier path (the alternative path is itself
        # a different bias-mitigation rotation and we don't compound them).
        if classifier_cell is None and local_exploration_pct > 0:
            local_cells = [c for c in compatible_cells if _infer_runtime_kind(c) == "local"]
            if local_cells and random.random() < local_exploration_pct:
                chosen = random.choice(local_cells)
                self._stats["local_exploration_calls"] = (
                    self._stats.get("local_exploration_calls", 0) + 1
                )
                self._record_recommendation(chosen)
                return Recommendation(
                    cell=chosen,
                    source="local_exploration",
                    cache_key=None,
                    latency_s=time.time() - t0,
                    classifier_cell=None,
                    raw_output=None,
                    candidates=_candidates_ordered(chosen, compatible_cells),
                )

        key = _cache_key(prompt_text)
        is_alternative = classifier_cell is not None
        if is_alternative:
            cls = classifier_cell
            classifier_key = f"{cls.model} {cls.reasoning_effort}"
            self._stats["alternative_calls"] = (
                self._stats.get("alternative_calls", 0) + 1
            )
            self._classifier_call_counts[classifier_key] = (
                self._classifier_call_counts.get(classifier_key, 0) + 1
            )
        else:
            # Resolve which cell actually serves as the cheap classifier for
            # this call against the CURRENT live cell list — handles the
            # case where the configured cheap_cell isn't in the live grid
            # anymore (model renamed, retired, or never advertised by any
            # backend on this machine).
            cls = self._resolve_cheap_cell(allowed_cells)
            self._classifier_call_counts[f"{cls.model} {cls.reasoning_effort}"] = (
                self._classifier_call_counts.get(
                    f"{cls.model} {cls.reasoning_effort}", 0
                ) + 1
            )
            cached = self._cache_get(key)
            if cached is not None:
                self._stats["cache_hits"] += 1
                self._record_recommendation(cached)
                return Recommendation(
                    cell=cached,
                    source="cache",
                    cache_key=key,
                    latency_s=time.time() - t0,
                    candidates=_candidates_ordered(cached, compatible_cells),
                )

        chosen, raw_output = await self._ask_upstream(
            prompt_text, compatible_cells, classifier_cell=classifier_cell
        )
        if chosen is None:
            self._stats["fallback_count"] += 1
            self._record_recommendation(fallback)
            return Recommendation(
                cell=fallback,
                source="fallback" if not is_alternative else "fallback_alt",
                cache_key=key,
                latency_s=time.time() - t0,
                # Even on fallback, surface the classifier identity + its
                # garbled output (if any) so the request log records what
                # actually happened.
                classifier_cell=cls,
                raw_output=raw_output,
                candidates=_candidates_ordered(fallback, compatible_cells),
            )

        # Only cache decisions made by the cheap classifier; alternative
        # classifier decisions are one-off and intentionally don't
        # influence subsequent routing.
        if not is_alternative:
            self._cache_put(key, chosen)
        self._record_recommendation(chosen)
        return Recommendation(
            cell=chosen,
            source="upstream" if not is_alternative else "alternative",
            cache_key=key,
            latency_s=time.time() - t0,
            classifier_cell=cls,
            raw_output=raw_output,
            candidates=_candidates_ordered(chosen, compatible_cells),
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
