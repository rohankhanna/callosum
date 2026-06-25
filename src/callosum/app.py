from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from callosum import __version__
from callosum.auth import (
    ApiKeyInvalidError,
    AuthService,
    InvalidCredentialsError,
    SessionInvalidError,
)
from callosum.auth_db import ApiKey, Session
from callosum.backend import Backend, CallHandle, HealthStatus
from callosum.cell_grid import (
    REASONING_LEVELS,
    VIRTUAL_MODELS,
    Cell,
    ModelMetadata,
    build_cells,
    cell_sample_counts,
    live_completion_models,
    reasoning_levels_for,
)
from callosum.config import AutoRouterConfig
from callosum.errors import RETRYABLE, BackendError, ErrorClass
from callosum.fallback import FallbackExecutor, should_attempt_fallback
from callosum.label_ui import install_label_ui
from callosum.peer_quality import PeerQualityOpinion, extract_peer_quality_opinions
from callosum.routing.cost_estimator import (
    CompositeCostModelProvider,
    CompositeCostUsageEstimator,
    CostUsageEstimator,
)
from callosum.routing.cost_model import CostRankProvider
from callosum.routing.factory import build_router
from callosum.routing.features import _approx_tokens
from callosum.routing.protocols import CellCapabilities
from callosum.routing.quota import select_quota_deficit_cell
from callosum.routing.router import NoCompatibleCellError, Router
from callosum.routing.time_estimator import TimeModelProvider, TimeUsageEstimator
from callosum.routing.usage_estimate import EstimateInput, OutputTokenForecaster
from callosum.routing.usage_rates import usage_rate_report
from callosum.selector import BackendSnapshot, blocking_meters, constraining_meter, select
from callosum.selectors import SelectorError, is_selector, parse_selector
from callosum.session import SessionRegistry
from callosum.tokenization import count_tokens
from callosum.usage_log import RoutingAttempt, SessionAssistantTurn, UsageLog, UsageLogEntry

logger = logging.getLogger("callosum.startup")


def _utc_timestamp() -> str:
    """Return current time in ISO 8601 UTC format with Z suffix."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# Clients opt into sticky routing by sending this header. When absent, every
# request is a fresh selection. There is no server-side "session mode" knob.
# Codex CLI sends `session-id` per-instance (one stable value for the
# lifetime of one codex process) — confirmed in the upstream codex source
# at codex-rs/codex-api/src/requests/headers.rs. Earlier code looked for
# `x-codex-session-id` which never matched real Codex traffic, so every
# request's session_id column ended up NULL despite the per-instance
# value being right there on the wire. Accept both names so any older
# custom client that happens to send the prefixed form keeps working.
# Order matters: the unprefixed `session-id` (what real Codex sends) is
# checked first; the prefixed form is a fallback. HTTP headers are
# case-insensitive at the request.headers layer.
SESSION_HEADERS: tuple[str, ...] = ("session-id", "x-codex-session-id")

NonstreamCall = Callable[[Backend, dict[str, Any], CallHandle], Awaitable[dict[str, Any]]]
StreamCall = Callable[[Backend, dict[str, Any], CallHandle], AsyncIterator[bytes]]

_EXHAUSTED_STATUS: dict[ErrorClass, int] = {
    "auth_invalid": 502,
    "unknown_model": 400,
    "rate_limited": 429,
    "transient": 502,
    "client_error": 400,
}

# Context variable to track the current request's database rowid, set during dispatch
# and read by response handlers to include in X-Proxy-Request-ID header.
_request_id_context: ContextVar[int | None] = ContextVar("request_id", default=None)

# The marker-only prompt-complexity label is retired (): the
# bare {{{1|2|3}}} class lacked judge/model provenance and was never an input to
# the live router, so collection is removed. Output scrubbers below still strip
# any stray marker a model voluntarily emits, and the legacy
# `prompt_complexity_class` column is preserved (always NULL for new rows) until
# an explicit data migration is approved.

# Context variable carrying the per-request effective routing mode — the
# mode this specific request was processed under. Set at the routing
# entry point after the canary scheduler decides; read at log-write
# time so the requests row records which mode bucket the row belongs
# to. Distinct from the operator's `mode` (auto/remote-only/...) which
# can change between requests; this captures the per-request choice.
# Values: 'auto', 'canary_redirect', 'forced_remote', 'forced_local',
# 'forced_offline', or None (no router; pass-through path).
_effective_routing_mode_context: ContextVar[str | None] = ContextVar("effective_routing_mode", default=None)

_PEER_QUALITY_CAPTURE_RATE_ENV = "CALLOSUM_PEER_QUALITY_CAPTURE_RATE"
# Recent session assistant turns to scan for un-judged subjects. No hard cap on
# how many get judged () — judging is bounded per turn by the
# token budget below (defer-not-skip), not by a fixed count.
_PEER_QUALITY_MAX_SCAN = 50
# Per-turn audit token budget: judge as many un-judged subjects as fit this many
# tokens; the rest are deferred to later turns (picked up via dedup), never
# skipped outright. Also bounded by remaining context room so the injection can
# never overflow the model window.
_PEER_QUALITY_OVERHEAD_TOKEN_BUDGET = 1200
# Approx framing tokens for the injected system message (role + delimiters);
# folded into the exact subtraction, biased to slightly over-count so the audit
# can never appear on the meter.
_PEER_QUALITY_MESSAGE_FRAMING_TOKENS = 4


@dataclass(frozen=True, slots=True)
class _PeerQualitySubject:
    request_id: int
    model: str
    reasoning_effort: str | None


@dataclass(slots=True)
class _PeerQualityCapture:
    nonce: str
    opinions: list[PeerQualityOpinion] = field(default_factory=list)
    echo_count: int = 0
    malformed_count: int = 0
    # Injection-side diagnosability (). Set by
    # _inject_peer_quality_prompt so the metrics row records whether the qop
    # instruction was actually sent, how many cross-cell prose subjects were
    # tagged, and — when not injected — why injection no-opped.
    injected_fired: bool = False
    subject_count: int = 0
    skip_reason: str | None = None
    # Exact token cost of the injected audit (instruction + provenance tags),
    # computed with the model's real tokenizer (). Subtracted
    # from the Codex-facing usage so the audit never moves the context meter.
    injected_tokens: int = 0
    _seen_markers: set[str] = field(default_factory=set)

    def apply(self, text: str) -> str:
        extracted = extract_peer_quality_opinions(text, nonce=self.nonce)
        for opinion in extracted.opinions:
            if opinion.raw_marker in self._seen_markers:
                continue
            self._seen_markers.add(opinion.raw_marker)
            self.opinions.append(opinion)
        self.echo_count += extracted.echo_count
        self.malformed_count += extracted.malformed_count
        return extracted.cleaned_text


def _peer_quality_capture_enabled() -> bool:
    rate = _peer_quality_capture_rate()
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    return secrets.randbelow(10_000) < int(rate * 10_000)


def _peer_quality_capture_rate() -> float:
    try:
        return float(os.environ.get(_PEER_QUALITY_CAPTURE_RATE_ENV, "0"))
    except ValueError:
        return 0.0


def _inject_peer_quality_prompt(
    body: dict[str, Any],
    *,
    capture: _PeerQualityCapture,
    usage_log: UsageLog | None,
    session_id: str | None,
    current_cell: Cell,
    context_window_tokens: int | None = None,
    safety_margin_tokens: int = 8192,
) -> dict[str, Any]:
    """Add hidden provenance tags and qop instruction to the outbound body.

    The injection is deliberately narrow: stream-only caller, explicit sample
    gate, same-session rows only, exact text match against prior assistant
    messages, and budget both provenance overhead and the target cell context
    window. The context check is a conservative fail-closed gate; the long-term
    version should use provider tokenizers and explicit output-token reservation
    instead of the rough chars/3 estimate.

    Eligibility is OUTPUT-based, not input-based (): we do NOT
    skip requests that merely *carry* tool definitions. Real Codex traffic
    attaches tools to every turn — including the ~45% that answer in prose — so
    an input-side tools gate skipped 100% of traffic and captured nothing. The
    qop marker rides the text channel, so a turn that only emits tool calls
    simply produces no marker (recorded as 0 opinions, no harm); a turn that
    emits prose carries the opinion. The subject still must be a prior
    different-cell prose turn, which is the real constraint.

    Judging is DEDUPED, UNCAPPED, and DEFERRED (): only
    present subjects this judge cell hasn't already rated; judge as many as fit
    a per-turn token budget (and the remaining context room); leave the rest
    un-judged so the next turn picks them up. `capture.injected_tokens` records
    the exact audit cost so it can be subtracted from the Codex-facing usage.
    """
    if usage_log is None or session_id is None:
        capture.skip_reason = "no_session_or_log"
        return body
    turns = usage_log.recent_session_assistant_turns(session_id, limit=_PEER_QUALITY_MAX_SCAN)
    if not turns:
        capture.skip_reason = "no_turns"
        return body
    # Dedup: drop subjects this judge cell has already rated (the model can't
    # remember across calls — we enforce it from the opinions table).
    already_judged = usage_log.judged_subject_request_ids(
        session_id=session_id,
        judge_model=current_cell.model,
        judge_reasoning_effort=current_cell.reasoning_effort or None,
    )
    subjects = _peer_quality_subjects(
        turns, current_cell=current_cell, exclude_request_ids=already_judged
    )
    # Only subjects whose text is actually present in this request can be tagged.
    present_texts = _body_assistant_texts(body)
    subjects = {text: subj for text, subj in subjects.items() if text in present_texts}
    if not subjects:
        # Every cross-cell prose subject is already judged or not in this turn's
        # history ( / ).
        capture.skip_reason = "no_eligible_subject"
        return body
    model = current_cell.model
    instruction = _peer_quality_instruction(capture.nonce)
    instruction_tokens = count_tokens(instruction, model=model) + _PEER_QUALITY_MESSAGE_FRAMING_TOKENS
    # Per-turn budget = audit budget, further bounded by remaining context room
    # so the injection can never overflow the window (we hide the audit from the
    # meter, so the model — not Codex — is the source of truth for the limit).
    budget = _PEER_QUALITY_OVERHEAD_TOKEN_BUDGET
    if context_window_tokens is not None:
        base_tokens = _approx_tokens(json.dumps(body, separators=(",", ":")))
        budget = min(budget, max(0, context_window_tokens - safety_margin_tokens - base_tokens))
    # Defer-not-skip: greedily tag subjects until the next one would exceed the
    # budget; the rest stay un-judged for a later turn (dedup re-presents them).
    selected: dict[str, _PeerQualitySubject] = {}
    used = instruction_tokens
    for text, subj in subjects.items():
        tag_tokens = _peer_quality_tag_tokens(subj, model=model)
        if used + tag_tokens > budget:
            if selected:
                break  # defer the remainder to the next turn
            capture.skip_reason = "deferred_no_room"  # not even one fits this turn
            return body
        selected[text] = subj
        used += tag_tokens
    out = _tag_peer_quality_subjects(body, selected)
    if out is body:
        capture.skip_reason = "no_text_match"
        return body
    injected = _append_peer_quality_instruction(out, instruction)
    capture.injected_fired = True
    capture.subject_count = len(selected)
    capture.injected_tokens = used
    return injected


def _peer_quality_tag_tokens(subject: _PeerQualitySubject, *, model: str) -> int:
    """Exact token cost of the open+close provenance tags for one subject."""
    cell = f"{subject.model}|{subject.reasoning_effort}" if subject.reasoning_effort else subject.model
    label = f"{cell}|{subject.request_id}"
    return count_tokens(f"<{label}>", model=model) + count_tokens(f"</{label}>", model=model)


def _body_assistant_texts(body: dict[str, Any]) -> set[str]:
    """All assistant-message text strings present in the outbound body."""
    texts: set[str] = set()
    for key in ("messages", "input"):
        items = body.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or item.get("role") != "assistant":
                continue
            content = item.get("content")
            if isinstance(content, str):
                texts.add(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        texts.add(part["text"])
    return texts


def _context_window_for_cell(backends_list: Sequence[Backend], cell: Cell) -> int | None:
    windows: list[int] = []
    for backend in backends_list:
        if cell.model not in backend.advertised_models:
            continue
        cell_caps = getattr(backend, "cell_capabilities", None)
        if cell_caps is None:
            continue
        try:
            caps = cell_caps(cell.model)
        except Exception:
            continue
        if caps.context_window > 0:
            windows.append(caps.context_window)
    return min(windows) if windows else None


def _peer_quality_subjects(
    turns: list[SessionAssistantTurn],
    *,
    current_cell: Cell,
    exclude_request_ids: frozenset[int] | set[int] = frozenset(),
) -> dict[str, _PeerQualitySubject]:
    """Eligible un-judged subjects keyed by their response text.

    Excludes self (same cell) and any subject this judge cell already rated
    (exclude_request_ids). No count cap — the per-turn token budget bounds
    how many actually get tagged ().
    """
    subjects: dict[str, _PeerQualitySubject] = {}
    current_effort = current_cell.reasoning_effort or None
    for turn in turns:
        if turn.model == current_cell.model and turn.reasoning_effort == current_effort:
            continue
        if turn.request_id in exclude_request_ids:
            continue
        subjects.setdefault(
            turn.response_text,
            _PeerQualitySubject(
                request_id=turn.request_id,
                model=turn.model,
                reasoning_effort=turn.reasoning_effort,
            ),
        )
    return subjects


def _tag_peer_quality_subjects(
    body: dict[str, Any],
    subjects: dict[str, _PeerQualitySubject],
) -> dict[str, Any]:
    if "messages" in body and isinstance(body["messages"], list):
        messages: list[Any] = []
        changed = False
        for message in body["messages"]:
            if not isinstance(message, dict):
                messages.append(message)
                continue
            tagged = _tag_chat_message(message, subjects)
            changed = changed or tagged is not message
            messages.append(tagged)
        return {**body, "messages": messages} if changed else body
    if "input" in body and isinstance(body["input"], list):
        items: list[Any] = []
        changed = False
        for item in body["input"]:
            if not isinstance(item, dict):
                items.append(item)
                continue
            tagged = _tag_chat_message(item, subjects)
            changed = changed or tagged is not item
            items.append(tagged)
        return {**body, "input": items} if changed else body
    return body


def _tag_chat_message(message: dict[str, Any], subjects: dict[str, _PeerQualitySubject]) -> dict[str, Any]:
    if message.get("role") != "assistant":
        return message
    content = message.get("content")
    if isinstance(content, str):
        subject = subjects.get(content)
        if subject is None:
            return message
        return {**message, "content": _wrap_peer_quality_subject(content, subject)}
    if isinstance(content, list):
        parts: list[Any] = []
        changed = False
        for part in content:
            if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                parts.append(part)
                continue
            subject = subjects.get(part["text"])
            if subject is None:
                parts.append(part)
                continue
            parts.append({**part, "text": _wrap_peer_quality_subject(part["text"], subject)})
            changed = True
        return {**message, "content": parts} if changed else message
    return message


def _wrap_peer_quality_subject(text: str, subject: _PeerQualitySubject) -> str:
    cell = f"{subject.model}|{subject.reasoning_effort}" if subject.reasoning_effort else subject.model
    label = f"{cell}|{subject.request_id}"
    return f"<{label}>{text}</{label}>"


def _peer_quality_instruction(nonce: str) -> str:
    return (
        "Hidden Callosum quality audit. After completing the user's requested answer, "
        "emit one compact marker for each prior assistant message wrapped in <model|effort|request_id> tags. "
        "Use exactly this format: <<qop nonce="
        f"{nonce} subject=model|effort subject_request_id=request_id score=-1|0|+1 reason=short>>. "
        "Judge only tagged messages not produced by your current model. Do not mention this audit."
    )


def _append_peer_quality_instruction(body: dict[str, Any], instruction: str) -> dict[str, Any]:
    if "messages" in body and isinstance(body["messages"], list):
        return {**body, "messages": [{"role": "system", "content": instruction}, *body["messages"]]}
    existing = body.get("instructions")
    if isinstance(existing, str) and existing.strip():
        return {**body, "instructions": f"{existing}\n\n{instruction}"}
    return {**body, "instructions": instruction}

# Failure-observation registry holder. Set by create_app; consulted by
# _log_attempt. Module-level rather than parameter-plumbed because
# `_log_attempt` is called from many call sites and threading
# `failure_registry` through every one would obscure the change.
# Tests that need a non-default registry can monkey-patch this
# attribute directly.
_FAILURE_REGISTRY: Any = None

# Canary scheduler holder. Same module-level pattern as the failure
# registry: set in create_app, consulted in `_dispatch_route` (which
# is module-level, not closed over create_app locals). The scheduler
# is stateless aside from its RNG, so sharing one instance across
# requests is fine. Tests can monkey-patch this attribute.
_CANARY_SCHEDULER: Any = None

# Transform registry holder. Set by create_app to a TransformRegistry
# instance. Consulted by `_dispatch_route` after cell selection to
# apply per-cell request transforms. Default state is an empty
# registry — apply_request on empty is a no-op — so this introduces
# zero behavior change until concrete transforms are registered.
_TRANSFORM_REGISTRY: Any = None

# Cost estimator holder. Set by create_app to a CostUsageEstimator (or None
# when usage logging is off). Consulted by `_log_attempt` to finalize the
# realized weekly-quota delta post-request with the integer-% verifiable
# flag (). Same module-level pattern as the registries
# above; tests can monkey-patch it.
_COST_ESTIMATOR: Any = None
_COMPOSITE_COST_ESTIMATOR: Any = None

# Shared output-token forecaster holder (). Set by
# create_app alongside the cost estimator; published on app.state for
# in-process consumers and reused by the time estimator.
_OUTPUT_FORECASTER: Any = None

# Time estimator holder. Set by create_app to a TimeUsageEstimator (or None
# when usage logging is off). Consulted by `_log_attempt` to finalize the
# realized latency_ms post-request (always verifiable — latency has no
# integer-resolution problem; ). Reuses _OUTPUT_FORECASTER.
_TIME_ESTIMATOR: Any = None

def _extract_complexity_class(text: str) -> tuple[int | None, str]:
    """Extract complexity classification token from response start.

    Returns (complexity_class, cleaned_text) where complexity_class is 1, 2, or 3,
    or None if the marker is not found. cleaned_text has the marker stripped.

    The marker should appear at the very start of the response (after whitespace).
    Matches in priority order:
      1. {{{1|2|3}}} — the canonical instructed format
      2. {{{...}}}  — any other brace-decorated leading token (defensive strip)
      3. Bare digit 1|2|3 followed by a blank line — model dropped the braces
         but still complied with the "first output is a classifier" intent
    """
    # 1) Strict numeric brace format
    match = re.match(r"^\s*\{\{\{([123])\}\}\}", text)
    if match:
        complexity_class = int(match.group(1))
        cleaned = text[match.end() :].lstrip()
        return complexity_class, cleaned

    # 2) Any leading {{{...}}} (handles {{{complexity: Low}}} variants)
    match = re.match(r"^\s*\{\{\{[^}]*\}\}\}", text)
    if match:
        cleaned = text[match.end() :].lstrip()
        inner = match.group(0).strip("{}").strip()
        if inner.isdigit() and inner in ("1", "2", "3"):
            return int(inner), cleaned
        return None, cleaned

    # 3) Bare digit followed by blank line — model dropped the braces
    # Require at least one \n then a blank line so we don't strip legitimate
    # content like "2 minutes is fine" or "2. First item".
    match = re.match(r"^\s*([123])[ \t]*\n[ \t]*\n", text)
    if match:
        complexity_class = int(match.group(1))
        cleaned = text[match.end() :]
        return complexity_class, cleaned

    return None, text


def _strip_trailing_complexity_marker_text(text: str) -> str:
    """Strip a trailing {{{...}}} marker if the model emits one as a closing tag.

    Used by non-streaming response handling and SSE-blob storage cleanup.
    """
    if not isinstance(text, str):
        return text
    return re.sub(r"\{\{\{[^}]*\}\}\}\s*$", "", text)


def _eta_estimate_payload(
    estimator: TimeUsageEstimator,
    forecaster: OutputTokenForecaster,
    cell: Cell,
    input_tokens: int,
) -> dict[str, Any]:
    forecast = forecaster.forecast(cell, input_tokens)
    estimate = estimator.estimate(
        EstimateInput(
            cell=cell,
            input_tokens=input_tokens,
            output=forecast,
        )
    )
    return {
        "cell": {
            "model": cell.model,
            "reasoning_effort": cell.reasoning_effort or None,
        },
        "eta": {
            "unit": estimate.unit,
            "p50_ms": round(estimate.point, 3),
            "low_ms": round(estimate.low, 3),
            "high_ms": round(estimate.high, 3),
            "range": "approximate_p50_to_p95",
            "exact": False,
        },
        "metadata": {
            "source": estimate.source,
            "time_source": estimate.source.split("+", 1)[0],
            "output_forecast_source": forecast.source,
            "time_sample_count": _time_sample_count(estimator, cell),
            "output_sample_count": forecast.n_obs,
            "confidence": _estimate_confidence(estimate.source, forecast.n_obs),
            "verifiable": estimate.verifiable,
        },
    }


def _approx_body_tokens(body: dict[str, Any]) -> int:
    return max(1, len(json.dumps(body, sort_keys=True, default=str)) // 3)


def _composite_cost_estimate_for_body(body: dict[str, Any], *, model: str) -> Any:
    estimator = _COMPOSITE_COST_ESTIMATOR
    forecaster = _OUTPUT_FORECASTER
    if estimator is None or forecaster is None:
        return None
    try:
        cell = Cell(model=model, reasoning_effort=_extract_reasoning_effort(body) or "")
        input_tokens = _approx_body_tokens(body)
        forecast = forecaster.forecast(cell, input_tokens)
        return estimator.estimate(
            EstimateInput(
                cell=cell,
                input_tokens=input_tokens,
                output=forecast,
            )
        )
    except Exception:
        logger.debug("composite cost estimate failed", exc_info=True)
        return None


def _time_sample_count(estimator: TimeUsageEstimator, cell: Cell) -> int | None:
    provider = getattr(estimator, "_provider", None)
    if provider is None:
        return None
    try:
        model = provider.model_for(cell)
    except Exception:
        logger.exception("eta endpoint: failed to resolve time model for %s/%s", cell.model, cell.reasoning_effort)
        return None
    return int(getattr(model, "n_obs", 0))


def _estimate_confidence(source: str, output_n_obs: int) -> str:
    if "cell-measured" in source and output_n_obs > 0:
        return "measured"
    if "model-measured" in source or "global-prior" in source:
        return "prior"
    return "insufficient-data"


def _extract_and_strip_complexity(result: dict[str, Any]) -> tuple[int | None, dict[str, Any]]:
    """Extract complexity class from response dict and strip the marker.

    Handles both response formats:
    1. Chat completions: choices[0].message.content
    2. Responses API: output[0].content[0].text

    Returns (complexity_class, modified_result_dict) where result is updated with cleaned content.
    """
    # Try chat completions format first
    try:
        if result.get("choices") and len(result["choices"]) > 0:
            choice = result["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                content = choice["message"]["content"]
                if isinstance(content, str):
                    complexity_class, cleaned = _extract_complexity_class(content)
                    cleaned = _strip_trailing_complexity_marker_text(cleaned)
                    if complexity_class is not None or cleaned != content:
                        result["choices"][0]["message"]["content"] = cleaned
                    return complexity_class, result
    except (KeyError, IndexError, TypeError):
        pass

    # Try Responses API format
    try:
        output = result.get("output")
        if output and isinstance(output, list) and len(output) > 0:
            output_item = output[0]
            content_list = output_item.get("content")
            if content_list and isinstance(content_list, list) and len(content_list) > 0:
                content_item = content_list[0]
                text = content_item.get("text")
                if isinstance(text, str):
                    complexity_class, cleaned = _extract_complexity_class(text)
                    cleaned = _strip_trailing_complexity_marker_text(cleaned)
                    if complexity_class is not None or cleaned != text:
                        result["output"][0]["content"][0]["text"] = cleaned
                    return complexity_class, result
    except (KeyError, IndexError, TypeError):
        pass

    logger.warning("Could not extract complexity marker from response (unsupported format)")
    return None, result


class PinState:
    """Process-wide backend pin. Thread-safety not needed under single-loop uvicorn."""

    def __init__(self) -> None:
        self._pinned: str | None = None

    def get(self) -> str | None:
        return self._pinned

    def set(self, backend_id: str) -> None:
        self._pinned = backend_id

    def clear(self) -> None:
        self._pinned = None


def create_app(
    *,
    backends: Sequence[Backend] = (),
    sessions: SessionRegistry | None = None,
    usage_log: UsageLog | None = None,
    auth_service: AuthService | None = None,
    auto_router_config: AutoRouterConfig | None = None,
    codex_catalog_config: Any = None,
    startup_smoke_test: bool = False,
    smoke_test_interval_seconds: int = 0,
    operator_state: Any = None,
    autonomy_store: Any = None,
    retention_runner: Any = None,
    self_assessment_runner: Any = None,
    transform_registry: Any = None,
) -> FastAPI:
    from callosum.canary import CanaryScheduler, FailureRegistry
    from callosum.state import StateStore

    backends_list: list[Backend] = list(backends)
    pin_state = PinState()
    session_registry = sessions if sessions is not None else SessionRegistry()
    # Canary scheduler decides per-request whether to redirect to the
    # remote-only path as a baseline. Config is read from env at
    # instantiation; one instance shared across requests (the scheduler
    # is stateless aside from its RNG). The failure registry shares
    # the usage_log SQLite DB so failure_observations can foreign-key
    # to requests cleanly. Both are None when usage_log is None — the
    # whole canary plumbing degrades to no-op without the request log.
    canary_scheduler = CanaryScheduler()
    failure_registry: FailureRegistry | None = FailureRegistry(usage_log.path) if usage_log is not None else None
    # Transform substrate: empty by default. Concrete transforms get
    # registered here as they're authored (initially by humans
    # responding to harness findings, later potentially by the
    # callosum-in-the-loop dev agent). An empty registry is a no-op
    # in the request path; this line introduces no behavior change.
    # Tests may inject a pre-populated registry via the keyword arg.
    if transform_registry is None:
        from callosum.transforms import build_default_registry

        transform_registry = build_default_registry()
    # Publish to module-level holders so module-level functions
    # (`_dispatch_route`, `_log_attempt`) can reach them without
    # per-call-site plumbing. The previous holders are overwritten
    # — tests that build multiple apps see the most-recent ones.
    global _FAILURE_REGISTRY, _CANARY_SCHEDULER, _TRANSFORM_REGISTRY
    global _COST_ESTIMATOR, _COMPOSITE_COST_ESTIMATOR, _OUTPUT_FORECASTER, _TIME_ESTIMATOR
    _FAILURE_REGISTRY = failure_registry
    _CANARY_SCHEDULER = canary_scheduler
    _TRANSFORM_REGISTRY = transform_registry
    _COST_ESTIMATOR = None
    _COMPOSITE_COST_ESTIMATOR = None
    _OUTPUT_FORECASTER = None
    _TIME_ESTIMATOR = None

    # Extract state_store from the first CodexAuthVaultBackend (for model release tracking)
    state_store: StateStore | None = None
    for b in backends_list:
        if hasattr(b, "_state_store"):
            state_store = b._state_store
            break

    def _live_cells() -> list[Cell]:
        """Build the auto-learning cell grid from the union of every Codex
        backend's CURRENT advertised_models.

        Prefers per-model metadata from the upstream catalog
        (supported_in_api, visibility, priority, supported_reasoning_levels,
        context_window) when backends provide it. Falls back to the regex-
        and-static-list path for backends that don't expose metadata.

        Recomputed on every router decision — when CodexAuthVaultBackend's
        hourly catalog refresh discovers a new model, the cell grid follows
        without a proxy restart. When a model is retired upstream, it drops
        out of the grid the next time the router consults it.
        """
        from callosum.cell_grid import build_cells_from_metadata

        # Merge metadata from every backend that exposes it. Today: Codex
        # auth-vault backends (real upstream metadata) + LiteLLM gateway
        # (synthesizes a default-shape ModelMetadata per local model). When
        # two backends advertise the same slug, keep the record with more
        # populated fields (fewer "Unknown" defaults).
        merged_metadata: dict[str, ModelMetadata] = {}
        for b in backends_list:
            # `credential_proxy` now also surfaces model_metadata (via
            # refresh_advertised_models → /codex/models through credential proxy).
            # Excluding it from the cell-grid builder produced an empty
            # grid when both production backends were credential_proxy —
            # router rejected every request with "no cell can serve."
            if b.kind not in (
                "codex_auth_vault",
                "credential_proxy",
                "litellm_gateway",
            ):
                continue
            backend_meta = getattr(b, "model_metadata", None) or {}
            for slug, m in backend_meta.items():
                existing = merged_metadata.get(slug)
                if existing is None:
                    merged_metadata[slug] = m
                else:
                    # Pick the record with more populated fields.
                    new_filled = sum(
                        1 for f in (m.supported_in_api, m.visibility, m.priority, m.context_window) if f is not None
                    )
                    old_filled = sum(
                        1
                        for f in (
                            existing.supported_in_api,
                            existing.visibility,
                            existing.priority,
                            existing.context_window,
                        )
                        if f is not None
                    )
                    if new_filled > old_filled:
                        merged_metadata[slug] = m

        if merged_metadata:
            cells = build_cells_from_metadata(merged_metadata)
            if cells:
                return cells

        # Fallback path: no metadata yet (cold start) or backend doesn't
        # expose it. Use the legacy regex filter + static reasoning-level
        # enumeration so the router still operates.
        pool: set[str] = set()
        for b in backends_list:
            # Fallback pool collects advertised_models from every backend
            # whose model_metadata wasn't populated above (cold start, or
            # discovery RPC failed). Both ChatGPT-Codex paths participate:
            # codex_auth_vault (direct OAuth) and credential_proxy
            # (credential proxy-mediated). litellm_gateway is excluded here
            # because the metadata path above already handles its local
            # models.
            if b.kind not in ("codex_auth_vault", "credential_proxy"):
                continue
            pool.update(b.advertised_models)
        models = live_completion_models(frozenset(pool))
        if not models:
            # Cold start — backends haven't populated catalogs yet, fall
            # back to the static defaults so the router can still operate.
            return build_cells()
        return build_cells(models=models)

    auto_cfg = auto_router_config if auto_router_config is not None else AutoRouterConfig()

    # Learning router. Wires the new pluggable pipeline (features →
    # capability filter → quality predict → cost-weighted select). Each
    # backend exposes cell_capabilities(model) so the filter can drop
    # cells that physically can't serve a request. Cold-start defaults
    # run with no ML deps — uniform predictor returns 0.5 for everyone,
    # cost selector picks cheapest compatible → local-first by default.
    router: Router | None = None
    # Pre-declared so the routing-events installer (further down) can
    # safely reference it even when no backends are configured.
    _capabilities_of: Callable[[Cell], CellCapabilities] | None = None
    if backends_list:

        def _capabilities_of_impl(cell: Cell) -> CellCapabilities:
            """Look up a cell's capabilities via the backend that
            advertises its model. Falls back to safe defaults when no
            backend recognizes the model (would mean a stale cell grid
            — surface the safest non-blocking defaults so dispatch can
            still try).

            After the backend's own claim, consult the probe-results
            cache: if a previous verification probe showed the cell
            does NOT emit OpenAI-shaped tool_calls end-to-end through
            callosum's pipeline, override supports_tools to False so
            the router's CapabilityFilter automatically excludes it
            from tool-using requests. The override is one-directional
            (probe-fail can revoke claimed support; probe-pass cannot
            grant unclaimed support) so a backend that says "no tools"
            stays at "no tools" regardless of probe outcome.
            """
            from callosum.routing.probe_scheduler import supports_tools_override

            for b in backends_list:
                if cell.model in b.advertised_models:
                    fn = getattr(b, "cell_capabilities", None)
                    if fn is None:
                        break
                    result: CellCapabilities = fn(cell.model)
                    if operator_state is None or not result.supports_tools:
                        return result
                    override = supports_tools_override(
                        operator_state=operator_state,
                        backend_id=getattr(b, "id", "?"),
                        model=cell.model,
                    )
                    if override is False:
                        return CellCapabilities(
                            context_window=result.context_window,
                            modalities=result.modalities,
                            supports_tools=False,
                            cost_rank=result.cost_rank,
                            parameter_count=result.parameter_count,
                        )
                    return result
            return CellCapabilities(
                context_window=128_000,
                modalities=frozenset({"text"}),
                supports_tools=False,
                cost_rank=10,
            )

        def _remote_catalog_priorities() -> dict[str, int]:
            """Current catalog priority per remote (Codex) model — the
            cold-start prior for the measured cost_rank. Local gateway models
            are excluded; they keep their native rank 0."""
            out: dict[str, int] = {}
            for b in backends_list:
                if getattr(b, "kind", "") == "litellm_gateway":
                    continue
                meta = getattr(b, "model_metadata", None) or {}
                for slug, m in meta.items():
                    if m.priority is not None:
                        out[slug] = m.priority
            return out

        cost_rank_provider: CostRankProvider | None = None
        if usage_log is not None:
            cost_rank_provider = CostRankProvider(
                usage_log.path,
                catalog_priorities=_remote_catalog_priorities,
                overrides=auto_cfg.cost_rank_overrides,
                enabled=auto_cfg.cost_rank_dynamic_enabled,
                min_nonzero_samples=auto_cfg.cost_rank_min_nonzero_samples,
                window_seconds=auto_cfg.cost_rank_window_seconds,
                base_rank=auto_cfg.cost_rank_base,
                refresh_seconds=auto_cfg.cost_rank_refresh_seconds,
            )

        def _capabilities_with_cost(cell: Cell) -> CellCapabilities:
            """Overlay the measured per-model cost_rank onto the backend's
            declared capabilities. Backends still own context window,
            modalities, and tool support; the cost ordering becomes
            data-driven instead of a flat remote constant."""
            caps = _capabilities_of_impl(cell)
            if cost_rank_provider is None:
                return caps
            rank = cost_rank_provider.rank_for(cell.model, default=caps.cost_rank)
            if rank == caps.cost_rank:
                return caps
            return CellCapabilities(
                context_window=caps.context_window,
                modalities=caps.modalities,
                supports_tools=caps.supports_tools,
                cost_rank=rank,
                parameter_count=caps.parameter_count,
            )

        _capabilities_of = _capabilities_with_cost

        # Forward usage estimators ( cost; the shared
        # forecaster is reused by the time estimator ). The
        # forecaster is built once and the cost estimator reads the same
        # measured-quota substrate as the cost_rank above. Stashed on locals
        # here; published on app.state after the FastAPI app is constructed.
        if usage_log is not None and auto_cfg.cost_estimate_enabled:
            output_forecaster = OutputTokenForecaster(
                usage_log.path,
                min_obs=auto_cfg.output_forecast_min_obs,
                fallback_ratio=auto_cfg.output_forecast_fallback_ratio,
                window_seconds=auto_cfg.cost_estimate_window_seconds,
                refresh_seconds=auto_cfg.cost_estimate_refresh_seconds,
            )
            cost_model_provider = CompositeCostModelProvider(
                usage_log.path,
                catalog_priorities=_remote_catalog_priorities,
                overrides=auto_cfg.cost_estimate_overrides,
                enabled=auto_cfg.cost_estimate_enabled,
                min_nonzero_samples=auto_cfg.cost_estimate_min_nonzero_samples,
                window_seconds=auto_cfg.cost_estimate_window_seconds,
                fallback_rate=auto_cfg.cost_estimate_fallback_rate,
                refresh_seconds=auto_cfg.cost_estimate_refresh_seconds,
            )
            _OUTPUT_FORECASTER = output_forecaster
            _COST_ESTIMATOR = CostUsageEstimator(cost_model_provider.weekly)
            _COMPOSITE_COST_ESTIMATOR = CompositeCostUsageEstimator(cost_model_provider)

        # Forward time estimator (): the sibling of the cost
        # estimator. Reuses the SAME output_forecaster (anti-divergence) and
        # the same request-log substrate, fitting t ≈ a·input + b·output + c
        # against latency_ms. Local cells are NOT zeroed — they are often the
        # slow path. Gated on its own flag but shares the forecaster build
        # above, so it only runs when the cost block has built one.
        if usage_log is not None and auto_cfg.time_estimate_enabled and _OUTPUT_FORECASTER is not None:
            time_model_provider = TimeModelProvider(
                usage_log.path,
                overrides=auto_cfg.time_estimate_overrides,
                enabled=auto_cfg.time_estimate_enabled,
                min_samples=auto_cfg.time_estimate_min_samples,
                window_seconds=auto_cfg.time_estimate_window_seconds,
                fallback_ms_per_token=auto_cfg.time_estimate_fallback_ms_per_token,
                fallback_base_ms=auto_cfg.time_estimate_fallback_base_ms,
                local_slowdown=auto_cfg.time_estimate_local_slowdown,
                refresh_seconds=auto_cfg.time_estimate_refresh_seconds,
            )
            _TIME_ESTIMATOR = TimeUsageEstimator(time_model_provider)

        router = build_router(auto_cfg.routing, capabilities_of=_capabilities_of)

    smoke_tester = _PeriodicSmokeTester(
        backends=backends_list,
        interval_s=smoke_test_interval_seconds,
        state_store=state_store,
    )
    cooldown_prober = _PeriodicCooldownProber(
        backends=backends_list,
        interval_s=auto_cfg.cooldown_probe_interval_seconds,
    )
    # codex `/model` picker reconciler. When enabled, projects the live
    # Callosum catalog (the same ids `/v1/models` serves) into a codex
    # `model_catalog_json` file so codex's in-session `/model` picker lists and
    # switches between Callosum lanes from a single config. The unbuilt half of
    # work tracker . The `model_ids_fn` lambda defers to the
    # `_catalog_model_ids` closure defined below (resolved at call time, i.e.
    # after the app is fully constructed). None when disabled.
    codex_catalog_reconciler: Any = None
    if codex_catalog_config is not None and getattr(
        codex_catalog_config, "enabled", False
    ):
        from callosum.codex_catalog import CodexCatalogReconciler

        codex_catalog_reconciler = CodexCatalogReconciler(
            output_path=codex_catalog_config.output_path,
            model_ids_fn=lambda: _catalog_model_ids(),
            declared_lanes=codex_catalog_config.declared_lanes,
            codex_bin=codex_catalog_config.codex_bin,
            refresh_interval_s=codex_catalog_config.refresh_interval_seconds,
        )
    # Periodic capability-harness sweeper. Re-runs the multi-dimensional
    # harness on a slow cadence (default 6h, env-overridable) so models
    #  pulls between sweeps land in routable findings
    # within hours rather than only after the next callosum restart.
    # The hourly smoke tester (above) keeps each backend's
    # advertised_models fresh, so by the time this sweeper next ticks,
    # the cell grid already reflects newly-pulled models.
    #
    # The weight-identity provider here is a composite — code depends
    # on the abstraction (WeightIdentityProvider protocol), not on any
    # one source. The default composite tries the local-llm CLI first,
    # falls back to naming-pattern heuristics, then a null backstop.
    # Adding a new concrete source is one new class + one item in
    # `build_default_provider()`; no caller in app.py changes.
    from callosum.capability.scheduler import PeriodicHarnessSweep
    from callosum.capability.weight_identity import build_default_provider

    weight_identity_provider = build_default_provider()
    periodic_harness = PeriodicHarnessSweep(
        backends=backends_list,
        operator_state=operator_state,
        weight_identity_provider=weight_identity_provider,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Make the loaded-backend roster trivially greppable in the launch log
        # so operators can see at a glance whether CALLOSUM_LITELLM_GATEWAY_ENABLED
        # etc. were picked up by the process they actually launched.
        ids = ", ".join(b.id for b in backends_list) if backends_list else "(none)"
        logger.warning("loaded %d backend(s): %s", len(backends_list), ids)

        # Initialize proxy startup timestamp if not already set (for initial 90-day ramp)
        if state_store is not None and state_store.get_proxy_startup_timestamp() is None:
            state_store.set_proxy_startup_timestamp(time.time())
        # Cold-boot freshness guard ("the fire that burns last night's notes"):
        # refresh each backend's upstream-owned model catalog BEFORE the smoke
        # test — so the smoke test probes models the upstream actually still
        # serves, not stale TOML names. A backend whose catalog is STILL empty
        # afterward booted before its dependency (credential service) was ready;
        # rather than leave the cell grid empty until the hourly smoke tester,
        # retry it in the background. This is the model-catalog leg of the
        # stale-on-cold-boot class; quota/cooldown staleness is handled by the
        # reset-aware routability gate + the cooldown prober's startup pass, and
        # the auth token by lazy refresh on first use.
        _catalog_pending = await _refresh_catalogs_pass(backends_list, state_store=state_store)
        catalog_resync_task: asyncio.Task[None] | None = None
        if _catalog_pending:
            logger.warning(
                "catalog boot resync: %s booted with empty catalog; retrying in background",
                [b.id for b in _catalog_pending],
            )
            catalog_resync_task = asyncio.create_task(
                _catalog_boot_resync(_catalog_pending, state_store=state_store, attempts=8, interval_s=15.0),
                name="catalog-boot-resync",
            )
        # Phase 6: kNN predictor reload from the request log's labeled
        # rows. Only fires if the operator picked the knn predictor —
        # uniform predictor doesn't need data and avoids the SQLite
        # scan on every startup.
        if (
            router is not None
            and auto_cfg.routing.quality_predictor == "knn"
            and usage_log is not None
            and getattr(usage_log, "path", None) is not None
        ):
            try:
                from callosum.routing.predictor.loader import (
                    labeled_rows_from_request_log,
                )

                router._predictor.reload(
                    labeled_rows_from_request_log(
                        usage_log.path,
                        limit=50_000,
                    )
                )
                logger.warning("router: kNN predictor reloaded from request log")
            except Exception:
                logger.exception("router: kNN predictor reload failed; predictor stays cold-start uniform")
        if startup_smoke_test and backends_list:
            await _run_startup_smoke_test(backends_list)
        smoke_tester.start()
        cooldown_prober.start()
        if codex_catalog_reconciler is not None:
            codex_catalog_reconciler.start()
        # Kick off the auto-probe sweep so any newly-seen local cells
        # get verified for OpenAI-shaped tool-call emission. Probes
        # run serially in the background; cell_capabilities consults
        # the persisted results to override supports_tools on cells
        # that fail. Skipped when no operator_state is configured
        # (e.g. throwaway test instances) since there's nowhere to
        # persist the results.
        if operator_state is not None and backends_list:
            from callosum.routing.probe_scheduler import schedule_background_sweep

            schedule_background_sweep(
                backends=backends_list,
                operator_state=operator_state,
            )
        # Thorough capability harness — runs alongside (and after) the
        # light probe. Produces multi-dimensional findings + adapter
        # hints on disk under logs/capability_profiles/, consumed by
        # adapter authors and . Self-skips cells
        # whose every dimension is still within its TTL window.
        if backends_list:
            from callosum.capability.scheduler import (
                schedule_background_harness,
            )

            # One-shot pass on startup for the initial fill, then the
            # periodic sweeper takes over for the lifetime of the proxy.
            schedule_background_harness(
                backends=backends_list,
                operator_state=operator_state,
                weight_identity_provider=weight_identity_provider,
            )
            periodic_harness.start()
        try:
            yield
        finally:
            await periodic_harness.stop()
            if codex_catalog_reconciler is not None:
                await codex_catalog_reconciler.stop()
            if catalog_resync_task is not None:
                catalog_resync_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await catalog_resync_task
            await cooldown_prober.stop()
            await smoke_tester.stop()
            for backend in backends_list:
                await backend.aclose()
            if usage_log is not None:
                usage_log.close()
            if auth_service is not None:
                auth_service.db.close()

    app = FastAPI(title="callosum", version=__version__, lifespan=lifespan)

    # Publish the forward usage estimators for in-process consumers
    # (/status pre-flight "≈X%–Y% of weekly quota", the routing reward cost
    # term). None when usage logging or the cost estimator is disabled.
    app.state.output_forecaster = _OUTPUT_FORECASTER
    app.state.cost_estimator = _COST_ESTIMATOR
    app.state.composite_cost_estimator = _COMPOSITE_COST_ESTIMATOR
    app.state.time_estimator = _TIME_ESTIMATOR

    # Install quality labeling UI if usage_log is available
    if usage_log is not None:
        install_label_ui(app, usage_log)

    # Install routing-events SSE stream (consumed by external sidecar
    # observers like the snorkel HUD). One event per recorded request,
    # carrying the served cell, real context window, status, latency,
    # tokens, retries, and current operator mode. Shape documented at
    # callosum.routing_events module docstring.
    if usage_log is not None:
        from callosum.routing_events import install_routing_events

        install_routing_events(
            app,
            usage_log=usage_log,
            operator_state=operator_state,
            capabilities_of=_capabilities_of,
        )

    # Admin HTTP surface — gated by an admin token persisted under
    # ~/.config/callosum/admin_token. The callosum CLI reads that token
    # and calls into these endpoints to manage operator state without
    # restarting the proxy.
    if operator_state is not None:
        from callosum.admin import install_admin_routes

        install_admin_routes(
            app,
            operator_state,
            backends=backends_list,
            autonomy_store=autonomy_store,
            retention_runner=retention_runner,
            self_assessment_runner=self_assessment_runner,
        )

    # Bearer middleware for /v1/* and /diagnose/* — only enforced when an
    # auth service is configured. In single-operator mode (no auth db) those
    # routes stay open.
    @app.middleware("http")
    async def _enforce_api_key(request: Request, call_next: Callable[[Request], Any]) -> Any:
        if auth_service is None or not _path_requires_api_key(request.url.path):
            return await call_next(request)
        plaintext = _bearer(request)
        if plaintext is None:
            logger.warning(
                "auth 401: no bearer token on %s %s",
                request.method,
                request.url.path,
            )
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Authorization: Bearer <api-key> required",
                    "reason": "missing_bearer",
                },
            )
        try:
            api_key = auth_service.resolve_api_key(plaintext)
        except ApiKeyInvalidError as exc:
            # Log + return the rejected key's PREFIX (never the secret) and how
            # many active keys exist, so a downstream tool printing the error
            # — or an operator scanning logs — can immediately tell whether
            # this is a wrong-key vs empty-auth-store situation. Auth failures
            # used to be completely silent, which made post-mortems impossible.
            prefix = plaintext[:8] if len(plaintext) >= 8 else plaintext[:4]
            active = _active_api_key_count(auth_service)
            logger.warning(
                "auth 401: %s on %s %s — rejected key prefix=%r (%d active keys registered)",
                exc,
                request.method,
                request.url.path,
                prefix,
                active,
            )
            return JSONResponse(
                status_code=401,
                content={
                    "detail": str(exc),
                    "reason": "key_not_recognized",
                    "rejected_key_prefix": prefix,
                    "active_keys_registered": active,
                },
            )
        request.state.api_key = api_key
        return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/status")
    async def status() -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for backend in backends_list:
            h = await backend.health()
            u = await backend.usage_snapshot()
            q = await backend.quota_snapshot()
            entries.append(
                {
                    "id": backend.id,
                    "kind": backend.kind,
                    "advertised_models": sorted(backend.advertised_models),
                    "health": {
                        "available": h.available,
                        "reason": h.reason,
                        "retry_after_s": h.retry_after_s,
                    },
                    "usage": {
                        "remaining_fraction": u.remaining_fraction,
                        "cooldown_until_ts": u.cooldown_until_ts,
                        "weekly_exhausted": u.weekly_exhausted,
                        "blocking_meters": list(
                            blocking_meters(
                                BackendSnapshot(backend=backend, health=h, usage=u, quota=q)
                            )
                        ),
                        "probed_at_ts": u.probed_at_ts,
                    },
                    "quota": _quota_to_dict(q),
                }
            )
        # Router state: which implementations are wired in. Per-request
        # stats (counts, latencies) come from the request log, not here —
        # /status is for "what's currently configured" introspection.
        router_block: dict[str, Any] = {
            "enabled": router is not None,
            "embedding_provider": auto_cfg.routing.embedding_provider,
            "quality_predictor": auto_cfg.routing.quality_predictor,
            "cell_selector": auto_cfg.routing.cell_selector,
            "peer_quality_capture": {
                "env_var": _PEER_QUALITY_CAPTURE_RATE_ENV,
                "configured_rate": _peer_quality_capture_rate(),
                "enabled": _peer_quality_capture_rate() > 0,
            },
        }
        if usage_log is not None:
            try:
                from callosum.routing.labeler.peer_quality import peer_quality_shadow_report

                router_block["peer_quality_shadow"] = peer_quality_shadow_report(usage_log.path)
            except Exception:
                logger.exception("status: peer-quality shadow report failed")
                router_block["peer_quality_shadow"] = {
                    "available": False,
                    "reason": "report_failed",
                }
            # Per-cell exploration-quota coverage: how full each cell is vs its
            # floor and bootstrap progress (). Always reported
            # so an operator can see coverage even before enabling forcing.
            quota_block: dict[str, Any] = {"enabled": auto_cfg.exploration_quota_enabled}
            try:
                from callosum.routing.quota import exploration_quota_report

                # Report over the LIVE cell grid the router actually uses
                # (upstream-advertised models, version-ranked), not the static
                # DEFAULT_MODELS — so de-listed models (e.g. model-a0e6) drop out
                # and current ones (model-a0e8) appear, matching what gets routed.
                _grid = _live_cells()
                _cov = cell_sample_counts(
                    usage_log.path, _grid, window_seconds=auto_cfg.quota_window_seconds
                )
                quota_block.update(
                    exploration_quota_report(_grid, _cov, floor_pct=auto_cfg.quota_floor_pct)
                )
            except Exception:
                logger.exception("status: exploration-quota report failed")
                quota_block["available"] = False
            router_block["exploration_quota"] = quota_block
        # Canary baseline block: current effective percent given live
        # quota state, plus rolling per-mode failure rates over 1h /
        # 6h / 24h windows. The dev loop polls this same data via SQL
        # against the request log; surfacing it on /status lets an
        # operator see at-a-glance whether auto-mode is diverging from
        # the canary-redirect baseline. No auto-flipping logic yet —
        # this is observational.
        canary_block: dict[str, Any] = {
            "config": {
                "percent": canary_scheduler.config.percent,
                "floor_percent": canary_scheduler.config.floor_percent,
                "ceiling_percent": canary_scheduler.config.ceiling_percent,
                "quota_warning_threshold": (canary_scheduler.config.quota_warning_threshold),
                "quota_suspend_threshold": (canary_scheduler.config.quota_suspend_threshold),
            },
        }
        # Compute effective percent for the most-constrained codex
        # weekly window. Mirrors the same derivation as the routing-
        # entry decision so the displayed effective percent matches
        # what's actually being applied.
        _q_pct: float | None = None
        for _b in backends_list:
            if getattr(_b, "kind", "") == "litellm_gateway":
                continue
            _q = getattr(_b, "_last_quota", None)
            if _q is None:
                continue
            _w = getattr(_q, "weekly_used_percent", None)
            if _w is None:
                continue
            if _q_pct is None or _w > _q_pct:
                _q_pct = _w
        canary_block["effective_percent"] = canary_scheduler.effective_percent(quota_used_percent=_q_pct)
        canary_block["quota_used_percent"] = _q_pct
        # Rolling per-mode stats. Skipped when usage_log is absent —
        # /status still returns a partial canary block, just without
        # the windows.
        if usage_log is not None:
            now_ts = time.time()
            canary_block["windows"] = {}
            for label, window_s in (
                ("1h", 3600.0),
                ("6h", 6 * 3600.0),
                ("24h", 24 * 3600.0),
            ):
                stats = usage_log.per_mode_stats_since(
                    since_ts=now_ts - window_s,
                )
                # Compute failure_rate per mode for at-a-glance reading.
                # rate is None when total = 0 so consumers can
                # distinguish "0% failure" from "no data."
                annotated: dict[str, dict[str, Any]] = {}
                for mode, counts in stats.items():
                    total = counts["total"]
                    failure_rate = counts["failure"] / total if total > 0 else None
                    annotated[mode] = {
                        **counts,
                        "failure_rate": failure_rate,
                    }
                canary_block["windows"][label] = annotated
        return {
            "backends": entries,
            "pinned": pin_state.get(),
            "sessions": session_registry.snapshot(),
            "router": router_block,
            "canary": canary_block,
        }

    if auth_service is not None:
        _install_auth_routes(app, auth_service)
        _install_web_ui(app)

    @app.post("/control/pin")
    async def control_pin(body: dict[str, Any]) -> dict[str, str | None]:
        backend_id = body.get("backend_id")
        if not isinstance(backend_id, str):
            raise HTTPException(status_code=400, detail="'backend_id' must be a string")
        if not any(b.id == backend_id for b in backends_list):
            raise HTTPException(status_code=404, detail=f"backend {backend_id!r} not in pool")
        pin_state.set(backend_id)
        return {"pinned": backend_id}

    @app.post("/control/unpin")
    async def control_unpin() -> dict[str, str | None]:
        pin_state.clear()
        return {"pinned": None}

    @app.post("/control/clear-cooldown/{backend_id}")
    async def control_clear_cooldown(backend_id: str) -> dict[str, Any]:
        """Operator override: clear a stale cooldown on one backend.

        Companion to the periodic cooldown prober (`_PeriodicCooldownProber`)
        for cases where you'd rather not wait an interval for the next probe
        — e.g. you know out of band that the account was just topped up.
        Returns 404 if no backend with that id is configured.
        """
        backend = next((b for b in backends_list if b.id == backend_id), None)
        if backend is None:
            raise HTTPException(status_code=404, detail=f"backend {backend_id!r} not in pool")
        clear = getattr(backend, "clear_cooldown", None)
        if clear is None:
            raise HTTPException(
                status_code=400,
                detail=f"backend {backend_id!r} does not support clear_cooldown",
            )
        snap = clear()
        return {
            "id": backend_id,
            "cleared": True,
            "snapshot": {
                "cooldown_until_ts": snap.cooldown_until_ts,
                "weekly_exhausted": snap.weekly_exhausted,
                "probed_at_ts": snap.probed_at_ts,
            },
        }

    @app.get("/diagnose/upstream")
    async def diagnose_upstream() -> dict[str, Any]:
        """Probe each backend with a tiny real request and verify the upstream
        contract still holds (HTTP 200, x-codex-* headers parse, response.completed
        SSE event with usage block, model name still accepted).

        Intended for a daily cron — run it, alert on any backend's `ok=false`.
        Each invocation makes one real upstream call per backend, costing a
        small number of quota tokens. Cooldown'd backends are reported as
        skipped (the daily run shouldn't kick a backend that's already
        recovering).
        """
        results = []
        for backend in backends_list:
            results.append(await _diagnose_backend(backend))
        all_ok = all(r["ok"] for r in results if not r.get("skipped"))
        return {"ok": all_ok, "backends": results}

    def _catalog_model_ids() -> list[str]:
        """The canonical Callosum catalog ids, recomputed on each call.

        Built from the same backend `advertised_models` + `model_metadata` the
        cell grid uses, so the catalog and the router agree on what exists:

        - strategy selectors: callosum:auto / local-only / remote-only
        - remote concrete pins: callosum:remote/<model>:<effort> (one per
          supported reasoning level, from model_metadata with REASONING_LEVELS
          fallback)
        - local concrete pins: callosum:local/<model>, plus a
          callosum:local/<model>:<effort> variant for each non-default
          reasoning level the model advertises (from model_metadata). Models
          that advertise only ("default",) get the bare pin alone.
        - raw passthrough ids, still listed + resolvable for back-compat
        """
        raw: set[str] = set()
        remote_models: set[str] = set()
        local_models: set[str] = set()
        remote_meta: dict[str, ModelMetadata] = {}
        local_meta: dict[str, ModelMetadata] = {}
        for backend in backends_list:
            is_local = getattr(backend, "kind", "") == "litellm_gateway"
            for m in backend.advertised_models:
                if m in VIRTUAL_MODELS or is_selector(m):
                    continue
                raw.add(m)
                (local_models if is_local else remote_models).add(m)
            meta = getattr(backend, "model_metadata", None) or {}
            target = local_meta if is_local else remote_meta
            for slug, md in meta.items():
                existing = target.get(slug)
                # Prefer the record that actually carries reasoning levels.
                if existing is None or (
                    md.supported_reasoning_levels
                    and not existing.supported_reasoning_levels
                ):
                    target[slug] = md

        ids: list[str] = [
            "callosum:auto",
            "callosum:local-only",
            "callosum:remote-only",
        ]
        for m in sorted(remote_models):
            for level in reasoning_levels_for(m, remote_meta):
                ids.append(f"callosum:remote/{m}:{level}")
        for m in sorted(local_models):
            ids.append(f"callosum:local/{m}")
            # Effort variants come straight from the model's advertised
            # supported_reasoning_levels (NOT the REASONING_LEVELS fallback):
            # advertise a pin only for levels the model genuinely exposes.
            # "default" is the implicit base pin above, so it's skipped here;
            # any non-default level must also be a recognized REASONING_LEVELS
            # value to be pinnable via the selector grammar.
            md = local_meta.get(m)
            levels = md.supported_reasoning_levels if md is not None else ()
            for level in levels:
                if level != "default" and level in REASONING_LEVELS:
                    ids.append(f"callosum:local/{m}:{level}")
        ids.extend(sorted(raw))
        return ids

    def _live_catalog() -> dict[str, Any]:
        """OpenAI-compatible model list: the canonical Callosum catalog
        (selectors + concrete pins) plus raw passthrough ids for back-compat.
        """
        now_ts = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": now_ts,
                    "owned_by": "callosum",
                }
                for model_id in _catalog_model_ids()
            ],
        }

    @app.get("/v1/models")
    async def list_models_v1() -> dict[str, Any]:
        """OpenAI-standard catalog endpoint."""
        return _live_catalog()

    @app.get("/models")
    async def list_models_root() -> dict[str, Any]:
        """Some clients probe /models without the /v1 prefix."""
        return _live_catalog()

    @app.get("/api/v1/models")
    async def list_models_api_v1() -> dict[str, Any]:
        """Ollama-style /api/v1/models prefix some clients try."""
        return _live_catalog()

    @app.get("/api/tags", include_in_schema=False)
    async def ollama_tags() -> dict[str, Any]:
        """Ollama compatibility probe. Returning an empty `models: []` is
        valid Ollama shape and signals "I'm not Ollama" without 404'ing.
        """
        return {"models": []}

    @app.get("/version", include_in_schema=False)
    @app.get("/api/version", include_in_schema=False)
    async def version_probe() -> dict[str, str]:
        """Version probe — Ollama, model-a0e0, and others all hit this path."""
        return {"version": __version__}

    @app.get("/v1/props", include_in_schema=False)
    @app.get("/props", include_in_schema=False)
    async def llamacpp_props() -> dict[str, Any]:
        """model-a0e0 /props probe. Empty dict is a valid response that
        signals "I don't speak model-a0e0" without 404'ing.
        """
        return {}

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        """Root probe — common reachability check; surface enough that an
        operator hitting the proxy in a browser sees something useful.
        """
        return {
            "service": "callosum",
            "version": __version__,
            "docs": "/docs",
            "status": "/status",
        }

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        """No favicon, but don't 404 — that just clutters the log."""
        return Response(status_code=204)

    @app.get("/v1/models/{model_id:path}")
    async def get_model(model_id: str) -> dict[str, Any]:
        """OpenAI-compatible single-model lookup. Resolves `callosum:`
        selector ids against the canonical catalog; otherwise falls back to
        raw advertised-model lookup. Returns 404 when unknown.
        """
        if is_selector(model_id):
            try:
                sel = parse_selector(model_id)
            except SelectorError:
                sel = None
            # Strategy selectors are always valid; concrete pins must resolve
            # to a catalog entry (pinned model+effort actually advertised).
            if sel is not None and (
                sel.strategy is not None or model_id in _catalog_model_ids()
            ):
                return {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "callosum",
                }
            raise HTTPException(status_code=404, detail=f"model {model_id!r} not found")
        for backend in backends_list:
            if model_id in backend.advertised_models:
                return {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "callosum",
                }
        raise HTTPException(status_code=404, detail=f"model {model_id!r} not found")

    @app.post("/v1/eta")
    async def estimate_eta(body: dict[str, Any]) -> dict[str, Any]:
        """Pre-flight approximate latency ranges for visible cells.

        This is intentionally range-first and metadata-heavy: it exposes the
        existing TimeUsageEstimator without implying exact per-request timing.
        """
        estimator = getattr(app.state, "time_estimator", None)
        forecaster = getattr(app.state, "output_forecaster", None)
        if estimator is None or forecaster is None:
            return {
                "available": False,
                "reason": "time_estimator_unavailable",
                "unit": "ms",
                "input_tokens": None,
                "estimates": [],
            }

        raw_tokens = body.get("input_tokens", 1000)
        if not isinstance(raw_tokens, int) or raw_tokens < 0:
            raise HTTPException(status_code=400, detail="input_tokens must be a non-negative integer")
        requested_model = body.get("model")
        requested_effort = body.get("reasoning_effort")
        if requested_model is not None and not isinstance(requested_model, str):
            raise HTTPException(status_code=400, detail="model must be a string when provided")
        if requested_effort is not None and not isinstance(requested_effort, str):
            raise HTTPException(status_code=400, detail="reasoning_effort must be a string when provided")

        cells = _live_cells()
        if requested_model is not None:
            cells = [cell for cell in cells if cell.model == requested_model]
        if requested_effort is not None:
            cells = [cell for cell in cells if cell.reasoning_effort == requested_effort]
        estimates = [_eta_estimate_payload(estimator, forecaster, cell, raw_tokens) for cell in cells]
        return {
            "available": bool(estimates),
            "reason": None if estimates else "no_matching_cells",
            "unit": "ms",
            "input_tokens": raw_tokens,
            "estimates": estimates,
        }

    @app.get("/v1/usage")
    async def usage_rates() -> dict[str, Any]:
        """Empirical weekly-quota token-rate table from the local request log."""
        if usage_log is None:
            return {
                "available": False,
                "reason": "usage_log_unavailable",
                "unit": "weekly_used_percent",
                "basis": "empirical_request_log",
                "limitations": [
                    "usage logging is disabled, so no empirical request-log rates can be computed"
                ],
                "rates": [],
                "meters": {},
                "relationships": [],
            }
        return usage_rate_report(usage_log.path, cells=_live_cells())

    @app.post("/v1/feedback")
    async def feedback(body: dict[str, Any]) -> dict[str, str]:
        """Record user feedback (quality label) for a request.

        Expected body: {"request_id": <int>, "rating": <-1|0|1>}
        """
        if usage_log is None:
            raise HTTPException(status_code=503, detail="usage logging disabled")
        request_id = body.get("request_id")
        rating = body.get("rating")
        if not isinstance(request_id, int) or request_id <= 0:
            raise HTTPException(status_code=400, detail="request_id must be a positive integer")
        if rating not in (-1, 0, 1):
            raise HTTPException(status_code=400, detail="rating must be -1, 0, or 1")
        try:
            usage_log.record_quality(request_id, rating, "user")
        except Exception as exc:
            logging.getLogger("callosum.app").warning("feedback record failed: %s", exc)
            raise HTTPException(
                status_code=400,
                detail="request_id not found or feedback failed",
            ) from exc
        return {"status": "recorded", "request_id": str(request_id)}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, body: dict[str, Any]) -> Any:
        _request_id_context.set(None)  # Reset context for this request
        result = await _dispatch_route(
            body,
            request=request,
            route_name="chat_completions",
            backends_list=backends_list,
            pin_state=pin_state,
            session_registry=session_registry,
            usage_log=usage_log,
            nonstream=lambda b, p, h: b.chat_completions(p, h),
            stream=lambda b, p, h: b.chat_completions_stream(p, h),
            router_context_safety_margin=auto_cfg.router_context_safety_margin,
            state_store=state_store,
            auto_cfg=auto_cfg,
            router=router,
            operator_state=operator_state,
            live_cells_fn=_live_cells,
        )
        # Add X-Proxy-Request-ID header if a request was logged
        request_id = _request_id_context.get()
        if request_id is not None:
            headers = {"X-Proxy-Request-ID": str(request_id)}
            if isinstance(result, dict):
                return JSONResponse(result, headers=headers)
            # Result is already a StreamingResponse from _dispatch_stream
            if isinstance(result, StreamingResponse):
                result.headers.update(headers)
                return result
            # Fallback: shouldn't reach here, but wrap just in case
            return StreamingResponse(result, media_type="text/event-stream", headers=headers)
        return result

    @app.post("/codex")
    async def codex_endpoint(request: Request, body: dict[str, Any]) -> Any:
        """Codex CLI's dedicated endpoint. Accepts the Responses-API
        request body codex sends, returns the SSE stream codex parses.
        Same dispatch core as /v1/responses, but tagged with
        client_endpoint="codex" so codex-specific response transforms
        can scope themselves to this endpoint.

        Generic OpenAI clients should keep using /v1/responses (no
        client-specific transforms applied)."""
        _request_id_context.set(None)
        result = await _dispatch_route(
            body,
            request=request,
            route_name="responses",
            backends_list=backends_list,
            pin_state=pin_state,
            session_registry=session_registry,
            usage_log=usage_log,
            nonstream=lambda b, p, h: b.responses(p, h),
            stream=lambda b, p, h: b.responses_stream(p, h),
            router_context_safety_margin=auto_cfg.router_context_safety_margin,
            state_store=state_store,
            auto_cfg=auto_cfg,
            router=router,
            operator_state=operator_state,
            live_cells_fn=_live_cells,
            client_endpoint="codex",
        )
        request_id = _request_id_context.get()
        if request_id is not None:
            headers = {"X-Proxy-Request-ID": str(request_id)}
            if isinstance(result, dict):
                return JSONResponse(result, headers=headers)
            if isinstance(result, StreamingResponse):
                result.headers.update(headers)
                return result
            return StreamingResponse(result, media_type="text/event-stream", headers=headers)
        return result

    @app.post("/v1/responses")
    async def responses(request: Request, body: dict[str, Any]) -> Any:
        _request_id_context.set(None)  # Reset context for this request
        result = await _dispatch_route(
            body,
            request=request,
            route_name="responses",
            backends_list=backends_list,
            pin_state=pin_state,
            session_registry=session_registry,
            usage_log=usage_log,
            nonstream=lambda b, p, h: b.responses(p, h),
            stream=lambda b, p, h: b.responses_stream(p, h),
            router_context_safety_margin=auto_cfg.router_context_safety_margin,
            state_store=state_store,
            auto_cfg=auto_cfg,
            router=router,
            operator_state=operator_state,
            live_cells_fn=_live_cells,
        )
        # Add X-Proxy-Request-ID header if a request was logged
        request_id = _request_id_context.get()
        if request_id is not None:
            headers = {"X-Proxy-Request-ID": str(request_id)}
            if isinstance(result, dict):
                return JSONResponse(result, headers=headers)
            # Result is already a StreamingResponse from _dispatch_stream
            if isinstance(result, StreamingResponse):
                result.headers.update(headers)
                return result
            # Fallback: shouldn't reach here, but wrap just in case
            return StreamingResponse(result, media_type="text/event-stream", headers=headers)
        return result

    return app


async def _routable_backends(backends_list: Sequence[Backend], *, now: float | None = None) -> list[Backend]:
    """Return the subset of currently-routable backends.

    Routable = not weekly-exhausted, not in cooldown. Reads cached usage
    state from each backend (no upstream calls) so this is cheap enough
    to run on every routing decision.

    The point is to keep the cell grid honest when Codex is exhausted or
    on cooldown: cells whose only serving backends are unroutable get
    filtered out of `cells_now` before the classifier sees them, so the
    recommender can only pick something dispatch can actually reach.
    Phase 4d-v1; v2 may add quick network-failure marking so offline
    transitions are sub-second rather than waiting for the classifier-
    call timeout to fail.
    """
    n = now if now is not None else time.time()
    out: list[Backend] = []
    for b in backends_list:
        try:
            u = await b.usage_snapshot()
            q = await b.quota_snapshot()
        except Exception:
            continue  # if we can't even read state, treat as unroutable
        if blocking_meters(BackendSnapshot(backend=b, health=HealthStatus(True, "ok"), usage=u, quota=q), now_ts=n):
            continue
        if u.cooldown_until_ts is not None and u.cooldown_until_ts > n:
            continue
        out.append(b)
    return out


def _filter_cells_to_routable(cells: list[Cell], routable_backends: Sequence[Backend]) -> list[Cell]:
    """Keep only cells whose model is advertised by at least one
    currently-routable backend. When every backend serving a cell's
    model is unroutable (e.g. Codex weekly_exhausted and the cell is
    Codex-only), the cell is dropped — the classifier can't pick a
    cell dispatch would fail to serve."""
    routable_models: set[str] = set()
    for b in routable_backends:
        routable_models.update(b.advertised_models)
    return [c for c in cells if c.model in routable_models]


async def _dispatch_route(
    body: dict[str, Any],
    *,
    request: Request,
    route_name: str,
    backends_list: Sequence[Backend],
    pin_state: PinState,
    session_registry: SessionRegistry,
    usage_log: UsageLog | None,
    nonstream: NonstreamCall,
    stream: StreamCall,
    router_context_safety_margin: int = 8192,
    state_store: Any | None = None,
    auto_cfg: AutoRouterConfig | None = None,
    router: Router | None = None,
    operator_state: Any = None,
    live_cells_fn: Callable[[], list[Cell]] | None = None,
    client_endpoint: str | None = None,
) -> Any:
    """HTTP entry-point. Pulls session_id + api_key off the Request, then
    hands off to _dispatch_internal for the rewrite + dispatch logic.

    `client_endpoint` identifies which entry-point the request arrived on
    (e.g. "codex" for POST /codex; None for the generic /v1/* endpoints).
    Threaded through to the transform context so per-CLI transforms can
    scope themselves to the CLI they were written for.
    """
    pinned_now = pin_state.get()
    session_id = _session_id_from_request(request, pinned=pinned_now)
    api_key: ApiKey | None = getattr(request.state, "api_key", None)
    user_id = api_key.user_id if api_key is not None else None
    api_key_id = api_key.id if api_key is not None else None
    return await _dispatch_internal(
        body,
        route_name=route_name,
        backends_list=backends_list,
        pin_state=pin_state,
        session_registry=session_registry,
        usage_log=usage_log,
        nonstream=nonstream,
        stream=stream,
        session_id=session_id,
        user_id=user_id,
        api_key_id=api_key_id,
        router_context_safety_margin=router_context_safety_margin,
        state_store=state_store,
        auto_cfg=auto_cfg,
        router=router,
        operator_state=operator_state,
        live_cells_fn=live_cells_fn,
        client_endpoint=client_endpoint,
    )


async def _dispatch_internal(
    body: dict[str, Any],
    *,
    route_name: str,
    backends_list: Sequence[Backend],
    pin_state: PinState,
    session_registry: SessionRegistry,
    usage_log: UsageLog | None,
    nonstream: NonstreamCall,
    stream: StreamCall,
    session_id: str | None,
    user_id: int | None,
    api_key_id: int | None,
    forced_backend_id: str | None = None,
    router_context_safety_margin: int = 8192,
    state_store: Any | None = None,
    auto_cfg: AutoRouterConfig | None = None,
    router: Router | None = None,
    operator_state: Any = None,
    live_cells_fn: Callable[[], list[Cell]] | None = None,
    client_endpoint: str | None = None,
) -> Any:
    if auto_cfg is None:
        auto_cfg = AutoRouterConfig()
    if live_cells_fn is None:
        live_cells_fn = build_cells
    """Dispatch core, no Request dependency. Used by the HTTP entry-points.

    `forced_backend_id` constrains the selector to a single backend. HTTP
    entry-points never set it; it remains for in-process callers that need to
    pin dispatch to one account.
    """
    requested_model = _require_model(body)
    requested_reasoning = _extract_reasoning_effort(body)
    # Client-driven routing selector. A `callosum:` model id expresses routing
    # intent for THIS request only (strategy engine and/or concrete pin) and
    # overrides the operator default below without mutating any global state.
    # None = legacy pass-through (behavior unchanged).
    try:
        _selector = parse_selector(requested_model)
    except SelectorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    routing_mode = "pass-through"
    # Router provenance for the request log. Populated when the Router
    # makes a decision; stays None for the no-router path (cold-start
    # state when no backends are loaded).
    recommender_classifier_cell: str | None = None
    recommender_raw_output: str | None = None
    recommender_source: str | None = None
    # Embedding for the prompt — populated by the EmbeddingProvider, lands
    # in the prompt_embedding BLOB column. Stays None when noop provider
    # is configured. Feeds the kNN predictor at training time.
    prompt_embedding: bytes | None = None
    cell_candidates: tuple[Cell, ...] = ()

    # `session_prompt_tokens` lookup used to feed the now-removed hard
    # context-window capability filter (see capability.py module
    # docstring). Window fit is now a soft preference applied by the
    # Router using the request body's own size, so the previous
    # per-session lookup is no longer needed on the hot path.

    # Route ALL requests through the learning router. Codex CLI sending
    # `model-a0e8`, Hermes sending `model-a0c3`, and explicit `auto`
    # requests all hit the same pipeline — the router's decision is
    # authoritative, and the original requested model only appears in
    # the request log's `routing_mode` column for provenance.
    if router is not None:
        cells_now = live_cells_fn()
        # Operator denylist + mode filters BEFORE routability — operator
        # decisions are absolute. denylist drops named cells; mode
        # constrains which backend kinds are eligible.
        #
        # Canary baseline: when operator is in auto, a configurable
        # fraction of requests are diverted to remote-only for this
        # request only (see callosum.canary). The fraction is bounded
        # [2%, 10%] and scales down as Codex weekly quota approaches
        # exhaustion. Canary requests skip local cells entirely, which
        # is structurally what makes them a clean baseline: any local
        # code change (transform, router tweak, harness adapter)
        # cannot affect this request's outcome.
        if operator_state is not None:
            _routing = operator_state.get_routing()
            # Per-request client selector overrides the operator default for
            # THIS request only (no global mutation). Strategy selectors set the
            # routing engine; concrete pins additionally constrain the cell pool
            # below and imply their source's engine.
            if _selector is not None:
                if _selector.strategy is not None:
                    _routing = _selector.strategy
                elif _selector.source == "remote":
                    _routing = "remote-only"
                elif _selector.source == "local":
                    _routing = "local-only"
            # Map routing mode to effective_routing_mode for the log.
            # Default mapping: 'auto' → 'auto', everything else → forced_*.
            _effective_mode = _routing if _routing == "auto" else f"forced_{_routing.replace('-only', '')}"
            if _selector is not None:
                _effective_mode = f"selector_{_selector.strategy or _selector.source}"
            # Quota for the canary scheduler. Walk codex_auth_vault
            # backends, take the highest weekly_used_percent (the
            # most-constrained). None means "unknown" — scheduler
            # treats unknown as unconstrained.
            _quota_pct: float | None = None
            for _b in backends_list:
                if getattr(_b, "kind", "") == "litellm_gateway":
                    continue
                _q = getattr(_b, "_last_quota", None)
                if _q is None:
                    continue
                _w = getattr(_q, "weekly_used_percent", None)
                if _w is None:
                    continue
                if _quota_pct is None or _w > _quota_pct:
                    _quota_pct = _w
            from callosum.canary import CanaryDecision

            _sched = _CANARY_SCHEDULER
            if _sched is not None:
                _canary = _sched.decide(
                    routing=_routing,
                    quota_used_percent=_quota_pct,
                )
            else:
                _canary = CanaryDecision.NORMAL
            if _canary == CanaryDecision.REDIRECT_REMOTE:
                # Override _routing for THIS request only; operator
                # state stays untouched. effective_mode reflects the
                # redirect so the row lands in the canary bucket.
                _routing = "remote-only"
                _effective_mode = "canary_redirect"
            _effective_routing_mode_context.set(_effective_mode)
            if _routing in ("offline", "local-only"):
                cells_now = [
                    c
                    for c in cells_now
                    if any(
                        c.model in b.advertised_models and getattr(b, "kind", "") == "litellm_gateway"
                        for b in backends_list
                    )
                ]
            elif _routing == "remote-only":
                cells_now = [
                    c
                    for c in cells_now
                    if any(
                        c.model in b.advertised_models and getattr(b, "kind", "") != "litellm_gateway"
                        for b in backends_list
                    )
                ]
            cells_now = [c for c in cells_now if not operator_state.is_denied(c.model)]
            # Concrete client pin: narrow the cell pool to the pinned model
            # (and effort, for remote pins). The source constraint was already
            # applied above via the selector-derived `_routing`. An empty pool
            # here surfaces as a clean 503 below (same as any unroutable state),
            # never a silent fall-through to a different model.
            if _selector is not None and _selector.pinned_model is not None:
                cells_now = [c for c in cells_now if c.model == _selector.pinned_model]
                if _selector.pinned_effort is not None:
                    cells_now = [c for c in cells_now if c.reasoning_effort == _selector.pinned_effort]
        _process_pin = pin_state.get()
        _process_pin_concrete_model = (
            _process_pin is not None
            and _selector is None
            and requested_model not in VIRTUAL_MODELS
        )
        # The process-wide operator pin is stronger than the learning
        # router's model rewrite. For concrete model requests, preserve the
        # requested model and let an incompatible pinned backend surface as
        # no viable backend. Virtual router names (auto/auto-learning) still
        # choose among the pinned backend's advertised cells.
        if _process_pin_concrete_model:
            cells_now = [c for c in cells_now if c.model == requested_model]
        # Drop cells whose only serving backends are currently unroutable
        # (Codex weekly-exhausted, on cooldown, gateway down). The Router's
        # capability filter further drops cells whose physical capabilities
        # can't serve THIS request.
        #
        # The filter is UNCONDITIONAL. An earlier version of this block
        # gated the filter behind `if _routable:` and `if _filtered:` —
        # the intent was "don't shrink to empty," but the effect was
        # "leave dead cells in the grid when everything is exhausted."
        # The router then accepted a dead cell, dispatch failed, the
        # error surfaced as 400 to the client, codex CLI retried, and
        # the cycle repeated until manual interruption. Documented in
        # docs/investigations/2026-06-08-routing-symptoms.md (Finding A).
        #
        # The correct behavior: filter unconditionally, then distinguish
        # the empty-cells causes so the response code matches the actual
        # reason (transient unavailability → 503 with Retry-After, vs.
        # fundamental capability mismatch → 400).
        _routable = await _routable_backends(backends_list)
        # Intersect routable backends with the active routing mode at the
        # backend level. Without this intersection, a remote-only mode
        # request could survive when only local backends are routable
        # (the cell-level mode filter ABOVE keeps cells served by a
        # remote-kind backend; the routability filter BELOW then keeps
        # cells served by ANY routable backend, including local — net
        # effect: mode-violating dispatch). Intersection here makes
        # routable = "satisfies both routability AND the operator's
        # current mode choice."
        if operator_state is not None:
            if _routing in ("offline", "local-only"):
                _routable = [b for b in _routable if getattr(b, "kind", "") == "litellm_gateway"]
            elif _routing == "remote-only":
                _routable = [b for b in _routable if getattr(b, "kind", "") != "litellm_gateway"]
        if _process_pin is not None:
            _routable = [b for b in _routable if b.id == _process_pin]
        cells_now = _filter_cells_to_routable(cells_now, _routable)
        if not cells_now:
            if _process_pin_concrete_model:
                _pinned_backend = next((b for b in backends_list if b.id == _process_pin), None)
                if _pinned_backend is not None and requested_model not in _pinned_backend.advertised_models:
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            f"pinned backend {_process_pin!r} cannot serve "
                            f"requested model {requested_model!r}"
                        ),
                        headers={"Retry-After": "60"},
                    )
            # A concrete client pin that no backend currently serves: the pin
            # filter (above) emptied the pool. Surface that directly — the lane
            # may be operator-declared in the `/model` picker but not yet live
            # (requirement of the client-driven routing catalog, ).
            # Without this branch the cause would be misreported as a routing-mode
            # exclusion below.
            if _selector is not None and _selector.pinned_model is not None:
                _effort_note = (
                    f" at {_selector.pinned_effort} reasoning"
                    if _selector.pinned_effort is not None
                    else ""
                )
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"selected lane {requested_model!r} is not available yet: "
                        f"no backend currently serves model "
                        f"{_selector.pinned_model!r}{_effort_note}"
                    ),
                    headers={"Retry-After": "60"},
                )
            if not _routable:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"no backend is routable under routing mode "
                        f"{_routing!r} "
                        "(all eligible backends on cooldown, "
                        "quota-exhausted, or excluded by mode)"
                    ),
                    headers={"Retry-After": "60"},
                )
            raise HTTPException(
                status_code=503,
                detail=(f"current routing mode {_routing!r} excludes all currently-routable backends"),
                headers={"Retry-After": "60"},
            )
        try:
            decision = await router.route(body, cells_now)
        except NoCompatibleCellError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"no cell can serve this request: {exc}",
            ) from exc
        chosen = decision.cell
        # Per-cell exploration quota (): the router's
        # cost/quality-optimal pick stands by default; when any compatible cell
        # is below its floor, steer this turn to it so every cell gets at least
        # quota_floor_pct of traffic. Lane scope is implicit: decision.candidates
        # is already the post-gate, lane-filtered pool. See
        # docs/architecture/exploration_quota.md.
        _ordered_candidates: tuple[Cell, ...] = decision.candidates
        if (
            auto_cfg.exploration_quota_enabled
            and usage_log is not None
            and len(decision.candidates) > 1
        ):
            _quota_coverage = cell_sample_counts(
                usage_log.path,
                list(decision.candidates),
                window_seconds=auto_cfg.quota_window_seconds,
            )
            _forced = select_quota_deficit_cell(
                decision.candidates,
                _quota_coverage,
                floor_pct=auto_cfg.quota_floor_pct,
            )
            if _forced is not None and _forced != chosen:
                chosen = _forced
                _ordered_candidates = (_forced, *(c for c in decision.candidates if c != _forced))
                # Mark this turn as quota-forced so it is distinguishable in the
                # log: excludable from natural-routing baselines while its
                # quality label stays usable.
                _effective_routing_mode_context.set("quota_explore")
        body["model"] = chosen.model
        _stamp_reasoning_effort(body, chosen.reasoning_effort)
        # Per-cell transforms. Empty registry → no-op. Each registered
        # transform's `applies_to(ctx)` decides whether it fires for
        # this cell+endpoint combination. Errors inside a transform are
        # isolated: the registry logs and skips. Streaming response
        # transforms are deferred — translating SSE events on the fly
        # requires careful design (chunk boundaries, partial events);
        # for v1, response transforms apply ONLY on the non-stream path
        # (which buffers the full response before returning).
        _tr = _TRANSFORM_REGISTRY
        _transform_ctx: Any = None
        if _tr is not None and len(_tr) > 0:
            from callosum.capability.profile import (
                load_profile as _load_profile,
            )
            from callosum.capability.profile import (
                profile_path as _profile_path,
            )
            from callosum.transforms import TransformContext

            # Build the transform context once per request. Loading
            # the profile from disk is cheap (small JSON), so we
            # accept it on the hot path; if it ever shows up in
            # profiling we can cache by cell name with mtime
            # invalidation like the gating reader.
            _profile = None
            try:
                _path = _profile_path(chosen.model)
                if _path.exists():
                    _profile = _load_profile(chosen.model, profile_dir=_path.parent)
            except Exception:
                logger.exception(
                    "transform context: failed to load profile for %s",
                    chosen.model,
                )
            _transform_ctx = TransformContext(
                cell=chosen,
                weight_identity=(_profile.weight_identity if _profile is not None else None),
                capability_profile=_profile,
                endpoint=client_endpoint,
            )
            body = _tr.apply_request(body, _transform_ctx)
        # Provenance for the request log — predictor_id distinguishes
        # cold-start (uniform) from learned (knn / gbm / ...) decisions
        # so downstream analysis can weight them differently. The
        # predictions map is the predictor's per-cell P(satisfy) over
        # the post-filter candidate set, serialized as compact JSON.
        recommender_classifier_cell = decision.predictor_id or None
        recommender_raw_output = (
            json.dumps(decision.predictions, separators=(",", ":"))[:500] if decision.predictions else None
        )
        recommender_source = "router"
        # Persist the prompt embedding for the kNN predictor's training
        # corpus. None when the noop embedding provider is selected.
        prompt_embedding = decision.features.embedding
        # Candidate cells the dispatch layer walks if the primary's
        # backend pool 5xxs. Capped at MAX_CELL_ATTEMPTS so retry
        # latency stays bounded.
        cell_candidates = _ordered_candidates[:MAX_CELL_ATTEMPTS]
        routing_mode = requested_model
    model = _require_model(body)
    pinned = pin_state.get()
    active = _active_pool(backends_list, pinned)
    # Honor the client selector's source at the dispatch pool too. The cell
    # filter above constrains which (model, effort) cells are eligible, but the
    # final backend ranking walks the active pool — when a model is served by
    # both a local and a remote backend, narrow the pool so the selector's
    # source wins regardless of catalog overlap. (Operator-mode requests rely
    # on disjoint local/remote catalogs and are left unchanged.)
    if _selector is not None:
        _want_local = _selector.strategy == "local-only" or _selector.source == "local"
        _want_remote = _selector.strategy == "remote-only" or _selector.source == "remote"
        if _want_local:
            active = [b for b in active if getattr(b, "kind", "") == "litellm_gateway"]
        elif _want_remote:
            active = [b for b in active if getattr(b, "kind", "") != "litellm_gateway"]
    if forced_backend_id is not None:
        active = [b for b in active if b.id == forced_backend_id]
        if not active:
            raise HTTPException(
                status_code=503,
                detail=f"forced backend {forced_backend_id!r} not in active pool",
            )
    preferred_id = session_registry.get(session_id) if session_id is not None else None
    if body.get("stream") is True:
        peer_quality_capture: _PeerQualityCapture | None = None
        if _peer_quality_capture_enabled():
            peer_quality_capture = _PeerQualityCapture(nonce=secrets.token_urlsafe(8))
            peer_quality_cell = Cell(model=model, reasoning_effort=_extract_reasoning_effort(body) or "")
            body = _inject_peer_quality_prompt(
                body,
                capture=peer_quality_capture,
                usage_log=usage_log,
                session_id=session_id,
                current_cell=peer_quality_cell,
                context_window_tokens=_context_window_for_cell(active, peer_quality_cell),
                safety_margin_tokens=router_context_safety_margin,
            )
        return await _dispatch_stream_with_cell_retry(
            body,
            candidates=cell_candidates,
            model=model,
            route_name=route_name,
            backends_list=active,
            preferred_id=preferred_id,
            session_id=session_id,
            session_registry=session_registry,
            usage_log=usage_log,
            call=stream,
            user_id=user_id,
            api_key_id=api_key_id,
            requested_model=requested_model,
            requested_reasoning_effort=requested_reasoning,
            routing_mode=routing_mode,
            recommender_classifier_cell=recommender_classifier_cell,
            recommender_raw_output=recommender_raw_output,
            recommender_source=recommender_source,
            prompt_embedding=prompt_embedding,
            peer_quality_capture=peer_quality_capture,
        )
    result = await _dispatch_nonstream_with_cell_retry(
        body,
        candidates=cell_candidates,
        model=model,
        route_name=route_name,
        backends_list=active,
        preferred_id=preferred_id,
        session_id=session_id,
        session_registry=session_registry,
        usage_log=usage_log,
        call=nonstream,
        user_id=user_id,
        api_key_id=api_key_id,
        requested_model=requested_model,
        requested_reasoning_effort=requested_reasoning,
        routing_mode=routing_mode,
        recommender_classifier_cell=recommender_classifier_cell,
        recommender_raw_output=recommender_raw_output,
        recommender_source=recommender_source,
        prompt_embedding=prompt_embedding,
    )
    # Response-side transforms (non-stream only — see comment above on
    # the request transform for why streaming is deferred). Empty
    # registry or no-applicable-transforms is a clean no-op.
    if _transform_ctx is not None and _tr is not None and isinstance(result, dict):
        result = _tr.apply_response(result, _transform_ctx)
    return result


def _require_model(body: dict[str, Any]) -> str:
    model = body.get("model")
    if not isinstance(model, str):
        raise HTTPException(status_code=400, detail="'model' must be a string")
    return model


def _active_pool(backends_list: Sequence[Backend], pinned: str | None) -> Sequence[Backend]:
    if pinned is None:
        return backends_list
    return [b for b in backends_list if b.id == pinned]


def _session_id_from_request(request: Request, *, pinned: str | None) -> str | None:
    # A pin is an operator override: it wins over any client-declared session.
    if pinned is not None:
        return None
    # Walk the accepted header names in preference order. First non-empty
    # value wins. Empty values fall through to None so callers can treat
    # "no client session" uniformly.
    for name in SESSION_HEADERS:
        raw = request.headers.get(name)
        if raw is None:
            continue
        session_id = raw.strip()
        if session_id:
            return session_id
    return None


def _remember_binding(registry: SessionRegistry, session_id: str | None, backend_id: str) -> None:
    if session_id is None:
        return
    registry.set(session_id, backend_id)


# Maximum number of cells the dispatch layer will try before giving up on
# a request. The recommender returns its primary pick at index 0 plus the
# rest of the compatible cell set in priority order; the dispatch layer
# walks them on retryable failures (5xx from a cell's backend pool). Cap
# is intentionally small — three attempts cover the common "primary cell
# is rate-limited" + "next-best cell unhealthy" sequence without blowing
# user-perceived latency. There's no wall-clock cap; latency is bounded
# only by upstream timeouts.
MAX_CELL_ATTEMPTS = 3


async def _dispatch_nonstream_with_cell_retry(
    body: dict[str, Any],
    *,
    candidates: tuple[Cell, ...],
    usage_log: UsageLog | None,
    model: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Walk up to MAX_CELL_ATTEMPTS candidate cells, rerouting on 5xx.

    Empty `candidates` short-circuits to a single _dispatch_nonstream call,
    preserving existing behavior for pass-through requests. Per-cell
    attempt history is persisted to request_routing_attempts via the
    final request_id (success row, or last failure row written by the
    inner dispatch's _log_attempt path).
    """
    cells_to_try = list(candidates[:MAX_CELL_ATTEMPTS])
    if not cells_to_try:
        return await _dispatch_nonstream(body, model=model, usage_log=usage_log, **kwargs)

    attempts: list[RoutingAttempt] = []
    final_request_id: int | None = None

    for cell_idx, cell in enumerate(cells_to_try):
        body["model"] = cell.model
        _stamp_reasoning_effort(body, cell.reasoning_effort)
        attempt_start = time.time()
        try:
            result = await _dispatch_nonstream(body, model=cell.model, usage_log=usage_log, **kwargs)
        except HTTPException as exc:
            attempt_ms = int((time.time() - attempt_start) * 1000)
            final_request_id = _request_id_context.get()
            attempts.append(
                RoutingAttempt(
                    attempt_idx=cell_idx,
                    # Cell may have spanned multiple backends inside the inner
                    # loop; backend_id at the cell level is intentionally NULL.
                    # Consumers wanting that join requests on (request_id, model).
                    backend_id=None,
                    model=cell.model,
                    reasoning_effort=cell.reasoning_effort,
                    status=exc.status_code,
                    classification=(
                        "retried_next_cell" if exc.status_code >= 500 and cell_idx + 1 < len(cells_to_try) else "failed"
                    ),
                    latency_ms=attempt_ms,
                    error_message=(str(exc.detail)[:500] if exc.detail else None),
                )
            )
            # 4xx → non-retryable (auth_invalid, malformed request, etc).
            # 5xx + cells remaining → reroute. 5xx + no cells left → propagate.
            if exc.status_code < 500 or cell_idx + 1 >= len(cells_to_try):
                if usage_log is not None and final_request_id is not None:
                    usage_log.record_routing_attempts(final_request_id, attempts)
                raise
            continue
        # Success.
        attempt_ms = int((time.time() - attempt_start) * 1000)
        final_request_id = _request_id_context.get()
        attempts.append(
            RoutingAttempt(
                attempt_idx=cell_idx,
                backend_id=None,
                model=cell.model,
                reasoning_effort=cell.reasoning_effort,
                status=200,
                classification="ok",
                latency_ms=attempt_ms,
            )
        )
        # Persist only multi-attempt histories. Single-attempt requests are the
        # common case and the requests row already tells the whole story —
        # keeping the sibling table to the reroute cohort makes "how often did
        # we have to reroute?" a one-line query.
        if usage_log is not None and final_request_id is not None and len(attempts) > 1:
            usage_log.record_routing_attempts(final_request_id, attempts)
        return result

    # Unreachable: the loop above either returns on success or raises after
    # the final cell. Kept for type-checker clarity.
    raise RuntimeError("cell-level retry exhausted without raising")  # pragma: no cover


async def _dispatch_stream_with_cell_retry(
    body: dict[str, Any],
    *,
    candidates: tuple[Cell, ...],
    usage_log: UsageLog | None,
    model: str,
    **kwargs: Any,
) -> StreamingResponse:
    """Stream-path mirror of _dispatch_nonstream_with_cell_retry.

    Reroute happens only on pre-first-chunk HTTPException from the inner
    dispatch — once StreamingResponse is committed and bytes start flowing,
    mid-stream failover stays a non-goal (per The Project Documentation). The inner
    dispatch raises HTTPException only before first-chunk; after that, it
    returns StreamingResponse and any backend failure is bubbled in-band.
    """
    cells_to_try = list(candidates[:MAX_CELL_ATTEMPTS])
    if not cells_to_try:
        return await _dispatch_stream(body, model=model, usage_log=usage_log, **kwargs)

    attempts: list[RoutingAttempt] = []
    final_request_id: int | None = None

    for cell_idx, cell in enumerate(cells_to_try):
        body["model"] = cell.model
        _stamp_reasoning_effort(body, cell.reasoning_effort)
        attempt_start = time.time()
        try:
            result = await _dispatch_stream(body, model=cell.model, usage_log=usage_log, **kwargs)
        except HTTPException as exc:
            attempt_ms = int((time.time() - attempt_start) * 1000)
            final_request_id = _request_id_context.get()
            attempts.append(
                RoutingAttempt(
                    attempt_idx=cell_idx,
                    backend_id=None,
                    model=cell.model,
                    reasoning_effort=cell.reasoning_effort,
                    status=exc.status_code,
                    classification=(
                        "retried_next_cell" if exc.status_code >= 500 and cell_idx + 1 < len(cells_to_try) else "failed"
                    ),
                    latency_ms=attempt_ms,
                    error_message=(str(exc.detail)[:500] if exc.detail else None),
                )
            )
            if exc.status_code < 500 or cell_idx + 1 >= len(cells_to_try):
                if usage_log is not None and final_request_id is not None:
                    usage_log.record_routing_attempts(final_request_id, attempts)
                raise
            continue
        attempt_ms = int((time.time() - attempt_start) * 1000)
        final_request_id = _request_id_context.get()
        attempts.append(
            RoutingAttempt(
                attempt_idx=cell_idx,
                backend_id=None,
                model=cell.model,
                reasoning_effort=cell.reasoning_effort,
                status=200,
                classification="ok",
                latency_ms=attempt_ms,
            )
        )
        if usage_log is not None and final_request_id is not None and len(attempts) > 1:
            usage_log.record_routing_attempts(final_request_id, attempts)
        return result

    raise RuntimeError("cell-level stream retry exhausted without raising")  # pragma: no cover


async def _dispatch_nonstream(
    body: dict[str, Any],
    *,
    model: str,
    route_name: str,
    backends_list: Sequence[Backend],
    preferred_id: str | None,
    session_id: str | None,
    session_registry: SessionRegistry,
    usage_log: UsageLog | None,
    call: NonstreamCall,
    user_id: int | None = None,
    api_key_id: int | None = None,
    requested_model: str | None = None,
    requested_reasoning_effort: str | None = None,
    routing_mode: str = "pass-through",
    recommender_classifier_cell: str | None = None,
    recommender_raw_output: str | None = None,
    recommender_source: str | None = None,
    prompt_embedding: bytes | None = None,
) -> dict[str, Any]:
    excluded: set[str] = set()
    excluded_errors: dict[str, BackendError] = {}
    last_error: BackendError | None = None
    cost_estimate = _composite_cost_estimate_for_body(body, model=model)
    while True:
        backend = await select(
            backends_list,
            model=model,
            excluded=frozenset(excluded),
            preferred_id=preferred_id,
            cost_estimate=cost_estimate,
        )
        if backend is None:
            break
        handle = CallHandle()
        ts_start = time.time()
        try:
            result = await call(backend, body, handle)
        except BackendError as exc:
            ts_end = time.time()
            _log_attempt(
                usage_log,
                body=body,
                model=model,
                route_name=route_name,
                stream=False,
                session_id=session_id,
                backend=backend,
                handle=handle,
                ts_start=ts_start,
                ts_end=ts_end,
                error=exc,
                resp_body=None,
                user_id=user_id,
                api_key_id=api_key_id,
                requested_model=requested_model,
                requested_reasoning_effort=requested_reasoning_effort,
                routing_mode=routing_mode,
                prompt_complexity_class=None,
                recommender_classifier_cell=recommender_classifier_cell,
                recommender_raw_output=recommender_raw_output,
                recommender_source=recommender_source,
                prompt_embedding=prompt_embedding,
            )
            last_error = exc
            if exc.classification not in RETRYABLE:
                raise _terminal_http(exc) from exc
            excluded.add(backend.id)
            excluded_errors[backend.id] = exc
            continue
        ts_end = time.time()
        _remember_binding(session_registry, session_id, backend.id)

        # Defensively strip any stray complexity marker a model voluntarily
        # emits so it never leaks to the client. The marker-only class itself is
        # retired () and no longer recorded.
        _, result = _extract_and_strip_complexity(result)

        _log_attempt(
            usage_log,
            body=body,
            model=model,
            route_name=route_name,
            stream=False,
            session_id=session_id,
            backend=backend,
            handle=handle,
            ts_start=ts_start,
            ts_end=ts_end,
            error=None,
            resp_body=result,
            user_id=user_id,
            api_key_id=api_key_id,
            requested_model=requested_model,
            requested_reasoning_effort=requested_reasoning_effort,
            routing_mode=routing_mode,
            prompt_complexity_class=None,  # retired: marker-only label no longer collected
            recommender_classifier_cell=recommender_classifier_cell,
            recommender_raw_output=recommender_raw_output,
            recommender_source=recommender_source,
            prompt_embedding=prompt_embedding,
        )
        return result

    # All primary backends exhausted. Compute recovery timestamp.
    recovery_ts: float | None = None
    for backend in backends_list:
        try:
            usage = await backend.usage_snapshot()
            if usage and usage.cooldown_until_ts and (recovery_ts is None or usage.cooldown_until_ts < recovery_ts):
                recovery_ts = usage.cooldown_until_ts
        except Exception:
            pass  # Skip backends that fail to report usage

    # Try fallback strategies.
    error_classifications = {backend_id: error.classification for backend_id, error in excluded_errors.items()}

    backend_status = await _collect_backend_status(backends_list)
    if should_attempt_fallback(error_classifications):
        fallback = FallbackExecutor()
        # TODO: Implement fallback retry logic here
        # For now, just log that we attempted it
        fallback.record_attempt(
            "considered_fallback",
            "skipped",
            reason="fallback_not_yet_implemented",
        )
        raise _no_viable(
            model=model,
            last_error=last_error,
            excluded_backends=excluded_errors,
            fallback_executor=fallback,
            recovery_ts=recovery_ts,
            backend_status=backend_status,
        )

    raise _no_viable(
        model=model,
        last_error=last_error,
        excluded_backends=excluded_errors,
        fallback_executor=None,
        recovery_ts=recovery_ts,
        backend_status=backend_status,
    )


async def _dispatch_stream(
    body: dict[str, Any],
    *,
    model: str,
    route_name: str,
    backends_list: Sequence[Backend],
    preferred_id: str | None,
    session_id: str | None,
    session_registry: SessionRegistry,
    usage_log: UsageLog | None,
    call: StreamCall,
    user_id: int | None = None,
    api_key_id: int | None = None,
    requested_model: str | None = None,
    requested_reasoning_effort: str | None = None,
    routing_mode: str = "pass-through",
    recommender_classifier_cell: str | None = None,
    recommender_raw_output: str | None = None,
    recommender_source: str | None = None,
    prompt_embedding: bytes | None = None,
    peer_quality_capture: _PeerQualityCapture | None = None,
) -> StreamingResponse:
    excluded: set[str] = set()
    excluded_errors: dict[str, BackendError] = {}
    last_error: BackendError | None = None
    cost_estimate = _composite_cost_estimate_for_body(body, model=model)
    while True:
        backend = await select(
            backends_list,
            model=model,
            excluded=frozenset(excluded),
            preferred_id=preferred_id,
            cost_estimate=cost_estimate,
        )
        if backend is None:
            break
        handle = CallHandle()
        ts_start = time.time()
        iterator = call(backend, body, handle)
        try:
            first_chunk = await iterator.__anext__()
        except StopAsyncIteration:
            ts_end = time.time()
            _log_attempt(
                usage_log,
                body=body,
                model=model,
                route_name=route_name,
                stream=True,
                session_id=session_id,
                backend=backend,
                handle=handle,
                ts_start=ts_start,
                ts_end=ts_end,
                error=None,
                resp_body=None,
                user_id=user_id,
                api_key_id=api_key_id,
                requested_model=requested_model,
                requested_reasoning_effort=requested_reasoning_effort,
                routing_mode=routing_mode,
                prompt_complexity_class=None,
                recommender_classifier_cell=recommender_classifier_cell,
                recommender_raw_output=recommender_raw_output,
                recommender_source=recommender_source,
                prompt_embedding=prompt_embedding,
            )
            _remember_binding(session_registry, session_id, backend.id)
            return StreamingResponse(_empty_iter(), media_type="text/event-stream")
        except BackendError as exc:
            ts_end = time.time()
            _log_attempt(
                usage_log,
                body=body,
                model=model,
                route_name=route_name,
                stream=True,
                session_id=session_id,
                backend=backend,
                handle=handle,
                ts_start=ts_start,
                ts_end=ts_end,
                error=exc,
                resp_body=None,
                user_id=user_id,
                api_key_id=api_key_id,
                requested_model=requested_model,
                requested_reasoning_effort=requested_reasoning_effort,
                routing_mode=routing_mode,
                prompt_complexity_class=None,
                recommender_classifier_cell=recommender_classifier_cell,
                recommender_raw_output=recommender_raw_output,
                recommender_source=recommender_source,
                prompt_embedding=prompt_embedding,
            )
            last_error = exc
            if exc.classification not in RETRYABLE:
                raise _terminal_http(exc) from exc
            excluded.add(backend.id)
            excluded_errors[backend.id] = exc
            continue
        _remember_binding(session_registry, session_id, backend.id)

        # Always extract and strip complexity markers from streaming responses
        stream = _prepend(first_chunk, iterator)
        stream = _extract_complexity_from_stream(stream)
        # Also scrub a trailing {{{...}}} marker if the model emits one as a
        # closing tag (e.g. {{{/2}}} at the very end of the answer).
        stream = _strip_trailing_complexity_marker(stream)
        # Final scrub: full-text .done events (which Hermes / codex-cli
        # often read for the final UI render). The delta filters above
        # never touch these — see _scrub_full_text_events docstring.
        stream = _scrub_full_text_events(stream)
        if peer_quality_capture is None:
            peer_quality_capture = _PeerQualityCapture(nonce="")
        stream = _strip_peer_quality_markers_from_stream(stream, capture=peer_quality_capture)

        # Wrap stream with safe error handling for peer disconnections
        stream = _safe_stream(stream, backend_id=backend.id)

        return StreamingResponse(
            _log_on_complete(
                stream,
                usage_log=usage_log,
                body=body,
                model=model,
                route_name=route_name,
                session_id=session_id,
                backend=backend,
                handle=handle,
                ts_start=ts_start,
                user_id=user_id,
                api_key_id=api_key_id,
                requested_model=requested_model,
                requested_reasoning_effort=requested_reasoning_effort,
                routing_mode=routing_mode,
                recommender_classifier_cell=recommender_classifier_cell,
                recommender_raw_output=recommender_raw_output,
                recommender_source=recommender_source,
                prompt_embedding=prompt_embedding,
                peer_quality_capture=peer_quality_capture,
            ),
            media_type="text/event-stream",
        )

    # All primary backends exhausted. Compute recovery timestamp.
    recovery_ts: float | None = None
    for backend in backends_list:
        try:
            usage = await backend.usage_snapshot()
            if usage and usage.cooldown_until_ts and (recovery_ts is None or usage.cooldown_until_ts < recovery_ts):
                recovery_ts = usage.cooldown_until_ts
        except Exception:
            pass  # Skip backends that fail to report usage

    # Try fallback strategies.
    error_classifications = {backend_id: error.classification for backend_id, error in excluded_errors.items()}

    backend_status = await _collect_backend_status(backends_list)
    if should_attempt_fallback(error_classifications):
        fallback = FallbackExecutor()
        # TODO: Implement fallback retry logic here
        # For now, just log that we attempted it
        fallback.record_attempt(
            "considered_fallback",
            "skipped",
            reason="fallback_not_yet_implemented",
        )
        raise _no_viable(
            model=model,
            last_error=last_error,
            excluded_backends=excluded_errors,
            fallback_executor=fallback,
            recovery_ts=recovery_ts,
            backend_status=backend_status,
        )

    raise _no_viable(
        model=model,
        last_error=last_error,
        excluded_backends=excluded_errors,
        fallback_executor=None,
        recovery_ts=recovery_ts,
        backend_status=backend_status,
    )


async def _prepend(first: bytes, rest: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    yield first
    async for chunk in rest:
        yield chunk


async def _safe_stream(source: AsyncIterator[bytes], backend_id: str = "unknown") -> AsyncIterator[bytes]:
    """Safely stream chunks, gracefully handling peer disconnections.

    When a peer closes connection without completing the response body,
    log the error but don't crash the ASGI app. The client sees the partial
    response (already sent headers are committed).
    """
    try:
        async for chunk in source:
            yield chunk
    except Exception as exc:
        # Peer closed connection or other streaming error
        ts = _utc_timestamp()
        logger.warning(f"[{ts}] streaming error from {backend_id}: {type(exc).__name__}: {exc}")
        # Don't re-raise; client already got partial response. Just stop streaming.


# Responses API event types whose `delta` field carries text-like content that
# the model may inadvertently lead with the classifier marker. We strip from
# all of these. We deliberately do NOT include
# `response.function_call_arguments.delta` because that field carries
# tool-call JSON fragments — removing `{` / `}` would corrupt the JSON.
_TEXT_BEARING_DELTA_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "response.output_text.delta",
        "response.reasoning_summary_text.delta",
    }
)

# Final / accumulated-text events. These carry the FULL response text after
# streaming completes, and Hermes / codex-cli often read from them for the
# final display (bypassing the delta stream). Marker scrubbing has to cover
# these too or a model that voluntarily echoes "{{{N}}}\n\n..." at the start
# of its response (because its conversation history is poisoned with prior
# marker-prefixed turns) will leak through to the user even though every
# delta event was stripped clean.
#
# Map: event_type -> JSON path to the string field that holds the full text.
# Path syntax is dot-separated keys, with `[N]` for list indexing.
_FULL_TEXT_EVENT_PATHS: dict[str, str] = {
    "response.output_text.done": "text",
    "response.content_part.done": "part.text",
    "response.output_item.done": "item.content[0].text",
}


def _get_at_path(obj: Any, path: str) -> Any:
    """Read a value out of a nested JSON-like dict/list using a 'a.b[0].c'
    style path. Returns None if any step is missing or wrongly typed.
    Cheap parser; not a full JSONPath impl."""
    cur = obj
    for part in path.split("."):
        # Split off any list indices like 'content[0]'
        while "[" in part and part.endswith("]"):
            head, idx_str = part[: part.index("[")], part[part.index("[") + 1 : -1]
            try:
                idx = int(idx_str)
            except ValueError:
                return None
            if head:
                cur = cur.get(head) if isinstance(cur, dict) else None
            if not isinstance(cur, list) or idx >= len(cur):
                return None
            cur = cur[idx]
            part = ""
        if part:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        if cur is None:
            return None
    return cur


def _set_at_path(obj: Any, path: str, value: Any) -> bool:
    """In-place set a value at the given path. Returns True on success."""
    cur = obj
    parts = path.split(".")
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        while "[" in part and part.endswith("]"):
            head, idx_str = part[: part.index("[")], part[part.index("[") + 1 : -1]
            try:
                idx = int(idx_str)
            except ValueError:
                return False
            if head:
                if not isinstance(cur, dict):
                    return False
                cur = cur.get(head)
            if not isinstance(cur, list) or idx >= len(cur):
                return False
            if last and "[" not in part[part.index("[") + 1 :]:
                cur[idx] = value
                return True
            cur = cur[idx]
            part = ""
        if part:
            if not isinstance(cur, dict):
                return False
            if last:
                cur[part] = value
                return True
            cur = cur.get(part)
            if cur is None:
                return False
    return False


def _scrub_full_text_event(data: dict[str, Any]) -> bool:
    """If this event is a known full-text 'done' event, strip leading +
    trailing {{{...}}} markers from its text field. Returns True if the
    event was mutated.
    """
    et = data.get("type")
    if not isinstance(et, str):
        return False
    path = _FULL_TEXT_EVENT_PATHS.get(et)
    if path is None:
        return False
    text = _get_at_path(data, path)
    if not isinstance(text, str) or not text:
        return False
    # Reuse the existing leading + trailing strippers.
    _, cleaned = _extract_complexity_class(text)
    cleaned = _strip_trailing_complexity_marker_text(cleaned)
    if cleaned == text:
        return False
    _set_at_path(data, path, cleaned)
    return True


def _scrub_peer_quality_from_event(data: dict[str, Any], *, capture: _PeerQualityCapture) -> bool:
    """Strip qop markers from user-visible text fields and collect valid opinions."""
    modified = False
    try:
        choices = data.get("choices")
        if choices and len(choices) > 0:
            delta = choices[0].get("delta", {})
            content = delta.get("content") or ""
            if isinstance(content, str) and "<<qop" in content:
                cleaned = capture.apply(content)
                if cleaned != content:
                    delta["content"] = cleaned
                    choices[0]["delta"] = delta
                    data["choices"] = choices
                    modified = True
        elif data.get("type") in _TEXT_BEARING_DELTA_EVENT_TYPES:
            delta_text = data.get("delta") or ""
            if isinstance(delta_text, str) and "<<qop" in delta_text:
                cleaned = capture.apply(delta_text)
                if cleaned != delta_text:
                    data["delta"] = cleaned
                    modified = True
        elif data.get("type") in _FULL_TEXT_EVENT_PATHS:
            path = _FULL_TEXT_EVENT_PATHS[data["type"]]
            text = _get_at_path(data, path)
            if isinstance(text, str) and "<<qop" in text:
                cleaned = capture.apply(text)
                if cleaned != text:
                    modified = _set_at_path(data, path, cleaned)
    except (AttributeError, KeyError, IndexError, TypeError):
        return False
    return modified


def _event_delta_text(event_bytes: bytes) -> str:
    """Concatenated delta text from one SSE event (chat-completions + Responses API).

    Reads both chat-completions choices[0].delta.content and the union of
    text-bearing Responses API delta event types defined above.
    """
    try:
        event_str = event_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    out = ""
    for line in event_str.split("\n"):
        if not line.startswith("data: "):
            continue
        json_str = line[6:]
        if json_str.strip() == "[DONE]":
            continue
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            continue
        try:
            choices = data.get("choices")
            if choices and len(choices) > 0:
                out += choices[0].get("delta", {}).get("content") or ""
                continue
            if data.get("type") in _TEXT_BEARING_DELTA_EVENT_TYPES:
                out += data.get("delta") or ""
        except (AttributeError, KeyError, IndexError, TypeError):
            pass
    return out


def _strip_chars_from_event(event_bytes: bytes, n: int) -> tuple[bytes, int]:
    """Strip up to n characters from delta content fields in this event.

    Walks data: lines in order; for each, removes up to (n - already_stripped)
    leading chars from delta.content (chat-completions) or delta (Responses API).
    Returns (modified_event_bytes, chars_actually_stripped).
    """
    if n <= 0:
        return event_bytes, 0
    try:
        event_str = event_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return event_bytes, 0
    new_lines: list[str] = []
    stripped_total = 0
    for line in event_str.split("\n"):
        if not (line.startswith("data: ") and stripped_total < n):
            new_lines.append(line)
            continue
        json_str = line[6:]
        if json_str.strip() == "[DONE]":
            new_lines.append(line)
            continue
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            new_lines.append(line)
            continue
        modified = False
        try:
            choices = data.get("choices")
            if choices and len(choices) > 0:
                delta = choices[0].get("delta", {})
                content = delta.get("content") or ""
                if content:
                    take = min(n - stripped_total, len(content))
                    delta["content"] = content[take:]
                    stripped_total += take
                    choices[0]["delta"] = delta
                    data["choices"] = choices
                    modified = True
            elif data.get("type") in _TEXT_BEARING_DELTA_EVENT_TYPES:
                delta_text = data.get("delta") or ""
                if delta_text:
                    take = min(n - stripped_total, len(delta_text))
                    data["delta"] = delta_text[take:]
                    stripped_total += take
                    modified = True
        except (AttributeError, KeyError, IndexError, TypeError):
            pass
        new_lines.append("data: " + json.dumps(data) if modified else line)
    return "\n".join(new_lines).encode("utf-8"), stripped_total


# Max delta chars to accumulate while searching for the leading marker.
# Numeric marker is 7 chars ("{{{N}}}"); allow generous slack for whitespace
# or unexpected variants. Once exceeded we give up and flush as-is.
_COMPLEXITY_MARKER_LOOKAHEAD = 32

# Sliding-window size (in delta chars) used by the trailing-marker stripper.
# The longest trailing marker we expect is ~16 chars ({{{/N}}} = 8, or
# {{{complexity: Medium}}} = 23); 32 covers all observed variants with slack.
_COMPLEXITY_TRAILING_WINDOW = 32


def _strip_chars_from_event_end(event_bytes: bytes, n: int) -> tuple[bytes, int]:
    """Strip up to n characters from the END of delta content fields in this event.

    Mirror of _strip_chars_from_event but operating on the tail. Walks data:
    lines in reverse so the LAST line's delta content is shaved first (it
    holds the trailing-most text); excess strip-budget then bleeds into the
    preceding data: line's delta tail.
    """
    if n <= 0:
        return event_bytes, 0
    try:
        event_str = event_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return event_bytes, 0
    lines = event_str.split("\n")
    stripped_total = 0
    for i in range(len(lines) - 1, -1, -1):
        if stripped_total >= n:
            break
        line = lines[i]
        if not line.startswith("data: "):
            continue
        json_str = line[6:]
        if json_str.strip() == "[DONE]":
            continue
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            continue
        modified = False
        try:
            choices = data.get("choices")
            if choices and len(choices) > 0:
                delta = choices[0].get("delta", {})
                content = delta.get("content") or ""
                if content:
                    take = min(n - stripped_total, len(content))
                    delta["content"] = content[: len(content) - take]
                    stripped_total += take
                    choices[0]["delta"] = delta
                    data["choices"] = choices
                    modified = True
            elif data.get("type") in _TEXT_BEARING_DELTA_EVENT_TYPES:
                delta_text = data.get("delta") or ""
                if delta_text:
                    take = min(n - stripped_total, len(delta_text))
                    data["delta"] = delta_text[: len(delta_text) - take]
                    stripped_total += take
                    modified = True
        except (AttributeError, KeyError, IndexError, TypeError):
            pass
        if modified:
            lines[i] = "data: " + json.dumps(data)
    return "\n".join(lines).encode("utf-8"), stripped_total


async def _extract_complexity_from_stream(
    source: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Strip a leading {{{N}}} / {{{...}}} marker from an SSE stream.

    This is now defensive cleanup only. Older prompts or context-poisoned
    threads may still cause a model to lead with marker-shaped text.
    Tokenizers usually split the marker across multiple SSE events (e.g.
    "{{{", "2", "}}}"), so we buffer events until either the full marker
    has arrived (then strip its bytes across whichever events carry them)
    or we can prove no marker is present (then flush as-is). Once decided,
    the rest of the stream is passed through unchanged.
    """
    buffered_events: list[bytes] = []
    accumulated_text = ""
    decided = False
    leftover = b""

    def _strip_and_flush(chars_to_strip: int) -> list[bytes]:
        nonlocal buffered_events, decided
        out: list[bytes] = []
        for ev in buffered_events:
            if chars_to_strip > 0:
                modified, stripped = _strip_chars_from_event(ev, chars_to_strip)
                chars_to_strip -= stripped
                out.append(modified + b"\n\n")
            else:
                out.append(ev + b"\n\n")
        buffered_events = []
        decided = True
        return out

    def _flush_as_is() -> list[bytes]:
        nonlocal buffered_events, decided
        out = [ev + b"\n\n" for ev in buffered_events]
        buffered_events = []
        decided = True
        return out

    def _decision_output(force: bool = False) -> list[bytes]:
        """Return bytes to yield once a decision is reachable, or [] if still buffering.

        Three accepted marker formats:
          A) {{{N}}} where N ∈ {1,2,3}                — canonical
          B) {{{...}}} (any other braced leading token) — defensive strip
          C) bare digit 1|2|3 followed by a blank line  — model dropped braces

        When force=True (end of stream / [DONE]), commits to a decision
        unconditionally: strip the best match available, otherwise flush as-is.
        """
        # A) strict {{{N}}}
        match = re.match(r"^\s*\{\{\{([123])\}\}\}", accumulated_text)
        if match:
            return _strip_and_flush(match.end())

        # B) any {{{...}}}
        match = re.match(r"^\s*\{\{\{[^}]*\}\}\}", accumulated_text)
        if match:
            return _strip_and_flush(match.end())

        # C) bare digit followed by blank line
        match = re.match(r"^\s*([123])[ \t]*\n[ \t]*\n", accumulated_text)
        if match:
            return _strip_and_flush(match.end())

        stripped_acc = accumulated_text.lstrip()

        # Decide whether to keep buffering or flush as-is.
        if not stripped_acc:
            return _flush_as_is() if force else []

        first = stripped_acc[0]
        could_be_marker = first == "{" or first in "123"

        # If digit-starting and we have 2+ chars, the next char tells us whether
        # this is a bare-digit marker candidate (followed by whitespace/newline)
        # or legitimate content like "2 minutes" / "2." / "20 things".
        if first in "123" and len(stripped_acc) >= 2:
            nxt = stripped_acc[1]
            if nxt not in (" ", "\t", "\n", "\r"):
                return _flush_as_is()
            # Else: still a bare-digit candidate, keep buffering for the blank line.

        if not could_be_marker:
            return _flush_as_is()

        if force or len(accumulated_text) >= _COMPLEXITY_MARKER_LOOKAHEAD:
            return _flush_as_is()

        return []

    async for chunk in source:
        if decided:
            yield chunk
            continue

        combined = leftover + chunk
        parts = combined.split(b"\n\n")
        leftover = parts[-1]
        events = parts[:-1]

        for ev in events:
            if decided:
                yield ev + b"\n\n"
                continue
            buffered_events.append(ev)
            accumulated_text += _event_delta_text(ev)
            force = b"data: [DONE]" in ev
            for out_bytes in _decision_output(force=force):
                yield out_bytes

        if decided and leftover:
            yield leftover
            leftover = b""

    # Stream ended; force a final decision on anything still buffered.
    if buffered_events:
        for out_bytes in _decision_output(force=True):
            yield out_bytes
    if leftover:
        yield leftover


async def _scrub_full_text_events(
    source: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Strip leading / trailing {{{...}}} markers from full-text 'done' events.

    The streaming filters (_extract_complexity_from_stream and
    _strip_trailing_complexity_marker) scrub the .delta event stream only.
    Codex also emits accumulated-text events at the end of each output:
    response.output_text.done, response.content_part.done,
    response.output_item.done — each carries the FULL response text and
    Hermes / codex-cli often read from those for the final display. If a
    marker was in the deltas (because the model voluntarily echoed it from
    its conversation history), it ends up in these too, bypassing both
    delta filters. This pass rewrites those events in-place so the
    consumer never sees the marker.

    Single-pass, no buffering: every event is parsed, the known .done
    event types have their text field scrubbed, then the (possibly
    modified) event is reserialized and yielded.
    """
    leftover = b""
    async for chunk in source:
        combined = leftover + chunk
        parts = combined.split(b"\n\n")
        leftover = parts[-1]
        events = parts[:-1]
        for ev in events:
            try:
                ev_str = ev.decode("utf-8")
            except UnicodeDecodeError:
                yield ev + b"\n\n"
                continue
            lines = ev_str.split("\n")
            new_lines: list[str] = []
            for line in lines:
                if not line.startswith("data: "):
                    new_lines.append(line)
                    continue
                json_str = line[6:]
                if json_str.strip() == "[DONE]":
                    new_lines.append(line)
                    continue
                try:
                    data = json.loads(json_str)
                except (json.JSONDecodeError, ValueError):
                    new_lines.append(line)
                    continue
                if _scrub_full_text_event(data):
                    new_lines.append("data: " + json.dumps(data))
                else:
                    new_lines.append(line)
            yield ("\n".join(new_lines)).encode("utf-8") + b"\n\n"
    if leftover:
        yield leftover


async def _strip_peer_quality_markers_from_stream(
    source: AsyncIterator[bytes],
    *,
    capture: _PeerQualityCapture,
) -> AsyncIterator[bytes]:
    """Strip nonce-bound peer-quality markers from text-bearing SSE events.

    This pass is intentionally limited to user-visible text fields. Tool-call
    argument deltas and reasoning structures are passed through unchanged.
    """
    leftover = b""
    async for chunk in source:
        combined = leftover + chunk
        parts = combined.split(b"\n\n")
        leftover = parts[-1]
        events = parts[:-1]
        for ev in events:
            try:
                ev_str = ev.decode("utf-8")
            except UnicodeDecodeError:
                yield ev + b"\n\n"
                continue
            lines = ev_str.split("\n")
            new_lines: list[str] = []
            for line in lines:
                if not line.startswith("data: "):
                    new_lines.append(line)
                    continue
                json_str = line[6:]
                if json_str.strip() == "[DONE]":
                    new_lines.append(line)
                    continue
                try:
                    data = json.loads(json_str)
                except (json.JSONDecodeError, ValueError):
                    new_lines.append(line)
                    continue
                mutated = _scrub_peer_quality_from_event(data, capture=capture)
                # Subtract the audit's token cost from the usage the client sees,
                # so the hidden injection never moves Codex's context meter
                # (; Codex reads usage.input_tokens).
                mutated = _subtract_audit_tokens_from_event(data, capture=capture) or mutated
                if mutated:
                    new_lines.append("data: " + json.dumps(data))
                else:
                    new_lines.append(line)
            yield ("\n".join(new_lines)).encode("utf-8") + b"\n\n"
    if leftover:
        yield leftover


def _subtract_audit_tokens_from_event(data: dict[str, Any], *, capture: _PeerQualityCapture) -> bool:
    """Subtract the injected audit tokens from a response's usage block.

    Codex derives "Context X% left" from the response usage.input_tokens
    (verified against codex-rs core/src/client.rs). The audit's input tokens are
    real, so without this they would shrink the meter. We subtract the exact
    injected count from the client-facing input_tokens/total_tokens so
    the meter reflects only the user's real conversation. callosum's own quota
    accounting reads the upstream usage separately and is unaffected.
    """
    n = capture.injected_tokens
    if n <= 0:
        return False
    usage = data.get("usage")
    if not isinstance(usage, dict):
        resp = data.get("response")
        usage = resp.get("usage") if isinstance(resp, dict) else None
    if not isinstance(usage, dict):
        return False
    changed = False
    for key in ("input_tokens", "prompt_tokens", "total_tokens"):
        val = usage.get(key)
        if isinstance(val, int):
            usage[key] = max(0, val - n)
            changed = True
    return changed


async def _strip_trailing_complexity_marker(
    source: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Strip a trailing {{{...}}} marker if the model emits one as a closing tag.

    Maintains a sliding window of the most recent SSE events whose accumulated
    delta text is at least _COMPLEXITY_TRAILING_WINDOW characters. Older
    events are flushed downstream the moment they fall out of the window — so
    streaming latency only increases by ~32 chars worth of buffering. On
    stream end (or [DONE]), checks the buffered tail for a trailing
    {{{...}}} pattern and shaves its bytes off the appropriate event(s)
    before flushing the rest.
    """
    trail: list[tuple[bytes, str]] = []
    trail_chars = 0
    leftover = b""

    def _flush(final: bool) -> list[bytes]:
        nonlocal trail, trail_chars
        if final and trail:
            tail_text = "".join(t for _, t in trail)
            match = re.search(r"\{\{\{[^}]*\}\}\}\s*$", tail_text)
            if match:
                chars_to_strip = len(match.group(0))
                for i in range(len(trail) - 1, -1, -1):
                    if chars_to_strip <= 0:
                        break
                    ev_b, ev_text = trail[i]
                    new_ev, stripped = _strip_chars_from_event_end(ev_b, chars_to_strip)
                    if stripped > 0:
                        ev_text = ev_text[: len(ev_text) - stripped] if stripped <= len(ev_text) else ""
                        trail[i] = (new_ev, ev_text)
                        chars_to_strip -= stripped
        out = [ev + b"\n\n" for ev, _ in trail]
        trail = []
        trail_chars = 0
        return out

    async for chunk in source:
        combined = leftover + chunk
        parts = combined.split(b"\n\n")
        leftover = parts[-1]
        events = parts[:-1]

        for ev in events:
            is_done = b"data: [DONE]" in ev
            delta_text = _event_delta_text(ev)
            trail.append((ev, delta_text))
            trail_chars += len(delta_text)

            if is_done:
                for out_bytes in _flush(final=True):
                    yield out_bytes
                continue

            while len(trail) > 1 and trail_chars - len(trail[0][1]) >= _COMPLEXITY_TRAILING_WINDOW:
                old_ev, old_text = trail.pop(0)
                trail_chars -= len(old_text)
                yield old_ev + b"\n\n"

    for out_bytes in _flush(final=True):
        yield out_bytes
    if leftover:
        yield leftover


async def _empty_iter() -> AsyncIterator[bytes]:
    if False:
        yield b""


async def _log_on_complete(
    source: AsyncIterator[bytes],
    *,
    usage_log: UsageLog | None,
    body: dict[str, Any],
    model: str,
    route_name: str,
    session_id: str | None,
    backend: Backend,
    handle: CallHandle,
    ts_start: float,
    user_id: int | None = None,
    api_key_id: int | None = None,
    requested_model: str | None = None,
    requested_reasoning_effort: str | None = None,
    routing_mode: str = "pass-through",
    recommender_classifier_cell: str | None = None,
    recommender_raw_output: str | None = None,
    recommender_source: str | None = None,
    prompt_embedding: bytes | None = None,
    peer_quality_capture: _PeerQualityCapture | None = None,
) -> AsyncIterator[bytes]:
    """Pass-through wrapper that writes the usage log row when the stream ends.

    The backend's `*_stream` methods populate `handle.stream_summary` after the
    final chunk is yielded, so we log on the other side of the async-for loop.
    """
    async for chunk in source:
        yield chunk
    ts_end = time.time()
    _log_attempt(
        usage_log,
        body=body,
        model=model,
        route_name=route_name,
        stream=True,
        session_id=session_id,
        backend=backend,
        handle=handle,
        ts_start=ts_start,
        ts_end=ts_end,
        error=None,
        resp_body=None,  # for streams the raw body lives in handle.stream_summary
        user_id=user_id,
        api_key_id=api_key_id,
        requested_model=requested_model,
        requested_reasoning_effort=requested_reasoning_effort,
        routing_mode=routing_mode,
        prompt_complexity_class=None,  # retired: marker-only label no longer collected
        recommender_classifier_cell=recommender_classifier_cell,
        recommender_raw_output=recommender_raw_output,
        recommender_source=recommender_source,
        prompt_embedding=prompt_embedding,
        peer_quality_capture=peer_quality_capture,
    )


def _clean_sse_blob(blob: bytes | None, *, peer_quality_capture: _PeerQualityCapture | None = None) -> bytes | None:
    """Remove complexity markers from SSE blob before storage.

    Parses SSE format, extracts and cleans response content from delta/text fields,
    and reconstructs the blob for database storage. Ensures markers never persist
    in the database even if they leak through the stream cleaning phase.
    """
    if blob is None:
        return None

    try:
        text = blob.decode("utf-8")
        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Check if this is a data line with JSON content
            if line.startswith("data: "):
                try:
                    json_str = line[6:]  # Strip 'data: '
                    data = json.loads(json_str)

                    # Handle text-bearing Responses API delta events
                    if data.get("type") in _TEXT_BEARING_DELTA_EVENT_TYPES:
                        delta = data.get("delta", "")
                        if isinstance(delta, str) and delta:
                            _, cleaned_delta = _extract_complexity_class(delta)
                            cleaned_delta = _strip_trailing_complexity_marker_text(cleaned_delta)
                            if peer_quality_capture is not None and "<<qop" in cleaned_delta:
                                cleaned_delta = peer_quality_capture.apply(cleaned_delta)
                            data["delta"] = cleaned_delta

                    # Handle full-text 'done' events (Hermes / codex-cli often
                    # read these for final UI render — must be scrubbed too)
                    elif data.get("type") in _FULL_TEXT_EVENT_PATHS:
                        _scrub_full_text_event(data)
                        if peer_quality_capture is not None:
                            _scrub_peer_quality_from_event(data, capture=peer_quality_capture)

                    # Handle chat completions format (choices[0].delta.content)
                    elif "choices" in data and len(data.get("choices", [])) > 0:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if isinstance(content, str) and content:
                            _, cleaned_content = _extract_complexity_class(content)
                            cleaned_content = _strip_trailing_complexity_marker_text(cleaned_content)
                            if peer_quality_capture is not None and "<<qop" in cleaned_content:
                                cleaned_content = peer_quality_capture.apply(cleaned_content)
                            delta["content"] = cleaned_content
                            data["choices"][0]["delta"] = delta

                    cleaned_lines.append("data: " + json.dumps(data))
                except (json.JSONDecodeError, KeyError, TypeError):
                    # Not JSON or unexpected format, pass through unchanged
                    cleaned_lines.append(line)
            else:
                # Non-data lines pass through unchanged
                cleaned_lines.append(line)

        cleaned_text = "\n".join(cleaned_lines)
        return cleaned_text.encode("utf-8")
    except (UnicodeDecodeError, AttributeError):
        # If we can't decode, return original blob
        return blob


def _log_attempt(
    usage_log: UsageLog | None,
    *,
    body: dict[str, Any],
    model: str,
    route_name: str,
    stream: bool,
    session_id: str | None,
    backend: Backend,
    handle: CallHandle,
    ts_start: float,
    ts_end: float,
    error: BackendError | None,
    resp_body: dict[str, Any] | None,
    user_id: int | None = None,
    api_key_id: int | None = None,
    requested_model: str | None = None,
    requested_reasoning_effort: str | None = None,
    routing_mode: str = "pass-through",
    prompt_complexity_class: int | None = None,
    recommender_classifier_cell: str | None = None,
    recommender_raw_output: str | None = None,
    recommender_source: str | None = None,
    prompt_embedding: bytes | None = None,
    peer_quality_capture: _PeerQualityCapture | None = None,
) -> None:
    if usage_log is None:
        return
    req_payload = json.dumps(body).encode()
    if stream and handle.stream_summary is not None:
        resp_payload: bytes | None = handle.stream_summary.raw_blob
        # Clean markers from raw blob before storing in database
        resp_payload = _clean_sse_blob(resp_payload, peer_quality_capture=peer_quality_capture)
        response_bytes = handle.stream_summary.total_bytes
        completed = handle.stream_summary.completed_response
        tokens = _extract_tokens(completed.get("usage") if completed else None)
    elif resp_body is not None:
        resp_payload = json.dumps(resp_body).encode()
        response_bytes = len(resp_payload)
        tokens = _extract_tokens(resp_body.get("usage"))
    else:
        resp_payload = None
        response_bytes = None
        tokens = _Tokens(None, None, None, None, None)
    status = error.status_code if error is not None else 200
    classification = error.classification if error is not None else "ok"
    # Per-request effective mode for the canary baseline accounting.
    # Pulled from the ContextVar set at the routing entry; None means
    # the request never went through the router (no-router path /
    # pass-through), which we record as such rather than fabricating
    # an effective mode.
    effective_routing_mode = _effective_routing_mode_context.get()
    entry = UsageLogEntry(
        ts_start=ts_start,
        ts_end=ts_end,
        route=route_name,
        stream=stream,
        session_id=session_id,
        backend_id=backend.id,
        model=model,
        reasoning_effort=_extract_reasoning_effort(body),
        status=int(status) if status is not None else 0,
        classification=classification,
        request_bytes=len(req_payload),
        response_bytes=response_bytes,
        prompt_tokens=tokens.prompt,
        completion_tokens=tokens.completion,
        total_tokens=tokens.total,
        cached_tokens=tokens.cached,
        reasoning_tokens=tokens.reasoning,
        quota_before=handle.quota_before,
        quota_after=handle.quota_after,
        req_payload=req_payload,
        resp_payload=resp_payload,
        upstream_headers=dict(handle.upstream_headers) if handle.upstream_headers else None,
        user_id=user_id,
        api_key_id=api_key_id,
        requested_model=requested_model,
        requested_reasoning_effort=requested_reasoning_effort,
        routing_mode=routing_mode,
        prompt_complexity_class=prompt_complexity_class,
        client_request=body,
        recommender_classifier_cell=recommender_classifier_cell,
        recommender_raw_output=recommender_raw_output,
        recommender_source=recommender_source,
        prompt_embedding=prompt_embedding,
        effective_routing_mode=effective_routing_mode,
    )
    request_id = usage_log.record(entry)
    if peer_quality_capture is not None and peer_quality_capture.nonce:
        if peer_quality_capture.opinions:
            usage_log.record_peer_quality_opinions(
                request_id=request_id,
                session_id=session_id,
                judge_backend_id=backend.id,
                judge_model=model,
                judge_reasoning_effort=entry.reasoning_effort,
                opinions=peer_quality_capture.opinions,
                created_at=ts_end,
            )
        usage_log.record_peer_quality_capture_metrics(
            request_id=request_id,
            session_id=session_id,
            judge_backend_id=backend.id,
            judge_model=model,
            judge_reasoning_effort=entry.reasoning_effort,
            nonce=peer_quality_capture.nonce,
            opinion_count=len(peer_quality_capture.opinions),
            echo_count=peer_quality_capture.echo_count,
            malformed_count=peer_quality_capture.malformed_count,
            created_at=ts_end,
            injected_fired=peer_quality_capture.injected_fired,
            subject_count=peer_quality_capture.subject_count,
            skip_reason=peer_quality_capture.skip_reason,
        )
    # Store request_id in context for response handlers to access
    _request_id_context.set(request_id)
    # Finalize the forward cost estimate (): record the
    # realized weekly-quota delta with the integer-% verifiable flag. A 0
    # delta is unverifiable (NOT a 0-cost label) and only ever feeds
    # aggregate calibration. Best-effort; the row is already persisted, so a
    # finalize hiccup never affects logging.
    _ce = _COST_ESTIMATOR
    if _ce is not None and status == 200:
        try:
            from callosum.cell_grid import Cell as _Cell
            from callosum.usage_log import _is_reset_crossover

            qb = handle.quota_before
            qa = handle.quota_after
            delta: float | None = None
            verifiable = False
            if (
                qb is not None
                and qa is not None
                and qb.weekly_used_percent is not None
                and qa.weekly_used_percent is not None
                and not _is_reset_crossover(qb, qa)
            ):
                delta = float(qa.weekly_used_percent - qb.weekly_used_percent)
                verifiable = delta > 0
            _ce.finalize(
                request_id,
                cell=_Cell(model=model, reasoning_effort=entry.reasoning_effort or ""),
                observed_output_tokens=tokens.completion,
                observed_value=delta,
                verifiable=verifiable,
            )
        except Exception:
            logger.debug("cost.finalize failed for request_id=%s", request_id, exc_info=True)
    # Finalize the forward time estimate (): record the
    # realized wall-clock latency. Latency is always observable, so this is
    # always verifiable (no integer-resolution / unverifiable case like cost)
    # and has no local-zero branch — local cells carry real, often large
    # latency. Best-effort; the row is already persisted.
    _te = _TIME_ESTIMATOR
    if _te is not None and status == 200:
        try:
            from callosum.cell_grid import Cell as _Cell

            latency_ms = (ts_end - ts_start) * 1000.0
            _te.finalize(
                request_id,
                cell=_Cell(model=model, reasoning_effort=entry.reasoning_effort or ""),
                observed_output_tokens=tokens.completion,
                observed_value=latency_ms,
            )
        except Exception:
            logger.debug("time.finalize failed for request_id=%s", request_id, exc_info=True)
    # Failure observation: if this row represents a failed request,
    # write a structured record so the dev loop and operator
    # dashboards can detect regressions empirically. Best-effort —
    # registry write failures are swallowed inside `record()`.
    _fr = _FAILURE_REGISTRY
    if _fr is not None and ((isinstance(status, int) and status >= 500) or error is not None):
        from callosum.canary import FailureObservation

        # Symptom: distinguish the well-known classes; everything
        # else is `http_5xx` as a catch-all that an operator can
        # refine later by annotating the row. `sym` is intentionally
        # typed `str` rather than `BackendErrorClassification` because
        # the symptom taxonomy is open-ended and grows over time —
        # see failure_registry module docstring.
        sym: str
        if error is not None:
            sym = error.classification or "http_5xx"
        elif status == 408 or status == 504:
            sym = "timeout"
        else:
            sym = "http_5xx"
        # Responsible-layer attribution: simple kind-based heuristic.
        # Local backend serving a 5xx is `callosum-local`; remote
        # backend serving a 5xx is `upstream-remote` (Codex /
        # OpenAI incident is the most likely cause, callosum bug is
        # less likely). The dev loop filters out `upstream-remote`
        # rows when comparing local-side regressions.
        backend_kind = getattr(backend, "kind", "")
        if backend_kind == "litellm_gateway":
            resp_layer = "callosum-local"
        elif backend_kind == "codex_auth_vault":
            resp_layer = "upstream-remote"
        else:
            resp_layer = "unknown"
        _fr.record(
            FailureObservation(
                ts=ts_end,
                effective_mode=effective_routing_mode or "pass-through",
                symptom=sym,
                responsible_layer=resp_layer,
                request_id=request_id,
                detail=(error.message[:300] if error and error.message else None),
            )
        )


class _Tokens:
    __slots__ = ("prompt", "completion", "total", "cached", "reasoning")

    def __init__(
        self,
        prompt: int | None,
        completion: int | None,
        total: int | None,
        cached: int | None,
        reasoning: int | None,
    ) -> None:
        self.prompt = prompt
        self.completion = completion
        self.total = total
        self.cached = cached
        self.reasoning = reasoning


def _extract_tokens(usage: Any) -> _Tokens:
    """Map either a Responses-API or chat-completions usage block to a common struct."""
    if not isinstance(usage, dict):
        return _Tokens(None, None, None, None, None)
    prompt = _int(usage.get("input_tokens")) or _int(usage.get("prompt_tokens"))
    completion = _int(usage.get("output_tokens")) or _int(usage.get("completion_tokens"))
    total = _int(usage.get("total_tokens"))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    # Cached input tokens appear under input_tokens_details.cached_tokens on
    # the Responses API; some chat responses expose prompt_tokens_details.
    cached: int | None = None
    details_in = usage.get("input_tokens_details")
    if isinstance(details_in, dict):
        cached = _int(details_in.get("cached_tokens"))
    if cached is None:
        details_p = usage.get("prompt_tokens_details")
        if isinstance(details_p, dict):
            cached = _int(details_p.get("cached_tokens"))
    reasoning: int | None = None
    details_out = usage.get("output_tokens_details")
    if isinstance(details_out, dict):
        reasoning = _int(details_out.get("reasoning_tokens"))
    return _Tokens(prompt, completion, total, cached, reasoning)


def _stamp_reasoning_effort(body: dict[str, Any], effort: str) -> None:
    """Set the routed reasoning effort on the request body, tolerating a
    client-supplied reasoning: null.

    body.setdefault("reasoning", {}) returns the EXISTING value when the
    key is present, so a body carrying "reasoning": null yields None and
    the follow-on ["effort"] = ... raises
    TypeError: 'NoneType' object does not support item assignment — an
    uncaught 500 that Codex renders as "high demand". Coerce any non-dict
    reasoning to a fresh dict, preserving a client-provided reasoning
    object when it already is one.
    """
    reasoning = body.get("reasoning")
    if not isinstance(reasoning, dict):
        reasoning = {}
        body["reasoning"] = reasoning
    reasoning["effort"] = effort


def _extract_reasoning_effort(body: dict[str, Any]) -> str | None:
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if isinstance(effort, str):
            return effort
    # Some clients put reasoning_effort at the top level.
    effort_top = body.get("reasoning_effort")
    if isinstance(effort_top, str):
        return effort_top
    return None


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _quota_to_dict(q: Any) -> dict[str, Any] | None:
    """Serialize a CodexQuotaSnapshot to a JSON-friendly dict, or None if no
    snapshot is available yet. Backend type is loosely typed because only the
    codex_auth_vault backend produces snapshots; everything else returns None.
    """
    if q is None:
        return None
    return {
        "plan_type": q.plan_type,
        "active_limit": q.active_limit,
        "five_hourly_used_percent": q.five_hourly_used_percent,
        "weekly_used_percent": q.weekly_used_percent,
        "five_hourly_window_minutes": q.five_hourly_window_minutes,
        "weekly_window_minutes": q.weekly_window_minutes,
        "five_hourly_reset_at": q.five_hourly_reset_at,
        "weekly_reset_at": q.weekly_reset_at,
        "five_hourly_reset_after_seconds": q.five_hourly_reset_after_seconds,
        "weekly_reset_after_seconds": q.weekly_reset_after_seconds,
        "five_hourly_over_weekly_limit_percent": q.five_hourly_over_weekly_limit_percent,
        "credits_balance": q.credits_balance,
        "credits_has_credits": q.credits_has_credits,
        "credits_unlimited": q.credits_unlimited,
        "observed_at": q.observed_at,
    }


def _active_api_key_count(auth_service: Any) -> int:
    """Best-effort count of active API keys for 401 diagnostics. Returns -1
    if the count can't be obtained — the error path must never raise."""
    try:
        return int(auth_service.db.count_active_api_keys())
    except Exception:
        return -1


def _bearer(request: Request) -> str | None:
    raw = request.headers.get("authorization")
    if raw is None:
        return None
    parts = raw.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def _path_requires_api_key(path: str) -> bool:
    """Routes the bearer middleware enforces when auth is enabled."""
    return path.startswith("/v1/") or path.startswith("/diagnose/")


# Tiny prompt + instructions the diagnostic uses. Kept short to limit quota
# burn. The Codex Responses API requires `instructions` to be present and
# non-empty; an absent or empty value gets rejected with "Instructions are
# required" upstream.
_DIAGNOSE_PROMPT = "say only: ok"
_DIAGNOSE_INSTRUCTIONS = "You are a smoke-test probe. Reply minimally."


async def _diagnose_backend(backend: Backend, *, force: bool = False) -> dict[str, Any]:
    """Send one minimal streaming request and check upstream contract holds.

    When `force=True` the cooldown-skip guard is bypassed: the probe goes
    out even if the persisted snapshot claims the backend is still in
    cooldown. The periodic cooldown prober uses this to break out of a
    stale-snapshot lockout (see CodexAuthVaultBackend.clear_cooldown).
    """
    if not backend.advertised_models:
        return {
            "id": backend.id,
            "ok": False,
            "skipped": False,
            "stage": "config",
            "reason": "backend has no advertised_models",
        }
    usage = await backend.usage_snapshot()
    now = time.time()
    if not force and usage.cooldown_until_ts is not None and usage.cooldown_until_ts > now:
        return {
            "id": backend.id,
            "ok": True,
            "skipped": True,
            "stage": "cooldown",
            "reason": "backend in cooldown; not probed",
            "cooldown_until_ts": usage.cooldown_until_ts,
        }
    # Pick a real model for the probe — skip virtual selectors like
    # `auto-fallback`/`auto-learning`, which would route through the picker
    # rather than exercise the upstream contract directly.
    real_models = sorted(m for m in backend.advertised_models if m not in VIRTUAL_MODELS)
    if not real_models:
        return {
            "id": backend.id,
            "ok": False,
            "skipped": False,
            "stage": "config",
            "reason": "backend advertises only virtual models",
        }
    model = real_models[0]
    handle = CallHandle()
    body = {
        "model": model,
        "instructions": _DIAGNOSE_INSTRUCTIONS,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": _DIAGNOSE_PROMPT}],
            }
        ],
        "stream": True,
        "store": False,
    }
    try:
        async for _chunk in backend.responses_stream(body, handle):
            pass
    except BackendError as exc:
        # Build detailed error result with quota information if available
        result = {
            "id": backend.id,
            "ok": False,
            "skipped": False,
            "stage": "upstream",
            "classification": exc.classification,
            "status_code": exc.status_code,
            "reason": exc.message or exc.classification,
            "model": model,
        }
        # Include quota snapshots for rate-limited errors
        if exc.classification == "rate_limited" and handle.quota_after is not None:
            quota = handle.quota_after
            result["quota_after"] = quota
        return result
    return _evaluate_diagnostic(backend.id, handle, model, kind=backend.kind)


class _PeriodicSmokeTester:
    """Background asyncio task that re-runs the smoke test on a fixed interval
    so operators see live backend state (weekly resets, auth refreshes, model
    catalog churn) without restarting the proxy.

    The startup pass runs synchronously in `lifespan` before connections are
    accepted, so the operator sees current state immediately on launch. This
    class only handles the recurring follow-up ticks. interval_s=0 disables.
    """

    def __init__(self, *, backends: Sequence[Backend], interval_s: int, state_store: Any | None = None) -> None:
        self._backends = list(backends)
        self._interval_s = interval_s
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._state_store = state_store

    @property
    def enabled(self) -> bool:
        return self._interval_s > 0 and bool(self._backends)

    def start(self) -> None:
        if not self.enabled:
            return
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="periodic-smoke-test")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            # Sleep BEFORE the first periodic tick. The startup pass already
            # ran synchronously; the first re-run should land an interval later.
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval_s)
            except TimeoutError:
                pass
            else:
                # Stop event fired during the wait → exit cleanly.
                return
            try:
                # Refresh dynamic model lists and detect new models
                for backend in self._backends:
                    refresh = getattr(backend, "refresh_advertised_models", None)
                    if refresh is not None:
                        try:
                            models_before = set(backend.advertised_models)
                            await refresh()
                            models_after = set(backend.advertised_models)
                            if models_after > models_before and self._state_store is not None:
                                new_models = models_after - models_before
                                logger.info(
                                    "new models detected in periodic refresh for backend %r: %s",
                                    backend.id,
                                    sorted(new_models),
                                )
                                self._state_store.set_model_release_timestamp(time.time())
                        except Exception:
                            logger.exception("periodic model-list refresh failed for %r", backend.id)
                logger.info("periodic smoke test cycle")
                await _run_startup_smoke_test(self._backends)
            except Exception:
                logger.exception("periodic smoke test cycle failed")


class _PeriodicCooldownProber:
    """Background task that re-probes cooldown'd backends so the proxy can
    self-heal from stale-snapshot lockouts.

    The dispatcher excludes any backend whose persisted cooldown_until_ts is
    in the future, and every other probe path (startup smoke test, periodic
    smoke test, /diagnose/upstream) also skips cooldown'd backends by design
    — they're meant to respect a real cooldown rather than burn quota on a
    backend that just said no. That defensive behavior turns into a
    chicken-and-egg lockout whenever the persisted snapshot stops matching
    reality (upstream's reported weekly_reset_at was wrong, account was
    topped up out of band, original 429 was transient, etc.): the proxy
    can't learn that headroom returned because nothing inside it is allowed
    to probe.

    This task is the deliberate counter to that: every `interval_s` it
    sends a minimal forced probe to each cooldown'd backend and clears the
    cooldown if the probe comes back clean. Cost is tiny (a handful of
    tokens per backend per cycle); the failure mode it prevents is days-of-
    blocked-traffic stuck on stale state. interval_s=0 disables.
    """

    def __init__(self, *, backends: Sequence[Backend], interval_s: int) -> None:
        self._backends = list(backends)
        self._interval_s = interval_s
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @property
    def enabled(self) -> bool:
        return self._interval_s > 0 and bool(self._backends)

    def start(self) -> None:
        if not self.enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="periodic-cooldown-prober")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        # Probe once immediately at startup, THEN settle into the interval
        # cadence. After a cold boot (e.g. an overnight host shutdown) the
        # persisted cooldown/exhaustion snapshot is stale — quota windows
        # reset while the machine was off — but the startup smoke test skips
        # cooldown'd backends by design, so without this first pass nothing
        # re-validates the lockout until a full `interval_s` (default 1h)
        # has elapsed. That hour-after-every-boot window is exactly when the
        # operator hits "works last night, 503 every morning."
        await self._probe_cooldowned()
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval_s)
            except TimeoutError:
                pass
            else:
                return
            await self._probe_cooldowned()

    async def _probe_cooldowned(self) -> None:
        for backend in self._backends:
            try:
                usage = await backend.usage_snapshot()
            except Exception:
                logger.exception("cooldown prober: usage_snapshot failed for %r", backend.id)
                continue
            now = time.time()
            in_active_cooldown = usage.cooldown_until_ts is not None and usage.cooldown_until_ts > now
            if not in_active_cooldown and not usage.weekly_exhausted:
                continue
            try:
                result = await _diagnose_backend(backend, force=True)
            except Exception:
                logger.exception("cooldown prober: probe raised for %r", backend.id)
                continue
            if result.get("ok") and not result.get("skipped"):
                clear = getattr(backend, "clear_cooldown", None)
                if clear is not None:
                    try:
                        clear()
                        logger.warning(
                            "cooldown prober: %r probe succeeded; cooldown cleared "
                            "(was until %s, weekly_exhausted=%s)",
                            backend.id,
                            usage.cooldown_until_ts,
                            usage.weekly_exhausted,
                        )
                    except Exception:
                        logger.exception("cooldown prober: clear_cooldown raised for %r", backend.id)


async def _refresh_catalogs_pass(backends: Sequence[Backend], *, state_store: Any | None) -> list[Backend]:
    """Run one upstream model-catalog refresh per backend. Returns the
    backends whose catalog is STILL empty afterward — the signal that the
    backend's dependency (credential service) was not ready at boot and the
    fetch needs retrying. Also records a model-release timestamp when new
    models appear. Best-effort: a backend that raises keeps its cold-start set.
    """
    pending: list[Backend] = []
    for backend in backends:
        refresh = getattr(backend, "refresh_advertised_models", None)
        if refresh is None:
            continue
        try:
            models_before = set(backend.advertised_models)
            await refresh()
            models_after = set(backend.advertised_models)
            if models_after > models_before and state_store is not None:
                logger.info(
                    "new models detected for backend %r: %s — recording model release timestamp",
                    backend.id,
                    sorted(models_after - models_before),
                )
                state_store.set_model_release_timestamp(time.time())
        except Exception:
            logger.exception("startup model-list refresh failed for %r", backend.id)
        if not backend.advertised_models:
            pending.append(backend)
    return pending


async def _catalog_boot_resync(
    backends: Sequence[Backend],
    *,
    state_store: Any | None,
    attempts: int,
    interval_s: float,
) -> None:
    """Background cold-boot retry for backends that booted with an empty model
    catalog. Keeps re-fetching the upstream model list until every backend has
    one (or the attempt budget is spent), so the cell grid recovers within
    minutes of a boot-time dependency lag instead of waiting on the hourly
    smoke tester. The model-catalog leg of the stale-on-cold-boot class.
    """
    pending = list(backends)
    for attempt in range(1, attempts + 1):
        await asyncio.sleep(interval_s)
        pending = await _refresh_catalogs_pass(pending, state_store=state_store)
        if not pending:
            logger.warning("catalog boot resync: all catalogs populated after %d retr(y/ies)", attempt)
            return
        logger.warning(
            "catalog boot resync: attempt %d/%d, still-empty: %s",
            attempt,
            attempts,
            [b.id for b in pending],
        )
    logger.error(
        "catalog boot resync: gave up after %d attempts; still-empty: %s",
        attempts,
        [b.id for b in pending],
    )


async def _run_startup_smoke_test(backends_list: Sequence[Backend]) -> None:
    """Probe each non-cooldown backend with one minimal upstream call so the
    operator sees auth health (or quota exhaustion, or any other backend
    issue) immediately in the launch log. Doesn't fail startup — just logs.

    Reuses _diagnose_backend so the smoke test and the on-demand
    /diagnose/upstream endpoint share contract definitions.
    """
    logger.info("startup smoke test: probing %d backend(s)", len(backends_list))
    results = []
    for backend in backends_list:
        try:
            result = await _diagnose_backend(backend)
        except Exception as exc:
            logger.exception("startup smoke test: backend %r raised", backend.id)
            results.append({"id": backend.id, "ok": False, "stage": "exception", "reason": str(exc)})
            continue
        results.append(result)
        _log_smoke_result(result)
    n_ok = sum(1 for r in results if r.get("ok"))
    n_skipped = sum(1 for r in results if r.get("skipped"))
    n_failed = len(results) - n_ok - n_skipped
    logger.info(
        "startup smoke test: %d ok, %d skipped (cooldown), %d failed",
        n_ok - n_skipped,  # 'ok' counts skipped too in _diagnose_backend's contract
        n_skipped,
        n_failed,
    )


def _log_smoke_result(result: dict[str, Any]) -> None:
    """One human-readable line per backend smoke test result."""
    backend_id = result.get("id", "?")
    if result.get("skipped"):
        cooldown_until = result.get("cooldown_until_ts")
        reason = result.get("reason", "in cooldown")
        if cooldown_until is not None:
            from datetime import datetime

            try:
                reset_time = datetime.fromtimestamp(cooldown_until, tz=UTC).isoformat()
                reason = f"{reason} (reset at {reset_time})"
            except (OverflowError, ValueError, OSError):
                # Bogus/very-large cooldown timestamps (e.g. test sentinels or
                # stuck-clock state) shouldn't crash the smoke-test log line.
                reason = f"{reason} (reset at ts={cooldown_until})"
        logger.info("  [%s] SKIPPED — %s", backend_id, reason)
        return
    if result.get("ok"):
        upstream = result.get("upstream_status")
        logger.info("  [%s] OK — upstream %s", backend_id, upstream)
        return

    stage = result.get("stage", "?")
    classification = result.get("classification")
    status_code = result.get("status_code")
    reason = result.get("reason") or result.get("failed_checks") or "(no reason)"

    # For rate_limited errors, include quota exhaustion details
    quota_msg = ""
    if classification == "rate_limited":
        quota = result.get("quota_after")
        if quota is not None:
            from datetime import datetime

            exhaustion_info = []

            # 5-hour quota status
            if quota.five_hourly_used_percent is not None:
                pct = quota.five_hourly_used_percent
                exhaustion_info.append(f"5h-window {pct}%")
                if pct >= 99:
                    if quota.five_hourly_reset_at is not None:
                        reset = datetime.fromtimestamp(quota.five_hourly_reset_at, tz=UTC)
                        exhaustion_info.append(f"(resets {reset.isoformat()})")
                    else:
                        exhaustion_info.append("(resets ~5 hours)")

            # Weekly quota status
            if quota.weekly_used_percent is not None:
                pct = quota.weekly_used_percent
                exhaustion_info.append(f"weekly {pct}%")
                if pct >= 99:
                    if quota.weekly_reset_at is not None:
                        reset = datetime.fromtimestamp(quota.weekly_reset_at, tz=UTC)
                        exhaustion_info.append(f"(resets {reset.isoformat()})")
                    else:
                        exhaustion_info.append("(resets ~7 days)")

            if exhaustion_info:
                quota_msg = f" [{' | '.join(exhaustion_info)}]"
        reason = f"{reason}{quota_msg}"

    bits = [f"stage={stage}"]
    if classification is not None:
        bits.append(f"class={classification}")
    if status_code is not None:
        bits.append(f"status={status_code}")
    bits.append(f"detail={reason}")
    logger.warning("  [%s] FAILED — %s", backend_id, " ".join(bits))


def _evaluate_diagnostic(
    backend_id: str, handle: CallHandle, model: str, *, kind: str = "codex_auth_vault"
) -> dict[str, Any]:
    """Per-backend-kind check over a CallHandle from a diagnostic request.

    `codex_auth_vault` backends require Codex-shape contract guarantees —
    quota headers + Responses-API SSE terminal events. Non-Codex backends
    (e.g. `litellm_gateway`) only need to confirm a 2xx came back; they
    don't carry Codex-specific headers and the contract definition differs.
    """
    if kind == "codex_auth_vault":
        return _evaluate_codex_diagnostic(backend_id, handle, model)
    return _evaluate_generic_diagnostic(backend_id, handle, model)


def _evaluate_codex_diagnostic(backend_id: str, handle: CallHandle, model: str) -> dict[str, Any]:
    summary = handle.stream_summary
    completed = summary.completed_response if summary is not None else None
    usage_block = completed.get("usage") if isinstance(completed, dict) else None
    quota = handle.quota_after
    checks = {
        "http_2xx": handle.upstream_status is not None and 200 <= handle.upstream_status < 300,
        "quota_headers_present": quota is not None,
        "five_hourly_used_percent_present": quota is not None and quota.five_hourly_used_percent is not None,
        "weekly_used_percent_present": quota is not None and quota.weekly_used_percent is not None,
        "response_completed_event_present": completed is not None,
        "usage_block_present": isinstance(usage_block, dict),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "id": backend_id,
        "ok": not failed,
        "skipped": False,
        "stage": "evaluate",
        "model": model,
        "checks": checks,
        "failed_checks": failed,
        "upstream_status": handle.upstream_status,
    }


def _evaluate_generic_diagnostic(backend_id: str, handle: CallHandle, model: str) -> dict[str, Any]:
    """For non-Codex backends the contract is much weaker — only verify a 2xx
    came back. Stream-summary-style checks are Codex-specific (the LiteLLM
    gateway backend yields synthetic SSE events that don't flow through
    ResponsesStreamCollector, so requiring stream_summary here would be a
    false negative).
    """
    http_2xx = handle.upstream_status is not None and 200 <= handle.upstream_status < 300
    checks = {"http_2xx": http_2xx}
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "id": backend_id,
        "ok": not failed,
        "skipped": False,
        "stage": "evaluate",
        "model": model,
        "checks": checks,
        "failed_checks": failed,
        "upstream_status": handle.upstream_status,
    }


def _install_auth_routes(app: FastAPI, auth_service: AuthService) -> None:
    """Add /auth/* routes when an AuthService is configured.

    Endpoints:
    - POST /auth/register {username, password} -> {user_id, username}
    - POST /auth/login    {username, password} -> {session_token, expires_at}
    - POST /auth/logout                         -> 204 (session-bearer)
    - POST /auth/keys     {label?}             -> issued key (plaintext shown ONCE)
    - GET  /auth/keys                          -> list keys for the session's user
    - DELETE /auth/keys/{key_id}               -> revoke
    """

    def _require_session(request: Request) -> Session:
        plaintext = _bearer(request)
        if plaintext is None:
            raise HTTPException(status_code=401, detail="session token required")
        try:
            return auth_service.resolve_session(plaintext)
        except SessionInvalidError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @app.post("/auth/register", status_code=201)
    async def register(body: dict[str, Any]) -> dict[str, Any]:
        username = body.get("username")
        password = body.get("password")
        if not isinstance(username, str) or not isinstance(password, str):
            raise HTTPException(status_code=400, detail="'username' and 'password' must be strings")
        try:
            user = auth_service.register(username=username, password=password)
        except InvalidCredentialsError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"user_id": user.id, "username": user.username}

    @app.post("/auth/login")
    async def login(body: dict[str, Any]) -> dict[str, Any]:
        username = body.get("username")
        password = body.get("password")
        if not isinstance(username, str) or not isinstance(password, str):
            raise HTTPException(status_code=400, detail="'username' and 'password' must be strings")
        try:
            issued = auth_service.login(username=username, password=password)
        except InvalidCredentialsError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return {
            "session_token": issued.plaintext,
            "expires_at": issued.session.expires_at,
        }

    @app.post("/auth/logout", status_code=204)
    async def logout(request: Request) -> None:
        plaintext = _bearer(request)
        if plaintext is not None:
            auth_service.logout(plaintext)

    @app.post("/auth/keys", status_code=201)
    async def create_key(request: Request, body: dict[str, Any]) -> dict[str, Any]:
        session = _require_session(request)
        label = body.get("label") if isinstance(body.get("label"), str) else None
        issued = auth_service.create_api_key(user_id=session.user_id, label=label)
        return {
            "id": issued.api_key.id,
            "api_key": issued.plaintext,  # plaintext shown ONCE
            "prefix": issued.api_key.key_prefix,
            "label": issued.api_key.label,
            "created_at": issued.api_key.created_at,
        }

    @app.get("/auth/keys")
    async def list_keys(request: Request) -> dict[str, Any]:
        session = _require_session(request)
        keys = auth_service.list_api_keys(user_id=session.user_id)
        return {
            "keys": [
                {
                    "id": k.id,
                    "prefix": k.key_prefix,
                    "label": k.label,
                    "created_at": k.created_at,
                    "last_used_at": k.last_used_at,
                    "revoked_at": k.revoked_at,
                }
                for k in keys
            ]
        }

    @app.delete("/auth/keys/{key_id}")
    async def revoke_key(request: Request, key_id: int) -> dict[str, bool]:
        session = _require_session(request)
        revoked = auth_service.revoke_api_key(key_id=key_id, user_id=session.user_id)
        if not revoked:
            raise HTTPException(status_code=404, detail="key not found or already revoked")
        return {"revoked": True}


_STATIC_DIR = Path(__file__).parent / "static"


def _install_web_ui(app: FastAPI) -> None:
    """Serve a small browser UI at /ui/ for users who don't want to curl
    the /auth/* endpoints by hand. Single-page, vanilla HTML+JS, no build
    step. Calls the same /auth/* JSON endpoints the curl flow uses; the
    browser stores the session token in localStorage. Only mounted when
    auth is enabled — without auth there's nothing to register or log in
    against.
    """
    index = _STATIC_DIR / "index.html"

    @app.get("/ui", include_in_schema=False)
    async def ui_root_redirect() -> HTMLResponse:
        return HTMLResponse(index.read_text(encoding="utf-8"))

    @app.get("/ui/", include_in_schema=False)
    async def ui_root() -> HTMLResponse:
        return HTMLResponse(index.read_text(encoding="utf-8"))


def _terminal_http(exc: BackendError) -> HTTPException:
    status = exc.status_code or _EXHAUSTED_STATUS.get(exc.classification, 500)
    return HTTPException(status_code=status, detail=exc.message or exc.classification)


async def _collect_backend_status(
    backends_list: Sequence[Backend],
) -> list[dict[str, Any]]:
    """Snapshot each backend's diagnosis-relevant state for an error response.

    Built so a 503 detail can be self-explanatory: every downstream tool
    (Hermes, codex-cli, Cursor) prints the proxy's error verbatim, so the
    proxy is the only place that can pack this context in once.
    """
    out: list[dict[str, Any]] = []
    now = time.time()
    for b in backends_list:
        try:
            usage = await b.usage_snapshot()
        except Exception:
            usage = None
        try:
            quota = await b.quota_snapshot()
        except Exception:
            quota = None
        cd = getattr(usage, "cooldown_until_ts", None)
        cd_in_s = (cd - now) if cd else None
        out.append(
            {
                "id": b.id,
                "kind": getattr(b, "kind", "unknown"),
                "advertised_models": sorted(b.advertised_models),
                "cooldown_until_ts": cd,
                "cooldown_in_seconds": int(cd_in_s) if cd_in_s and cd_in_s > 0 else None,
                "weekly_exhausted": bool(getattr(usage, "weekly_exhausted", False)),
                "blocking_meters": (
                    list(
                        blocking_meters(
                            BackendSnapshot(
                                backend=b,
                                health=HealthStatus(available=True, reason="ok"),
                                usage=usage,
                                quota=quota,
                            )
                        )
                    )
                    if usage is not None
                    else []
                ),
                "constraining_meter": (
                    constraining_meter(
                        BackendSnapshot(
                            backend=b,
                            health=HealthStatus(available=True, reason="ok"),
                            usage=usage,
                            quota=quota,
                        )
                    )
                    if usage is not None
                    else None
                ),
                "five_hourly_used_percent": getattr(quota, "five_hourly_used_percent", None),
                "five_hourly_reset_after_seconds": getattr(quota, "five_hourly_reset_after_seconds", None),
                "weekly_used_percent": getattr(quota, "weekly_used_percent", None),
                "weekly_reset_after_seconds": getattr(quota, "weekly_reset_after_seconds", None),
            }
        )
    return out


def _no_viable(
    *,
    model: str,
    last_error: BackendError | None,
    excluded_backends: dict[str, BackendError] | None = None,
    fallback_executor: FallbackExecutor | None = None,
    recovery_ts: float | None = None,
    backend_status: list[dict[str, Any]] | None = None,
) -> HTTPException:
    """Log all failed backends and return appropriate error with Retry-After header.

    When `backend_status` is provided, the response body's `detail` becomes a
    structured dict including each backend's cooldown + quota state so
    downstream tools printing the error verbatim have enough information
    to diagnose without separately curling /status.
    """
    if excluded_backends:
        # Build error classification map for logging
        error_classifications = {backend_id: error.classification for backend_id, error in excluded_backends.items()}

        if fallback_executor:
            fallback_executor.log_final_exhaustion(model, error_classifications)
        else:
            # Fallback not attempted, log with recovery info if available
            failures = []
            for backend_id, error in excluded_backends.items():
                failures.append(f"{backend_id}: {error.classification}")
            ts = _utc_timestamp()
            msg = f"[{ts}] all backends exhausted for model {model!r}. Failures: {'; '.join(failures)}"
            if recovery_ts:
                recovery_dt = datetime.fromtimestamp(recovery_ts, tz=UTC)
                recovery_s = max(1, int(recovery_ts - time.time()))
                msg += f" | Earliest recovery: {recovery_dt.isoformat()} (in {recovery_s}s)"
            logger.warning(msg)

    def _build_detail(short: str) -> Any:
        if backend_status is None:
            return short
        summary = short
        if recovery_ts:
            recovery_s = max(1, int(recovery_ts - time.time()))
            summary += f" | earliest recovery in {recovery_s}s ({recovery_s / 60:.1f}min)"
        return {
            "error": short,
            "summary": summary,
            "model": model,
            "backends": backend_status,
            "recovery_in_seconds": (max(1, int(recovery_ts - time.time())) if recovery_ts else None),
        }

    if last_error is None:
        return HTTPException(
            status_code=503,
            detail=_build_detail(f"no viable backend for model {model!r}"),
        )
    status = _EXHAUSTED_STATUS.get(last_error.classification, 502)

    # Build response headers with Retry-After if available
    headers: dict[str, str] = {}
    if recovery_ts and status == 429:
        retry_after_s = max(1, int(recovery_ts - time.time()))
        headers["Retry-After"] = str(retry_after_s)
        recovery_dt = datetime.fromtimestamp(recovery_ts, tz=UTC)
        headers["X-Retry-After-UTC"] = recovery_dt.isoformat()

    return HTTPException(
        status_code=status,
        detail=_build_detail(last_error.message or last_error.classification),
        headers=headers or None,
    )
