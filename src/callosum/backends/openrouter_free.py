"""OpenRouter free-tier backend with auto-discovery and per-request model selection.

The user opts into this backend by setting OPENROUTER_API_KEY in env. The
backend then:

- Periodically fetches OpenRouter's /api/v1/models catalog
- Filters to free-tier models (prompt + completion pricing both 0)
- Exposes the resulting set as advertised_models
- For each call, picks the best free model that matches the request's
  needs (context length, tool support, code-friendliness)

This is the "snail's pace fallback" tier: when all Codex backends are
weekly-exhausted, the proxy auto-fails-over here so the user keeps working.

OpenRouter is OpenAI-compatible at /api/v1/chat/completions. It does NOT
support OpenAI's /v1/responses API; clients hitting /v1/responses will get
a translated chat-completions call out, and the response is translated back
into Responses-API shape.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, cast

import httpx

from callosum.backend import BackendKind, CallHandle, HealthStatus, UsageSnapshot
from callosum.backends._http import error_from_response
from callosum.errors import BackendError

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Refresh the catalog hourly. Free model lineup churns; this keeps us current
# without hammering OpenRouter.
DEFAULT_CATALOG_REFRESH_S = 3600.0

# Provider prefixes whose models are excluded from the free-pool catalog.
# Operator preference: no Chinese-origin models from cloud providers
# (OpenRouter is cloud). If a separate "local models" backend ships later,
# it can apply different rules — this exclusion is OpenRouter-specific.
_BLOCKED_PROVIDER_PREFIXES: frozenset[str] = frozenset(
    {
        "model-a0g3",  # Alibaba
        "alibaba",
        "model-a0e2",  # DeepSeek
        "01-ai",  # 01.AI (Yi)
        "baichuan-inc",
        "baidu",  # Baidu (Ernie/Qianfan)
        "thudm",  # Tsinghua/Zhipu (ChatGLM, GLM)
        "zhipu",
        "zhipuai",
        "z-ai",  # Zhipu's brand on OpenRouter (GLM-4.5 etc.)
        "zai",
        "internlm",
        "shanghai-ai-laboratory",
        "bytedance",  # Doubao
        "tencent",  # Hunyuan
        "minimax",
        "stepfun",
        "moonshot",  # Kimi
        "moonshotai",
        "yi",
        "inclusionai",  # Inclusion AI (Ling family, Chinese)
        "inclusion-ai",
        "01ai",
    }
)

# Code-keyword detection — model ids/names containing these substrings get
# a "this is a code-tuned model" bonus. Generic, applies to any provider.
_CODE_KEYWORDS: tuple[str, ...] = ("coder", "-code-", " code ", "starcoder", "codellama")


@dataclass(frozen=True, slots=True)
class FreeModel:
    id: str
    context_length: int
    supports_tools: bool
    supports_vision: bool
    code_score: int


class OpenRouterFreeBackend:
    """Auto-discovering free-tier OpenRouter backend.

    Construct one with an API key. On first call, fetches and caches the model
    catalog. Subsequent calls within DEFAULT_CATALOG_REFRESH_S use the cache.
    """

    kind: BackendKind = "openrouter_free"

    def __init__(
        self,
        *,
        id: str,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        catalog_refresh_s: float = DEFAULT_CATALOG_REFRESH_S,
        timeout_s: float = 60.0,
        shadow_models: frozenset[str] | Callable[[], frozenset[str]] | None = None,
    ) -> None:
        self.id = id
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._catalog_refresh_s = catalog_refresh_s
        # Models the OpenRouter backend "shadow-advertises" so it can serve
        # requests targeting other backends' model names. Used so that when
        # all Codex backends are in cooldown, requests for `model-a0e7` etc. are
        # still routable — OpenRouter takes them and substitutes a free model
        # at call time. The selector keeps picking Codex first whenever Codex
        # is viable, because OpenRouter's reported remaining_fraction is the
        # smallest possible (0.001).
        #
        # `shadow_models` can be a frozenset (static, evaluated once) or a
        # callable returning a frozenset (re-evaluated on every advertised_models
        # access). The callable form is what __main__.py passes so that
        # dynamic Codex model discovery propagates here without restart.
        self._shadow_models: frozenset[str] | Callable[[], frozenset[str]] = (
            shadow_models if shadow_models is not None else frozenset()
        )
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(transport=transport, timeout=timeout_s)
            self._owns_client = True
        # Catalog state.
        self._catalog: list[FreeModel] = []
        self._catalog_fetched_at: float = 0.0
        # remaining_fraction is set to a tiny non-zero value so the selector
        # always prefers a healthy Codex backend over OpenRouter, but falls
        # over here once Codex is in cooldown.
        self._usage = UsageSnapshot(
            remaining_fraction=0.001,
            cooldown_until_ts=None,
            weekly_exhausted=False,
            probed_at_ts=time.time(),
        )

    @property
    def advertised_models(self) -> frozenset[str]:
        """Union of (free catalog ids) ∪ (shadow_models, e.g. Codex names) ∪
        the virtual selector "auto-fallback". When shadow_models is a callable,
        it's invoked on every access so dynamic upstream catalog updates from
        sibling backends propagate immediately.
        """
        shadow = self._shadow_models() if callable(self._shadow_models) else self._shadow_models
        return frozenset(shadow | {m.id for m in self._catalog} | {"auto-fallback"})

    async def health(self) -> HealthStatus:
        return HealthStatus(available=True, reason="ok")

    async def usage_snapshot(self) -> UsageSnapshot:
        return self._usage

    async def quota_snapshot(self) -> Any:  # CodexQuotaSnapshot | None
        # OpenRouter has no Codex-shape quota headers.
        return None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def chat_completions(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> dict[str, Any]:
        await self._refresh_catalog_if_stale()
        chosen = self._pick_model(body)
        out_body = {**body, "model": chosen.id, "stream": False}
        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=out_body,
                headers=self._build_headers(),
            )
        except httpx.HTTPError as exc:
            raise BackendError(classification="transient", message=str(exc)) from exc
        if handle is not None:
            handle.upstream_status = response.status_code
            handle.upstream_headers = dict(response.headers)
        if response.status_code >= 400:
            raise error_from_response(response)
        return cast(dict[str, Any], response.json())

    async def chat_completions_stream(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> AsyncIterator[bytes]:
        await self._refresh_catalog_if_stale()
        chosen = self._pick_model(body)
        out_body = {**body, "model": chosen.id, "stream": True}
        try:
            stream_ctx = self._client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=out_body,
                headers=self._build_headers(),
            )
            async with stream_ctx as response:
                if handle is not None:
                    handle.upstream_status = response.status_code
                    handle.upstream_headers = dict(response.headers)
                if response.status_code >= 400:
                    await response.aread()
                    raise error_from_response(response)
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.HTTPError as exc:
            raise BackendError(classification="transient", message=str(exc)) from exc

    async def responses(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> dict[str, Any]:
        # OpenRouter doesn't support /v1/responses. Translate to chat-completions,
        # call, translate back to Responses-API shape.
        chat_body = _responses_to_chat_request(body)
        chat_response = await self.chat_completions(chat_body, handle)
        return _chat_to_responses_response(chat_response)

    async def responses_stream(
        self, body: dict[str, Any], handle: CallHandle | None = None
    ) -> AsyncIterator[bytes]:
        # Buffered translation: call non-stream, synthesize Responses-API SSE.
        full = await self.responses({**body, "stream": False}, handle)
        # Minimal SSE: response.created → response.completed with the full payload.
        created_event = {"type": "response.created", "id": full.get("id", "resp-or")}
        completed_event = {"type": "response.completed", "response": full}
        for ev in (created_event, completed_event):
            yield f"data: {json.dumps(ev)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    # --------- internals ----------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            # OpenRouter recommends these for traffic attribution; harmless if
            # missing on the server side.
            "HTTP-Referer": "https://github.com/anthropics/callosum",
            "X-Title": "callosum",
        }

    async def _refresh_catalog_if_stale(self, *, now: float | None = None) -> None:
        ts = now if now is not None else time.time()
        if self._catalog and ts - self._catalog_fetched_at < self._catalog_refresh_s:
            return
        try:
            response = await self._client.get(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.HTTPError:
            # If the catalog endpoint is down but we have a cached catalog,
            # keep using it. If we have nothing, advertised_models stays empty.
            return
        if response.status_code != 200:
            return
        try:
            payload = response.json()
        except json.JSONDecodeError:
            return
        models_raw = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models_raw, list):
            return
        self._catalog = [
            m for m in (_parse_model_entry(entry) for entry in models_raw) if m is not None
        ]
        self._catalog_fetched_at = ts

    def _pick_model(self, body: dict[str, Any]) -> FreeModel:
        """Score the catalog against the request. Highest score wins.

        Filters: model must support tools if the request has any. Model's
        context_length must comfortably exceed the request's input token
        budget (we use a heuristic — exact tokenization isn't worth the cost
        here).
        """
        if not self._catalog:
            raise BackendError(
                classification="transient",
                message=(
                    "OpenRouter free catalog is empty — either no free models "
                    "are currently available or the catalog has not been fetched yet"
                ),
            )
        wants_tools = bool(body.get("tools")) or bool(body.get("tool_choice"))
        approx_input_tokens = _approx_token_budget(body)
        candidates = [
            m
            for m in self._catalog
            if (not wants_tools or m.supports_tools) and m.context_length >= approx_input_tokens
        ]
        if not candidates:
            # Relax the context constraint — better to truncate than to refuse.
            candidates = [m for m in self._catalog if not wants_tools or m.supports_tools]
        if not candidates:
            # Tools required but no free model supports them. Fall back to the
            # raw catalog and hope the request works (it likely won't, but the
            # caller will see a clearer error).
            candidates = list(self._catalog)
        # Sort: code_score desc, context_length desc, supports_tools desc, id asc.
        candidates.sort(
            key=lambda m: (
                -m.code_score,
                -m.context_length,
                0 if m.supports_tools else 1,
                m.id,
            )
        )
        return candidates[0]


def _parse_model_entry(entry: Any, *, now_ts: float | None = None) -> FreeModel | None:
    if not isinstance(entry, dict):
        return None
    model_id = entry.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None
    if _is_blocked_provider(model_id):
        return None
    pricing = entry.get("pricing") if isinstance(entry.get("pricing"), dict) else {}
    if not _is_free(pricing):
        return None
    context_length = _coerce_int(entry.get("context_length"), default=4096)
    supports_tools = _supports_tools(entry)
    supports_vision = _supports_vision(entry)
    score = _score_model(entry, now_ts=now_ts if now_ts is not None else time.time())
    return FreeModel(
        id=model_id,
        context_length=context_length,
        supports_tools=supports_tools,
        supports_vision=supports_vision,
        code_score=score,
    )


def _is_blocked_provider(model_id: str) -> bool:
    """OpenRouter ids are formatted `provider/model:tag`. Reject if the provider
    prefix is on the blocklist (case-insensitive).
    """
    provider = model_id.split("/", 1)[0].lower() if "/" in model_id else model_id.lower()
    return provider in _BLOCKED_PROVIDER_PREFIXES


def _is_free(pricing: Any) -> bool:
    if not isinstance(pricing, dict):
        return False

    def _zero(v: Any) -> bool:
        try:
            return float(v) == 0.0
        except (TypeError, ValueError):
            return False

    # Both fields must be present AND zero. An empty pricing dict isn't "free" —
    # it's "we don't know," and we err toward conservative.
    if "prompt" not in pricing or "completion" not in pricing:
        return False
    return _zero(pricing["prompt"]) and _zero(pricing["completion"])


def _supports_tools(entry: dict[str, Any]) -> bool:
    # OpenRouter exposes tool support in different fields across model entries.
    # Check the most common spots.
    sup = entry.get("supported_parameters")
    if isinstance(sup, list) and any(s in sup for s in ("tools", "tool_choice")):
        return True
    arch = entry.get("architecture")
    if isinstance(arch, dict):
        modality = arch.get("modality")
        if isinstance(modality, str) and "tool" in modality:
            return True
    return False


def _supports_vision(entry: dict[str, Any]) -> bool:
    arch = entry.get("architecture")
    if isinstance(arch, dict):
        modality = arch.get("modality")
        if isinstance(modality, str) and "image" in modality:
            return True
        input_modalities = arch.get("input_modalities")
        if isinstance(input_modalities, list) and "image" in input_modalities:
            return True
    return False


def _coerce_int(v: Any, *, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _score_model(entry: dict[str, Any], *, now_ts: float) -> int:
    """Score a catalog entry by features extracted from its metadata, NOT by
    matching specific model family names. New families released to the free
    tier get scored on their own merits the moment they appear in the catalog.

    Components (all derived from per-entry data, never from a hardcoded family
    list except the generic "code"-keyword bonus):

    - Context length: log-scale bonus (more context → bigger window for
      reading codebases).
    - Parameter count parsed from the model id (e.g. `model-a0c2` → 70B):
      log-scale, bigger = better with diminishing returns.
    - Instruct-tuned bonus: presence of `architecture.instruct_type` indicates
      the model is chat/instruction-tuned, not a base model. Base models are
      poor for code workflows.
    - Tool support: presence of `tools` / `tool_choice` in
      `supported_parameters` (matters for agentic workflows).
    - Recency: newer `created` timestamp → higher score, decaying over months.
    - Generic code-keyword bonus: model id/name contains `coder`/`code`/etc.
    """
    score = 0

    # Context length, log-scale, capped at +40.
    ctx = _coerce_int(entry.get("context_length"), default=4096)
    if ctx > 4096:
        score += min(40, max(0, int(math.log2(ctx / 4096) * 10)))

    # Parameter count, log-scale, capped at +40.
    param_count_b = _parse_param_count_billions(str(entry.get("id", "")))
    if param_count_b > 0:
        # log2(70B) ≈ 6.13, *5 = ~30; log2(405B) ≈ 8.66, *5 = ~43 → capped at 40.
        score += min(40, max(0, int(math.log2(param_count_b) * 5)))

    # Instruct-tuned bonus.
    arch = entry.get("architecture")
    if isinstance(arch, dict) and arch.get("instruct_type"):
        score += 20

    # Tool support.
    sup = entry.get("supported_parameters")
    if isinstance(sup, list) and any(s in sup for s in ("tools", "tool_choice")):
        score += 20

    # Recency: 30 points if released in the last 30 days, decaying linearly to
    # 0 at 360 days. Older models still useful but not preferred.
    created = entry.get("created")
    if isinstance(created, (int, float)) and created > 0:
        days_old = max(0.0, (now_ts - float(created)) / 86400.0)
        score += max(0, 30 - int(days_old / 12))

    # Generic code-keyword bonus (substring match on id and name).
    name_lower = (str(entry.get("id", "")) + " " + str(entry.get("name", ""))).lower()
    if any(kw in name_lower for kw in _CODE_KEYWORDS):
        score += 30

    return score


_PARAM_COUNT_RE = re.compile(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)


def _parse_param_count_billions(model_id: str) -> float:
    """Extract parameter count in billions from a model id, e.g.
    `meta-model-a0g1/model-a0a5:free` → 70.0,
    `mistralai/mistral-7b-instruct:free`     → 7.0,
    `nousresearch/hermes-3:free`             → 0.0 (no clear count).
    """
    matches = _PARAM_COUNT_RE.findall(model_id)
    if not matches:
        return 0.0
    # If multiple matches (e.g. "model-a0c2" has both "3" not matched and
    # "70" matched), take the largest — biggest plausible param count.
    try:
        return max(float(m) for m in matches)
    except ValueError:
        return 0.0


def _approx_token_budget(body: dict[str, Any]) -> int:
    """Rough estimate of input token count. Used only for catalog filtering;
    if we're off by 2x it just biases toward larger-context models which is
    fine.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return 4096
    total_chars = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        total_chars += len(text)
    # ~4 chars/token for English; double for code/JSON noise; round up.
    return max(1024, total_chars // 2)


# --------- Responses-API ↔ chat-completions translation ---------------------


def _responses_to_chat_request(body: dict[str, Any]) -> dict[str, Any]:
    """Minimal translation of /v1/responses request body to /v1/chat/completions
    shape. Sufficient for the simple "give me text back" path; advanced
    Responses-API features (file inputs, structured outputs, etc.) won't
    survive this translation.
    """
    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    input_block = body.get("input")
    if isinstance(input_block, str):
        messages.append({"role": "user", "content": input_block})
    elif isinstance(input_block, list):
        for item in input_block:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            role = item.get("role", "user")
            content = item.get("content")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and isinstance(p.get("text"), str)
                ]
                text = "".join(parts)
            messages.append({"role": role, "content": text})
    if not messages:
        messages.append({"role": "user", "content": ""})
    chat_body: dict[str, Any] = {
        "model": body.get("model", ""),
        "messages": messages,
    }
    # Pass through anything else that's chat-completions-shaped.
    for key in ("temperature", "max_tokens", "top_p", "tools", "tool_choice"):
        if key in body:
            chat_body[key] = body[key]
    return chat_body


def _chat_to_responses_response(chat: dict[str, Any]) -> dict[str, Any]:
    """Inverse of _responses_to_chat_request, on the response side."""
    choices = chat.get("choices")
    if not isinstance(choices, list) or not choices:
        text = ""
    else:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        raw = message.get("content") if isinstance(message, dict) else None
        text = raw if isinstance(raw, str) else ""
    return {
        "id": chat.get("id", "resp-or"),
        "object": "response",
        "model": chat.get("model", ""),
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": chat.get("usage"),
    }
