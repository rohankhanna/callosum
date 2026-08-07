"""Shared Responses-API ↔ Chat-Completions translation for chat-shaped backends.

Backends whose upstream speaks Chat Completions (the LiteLLM gateway, the
local ollama daemon serving Ollama Cloud models) must translate Codex CLI's
Responses-API requests to Chat shape and back. The pure translators
(`_strip_codex_only_fields`, `_responses_to_chat_request`,
`_chat_to_responses_response`, and helpers) and the chat→Responses *streaming*
generator (`chat_to_responses_stream`) live here so every chat-shaped backend
shares ONE copy. Duplication would leave two subtly-divergent streaming state
machines that must stay in sync as the Responses API evolves — see the
"reuse existing code before writing new" operator instruction.

This is the refactor `litellm_gateway.py` anticipated when it was the only
chat-shaped backend: those translators were originally inlined there with a
note that they'd move to a shared module once a third backend needed them.
The local ollama daemon (Ollama Cloud) is that backend.

Module-private (`_`-prefixed) translator names are preserved verbatim so
existing imports from `callosum.backends.litellm_gateway` (which now re-imports
them from here) and `callosum.backends.local_direct` keep working without a
rename sweep.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager as AsyncContextManager
from typing import Any

import httpx

from callosum.backend import CallHandle
from callosum.backends._http import error_from_response, stall_guarded
from callosum.errors import BackendError

# ---------- body hygiene ------------------------------------------------
# Local/cloud backends (ollama / vLLM / model-a0e0 behind LiteLLM, or the
# ollama daemon serving cloud models) don't honor every Chat-Completions
# request field OpenAI advertises. LiteLLM in `drop_params: false` mode (the
# default in local LLM gateway's config) errors hard instead of silently
# dropping — `litellm.UnsupportedParamsError` bubbles up as a 400 to the
# caller. The ollama daemon likewise rejects unknown fields. Strip the fields
# known to trigger this BEFORE sending. Currently includes:
#   * `reasoning`: Codex-specific routing hint; no chat-shaped backend uses it.
#   * `parallel_tool_calls`: standard Chat-Completions field (controls
#     whether the model emits multiple tool calls in one turn) but ollama
#     does not implement it. The model defaults to its native behavior
#     either way; dropping the flag is harmless because no chat-shaped
#     backend can honor it. Add new keys as more upstream incompatibilities
#     surface in real traffic.
_CODEX_ONLY_BODY_KEYS = ("reasoning", "parallel_tool_calls")


def _strip_codex_only_fields(body: dict[str, Any]) -> dict[str, Any]:
    """Drop request keys chat-shaped backends refuse from a body destined for
    a chat-shaped backend.

    Pure-fn returns a new dict; the caller's `body` is untouched. The
    name is historical — the strip list started as Codex-only fields but
    has grown to cover any field that OpenAI Chat Completions advertises
    but chat-shaped servers reject. See _CODEX_ONLY_BODY_KEYS
    docstring for the current contents and why each is dropped.
    """
    if not any(k in body for k in _CODEX_ONLY_BODY_KEYS):
        return body
    return {k: v for k, v in body.items() if k not in _CODEX_ONLY_BODY_KEYS}


# ---------- /v1/responses ↔ /v1/chat/completions translation -------------
# Shared by every chat-shaped backend (litellm_gateway, ollama_cloud) so the
# streaming translation state machine stays in one place.


# Keys that are part of the Responses-API request envelope and have no
# meaning for /v1/chat/completions. Stripped during translation so the
# Chat backend doesn't reject the body or silently ignore them. Everything
# else (stream, stream_options, temperature, tools, tool_choice, max_tokens,
# top_p, parallel_tool_calls, n, seed, response_format, …) is preserved
# verbatim — keeps the translator small as new request fields are added
# upstream.
_RESPONSES_ONLY_KEYS: frozenset[str] = frozenset(
    {
        "input",
        "instructions",
        "store",
        "include",
        "prompt_cache_key",
        "client_metadata",
        "text",  # Responses-API response_format equivalent; not the same field
        "reasoning",  # Codex-only request hint; redundant since _strip_codex_only_fields
    }
)


def _extract_reasoning_text(payload: dict[str, Any]) -> str:
    """Return the first non-empty reasoning alias from a chat payload."""
    for key in ("thinking", "reasoning_content", "reasoning"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _extract_text_from_content(content: Any) -> str:
    """Pull all text from a Responses-API `content` field.

    The Responses API allows content to be either a plain string OR a list
    of part dicts like `{"type": "input_text", "text": "..."}`. Both shapes
    appear in Codex CLI traffic. Return the concatenation of every text-
    bearing part; non-text parts (images, audio) are silently skipped here
    because the chat-completions translator can't represent them in a
    text-only role.content slot.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "".join(parts)
    return ""


def _responses_to_chat_request(body: dict[str, Any]) -> dict[str, Any]:
    """Translate a Codex /v1/responses request body to /v1/chat/completions shape.

    Handles the item types Codex CLI actually sends:

      * `message` (user / assistant / developer / system): copied to a
        chat message with the same role + extracted text.
      * `function_call`: collected into a pending tool_calls list; consecutive
        function_calls collapse into ONE assistant message with multiple
        tool_calls (matches OpenAI's parallel-tool-call shape so downstream
        models that paid attention to its training know what to do).
      * `function_call_output`: flushes the pending tool_calls group, then
        emits a `role: tool` message with `tool_call_id` matching the call.
      * `custom_tool_call` / `custom_tool_call_output`: same as function_*,
        treated identically since the on-wire shape is the same.
      * `reasoning`: DROPPED. The encrypted_content blobs are OpenAI-server-
        side state with no chat-completions representation. The model loses
        its prior internal CoT but the visible assistant messages still
        carry the conclusions, which is what matters for continuation.
      * `compaction`: preserved as a system note so the model knows prior
        turns were summarized away.

    Preserves every other top-level key from the input body (stream,
    stream_options, temperature, tools, tool_choice, max_tokens, top_p,
    parallel_tool_calls, etc.) so future Responses-API fields that ALSO
    apply to chat-completions don't need a translator change.
    """
    # Start with every non-Responses-only key carried over unchanged. This
    # is the opposite of a key whitelist — we explicitly know what to
    # drop, and pass through everything else. Reduces translator churn as
    # the upstream API grows.
    chat_body: dict[str, Any] = {k: v for k, v in body.items() if k not in _RESPONSES_ONLY_KEYS}
    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    input_block = body.get("input")
    if isinstance(input_block, str):
        messages.append({"role": "user", "content": input_block})
    elif isinstance(input_block, list):
        # Pending group of consecutive function_call items, materialized as
        # a single assistant message with multiple tool_calls when something
        # non-function_call interrupts the run.
        pending_tool_calls: list[dict[str, Any]] = []

        def _flush_pending() -> None:
            if pending_tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": list(pending_tool_calls),
                    }
                )
                pending_tool_calls.clear()

        for item in input_block:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "message":
                _flush_pending()
                role = item.get("role", "user")
                text = _extract_text_from_content(item.get("content"))
                # Always emit even if text is empty — preserves turn
                # structure for models that expect alternation.
                messages.append({"role": role, "content": text})
            elif t in ("function_call", "custom_tool_call"):
                call_id = item.get("call_id") or item.get("id", "")
                args = item.get("arguments")
                if args is None:
                    # custom_tool_call uses `input` instead of `arguments`.
                    args = item.get("input", "")
                if isinstance(args, (dict, list)):
                    args = json.dumps(args)
                elif not isinstance(args, str):
                    args = "{}"
                pending_tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": args,
                        },
                    }
                )
            elif t in ("function_call_output", "custom_tool_call_output"):
                _flush_pending()
                output = item.get("output", "")
                if isinstance(output, (dict, list)):
                    output = json.dumps(output)
                elif not isinstance(output, str):
                    output = str(output)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": output,
                    }
                )
            elif t == "reasoning":
                # Encrypted CoT from prior OpenAI turns; no chat-completions
                # equivalent. Dropping it is correct, not lossy in the sense
                # that matters — visible assistant messages already encode
                # the externally-stated conclusions of whatever the prior
                # reasoning produced.
                continue
            elif t == "compaction":
                summary = _extract_text_from_content(item.get("content")) or (
                    item.get("summary") if isinstance(item.get("summary"), str) else ""
                )
                if summary:
                    _flush_pending()
                    messages.append(
                        {
                            "role": "system",
                            "content": f"[compacted prior turns: {summary}]",
                        }
                    )
            # Unknown types are intentionally dropped. Add a branch here
            # the first time a new type surfaces in real traffic.
        _flush_pending()
    if not messages:
        messages.append({"role": "user", "content": ""})
    chat_body["model"] = body.get("model", "")
    chat_body["messages"] = messages
    return chat_body


def _chat_to_responses_response(chat: dict[str, Any]) -> dict[str, Any]:
    """Inverse of _responses_to_chat_request, on the response side.

    Translates BOTH content text AND tool_calls. Earlier versions only
    extracted message.content and dropped tool_calls on the floor —
    which manifested as Codex CLI receiving a response.completed event
    with empty output and silently displaying nothing. Tool-use prompts
    (Codex sends a tools array on every request, then the model picks
    a tool) require the function_call items in output[].
    """
    choices = chat.get("choices")
    text = ""
    thinking = ""
    tool_calls: list[dict[str, Any]] = []
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        if isinstance(message, dict):
            raw_content = message.get("content")
            if isinstance(raw_content, str):
                text = raw_content
            # Some local models emit chain-of-thought in a separate chat
            # field. Preserve the common OpenAI-compatible aliases as one
            # reasoning output item so the translator never drops it.
            thinking = _extract_reasoning_text(message)
            raw_tool_calls = message.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                for tc in raw_tool_calls:
                    if isinstance(tc, dict):
                        tool_calls.append(tc)
    # Build the Responses-API output list. Reasoning (chain-of-thought
    # from thinking-mode models) comes first if present — matches
    # OpenAI's o1/o3 reasoning-summary convention. function_call items
    # come next so Codex CLI executes them in order. A message item
    # with the assistant's visible text follows. When none of these
    # are present, emit an empty message so output[] is never empty.
    output: list[dict[str, Any]] = []
    if thinking:
        output.append(
            {
                "type": "reasoning",
                "id": f"rs_{chat.get('id', 'reasoning')}",
                "summary": [{"type": "summary_text", "text": thinking}],
            }
        )
    for tc in tool_calls:
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        if not isinstance(fn, dict):
            fn = {}
        # Arguments arrive as a JSON string from ollama-tool-calling
        # convention; some serving stacks return them as dicts instead.
        # The Responses-API spec wants a string.
        args = fn.get("arguments", "{}")
        if isinstance(args, (dict, list)):
            args = json.dumps(args)
        elif not isinstance(args, str):
            args = "{}"
        call_id = tc.get("id") or fn.get("name", "") or "fc-unknown"
        output.append(
            {
                "type": "function_call",
                "id": f"fc_{call_id}",
                "call_id": call_id,
                "name": fn.get("name", ""),
                "arguments": args,
                "status": "completed",
            }
        )
    if text or not tool_calls:
        # Emit the message even when empty if there were no tool calls,
        # so output[] is never an empty list (Codex parsers vary on
        # how strictly they require at least one item).
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        )
    # Translate usage from chat-completions shape (prompt_tokens /
    # completion_tokens) to Responses-API shape (input_tokens /
    # output_tokens). Codex CLI's stream parser hard-fails with
    # "missing field 'input_tokens'" when the response.completed event
    # lacks it, so we ALWAYS emit at least zeros — even when ollama
    # omits usage from its chat-completions reply.
    _maybe_usage = chat.get("usage")
    chat_usage: dict[str, Any] = _maybe_usage if isinstance(_maybe_usage, dict) else {}
    usage: dict[str, Any] = {
        "input_tokens": int(chat_usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(chat_usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(chat_usage.get("total_tokens", 0) or 0),
    }
    # Preserve any cache / reasoning-token sub-fields the upstream
    # included — they're optional in the Responses API but if present
    # they help downstream cost accounting.
    prompt_details = chat_usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        usage["input_tokens_details"] = prompt_details
    completion_details = chat_usage.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        usage["output_tokens_details"] = completion_details
    return {
        "id": chat.get("id", "resp-litellm"),
        "object": "response",
        "model": chat.get("model", ""),
        "status": "completed",
        "output": output,
        "usage": usage,
    }


# ---------- chat → Responses streaming generator --------------------------


async def chat_to_responses_stream(
    *,
    client: httpx.AsyncClient,
    chat_url: str,
    body: dict[str, Any],
    handle: CallHandle | None,
    prep_body: Any,
    headers: dict[str, str],
    first_item_timeout_s: float,
    idle_timeout_s: float,
    what_label: str,
    on_success: Any,
    on_transport_error: Any,
    open_chat_stream: Callable[[dict[str, Any], dict[str, str]], AsyncContextManager[httpx.Response]] | None = None,
    upstream_status_of: Callable[[httpx.Response], int] | None = None,
) -> AsyncIterator[bytes]:
    """Translate a streamed Chat-Completions response into Responses-API SSE.

    Shared by every chat-shaped backend. The backend injects its own
    coupling points via the keyword-only params:

      * `client` / `chat_url` / `headers` — the upstream httpx client, the
        `/v1/chat/completions` URL, and the request headers (auth differs:
        litellm sends a master-key bearer; ollama_cloud sends none — the
        daemon holds the cloud auth).
      * `prep_body(body)` — returns the Chat body with `stream:True`, codex-only
        fields stripped, and (for litellm) inference params applied. Pure hook
        so the generator doesn't know about per-backend operator overrides.
      * `first_item_timeout_s` / `idle_timeout_s` / `what_label` — stall-guard
        tunables (the "LOCAL" in the constant names is historical; the guard
        applies to any streamed backend).
      * `on_success()` / `on_transport_error()` — health hooks called on
        clean completion vs an httpx transport error (before re-raising).
      * `open_chat_stream(out_body, headers)` / `upstream_status_of(response)`
        — optional (default `None`) hooks that replace the hardcoded
        direct-POST open + the bare `response.status_code` read. A backend
        that routes its upstream through a credential proxy (ollama_cloud via
        credential proxy) passes both so the generator opens the proxy stream instead
        of a direct upstream POST and classifies the *upstream* status from a
        proxy header rather than the proxy's own HTTP status. `None` reproduces
        the original `client.stream("POST", chat_url, json=out_body, headers=headers)` +
        `response.status_code` exactly (litellm local lane is byte-identical).

    `body` is the ORIGINAL Responses-API request body; the generator uses
    `body["model"]` (the cell id, before any runtime rewrite) to load the
    in-band-reasoning splitter, and `prep_body(body)` to build the Chat body
    actually POSTed. See `litellm_gateway.responses_stream` for the canonical
    call site + the `ResponsesStreamCollector` wrapper that populates
    `handle.stream_summary` for token accounting.
    """
    out_body = prep_body(body)
    # If the body arrived in Responses-API shape (has `input` instead
    # of `messages`), translate to Chat Completions shape before
    # forwarding. This is the streaming counterpart of what `responses`
    # already does on the non-stream path. Without this, the upstream
    # throws TypeError("missing required argument: 'messages'")
    # against any bare ollama / vLLM endpoint that doesn't have a
    # Responses-API translation wrapper running in front. Idempotent:
    # if the body is already in Chat shape, this branch is skipped.
    if "input" in out_body and "messages" not in out_body:
        out_body = _responses_to_chat_request(out_body)
    # OpenAI Chat Completions streaming omits the `usage` block
    # unless include_usage is explicitly requested. Without it, no
    # SSE chunk carries token counts, the terminal response.completed
    # event ships zeros, and downstream token accounting is NULL for
    # every streamed chat-shaped request. ollama / LiteLLM / vLLM
    # all honor this flag in their OpenAI-compatible mode. We set
    # it unconditionally because every streamed chat completion
    # benefits from it; if the operator already set it in the body,
    # ours is a no-op (same value).
    stream_options = out_body.get("stream_options")
    if not isinstance(stream_options, dict):
        stream_options = {}
    stream_options["include_usage"] = True
    out_body["stream_options"] = stream_options
    # In-band reasoning re-routing. callosum's transform framework does
    # not process streaming responses, and this generator is the sole
    # place callosum owns the chat→Responses *stream* translation (it is
    # never a byte-passed Responses stream), so the in-band splitter
    # hooks here. The gate loads the cell's capability profile and
    # returns a splitter only for `inband_tags` cells; for every other
    # cell (native/none/unknown/unprobed) it returns None and the
    # content path below is byte-for-byte unchanged. `body["model"]` is
    # the cell id the profile is keyed by (before any runtime rewrite).
    from callosum.transforms.inband_reasoning import inband_splitter_for_model

    reasoning_splitter = inband_splitter_for_model(str(body.get("model", "")))
    seq = 0

    def _emit(event_type: str, payload: dict[str, Any]) -> bytes:
        nonlocal seq
        seq += 1
        ev = {"type": event_type, "sequence_number": seq, **payload}
        return f"event: {event_type}\ndata: {json.dumps(ev)}\n\n".encode()

    # Per-stream accumulation state. Indexed by "output_index" in the
    # Responses-API sense; each output item gets a stable index from
    # the order it FIRST appeared in the stream.
    text_so_far = ""
    thinking_so_far = ""
    tool_calls_state: dict[int, dict[str, Any]] = {}  # idx → {id, name, args, output_index}
    output_items: list[dict[str, Any]] = []  # final assembled output for response.completed
    next_output_index = 0
    reasoning_output_index: int | None = None
    message_output_index: int | None = None
    message_item_id: str | None = None
    reasoning_item_id: str | None = None
    resp_id = "resp-litellm"
    upstream_model = out_body.get("model", "")
    usage: dict[str, Any] | None = None

    def _reasoning_events(text: str) -> list[bytes]:
        """Emit a reasoning-summary delta, lazily opening the reasoning
        output item on first use. Shared by the native `thinking`
        channel and the in-band splitter so both feed one reasoning
        item."""
        nonlocal reasoning_output_index, reasoning_item_id, next_output_index, thinking_so_far
        evs: list[bytes] = []
        if reasoning_output_index is None:
            reasoning_output_index = next_output_index
            next_output_index += 1
            reasoning_item_id = f"rs_{resp_id}_{reasoning_output_index}"
            evs.append(
                _emit(
                    "response.output_item.added",
                    {
                        "output_index": reasoning_output_index,
                        "item": {
                            "type": "reasoning",
                            "id": reasoning_item_id,
                            "summary": [],
                        },
                    },
                )
            )
        thinking_so_far += text
        evs.append(
            _emit(
                "response.reasoning_summary_text.delta",
                {
                    "item_id": reasoning_item_id,
                    "output_index": reasoning_output_index,
                    "summary_index": 0,
                    "delta": text,
                },
            )
        )
        return evs

    def _content_events(text: str) -> list[bytes]:
        """Emit a visible output_text delta, lazily opening the message
        output item on first use."""
        nonlocal message_output_index, message_item_id, next_output_index, text_so_far
        evs: list[bytes] = []
        if message_output_index is None:
            message_output_index = next_output_index
            next_output_index += 1
            message_item_id = f"msg_{resp_id}_{message_output_index}"
            evs.append(
                _emit(
                    "response.output_item.added",
                    {
                        "output_index": message_output_index,
                        "item": {
                            "type": "message",
                            "id": message_item_id,
                            "role": "assistant",
                            "content": [],
                            "status": "in_progress",
                        },
                    },
                )
            )
        text_so_far += text
        evs.append(
            _emit(
                "response.output_text.delta",
                {
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "delta": text,
                },
            )
        )
        return evs

    def _route_content(text: str) -> list[bytes]:
        """Route a content delta to the right channel. With no in-band
        splitter active this is just `_content_events(text)`. With one
        active, each segment the splitter yields goes to the reasoning
        or content channel; partial tags are held back across deltas."""
        if reasoning_splitter is None:
            return _content_events(text)
        evs: list[bytes] = []
        for channel, segment in reasoning_splitter.push(text):
            if channel == "reasoning":
                evs.extend(_reasoning_events(segment))
            else:
                evs.extend(_content_events(segment))
        return evs

    try:
        if open_chat_stream is not None:
            # Proxy-custody path (ollama_cloud via credential proxy): the backend opens
            # the upstream through a credential proxy, so the POST target,
            # auth, and body envelope are backend-controlled. The generator
            # stays shape-agnostic — it still parses `data:` SSE lines from
            # whatever the proxy forwards, regardless of media type.
            stream_ctx = open_chat_stream(out_body, headers)
        else:
            stream_ctx = client.stream(
                "POST",
                chat_url,
                json=out_body,
                headers=headers,
            )
        async with stream_ctx as response:
            upstream_status = upstream_status_of(response) if upstream_status_of is not None else response.status_code
            if handle is not None:
                handle.upstream_status = upstream_status
                handle.upstream_headers = dict(response.headers)
            if upstream_status >= 400:
                await response.aread()
                raise error_from_response(response, status_code=upstream_status)
            # Emit response.created early so clients show progress.
            base_response = {
                "id": resp_id,
                "object": "response",
                "model": upstream_model,
                "status": "in_progress",
                "output": [],
            }
            yield _emit("response.created", {"response": base_response})
            yield _emit("response.in_progress", {"response": base_response})

            async for line in stall_guarded(
                response.aiter_lines(),
                first_item_timeout_s=first_item_timeout_s,
                idle_timeout_s=idle_timeout_s,
                what=f"{what_label} {upstream_model}",
                handle=handle,
            ):
                if not line:
                    continue
                if line.startswith("data:"):
                    payload_str = line[5:].strip()
                else:
                    continue
                if payload_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk.get("id"), str) and chunk["id"]:
                    resp_id = chunk["id"]
                if isinstance(chunk.get("model"), str) and chunk["model"]:
                    upstream_model = chunk["model"]
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    delta = {}

                # Reasoning deltas (model-a0d5/model-a0g2/r1 and other
                # OpenAI-compatible local servers) arriving in a
                # native field.
                thinking_delta = _extract_reasoning_text(delta)
                if isinstance(thinking_delta, str) and thinking_delta:
                    for ev in _reasoning_events(thinking_delta):
                        yield ev

                # Tool-call deltas.
                tool_calls_delta = delta.get("tool_calls") or []
                if isinstance(tool_calls_delta, list):
                    for tc_delta in tool_calls_delta:
                        if not isinstance(tc_delta, dict):
                            continue
                        tc_idx = tc_delta.get("index", 0)
                        if not isinstance(tc_idx, int):
                            tc_idx = 0
                        state = tool_calls_state.get(tc_idx)
                        if state is None:
                            # First time we see this tool call — emit added.
                            call_id = tc_delta.get("id") or f"fc_{resp_id}_{tc_idx}"
                            fn = tc_delta.get("function") or {}
                            name = fn.get("name", "") if isinstance(fn, dict) else ""
                            out_idx = next_output_index
                            next_output_index += 1
                            state = {
                                "id": call_id,
                                "name": name,
                                "args": "",
                                "output_index": out_idx,
                                "item_id": f"fc_{call_id}",
                            }
                            tool_calls_state[tc_idx] = state
                            yield _emit(
                                "response.output_item.added",
                                {
                                    "output_index": out_idx,
                                    "item": {
                                        "type": "function_call",
                                        "id": state["item_id"],
                                        "call_id": call_id,
                                        "name": name,
                                        "arguments": "",
                                        "status": "in_progress",
                                    },
                                },
                            )
                        else:
                            fn = tc_delta.get("function") or {}
                            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                                state["name"] = fn["name"] or state["name"]
                        fn = tc_delta.get("function") or {}
                        args_delta = fn.get("arguments") if isinstance(fn, dict) else None
                        if isinstance(args_delta, str) and args_delta:
                            state["args"] += args_delta
                            yield _emit(
                                "response.function_call_arguments.delta",
                                {
                                    "item_id": state["item_id"],
                                    "output_index": state["output_index"],
                                    "delta": args_delta,
                                },
                            )

                # Message text delta. Routed through the in-band splitter
                # when the cell is an `inband_tags` cell (else verbatim).
                content_delta = delta.get("content")
                if isinstance(content_delta, str) and content_delta:
                    for ev in _route_content(content_delta):
                        yield ev

        # Stream ended cleanly. Release any tag-fragment text the
        # in-band splitter buffered at end-of-stream onto its active
        # channel before the per-item .done events.
        if reasoning_splitter is not None:
            for channel, segment in reasoning_splitter.flush():
                events = _reasoning_events(segment) if channel == "reasoning" else _content_events(segment)
                for ev in events:
                    yield ev
        # Emit per-item .done events in output order, then
        # response.output_item.done, then response.completed.
        if reasoning_output_index is not None:
            yield _emit(
                "response.reasoning_summary_text.done",
                {
                    "item_id": reasoning_item_id,
                    "output_index": reasoning_output_index,
                    "summary_index": 0,
                    "text": thinking_so_far,
                },
            )
            reasoning_item = {
                "type": "reasoning",
                "id": reasoning_item_id,
                "summary": [{"type": "summary_text", "text": thinking_so_far}],
            }
            output_items.append(reasoning_item)
            yield _emit(
                "response.output_item.done",
                {
                    "output_index": reasoning_output_index,
                    "item": reasoning_item,
                },
            )

        for tc_idx in sorted(tool_calls_state):
            state = tool_calls_state[tc_idx]
            yield _emit(
                "response.function_call_arguments.done",
                {
                    "item_id": state["item_id"],
                    "output_index": state["output_index"],
                    "arguments": state["args"],
                },
            )
            fn_item = {
                "type": "function_call",
                "id": state["item_id"],
                "call_id": state["id"],
                "name": state["name"],
                "arguments": state["args"],
                "status": "completed",
            }
            output_items.append(fn_item)
            yield _emit(
                "response.output_item.done",
                {
                    "output_index": state["output_index"],
                    "item": fn_item,
                },
            )

        if message_output_index is not None:
            yield _emit(
                "response.output_text.done",
                {
                    "item_id": message_item_id,
                    "output_index": message_output_index,
                    "content_index": 0,
                    "text": text_so_far,
                },
            )
            msg_item = {
                "type": "message",
                "id": message_item_id,
                "role": "assistant",
                "content": [{"type": "output_text", "text": text_so_far}],
                "status": "completed",
            }
            output_items.append(msg_item)
            yield _emit(
                "response.output_item.done",
                {
                    "output_index": message_output_index,
                    "item": msg_item,
                },
            )

        # Codex CLI requires input_tokens; map from chat usage shape.
        chat_usage = usage if isinstance(usage, dict) else {}
        usage_out = {
            "input_tokens": int(chat_usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(chat_usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(chat_usage.get("total_tokens", 0) or 0),
        }
        full_response = {
            "id": resp_id,
            "object": "response",
            "model": upstream_model,
            "status": "completed",
            "output": output_items,
            "usage": usage_out,
        }
        yield _emit("response.completed", {"response": full_response})
        yield b"data: [DONE]\n\n"
        # Success — notify the backend (clears any offline tracker).
        on_success()
    except httpx.HTTPError as exc:
        # Transport-level error during the stream — let the backend mark
        # itself unhealthy so the offline detector kicks in, then surface
        # a transient BackendError.
        on_transport_error()
        raise BackendError(classification="transient", message=str(exc)) from exc
