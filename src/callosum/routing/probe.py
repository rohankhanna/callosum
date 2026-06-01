"""Tool-call verification probe.

A small, focused capability check: does this specific cell actually
emit OpenAI-shaped `tool_calls` when asked to, end-to-end through
callosum's pipeline?

Why this exists despite ollama already exposing a `tools` capability
flag via /api/show: ollama's flag means "the model's template supports
tool prompting" — not "the model emits responses that parse to
structured tool_calls when callosum's gateway translator processes
them." The first time we wired a bare ollama cell into callosum,
model-a0c7 advertised tools-capable AND served requests, but emitted
the tool call as JSON text inside `message.content` instead of as a
structured `tool_calls` field. The router happily kept sending it
tool-using traffic; Codex CLI happily rendered the JSON-as-text as
visible junk to the user. The static capability flag lied.

The probe closes that loop. Given a function that calls
`backend.responses(body)` for a cell, it constructs a minimal
tool-using request, runs it, and parses the response for structured
function_call items. Returns True only when the cell's actual
end-to-end behavior matches the OpenAI tool-call contract.

Used by the router (or an operator's `callosum-ctl probe-all` command)
to override `supports_tools` flags coming from upstream catalogs when
they don't match the cell's observed runtime behavior. Defense in
depth against per-model integration breaks.

Out of scope: streaming (non-stream `responses` is sufficient to
verify the format); modalities other than text (vision-probing is a
follow-up); quality (whether the model picked the RIGHT tool or
passed the RIGHT arguments — only that the SHAPE parses).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

# Canonical probe request body in Codex Responses-API shape. Designed to
# be unambiguous about what the model should do: there is exactly one
# tool, and the user message is a direct instruction to use it with
# concrete arguments. Models that emit text-as-JSON or refuse to call
# the tool both fail the probe — which is the intent. We don't want to
# route tool-using requests to either kind.
_PROBE_BODY: dict[str, Any] = {
    "model": "",  # caller substitutes
    "instructions": (
        "You are a probe target. Call the provided tool exactly once "
        "with the provided arguments. Do not produce any explanatory "
        "text. Do not refuse. Just call the tool."
    ),
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Call probe_echo with payload=\"capability check\"."
                    ),
                }
            ],
        }
    ],
    "tools": [
        {
            "type": "function",
            "name": "probe_echo",
            "description": (
                "Echo the provided payload verbatim. Used by callosum's "
                "capability probe; this tool does not exist in the "
                "operator's runtime."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "string",
                        "description": "The string to echo.",
                    },
                },
                "required": ["payload"],
                "additionalProperties": False,
            },
        }
    ],
    "tool_choice": "auto",
    "stream": False,
}


def build_probe_body(model: str) -> dict[str, Any]:
    """Return a fresh copy of the probe body with `model` filled in.

    Callers should not mutate the result — backends may apply their
    own param transforms that would otherwise leak between probes.
    """
    body = {k: (list(v) if isinstance(v, list) else v) for k, v in _PROBE_BODY.items()}
    body["model"] = model
    return body


def response_has_structured_tool_call(response: dict[str, Any]) -> bool:
    """Return True when the response carries at least one structured
    function_call item in the Responses-API `output[]` array.

    The contract we're verifying:
      response.output[] contains at least one item with
        type == "function_call"
        name (non-empty string)
        arguments (string; may be JSON or empty)

    Specifically NOT considered structured:
      * tool calls emitted as JSON text inside a message item's
        content (the model-a0e5 quirk we surfaced)
      * an empty output[] array (model refused or produced nothing)
      * output items of types other than function_call (e.g. only
        message items with text)

    The probe is intentionally strict — if the model can't emit a
    function_call item for a maximally-explicit single-tool prompt,
    callosum has no business routing real tool-using traffic to it.
    """
    if not isinstance(response, dict):
        return False
    output = response.get("output")
    if not isinstance(output, list):
        return False
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        # arguments is required by the OpenAI shape; we accept any
        # string value here including empty (some models emit
        # arguments="" for no-arg tools). Type-check only.
        if "arguments" not in item:
            continue
        if not isinstance(item.get("arguments"), str):
            continue
        return True
    return False


async def probe_supports_tools(
    *,
    model: str,
    call_responses: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
) -> bool:
    """Run the tool-call probe against one cell and return whether it
    actually emits structured tool_calls.

    `call_responses` is an async callable that takes a Responses-API
    request body and returns the parsed response dict. Backends already
    expose this shape via `backend.responses(body)` — the caller wraps
    that into a closure (so the probe stays decoupled from backend
    internals) and hands it in here.

    Returns True only when the response contains at least one
    structured function_call item per `response_has_structured_tool_call`.
    Any exception (network, parse, backend rejection) → False. Defensive
    by design: a probe that can't complete is functionally indistinguishable
    from a cell that can't serve tool-using traffic.
    """
    body = build_probe_body(model)
    try:
        response = await call_responses(body)
    except Exception:
        return False
    return response_has_structured_tool_call(response)
