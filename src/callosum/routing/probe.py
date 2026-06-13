"""Tool-call verification probe.

A capability check: does this specific cell actually emit OpenAI-shaped
`tool_calls` end-to-end through callosum's pipeline, under conditions
that resemble real Codex CLI traffic?

Why this exists despite ollama's static `tools` capability flag:
ollama's flag means "the model's template supports tool prompting" —
not "the model emits responses that parse to structured tool_calls
when callosum's gateway translator processes them." Without runtime
verification, broken cells stay routable and Codex CLI ends up
rendering JSON-as-text in the user's terminal.

# Why the probe is shaped the way it is

An earlier, simpler probe (single tool, directive prompt, trivial
arguments) produced false positives in production: model-a0c8 passed
the probe but emitted JSON-as-text under real Codex CLI traffic. The
LiteLLM `add_function_to_prompt` fallback was strong enough to parse
the trivial canonical case but broke down on harder inputs.

The probe is now shaped to resemble realistic Codex CLI requests:

  * **Multiple tools** in the request (~6), not one. Forces the model
    to actually parse the tool definitions and pick one — single-tool
    probes degenerate into "always call that tool" which is easier
    than real traffic.
  * **Longer context** (~5K tokens of system+user prompt). Not as long
    as a real Codex session (which can be 200K+) but long enough to
    expose models that work on tiny prompts and break on realistic
    ones. Probe latency stays under ~60s for 31B-class models on a
    typical GPU.
  * **Less directive prompt**. The user message describes a task that
    naturally requires tool use, instead of literally saying "call
    this tool with these arguments". Models that need explicit
    spoon-feeding fail.
  * **JSON-as-text rejection**. Even if the model emits one structured
    function_call by accident, if it ALSO emits text content that
    parses as a tool-call JSON object, the probe fails. Real Codex
    CLI traffic hits this case when the model is "almost working" —
    it produces a structured call AND duplicates the call as JSON
    in its message text, which is the visible failure mode.

# What the probe is NOT testing

  * Streaming (non-stream `responses` is sufficient for shape
    verification).
  * Modalities other than text.
  * Tool-call CORRECTNESS (did the model pick the right tool, pass
    the right arguments?). Only that the SHAPE parses.
  * Long-context behavior past ~5K tokens. A model that works at 5K
    but breaks at 100K is currently classified as tools-capable —
    revisit if real-traffic data shows the boundary actually matters.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

# Six tools mirroring the shape Codex CLI sends — names and parameter
# schemas are synthetic but deliberately resemble realistic tools so
# the probe stresses tool-disambiguation logic the same way real
# traffic does. Models that can only handle a single-tool request
# (the previous probe surface) get filtered out here.
_PROBE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "exec_command",
        "description": (
            "Run a shell command and return its stdout, stderr, exit code, "
            "and wall time. Use for any read-only inspection of the local "
            "filesystem or environment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout_s": {
                    "type": "number",
                    "description": "Hard kill timeout in seconds.",
                },
            },
            "required": ["cmd"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": (
            "Read a UTF-8 file from disk and return its contents. Returns an error if the path is missing or not text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or working-directory-relative path.",
                },
                "max_bytes": {
                    "type": "integer",
                    "description": "Cap on bytes returned; default 64KB.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_directory",
        "description": (
            "List the entries in a directory. Returns each entry's name and whether it is a file or directory."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path.",
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Include dotfiles. Default false.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_text",
        "description": (
            "Search the working directory tree for text matching a regex. "
            "Wraps ripgrep. Returns matches with file path and line number."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern.",
                },
                "path": {
                    "type": "string",
                    "description": "Root to search under. Defaults to cwd.",
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "git_status",
        "description": (
            "Return the current branch, list of modified files, and list "
            "of untracked files in the working directory's git repo."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "update_plan",
        "description": (
            "Update the visible plan / progress checklist shown to the user. Use whenever the high-level plan changes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["title", "status"],
                    },
                },
            },
            "required": ["items"],
            "additionalProperties": False,
        },
    },
]


# System instructions mimicking what Codex CLI sends. Deliberately
# verbose to push the context into a few KB without contrived padding.
# Models that can only handle short prompts fail at this boundary.
_PROBE_INSTRUCTIONS = """\
You are an autonomous engineering assistant operating inside a
terminal-based CLI. The operator has given you access to a small
toolset that lets you inspect the local environment and report
findings. Use the tools to answer the operator's question
correctly — do not guess or make assumptions about file contents,
directory structure, or repository state when you can verify with
a tool call.

Operating rules:

  * Always favor a tool call over speculating.
  * Tool calls are structured: emit them via the OpenAI function-
    calling protocol (the `tool_calls` field in your message), not
    as text. Never write a JSON tool-call object as message content.
    If you find yourself about to write `{"name":"...","arguments":...}`
    in text, stop and emit the tool call structurally instead.
  * After a tool returns, briefly explain what you found and what
    you'll do next.
  * Keep responses concise. The operator is reading them in a
    terminal pane.

Available tools are passed in the `tools` field of this request.
Each tool's description tells you when to use it.
"""


# User message designed to make the model pick a tool naturally —
# the operator's question doesn't name a tool, doesn't spoon-feed
# arguments, and the obvious next action is to call one of the
# inspection tools. Probe passes if the model emits a structured
# function_call for ANY of the six tools.
_PROBE_USER_MESSAGE = """\
I'm starting a debugging session in this directory and I need to
understand the current state of the repo before I can decide what
to investigate. Specifically:

  1. What branch is currently checked out?
  2. Are there uncommitted changes I should be aware of?
  3. Are there any untracked files?

Please answer these by inspecting the repo, not by guessing.
Pick the best tool from the ones available and use it.
"""


# Canonical probe request body in Codex Responses-API shape, sized
# to ~5K tokens (instructions + user message + tool definitions).
# Caller substitutes `model` before sending.
_PROBE_BODY: dict[str, Any] = {
    "model": "",
    "instructions": _PROBE_INSTRUCTIONS,
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": _PROBE_USER_MESSAGE},
            ],
        }
    ],
    "tools": _PROBE_TOOLS,
    "tool_choice": "auto",
    "stream": False,
}


def build_probe_body(model: str) -> dict[str, Any]:
    """Return a fresh copy of the probe body with `model` filled in.

    Callers should not mutate the result — backends may apply their
    own param transforms that would otherwise leak between probes.
    Returns a deep-enough copy that test mutations on `input` /
    `tools` don't leak into subsequent probes.
    """
    body: dict[str, Any] = {}
    for k, v in _PROBE_BODY.items():
        if isinstance(v, list):
            body[k] = [dict(item) if isinstance(item, dict) else item for item in v]
        else:
            body[k] = v
    body["model"] = model
    return body


def _text_looks_like_tool_call_json(text: str) -> bool:
    """Return True when `text` parses as JSON in the shape of an
    OpenAI tool call, i.e. an object with both a `name` (string) and
    `arguments` (string or object) key.

    This is the model-a0e5 failure pattern: the model emits a real-
    looking tool call BUT inside `message.content` as text rather
    than as a structured `function_call` item. Strict-mode tooling
    (Codex CLI's parser) drops these, leaving the user with visible
    JSON junk in the terminal.

    We do a strict parse rather than a regex because regex on JSON
    is fragile (multi-line content, escaped quotes, etc.) and the
    probe is a one-shot offline check where parse cost doesn't
    matter. Non-JSON text returns False quickly via the try.
    """
    stripped = text.strip()
    if not stripped:
        return False
    # Cheap pre-check: only attempt JSON parse if the text starts
    # like a JSON object. Avoids paying full json.loads cost for
    # the overwhelming majority of normal model responses.
    if not stripped.startswith("{"):
        return False
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(parsed, dict):
        return False
    if not isinstance(parsed.get("name"), str) or not parsed["name"]:
        return False
    args = parsed.get("arguments")
    return isinstance(args, (str, dict))


def response_has_structured_tool_call(response: dict[str, Any]) -> bool:
    """Return True when the response carries at least one structured
    function_call item AND does not also emit a tool-call-shaped
    JSON object inside message text content.

    The dual condition matters: some models (model-a0e5 the canonical
    example) sometimes emit BOTH a structured function_call AND
    duplicate the call as JSON in their text content. Codex CLI's
    parser handles the structured call but renders the duplicate
    JSON as visible junk. The probe must fail those cells — emitting
    correct structure isn't enough if the text channel also leaks.

    Specifically passes when:
      * response.output[] contains at least one item with
          type == "function_call"
          name (non-empty string)
          arguments (string)
      * AND no message item's text content parses as a tool-call
        JSON object (`{"name": ..., "arguments": ...}`).

    Fails on:
      * empty output[] (refused or produced nothing)
      * output items that are only `message` type with text (no
        structured call)
      * malformed function_call items (missing name/arguments,
        non-string arguments)
      * structured function_call PLUS text that looks like a
        tool-call JSON (the dual-emit failure mode)
    """
    if not isinstance(response, dict):
        return False
    output = response.get("output")
    if not isinstance(output, list):
        return False

    has_structured_call = False
    has_text_tool_call_leak = False

    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")

        if item_type == "function_call":
            name = item.get("name")
            if not isinstance(name, str) or not name:
                continue
            if "arguments" not in item:
                continue
            if not isinstance(item.get("arguments"), str):
                continue
            has_structured_call = True
            continue

        if item_type == "message":
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if not isinstance(text, str):
                    continue
                if _text_looks_like_tool_call_json(text):
                    has_text_tool_call_leak = True

    return has_structured_call and not has_text_tool_call_leak


async def probe_supports_tools(
    *,
    model: str,
    call_responses: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
) -> bool:
    """Run the tool-call probe against one cell and return whether it
    actually emits structured tool_calls under realistic conditions.

    `call_responses` is an async callable that takes a Responses-API
    request body and returns the parsed response dict. Backends already
    expose this shape via `backend.responses(body)` — the caller wraps
    that into a closure (so the probe stays decoupled from backend
    internals) and hands it in here.

    Returns True only when the response contains at least one
    structured function_call item AND does not also emit tool-call-
    shaped JSON in message text content. See
    `response_has_structured_tool_call` for the full contract.

    Any exception (network, parse, backend rejection) → False.
    Defensive by design: a probe that can't complete is functionally
    indistinguishable from a cell that can't serve tool-using traffic.
    """
    body = build_probe_body(model)
    try:
        response = await call_responses(body)
    except Exception:
        return False
    return response_has_structured_tool_call(response)
