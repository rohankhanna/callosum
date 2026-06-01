"""Canonical Codex-CLI-shape request bodies for capability tests.

These bodies mirror what real Codex CLI traffic looks like on the
wire so capability tests stress the same code paths real requests
exercise. Tests select an appropriate body based on the dimension
they're probing and the cell they're targeting.

The bodies are static dicts rather than rendered from templates so
test failures point at line-stable evidence in this file.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------
# Tool definitions — a realistic Codex-CLI-shape set. Six tools so
# disambiguation is non-trivial; matches the probe-scheduler's view of
# realistic tool surface.
# --------------------------------------------------------------------

_CODEX_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "exec_command",
        "description": (
            "Run a shell command and return its stdout, stderr, exit "
            "code, and wall time. Use for any read-only inspection of "
            "the local filesystem or environment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string"},
                "timeout_s": {"type": "number"},
            },
            "required": ["cmd"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Read a UTF-8 file and return its contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_bytes": {"type": "integer"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "list_directory",
        "description": "List entries in a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "show_hidden": {"type": "boolean"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "search_text",
        "description": "Search the directory tree for text matching a regex.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "git_status",
        "description": (
            "Return the current branch and list of modified/untracked files."
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
        "description": "Update the visible plan / progress checklist.",
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


# --------------------------------------------------------------------
# System instructions — verbose Codex-CLI-style, including the
# critical contract that tool calls must be structured (not text).
# --------------------------------------------------------------------

_CODEX_INSTRUCTIONS = """\
You are an autonomous engineering assistant operating inside a
terminal-based CLI. The operator has given you access to a toolset
that lets you inspect the local environment. Use the tools to answer
the operator's question — do not guess or make assumptions about
file contents, directory structure, or repository state when you can
verify with a tool call.

Operating rules:

  * Always favor a tool call over speculating.
  * Tool calls are STRUCTURED: emit them via the OpenAI function-
    calling protocol (the `tool_calls` field of the assistant
    message), not as text content. Never write a JSON tool-call
    object as message content. If you find yourself about to write
    `{"name":"...","arguments":...}` in text, stop and emit the
    tool call structurally instead.
  * After a tool returns, briefly explain what you found and what
    you'll do next.
  * Keep responses concise. The operator is reading them in a
    terminal pane.

Available tools are passed in the `tools` field of this request.
Each tool's description tells you when to use it.
"""


# --------------------------------------------------------------------
# Body builders — one per dimension test, each scaled appropriately.
# --------------------------------------------------------------------


def tool_call_simple_body() -> dict[str, Any]:
    """A small (~3KB) tool-using request. Tests baseline tool-call
    emission. A cell that fails this isn't using callosum's tool path
    at all."""
    return {
        "model": "",  # caller sets via /admin/cell-call
        "instructions": _CODEX_INSTRUCTIONS,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "I'm debugging this repo. What's the current "
                            "git branch and are there uncommitted changes? "
                            "Please find out by inspecting, not by guessing."
                        ),
                    }
                ],
            }
        ],
        "tools": _CODEX_TOOLS,
        "tool_choice": "auto",
        "stream": False,
    }


def tool_call_with_context_body(target_user_chars: int) -> dict[str, Any]:
    """A tool-using request padded with a fake-but-coherent prior
    conversation history so total prompt size approaches
    `target_user_chars` characters. Used to test how tool-call
    behavior degrades as input grows.

    `target_user_chars` is rough (we don't tokenize) — but at ~3.5
    chars/token, 80000 chars ≈ 22K tokens, 200000 chars ≈ 55K tokens.
    Real Codex sessions can reach 200K tokens and the structured-
    tool-call shape often breaks there even when small prompts work.
    """
    # Pad the conversation with synthetic prior turns. Each turn is
    # a small interaction so the structure is realistic — a single
    # giant user message would compress badly under the model's
    # tokenizer and not stress the same code path.
    padding_turn_text = (
        "Earlier in this session I asked about the project structure. "
        "You inspected the layout, identified the src/ and tests/ "
        "directories, and noted the pyproject.toml configuration. "
        "We then briefly discussed the cell-grid abstraction and how "
        "the recommender pipeline maps prompts to cells via the "
        "capability filter. After that we moved on to the auth "
        "middleware and the loopback-only design. "
    )
    turns: list[dict[str, Any]] = []
    accumulated = 0
    turn_idx = 0
    while accumulated < target_user_chars:
        turn_idx += 1
        text = (
            f"[Earlier turn {turn_idx}] {padding_turn_text}"
            f"Tagged with index {turn_idx} so the model can verify "
            "context continuity later if it wants to."
        )
        turns.append({
            "type": "message",
            "role": "user" if turn_idx % 2 == 1 else "assistant",
            "content": [{
                "type": "input_text" if turn_idx % 2 == 1 else "output_text",
                "text": text,
            }],
        })
        accumulated += len(text)
    # Final user message — the actual ask that should trigger a tool call.
    turns.append({
        "type": "message",
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": (
                    "OK, given everything we discussed above — I need "
                    "you to check the current git branch and uncommitted "
                    "changes. Inspect with a tool, don't guess."
                ),
            }
        ],
    })
    return {
        "model": "",
        "instructions": _CODEX_INSTRUCTIONS,
        "input": turns,
        "tools": _CODEX_TOOLS,
        "tool_choice": "auto",
        "stream": False,
    }
