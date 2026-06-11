from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "codex_local_only_repro.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("codex_local_only_repro", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_status_parses_successful_status_json() -> None:
    mod = _load_script_module()
    assert mod._load_status({"returncode": 0, "stdout": '{"routing":"local-only"}'}) == {
        "routing": "local-only"
    }


def test_load_status_returns_none_on_failed_status_command() -> None:
    mod = _load_script_module()
    assert mod._load_status({"returncode": 1, "stdout": ""}) is None


def test_summarize_models_separates_responses_and_chat_only() -> None:
    mod = _load_script_module()
    raw = {
        "returncode": 0,
        "stdout": """
        {
          "entries": [
            {"model": {"id": "chat", "enabled": true, "api_surfaces": ["chat"]}},
            {"model": {"id": "responses", "enabled": true, "api_surfaces": ["responses", "chat"]}},
            {"model": {"id": "disabled", "enabled": false, "api_surfaces": ["responses"]}}
          ]
        }
        """,
    }
    summary = mod._summarize_models(raw)
    assert summary == {
        "ok": True,
        "enabled_responses": ["responses"],
        "enabled_chat_only": ["chat"],
        "disabled_count": 1,
    }


def test_without_large_text_replaces_stdout_and_stderr_with_paths() -> None:
    mod = _load_script_module()
    slim = mod._without_large_text({"stdout": "large", "stderr": "large", "returncode": 0})
    assert "stdout" not in slim
    assert "stderr" not in slim
    assert slim["stdout_path"] == "codex.stdout.txt"
    assert slim["stderr_path"] == "codex.stderr.txt"
    assert slim["returncode"] == 0
