"""Tests for the local LLM gateway CLI wrapper.

Stubs the subprocess call so tests don't depend on local LLM gateway being
installed. Verifies parsing, caching, failure handling.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from callosum.local import LocalModelRegistrySource, ModelEntry


def _make_payload(entries: list[dict[str, Any]]) -> str:
    return json.dumps({"entries": entries, "action": "models", "contract_version": 1})


def _stub_subprocess(monkeypatch, stdout: str, returncode: int = 0) -> list[list[str]]:
    """Patch subprocess.run to return a canned result. Returns a list
    that gets appended with each invocation's argv so tests can assert
    on call counts / arguments."""
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(
            args=cmd, returncode=returncode, stdout=stdout, stderr=""
        )

    monkeypatch.setattr("callosum.local.subprocess.run", fake_run)
    return calls


def test_models_returns_parsed_entries(monkeypatch) -> None:
    payload = _make_payload([
        {
            "artifacts": {"status": "present"},
            "model": {
                "id": "model-a0b0",
                "endpoint": "http://127.0.0.1:11434",
                "runtime": "ollama",
                "runtime_model": "model-a0d7",
                "family": "model-a0e5",
                "context_window": 262144,
                "api_surfaces": ["chat"],
                "enabled": True,
            },
        },
    ])
    _stub_subprocess(monkeypatch, payload)
    src = LocalModelRegistrySource()
    models = src.models(force=True)
    assert len(models) == 1
    m = models[0]
    assert m.id == "model-a0b0"
    assert m.endpoint == "http://127.0.0.1:11434"
    assert m.runtime == "ollama"
    assert m.runtime_model == "model-a0d7"
    assert m.context_window == 262144
    assert m.api_surfaces == ("chat",)


def test_models_skips_disabled_entries(monkeypatch) -> None:
    payload = _make_payload([
        {
            "model": {
                "id": "enabled-model", "endpoint": "http://127.0.0.1:11434",
                "runtime": "ollama", "runtime_model": "e:1b",
                "enabled": True, "api_surfaces": ["chat"],
            },
        },
        {
            "model": {
                "id": "disabled-model", "endpoint": "http://127.0.0.1:11434",
                "runtime": "ollama", "runtime_model": "d:1b",
                "enabled": False, "api_surfaces": ["chat"],
            },
        },
    ])
    _stub_subprocess(monkeypatch, payload)
    models = LocalModelRegistrySource().models(force=True)
    assert [m.id for m in models] == ["enabled-model"]


def test_models_handles_cli_failure(monkeypatch) -> None:
    _stub_subprocess(monkeypatch, "", returncode=1)
    models = LocalModelRegistrySource().models(force=True)
    assert models == []


def test_models_handles_malformed_json(monkeypatch) -> None:
    _stub_subprocess(monkeypatch, "not json at all")
    models = LocalModelRegistrySource().models(force=True)
    assert models == []


def test_models_caches_until_refresh_ttl(monkeypatch) -> None:
    payload = _make_payload([
        {
            "model": {
                "id": "m1", "endpoint": "http://127.0.0.1:11434",
                "runtime": "ollama", "runtime_model": "m1:1b",
                "enabled": True, "api_surfaces": ["chat"],
            },
        },
    ])
    calls = _stub_subprocess(monkeypatch, payload)
    src = LocalModelRegistrySource(refresh_s=60.0)
    src.models()  # first call: fetches
    src.models()  # second call within TTL: cached
    src.models()
    assert len(calls) == 1


def test_models_force_bypasses_cache(monkeypatch) -> None:
    payload = _make_payload([
        {
            "model": {
                "id": "m1", "endpoint": "http://127.0.0.1:11434",
                "runtime": "ollama", "runtime_model": "m1:1b",
                "enabled": True, "api_surfaces": ["chat"],
            },
        },
    ])
    calls = _stub_subprocess(monkeypatch, payload)
    src = LocalModelRegistrySource(refresh_s=60.0)
    src.models(force=True)
    src.models(force=True)
    assert len(calls) == 2


def test_models_handles_subprocess_filenotfound(monkeypatch) -> None:
    def raise_fnf(*args, **kwargs):
        raise FileNotFoundError("local-llm not found")

    monkeypatch.setattr("callosum.local.subprocess.run", raise_fnf)
    models = LocalModelRegistrySource().models(force=True)
    assert models == []


def test_is_available_returns_false_when_cli_missing(monkeypatch) -> None:
    def raise_fnf(*args, **kwargs):
        raise FileNotFoundError("not installed")

    monkeypatch.setattr("callosum.local.subprocess.run", raise_fnf)
    assert LocalModelRegistrySource.is_available() is False


def test_is_available_returns_true_on_zero_exit(monkeypatch) -> None:
    _stub_subprocess(monkeypatch, "Usage: local-llm", returncode=0)
    assert LocalModelRegistrySource.is_available() is True


def test_model_entry_from_cli_returns_none_when_id_missing() -> None:
    entry = {"model": {"endpoint": "http://x"}}
    assert ModelEntry.from_cli_entry(entry) is None


def test_model_entry_from_cli_accepts_legacy_api_field() -> None:
    """Some local LLM gateway entries have `api: "responses"` instead of
    `api_surfaces: ["responses"]`. The parser should accept either."""
    entry = {
        "model": {
            "id": "x", "endpoint": "http://h:8090", "runtime": "responses_proxy",
            "runtime_model": "x", "api": "responses",
        },
    }
    parsed = ModelEntry.from_cli_entry(entry)
    assert parsed is not None
    assert parsed.api_surfaces == ("responses",)
