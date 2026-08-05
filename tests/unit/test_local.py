"""Tests for the local LLM gateway CLI wrapper.

Stubs the subprocess call so tests don't depend on local LLM gateway being
installed. Verifies parsing, caching, failure handling.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from callosum.local import CapabilityRow, LocalModelRegistrySource, ModelEntry


def _make_cap_row(**overrides: Any) -> dict[str, Any]:
    """Canonical minimal capability-matrix row dict for `from_cli_row`."""
    base: dict[str, Any] = {"model_id": "m1"}
    base.update(overrides)
    return base


def _make_payload(entries: list[dict[str, Any]]) -> str:
    return json.dumps({"entries": entries, "action": "models", "contract_version": 1})


def _stub_subprocess(monkeypatch, stdout: str, returncode: int = 0) -> list[list[str]]:
    """Patch subprocess.run to return a canned result. Returns a list
    that gets appended with each invocation's argv so tests can assert
    on call counts / arguments."""
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=returncode, stdout=stdout, stderr="")

    monkeypatch.setattr("callosum.local.subprocess.run", fake_run)
    return calls


def test_models_returns_parsed_entries(monkeypatch) -> None:
    payload = _make_payload(
        [
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
        ]
    )
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
    payload = _make_payload(
        [
            {
                "model": {
                    "id": "enabled-model",
                    "endpoint": "http://127.0.0.1:11434",
                    "runtime": "ollama",
                    "runtime_model": "e:1b",
                    "enabled": True,
                    "api_surfaces": ["chat"],
                },
            },
            {
                "model": {
                    "id": "disabled-model",
                    "endpoint": "http://127.0.0.1:11434",
                    "runtime": "ollama",
                    "runtime_model": "d:1b",
                    "enabled": False,
                    "api_surfaces": ["chat"],
                },
            },
        ]
    )
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


def test_models_loads_once_then_never_expires(monkeypatch) -> None:
    """A healthy snapshot is returned indefinitely: no TTL auto-refresh.
    Reads far past `refresh_s` still serve the cached snapshot without a
    re-fetch."""
    payload = _make_payload(
        [
            {
                "model": {
                    "id": "m1",
                    "endpoint": "http://127.0.0.1:11434",
                    "runtime": "ollama",
                    "runtime_model": "m1:1b",
                    "enabled": True,
                    "api_surfaces": ["chat"],
                },
            },
        ]
    )
    calls = _stub_subprocess(monkeypatch, payload)
    clock = [0.0]

    def fake_time():
        return clock[0]

    monkeypatch.setattr("callosum.local.time.time", fake_time)
    src = LocalModelRegistrySource(refresh_s=60.0)
    src.models()  # first call: fetches (models + capabilities)
    clock[0] = 10_000.0  # far past refresh_s
    src.models()  # healthy snapshot stays valid forever
    src.models()
    assert len(calls) == 2  # one fetch only


def test_models_unhealthy_cache_retries_after_refresh_s(monkeypatch) -> None:
    """A failed (unhealthy) snapshot retries on read, but throttled to once
    per `refresh_s` so a down local-llm isn't hammered on every lookup."""
    calls = _stub_subprocess(monkeypatch, "", returncode=1)
    clock = [1_000.0]

    def fake_time():
        return clock[0]

    monkeypatch.setattr("callosum.local.time.time", fake_time)
    src = LocalModelRegistrySource(refresh_s=60.0)
    assert src.models() == []  # first read: fetches, unhealthy
    first_fetches = len(calls)
    src.models()  # within refresh_s: throttled, no re-fetch
    assert len(calls) == first_fetches
    clock[0] = 1_100.0  # past refresh_s: retries
    assert src.models() == []
    assert len(calls) == 2 * first_fetches


def test_models_force_bypasses_cache(monkeypatch) -> None:
    payload = _make_payload(
        [
            {
                "model": {
                    "id": "m1",
                    "endpoint": "http://127.0.0.1:11434",
                    "runtime": "ollama",
                    "runtime_model": "m1:1b",
                    "enabled": True,
                    "api_surfaces": ["chat"],
                },
            },
        ]
    )
    calls = _stub_subprocess(monkeypatch, payload)
    src = LocalModelRegistrySource(refresh_s=60.0)
    src.models(force=True)
    src.models(force=True)
    assert len(calls) == 4


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
            "id": "x",
            "endpoint": "http://h:8090",
            "runtime": "responses_proxy",
            "runtime_model": "x",
            "api": "responses",
        },
    }
    parsed = ModelEntry.from_cli_entry(entry)
    assert parsed is not None
    assert parsed.api_surfaces == ("responses",)


def test_models_enrich_entries_with_capability_matrix(monkeypatch) -> None:
    models_payload = _make_payload(
        [
            {
                "model": {
                    "id": "m1",
                    "endpoint": "http://127.0.0.1:11434",
                    "runtime": "ollama",
                    "runtime_model": "m1:1b",
                    "enabled": True,
                    "api_surfaces": ["responses"],
                },
            },
        ]
    )
    caps_payload = json.dumps(
        {
            "host": {"total_vram_gb": 24, "total_ram_gb": 64, "unified_memory": False},
            "rows": [
                {
                    "model_id": "m1",
                    "status": "unmeasured",
                    "quantization": {"label": "ollama-q4_k_m"},
                    "host_fit": {"runnable_on_host": True},
                    "graph_metrics": {
                        "estimated_clean_total_tokens_per_second": 12.5,
                        "ceiling_search_prompt_tokens": 8192,
                        "measured_clean_total_tokens": 4096,
                        "local_first_avg_duration_seconds": 8.192,
                        "ceiling_search_completion_tokens_per_second": 10.0,
                    },
                }
            ]
        }
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if "capabilities" in cmd:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=caps_payload, stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=models_payload, stderr="")

    monkeypatch.setattr("callosum.local.subprocess.run", fake_run)
    models = LocalModelRegistrySource().models(force=True)
    assert len(models) == 1
    assert models[0].local_quantization == "ollama-q4_k_m"
    assert models[0].local_runnable_on_host is True
    assert models[0].local_status == "unmeasured"
    assert models[0].estimated_tokens_per_second == 12.5
    assert models[0].local_pool_bytes == 24 * 1024**3
    assert models[0].local_fit_limit_tokens == 8192
    assert models[0].local_prefill_ms_per_token == 2.0
    assert models[0].local_decode_bandwidth_kappa == 1.25
    assert len(calls) == 2


def test_capability_row_from_cli_row_defaults_when_keys_absent() -> None:
    """Today the hub emits neither modalities nor supports_tools; both
    must parse to None (no guesswork)."""
    row = _make_cap_row()
    cap = CapabilityRow.from_cli_row(row)
    assert cap is not None
    assert cap.modalities is None
    assert cap.supports_tools is None


def test_capability_row_parses_modalities_list_of_strings() -> None:
    row = _make_cap_row(modalities=["text", "image"])
    cap = CapabilityRow.from_cli_row(row)
    assert cap is not None
    assert cap.modalities == frozenset({"text", "image"})
    assert cap.supports_tools is None


def test_capability_row_normalizes_modalities_lowercase_and_text() -> None:
    row = _make_cap_row(modalities=["IMAGE"])
    cap = CapabilityRow.from_cli_row(row)
    assert cap is not None
    assert cap.modalities == frozenset({"text", "image"})


def test_capability_row_modalities_adds_text_when_missing() -> None:
    row = _make_cap_row(modalities=["image"])
    cap = CapabilityRow.from_cli_row(row)
    assert cap is not None
    assert cap.modalities == frozenset({"text", "image"})


def test_capability_row_supports_tools_real_bool_true() -> None:
    row = _make_cap_row(supports_tools=True)
    cap = CapabilityRow.from_cli_row(row)
    assert cap is not None
    assert cap.supports_tools is True


def test_capability_row_supports_tools_real_bool_false() -> None:
    row = _make_cap_row(supports_tools=False)
    cap = CapabilityRow.from_cli_row(row)
    assert cap is not None
    assert cap.supports_tools is False


def test_capability_row_supports_tools_string_false_is_none() -> None:
    """Regression guard: the string 'false' is truthy in Python and must
    NOT become True. Strings are never trusted as bools."""
    row = _make_cap_row(supports_tools="false")
    cap = CapabilityRow.from_cli_row(row)
    assert cap is not None
    assert cap.supports_tools is None


def test_capability_row_supports_tools_string_true_is_none() -> None:
    row = _make_cap_row(supports_tools="true")
    cap = CapabilityRow.from_cli_row(row)
    assert cap is not None
    assert cap.supports_tools is None


def test_capability_row_modalities_string_not_list_is_none() -> None:
    row = _make_cap_row(modalities="text")
    cap = CapabilityRow.from_cli_row(row)
    assert cap is not None
    assert cap.modalities is None


def test_capability_row_modalities_empty_list_is_none() -> None:
    row = _make_cap_row(modalities=[])
    cap = CapabilityRow.from_cli_row(row)
    assert cap is not None
    assert cap.modalities is None
