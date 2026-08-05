"""Tests for the model fit probe executor + runner.

The subprocess/CLI surface is mocked; these tests cover the pure parsing,
content-hash, preflight, and dedup logic, plus one mocked end-to-end probe
that verifies the vLLM log is parsed and the result is persisted.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import callosum.jobs.model_probe as job
import callosum.model_probe as mp
from callosum.local import ModelEntry
from callosum.usage_log import ModelFitProbe, UsageLog

FIT_LOG = (
    "engine core.py:97] Initializing a V1 LLM engine ... max_seq_len=131072 ...\n"
    "kv_cache_utils.py:1307] GPU KV cache size: 87,376 tokens\n"
    "kv_cache_utils.py:1312] Maximum concurrency for 131,072 tokens per request: 1.31x\n"
)
NOFIT_LOG = (
    "core.py:97] ... max_seq_len=131072 ...\n"
    "kv_cache_utils.py:1307] GPU KV cache size: 20,000 tokens\n"
    "kv_cache_utils.py:1312] Maximum concurrency for 131,072 tokens per request: 0.15x\n"
)


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _model(model_id: str = "m1", pool_bytes: int | None = 100 * 1024**3) -> ModelEntry:
    return ModelEntry(
        id=model_id,
        endpoint="http://127.0.0.1:8000",
        runtime="vllm",
        runtime_model=model_id,
        family="fam",
        context_window=131_072,
        api_surfaces=("responses",),
        enabled=True,
        local_pool_bytes=pool_bytes,
    )


def test_parse_fit_log_extracts_numbers() -> None:
    fit = mp._parse_vllm_fit_log(FIT_LOG)
    assert fit.max_seq_len == 131_072
    assert fit.kv_cache_tokens == 87_376
    assert fit.max_concurrency == 1.31


def test_parse_fit_log_no_fit_case() -> None:
    fit = mp._parse_vllm_fit_log(NOFIT_LOG)
    assert fit.max_seq_len == 131_072
    assert fit.kv_cache_tokens == 20_000
    assert fit.max_concurrency == 0.15


def test_parse_fit_log_missing_lines() -> None:
    fit = mp._parse_vllm_fit_log("nothing relevant here")
    assert fit.max_seq_len is None
    assert fit.kv_cache_tokens is None
    assert fit.max_concurrency is None


def test_content_hash_and_weight_bytes(tmp_path: Path) -> None:
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"y" * 50)
    h1 = mp._content_hash(str(tmp_path))
    assert h1 is not None
    assert mp._weight_bytes(str(tmp_path)) == 150
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"x" * 200)
    h2 = mp._content_hash(str(tmp_path))
    assert h2 is not None and h1 != h2
    assert mp._weight_bytes(str(tmp_path)) == 250


def test_preflight_admits_blocks_when_weights_too_big() -> None:
    admitted, reason = mp.preflight_admits(
        pool_bytes=10_000, weight_bytes=9_500, utilization_fraction=0.8, activation_margin_bytes=1_000
    )
    assert admitted is False
    assert reason


def test_preflight_admits_when_headroom_ok(monkeypatch) -> None:
    monkeypatch.setattr(mp, "_read_memavailable_bytes", lambda: 80 * 1024**3)
    admitted, _ = mp.preflight_admits(
        pool_bytes=100 * 1024**3, weight_bytes=20 * 1024**3, utilization_fraction=0.8,
        activation_margin_bytes=2 * 1024**3,
    )
    assert admitted is True


def test_kv_budget_subtracts_weights_and_margin() -> None:
    gib = 1024**3
    budget = mp._kv_budget_bytes(
        pool_bytes=200 * gib, weight_bytes=50 * gib, utilization_fraction=0.8, activation_margin_bytes=4 * gib
    )
    assert budget == int(200 * gib * 0.8) - 50 * gib - 4 * gib  # 106 GiB
    # floored at DEFAULT_KV_FLOOR_BYTES when the computed budget would be tiny
    small = mp._kv_budget_bytes(
        pool_bytes=100, weight_bytes=90, utilization_fraction=0.8, activation_margin_bytes=5
    )
    assert small == mp.DEFAULT_KV_FLOOR_BYTES


def _fake_cli_factory(monkeypatch, *, mdir: Path, logfile: Path) -> dict[str, bool]:
    """Patch _run_cli to simulate local-llm; returns a 'serve called' marker."""
    called: dict[str, bool] = {"serve": False, "stop": False}

    def fake_run_cli(cli, args, *, timeout_s, env=None, capture=True):  # type: ignore[no-untyped-def]
        if args[:2] == ["models", "local"]:
            return _completed(
                0,
                stdout=json.dumps(
                    {
                        "entries": [
                            {
                                "model": {"id": "m1", "endpoint": "http://x", "runtime": "vllm"},
                                "artifacts": {"detail": f"present | target={mdir}"},
                            }
                        ]
                    }
                ),
            )
        if args[:1] == ["serve"]:
            called["serve"] = True
            return _completed(0, stdout=json.dumps({"ready": True, "log_file": str(logfile)}))
        if args[:1] == ["stop"]:
            called["stop"] = True
            return _completed(0)
        return _completed(1, stderr="unknown cli")

    monkeypatch.setattr(mp, "_run_cli", fake_run_cli)
    return called


def test_execute_model_probe_parses_and_persists(tmp_path: Path, monkeypatch) -> None:
    mdir = tmp_path / "model"
    mdir.mkdir()
    (mdir / "model.safetensors").write_bytes(b"x" * 1024)
    logfile = tmp_path / "serve.log"
    logfile.write_text(FIT_LOG)
    _fake_cli_factory(monkeypatch, mdir=mdir, logfile=logfile)
    monkeypatch.setattr(mp, "_read_memavailable_bytes", lambda: 100 * 1024**3)

    log = UsageLog(tmp_path / "u.db")
    probe = mp.execute_model_probe(_model(), usage_log=log, serve_timeout_s=10.0)
    assert probe is not None
    assert probe.fits_full_context is True
    assert probe.advertised_context_tokens == 131_072
    assert probe.achievable_context_tokens == 87_376
    assert log.latest_model_fit_probe(model_id="m1") is not None


def test_execute_model_probe_reports_no_fit(tmp_path: Path, monkeypatch) -> None:
    mdir = tmp_path / "model"
    mdir.mkdir()
    (mdir / "model.safetensors").write_bytes(b"x" * 1024)
    logfile = tmp_path / "serve.log"
    logfile.write_text(NOFIT_LOG)
    _fake_cli_factory(monkeypatch, mdir=mdir, logfile=logfile)
    monkeypatch.setattr(mp, "_read_memavailable_bytes", lambda: 100 * 1024**3)

    log = UsageLog(tmp_path / "u.db")
    probe = mp.execute_model_probe(_model(), usage_log=log, serve_timeout_s=10.0)
    assert probe is not None
    assert probe.fits_full_context is False
    assert probe.max_concurrency == 0.15


def test_execute_model_probe_skips_already_probed(tmp_path: Path, monkeypatch) -> None:
    mdir = tmp_path / "model"
    mdir.mkdir()
    (mdir / "model.safetensors").write_bytes(b"x" * 1024)
    log = UsageLog(tmp_path / "u.db")
    content_hash = mp._content_hash(str(mdir))
    assert content_hash is not None
    # Pre-record a result for this exact artifact -> must skip without serving.
    log.upsert_model_fit_probe(
        model_id="m1",
        content_hash=content_hash,
        advertised_context_tokens=131_072,
        achievable_context_tokens=87_376,
        max_concurrency=1.3,
        fits_full_context=True,
        probed_at=1.0,
    )
    called = _fake_cli_factory(monkeypatch, mdir=mdir, logfile=tmp_path / "serve.log")
    monkeypatch.setattr(mp, "_read_memavailable_bytes", lambda: 100 * 1024**3)

    probe = mp.execute_model_probe(_model(), usage_log=log, serve_timeout_s=10.0)
    assert probe is None
    assert called["serve"] is False


def test_probe_candidates_filters_to_enabled_vllm_responses() -> None:
    class FakeSource:
        def models(self, *, force: bool = False) -> list[ModelEntry]:
            return [
                _model("vllm-good"),
                ModelEntry(
                    id="chat-only",
                    endpoint="http://x",
                    runtime="vllm",
                    runtime_model="chat-only",
                    family="f",
                    context_window=8,
                    api_surfaces=("chat",),
                    enabled=True,
                ),
                ModelEntry(
                    id="ollama-one",
                    endpoint="http://x",
                    runtime="ollama",
                    runtime_model="ollama-one",
                    family="f",
                    context_window=8,
                    api_surfaces=("responses",),
                    enabled=True,
                ),
                ModelEntry(
                    id="disabled",
                    endpoint="http://x",
                    runtime="vllm",
                    runtime_model="disabled",
                    family="f",
                    context_window=8,
                    api_surfaces=("responses",),
                    enabled=False,
                ),
            ]

        def capabilities(self, *, force: bool = False) -> dict[str, object]:
            return {}

    ids = [m.id for m in job._probe_candidates(FakeSource())]  # type: ignore[attr-defined]
    assert ids == ["vllm-good"]


def test_runner_respects_max_models_and_sigterm(tmp_path: Path, monkeypatch) -> None:
    # Force the runner to see candidates and a no-op probe; verify --max-models
    # and SIGTERM both stop the loop early.
    monkeypatch.setattr(job, "_probe_candidates", lambda source: [_model("m1"), _model("m2"), _model("m3")])

    seen: list[str] = []

    def fake_execute(model, *, usage_log, cli_command, serve_timeout_s, pool_bytes):  # type: ignore[no-untyped-def]
        seen.append(model.id)
        return ModelFitProbe(
            model_id=model.id,
            content_hash="x",
            advertised_context_tokens=8,
            achievable_context_tokens=4,
            max_concurrency=1.0,
            fits_full_context=True,
            probed_at=0.0,
        )

    monkeypatch.setattr(job, "execute_model_probe", fake_execute)

    class _DummyLog:
        def __init__(self, path: Path) -> None:
            self.path = path

    monkeypatch.setattr(job, "UsageLog", _DummyLog)
    (tmp_path / "u.db").touch()  # the runner refuses a missing db path

    job._should_exit = True  # SIGTERM set before the loop -> no probes
    _summary, rc = job.run(db_path=tmp_path / "u.db", max_models=2)
    assert rc == 0
    assert seen == []
    job._should_exit = False

    seen.clear()
    _summary, rc = job.run(db_path=tmp_path / "u.db", max_models=2)
    assert rc == 0
    assert seen == ["m1", "m2"]