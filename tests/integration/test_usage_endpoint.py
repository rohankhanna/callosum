from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from fastapi.testclient import TestClient

from callosum.app import create_app
from callosum.fakes import InMemoryFakeBackend
from callosum.usage_log import UsageLog


def _usage_log(tmp_path: Path) -> UsageLog:
    return UsageLog(tmp_path / "requests.sqlite", capture_bodies=False)


def _insert_row(
    path: Path,
    *,
    model: str,
    effort: str,
    prompt: int,
    cached: int,
    completion: int,
    reasoning: int,
    delta: int,
    five_hourly_delta: int | None = None,
    crossover: int = 0,
    ts_offset: int = 0,
    backend_id: str = "fake",
    duration: int = 1,
) -> None:
    now = time.time()
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "INSERT INTO requests ("
            "ts_start, ts_end, latency_ms, route, stream, backend_id, model,"
            "reasoning_effort, status, request_bytes, response_bytes,"
            "prompt_tokens, completion_tokens, total_tokens, cached_tokens,"
            "reasoning_tokens, five_hourly_used_percent_before, five_hourly_used_percent_after,"
            "weekly_used_percent_before, weekly_used_percent_after,"
            "quota_reset_crossover, classification"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                now + ts_offset,
                now + ts_offset + duration,
                100,
                "responses",
                0,
                backend_id,
                model,
                effort,
                200,
                10,
                10,
                prompt,
                completion,
                prompt + completion,
                cached,
                reasoning,
                10,
                10 + (delta if five_hourly_delta is None else five_hourly_delta),
                20,
                20 + delta,
                crossover,
                "ok",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _rate(body: dict, model: str = "model-a0e7", effort: str | None = "medium") -> dict:
    for row in body["rates"]:
        if row["model"] == model and row["reasoning_effort"] == effort:
            return row
    raise AssertionError(f"rate row not found for {model}/{effort}: {body}")


def test_usage_endpoint_filters_historical_rows_to_current_catalog(tmp_path: Path) -> None:
    usage_log = _usage_log(tmp_path)
    for i, model in enumerate(["model-a0e7", "retired-model"]):
        for j, (uncached, cached, output, reasoning) in enumerate(
            [(100, 0, 20, 5), (60, 50, 25, 7), (140, 20, 15, 2), (30, 120, 40, 9), (90, 80, 30, 4)]
        ):
            delta = int((uncached * 0.10) + (cached * 0.02) + (output * 0.30) + (reasoning * 0.40))
            _insert_row(
                usage_log.path,
                model=model,
                effort="medium",
                prompt=uncached + cached,
                cached=cached,
                completion=output,
                reasoning=reasoning,
                delta=delta,
                ts_offset=(i * 10) + j,
            )
    backend = InMemoryFakeBackend(id="primary", advertised_models=frozenset({"model-a0e7"}))

    with TestClient(create_app(backends=[backend], usage_log=usage_log)) as client:
        body = client.get("/v1/usage").json()

    models = {row["model"] for row in body["rates"]}
    assert "model-a0e7" in models
    assert "retired-model" not in models
    for meter in body["meters"].values():
        assert "retired-model" not in {row["model"] for row in meter["rates"]}
    assert "retired-model" not in {row["model"] for row in body["relationships"]}


def test_usage_endpoint_reports_cached_split_only_when_identifiable(tmp_path: Path) -> None:
    usage_log = _usage_log(tmp_path)
    for i in range(5):
        _insert_row(
            usage_log.path,
            model="model-a0e7",
            effort="medium",
            prompt=100 + i,
            cached=50,
            completion=20,
            reasoning=5,
            delta=10 + i,
            ts_offset=i,
        )
    backend = InMemoryFakeBackend(id="primary", advertised_models=frozenset({"model-a0e7"}))

    with TestClient(create_app(backends=[backend], usage_log=usage_log)) as client:
        row = _rate(client.get("/v1/usage").json())

    assert row["input_cached"]["available"] is False
    assert row["cache_effect"]["available"] is False
    assert row["confidence"] == "insufficient_data"


def test_usage_endpoint_reports_local_cells_as_zero_not_applicable(tmp_path: Path) -> None:
    usage_log = _usage_log(tmp_path)
    backend = InMemoryFakeBackend(id="local", advertised_models=frozenset({"model-a0g1-local"}))
    backend.kind = "litellm_gateway"

    with TestClient(create_app(backends=[backend], usage_log=usage_log)) as client:
        row = _rate(client.get("/v1/usage").json(), model="model-a0g1-local")

    assert row["input_uncached"]["available"] is True
    assert row["input_uncached"]["rate"] == 0.0
    assert row["input_cached"]["rate"] == 0.0
    assert row["cache_effect"]["available"] is False
    assert row["source"] == "local_zero"


def test_usage_endpoint_excludes_quota_reset_crossover_rows(tmp_path: Path) -> None:
    usage_log = _usage_log(tmp_path)
    for i in range(4):
        _insert_row(
            usage_log.path,
            model="model-a0e7",
            effort="medium",
            prompt=100 + i,
            cached=0,
            completion=20,
            reasoning=2,
            delta=1,
            crossover=1,
            ts_offset=i,
        )
    backend = InMemoryFakeBackend(id="primary", advertised_models=frozenset({"model-a0e7"}))

    with TestClient(create_app(backends=[backend], usage_log=usage_log)) as client:
        row = _rate(client.get("/v1/usage").json())

    assert row["samples"] == {"total": 0, "usable": 0}
    assert row["confidence"] == "insufficient_data"


def test_usage_endpoint_includes_quantization_and_cache_policy_limitations(tmp_path: Path) -> None:
    usage_log = _usage_log(tmp_path)
    backend = InMemoryFakeBackend(id="primary", advertised_models=frozenset({"model-a0e7"}))

    with TestClient(create_app(backends=[backend], usage_log=usage_log)) as client:
        body = client.get("/v1/usage").json()

    assert body["metadata"]["integer_percent_quantized"] is True
    assert body["metadata"]["upstream_cache_policy_unverified"] is True
    limitations = " ".join(body["limitations"])
    assert "integer-percent quantized" in limitations
    assert "upstream prompt-cache policy" in limitations
