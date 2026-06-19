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


def _seed_rows(path: Path, *, model: str, effort: str, n: int = 24, latency_base: int = 900) -> None:
    now = time.time()
    conn = sqlite3.connect(path)
    try:
        for i in range(n):
            prompt = 100 + i
            completion = 50 + i
            latency = latency_base + i * 25
            conn.execute(
                "INSERT INTO requests ("
                "ts_start, ts_end, latency_ms, route, stream, backend_id, model,"
                "reasoning_effort, status, request_bytes, response_bytes,"
                "prompt_tokens, completion_tokens, total_tokens, classification"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    now - 1000 + i,
                    now - 999 + i,
                    latency,
                    "responses",
                    0,
                    "fake",
                    model,
                    effort,
                    200,
                    10,
                    10,
                    prompt,
                    completion,
                    prompt + completion,
                    "ok",
                ),
            )
        conn.commit()
    finally:
        conn.close()


def test_eta_endpoint_returns_approximate_ranges_when_estimator_present(tmp_path: Path) -> None:
    usage_log = _usage_log(tmp_path)
    _seed_rows(usage_log.path, model="model-a0e7", effort="medium")
    backend = InMemoryFakeBackend(id="primary", advertised_models=frozenset({"model-a0e7"}))

    with TestClient(create_app(backends=[backend], usage_log=usage_log)) as client:
        response = client.post(
            "/v1/eta",
            json={"model": "model-a0e7", "reasoning_effort": "medium", "input_tokens": 120},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["unit"] == "ms"
    assert len(body["estimates"]) == 1
    estimate = body["estimates"][0]
    assert estimate["cell"] == {"model": "model-a0e7", "reasoning_effort": "medium"}
    assert estimate["eta"]["p50_ms"] > 0
    assert estimate["eta"]["high_ms"] >= estimate["eta"]["p50_ms"]
    assert estimate["eta"]["range"] == "approximate_p50_to_p95"
    assert estimate["eta"]["exact"] is False
    assert estimate["metadata"]["time_sample_count"] >= 10
    assert estimate["metadata"]["output_sample_count"] >= 20
    assert "cell-measured" in estimate["metadata"]["source"]


def test_eta_endpoint_handles_unavailable_estimator_cleanly() -> None:
    backend = InMemoryFakeBackend(id="primary", advertised_models=frozenset({"model-a0e7"}))
    with TestClient(create_app(backends=[backend], usage_log=None)) as client:
        response = client.post("/v1/eta", json={"input_tokens": 100})

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "available": False,
        "reason": "time_estimator_unavailable",
        "unit": "ms",
        "input_tokens": None,
        "estimates": [],
    }


def test_eta_endpoint_marks_insufficient_data_without_exact_claim(tmp_path: Path) -> None:
    usage_log = _usage_log(tmp_path)
    backend = InMemoryFakeBackend(id="primary", advertised_models=frozenset({"model-a0e7"}))

    with TestClient(create_app(backends=[backend], usage_log=usage_log)) as client:
        response = client.post(
            "/v1/eta",
            json={"model": "model-a0e7", "reasoning_effort": "low", "input_tokens": 50},
        )

    assert response.status_code == 200
    estimate = response.json()["estimates"][0]
    assert estimate["eta"]["exact"] is False
    assert estimate["eta"]["p50_ms"] > 0
    assert estimate["metadata"]["confidence"] == "insufficient-data"
    assert estimate["metadata"]["time_sample_count"] == 0
    assert estimate["metadata"]["output_sample_count"] == 0
    assert "fallback" in estimate["metadata"]["source"]


def test_eta_endpoint_local_cells_can_have_nonzero_time_estimates(tmp_path: Path) -> None:
    usage_log = _usage_log(tmp_path)
    backend = InMemoryFakeBackend(id="local", advertised_models=frozenset({"model-a0g1-local"}))
    backend.kind = "litellm_gateway"

    with TestClient(create_app(backends=[backend], usage_log=usage_log)) as client:
        response = client.post(
            "/v1/eta",
            json={"model": "model-a0g1-local", "reasoning_effort": "medium", "input_tokens": 100},
        )

    assert response.status_code == 200
    estimate = response.json()["estimates"][0]
    assert estimate["cell"] == {"model": "model-a0g1-local", "reasoning_effort": "medium"}
    assert estimate["eta"]["p50_ms"] > 0
    assert estimate["eta"]["high_ms"] > 0
    assert estimate["metadata"]["confidence"] == "insufficient-data"
