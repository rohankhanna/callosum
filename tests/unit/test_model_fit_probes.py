"""Tests for the model_fit_probes table + accessors in usage_log."""

from __future__ import annotations

from pathlib import Path

from callosum.usage_log import UsageLog


def _upsert(
    log: UsageLog,
    *,
    model_id: str = "m1",
    content_hash: str = "h1",
    fits: bool = True,
    probed_at: float = 1000.0,
) -> None:
    log.upsert_model_fit_probe(
        model_id=model_id,
        content_hash=content_hash,
        advertised_context_tokens=131_072,
        achievable_context_tokens=87_376 if fits else 2_048,
        max_concurrency=1.3 if fits else 0.2,
        fits_full_context=fits,
        probed_at=probed_at,
    )


def test_upsert_then_get_for_matching_hash(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.db")
    _upsert(log, model_id="m1", content_hash="h1", fits=True)
    got = log.get_model_fit_probe_for_hash(model_id="m1", content_hash="h1")
    assert got is not None
    assert got.fits_full_context is True
    assert got.advertised_context_tokens == 131_072
    assert got.max_concurrency == 1.3
    # a different hash must not match (artifact changed -> not yet re-probed)
    assert log.get_model_fit_probe_for_hash(model_id="m1", content_hash="other") is None


def test_upsert_overwrites_on_artifact_change(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.db")
    _upsert(log, model_id="m1", content_hash="h1", fits=True, probed_at=1000.0)
    _upsert(log, model_id="m1", content_hash="h2", fits=False, probed_at=2000.0)
    latest = log.latest_model_fit_probe(model_id="m1")
    assert latest is not None
    assert latest.content_hash == "h2"
    assert latest.fits_full_context is False
    assert latest.probed_at == 2000.0
    # old hash no longer matches the stored row
    assert log.get_model_fit_probe_for_hash(model_id="m1", content_hash="h1") is None
    assert log.get_model_fit_probe_for_hash(model_id="m1", content_hash="h2") is not None


def test_all_model_fit_probes_and_last_request_end(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.db")
    _upsert(log, model_id="m1", content_hash="h1", fits=True)
    _upsert(log, model_id="m2", content_hash="h1", fits=False)
    all_probes = log.all_model_fit_probes()
    assert set(all_probes) == {"m1", "m2"}
    assert all_probes["m1"].fits_full_context is True
    assert all_probes["m2"].fits_full_context is False
    # No requests logged yet -> idle gate sees None.
    assert log.last_request_end_ts() is None


def test_get_for_unknown_model_is_none(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.db")
    assert log.latest_model_fit_probe(model_id="nope") is None
    assert log.get_model_fit_probe_for_hash(model_id="nope", content_hash="x") is None