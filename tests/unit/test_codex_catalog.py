"""Unit tests for the codex `/model` picker catalog reconciler.

These exercise the pure projection logic (lane labels, ordering, catalog
build, digest, atomic write) without invoking the codex binary. Template
sourcing (`load_codex_template`) is covered only for its failure modes — the
happy path shells out to codex and is verified end-to-end elsewhere.
"""

from __future__ import annotations

import json

import pytest

from callosum import codex_catalog as cc


def _template() -> dict:
    """A minimal-but-representative codex ModelInfo template."""
    return {
        "slug": "model-a0e8",
        "display_name": "MODEL-A0E8",
        "description": "headline",
        "default_reasoning_level": "medium",
        "supported_reasoning_levels": [
            {"effort": "low", "description": "low"},
            {"effort": "medium", "description": "medium"},
            {"effort": "high", "description": "high"},
            {"effort": "xhigh", "description": "xhigh"},
        ],
        "visibility": "list",
        "priority": 0,
        "base_instructions": "be a coding agent",
        "availability_nux": {"message": "shiny new model"},
        "upgrade": {"slug": "model-a0f9"},
        "context_window": 400000,
    }


@pytest.mark.parametrize(
    "model_id,expected_name,expected_priority",
    [
        ("callosum:auto", "Callosum: Auto", 0),
        ("callosum:remote-only", "Callosum: Remote only", 1),
        ("callosum:local-only", "Callosum: Local only", 2),
        ("callosum:remote/model-a0e8:high", "Callosum remote · model-a0e8 · high", 10),
        ("callosum:remote/model-a0e8", "Callosum remote · model-a0e8", 10),
        ("callosum:local/model-a0g2", "Callosum local · model-a0g2", 20),
        ("callosum:local/model-a0d2:high", "Callosum local · model-a0d2 · high", 20),
    ],
)
def test_lane_metadata_known_lanes(model_id, expected_name, expected_priority):
    meta = cc.lane_metadata(model_id)
    assert meta is not None
    name, _desc, priority = meta
    assert name == expected_name
    assert priority == expected_priority


@pytest.mark.parametrize(
    "model_id",
    ["model-a0e7", "raw-passthrough-id", "callosum:bogus", "callosum:offline"],
)
def test_lane_metadata_excludes_non_menu_ids(model_id):
    # Raw ids, unknown selectors, and rejected selectors (offline) are kept out
    # of the picker.
    assert cc.lane_metadata(model_id) is None


def test_ordered_lane_ids_strategy_first_then_pins_then_declared():
    ids = [
        "model-a0e7",  # raw → dropped
        "callosum:local/model-a0g2",
        "callosum:remote/model-a0e8:high",
        "callosum:local-only",
        "callosum:auto",
        "callosum:remote-only",
    ]
    declared = ["callosum:remote/model-a0e9:xhigh", "callosum:auto"]
    out = cc._ordered_lane_ids(ids, declared)
    # Strategy selectors lead in fixed order.
    assert out[:3] == [
        "callosum:auto",
        "callosum:remote-only",
        "callosum:local-only",
    ]
    # Declared-but-not-live lane is present exactly once; no dupes.
    assert out.count("callosum:auto") == 1
    assert "callosum:remote/model-a0e9:xhigh" in out
    assert "model-a0e7" not in out
    assert len(out) == len(set(out))


def test_build_catalog_restamps_identity_and_keeps_rich_fields():
    cat = cc.build_codex_catalog(
        model_ids=["callosum:auto"],
        declared_lanes=[],
        template=_template(),
    )
    assert list(cat) == ["models"]
    (entry,) = cat["models"]
    assert entry["slug"] == "callosum:auto"
    assert entry["display_name"] == "Callosum: Auto"
    assert entry["visibility"] == "list"
    # Identity-noise fields are cleared so a Callosum lane never inherits the
    # headline model's "shiny new model" nux or an upgrade pointer.
    assert entry["availability_nux"] is None
    assert entry["upgrade"] is None
    # Rich behavioral fields are inherited from the template.
    assert entry["base_instructions"] == "be a coding agent"
    assert entry["context_window"] == 400000


def test_remote_pin_restricts_reasoning_levels_to_baked_effort():
    cat = cc.build_codex_catalog(
        model_ids=["callosum:remote/model-a0e8:high"],
        declared_lanes=[],
        template=_template(),
    )
    (entry,) = cat["models"]
    levels = entry["supported_reasoning_levels"]
    assert [lvl["effort"] for lvl in levels] == ["high"]
    assert entry["default_reasoning_level"] == "high"


def test_local_pin_restricts_reasoning_levels_to_baked_effort():
    # Symmetric to the remote case: a local pin that bakes an effort offers only
    # that effort in the picker ().
    cat = cc.build_codex_catalog(
        model_ids=["callosum:local/model-a0d2:high"],
        declared_lanes=[],
        template=_template(),
    )
    (entry,) = cat["models"]
    assert entry["slug"] == "callosum:local/model-a0d2:high"
    assert entry["display_name"] == "Callosum local · model-a0d2 · high"
    levels = entry["supported_reasoning_levels"]
    assert [lvl["effort"] for lvl in levels] == ["high"]
    assert entry["default_reasoning_level"] == "high"


def test_strategy_selector_keeps_full_reasoning_levels():
    cat = cc.build_codex_catalog(
        model_ids=["callosum:auto"],
        declared_lanes=[],
        template=_template(),
    )
    (entry,) = cat["models"]
    assert [lvl["effort"] for lvl in entry["supported_reasoning_levels"]] == [
        "low",
        "medium",
        "high",
        "xhigh",
    ]


def test_declared_lane_appears_even_when_not_live():
    cat = cc.build_codex_catalog(
        model_ids=["callosum:auto"],
        declared_lanes=["callosum:remote/model-a0e9:xhigh"],
        template=_template(),
    )
    slugs = [m["slug"] for m in cat["models"]]
    assert "callosum:remote/model-a0e9:xhigh" in slugs


def test_digest_is_stable_and_order_independent():
    a = cc.build_codex_catalog(
        model_ids=["callosum:auto", "callosum:remote-only"],
        declared_lanes=[],
        template=_template(),
    )
    b = cc.build_codex_catalog(
        model_ids=["callosum:remote-only", "callosum:auto"],
        declared_lanes=[],
        template=_template(),
    )
    assert cc.catalog_digest(a) == cc.catalog_digest(b)


def test_write_catalog_atomic_roundtrips(tmp_path):
    cat = cc.build_codex_catalog(model_ids=["callosum:auto"], declared_lanes=[], template=_template())
    out = tmp_path / "nested" / "callosum-catalog.json"
    cc.write_catalog_atomic(out, cat)
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded == cat
    # No stray temp file left behind.
    assert not (out.parent / (out.name + ".tmp")).exists()


def test_load_template_returns_none_when_codex_missing():
    # A nonexistent binary must degrade gracefully (the reconciler then skips
    # the write rather than emitting an invalid catalog).
    assert cc.load_codex_template(codex_bin="definitely-not-a-real-binary-xyz") is None


def test_reconcile_once_skips_write_when_template_unavailable(tmp_path, monkeypatch):
    out = tmp_path / "callosum-catalog.json"
    monkeypatch.setattr(cc, "load_codex_template", lambda **_: None)
    rec = cc.CodexCatalogReconciler(
        output_path=out,
        model_ids_fn=lambda: ["callosum:auto"],
    )
    assert rec.reconcile_once() is False
    assert not out.exists()


def test_reconcile_once_writes_then_noops_until_change(tmp_path, monkeypatch):
    out = tmp_path / "callosum-catalog.json"
    monkeypatch.setattr(cc, "load_codex_template", lambda **_: _template())
    ids = ["callosum:auto"]
    rec = cc.CodexCatalogReconciler(
        output_path=out,
        model_ids_fn=lambda: list(ids),
    )
    # First call writes.
    assert rec.reconcile_once() is True
    assert out.exists()
    # Unchanged catalog → no rewrite.
    assert rec.reconcile_once() is False
    # Catalog changes → rewrite.
    ids.append("callosum:remote-only")
    assert rec.reconcile_once() is True
    slugs = [m["slug"] for m in json.loads(out.read_text())["models"]]
    assert "callosum:remote-only" in slugs
