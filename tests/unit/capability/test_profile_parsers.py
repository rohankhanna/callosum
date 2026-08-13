"""Hermetic unit tests for the PURE defensive parsers in profile.

CapabilityProfile.from_dict rehydrates persisted JSON defensively so a
corrupt or old profile never breaks loading. The three module-level helpers
below are the pure coercion seam — no sqlite, no network, no app, no disk —
and each fails CLOSED to None on anything unexpected:

* _parse_optional_str — str | None for the optional gap fields
  (empty string and non-str both collapse to None).
* _parse_gap_class — Literal["A", "B"] | None; only the two valid
  class labels pass through, everything else (including old profiles with
  no field) loads as None.
* _parse_suggested_action — ContractAction | None; valid enum
  members coerce, unknown/empty/non-str values load as None so the gap
  classifier falls back to inferring from adapter_hint.

profile_path and load_profile/save_profile touch the filesystem
and are out of scope for this pure-logic pin.
"""

from __future__ import annotations

from callosum.capability.profile import (
    _parse_gap_class,
    _parse_optional_str,
    _parse_suggested_action,
)
from callosum.substrate_contract import ContractAction

# ---------- _parse_optional_str ------------------------------------------


def test_parse_optional_str_valid_passes_through() -> None:
    assert _parse_optional_str("upstream-team") == "upstream-team"


def test_parse_optional_str_none_returns_none() -> None:
    # JSON null / missing key both arrive as None
    assert _parse_optional_str(None) is None


def test_parse_optional_str_empty_string_returns_none() -> None:
    # empty str is falsy -> collapsed to None (no empty-string leakage)
    assert _parse_optional_str("") is None


def test_parse_optional_str_non_str_returns_none() -> None:
    # ints/floats/lists/dicts are all rejected defensively
    assert _parse_optional_str(42) is None
    assert _parse_optional_str(3.14) is None
    assert _parse_optional_str(["a"]) is None
    assert _parse_optional_str({"k": "v"}) is None
    assert _parse_optional_str(True) is None


# ---------- _parse_gap_class ---------------------------------------------


def test_parse_gap_class_valid_members_pass_through() -> None:
    assert _parse_gap_class("A") == "A"
    assert _parse_gap_class("B") == "B"


def test_parse_gap_class_none_returns_none() -> None:
    assert _parse_gap_class(None) is None


def test_parse_gap_class_invalid_value_returns_none() -> None:
    # anything outside the Literal["A","B"] set fails closed to None
    assert _parse_gap_class("C") is None
    assert _parse_gap_class("a") is None  # case-sensitive
    assert _parse_gap_class("AB") is None
    assert _parse_gap_class("") is None
    assert _parse_gap_class(1) is None  # non-str ignored
    assert _parse_gap_class(["A"]) is None


# ---------- _parse_suggested_action ---------------------------------------


def test_parse_suggested_action_valid_members_coerce() -> None:
    # every ContractAction member round-trips through its string value
    for action in ContractAction:
        assert _parse_suggested_action(action.value) is action


def test_parse_suggested_action_none_returns_none() -> None:
    assert _parse_suggested_action(None) is None


def test_parse_suggested_action_empty_string_returns_none() -> None:
    # falsy -> short-circuits to None (no ValueError raised)
    assert _parse_suggested_action("") is None


def test_parse_suggested_action_unknown_value_returns_none() -> None:
    # not a ContractAction member -> ValueError swallowed -> None
    assert _parse_suggested_action("not_a_real_action") is None
    assert _parse_suggested_action("route_native_v2") is None


def test_parse_suggested_action_non_str_returns_none() -> None:
    # non-str is rejected before ContractAction() is attempted
    assert _parse_suggested_action(42) is None
    assert _parse_suggested_action(["route_native"]) is None
    assert _parse_suggested_action({"action": "route_native"}) is None
    assert _parse_suggested_action(True) is None
