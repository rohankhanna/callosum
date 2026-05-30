"""Tests for the session-id header extraction.

The proxy reads `session-id` from the request as the per-instance
identifier Codex CLI declares (one stable value per Codex process).
Earlier code looked for `x-codex-session-id` which never matched real
Codex traffic, so every request's session_id column was silently NULL
even though Codex was sending the value on the wire. These tests
codify the contract so the bug can't quietly reappear.
"""

from __future__ import annotations

from types import SimpleNamespace

from callosum.app import SESSION_HEADERS, _session_id_from_request


class _CIHeaders:
    """Case-insensitive header lookup stub matching the subset of
    Starlette's Headers interface that _session_id_from_request uses."""

    def __init__(self, **headers: str) -> None:
        self._h = {k.lower(): v for k, v in headers.items()}

    def get(self, name: str, default: str | None = None) -> str | None:
        return self._h.get(name.lower(), default)


def _request_with_headers(**headers: str) -> SimpleNamespace:
    return SimpleNamespace(headers=_CIHeaders(**headers))


def test_unprefixed_session_id_header_is_captured() -> None:
    """The header Codex CLI actually sends.
    Source: codex-rs/codex-api/src/requests/headers.rs."""
    req = _request_with_headers(**{"session-id": "sess-real-codex"})
    assert _session_id_from_request(req, pinned=None) == "sess-real-codex"


def test_legacy_x_codex_session_id_header_still_works() -> None:
    """Backward-compat for any custom client still sending the prefixed
    form. Won't be hit by real Codex CLI but the proxy shouldn't break
    callers that previously worked."""
    req = _request_with_headers(**{"x-codex-session-id": "sess-legacy"})
    assert _session_id_from_request(req, pinned=None) == "sess-legacy"


def test_unprefixed_form_wins_when_both_present() -> None:
    """If both headers are sent, the unprefixed form (Codex's actual
    value) takes precedence over the legacy prefixed form."""
    req = _request_with_headers(**{
        "session-id": "sess-codex-real",
        "x-codex-session-id": "sess-legacy-ignored",
    })
    assert _session_id_from_request(req, pinned=None) == "sess-codex-real"


def test_no_header_returns_none() -> None:
    req = _request_with_headers()
    assert _session_id_from_request(req, pinned=None) is None


def test_empty_header_value_returns_none() -> None:
    """Whitespace-only or empty values shouldn't pin a session."""
    req = _request_with_headers(**{"session-id": "   "})
    assert _session_id_from_request(req, pinned=None) is None


def test_operator_pin_overrides_client_session() -> None:
    """Pinning is an operator override; the client header is ignored
    when a pin is in place. Documented existing behavior."""
    req = _request_with_headers(**{"session-id": "sess-real"})
    assert _session_id_from_request(req, pinned="some-pin") is None


def test_session_headers_constant_ordering_is_stable() -> None:
    """If this assertion breaks, the precedence order changed —
    something a downstream caller may be relying on."""
    assert SESSION_HEADERS == ("session-id", "x-codex-session-id")
