"""Pytest harness for model-capability testing.

These tests probe individual local cells through callosum's admin
`/admin/cell-call` endpoint and record findings to per-model JSON
profiles under logs/capability_profiles/. They take MINUTES per
model and don't run on the default test suite — they are gated by
the `model_probe` marker.

Run with:

  uv run pytest -m model_probe                    # probe every cell
  uv run pytest -m model_probe --probe-cell=NAME  # probe one cell

Skipping behavior:

  * The full suite (`uv run pytest`) does NOT run these — the marker
    excludes them by default (see pyproject.toml addopts).
  * If the proxy isn't reachable on http://127.0.0.1:8765, the tests
    fail with a clear message rather than hanging on connection
    attempts.
  * If no admin token is available (proxy was never started), the
    tests skip with a clear message.

Tests do NOT enforce behavior. They observe and record. The operator
reviews the resulting profile and decides denylist / adapter / accept.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

CALLOSUM_BASE_URL_DEFAULT = "http://127.0.0.1:8765"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--probe-cell",
        action="store",
        default=None,
        help=(
            "Restrict model_probe tests to a single cell by name. "
            "Default probes every cell advertised by callosum at probe time."
        ),
    )
    parser.addoption(
        "--probe-base-url",
        action="store",
        default=os.environ.get("CALLOSUM_BASE_URL", CALLOSUM_BASE_URL_DEFAULT),
        help="callosum base URL. Default 127.0.0.1:8765.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "model_probe: capability-probe tests; expensive (minutes per cell), "
        "gated off the default suite. Run with `pytest -m model_probe`.",
    )


def _admin_token() -> str | None:
    """Read the admin token the same way callosum-ctl does. Returns
    None when no token is available — caller skips."""
    env = os.environ.get("CALLOSUM_ADMIN_TOKEN")
    if env:
        return env.strip()
    path = Path("~/.config/callosum/admin_token").expanduser()
    if not path.exists():
        return None
    text = path.read_text().strip()
    return text or None


@pytest.fixture(scope="session")
def callosum_base_url(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--probe-base-url"))


@pytest.fixture(scope="session")
def admin_token() -> str:
    tok = _admin_token()
    if tok is None:
        pytest.skip(
            "no admin token at ~/.config/callosum/admin_token or "
            "$CALLOSUM_ADMIN_TOKEN. Start the proxy at least once "
            "before running model-capability probes."
        )
    return tok


@pytest.fixture(scope="session")
def callosum_client(callosum_base_url: str, admin_token: str) -> Iterator[httpx.Client]:
    """HTTP client pre-loaded with the admin bearer. Long timeout
    because real model calls can be 60s+ for 31B-class cells."""
    with httpx.Client(
        base_url=callosum_base_url,
        headers={"Authorization": f"Bearer {admin_token}"},
        timeout=httpx.Timeout(connect=5.0, read=600.0, write=10.0, pool=5.0),
    ) as client:
        # Sanity-check that the proxy is reachable before any probe.
        # Fails fast with a clear message instead of leaving each
        # individual test hanging on connection attempts.
        try:
            r = client.get("/status", timeout=5.0)
            if r.status_code >= 500:
                pytest.skip(f"callosum returned {r.status_code} from /status — proxy is up but unhealthy")
        except httpx.RequestError as exc:
            pytest.skip(
                f"callosum unreachable at {callosum_base_url}: {exc}. "
                "Start the proxy before running model-capability probes."
            )
        yield client


def _discover_cells(client: httpx.Client) -> list[str]:
    """Pull the live advertised cell list from /status. The probe
    suite tests whatever the proxy currently exposes — no hard-coded
    cell lists to drift out of sync."""
    r = client.get("/status", timeout=5.0)
    r.raise_for_status()
    payload = r.json()
    seen: set[str] = set()
    for b in payload.get("backends", []):
        if b.get("kind") != "litellm_gateway":
            continue
        for m in b.get("advertised_models", []) or []:
            if isinstance(m, str):
                seen.add(m)
    return sorted(seen)


@pytest.fixture(scope="session")
def cells_to_probe(request: pytest.FixtureRequest, callosum_client: httpx.Client) -> list[str]:
    requested = request.config.getoption("--probe-cell")
    if requested:
        return [str(requested)]
    cells = _discover_cells(callosum_client)
    if not cells:
        pytest.skip("no local cells advertised by callosum; nothing to probe")
    return cells


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """When `pytest -m model_probe` is NOT used, skip all
    model_probe-marked items so the default `uv run pytest` stays fast.

    The standard pytest mechanism (`-m "not model_probe"` by default
    in pyproject.toml) would also work but it's slightly less
    discoverable. Doing it here keeps the gating explicit.
    """
    marker_expression = config.getoption("-m", default="") or ""
    if "model_probe" in marker_expression:
        return
    skip_marker = pytest.mark.skip(
        reason=("model_probe tests are gated off the default suite; run with `pytest -m model_probe` to enable.")
    )
    for item in items:
        if item.get_closest_marker("model_probe") is not None:
            item.add_marker(skip_marker)


# Make `tests/model_capability/` importable as a package so test cases
# can import from `tests.model_capability.profile`. Adding the tests/
# directory to sys.path makes this work even when run from arbitrary
# cwds.
_TESTS_DIR = Path(__file__).resolve().parent.parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
