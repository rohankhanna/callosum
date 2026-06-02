"""Pytest entry into the canonical capability runner.

This file used to host per-dimension pytest functions that each
duplicated the runner's orchestration logic. After the refactor that
moved the harness into src/callosum/capability/, this file is a thin
driver: pytest just invokes the same runner the background scheduler
uses, with `call_responses` wired to POST through /admin/cell-call.

If you're adding a new capability dimension, add it under
src/callosum/capability/dimensions/ and register it in
src/callosum/capability/dimensions/__init__.py. The runner picks it
up automatically; no new pytest file is needed.

Run:

  uv run pytest -m model_probe                    # probe every cell
  uv run pytest -m model_probe --probe-cell=NAME  # probe one cell
"""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

import httpx
import pytest

from callosum.capability.runner import run_dimensions


@pytest.mark.model_probe
def test_run_all_dimensions_per_cell(
    callosum_client: httpx.Client, cells_to_probe: list[str]
) -> None:
    """Iterate every advertised cell, running every registered dimension
    against it. Findings persist as side effects; the test passes as
    long as the runner completes without throwing.

    The "test passes if it completes" contract is intentional: the
    capability harness is OBSERVATIONAL — it produces structured
    findings that downstream consumers act on. A cell that fails
    a dimension is not a test failure, it's information.
    """
    for cell in cells_to_probe:
        served_by: str | None = None

        async def _call_responses(
            body: dict[str, Any], _cell: str = cell
        ) -> dict[str, Any]:
            """Closure over httpx client and the current cell. Posts
            to /admin/cell-call (which bypasses the router) so the
            probe lands on the intended cell deterministically.
            """
            nonlocal served_by
            # Offload the blocking sync-httpx call so we don't pin the
            # event loop while the cell takes minutes to respond.
            r = await asyncio.to_thread(
                partial(
                    callosum_client.post,
                    "/admin/cell-call",
                    json={"model": _cell, "body": body},
                )
            )
            r.raise_for_status()
            outcome = r.json()
            served_by = outcome.get("served_by") or served_by
            if outcome.get("status") != "ok":
                raise RuntimeError(
                    f"cell-call failed: {outcome.get('error')}"
                )
            response = outcome.get("response")
            if not isinstance(response, dict):
                raise RuntimeError(
                    "cell-call returned no response body"
                )
            return response

        asyncio.run(
            run_dimensions(
                cell=cell,
                backend_id=None,  # filled in via call_responses side effect
                call_responses=_call_responses,
            )
        )
        # We could thread served_by through to the profile, but the
        # runner already saves the profile after each dimension with
        # backend_id=None when not provided. Operators who care about
        # the backend_id can read it from request logs.
