"""Regression tests for reasoning-effort stamping on the request body.

Root cause (2026-06-18): the routed reasoning effort was written with
body.setdefault("reasoning", {})["effort"] = effort at three dispatch
sites. dict.setdefault returns the EXISTING value when the key is
present, so a client body carrying "reasoning": null (which the Codex
CLI sends on some Responses requests) yielded None and the follow-on
["effort"] = ... raised
TypeError: 'NoneType' object does not support item assignment — an
uncaught HTTP 500 that Codex surfaced as "We're currently experiencing
high demand". The fix routes all three sites through
_stamp_reasoning_effort, which coerces a non-dict reasoning to a
fresh dict first.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from callosum import app as app_module
from callosum.app import _dispatch_nonstream_with_cell_retry, _stamp_reasoning_effort
from callosum.cell_grid import Cell
from callosum.usage_log import UsageLog

# ---------- pure helper -----------------------------------------------------


def test_stamp_reasoning_null_does_not_crash() -> None:
    """The regression case: a present-but-null reasoning must be coerced,
    not item-assigned into (which raised TypeError -> 500)."""
    body: dict[str, Any] = {"model": "m", "reasoning": None}
    _stamp_reasoning_effort(body, "high")
    assert body["reasoning"] == {"effort": "high"}


def test_stamp_reasoning_missing_creates_dict() -> None:
    body: dict[str, Any] = {"model": "m"}
    _stamp_reasoning_effort(body, "medium")
    assert body["reasoning"] == {"effort": "medium"}


def test_stamp_reasoning_preserves_existing_client_object() -> None:
    """A client-provided reasoning dict keeps its other keys; only effort is
    (re)stamped to the routed value."""
    body: dict[str, Any] = {"model": "m", "reasoning": {"summary": "auto", "effort": "low"}}
    _stamp_reasoning_effort(body, "high")
    assert body["reasoning"] == {"summary": "auto", "effort": "high"}


@pytest.mark.parametrize("bad", [[], "high", 3])
def test_stamp_reasoning_non_dict_is_replaced(bad: Any) -> None:
    """Any non-dict reasoning (list/str/int) is replaced rather than crashing."""
    body: dict[str, Any] = {"model": "m", "reasoning": bad}
    _stamp_reasoning_effort(body, "low")
    assert body["reasoning"] == {"effort": "low"}


# ---------- end-to-end through the real crash site --------------------------


def test_cell_retry_dispatch_tolerates_reasoning_null(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """_dispatch_nonstream_with_cell_retry stamps effort at the site that
    used to raise on reasoning: null. Drive it with a null reasoning and
    assert it dispatches cleanly with the routed effort applied."""
    seen: list[dict[str, Any]] = []

    async def _stub(body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        seen.append({"model": body.get("model"), "reasoning": body.get("reasoning")})
        return {"served": body.get("model")}

    monkeypatch.setattr(app_module, "_dispatch_nonstream", _stub)
    log = UsageLog(tmp_path / "u.sqlite")
    body: dict[str, Any] = {"model": "auto", "reasoning": None}
    out = asyncio.run(
        _dispatch_nonstream_with_cell_retry(
            body,
            candidates=(Cell(model="model-a", reasoning_effort="high", context_window=128_000),),
            usage_log=log,
            model="model-a",
            route_name="/v1/responses",
            backends_list=[],
            preferred_id=None,
            session_id=None,
            session_registry=object(),
            call=None,
        )
    )
    assert out == {"served": "model-a"}
    assert seen[0]["reasoning"] == {"effort": "high"}
