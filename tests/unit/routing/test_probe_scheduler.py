"""Tests for the auto-probe scheduler.

Covers the contract `_capabilities_of` relies on:
  * scheduler picks the right cells to probe (local-backend cells only,
    fresh cached results skipped, stale ones re-probed)
  * probe failures are persisted as supports_tools=False
  * the override helper returns False only for failed probes — passed
    probes return None so the caller trusts the backend's own claim
  * probe + persist path is defensive (exceptions never propagate)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from callosum.operator_state import OperatorState
from callosum.routing.probe_scheduler import (
    _cells_needing_probe,
    run_probe_sweep,
    supports_tools_override,
)


def _state(tmp_path: Path) -> OperatorState:
    return OperatorState(tmp_path / "operator_state.sqlite")


class _FakeBackend:
    """Minimal backend stub. Implements the surface the scheduler reads:
    `kind`, `id`, `advertised_models`, and `responses()` (async)."""

    def __init__(
        self,
        *,
        id: str,
        kind: str,
        models: frozenset[str],
        probe_response: dict[str, Any] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.id = id
        self.kind = kind
        self.advertised_models = models
        self._probe_response = probe_response
        self._raises = raises
        self.responses_calls = 0

    async def responses(self, body: dict[str, Any]) -> dict[str, Any]:
        self.responses_calls += 1
        if self._raises is not None:
            raise self._raises
        return self._probe_response or {"output": []}


def test_cells_needing_probe_skips_remote_backends(tmp_path: Path) -> None:
    """codex_auth_vault cells are NOT probed — their capability claims
    come from a curated catalog and are trustworthy."""
    state = _state(tmp_path)
    remote = _FakeBackend(
        id="primary", kind="codex_auth_vault", models=frozenset({"model-a0e8"})
    )
    todo = _cells_needing_probe(
        backends=[remote],
        operator_state=state,
        probe_ttl_s=3600.0,
        now=time.time(),
    )
    assert todo == []


def test_cells_needing_probe_includes_local_cells(tmp_path: Path) -> None:
    state = _state(tmp_path)
    local = _FakeBackend(
        id="local",
        kind="litellm_gateway",
        models=frozenset({"model-a0b0", "model-a0a9"}),
    )
    todo = _cells_needing_probe(
        backends=[local],
        operator_state=state,
        probe_ttl_s=3600.0,
        now=time.time(),
    )
    assert {(b.id, m) for b, m in todo} == {
        ("local", "model-a0b0"),
        ("local", "model-a0a9"),
    }


def test_cells_needing_probe_skips_fresh_cached_results(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.set_probe_result("local", "model-a0d5-x", supports_tools=True)
    local = _FakeBackend(
        id="local", kind="litellm_gateway", models=frozenset({"model-a0d5-x"})
    )
    todo = _cells_needing_probe(
        backends=[local],
        operator_state=state,
        probe_ttl_s=3600.0,
        now=time.time(),
    )
    assert todo == []


def test_cells_needing_probe_re_probes_stale_results(tmp_path: Path) -> None:
    """A cached result older than the TTL must trigger a fresh probe."""
    state = _state(tmp_path)
    state.set_probe_result("local", "model-a0d5-x", supports_tools=True)
    local = _FakeBackend(
        id="local", kind="litellm_gateway", models=frozenset({"model-a0d5-x"})
    )
    far_future = time.time() + 100 * 24 * 3600
    todo = _cells_needing_probe(
        backends=[local],
        operator_state=state,
        probe_ttl_s=24 * 3600.0,
        now=far_future,
    )
    assert [(b.id, m) for b, m in todo] == [("local", "model-a0d5-x")]


def test_cells_needing_probe_dedupes_when_two_backends_advertise_same_model(
    tmp_path: Path,
) -> None:
    """Two LiteLLMGatewayBackend instances can both advertise the same
    cell name (this is exactly the situation that hid the model-a0d5 probe
    miss in production). Dedupe by (backend_id, model) so we don't
    waste probes."""
    state = _state(tmp_path)
    b1 = _FakeBackend(
        id="local LLM gateway", kind="litellm_gateway", models=frozenset({"shared-model"})
    )
    b2 = _FakeBackend(
        id="local LLM gateway", kind="litellm_gateway", models=frozenset({"shared-model"})
    )
    todo = _cells_needing_probe(
        backends=[b1, b2],
        operator_state=state,
        probe_ttl_s=3600.0,
        now=time.time(),
    )
    assert len(todo) == 1  # deduped by (backend_id, model)


@pytest.mark.asyncio
async def test_run_probe_sweep_passes_when_response_has_structured_tool_call(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    local = _FakeBackend(
        id="local",
        kind="litellm_gateway",
        models=frozenset({"good-model"}),
        probe_response={
            "output": [
                {
                    "type": "function_call",
                    "name": "probe_echo",
                    "arguments": '{"payload":"x"}',
                }
            ]
        },
    )
    probed = await run_probe_sweep(backends=[local], operator_state=state)
    assert probed == 1
    cached = state.get_probe_result("local", "good-model")
    assert cached is not None
    assert cached[0] is True  # supports_tools


@pytest.mark.asyncio
async def test_run_probe_sweep_fails_when_response_has_text_only(
    tmp_path: Path,
) -> None:
    """The model-a0d5-shape failure: model emits a text message, no
    structured function_call. The sweep must record supports_tools=False
    so the router excludes the cell from tool-using traffic."""
    state = _state(tmp_path)
    local = _FakeBackend(
        id="local",
        kind="litellm_gateway",
        models=frozenset({"bad-model"}),
        probe_response={
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "I'd be happy to..."}],
                }
            ]
        },
    )
    await run_probe_sweep(backends=[local], operator_state=state)
    cached = state.get_probe_result("local", "bad-model")
    assert cached is not None
    assert cached[0] is False


@pytest.mark.asyncio
async def test_run_probe_sweep_handles_backend_exception(tmp_path: Path) -> None:
    """A backend raising mid-probe must NOT crash the sweep — the
    probe is recorded as failed and the next cell continues."""
    state = _state(tmp_path)
    boom = _FakeBackend(
        id="local",
        kind="litellm_gateway",
        models=frozenset({"unreachable"}),
        raises=RuntimeError("connection refused"),
    )
    probed = await run_probe_sweep(backends=[boom], operator_state=state)
    assert probed == 1
    cached = state.get_probe_result("local", "unreachable")
    assert cached is not None
    assert cached[0] is False
    # The error reason should be persisted for the operator to see.
    rows = state.list_probe_results()
    assert rows[0][4] is not None  # error column populated
    assert "RuntimeError" in rows[0][4]


def test_supports_tools_override_returns_false_only_for_failed_probes(
    tmp_path: Path,
) -> None:
    """Asymmetric override: probe-pass returns None (don't reach UP),
    probe-fail returns False (reach DOWN to revoke false claim)."""
    state = _state(tmp_path)
    state.set_probe_result("local", "good", supports_tools=True)
    state.set_probe_result("local", "bad", supports_tools=False)
    assert supports_tools_override(
        operator_state=state, backend_id="local", model="good"
    ) is None
    assert supports_tools_override(
        operator_state=state, backend_id="local", model="bad"
    ) is False


def test_supports_tools_override_returns_none_for_unprobed_cells(
    tmp_path: Path,
) -> None:
    """Never-probed cells get None so the caller trusts the backend's
    own claim. The override only fires when there's evidence."""
    state = _state(tmp_path)
    assert supports_tools_override(
        operator_state=state, backend_id="local", model="never-probed"
    ) is None
