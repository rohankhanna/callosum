"""POST /codex — per-CLI endpoint for codex CLI traffic.

Pins the architectural property: the /codex endpoint shares the same
dispatch core as /v1/responses, but tags requests with
client_endpoint="codex" so codex-specific response transforms can
scope themselves to this endpoint and not bleed into generic clients.

Verification: a transform that fires only on endpoint=="codex" applies
to /codex requests and does NOT apply to /v1/responses requests, even
when the backend, model, and request body are otherwise identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from callosum.app import create_app
from callosum.cell_grid import DEFAULT_MODELS
from callosum.fakes import InMemoryFakeBackend
from callosum.transforms.protocol import (
    TransformBase,
    TransformContext,
)
from callosum.transforms.registry import TransformRegistry


class _CodexOnlyResponseMarker(TransformBase):
    """Test transform: appends a marker to response output_text content,
    but ONLY for requests on the codex endpoint. Lets us verify that
    /codex and /v1/responses dispatch through different transform sets
    without behavior leaking across endpoints."""

    @property
    def name(self) -> str:
        return "test:codex-only-response-marker"

    def applies_to(self, ctx: TransformContext) -> bool:
        return ctx.endpoint == "codex"

    def transform_response(self, body: dict[str, Any], ctx: TransformContext) -> dict[str, Any]:
        # Walk the Responses-API output array, append a marker to any
        # output_text content. Done in-place; safe for a test fake.
        for item in body.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []) or []:
                if isinstance(content, dict) and content.get("type") == "output_text":
                    content["text"] = f"{content.get('text', '')} [codex-marker]"
        return body


def _backend_returning_text(text: str) -> InMemoryFakeBackend:
    """Backend that returns a fixed Responses-API-shaped non-stream
    response. Lets transforms operate on a predictable input."""
    return InMemoryFakeBackend(
        id="fake",
        advertised_models=frozenset(DEFAULT_MODELS),
        canned_responses_response={
            "id": "resp_test",
            "object": "response",
            "model": "model-a0e8",
            "status": "completed",
            "output": [
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
        },
    )


def test_codex_endpoint_applies_codex_only_transforms(tmp_path: Path) -> None:
    """A transform with applies_to(ctx.endpoint=='codex') runs when
    invoked via /codex."""
    registry = TransformRegistry()
    registry.register(_CodexOnlyResponseMarker())
    app = create_app(
        backends=[_backend_returning_text("hello")],
        transform_registry=registry,
    )
    with TestClient(app) as client:
        r = client.post(
            "/codex",
            json={
                "model": "auto-learning",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hi"}],
                    }
                ],
            },
        )
    assert r.status_code == 200, r.text
    text = r.json()["output"][0]["content"][0]["text"]
    assert "[codex-marker]" in text


def test_v1_responses_does_not_apply_codex_transforms(tmp_path: Path) -> None:
    """Endpoint isolation: the same transform registered does NOT fire
    when the request comes in on /v1/responses (generic client),
    because ctx.endpoint is None there, not 'codex'."""
    registry = TransformRegistry()
    registry.register(_CodexOnlyResponseMarker())
    app = create_app(
        backends=[_backend_returning_text("hello")],
        transform_registry=registry,
    )
    with TestClient(app) as client:
        r = client.post(
            "/v1/responses",
            json={
                "model": "auto-learning",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hi"}],
                    }
                ],
            },
        )
    assert r.status_code == 200, r.text
    text = r.json()["output"][0]["content"][0]["text"]
    assert "[codex-marker]" not in text
    assert text == "hello"


def test_codex_endpoint_empty_registry_is_passthrough(tmp_path: Path) -> None:
    """Sanity: with no transforms registered, /codex behaves exactly
    like /v1/responses on the response body. (No regression on the
    common case of 'transforms not configured yet'.)"""
    app = create_app(backends=[_backend_returning_text("ok")])
    with TestClient(app) as client:
        r = client.post(
            "/codex",
            json={
                "model": "auto-learning",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "x"}],
                    }
                ],
            },
        )
    assert r.status_code == 200
    assert r.json()["output"][0]["content"][0]["text"] == "ok"
