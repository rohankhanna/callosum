"""Admin HTTP surface — loopback-only operator endpoints.

The callosum CLI talks to these endpoints to read and modify
OperatorState (inference overrides, denylist, mode) without restarting
the proxy. Endpoints are gated by an admin token stored in
~/.config/callosum/admin_token (generated on first proxy start if
absent), so anyone with loopback access + filesystem read on the
operator's home directory can call them — same trust boundary as the
existing auth.sqlite.

Endpoints (all under /admin):

  GET  /admin/status           → consolidated operator-state snapshot
  GET  /admin/params           → list inference overrides
  POST /admin/params           → set/clear inference override for one model
  GET  /admin/denylist         → list denied cells
  POST /admin/denylist         → add/remove a denied cell
  GET  /admin/mode             → current mode
  POST /admin/mode             → set mode

Designed for the CLI; not intended for browser / human consumption.
JSON in, JSON out.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request, status

from callosum.operator_state import VALID_OPERATOR_MODES, OperatorState


def _admin_token_path() -> Path:
    """Where the admin token is persisted across restarts."""
    return Path("~/.config/callosum/admin_token").expanduser()


def ensure_admin_token() -> str:
    """Read the existing admin token from disk; if absent, generate one,
    save it with 0600 perms, and return it. Idempotent across restarts —
    the CLI reads the same file."""
    path = _admin_token_path()
    if path.exists():
        token = path.read_text().strip()
        if token:
            return token
    # Generate fresh.
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    path.write_text(token)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return token


def install_admin_routes(app: FastAPI, operator_state: OperatorState) -> None:
    """Mount the /admin/* endpoints on `app`, gated by the admin token."""
    token = ensure_admin_token()
    router = APIRouter(prefix="/admin", tags=["admin"])

    def _check(request: Request) -> None:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "admin token required")
        provided = auth[len("Bearer ") :].strip()
        if not secrets.compare_digest(provided, token):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "admin token invalid")

    @router.get("/status")
    async def admin_status(request: Request) -> dict[str, Any]:
        _check(request)
        return {
            "mode": operator_state.get_mode(),
            "inference_overrides": [
                {"model": m, "params": p, "force": f}
                for m, p, f in operator_state.list_inference_overrides()
            ],
            "denylist": [
                {"model": m, "reason": r}
                for m, r in operator_state.list_denied_cells()
            ],
        }

    @router.get("/params")
    async def admin_params_list(request: Request) -> list[dict[str, Any]]:
        _check(request)
        return [
            {"model": m, "params": p, "force": f}
            for m, p, f in operator_state.list_inference_overrides()
        ]

    @router.post("/params")
    async def admin_params_set(request: Request) -> dict[str, str]:
        _check(request)
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "expected JSON object")
        action = body.get("action")
        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise HTTPException(400, "model required")
        if action == "clear":
            operator_state.clear_inference_overrides(model)
            return {"status": "cleared"}
        if action == "set":
            params = body.get("params")
            force = bool(body.get("force", False))
            if not isinstance(params, dict):
                raise HTTPException(400, "params must be a JSON object")
            operator_state.set_inference_overrides(model, params, force=force)
            return {"status": "set"}
        raise HTTPException(400, f"unknown action {action!r}")

    @router.get("/denylist")
    async def admin_denylist_list(request: Request) -> list[dict[str, Any]]:
        _check(request)
        return [
            {"model": m, "reason": r}
            for m, r in operator_state.list_denied_cells()
        ]

    @router.post("/denylist")
    async def admin_denylist_modify(request: Request) -> dict[str, str]:
        _check(request)
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "expected JSON object")
        action = body.get("action")
        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise HTTPException(400, "model required")
        if action == "add":
            reason = body.get("reason")
            if reason is not None and not isinstance(reason, str):
                raise HTTPException(400, "reason must be string if given")
            operator_state.add_denied_cell(model, reason)
            return {"status": "added"}
        if action == "remove":
            operator_state.remove_denied_cell(model)
            return {"status": "removed"}
        raise HTTPException(400, f"unknown action {action!r}")

    @router.get("/mode")
    async def admin_mode_get(request: Request) -> dict[str, str]:
        _check(request)
        return {"mode": operator_state.get_mode()}

    @router.post("/mode")
    async def admin_mode_set(request: Request) -> dict[str, str]:
        _check(request)
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "expected JSON object")
        mode = body.get("mode")
        if not isinstance(mode, str) or mode not in VALID_OPERATOR_MODES:
            raise HTTPException(
                400,
                f"mode must be one of {sorted(VALID_OPERATOR_MODES)}",
            )
        operator_state.set_mode(mode)
        return {"status": "set", "mode": mode}

    app.include_router(router)
