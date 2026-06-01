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
  POST /admin/probe-tools      → run tool-call probe against advertised cells

Designed for the CLI; not intended for browser / human consumption.
JSON in, JSON out.
"""

from __future__ import annotations

import contextlib
import secrets
import time
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
    with contextlib.suppress(OSError):
        path.chmod(0o600)
    return token


def install_admin_routes(
    app: FastAPI,
    operator_state: OperatorState,
    *,
    backends: list[Any] | None = None,
) -> None:
    """Mount the /admin/* endpoints on `app`, gated by the admin token.

    `backends` is an optional list of loaded backend objects. When
    provided, the /admin/probe-tools endpoint can iterate them to run
    the verification probe against each cell. When None, that endpoint
    returns an empty result (callosum was started without any backends
    or the wiring isn't passing them through yet).
    """
    token = ensure_admin_token()
    router = APIRouter(prefix="/admin", tags=["admin"])
    backends_list: list[Any] = list(backends) if backends else []

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

    @router.post("/probe-tools")
    async def admin_probe_tools(request: Request) -> dict[str, Any]:
        """Run the tool-call verification probe against every advertised
        cell across the configured backends. Returns per-cell pass/fail
        plus timing and any error encountered.

        Optional JSON body: `{"models": ["name1", "name2", ...]}` to
        restrict probing to a subset. Default probes every model the
        loaded backends advertise.

        This is intentionally synchronous from the CLI's perspective:
        the endpoint awaits all probes before returning. For a pool of
        ~5-10 cells with 26B-class local models, expect 30s-3min total.
        Future work could surface streaming progress via SSE; for now
        the operator runs `callosum-ctl probe-tools` and waits.
        """
        _check(request)
        from callosum.routing.probe import probe_supports_tools

        body: dict[str, Any] = {}
        try:
            body = await request.json()
        except Exception:
            # Empty / non-JSON body = "probe everything" — friendlier
            # than 400'ing on a body that's allowed to be absent.
            body = {}
        wanted = body.get("models") if isinstance(body, dict) else None
        wanted_set: set[str] | None = (
            set(wanted) if isinstance(wanted, list) else None
        )

        results: list[dict[str, Any]] = []
        if not backends_list:
            return {
                "results": results,
                "note": (
                    "no backends wired into the admin surface; restart "
                    "the proxy to ensure backends are passed through"
                ),
            }
        for backend in backends_list:
            advertised: frozenset[str] = getattr(
                backend, "advertised_models", frozenset()
            )
            for model in sorted(advertised):
                if wanted_set is not None and model not in wanted_set:
                    continue
                if not hasattr(backend, "responses"):
                    # Backend doesn't expose the non-stream responses
                    # path the probe needs — skip rather than fail.
                    continue

                async def _call(
                    probe_body: dict[str, Any],
                    _b: Any = backend,
                ) -> dict[str, Any]:
                    # Closure binds the current backend so each probe
                    # hits the right one. The probe-supports-tools
                    # function signature is decoupled from backend
                    # internals — this thin closure is the seam.
                    result: dict[str, Any] = await _b.responses(probe_body)
                    return result

                t0 = time.time()
                error_msg: str | None = None
                supports: bool = False
                try:
                    supports = await probe_supports_tools(
                        model=model, call_responses=_call
                    )
                except Exception as exc:
                    error_msg = f"{type(exc).__name__}: {exc}"
                latency_ms = int((time.time() - t0) * 1000)
                results.append({
                    "model": model,
                    "backend": getattr(backend, "id", "?"),
                    "supports_tools": supports,
                    "latency_ms": latency_ms,
                    "error": error_msg,
                })
        return {"results": results}

    @router.post("/cell-call")
    async def admin_cell_call(request: Request) -> dict[str, Any]:
        """Send an arbitrary Responses-API body to a specific cell,
        bypassing the router. Returns the upstream response (or error)
        along with timing.

        Designed for the model-capability test harness: tests build
        a Codex-shape request body, target one cell explicitly, and
        record findings about how that cell responded. Without this
        endpoint, tests would have to mutate operator_state's denylist
        to force routing to a specific cell — destructive, slow, racy.

        Request body shape:
          {
            "model": "<cell-model-name>",
            "body": <full Responses-API request object>,
            "timeout_s": <optional float>
          }

        Response shape:
          {
            "status": "ok" | "error",
            "served_by": "<backend-id>",
            "latency_ms": int,
            "response": <upstream response dict on success>,
            "error": "<error class + message on failure>"
          }

        This endpoint is admin-token-gated (same as the rest of /admin/*)
        because it bypasses every other guardrail the router applies —
        denylist, mode filter, capability filter, all skipped. Operator
        intent is "I know what I'm doing, just hit this cell."
        """
        _check(request)
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(400, "expected JSON object")
        model = payload.get("model")
        body = payload.get("body")
        if not isinstance(model, str) or not model:
            raise HTTPException(400, "`model` (string) is required")
        if not isinstance(body, dict):
            raise HTTPException(400, "`body` (object) is required")

        target_backend: Any = None
        for backend in backends_list:
            advertised: frozenset[str] = getattr(
                backend, "advertised_models", frozenset()
            )
            if model in advertised and hasattr(backend, "responses"):
                target_backend = backend
                break
        if target_backend is None:
            raise HTTPException(
                404,
                f"no backend advertises cell {model!r} with a non-stream "
                "responses() entry point",
            )

        # Force the model field in the body so the test harness can
        # send a body with model="" or a different name and still
        # route correctly to the target cell.
        body = dict(body)
        body["model"] = model

        t0 = time.time()
        try:
            response = await target_backend.responses(body)
            latency_ms = int((time.time() - t0) * 1000)
            return {
                "status": "ok",
                "served_by": getattr(target_backend, "id", "?"),
                "latency_ms": latency_ms,
                "response": response,
                "error": None,
            }
        except Exception as exc:
            latency_ms = int((time.time() - t0) * 1000)
            return {
                "status": "error",
                "served_by": getattr(target_backend, "id", "?"),
                "latency_ms": latency_ms,
                "response": None,
                "error": f"{type(exc).__name__}: {exc}",
            }

    app.include_router(router)
