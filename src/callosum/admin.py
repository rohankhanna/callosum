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
  GET  /admin/routing          → current routing mode
  POST /admin/routing          → set routing mode
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

from callosum.autonomy import AutonomyLevel, AutonomyStore, PromotionNotReady
from callosum.operator_state import VALID_ROUTING_MODES, OperatorState
from callosum.retention import RetentionRunner, result_to_dict
from callosum.self_assessment import SelfAssessmentRunner


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
    autonomy_store: AutonomyStore | None = None,
    retention_runner: RetentionRunner | None = None,
    self_assessment_runner: SelfAssessmentRunner | None = None,
) -> None:
    """Mount the /admin/* endpoints on `app`, gated by the admin token.

    `backends` is an optional list of loaded backend objects. When
    provided, the /admin/probe-tools endpoint can iterate them to run
    the verification probe against each cell. When None, that endpoint
    returns an empty result (callosum was started without any backends
    or the wiring isn't passing them through yet).

    `autonomy_store` is the earned-autonomy ladder backing
    /admin/autonomy/*. When None, those endpoints return 503; the proxy
    can still operate at effective L1_MANUAL (manual everything) which
    is the safe default behavior when state isn't configured.
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
            "routing": operator_state.get_routing(),
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

    @router.get("/routing")
    async def admin_routing_get(request: Request) -> dict[str, str]:
        _check(request)
        return {"routing": operator_state.get_routing()}

    @router.post("/routing")
    async def admin_routing_set(request: Request) -> dict[str, str]:
        _check(request)
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "expected JSON object")
        routing = body.get("routing")
        if not isinstance(routing, str) or routing not in VALID_ROUTING_MODES:
            raise HTTPException(
                400,
                f"routing must be one of {sorted(VALID_ROUTING_MODES)}",
            )
        operator_state.set_routing(routing)
        return {"status": "set", "routing": routing}

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

    # ---------- autonomy ladder (Tier A) ---------------------------------

    def _autonomy_state_dict(store: AutonomyStore) -> dict[str, Any]:
        state = store.get_state()
        eligible, why = state.is_eligible_for_promotion()
        return {
            "current_level": int(state.current_level),
            "current_level_name": state.current_level.name,
            "last_changed_at": state.last_changed_at,
            "last_changed_reason": state.last_changed_reason,
            "ops_at_current_level": state.ops_at_current_level,
            "clean_streak": state.clean_streak,
            "promotion_threshold_k": state.promotion_threshold_k,
            "promotion_eligible": eligible,
            "promotion_eligibility_reason": why,
        }

    def _require_autonomy() -> AutonomyStore:
        if autonomy_store is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "autonomy store not configured; effective L1_MANUAL",
            )
        return autonomy_store

    @router.get("/autonomy")
    async def admin_autonomy_show(request: Request) -> dict[str, Any]:
        _check(request)
        return _autonomy_state_dict(_require_autonomy())

    @router.post("/autonomy/promote")
    async def admin_autonomy_promote(request: Request) -> dict[str, Any]:
        _check(request)
        store = _require_autonomy()
        try:
            store.promote(actor="user")
        except PromotionNotReady as exc:
            raise HTTPException(409, str(exc)) from exc
        return _autonomy_state_dict(store)

    @router.post("/autonomy/demote")
    async def admin_autonomy_demote(request: Request) -> dict[str, Any]:
        _check(request)
        store = _require_autonomy()
        body: dict[str, Any] = {}
        try:
            body = await request.json()
        except Exception:
            body = {}
        reason = body.get("reason") if isinstance(body, dict) else None
        reason_str = (
            str(reason) if isinstance(reason, str) and reason
            else "operator demote"
        )
        store.demote(reason=reason_str, actor="user")
        return _autonomy_state_dict(store)

    @router.post("/autonomy/set")
    async def admin_autonomy_set(request: Request) -> dict[str, Any]:
        _check(request)
        store = _require_autonomy()
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "expected JSON object")
        level_raw = body.get("level")
        if not isinstance(level_raw, int):
            raise HTTPException(
                400, "level required: integer 1..5"
            )
        try:
            level = AutonomyLevel(level_raw)
        except ValueError as exc:
            raise HTTPException(
                400, f"invalid level {level_raw}: {exc}"
            ) from exc
        reason = body.get("reason")
        reason_str = (
            str(reason) if isinstance(reason, str) and reason
            else "operator set"
        )
        store.set_level(level, reason=reason_str, actor="user")
        return _autonomy_state_dict(store)

    @router.get("/autonomy/history")
    async def admin_autonomy_history(request: Request) -> list[dict[str, Any]]:
        _check(request)
        store = _require_autonomy()
        return [
            {
                "ts": t.ts,
                "from_level": int(t.from_level),
                "from_level_name": t.from_level.name,
                "to_level": int(t.to_level),
                "to_level_name": t.to_level.name,
                "reason": t.reason,
                "actor": t.actor,
            }
            for t in store.history(limit=200)
        ]

    @router.get("/autonomy/audit")
    async def admin_autonomy_audit(request: Request) -> list[dict[str, Any]]:
        _check(request)
        store = _require_autonomy()
        return [
            {
                "ts": e.ts,
                "action": e.action,
                "level_at_time": int(e.level_at_time),
                "level_name": e.level_at_time.name,
                "outcome": e.outcome,
                "signal": e.signal,
                "details": e.details,
            }
            for e in store.audit_log(limit=200)
        ]

    # ---------- retention (Tier G) ---------------------------------------

    def _require_retention() -> RetentionRunner:
        if retention_runner is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "retention runner not configured",
            )
        return retention_runner

    @router.get("/retention")
    async def admin_retention_show(request: Request) -> dict[str, Any]:
        _check(request)
        runner = _require_retention()
        return {
            "archive_dir": str(runner.archive_dir),
            "policies": [
                {
                    "name": p.name,
                    "description": p.description,
                    "kind": p.kind.value,
                    "delete_after_days": p.delete_after_days,
                    "archive_on_delete": p.archive_on_delete,
                    "params": p.params,
                }
                for p in runner.policies()
            ],
        }

    @router.get("/retention/status")
    async def admin_retention_status(request: Request) -> list[dict[str, Any]]:
        _check(request)
        return _require_retention().status()

    @router.post("/retention/preview")
    async def admin_retention_preview(request: Request) -> list[dict[str, Any]]:
        _check(request)
        return [result_to_dict(r) for r in _require_retention().preview()]

    @router.post("/retention/run")
    async def admin_retention_run(request: Request) -> list[dict[str, Any]]:
        _check(request)
        return [result_to_dict(r) for r in _require_retention().run()]

    # ---------- self-assessment (Tier C) ---------------------------------

    def _require_self_assessment() -> SelfAssessmentRunner:
        if self_assessment_runner is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "self-assessment runner not configured",
            )
        return self_assessment_runner

    @router.get("/self-assessment/history")
    async def admin_self_assessment_history(
        request: Request,
    ) -> list[dict[str, Any]]:
        _check(request)
        store = _require_autonomy()
        return [
            {
                "id": r.id,
                "ts": r.ts,
                "window_start_ts": r.window_start_ts,
                "window_end_ts": r.window_end_ts,
                "metrics": r.metrics,
                "decision": r.decision,
                "notes": r.notes,
            }
            for r in store.list_self_assessments(limit=200)
        ]

    @router.post("/self-assessment/preview")
    async def admin_self_assessment_preview(
        request: Request,
    ) -> dict[str, Any]:
        """Dry-run: compute metrics + decision without persisting or
        firing demote. Useful for the operator to inspect what the
        next real run would do."""
        _check(request)
        runner = _require_self_assessment()
        metrics, decision = runner.run(dry_run=True)
        return {
            "metrics": _dataclass_to_dict(metrics),
            "decision": _dataclass_to_dict(decision),
        }

    @router.post("/self-assessment/run")
    async def admin_self_assessment_run(request: Request) -> dict[str, Any]:
        """Execute one self-assessment cycle: persist row, demote on
        bad signal, emit feedback artifact. Returns the metrics +
        decision so the cron wrapper logs them."""
        _check(request)
        runner = _require_self_assessment()
        metrics, decision = runner.run(dry_run=False)
        return {
            "metrics": _dataclass_to_dict(metrics),
            "decision": _dataclass_to_dict(decision),
        }

    app.include_router(router)


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Lightweight dataclass-to-dict that's safe for JSON serialization
    of the AssessmentMetrics / AssessmentDecision types (which contain
    only JSON-friendly primitives)."""
    from dataclasses import asdict, is_dataclass
    # is_dataclass() is True for both instances AND classes; asdict()
    # requires an instance. Narrow explicitly so callers passing a class
    # by mistake hit the dict() fallback instead of an asdict TypeError.
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return dict(obj)
