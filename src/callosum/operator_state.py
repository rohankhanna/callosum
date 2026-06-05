"""Operator-managed runtime state.

Persists operator decisions that should survive proxy restart and be
changeable without editing config.toml — denylists, per-cell inference
parameter overrides, mode toggles. Backed by its own SQLite database
so that frequent CLI writes don't contend with the read-heavy auth or
request-log databases.

Today (step 1 of the per-cell inference-params work) only the
`inference_overrides` table is populated. Future steps add:
  * cell_denylist
  * mode (online / offline / local-only / remote-only)
  * priority_overrides

The CLI (step 4) drives all writes; reads happen on the hot path
through the backends.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# Per-backend baseline parameters applied when nothing is operator-set.
# Keyed by the backend's `kind` so the same merge logic works for
# ollama-served local cells, future vLLM cells, etc.
#
# Intentionally empty by default. Earlier iterations defaulted
# `think: false` for ollama-served cells to defend against thinking-mode
# models monopolizing inference budget — but that hid valuable
# reasoning content the operator could otherwise see and act on.
#
# The actual fix for the "stall" symptom is the stream-through
# translator: thinking content now streams live as
# `response.reasoning_summary_text.delta` events, so the operator can
# WATCH the model think and Ctrl-C if it's looping. Cancellation
# propagates through httpx to ollama and stops the runner.
#
# Operators who STILL want to disable thinking for a specific cell can
# set it explicitly via the (future) CLI:
#     callosum params set <model> think=false
BACKEND_DEFAULT_INFERENCE_PARAMS: dict[str, dict[str, Any]] = {}


_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS inference_overrides (
        model TEXT PRIMARY KEY,
        params_json TEXT NOT NULL,
        force INTEGER NOT NULL DEFAULT 0,
        updated_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cell_denylist (
        model TEXT PRIMARY KEY,
        reason TEXT,
        added_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS operator_mode (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        mode TEXT NOT NULL,
        updated_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS probe_results (
        backend_id TEXT NOT NULL,
        model TEXT NOT NULL,
        probed_at REAL NOT NULL,
        supports_tools INTEGER NOT NULL,
        error TEXT,
        latency_ms INTEGER,
        PRIMARY KEY (backend_id, model)
    )
    """,
]

# Allowed values for the single-row operator_mode table. `auto` is the
# default — Router considers all routable backends. The others give the
# operator a kill-switch when they want to constrain routing without
# editing config.toml:
#   * offline:     ignore remote backends entirely (force local-only)
#   * local-only:  same as offline but explicit semantics
#   * remote-only: ignore local backends (e.g. while debugging a local
#                  serving stack)
#
# The CONCEPT was originally called "operator mode" — the name is
# preserved on the internal SQLite table because renaming requires
# a migration. The user-facing surface (CLI subcommand, admin endpoint,
# Python API) is now "routing" to disambiguate from the unrelated
# `callosum-ctl autonomy` ladder. See the project memory
# `the project notes` for the rename rationale.
VALID_ROUTING_MODES = frozenset({"auto", "offline", "local-only", "remote-only"})


class OperatorState:
    """SQLite-backed runtime state for operator decisions.

    Thread-safe via a single lock around all writes; reads are cheap
    SELECTs that take the lock briefly. The database is created on
    first construction; subsequent runs reuse it.
    """

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._lock = threading.Lock()
        with self._lock:
            for stmt in _SCHEMA:
                self._conn.execute(stmt)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------- inference overrides --------------------------------------

    def get_inference_overrides(self, model: str) -> tuple[dict[str, Any], bool]:
        """Return (params, force) for `model`. Empty dict + False when no
        override exists — caller then applies backend defaults only."""
        with self._lock:
            row = self._conn.execute(
                "SELECT params_json, force FROM inference_overrides WHERE model = ?",
                (model,),
            ).fetchone()
        if row is None:
            return ({}, False)
        try:
            params = json.loads(row[0])
            if not isinstance(params, dict):
                return ({}, False)
        except (TypeError, ValueError, json.JSONDecodeError):
            return ({}, False)
        return (params, bool(row[1]))

    def set_inference_overrides(
        self,
        model: str,
        params: Mapping[str, Any],
        *,
        force: bool = False,
    ) -> None:
        """Upsert an override row. force=True means operator values win
        over client request body for the listed keys; force=False (the
        default) means operator values act as DEFAULTS and yield to
        whatever the client explicitly set."""
        payload = json.dumps(dict(params))
        with self._lock:
            self._conn.execute(
                "INSERT INTO inference_overrides (model, params_json, force, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(model) DO UPDATE SET "
                "params_json=excluded.params_json, force=excluded.force, "
                "updated_at=excluded.updated_at",
                (model, payload, 1 if force else 0, time.time()),
            )
            self._conn.commit()

    def clear_inference_overrides(self, model: str) -> None:
        """Remove any override for `model`. Backend defaults still apply."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM inference_overrides WHERE model = ?", (model,)
            )
            self._conn.commit()

    # ---------- cell denylist --------------------------------------------

    def add_denied_cell(self, model: str, reason: str | None = None) -> None:
        """Add `model` to the denylist. Future routing decisions exclude
        any cell whose model matches before the capability filter runs."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO cell_denylist (model, reason, added_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(model) DO UPDATE SET "
                "reason=excluded.reason, added_at=excluded.added_at",
                (model, reason, time.time()),
            )
            self._conn.commit()

    def remove_denied_cell(self, model: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM cell_denylist WHERE model = ?", (model,)
            )
            self._conn.commit()

    def list_denied_cells(self) -> list[tuple[str, str | None]]:
        """Return [(model, reason), ...] in insertion order."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT model, reason FROM cell_denylist ORDER BY added_at"
            ).fetchall()
        return [(str(m), r if r is None or isinstance(r, str) else None) for m, r in rows]

    def is_denied(self, model: str) -> bool:
        """Hot-path check: is this model in the denylist? Sub-millisecond
        SELECT — called once per routing decision."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM cell_denylist WHERE model = ? LIMIT 1", (model,)
            ).fetchone()
        return row is not None

    # ---------- routing mode ---------------------------------------------

    def get_routing(self) -> str:
        """Return current routing mode (auto / offline / local-only /
        remote-only). Defaults to 'auto' if never set. The SQLite
        column is still named `mode` for backward compat with existing
        DBs; only the Python API and user-facing surfaces use the
        clearer 'routing' name."""
        with self._lock:
            row = self._conn.execute(
                "SELECT mode FROM operator_mode WHERE id = 1"
            ).fetchone()
        if row is None:
            return "auto"
        routing = str(row[0])
        return routing if routing in VALID_ROUTING_MODES else "auto"

    def set_routing(self, routing: str) -> None:
        """Set the routing mode. Raises ValueError on unknown value."""
        if routing not in VALID_ROUTING_MODES:
            raise ValueError(
                f"unknown routing {routing!r}; "
                f"valid: {sorted(VALID_ROUTING_MODES)}"
            )
        with self._lock:
            self._conn.execute(
                "INSERT INTO operator_mode (id, mode, updated_at) "
                "VALUES (1, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "mode=excluded.mode, updated_at=excluded.updated_at",
                (routing, time.time()),
            )
            self._conn.commit()

    # ---------- probe results --------------------------------------------

    def get_probe_result(
        self, backend_id: str, model: str
    ) -> tuple[bool, float] | None:
        """Hot-path read used by `_capabilities_of` to override
        `supports_tools` for cells that failed the verification probe.

        Returns (supports_tools, probed_at) when a result is cached.
        Returns None when this cell has never been probed — caller
        falls back to the backend's claimed capability.

        Probe results have no TTL enforcement in this read path —
        callers (the probe scheduler) decide when to re-probe by
        consulting `probed_at`.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT supports_tools, probed_at FROM probe_results "
                "WHERE backend_id = ? AND model = ?",
                (backend_id, model),
            ).fetchone()
        if row is None:
            return None
        return (bool(row[0]), float(row[1]))

    def set_probe_result(
        self,
        backend_id: str,
        model: str,
        *,
        supports_tools: bool,
        error: str | None = None,
        latency_ms: int | None = None,
    ) -> None:
        """Persist the outcome of a tool-call verification probe for one
        cell. Called by the probe scheduler after each probe completes.
        Idempotent: re-probing the same cell overwrites the prior row
        with a fresh `probed_at` timestamp.
        """
        with self._lock:
            self._conn.execute(
                "INSERT INTO probe_results "
                "(backend_id, model, probed_at, supports_tools, error, latency_ms) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(backend_id, model) DO UPDATE SET "
                "probed_at=excluded.probed_at, "
                "supports_tools=excluded.supports_tools, "
                "error=excluded.error, "
                "latency_ms=excluded.latency_ms",
                (
                    backend_id,
                    model,
                    time.time(),
                    1 if supports_tools else 0,
                    error,
                    latency_ms,
                ),
            )
            self._conn.commit()

    def list_probe_results(
        self,
    ) -> list[tuple[str, str, float, bool, str | None, int | None]]:
        """Return all probe results as
        (backend_id, model, probed_at, supports_tools, error, latency_ms).
        Used by `callosum-ctl probe-tools list` to show the cached
        view without re-probing.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT backend_id, model, probed_at, supports_tools, error, latency_ms "
                "FROM probe_results ORDER BY backend_id, model"
            ).fetchall()
        return [
            (str(r[0]), str(r[1]), float(r[2]), bool(r[3]),
             r[4] if r[4] is None else str(r[4]),
             r[5] if r[5] is None else int(r[5]))
            for r in rows
        ]

    # ---------- inference overrides (existing) ---------------------------

    def list_inference_overrides(self) -> list[tuple[str, dict[str, Any], bool]]:
        """Return all rows as (model, params, force). Used by the CLI's
        `params list` subcommand once it exists."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT model, params_json, force FROM inference_overrides ORDER BY model"
            ).fetchall()
        out: list[tuple[str, dict[str, Any], bool]] = []
        for model, params_json, force in rows:
            try:
                params = json.loads(params_json)
                if isinstance(params, dict):
                    out.append((model, params, bool(force)))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return out


def merge_inference_params(
    *,
    backend_defaults: Mapping[str, Any],
    operator_overrides: Mapping[str, Any],
    operator_force: bool,
    client_body: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge the three layers of inference parameter sources into the
    final body the backend should forward upstream.

    Layering rules (option (a) — preserve client intent by default):
      1. Start from a copy of the client body.
      2. Fill in operator-override keys ONLY if the client didn't set
         them — unless `operator_force=True`, in which case operator
         values overwrite client values.
      3. Fill in backend defaults ONLY for keys still absent — they
         never overwrite anything explicit.

    The client body is returned unchanged for any non-parameter field
    (`model`, `messages`, `tools`, etc.). Only the inference-param
    keys present in `backend_defaults` or `operator_overrides` get
    touched.
    """
    out: dict[str, Any] = dict(client_body)
    param_keys: set[str] = set(backend_defaults) | set(operator_overrides)
    for k in param_keys:
        if k in operator_overrides and (operator_force or k not in out):
            out[k] = operator_overrides[k]
        if k in backend_defaults and k not in out:
            out[k] = backend_defaults[k]
    return out
