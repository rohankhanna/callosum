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
# ollama-served local cells, future vLLM cells, etc. Ollama defaults
# `think: false` because the thinking-mode behavior of model-a0d5-class
# models silently consumes inference budget without producing visible
# output — bad for agentic workloads where every token matters. An
# operator who wants thinking on for a specific model can override
# via the CLI.
BACKEND_DEFAULT_INFERENCE_PARAMS: dict[str, dict[str, Any]] = {
    "litellm_gateway": {"think": False},
}


_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS inference_overrides (
        model TEXT PRIMARY KEY,
        params_json TEXT NOT NULL,
        force INTEGER NOT NULL DEFAULT 0,
        updated_at REAL NOT NULL
    )
    """,
]


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
        if k in operator_overrides:
            if operator_force or k not in out:
                out[k] = operator_overrides[k]
        if k in backend_defaults and k not in out:
            out[k] = backend_defaults[k]
    return out
