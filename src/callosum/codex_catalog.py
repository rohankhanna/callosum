"""Project the canonical Callosum routing catalog into a codex `/model` picker.

Background
----------
codex's interactive `/model` picker is *not* fed by the provider's
`/v1/models`. It is driven by a model catalog that codex resolves at startup,
which an operator can fully override from a single `~/.codex/config.toml` via:

    model_catalog_json = "<absolute path to a {\"models\": [...]} file>"

Each entry is a full codex ModelInfo (base_instructions is required).
codex debug models renders the resolved catalog as JSON — exactly what the
picker consumes — so the mechanism is verifiable without a TUI.

This module is the "background catalog reconciler that projects the catalog
into the client picker store" — the previously-unbuilt half of work tracker
````. It re-emits the codex catalog file from the live
Callosum catalog (the same ids `/v1/models` serves) on startup, whenever the
catalog hash changes, and on a periodic safety interval.

Design notes
------------
- The per-entry base_instructions (and the rest of the rich ModelInfo
  shape) are sourced live from codex debug models run under a disposable
  CODEX_HOME — that yields codex's *bundled default* catalog (offline, no
  auth) without inheriting our own model_catalog_json override (which would
  be circular). Nothing proprietary is committed to the repo: the template is
  read from the installed codex binary at emit time.
- If the template cannot be sourced (codex missing/unhealthy), the reconciler
  logs and SKIPS the write, leaving any existing file untouched, rather than
  emitting a catalog codex would reject.
- Raw passthrough ids (non-callosum: slugs) are intentionally excluded from
  the picker; they remain resolvable as a model string but should not clutter
  the menu. Strategy selectors and concrete pins are shown.
- Operator-declared "aspirational" lanes (declared_lanes) are merged in even
  when no backend serves them yet, so the menu can list not-yet-live lanes.
  Selecting one resolves through `selectors.py` and surfaces a clean
  "not available yet" message at dispatch (see app.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from callosum.cell_grid import REASONING_LEVELS
from callosum.selectors import SelectorError, is_selector, parse_selector

logger = logging.getLogger("callosum.codex_catalog")

#: Strategy selectors always lead the menu, in this fixed order.
_STRATEGY_ORDER: tuple[str, ...] = (
    "callosum:auto",
    "callosum:remote-only",
    "callosum:local-only",
)

_STRATEGY_LABELS: dict[str, tuple[str, str]] = {
    "callosum:auto": (
        "Callosum: Auto",
        "Router selects a local or remote backend per request.",
    ),
    "callosum:remote-only": (
        "Callosum: Remote only",
        "Force remote backends for this session (no global mode change).",
    ),
    "callosum:local-only": (
        "Callosum: Local only",
        "Force local backends for this session (no global mode change).",
    ),
}


def lane_metadata(model_id: str) -> tuple[str, str, int] | None:
    """Return (display_name, description, priority) for a Callosum id, or
    None if the id should not appear in the picker.

    Lower priority sorts earlier in codex's picker. Strategy selectors lead,
    then concrete remote pins, then local pins.
    """
    if model_id in _STRATEGY_LABELS:
        name, desc = _STRATEGY_LABELS[model_id]
        return name, desc, _STRATEGY_ORDER.index(model_id)
    try:
        sel = parse_selector(model_id)
    except SelectorError:
        return None
    if sel is None or sel.pinned_model is None:
        # Not a concrete pin (raw passthrough id, or unknown) → keep it out of
        # the menu.
        return None
    if sel.source == "remote":
        effort = sel.pinned_effort
        suffix = f" · {effort}" if effort else ""
        name = f"Callosum remote · {sel.pinned_model}{suffix}"
        desc = (
            f"Pin the remote model {sel.pinned_model}"
            + (f" at {effort} reasoning." if effort else ".")
        )
        return name, desc, 10
    # source == "local"
    name = f"Callosum local · {sel.pinned_model}"
    desc = f"Pin the local model {sel.pinned_model}."
    return name, desc, 20


def _pinned_effort(model_id: str) -> str | None:
    """The reasoning effort baked into a remote pin id, if any."""
    with contextlib.suppress(SelectorError):
        sel = parse_selector(model_id)
        if sel is not None:
            return sel.pinned_effort
    return None


def _reasoning_levels_for_lane(
    model_id: str, template_levels: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Reasoning-effort presets to advertise for a lane.

    A concrete remote pin bakes its effort into the id, so the picker should
    offer only that effort (codex's separately-sent reasoning_effort is
    overridden by the selector anyway). Every other lane inherits the template's
    full set.
    """
    effort = _pinned_effort(model_id)
    if effort is None:
        return template_levels
    matching = [lvl for lvl in template_levels if lvl.get("effort") == effort]
    if matching:
        return matching
    # Effort is a valid REASONING_LEVELS value the template didn't enumerate;
    # synthesize a minimal preset so the picker still shows it.
    if effort in REASONING_LEVELS:
        return [{"effort": effort, "description": effort}]
    return template_levels


def _ordered_lane_ids(
    model_ids: Sequence[str], declared_lanes: Sequence[str]
) -> list[str]:
    """Stable, de-duplicated menu order: strategy selectors first (fixed
    order), then live concrete pins (sorted), then operator-declared lanes not
    already present (sorted). Non-selector / non-pin ids are dropped.
    """
    live = {m for m in model_ids if is_selector(m)}
    declared = {d for d in declared_lanes if is_selector(d)}
    seen: set[str] = set()
    ordered: list[str] = []

    def _add(mid: str) -> None:
        if mid in seen:
            return
        if lane_metadata(mid) is None:
            return
        seen.add(mid)
        ordered.append(mid)

    for mid in _STRATEGY_ORDER:
        if mid in live or mid in declared:
            _add(mid)
    for mid in sorted(live):
        _add(mid)
    for mid in sorted(declared):
        _add(mid)
    return ordered


def build_codex_catalog(
    *,
    model_ids: Sequence[str],
    declared_lanes: Sequence[str],
    template: dict[str, Any],
) -> dict[str, Any]:
    """Build a codex model_catalog_json document from Callosum lane ids.

    template is one resolved codex ModelInfo (from load_codex_template);
    every lane inherits its rich fields (base_instructions, context window,
    tool config, ...) and re-stamps only the identity fields.
    """
    template_levels = list(template.get("supported_reasoning_levels") or [])
    models: list[dict[str, Any]] = []
    for mid in _ordered_lane_ids(model_ids, declared_lanes):
        meta = lane_metadata(mid)
        if meta is None:  # pragma: no cover - filtered by _ordered_lane_ids
            continue
        name, desc, priority = meta
        entry = dict(template)  # shallow copy is enough; we overwrite scalars
        entry["slug"] = mid
        entry["display_name"] = name
        entry["description"] = desc
        entry["priority"] = priority
        entry["visibility"] = "list"
        entry["availability_nux"] = None
        entry["upgrade"] = None
        levels = _reasoning_levels_for_lane(mid, template_levels)
        entry["supported_reasoning_levels"] = levels
        # Keep the default within the offered set so the picker has a valid
        # default selection.
        if levels and entry.get("default_reasoning_level") not in {
            lvl.get("effort") for lvl in levels
        }:
            entry["default_reasoning_level"] = levels[0].get("effort")
        models.append(entry)
    return {"models": models}


def load_codex_template(*, codex_bin: str = "codex") -> dict[str, Any] | None:
    """Source one rich codex ModelInfo to use as the lane template.

    Runs codex debug models under a disposable CODEX_HOME so it returns
    codex's bundled default catalog (offline, no auth) rather than our own
    model_catalog_json override. Returns the highest-priority entry, or
    None if the catalog could not be sourced.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="callosum-codex-home-") as home:
            env = dict(os.environ)
            env["CODEX_HOME"] = home
            proc = subprocess.run(
                [codex_bin, "debug", "models"],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
                check=False,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("codex catalog: could not run %r: %s", codex_bin, exc)
        return None
    if proc.returncode != 0:
        logger.warning(
            "codex catalog: %r exited %d: %s",
            codex_bin,
            proc.returncode,
            (proc.stderr or "").strip()[:300],
        )
        return None
    try:
        catalog = json.loads(proc.stdout)
        models = catalog["models"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("codex catalog: unparseable `codex debug models`: %s", exc)
        return None
    if not models:
        logger.warning("codex catalog: `codex debug models` returned no models")
        return None
    # Lowest priority value == codex's default/headline model: the richest,
    # most representative ModelInfo to clone.
    template = min(models, key=lambda m: m.get("priority", 1_000_000))
    if "base_instructions" not in template:
        logger.warning("codex catalog: template model missing base_instructions")
        return None
    return template


def catalog_digest(catalog: dict[str, Any]) -> str:
    """Stable content hash of a catalog document (order-independent)."""
    payload = json.dumps(catalog, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_catalog_atomic(path: Path, catalog: dict[str, Any]) -> None:
    """Write catalog to path atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


class CodexCatalogReconciler:
    """Background task that keeps a codex model_catalog_json file in sync
    with the live Callosum catalog.

    Mirrors the _PeriodicSmokeTester lifecycle (start() / await
    stop()). The first reconcile runs immediately so the file is fresh as
    soon as the proxy is up; subsequent reconciles run when the catalog hash
    changes or the safety interval elapses, whichever comes first.
    """

    def __init__(
        self,
        *,
        output_path: Path,
        model_ids_fn: Callable[[], Sequence[str]],
        declared_lanes: Sequence[str] = (),
        codex_bin: str = "codex",
        refresh_interval_s: int = 1800,
    ) -> None:
        self._output_path = output_path
        self._model_ids_fn = model_ids_fn
        self._declared_lanes = list(declared_lanes)
        self._codex_bin = codex_bin
        self._refresh_interval_s = refresh_interval_s
        self._last_digest: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    def reconcile_once(self) -> bool:
        """Re-emit the catalog file if the live catalog changed. Returns True
        if a write happened. Safe to call off-loop; does its own I/O.
        """
        try:
            model_ids = list(self._model_ids_fn())
        except Exception:
            logger.exception("codex catalog: failed to read live model ids")
            return False
        template = load_codex_template(codex_bin=self._codex_bin)
        if template is None:
            # Already logged; leave any existing file untouched.
            return False
        catalog = build_codex_catalog(
            model_ids=model_ids,
            declared_lanes=self._declared_lanes,
            template=template,
        )
        digest = catalog_digest(catalog)
        if digest == self._last_digest and self._output_path.exists():
            return False
        try:
            write_catalog_atomic(self._output_path, catalog)
        except OSError:
            logger.exception(
                "codex catalog: failed to write %s", self._output_path
            )
            return False
        self._last_digest = digest
        logger.info(
            "codex catalog: wrote %d lane(s) to %s",
            len(catalog["models"]),
            self._output_path,
        )
        return True

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name="codex-catalog-reconciler")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        # Immediate first emit (template sourcing + write run in a worker
        # thread so the subprocess never blocks the event loop).
        with contextlib.suppress(Exception):
            await asyncio.to_thread(self.reconcile_once)
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self._refresh_interval_s
                )
            except TimeoutError:
                pass
            else:
                return
            try:
                await asyncio.to_thread(self.reconcile_once)
            except Exception:
                logger.exception("codex catalog: periodic reconcile failed")
