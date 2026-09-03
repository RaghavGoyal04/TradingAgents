"""Run manifest lifecycle: the single source of truth for a dashboard run.

The manifest coordinates three independent resume layers (orchestrator batch
results, per-ticker LangGraph SQLite checkpoints, and the UI) under one
schema-versioned document. The UI reads only the manifest and the artifacts it
points to; it never infers run state from partial files.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .contracts import (
    EVENTS_LOG,
    FORECASTS,
    PHASES,
    PORTFOLIO_SNAPSHOT,
    RECOMMENDATIONS,
    RUN_MANIFEST,
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    atomic_write_json,
    read_json,
    stable_hash,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / RUN_MANIFEST


def portfolio_hash(snapshot: dict[str, Any]) -> str:
    """Hash the portfolio composition that a run depends on.

    Only the fields that would invalidate a resume are included: the analyzed
    symbols, their mapping status, and the watchlist. Position *values* change
    every tick and must not bust the hash, so they are excluded.
    """
    positions = sorted(
        (
            p.get("symbol") or p.get("broker_ticker") or "",
            p.get("mapping_status") or "",
            bool(p.get("watch_only")),
        )
        for p in snapshot.get("positions", [])
    )
    return stable_hash(
        {"positions": positions, "watchlist": sorted(snapshot.get("watchlist", []))}
    )


def settings_fingerprint(settings: dict[str, Any]) -> str:
    """Hash the run settings that must match for a resume to be valid."""
    return stable_hash(settings)


def new_manifest(
    run_dir: str | Path,
    *,
    analysis_date: str,
    snapshot: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Create and persist a fresh manifest for a run directory."""
    now = _now()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(uuid.uuid4()),
        "created_at": now,
        "updated_at": now,
        "analysis_date": analysis_date,
        "portfolio_hash": portfolio_hash(snapshot),
        "settings_fingerprint": settings_fingerprint(settings),
        "status": STATUS_PENDING,
        "phases": {
            name: {"status": STATUS_PENDING, "started_at": None, "updated_at": now}
            for name in PHASES
        },
        "artifacts": {
            "portfolio_snapshot": PORTFOLIO_SNAPSHOT,
            "recommendations": RECOMMENDATIONS,
            "forecasts": FORECASTS,
        },
        "settings": settings,
    }
    manifest["phases"]["agents"]["tickers"] = {}
    atomic_write_json(manifest_path(run_dir), manifest)
    return manifest


def load_manifest(run_dir: str | Path) -> dict[str, Any] | None:
    return read_json(manifest_path(run_dir))


def save_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _now()
    atomic_write_json(manifest_path(run_dir), manifest)


def set_phase(
    run_dir: str | Path,
    manifest: dict[str, Any],
    phase: str,
    status: str,
) -> dict[str, Any]:
    """Update one phase's status (and run-level status) and persist."""
    if phase not in manifest["phases"]:
        raise KeyError(f"unknown phase {phase!r}")
    now = _now()
    entry = manifest["phases"][phase]
    if status == STATUS_RUNNING and entry.get("started_at") is None:
        entry["started_at"] = now
    entry["status"] = status
    entry["updated_at"] = now
    manifest["status"] = _derive_run_status(manifest)
    save_manifest(run_dir, manifest)
    return manifest


def set_ticker_status(
    run_dir: str | Path,
    manifest: dict[str, Any],
    ticker: str,
    status: str,
) -> dict[str, Any]:
    manifest["phases"]["agents"].setdefault("tickers", {})[ticker] = status
    save_manifest(run_dir, manifest)
    return manifest


def _derive_run_status(manifest: dict[str, Any]) -> str:
    statuses = [p["status"] for p in manifest["phases"].values()]
    if all(s == STATUS_COMPLETE for s in statuses):
        return STATUS_COMPLETE
    if any(s == STATUS_RUNNING for s in statuses):
        return STATUS_RUNNING
    # No phase is still running, so the run has stopped. A failed phase must
    # surface as failed; without this the run stays "running" forever and the
    # dashboard shows a spinner for a process that already exited.
    if any(s == STATUS_FAILED for s in statuses):
        return STATUS_FAILED
    if any(s == STATUS_COMPLETE for s in statuses):
        return STATUS_RUNNING
    return STATUS_PENDING


def is_resumable(
    manifest: dict[str, Any],
    *,
    snapshot: dict[str, Any],
    settings: dict[str, Any],
) -> bool:
    """A run is resumable only if portfolio and settings are unchanged.

    This is the guard the staff review demanded: a model swap or portfolio change
    must not partially reuse stale results. ``load_previous_results`` in the
    watchlist runner adds a second, finer check at the ticker level.
    """
    return (
        manifest.get("portfolio_hash") == portfolio_hash(snapshot)
        and manifest.get("settings_fingerprint") == settings_fingerprint(settings)
    )


def append_event(run_dir: str | Path, event: dict[str, Any]) -> None:
    """Append one structured event to the run's append-only ``events.jsonl``."""
    path = Path(run_dir) / EVENTS_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": _now(), **event}
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
