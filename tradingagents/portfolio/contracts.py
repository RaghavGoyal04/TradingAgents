"""Versioned artifact contracts and atomic IO for the portfolio dashboard.

Every artifact carries ``schema_version`` and is written atomically (temp file
+ ``os.replace``) so a reader (the Streamlit UI) never observes a half-written
file, even if the orchestrator is killed mid-write. See the schemas in
docs/portfolio-intelligence-prd.md.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

# Canonical artifact filenames (relative to a run directory).
PORTFOLIO_SNAPSHOT = "portfolio_snapshot.json"
RUN_MANIFEST = "run_manifest.json"
FORECASTS = "forecasts.json"
EVENTS_LOG = "events.jsonl"
AGENTS_SUBDIR = "agents"
RECOMMENDATIONS = f"{AGENTS_SUBDIR}/recommendations.json"

# Phase names, in execution order.
PHASES = ("ingest", "agents", "forecast")

# Status vocabulary shared across manifest phases and the run itself.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Write ``value`` as JSON to ``path`` atomically.

    The temp file is created in the destination directory so ``os.replace`` is a
    same-filesystem rename (atomic on POSIX). A partially written temp file is
    never promoted to the destination name.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any | None:
    """Return parsed JSON at ``path``, or ``None`` if missing/corrupt.

    Corrupt reads return ``None`` rather than raising so the UI can degrade to a
    "no valid artifact yet" state instead of crashing on a torn write from a
    process that predates :func:`atomic_write_json`.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def stable_hash(value: Any, *, length: int = 16) -> str:
    """Deterministic short hash of any JSON-serializable value.

    Used for ``portfolio_hash`` and ``settings_fingerprint`` so resume can
    detect when the portfolio composition or run settings changed and must not
    be reused. Keys are sorted so ordering never changes the hash.
    """
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]
