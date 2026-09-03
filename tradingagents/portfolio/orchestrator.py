"""Resumable portfolio run orchestrator.

Owns one run directory and drives three phases under a single manifest:
``ingest`` (Trading212 snapshot), ``agents`` (the existing TradingAgents
subprocess pipeline), and ``forecast`` (baseline + optional TimesFM). The UI
launches this process and only reads the artifacts it writes.

Design choices:
- The agents phase shells out to ``scripts/run_watchlist.py`` so all of its
  battle-tested locking, per-ticker timeouts, retries, and LangGraph checkpoint
  resume are reused verbatim. The orchestrator never imports the heavy agent
  graph itself.
- Both the agents and forecast phases are injectable functions so tests can run
  a full end-to-end lifecycle without network or AWS.
- ``caffeinate`` keeps the Mac awake for the run's duration; resume covers the
  lid-close case macOS won't.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from . import sleep_guard, t212
from .contracts import (
    AGENTS_SUBDIR,
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_RUNNING,
    read_json,
)
from .manifest import (
    append_event,
    is_resumable,
    load_manifest,
    new_manifest,
    set_phase,
    set_ticker_status,
)


@contextlib.contextmanager
def exclusive_lock(run_dir: Path) -> Iterator[None]:
    """Prevent two orchestrators from writing the same run directory.

    Uses the same PID-file pattern as the watchlist runner: a stale lock owned
    by a dead PID is reclaimed, an active one is rejected.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".orchestrator.lock"
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
            break
        except FileExistsError:
            try:
                owner = int(lock_path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                owner = -1
            if owner > 0 and _pid_alive(owner):
                raise RuntimeError(
                    f"another portfolio run (PID {owner}) is using {run_dir}"
                ) from None
            lock_path.unlink(missing_ok=True)
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError, OSError, ValueError):
            if int(lock_path.read_text(encoding="utf-8").strip()) == os.getpid():
                lock_path.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def build_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    """Ingest the portfolio from Trading212 or a holdings file."""
    watchlist = list(args.watchlist or [])
    if args.trading212:
        return t212.fetch_portfolio_snapshot(watchlist=watchlist)
    import json

    holdings = json.loads(Path(args.holdings).read_text(encoding="utf-8"))
    holdings = {str(k).upper(): float(v) for k, v in holdings.items()}
    return t212.snapshot_from_holdings(
        holdings,
        account_value=args.capital,
        currency=args.currency or "USD",
        watchlist=watchlist,
    )


def _settings(args: argparse.Namespace, snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "quick_model": args.quick_model,
        "deep_model": args.deep_model,
        "depth": getattr(args, "depth", "shallow"),
        "llm_provider": getattr(args, "llm_provider", None),
        "effort": getattr(args, "effort", None),
        "concurrency": getattr(args, "concurrency", 6),
        "base_currency": snapshot["base_currency"],
        "analysis_date": args.analysis_date,
        "aws_profile": args.aws_profile,
        "aws_region": args.aws_region,
    }


def run_agents_phase(
    run_dir: Path,
    manifest: dict[str, Any],
    snapshot: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    """Launch the existing watchlist runner for the analyzed symbols.

    Returns the subprocess exit code. Ticker statuses are lifted from the
    incrementally written ``recommendations.json`` into the manifest.
    """
    symbols = t212.analyzed_symbols(snapshot)
    if not symbols:
        append_event(run_dir, {"phase": "agents", "event": "no_symbols"})
        return 0

    agents_dir = run_dir / AGENTS_SUBDIR
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[2] / "scripts" / "run_watchlist.py"),
        "--tickers",
        *symbols,
        "--output-dir",
        str(agents_dir),
        "--analysis-date",
        args.analysis_date,
        "--quick-model",
        args.quick_model,
        "--deep-model",
        args.deep_model,
        "--aws-profile",
        args.aws_profile,
        "--aws-region",
        args.aws_region,
        "--depth",
        getattr(args, "depth", "shallow"),
        "--concurrency",
        str(getattr(args, "concurrency", 6)),
    ]
    if getattr(args, "llm_provider", None):
        command.extend(["--llm-provider", args.llm_provider])
    if getattr(args, "llm_base_url", None):
        command.extend(["--llm-base-url", args.llm_base_url])
    if getattr(args, "effort", None):
        command.extend(["--effort", args.effort])
    if getattr(args, "force", False):
        command.append("--force")
    # Pass the portfolio source through so sizing uses live capital + holdings.
    if args.trading212:
        command.append("--trading212")
    elif args.holdings:
        command.extend(["--holdings", args.holdings])
    if args.capital is not None:
        command.extend(["--capital", str(args.capital)])

    append_event(run_dir, {"phase": "agents", "event": "launch", "symbols": symbols})
    completed = subprocess.run(command, check=False)
    return completed.returncode


def _sync_ticker_statuses(
    run_dir: Path,
    manifest: dict[str, Any],
    agents_dir: Path,
    symbols: list[str],
    since: float | None = None,
) -> dict[str, int]:
    """Copy per-ticker outcomes into the manifest and return a status tally.

    ``since`` is the time the phase started. Results written before it belong to
    an earlier run, and counting them let a phase that died on startup inherit
    the previous run's successes and report itself complete.
    """
    results_file = agents_dir / "recommendations.json"
    if since is not None and (
        not results_file.exists() or results_file.stat().st_mtime < since
    ):
        return dict.fromkeys(symbols and ["pending"] or [], len(symbols))
    payload = read_json(results_file) or {}
    by_ticker = {item["ticker"]: item for item in payload.get("results", [])}
    tally: dict[str, int] = {}
    for symbol in symbols:
        item = by_ticker.get(symbol)
        status = item["status"] if item else "pending"
        tally[status] = tally.get(status, 0) + 1
        set_ticker_status(run_dir, manifest, symbol, status)
    return tally


def orchestrate(
    args: argparse.Namespace,
    *,
    agents_phase: Callable[..., int] | None = None,
    forecast_phase: Callable[..., dict[str, Any]] | None = None,
) -> int:
    """Run (or resume) the full portfolio pipeline. Returns a process exit code.

    ``agents_phase`` and ``forecast_phase`` are injectable for tests; production
    defaults call the subprocess runner and the forecast module respectively.
    """
    run_dir = Path(args.run_dir).expanduser()
    agents_phase = agents_phase or run_agents_phase
    if forecast_phase is None:
        from ..forecast.run import generate_forecasts as forecast_phase  # lazy

    with exclusive_lock(run_dir), sleep_guard.keep_awake():
        snapshot = build_snapshot(args)
        t212.write_snapshot(run_dir, snapshot)

        unmapped_nonzero = [
            p["broker_ticker"]
            for p in snapshot["positions"]
            if p["mapping_status"] != "mapped" and p.get("value", 0) > 0
        ]

        settings = _settings(args, snapshot)
        manifest = load_manifest(run_dir)
        resuming = bool(
            args.resume
            and manifest
            and is_resumable(manifest, snapshot=snapshot, settings=settings)
        )
        if not resuming:
            manifest = new_manifest(
                run_dir,
                analysis_date=args.analysis_date,
                snapshot=snapshot,
                settings=settings,
            )
        append_event(
            run_dir,
            {"event": "start", "resuming": resuming, "run_id": manifest["run_id"]},
        )

        if unmapped_nonzero:
            # Do not brick the whole run for one unmappable instrument. Skip it
            # from analysis (never guess a wrong symbol), keep it visible in the
            # snapshot, and surface a prominent warning. It still counts toward
            # account value/cash.
            append_event(
                run_dir,
                {"phase": "ingest", "event": "unmapped_skipped",
                 "tickers": unmapped_nonzero},
            )
        set_phase(run_dir, manifest, "ingest", STATUS_COMPLETE)

        # Agents phase (skip if resuming an already-complete phase).
        if not (resuming and manifest["phases"]["agents"]["status"] == STATUS_COMPLETE):
            if not args.skip_agents:
                set_phase(run_dir, manifest, "agents", STATUS_RUNNING)
                phase_started = time.time()
                code = agents_phase(run_dir, manifest, snapshot, args)
                # Lift per-ticker results into the manifest regardless of which
                # phase implementation ran (subprocess runner or injected test).
                tally = _sync_ticker_statuses(
                    run_dir,
                    manifest,
                    run_dir / AGENTS_SUBDIR,
                    t212.analyzed_symbols(snapshot),
                    since=phase_started,
                )
                # A single un-analyzable holding (delisted, no market data) must
                # not condemn a run where everything else worked -- the per-ticker
                # errors are already recorded above and shown in the dashboard.
                # Only a phase that produced no successes at all is a failure.
                succeeded = tally.get("success", 0)
                status = (
                    STATUS_COMPLETE if (code == 0 or succeeded) else STATUS_FAILED
                )
                set_phase(run_dir, manifest, "agents", status)
                append_event(
                    run_dir,
                    {
                        "phase": "agents",
                        "event": "done",
                        "code": code,
                        "tickers": tally,
                    },
                )
            else:
                set_phase(run_dir, manifest, "agents", STATUS_COMPLETE)

        # Forecast phase.
        if not args.skip_forecast:
            set_phase(run_dir, manifest, "forecast", STATUS_RUNNING)
            try:
                forecast_phase(
                    snapshot,
                    run_dir,
                    settings=settings,
                    use_timesfm=getattr(args, "timesfm", False),
                )
                set_phase(run_dir, manifest, "forecast", STATUS_COMPLETE)
            except Exception as exc:  # forecast must never abort a whole run
                set_phase(run_dir, manifest, "forecast", STATUS_FAILED)
                append_event(
                    run_dir,
                    {"phase": "forecast", "event": "error", "error": repr(exc)},
                )
        else:
            set_phase(run_dir, manifest, "forecast", STATUS_COMPLETE)

        append_event(run_dir, {"event": "finished", "status": manifest["status"]})
    return 0 if manifest["status"] == STATUS_COMPLETE else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--trading212", action="store_true")
    source.add_argument("--holdings", help="JSON file mapping symbols to values")
    parser.add_argument("--watchlist", nargs="*", default=[])
    parser.add_argument("--analysis-date")
    parser.add_argument("--capital", type=float)
    parser.add_argument("--currency")
    parser.add_argument(
        "--depth",
        choices=("shallow", "medium", "deep"),
        default="shallow",
        help="Analysis effort per holding.",
    )
    parser.add_argument("--llm-provider")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--effort")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--quick-model", default="us.anthropic.claude-sonnet-4-6")
    parser.add_argument("--deep-model", default="us.anthropic.claude-opus-4-7")
    parser.add_argument(
        "--aws-profile",
        default=os.getenv("AWS_PROFILE", "wbd-boltcloud-data-processing-dev.GtDcpDca"),
    )
    parser.add_argument("--aws-region", default=os.getenv("AWS_REGION", "us-east-1"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-agents", action="store_true")
    parser.add_argument("--skip-forecast", action="store_true")
    parser.add_argument(
        "--timesfm",
        action="store_true",
        help="Back-test the TimesFM candidate too (minutes, CPU-bound).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-analyse every ticker instead of reusing today's completed results.",
    )
    args = parser.parse_args(argv)
    if args.analysis_date is None:
        from datetime import datetime

        args.analysis_date = datetime.now().astimezone().date().isoformat()
    return args


def main(argv: list[str] | None = None) -> int:
    return orchestrate(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
