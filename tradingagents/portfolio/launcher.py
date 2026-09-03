"""Pure launch helpers for the dashboard (no Streamlit import).

Keeping the command construction and run-state checks here means they are unit
tested without a UI. The Streamlit app is a thin shell over these functions.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .contracts import RUN_MANIFEST
from .orchestrator import _pid_alive  # reuse the single liveness check

# Lives here rather than in scripts/run_watchlist.py so both the runner and the
# dashboard can import it: Streamlit puts dashboard/ on sys.path, not the repo
# root, so scripts/ is not importable from the UI.
MAX_CONCURRENCY = 100


def runs_root(base: str | Path | None = None) -> Path:
    """Base directory that holds one subdirectory per run."""
    if base is not None:
        return Path(base).expanduser()
    root = os.getenv("TRADINGAGENTS_RESULTS_DIR", "~/.tradingagents/logs")
    return Path(root).expanduser() / "portfolio"


def new_run_dir(analysis_date: str, base: str | Path | None = None) -> Path:
    return runs_root(base) / analysis_date


def list_run_dirs(base: str | Path | None = None) -> list[Path]:
    root = runs_root(base)
    if not root.exists():
        return []
    return sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / RUN_MANIFEST).exists()),
        reverse=True,
    )


def is_run_active(run_dir: str | Path) -> bool:
    """True when an orchestrator lock is held by a live process."""
    lock = Path(run_dir) / ".orchestrator.lock"
    if not lock.exists():
        return False
    try:
        owner = int(lock.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    return owner > 0 and _pid_alive(owner)


def build_orchestrator_command(
    run_dir: str | Path,
    *,
    source: str,
    holdings_path: str | None = None,
    watchlist: list[str] | None = None,
    analysis_date: str | None = None,
    quick_model: str,
    deep_model: str,
    aws_profile: str | None = None,
    aws_region: str | None = None,
    resume: bool = False,
    skip_agents: bool = False,
    skip_forecast: bool = False,
    use_timesfm: bool = False,
    force: bool = False,
    depth: str = "shallow",
    llm_provider: str | None = None,
    llm_base_url: str | None = None,
    effort: str | None = None,
    concurrency: int = 6,
    python: str | None = None,
) -> list[str]:
    """Construct the ``python -m tradingagents.portfolio.orchestrator`` argv.

    ``source`` is ``"trading212"`` or ``"holdings"``. Invalid combinations
    (holdings source without a path) raise ``ValueError`` so the UI surfaces a
    clear error before spawning anything.
    """
    if source not in ("trading212", "holdings"):
        raise ValueError(f"unknown source {source!r}")
    if source == "holdings" and not holdings_path:
        raise ValueError("holdings source requires a holdings file path")

    command: list[str] = [
        python or sys.executable,
        "-m",
        "tradingagents.portfolio.orchestrator",
        "--run-dir",
        str(run_dir),
        "--quick-model",
        quick_model,
        "--deep-model",
        deep_model,
    ]
    if source == "trading212":
        command.append("--trading212")
    else:
        command.extend(["--holdings", holdings_path])
    if watchlist:
        command.extend(["--watchlist", *watchlist])
    if analysis_date:
        command.extend(["--analysis-date", analysis_date])
    if aws_profile:
        command.extend(["--aws-profile", aws_profile])
    if aws_region:
        command.extend(["--aws-region", aws_region])
    if resume:
        command.append("--resume")
    if skip_agents:
        command.append("--skip-agents")
    if skip_forecast:
        command.append("--skip-forecast")
    if use_timesfm:
        command.append("--timesfm")
    if force:
        command.append("--force")
    command.extend(["--depth", depth])
    if llm_provider:
        command.extend(["--llm-provider", llm_provider])
    if llm_base_url:
        command.extend(["--llm-base-url", llm_base_url])
    if effort:
        command.extend(["--effort", effort])
    command.extend(["--concurrency", str(concurrency)])
    return command


def build_watchlist_command(
    tickers: list[str],
    *,
    output_dir: str | Path,
    quick_model: str,
    deep_model: str,
    analysis_date: str | None = None,
    depth: str = "shallow",
    llm_provider: str | None = None,
    llm_base_url: str | None = None,
    effort: str | None = None,
    concurrency: int = 6,
    aws_profile: str | None = None,
    aws_region: str | None = None,
    python: str | None = None,
) -> list[str]:
    """Construct the ``scripts/run_watchlist.py`` argv for an explicit ticker list.

    Used to run the full analyst pipeline over candidates that came from
    somewhere other than a portfolio -- a screener, say -- so no holdings
    source is passed and position sizing is skipped.
    """
    if not tickers:
        raise ValueError("at least one ticker is required")

    command: list[str] = [
        python or sys.executable,
        str(Path(__file__).resolve().parents[2] / "scripts" / "run_watchlist.py"),
        "--tickers",
        *tickers,
        "--output-dir",
        str(output_dir),
        "--quick-model",
        quick_model,
        "--deep-model",
        deep_model,
        "--depth",
        depth,
    ]
    if analysis_date:
        command.extend(["--analysis-date", analysis_date])
    if llm_provider:
        command.extend(["--llm-provider", llm_provider])
    if llm_base_url:
        command.extend(["--llm-base-url", llm_base_url])
    if effort:
        command.extend(["--effort", effort])
    command.extend(["--concurrency", str(concurrency)])
    if aws_profile:
        command.extend(["--aws-profile", aws_profile])
    if aws_region:
        command.extend(["--aws-region", aws_region])
    return command


def launch(
    command: list[str],
    run_dir: str | Path,
    *,
    env: dict[str, str] | None = None,
    popen: Any = None,
    log_name: str = "orchestrator.log",
) -> int:
    """Spawn a detached orchestrator process; return its PID.

    Rejects launching when a run is already active in ``run_dir`` (the lock
    check), preventing the Streamlit double-rerun from starting two runs.

    ``env`` is merged over the current environment. Provider tokens travel this
    way rather than as command-line arguments, which any user on the machine
    could read out of ``ps``.
    """
    if is_run_active(run_dir):
        raise RuntimeError(f"a run is already active in {run_dir}")
    import subprocess

    runner = popen or subprocess.Popen
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    # The log handle must outlive this function: it is inherited by the detached
    # orchestrator process, so it is intentionally not closed here.
    log = open(Path(run_dir) / log_name, "a", encoding="utf-8")  # noqa: SIM115
    child_env = {**os.environ, **env} if env else None
    process = runner(command, stdout=log, stderr=subprocess.STDOUT, env=child_env)
    return process.pid
