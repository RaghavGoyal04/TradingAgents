"""Portfolio forecast entrypoint: writes ``forecasts.json`` for a run.

Loads an aligned, base-currency return panel for the analyzed symbols, runs the
always-available block-bootstrap portfolio simulation, evaluates whether TimesFM
should be promoted per horizon (leakage-safe walk-forward), and records
per-asset quantile bands plus the portfolio P&L distribution. The heavy TimesFM
model, if used, is loaded exactly once here -- never inside a ticker worker.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..portfolio import SCHEMA_VERSION
from ..portfolio.contracts import FORECASTS, atomic_write_json
from ..portfolio.prices import (
    build_close_panel,
    log_returns,
    to_base_currency_panel,
)
from . import evaluation, simulation, timesfm_adapter

HORIZONS = (1, 5, 20)
PORTFOLIO_HORIZON = 20
N_PATHS = 10000
# Walk-forward origins per symbol/horizon when the TimesFM candidate is in the
# race. Keeps a full back-test in the low minutes instead of the low hours.
TIMESFM_MAX_ORIGINS = 12


def generate_forecasts(
    snapshot: dict[str, Any],
    run_dir: str | Path,
    *,
    settings: dict[str, Any] | None = None,
    price_loader: Callable[..., pd.DataFrame] | None = None,
    fx_series_loader: Callable[..., Any] | None = None,
    timesfm_forecaster: Any | None = None,
    use_timesfm: bool = False,
    horizons: tuple[int, ...] = HORIZONS,
    n_paths: int = N_PATHS,
    seed: int = 12345,
) -> dict[str, Any]:
    """Build and persist ``forecasts.json``; also return the payload."""
    if timesfm_forecaster is None and use_timesfm:
        # Opt-in only: the walk-forward back-test needs thousands of CPU
        # inferences, so it must never be paid for by a routine run.
        timesfm_forecaster = timesfm_adapter.get_forecaster()
    from ..portfolio.t212 import analyzed_symbols

    base_currency = snapshot.get("base_currency", "USD")
    curr_date = (settings or {}).get("analysis_date") or datetime.now().date().isoformat()
    symbols = analyzed_symbols(snapshot)
    values = {
        p["symbol"]: p["value"]
        for p in snapshot.get("positions", [])
        if p.get("symbol") and not p.get("watch_only") and p.get("value", 0) > 0
    }

    warnings: list[str] = []
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_currency": base_currency,
        "data_cutoff": curr_date,
        "horizons": list(horizons),
        "model_used": "bootstrap_v1",
        "evaluation": {},
        "assets": {},
        "portfolio": {},
        "warnings": warnings,
    }

    if not symbols:
        warnings.append("no analyzable symbols in snapshot")
        _write(run_dir, payload)
        return payload

    local_panel, dropped = build_close_panel(
        symbols, curr_date, price_loader=price_loader
    )
    if dropped:
        warnings.append(f"insufficient history dropped: {', '.join(dropped)}")
    if local_panel.empty:
        warnings.append("no price history available for any symbol")
        _write(run_dir, payload)
        return payload

    panel, fx_warnings = to_base_currency_panel(
        local_panel, base_currency, fx_series_loader=fx_series_loader
    )
    warnings.extend(fx_warnings)
    returns = log_returns(panel)

    # Portfolio simulation (always available baseline).
    sim_values = {s: v for s, v in values.items() if s in returns.columns}
    if sim_values:
        try:
            portfolio = simulation.simulate_portfolio(
                returns,
                sim_values,
                horizon=PORTFOLIO_HORIZON,
                n_paths=n_paths,
                seed=seed,
            )
            portfolio["currency"] = base_currency
            portfolio["warnings"] = list(warnings)
            payload["portfolio"] = portfolio
            payload["assets"] = _asset_bands(portfolio, panel)
        except ValueError as exc:
            warnings.append(f"portfolio simulation skipped: {exc}")

    # Promotion gate per horizon (candidate optional).
    payload["evaluation"] = _evaluate(returns, horizons, timesfm_forecaster)
    if any(v.get("promoted") for v in payload["evaluation"].values()):
        payload["model_used"] = "timesfm_v3"
    # Record WHY the candidate was or was not usable, so the dashboard can
    # distinguish "not installed" from "installed but no weights".
    payload["timesfm"] = timesfm_adapter.status()

    _write(run_dir, payload)
    return payload


def _asset_bands(portfolio: dict[str, Any], panel: pd.DataFrame) -> dict[str, Any]:
    bands = portfolio.get("asset_quantiles", {})
    assets = {}
    for symbol, quantiles in bands.items():
        tail = panel[symbol].dropna().tail(60)
        assets[symbol] = {
            "history_tail": [round(float(x), 4) for x in tail.to_list()],
            "quantiles": {str(portfolio["horizon"]): quantiles},
        }
    return assets


def _evaluate(
    returns: pd.DataFrame,
    horizons: tuple[int, ...],
    candidate: Any | None,
) -> dict[str, Any]:
    # A TimesFM inference costs ~1000x a baseline draw, so cut the number of
    # walk-forward origins when it is in the race. Applied to every model so the
    # comparison stays like-for-like.
    max_origins = TIMESFM_MAX_ORIGINS if candidate is not None else 60
    evaluation_result: dict[str, Any] = {}
    for horizon in horizons:
        per_symbol = {
            symbol: evaluation.evaluate_symbol(
                returns[symbol].to_numpy(dtype=float),
                horizon,
                candidate,
                max_origins=max_origins,
            )
            for symbol in returns.columns
        }
        evaluation_result[str(horizon)] = evaluation.promotion_decision(
            per_symbol, horizon
        )
    return evaluation_result


def _write(run_dir: str | Path, payload: dict[str, Any]) -> None:
    atomic_write_json(Path(run_dir) / FORECASTS, payload)
