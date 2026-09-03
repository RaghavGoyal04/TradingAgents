"""Correlated block-bootstrap portfolio simulation and risk metrics.

Resamples blocks of consecutive historical return rows across all assets at
once, so cross-asset correlation and short-horizon autocorrelation (the shape of
a correlated drawdown) are preserved -- a plain per-asset Gaussian draw would
understate joint crash risk. Everything is seeded for reproducibility.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def sample_daily_returns(
    returns: np.ndarray,
    horizon: int,
    n_paths: int,
    *,
    block_size: int = 5,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return simulated daily log returns, shape ``(n_paths, horizon, n_assets)``.

    Overlapping blocks of length ``block_size`` are drawn from the historical
    rows and laid end to end until each path has ``horizon`` days. Because a
    whole row (all assets) is copied together, the contemporaneous correlation
    structure is retained exactly.
    """
    n_obs, n_assets = returns.shape
    if n_obs == 0:
        raise ValueError("cannot simulate from an empty return history")
    block_size = max(1, min(block_size, n_obs))
    n_blocks = int(np.ceil(horizon / block_size))
    max_start = n_obs - block_size + 1

    # Random block start indices for every (path, block).
    starts = rng.integers(0, max_start, size=(n_paths, n_blocks))
    # Build an index array (n_paths, n_blocks*block_size) then trim to horizon.
    offsets = np.arange(block_size)
    idx = starts[:, :, None] + offsets[None, None, :]
    idx = idx.reshape(n_paths, n_blocks * block_size)[:, :horizon]
    # Gather: result[p, t, :] = returns[idx[p, t], :]
    return returns[idx]


def simulate_portfolio(
    returns: pd.DataFrame,
    values: dict[str, float],
    *,
    horizon: int,
    n_paths: int = 10000,
    block_size: int = 5,
    seed: int = 12345,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Simulate horizon-ahead portfolio P&L in the currency of ``values``.

    ``values`` maps symbol -> current position value (base currency). Only
    symbols present in both ``returns`` and ``values`` are simulated. Returns a
    metrics dict matching the ``forecasts.json`` ``portfolio`` block.
    """
    symbols = [s for s in returns.columns if s in values and values[s] > 0]
    if not symbols:
        raise ValueError("no overlapping positive-value symbols to simulate")

    ret = returns[symbols].to_numpy(dtype=float)
    value_vec = np.array([values[s] for s in symbols], dtype=float)
    invested = float(value_vec.sum())
    weights = value_vec / invested

    rng = np.random.default_rng(seed)
    daily = sample_daily_returns(
        ret, horizon, n_paths, block_size=block_size, rng=rng
    )  # (n_paths, horizon, n_assets)

    # Cumulative log return per asset per day -> price multiplier path.
    cum_log = np.cumsum(daily, axis=1)  # (n_paths, horizon, n_assets)
    multiplier = np.exp(cum_log)  # relative price vs today

    # Per-day portfolio value = sum_i value_i * multiplier_i.
    port_value_path = multiplier @ value_vec  # (n_paths, horizon)
    endpoint_value = port_value_path[:, -1]
    pnl = endpoint_value - invested  # currency P&L at horizon

    # Per-asset horizon simple returns for asset quantile bands.
    asset_simple = multiplier[:, -1, :] - 1.0  # (n_paths, n_assets)

    alpha = 1.0 - confidence
    var = -float(np.quantile(pnl, alpha))
    tail = pnl[pnl <= np.quantile(pnl, alpha)]
    cvar = -float(tail.mean()) if tail.size else var

    # Median path max drawdown from the intraperiod equity curve.
    equity = np.concatenate(
        [np.full((n_paths, 1), invested), port_value_path], axis=1
    )
    running_max = np.maximum.accumulate(equity, axis=1)
    drawdowns = (equity - running_max) / running_max
    max_dd_per_path = drawdowns.min(axis=1)
    max_dd_median = float(np.median(max_dd_per_path))

    return {
        "horizon": horizon,
        "n_paths": n_paths,
        "invested_value": round(invested, 2),
        "symbols": symbols,
        "pnl": {
            "p5": round(float(np.quantile(pnl, 0.05)), 2),
            "p50": round(float(np.quantile(pnl, 0.50)), 2),
            "p95": round(float(np.quantile(pnl, 0.95)), 2),
            "mean": round(float(pnl.mean()), 2),
        },
        "return_pct": {
            "p5": round(float(np.quantile(pnl, 0.05)) / invested, 6),
            "p50": round(float(np.quantile(pnl, 0.50)) / invested, 6),
            "p95": round(float(np.quantile(pnl, 0.95)) / invested, 6),
        },
        "loss_probability": round(float((pnl < 0).mean()), 6),
        f"var_{int(confidence * 100)}": round(var, 2),
        f"cvar_{int(confidence * 100)}": round(cvar, 2),
        "max_drawdown_p50": round(max_dd_median, 6),
        "risk_contribution": risk_contribution(returns[symbols], weights, symbols),
        "asset_quantiles": {
            symbol: {
                "p10": round(float(np.quantile(asset_simple[:, i], 0.10)), 6),
                "p50": round(float(np.quantile(asset_simple[:, i], 0.50)), 6),
                "p90": round(float(np.quantile(asset_simple[:, i], 0.90)), 6),
            }
            for i, symbol in enumerate(symbols)
        },
    }


def risk_contribution(
    returns: pd.DataFrame, weights: np.ndarray, symbols: list[str]
) -> dict[str, float]:
    """Percentage contribution of each position to portfolio variance.

    Uses the standard Euler decomposition: RC_i = w_i (Sigma w)_i / (w' Sigma w).
    Contributions sum to 1 (verified in tests).
    """
    cov = returns.to_numpy(dtype=float)
    covariance = np.cov(cov, rowvar=False)
    if covariance.ndim == 0:  # single asset
        return {symbols[0]: 1.0}
    portfolio_var = float(weights @ covariance @ weights)
    if portfolio_var <= 0:
        # Degenerate (e.g. zero-variance history): fall back to weights.
        return {s: round(float(w), 6) for s, w in zip(symbols, weights, strict=False)}
    marginal = covariance @ weights
    contributions = weights * marginal / portfolio_var
    return {
        s: round(float(c), 6)
        for s, c in zip(symbols, contributions, strict=False)
    }
