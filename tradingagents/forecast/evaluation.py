"""Leakage-safe rolling-origin evaluation and the TimesFM promotion gate.

A forecaster is only ever handed data strictly before the forecast origin, so a
model cannot "see" the value it is scored against. Forecasts are probabilistic
(quantiles), scored with the pinball (quantile) loss, and TimesFM is promoted
per horizon only when it beats the best baseline by a margin on enough holdings.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

# Forecaster signature: (train_returns_1d, horizon, quantile_levels) ->
# {level: predicted_cumulative_log_return}. It must never receive future data.
Forecaster = Callable[[np.ndarray, int, tuple[float, ...]], dict[float, float]]

DEFAULT_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
MIN_OBS = 252
MIN_IMPROVEMENT = 0.05
MIN_FRACTION = 0.8


def pinball_loss(
    y_true: float, quantile_preds: dict[float, float]
) -> float:
    """Mean pinball loss of a set of quantile predictions against one outcome."""
    losses = []
    for level, pred in quantile_preds.items():
        error = y_true - pred
        losses.append(max(level * error, (level - 1) * error))
    return float(np.mean(losses)) if losses else float("nan")


def rolling_origin_score(
    series: np.ndarray,
    horizon: int,
    forecaster: Forecaster,
    *,
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES,
    min_train: int = MIN_OBS,
    max_origins: int = 60,
) -> float | None:
    """Mean pinball loss over rolling origins; ``None`` if too little history.

    At each origin ``t`` the forecaster sees only ``series[:t]`` and predicts the
    cumulative return over ``series[t:t+horizon]``. Origins are spaced to cap the
    number of evaluations at ``max_origins`` for speed.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    first = min_train
    last = n - horizon
    if last <= first:
        return None

    origins = list(range(first, last + 1))
    if len(origins) > max_origins:
        step = len(origins) // max_origins
        origins = origins[::step]

    losses = []
    for origin in origins:
        train = series[:origin]  # strictly past -- no leakage
        realized = float(series[origin : origin + horizon].sum())
        preds = forecaster(train, horizon, quantile_levels)
        losses.append(pinball_loss(realized, preds))
    return float(np.mean(losses)) if losses else None


def naive_forecaster(
    train: np.ndarray, horizon: int, quantile_levels: tuple[float, ...]
) -> dict[float, float]:
    """Random-walk baseline: zero-drift, empirical daily vol scaled by sqrt(h)."""
    sigma = float(np.std(train)) if len(train) > 1 else 0.0
    scale = sigma * np.sqrt(horizon)
    # Normal quantiles around a zero cumulative drift.
    from math import erf, sqrt

    def z(p: float) -> float:
        # Inverse standard-normal CDF via bisection on erf (no scipy dependency).
        lo, hi = -8.0, 8.0
        for _ in range(64):
            mid = (lo + hi) / 2
            if 0.5 * (1 + erf(mid / sqrt(2))) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    return {level: z(level) * scale for level in quantile_levels}


def block_bootstrap_forecaster(block_size: int = 5, n_paths: int = 2000, seed: int = 7):
    """Return a forecaster that predicts quantiles via block bootstrap."""

    def forecaster(
        train: np.ndarray, horizon: int, quantile_levels: tuple[float, ...]
    ) -> dict[float, float]:
        rng = np.random.default_rng(seed)
        n = len(train)
        bs = max(1, min(block_size, n))
        n_blocks = int(np.ceil(horizon / bs))
        max_start = n - bs + 1
        starts = rng.integers(0, max_start, size=(n_paths, n_blocks))
        offsets = np.arange(bs)
        idx = (starts[:, :, None] + offsets[None, None, :]).reshape(
            n_paths, n_blocks * bs
        )[:, :horizon]
        cum = train[idx].sum(axis=1)
        return {level: float(np.quantile(cum, level)) for level in quantile_levels}

    return forecaster


def evaluate_symbol(
    series: np.ndarray,
    horizon: int,
    candidate: Forecaster | None,
    *,
    quantile_levels: tuple[float, ...] = DEFAULT_QUANTILES,
    min_train: int = MIN_OBS,
    max_origins: int = 60,
) -> dict[str, Any]:
    """Score baselines (and the candidate if any) for one symbol/horizon.

    ``max_origins`` applies to every model equally so the comparison stays fair
    when it is lowered to keep an expensive candidate affordable.
    """
    if len(series) < min_train + horizon:
        return {"eligible": False, "reason": "insufficient_history"}

    scores = {
        "naive": rolling_origin_score(
            series, horizon, naive_forecaster,
            quantile_levels=quantile_levels, min_train=min_train,
            max_origins=max_origins,
        ),
        "bootstrap": rolling_origin_score(
            series, horizon, block_bootstrap_forecaster(),
            quantile_levels=quantile_levels, min_train=min_train,
            max_origins=max_origins,
        ),
    }
    if candidate is not None:
        scores["timesfm"] = rolling_origin_score(
            series, horizon, candidate,
            quantile_levels=quantile_levels, min_train=min_train,
            max_origins=max_origins,
        )
    valid = {k: v for k, v in scores.items() if v is not None}
    best_baseline = min(
        (v for k, v in valid.items() if k != "timesfm"), default=None
    )
    return {"eligible": True, "scores": valid, "best_baseline": best_baseline}


def promotion_decision(
    per_symbol: dict[str, dict[str, Any]],
    horizon: int,
    *,
    min_improvement: float = MIN_IMPROVEMENT,
    min_fraction: float = MIN_FRACTION,
) -> dict[str, Any]:
    """Decide whether TimesFM is promoted for ``horizon``.

    Promoted only when TimesFM beats the best baseline by >= ``min_improvement``
    (lower pinball loss) on >= ``min_fraction`` of eligible holdings. Otherwise
    the winning baseline is selected.
    """
    eligible = [s for s in per_symbol.values() if s.get("eligible")]
    improved = 0
    counted = 0
    for entry in eligible:
        scores = entry.get("scores", {})
        timesfm = scores.get("timesfm")
        baseline = entry.get("best_baseline")
        if timesfm is None or baseline is None or baseline <= 0:
            continue
        counted += 1
        if timesfm <= baseline * (1 - min_improvement):
            improved += 1

    fraction = improved / counted if counted else 0.0
    promoted = counted > 0 and fraction >= min_fraction

    # Winner among baselines aggregated across symbols (mean pinball).
    baseline_means = _mean_scores(eligible, exclude="timesfm")
    winner = "timesfm" if promoted else _argmin(baseline_means)
    return {
        "horizon": horizon,
        "promoted": promoted,
        "winner": winner or "bootstrap",
        "eligible_count": len(eligible),
        "counted": counted,
        "improved_fraction": round(fraction, 4),
        "baseline_mean_scores": baseline_means,
    }


def _mean_scores(entries: list[dict[str, Any]], *, exclude: str) -> dict[str, float]:
    sums: dict[str, list[float]] = {}
    for entry in entries:
        for name, value in entry.get("scores", {}).items():
            if name == exclude or value is None:
                continue
            sums.setdefault(name, []).append(value)
    return {name: round(float(np.mean(vals)), 6) for name, vals in sums.items()}


def _argmin(scores: dict[str, float]) -> str | None:
    return min(scores, key=scores.get) if scores else None
