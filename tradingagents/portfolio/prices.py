"""Aligned multi-asset price/return panel for portfolio forecasting.

Builds a dates x symbols matrix of adjusted closes from the existing cached
Yahoo loader (``load_ohlcv``), intersected onto a common trading calendar so the
correlation structure the simulator relies on is well defined. Log returns are
computed on the aligned panel. A ``price_loader`` is injectable so tests build
panels from synthetic series without any network.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

# (symbol, curr_date) -> DataFrame with Date and Close columns.
PriceLoader = Callable[[str, str], pd.DataFrame]


def _default_loader(symbol: str, curr_date: str) -> pd.DataFrame:
    from ..dataflows.stockstats_utils import load_ohlcv

    return load_ohlcv(symbol, curr_date)


def build_close_panel(
    symbols: list[str],
    curr_date: str,
    *,
    lookback_days: int = 504,
    min_obs: int = 60,
    price_loader: PriceLoader | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return (aligned close panel, dropped symbols).

    Symbols with fewer than ``min_obs`` observations in the window are dropped
    and reported so the caller can flag them rather than silently skewing the
    correlation matrix. The panel is inner-joined on dates (common calendar).
    """
    loader = price_loader or _default_loader
    series: dict[str, pd.Series] = {}
    dropped: list[str] = []
    cutoff = pd.to_datetime(curr_date) - pd.Timedelta(days=lookback_days)

    for symbol in symbols:
        try:
            frame = loader(symbol, curr_date)
        except Exception:
            dropped.append(symbol)
            continue
        if frame is None or frame.empty or "Close" not in frame.columns:
            dropped.append(symbol)
            continue
        frame = frame.copy()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date", "Close"])
        frame = frame[frame["Date"] >= cutoff]
        s = frame.set_index("Date")["Close"].astype(float)
        s = s[~s.index.duplicated(keep="last")]
        if len(s) < min_obs:
            dropped.append(symbol)
            continue
        series[symbol] = s

    if not series:
        return pd.DataFrame(), dropped

    panel = pd.DataFrame(series).sort_index()
    # Inner join on the common calendar, then forward-fill single-day gaps that
    # arise from differing exchange holidays before dropping any residual NaNs.
    panel = panel.ffill().dropna()
    return panel, dropped


def log_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns of an aligned close panel (drops the first row)."""
    if panel.empty:
        return panel
    return np.log(panel / panel.shift(1)).dropna()


# (local_currency, base_currency, index) -> aligned Series of base-per-local, or
# None when the FX series is unavailable.
FxSeriesLoader = Callable[[str, str, pd.DatetimeIndex], "pd.Series | None"]


def _default_fx_series_loader(
    local: str, base: str, index: pd.DatetimeIndex
) -> pd.Series | None:
    if local == base:
        return pd.Series(1.0, index=index)
    from ..dataflows.stockstats_utils import load_ohlcv

    curr_date = pd.Timestamp(index.max()).strftime("%Y-%m-%d")
    try:
        frame = load_ohlcv(f"{local}{base}=X", curr_date)
    except Exception:
        return None
    if frame is None or frame.empty or "Close" not in frame.columns:
        return None
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    series = frame.dropna(subset=["Date"]).set_index("Date")["Close"].astype(float)
    return series.reindex(index).ffill().bfill()


def to_base_currency_panel(
    panel: pd.DataFrame,
    base_currency: str,
    *,
    fx_series_loader: FxSeriesLoader | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Convert each local-currency price column into base currency.

    Returns ``(base_panel, warnings)``. Converting through the FX *series* (not a
    single rate) means the resulting returns correctly include currency drift for
    foreign listings. A symbol whose FX series is unavailable is kept in local
    currency and a warning is recorded, so nothing is silently mis-stated.
    """
    from .fx import currency_for_symbol

    if panel.empty:
        return panel, []
    loader = fx_series_loader or _default_fx_series_loader
    base_currency = base_currency.upper()
    converted = {}
    warnings: list[str] = []
    for symbol in panel.columns:
        local = currency_for_symbol(symbol)
        if local == base_currency:
            converted[symbol] = panel[symbol]
            continue
        fx = loader(local, base_currency, panel.index)
        if fx is None or fx.isna().all():
            converted[symbol] = panel[symbol]
            warnings.append(
                f"{symbol}: FX {local}->{base_currency} unavailable; kept in {local}"
            )
            continue
        converted[symbol] = panel[symbol] * fx.reindex(panel.index).ffill().bfill()
    return pd.DataFrame(converted).dropna(), warnings
