"""Explicit Yahoo vs Alpha Vantage latest-close reconciliation.

Unlike the vendor *router* (which silently falls back from one vendor to the
next), the dashboard must show BOTH prices and any disagreement. This module
fetches the latest close from each source and returns a structured record with a
status the UI renders as a badge. It never raises for a data problem: a missing
Alpha Vantage key or a rate limit degrades to ``yahoo_only`` with a note.

Alpha Vantage's free tier is ~25 requests/day, so callers should cross-check
only a few high-weight symbols per run and rely on the once-daily OHLCV cache.
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
from typing import Any

import pandas as pd

# Relative tolerance (fraction) beyond which two closes are "diverged".
DEFAULT_TOLERANCE = 0.01
# Yahoo close older than this many calendar days vs the reference date is stale.
STALE_DAYS = 5

STATUS_OK = "ok"
STATUS_DIVERGED = "diverged"
STATUS_STALE = "stale"
STATUS_YAHOO_ONLY = "yahoo_only"
STATUS_UNAVAILABLE = "unavailable"


def _yahoo_latest_close(symbol: str, curr_date: str) -> tuple[float | None, str | None]:
    """Latest Yahoo close on/before ``curr_date`` and its date, via the cache."""
    from .stockstats_utils import load_ohlcv
    from .symbol_utils import NoMarketDataError

    try:
        data = load_ohlcv(symbol, curr_date)
    except NoMarketDataError:
        return None, None
    except Exception:
        return None, None
    if data is None or data.empty or "Close" not in data.columns:
        return None, None
    row = data.dropna(subset=["Close"]).iloc[-1]
    date = pd.to_datetime(row["Date"]).date().isoformat() if "Date" in data.columns else None
    return float(row["Close"]), date


def _alpha_vantage_latest_close(
    symbol: str, curr_date: str
) -> tuple[float | None, str | None, str | None]:
    """Latest Alpha Vantage adjusted close; returns (close, date, degrade_note).

    ``degrade_note`` is set when Alpha Vantage is unavailable (no key / rate
    limited / error) so the caller can fall back to yahoo-only transparently.
    """
    try:
        from .alpha_vantage_common import (
            AlphaVantageNotConfiguredError,
            AlphaVantageRateLimitError,
        )
        from .alpha_vantage_stock import get_stock
        from .symbol_utils import normalize_symbol
    except ImportError as exc:  # pragma: no cover - defensive
        return None, None, f"alpha_vantage import failed: {exc}"

    start = (pd.to_datetime(curr_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    try:
        csv_data = get_stock(normalize_symbol(symbol), start, curr_date)
    except AlphaVantageNotConfiguredError:
        return None, None, "alpha_vantage_not_configured"
    except AlphaVantageRateLimitError:
        return None, None, "alpha_vantage_rate_limited"
    except Exception as exc:
        return None, None, f"alpha_vantage_error: {exc}"

    if not csv_data or not csv_data.strip():
        return None, None, "alpha_vantage_empty"
    try:
        frame = pd.read_csv(StringIO(csv_data))
    except Exception:
        return None, None, "alpha_vantage_parse_error"
    if frame.empty:
        return None, None, "alpha_vantage_empty"

    date_col = frame.columns[0]
    close_col = next(
        (c for c in frame.columns if c.lower() in ("adjusted_close", "close")),
        None,
    )
    if close_col is None:
        return None, None, "alpha_vantage_no_close_column"
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).sort_values(date_col)
    if frame.empty:
        return None, None, "alpha_vantage_empty"
    last = frame.iloc[-1]
    return float(last[close_col]), last[date_col].date().isoformat(), None


def classify(
    yahoo_close: float | None,
    yahoo_date: str | None,
    av_close: float | None,
    curr_date: str,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    stale_days: int = STALE_DAYS,
) -> tuple[str, float | None]:
    """Return (status, pct_diff) from the two closes. Pure and unit-tested."""
    if yahoo_close is None:
        return STATUS_UNAVAILABLE, None

    if yahoo_date is not None:
        age = (pd.to_datetime(curr_date).date() - pd.to_datetime(yahoo_date).date()).days
        if age > stale_days:
            return STATUS_STALE, None

    if av_close is None:
        return STATUS_YAHOO_ONLY, None

    denominator = abs(yahoo_close) or 1.0
    pct_diff = (av_close - yahoo_close) / denominator
    if abs(pct_diff) > tolerance:
        return STATUS_DIVERGED, pct_diff
    return STATUS_OK, pct_diff


def cross_check_price(
    symbol: str,
    curr_date: str,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    stale_days: int = STALE_DAYS,
) -> dict[str, Any]:
    """Reconcile the latest Yahoo and Alpha Vantage closes for one symbol."""
    yahoo_close, yahoo_date = _yahoo_latest_close(symbol, curr_date)
    av_close, av_date, degrade_note = _alpha_vantage_latest_close(symbol, curr_date)
    status, pct_diff = classify(
        yahoo_close,
        yahoo_date,
        av_close,
        curr_date,
        tolerance=tolerance,
        stale_days=stale_days,
    )
    return {
        "symbol": symbol,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "reference_date": curr_date,
        "yahoo_close": yahoo_close,
        "yahoo_date": yahoo_date,
        "alpha_vantage_close": av_close,
        "alpha_vantage_date": av_date,
        "pct_diff": round(pct_diff, 6) if pct_diff is not None else None,
        "status": status,
        "note": degrade_note,
    }
