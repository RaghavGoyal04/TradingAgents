"""Base-currency FX conversion for portfolio math.

Every listing's price series is quoted in its local exchange currency (USD for
US, EUR for ``.DE``, KRW for ``.KS``, ...). Portfolio risk/forecast aggregation
must happen in the account (base) currency, so returns from each listing are
kept in local currency (returns are currency-agnostic up to FX drift) and only
absolute value conversions use dated FX rates.

The converter takes an injectable ``rate_source`` so tests never hit the
network. The default source uses Yahoo ``<PAIR>=X`` daily closes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Mapping of an equity's listing suffix to its local trading currency. US
# tickers (no dotted suffix) are USD. Extend as new markets are held.
SUFFIX_CURRENCY = {
    ".L": "GBP",
    ".DE": "EUR",
    ".PA": "EUR",
    ".AS": "EUR",
    ".MC": "EUR",
    ".MI": "EUR",
    ".BR": "EUR",
    ".LS": "EUR",
    ".VI": "EUR",
    ".SW": "CHF",
    ".TO": "CAD",
    ".KS": "KRW",
    ".T": "JPY",
    ".HK": "HKD",
    ".SS": "CNY",
    ".SZ": "CNY",
    ".AX": "AUD",
    ".NS": "INR",
    ".BO": "INR",
}

# London quotes many equities in pence (GBp), not pounds. Callers that convert
# absolute prices should be aware; we expose the currency as GBP and leave pence
# handling to the price source, which uses Yahoo adjusted values consistently.


def currency_for_symbol(symbol: str) -> str:
    """Return the local trading currency for a Yahoo symbol (default USD)."""
    for suffix, currency in SUFFIX_CURRENCY.items():
        if symbol.upper().endswith(suffix):
            return currency
    return "USD"


RateSource = Callable[[str, str], float | None]


def _yahoo_rate_source(base: str, quote: str) -> float | None:
    """Latest FX rate to convert ``base`` into ``quote`` via Yahoo ``=X``."""
    if base == quote:
        return 1.0
    import yfinance as yf

    pair = f"{base}{quote}=X"
    try:
        data = yf.Ticker(pair).history(period="5d")
    except Exception:
        return None
    if data is None or data.empty or "Close" not in data.columns:
        return None
    return float(data["Close"].dropna().iloc[-1])


class FxConverter:
    """Convert amounts between currencies with a caching, injectable source."""

    def __init__(self, base_currency: str, rate_source: RateSource | None = None):
        self.base_currency = base_currency.upper()
        self._rate_source = rate_source or _yahoo_rate_source
        self._cache: dict[tuple[str, str], float | None] = {}

    def rate(self, from_currency: str, to_currency: str) -> float | None:
        """Rate that multiplies a ``from_currency`` amount into ``to_currency``."""
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        if from_currency == to_currency:
            return 1.0
        key = (from_currency, to_currency)
        if key not in self._cache:
            rate = self._rate_source(from_currency, to_currency)
            if rate is None:
                # Try the inverse pair before giving up.
                inverse = self._rate_source(to_currency, from_currency)
                rate = (1.0 / inverse) if inverse else None
            self._cache[key] = rate
        return self._cache[key]

    def to_base(self, amount: float, from_currency: str) -> float | None:
        rate = self.rate(from_currency, self.base_currency)
        return None if rate is None else amount * rate

    def rates_report(self, currencies: list[str]) -> dict[str, Any]:
        """Report the rate used for each source currency (for artifacts/UI)."""
        report = {}
        for currency in sorted(set(currencies)):
            report[currency] = {
                "to": self.base_currency,
                "rate": self.rate(currency, self.base_currency),
            }
        return report
