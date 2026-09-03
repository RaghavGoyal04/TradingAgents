"""Trading212 read-only client and portfolio snapshot builder.

This is the single, canonical Trading212 integration. It is deliberately
GET-only: the allowed endpoints are enumerated and every request is checked
against that allowlist, so no code path can ever place, modify, or cancel an
order. ``scripts/run_watchlist.py`` re-exports the symbol mapping from here so
there is one mapping implementation (the staff review flagged the previous
duplicate).
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from ..dataflows.symbol_utils import normalize_symbol
from . import SCHEMA_VERSION
from .contracts import PORTFOLIO_SNAPSHOT, atomic_write_json

logger = logging.getLogger(__name__)

TRADING212_BASE_URL = "https://live.trading212.com/api/v0"
TRADING212_TIMEOUT = 20

# Strict allowlist of the only endpoints this client may ever call. All are
# read-only. A guard rejects anything outside this set, so an accidental order
# endpoint cannot be introduced without deleting it from here on purpose.
ALLOWED_ENDPOINTS = frozenset(
    {
        "/equity/account/summary",
        "/equity/account/cash",
        "/equity/positions",
        "/equity/pies",
    }
)

# Pie detail endpoints carry a numeric id, so they cannot be enumerated above.
# This pattern is still read-only and is the ONLY dynamic path permitted.
_ALLOWED_ENDPOINT_PATTERN = re.compile(r"^/equity/pies/\d+$")

# The pies endpoints rate-limit far more aggressively than positions.
_PIE_RETRIES = 3
_PIE_BACKOFF_SECONDS = 6

# Exact broker-ticker aliases for instruments whose T212 identifier carries no
# recognizable market marker (e.g. LSE-listed UCITS ETFs encoded as a bare
# ``_EQ``). Checked before the suffix rules. Extend with verified mappings only.
_T212_TICKER_ALIASES = {
    "CNX1_EQ": "CNX1.L",  # iShares NASDAQ 100 UCITS ETF (LSE, quoted in GBp)
    # USA Rare Earth began trading as USAR after the Inflection Point
    # Acquisition Corp. II merger; Trading212 still exposes the legacy ticker.
    "IPXX_US_EQ": "USAR",
}

# Trading212 broker ticker suffix -> Yahoo Finance suffix. Order matters: the
# first matching suffix wins, so multi-char suffixes precede shorter ones.
_TRADING212_YAHOO_SUFFIXES = (
    ("_US_EQ", ""),
    ("_CA_EQ", ".TO"),
    ("_AT_EQ", ".VI"),
    ("_BE_EQ", ".BR"),
    ("_PT_EQ", ".LS"),
    ("l_EQ", ".L"),
    ("d_EQ", ".DE"),
    ("p_EQ", ".PA"),
    ("a_EQ", ".AS"),
    ("s_EQ", ".SW"),
    ("e_EQ", ".MC"),
    ("m_EQ", ".MI"),
)


class Trading212ConfigError(RuntimeError):
    """Raised when Trading212 credentials are missing."""


def trading212_to_yahoo_symbol(ticker: str) -> str | None:
    """Convert a Trading212 equity identifier to a Yahoo Finance symbol.

    Returns ``None`` when the broker suffix is unrecognized so the caller can
    record the position as unmapped rather than guessing a wrong symbol. The
    result is passed through :func:`normalize_symbol` so both mapping paths stay
    consistent.
    """
    if ticker in _T212_TICKER_ALIASES:
        return normalize_symbol(_T212_TICKER_ALIASES[ticker])
    for suffix, yahoo_suffix in _TRADING212_YAHOO_SUFFIXES:
        if ticker.endswith(suffix):
            symbol = ticker[: -len(suffix)]
            if not yahoo_suffix:
                symbol = symbol.replace(".", "-")
            return normalize_symbol(f"{symbol.upper()}{yahoo_suffix}")
    return None


def _credentials() -> tuple[str, str]:
    if not os.getenv("TRADING212_API_KEY") or not os.getenv("TRADING212_API_SECRET"):
        # Late .env load keeps credentials out of the process env until needed.
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    api_key = os.getenv("TRADING212_API_KEY")
    api_secret = os.getenv("TRADING212_API_SECRET")
    if not api_key or not api_secret:
        raise Trading212ConfigError(
            "TRADING212_API_KEY and TRADING212_API_SECRET must be set to read the "
            "Trading212 account."
        )
    return api_key, api_secret


def _get(endpoint: str, auth: tuple[str, str], *, session: Any = requests) -> Any:
    """Perform a guarded GET against an allowlisted Trading212 endpoint."""
    if endpoint not in ALLOWED_ENDPOINTS and not _ALLOWED_ENDPOINT_PATTERN.match(
        endpoint
    ):
        raise ValueError(
            f"refusing to call non-allowlisted Trading212 endpoint {endpoint!r}"
        )
    response = session.get(
        f"{TRADING212_BASE_URL}{endpoint}", auth=auth, timeout=TRADING212_TIMEOUT
    )
    response.raise_for_status()
    return response.json()


def _get_with_backoff(
    endpoint: str, auth: tuple[str, str], *, session: Any = requests
) -> Any | None:
    """GET an allowlisted endpoint, retrying the pies rate limit.

    Returns ``None`` instead of raising when the endpoint stays unavailable:
    pies are enrichment, and losing them must never fail a portfolio run.
    """
    for attempt in range(_PIE_RETRIES):
        try:
            return _get(endpoint, auth, session=session)
        except ValueError:
            raise  # allowlist violation is a bug, not a transient failure
        except Exception as exc:  # noqa: BLE001 - transient HTTP/rate-limit
            if attempt == _PIE_RETRIES - 1:
                logger.warning("Trading212 %s unavailable (%s).", endpoint, exc)
                return None
            time.sleep(_PIE_BACKOFF_SECONDS * (attempt + 1))
    return None


def fetch_pies(auth: tuple[str, str], *, session: Any = requests) -> list[dict[str, Any]]:
    """Fetch each pie with its name and per-instrument target vs actual shares.

    ``expectedShare``/``currentShare`` are shares WITHIN the pie (they sum to 1
    per pie), not shares of the whole account. Returns ``[]`` when pies are
    unavailable so callers degrade to an ungrouped view.
    """
    listing = _get_with_backoff("/equity/pies", auth, session=session)
    if not isinstance(listing, list):
        return []
    pies = []
    for entry in listing:
        pie_id = entry.get("id")
        if pie_id is None:
            continue
        detail = _get_with_backoff(f"/equity/pies/{pie_id}", auth, session=session)
        if not isinstance(detail, dict):
            continue
        settings = detail.get("settings") or {}
        instruments = []
        for instrument in detail.get("instruments") or []:
            result = instrument.get("result") or {}
            instruments.append(
                {
                    "broker_ticker": instrument.get("ticker"),
                    "target_share": instrument.get("expectedShare"),
                    "current_share": instrument.get("currentShare"),
                    "value": result.get("priceAvgValue"),
                }
            )
        pies.append(
            {
                "id": pie_id,
                "name": settings.get("name") or f"Pie {pie_id}",
                "cash": entry.get("cash") or 0.0,
                "instruments": instruments,
            }
        )
    return pies


def _attach_pies(
    positions: list[dict[str, Any]], pies: list[dict[str, Any]]
) -> None:
    """Annotate each position with the pie(s) holding it and its target share."""
    membership: dict[str, list[dict[str, Any]]] = {}
    for pie in pies:
        for instrument in pie["instruments"]:
            ticker = instrument["broker_ticker"]
            if not ticker:
                continue
            membership.setdefault(ticker, []).append(
                {
                    "pie": pie["name"],
                    "target_share": instrument["target_share"],
                    "current_share": instrument["current_share"],
                }
            )
    for position in positions:
        entries = membership.get(position["broker_ticker"], [])
        position["pies"] = entries
        position["pie"] = (
            " + ".join(e["pie"] for e in entries) if entries else "Not in a pie"
        )


def fetch_portfolio_snapshot(
    *,
    watchlist: list[str] | None = None,
    session: Any = requests,
    include_pies: bool = True,
) -> dict[str, Any]:
    """Fetch a structured, read-only portfolio snapshot from Trading212.

    The snapshot conforms to the ``portfolio_snapshot.json`` schema. Positions
    that cannot be mapped to a Yahoo symbol are still recorded (with
    ``mapping_status="unmapped"``) and their broker tickers listed under
    ``unmapped`` so the orchestrator can fail the run with a precise message.
    """
    auth = _credentials()
    summary = _get("/equity/account/summary", auth, session=session)
    positions_raw = _get("/equity/positions", auth, session=session)

    account_value = float(summary["totalValue"])
    currency = str(summary["currency"])

    positions: list[dict[str, Any]] = []
    unmapped: list[str] = []
    positions_value = 0.0
    for position in positions_raw:
        instrument = position["instrument"]
        broker_ticker = instrument["ticker"]
        value = float(position["walletImpact"]["currentValue"])
        positions_value += value
        symbol = trading212_to_yahoo_symbol(broker_ticker)
        quantity = position.get("quantity")
        entry = {
            "broker_ticker": broker_ticker,
            "symbol": symbol,
            # Human-readable instrument name straight from the broker, so the UI
            # never has to show a bare ticker.
            "name": instrument.get("name") or (symbol or broker_ticker),
            "isin": instrument.get("isin"),
            "quantity": float(quantity) if quantity is not None else None,
            # ``value`` is in the ACCOUNT currency (walletImpact); the listing
            # itself trades in ``instrument_currency``. Keep both distinct.
            "value": value,
            "currency": currency,
            "instrument_currency": instrument.get("currency"),
            # Quoted in the instrument's own currency, so it is directly
            # comparable with the agents' price targets.
            "current_price": _as_float(position.get("currentPrice")),
            "average_price": _as_float(position.get("averagePricePaid")),
            "unrealized_pct": _unrealized_pct(position),
            "mapping_status": "mapped" if symbol else "unmapped",
            "watch_only": False,
        }
        positions.append(entry)
        if symbol is None:
            unmapped.append(broker_ticker)

    # Aggregate duplicate symbols (e.g. two lines that resolve to the same
    # listing) so weights and analysis treat each symbol once.
    positions = _aggregate_by_symbol(positions)

    for entry in positions:
        entry["weight"] = (
            round(entry["value"] / account_value, 6) if account_value else 0.0
        )

    pies = fetch_pies(auth, session=session) if include_pies else []
    _attach_pies(positions, pies)

    watch_entries = _watchlist_entries(watchlist, held={p["symbol"] for p in positions})

    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source": "trading212",
        "base_currency": currency,
        "account_value": account_value,
        "cash": round(account_value - positions_value, 2),
        "positions": positions + watch_entries,
        "unmapped": unmapped,
        "watchlist": [e["symbol"] for e in watch_entries],
        "pies": pies,
    }


def snapshot_from_holdings(
    holdings: dict[str, float],
    *,
    account_value: float | None,
    currency: str,
    watchlist: list[str] | None = None,
) -> dict[str, Any]:
    """Build a snapshot from a static holdings mapping (offline / no API).

    Symbols are assumed already in Yahoo form (the JSON holdings file the CLI
    accepts). This keeps the dashboard usable without live credentials.
    """
    positions_value = sum(holdings.values())
    total = account_value if account_value is not None else positions_value
    positions = [
        {
            "broker_ticker": symbol,
            "symbol": normalize_symbol(symbol),
            "name": normalize_symbol(symbol),
            "quantity": None,
            "value": float(value),
            "currency": currency,
            "mapping_status": "mapped",
            "watch_only": False,
            "weight": round(value / total, 6) if total else 0.0,
        }
        for symbol, value in holdings.items()
    ]
    watch_entries = _watchlist_entries(
        watchlist, held={p["symbol"] for p in positions}
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source": "holdings_file",
        "base_currency": currency,
        "account_value": total,
        "cash": round(total - positions_value, 2),
        "positions": positions + watch_entries,
        "unmapped": [],
        "watchlist": [e["symbol"] for e in watch_entries],
    }


def analyzed_symbols(snapshot: dict[str, Any]) -> list[str]:
    """Symbols the run should analyze: mapped, non-zero positions + watchlist.

    Zero-value positions are excluded (unless they were added as watch-only),
    matching the PRD: holdings drive the analyzed universe.
    """
    symbols: list[str] = []
    for position in snapshot.get("positions", []):
        if position["mapping_status"] != "mapped":
            continue
        if position.get("watch_only") or position["value"] > 0:
            symbols.append(position["symbol"])
    # Deduplicate while preserving order.
    return list(dict.fromkeys(symbols))


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unrealized_pct(position: dict[str, Any]) -> float | None:
    """Unrealized gain/loss as a fraction of cost.

    Uses the ratio of current to average paid price, both quoted in the
    instrument's own currency, so the result is currency-agnostic and safe to
    show next to an account-currency value.
    """
    try:
        paid = float(position["averagePricePaid"])
        current = float(position["currentPrice"])
    except (KeyError, TypeError, ValueError):
        return None
    if paid <= 0:
        return None
    return round(current / paid - 1.0, 6)


def _aggregate_by_symbol(positions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for position in positions:
        key = position["symbol"] or f"__unmapped__{position['broker_ticker']}"
        if key not in merged:
            merged[key] = dict(position)
            order.append(key)
            continue
        target = merged[key]
        # Value-weight the unrealized percentage before the values are summed.
        combined = target["value"] + position["value"]
        if (
            target.get("unrealized_pct") is not None
            and position.get("unrealized_pct") is not None
            and combined
        ):
            target["unrealized_pct"] = round(
                (
                    target["unrealized_pct"] * target["value"]
                    + position["unrealized_pct"] * position["value"]
                )
                / combined,
                6,
            )
        target["value"] = combined
        if position.get("quantity") is not None:
            target["quantity"] = (target.get("quantity") or 0.0) + position["quantity"]
    return [merged[key] for key in order]


def _watchlist_entries(
    watchlist: list[str] | None, *, held: set[str | None]
) -> list[dict[str, Any]]:
    if not watchlist:
        return []
    entries = []
    for raw in watchlist:
        symbol = normalize_symbol(raw)
        if symbol in held:
            continue  # already a real position; don't duplicate as watch-only
        entries.append(
            {
                "broker_ticker": symbol,
                "symbol": symbol,
                "name": symbol,
                "quantity": None,
                "value": 0.0,
                "currency": None,
                "weight": 0.0,
                "mapping_status": "mapped",
                "watch_only": True,
                "pie": "Watchlist",
                "pies": [],
            }
        )
        held.add(symbol)
    return entries


def write_snapshot(run_dir: str | Path, snapshot: dict[str, Any]) -> Path:
    path = Path(run_dir) / PORTFOLIO_SNAPSHOT
    atomic_write_json(path, snapshot)
    return path
