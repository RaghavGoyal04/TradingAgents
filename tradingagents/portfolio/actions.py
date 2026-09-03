"""Turn run artifacts into a prioritized, plain-language "what to do next" list.

Kept out of the Streamlit layer so the rules are unit-testable. Every item is
advisory: nothing here places an order. Items are returned already sorted, most
important first.
"""

from __future__ import annotations

from typing import Any

PRIORITY_ORDER = {"high": 0, "medium": 1, "info": 2}

# Thresholds that turn portfolio structure into an actionable flag.
TOP5_CONCENTRATION_LIMIT = 0.40
SINGLE_POSITION_LIMIT = 0.15
HIGH_LOSS_PROBABILITY = 0.50

# A pie holding must drift at least this far from its own target share, and be
# worth at least this much, before it is worth a trade. Below these it is noise
# and the dealing cost/effort outweighs the correction.
DRIFT_THRESHOLD = 0.02
MIN_TRADE_VALUE = 25.0


# Conviction is derived purely from how far the analysts' price target sits from
# today's price. A move of this size or more scores 100; the score is a rescaled
# distance to target, NOT a probability.
CONVICTION_FULL_SCALE_UPSIDE = 0.40
HIGH_CONVICTION_SCORE = 50  # equals a 20% move to target
MEDIUM_CONVICTION_SCORE = 25  # equals a 10% move to target

# Band ordering for the default sort: strongest conviction at the top.
CONVICTION_RANK = {"High": 0, "Medium": 1, "Low": 2, "Unknown": 3}

# A price target more than this far from the current price almost certainly
# means the two numbers are quoted in different units (e.g. GBP vs GBX pence),
# so the upside is suppressed rather than shown as nonsense.
_MAX_PLAUSIBLE_RATIO = 5.0


def decision_table(
    snapshot: dict[str, Any] | None,
    recommendations: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Actionable buy/sell calls, strongest conviction first.

    Sorted by conviction band, then by money at stake within each band, so the
    top row is the most convincing call that also moves the portfolio. Both
    conviction and upside come from the analysts' own price target, never from
    a guess.
    """
    snapshot = snapshot or {}
    recommendations = recommendations or {}
    prices = {
        p.get("symbol"): p.get("current_price")
        for p in snapshot.get("positions", [])
        if p.get("symbol")
    }
    names = {
        p.get("symbol"): p.get("name") or p.get("symbol")
        for p in snapshot.get("positions", [])
    }

    rows = []
    for item in recommendations.get("results") or []:
        if item.get("status") != "success":
            continue
        action = (item.get("action") or "").upper()
        if action not in ("BUY", "SELL"):
            continue
        ticker = item.get("ticker")
        upside = _upside(item.get("price_target"), prices.get(ticker))
        score = _conviction_score(upside, action)
        rows.append(
            {
                "action": action,
                "ticker": ticker,
                "name": names.get(ticker) or item.get("company") or ticker,
                "amount": float(item.get("trade_value") or 0.0),
                "conviction": _conviction_band(score),
                "conviction_score": score,
                "upside": upside,
                "price_target": _to_float(item.get("price_target")),
                "current_price": prices.get(ticker),
                "horizon": item.get("time_horizon"),
                "rating": item.get("rating"),
                "sizing": item.get("sizing"),
                "plan": item.get("executive_summary") or "",
                "thesis": item.get("investment_thesis") or "",
            }
        )
    # Conviction band first, then money at stake within the band.
    rows.sort(
        key=lambda r: (CONVICTION_RANK.get(r["conviction"], 9), -r["amount"])
    )
    return rows


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _upside(price_target: Any, current_price: Any) -> float | None:
    """Fractional move from today's price to the analysts' target."""
    target = _to_float(price_target)
    current = _to_float(current_price)
    if not target or not current or current <= 0:
        return None
    ratio = target / current
    if ratio > _MAX_PLAUSIBLE_RATIO or ratio < 1 / _MAX_PLAUSIBLE_RATIO:
        return None  # unit mismatch, not a real 400% call
    return ratio - 1.0


def _conviction_score(upside: float | None, action: str) -> int | None:
    """0-100 rescaling of the distance to the price target.

    Deliberately NOT a probability: it says how far the analysts think the
    price can travel, not how likely they are to be right.

    Scored only when the target points the same way as the trade. On sells the
    target is often an exit level ABOVE today's price ("sell into the 44-46
    resistance"), which says nothing about downside -- treating that as
    conviction would rank a trim as if it were a strong directional call.
    """
    if upside is None:
        return None
    if (action == "BUY" and upside <= 0) or (action == "SELL" and upside >= 0):
        return None
    return min(100, round(abs(upside) / CONVICTION_FULL_SCALE_UPSIDE * 100))


def _conviction_band(score: int | None) -> str:
    """Band label derived from the score so the two can never disagree."""
    if score is None:
        return "Unknown"
    if score >= HIGH_CONVICTION_SCORE:
        return "High"
    if score >= MEDIUM_CONVICTION_SCORE:
        return "Medium"
    return "Low"


def rebalance_plan(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Buy/trim amounts that pull each pie back to ITS OWN target weights.

    This deliberately uses the targets you already set in Trading 212 rather
    than any view of our own: the drift between ``target_share`` and
    ``current_share`` is a fact about your stated plan, not a prediction. Rows
    are returned worst-drift-first.
    """
    snapshot = snapshot or {}
    pies = snapshot.get("pies") or []
    names = {
        p.get("broker_ticker"): p.get("name")
        for p in snapshot.get("positions", [])
        if p.get("broker_ticker")
    }
    rows: list[dict[str, Any]] = []
    for pie in pies:
        instruments = pie.get("instruments") or []
        pie_value = sum(float(i.get("value") or 0.0) for i in instruments)
        if pie_value <= 0:
            continue
        for instrument in instruments:
            target = instrument.get("target_share")
            current = instrument.get("current_share")
            if target is None or current is None:
                continue
            drift = float(current) - float(target)
            amount = -drift * pie_value  # positive => buy, negative => trim
            if abs(drift) < DRIFT_THRESHOLD or abs(amount) < MIN_TRADE_VALUE:
                action = "HOLD"
            else:
                action = "BUY" if amount > 0 else "TRIM"
            ticker = instrument.get("broker_ticker")
            rows.append(
                {
                    "pie": pie.get("name"),
                    "broker_ticker": ticker,
                    "name": names.get(ticker) or ticker,
                    "action": action,
                    "target_share": float(target),
                    "current_share": float(current),
                    "drift": drift,
                    "amount": amount,
                    "value": float(instrument.get("value") or 0.0),
                }
            )
    rows.sort(key=lambda r: abs(r["drift"]), reverse=True)
    return rows


def next_day_actions(
    snapshot: dict[str, Any] | None,
    forecasts: dict[str, Any] | None,
    recommendations: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Return prioritized advisory actions for the next trading day."""
    snapshot = snapshot or {}
    forecasts = forecasts or {}
    recommendations = recommendations or {}

    holdings = [
        p for p in snapshot.get("positions", []) if not p.get("watch_only")
    ]
    names = {p.get("symbol"): p.get("name") or p.get("symbol") for p in holdings}
    currency = snapshot.get("base_currency", "")
    actions: list[dict[str, str]] = []

    actions.extend(_trade_actions(recommendations))
    actions.extend(_concentration_actions(holdings, names))
    actions.extend(_data_quality_actions(snapshot, forecasts))
    actions.extend(_risk_actions(forecasts, currency))

    actions.sort(key=lambda a: PRIORITY_ORDER.get(a["priority"], 9))
    return actions


def _trade_actions(recommendations: dict[str, Any]) -> list[dict[str, str]]:
    """Only the 'is there analysis at all' note.

    The trades themselves live in :func:`decision_table`, which ranks them;
    repeating them here just buries the housekeeping items.
    """
    results = recommendations.get("results") or []
    if not results:
        return [
            {
                "priority": "info",
                "title": "No agent analysis in this run",
                "detail": (
                    "This run skipped the AI analyst team, so there are no "
                    "buy/sell calls. Choose 'Full analysis' in the sidebar and "
                    "press Run to get per-holding recommendations."
                ),
            }
        ]
    if not any((r.get("action") or "").upper() in ("BUY", "SELL") for r in results):
        return [
            {
                "priority": "info",
                "title": "No buy or sell calls",
                "detail": (
                    "The analyst team rated every holding hold/neutral, so no "
                    "trade is required."
                ),
            }
        ]
    return []


def _concentration_actions(
    holdings: list[dict[str, Any]], names: dict[str, str]
) -> list[dict[str, str]]:
    actions = []
    weights = sorted(
        ((p.get("weight") or 0.0), p.get("symbol")) for p in holdings
    )[::-1]
    top5 = sum(w for w, _ in weights[:5])
    if top5 > TOP5_CONCENTRATION_LIMIT:
        biggest = ", ".join(
            f"{names.get(s, s)} ({w:.1%})" for w, s in weights[:5] if s
        )
        actions.append(
            {
                "priority": "medium",
                "title": f"Your top 5 holdings are {top5:.0%} of the account",
                "detail": (
                    f"Concentrated in {biggest}. A single bad day in these moves "
                    "the whole portfolio. Consider trimming toward your target "
                    "weights."
                ),
            }
        )
    for weight, symbol in weights:
        if weight > SINGLE_POSITION_LIMIT and symbol:
            actions.append(
                {
                    "priority": "medium",
                    "title": (
                        f"{names.get(symbol, symbol)} alone is {weight:.0%} "
                        "of the account"
                    ),
                    "detail": (
                        f"Above the {SINGLE_POSITION_LIMIT:.0%} single-position "
                        "guideline. Consider trimming or hedging."
                    ),
                }
            )
    return actions


def _data_quality_actions(
    snapshot: dict[str, Any], forecasts: dict[str, Any]
) -> list[dict[str, str]]:
    actions = []
    unmapped = snapshot.get("unmapped") or []
    if unmapped:
        actions.append(
            {
                "priority": "high",
                "title": f"{len(unmapped)} holding(s) could not be priced",
                "detail": (
                    f"{', '.join(unmapped)} have no market symbol, so they are "
                    "excluded from forecasts and risk. Their value still counts "
                    "toward your account total."
                ),
            }
        )
    for warning in forecasts.get("warnings") or []:
        if "insufficient history" in warning:
            actions.append(
                {
                    "priority": "info",
                    "title": "Some holdings lack enough price history",
                    "detail": (
                        f"{warning}. These are left out of the risk simulation, "
                        "so the numbers below cover the rest of the portfolio."
                    ),
                }
            )
    return actions


def _risk_actions(forecasts: dict[str, Any], currency: str) -> list[dict[str, str]]:
    portfolio = forecasts.get("portfolio") or {}
    if not portfolio:
        return []
    loss = portfolio.get("loss_probability")
    var = portfolio.get("var_95")
    actions = []
    if isinstance(loss, (int, float)) and loss > HIGH_LOSS_PROBABILITY:
        actions.append(
            {
                "priority": "medium",
                "title": f"More likely than not to be down in 20 days ({loss:.0%})",
                "detail": (
                    "The simulation says a loss is the more likely outcome over "
                    "the next month of trading. Size new buys accordingly."
                ),
            }
        )
    if isinstance(var, (int, float)):
        actions.append(
            {
                "priority": "info",
                "title": f"Plan for a bad month of about {currency} {var:,.0f}",
                "detail": (
                    "On the worst 1 day in 20 (5% of simulated outcomes), losses "
                    "over the next 20 trading days reach at least this much. Make "
                    "sure you are comfortable holding through it."
                ),
            }
        )
    return actions
