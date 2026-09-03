"""Rank growth candidates from congressional stock disclosures.

Two stages, deliberately separate so the expensive one is opt-in:

1. ``scan`` hits the disclosure sources, aggregates every readable trade by
   ticker, and scores each one. No LLM, no cost.
2. ``rank_top3`` hands the shortlist to an LLM, which picks three names and
   argues for them. One call.

What the score can and cannot mean: a Periodic Transaction Report arrives up
to 45 days after the trade and discloses a size band, never an amount. So the
ranking answers "where has elected-official buying clustered recently", which
is a starting point for research -- not a forecast, and not a signal you can
front-run.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, Field

from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.dataflows.congress_trades import (
    CongressTrade,
    fetch_congress_trades,
)

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 60
DEFAULT_SHORTLIST_SIZE = 10
PICK_COUNT = 3

# How the score is built, in one place so the ranking can be argued with.
#
# Size enters on a log scale, measured in decades above the smallest band a
# member can disclose, so one minimum-sized buy scores 1.0 and every tenfold
# increase adds another point. Taking a plain log of the dollar amount instead
# compresses the range so hard that a $8k purchase outranks a $1.1M one on a
# single extra buyer, which is the wrong answer.
#
# Breadth multiplies rather than adds: one member buying is a portfolio
# decision, five buying the same name inside a month is a pattern, and that is
# the part of the signal hardest to explain away. Recency decays because a
# month-old disclosure of a month-old trade is already stale. Option purchases
# earn a premium -- a member reaching for leverage is expressing more
# conviction than one topping up a holding.
SIZE_REFERENCE_DOLLARS = 8_000.0  # midpoint of the smallest band, $1,001-$15,000
BREADTH_WEIGHT = 0.5
RECENCY_HALF_LIFE_DAYS = 30.0
OPTION_PREMIUM = 0.25

# Trades listed per ticker in the LLM's evidence block. Enough to show a
# cluster's shape without spending the context window on a single filer's
# 27-line portfolio rebalance.
EVIDENCE_TRADES_PER_TICKER = 6


class CongressPick(BaseModel):
    """One ranked growth candidate."""

    ticker: str = Field(
        description="Ticker symbol, exactly as it appears in the shortlist."
    )
    company: str = Field(description="Company name.")
    growth_thesis: str = Field(
        description=(
            "Two to four sentences on why this name could run from here. Argue "
            "from the disclosure pattern you were given plus what you know of "
            "the company and its sector. Do not invent prices, earnings "
            "figures, or trades that are not in the evidence."
        )
    )
    disclosure_evidence: str = Field(
        description=(
            "The specific filings behind the pick: which members bought, when, "
            "in what size bands, and whether any used options. Quote only what "
            "the shortlist shows."
        )
    )
    key_risk: str = Field(
        description=(
            "The most likely reason this pick disappoints, in one or two "
            "sentences. Include the disclosure lag if it is material here."
        )
    )


class CongressPicks(BaseModel):
    """The ranked shortlist verdict."""

    picks: list[CongressPick] = Field(
        description=(
            f"Exactly {PICK_COUNT} candidates, ordered strongest growth "
            f"potential first."
        )
    )
    what_would_change_this: str = Field(
        description=(
            "One paragraph: what new information would most change this "
            "ranking, and what the disclosure data structurally cannot tell you."
        )
    )


@dataclass(frozen=True)
class TickerSignal:
    """Every readable congressional trade in one ticker, aggregated."""

    ticker: str
    company: str
    buy_dollars: float
    sell_dollars: float
    buyers: tuple[str, ...]
    sellers: tuple[str, ...]
    option_buys: int
    latest_trade_date: date
    latest_filing_date: date
    trades: tuple[CongressTrade, ...]
    score: float

    @property
    def net_dollars(self) -> float:
        return self.buy_dollars - self.sell_dollars


@dataclass(frozen=True)
class Ranking:
    """The LLM's verdict on a shortlist."""

    markdown: str
    tickers: tuple[str, ...]
    """The picked tickers. Empty when the model answered as free text, in which
    case there is no reliable list to spend a full analysis run on."""


@dataclass(frozen=True)
class Scan:
    """Result of one pass over the disclosure sources."""

    as_of: date
    lookback_days: int
    source_notes: tuple[str, ...]
    signals: tuple[TickerSignal, ...]

    @property
    def buying(self) -> tuple[TickerSignal, ...]:
        """Net-bought names, strongest signal first."""
        return tuple(s for s in self.signals if s.net_dollars > 0)

    @property
    def selling(self) -> tuple[TickerSignal, ...]:
        """Net-sold names, heaviest selling first."""
        return tuple(
            sorted(
                (s for s in self.signals if s.net_dollars < 0),
                key=lambda s: s.net_dollars,
            )
        )

    def shortlist(self, size: int = DEFAULT_SHORTLIST_SIZE) -> tuple[TickerSignal, ...]:
        return self.buying[:size]


def _score(
    *,
    net_buy_dollars: float,
    buyer_count: int,
    days_since_trade: float,
    has_option_buys: bool,
) -> float:
    """Conviction-weighted score for net buying in one ticker.

    Zero for anything not net bought: this ranks growth candidates, and a name
    Congress is net selling is not one.
    """
    if net_buy_dollars <= 0 or buyer_count == 0:
        return 0.0

    size = max(0.0, math.log10(net_buy_dollars / SIZE_REFERENCE_DOLLARS) + 1)
    breadth = 1 + BREADTH_WEIGHT * (buyer_count - 1)
    recency = 0.5 ** (max(days_since_trade, 0) / RECENCY_HALF_LIFE_DAYS)
    conviction = 1 + (OPTION_PREMIUM if has_option_buys else 0)
    return size * breadth * recency * conviction


def aggregate(
    trades: list[CongressTrade], *, as_of: date | None = None
) -> list[TickerSignal]:
    """Group trades by ticker and score each, strongest signal first."""
    as_of = as_of or date.today()
    by_ticker: dict[str, list[CongressTrade]] = {}
    for trade in trades:
        by_ticker.setdefault(trade.ticker, []).append(trade)

    signals = []
    for ticker, group in by_ticker.items():
        buys = [t for t in group if t.action == "buy"]
        sells = [t for t in group if t.action == "sell"]
        buy_dollars = sum(t.amount_mid for t in buys)
        sell_dollars = sum(t.amount_mid for t in sells)
        buyers = tuple(sorted({t.member for t in buys}))
        option_buys = sum(1 for t in buys if t.asset_code == "OP")

        # The most-repeated description wins: filers write the same holding a
        # few different ways and the common spelling is the readable one.
        company = Counter(t.asset for t in group if t.asset).most_common(1)
        latest_buy = max((t.trade_date for t in buys), default=None)

        signals.append(
            TickerSignal(
                ticker=ticker,
                company=company[0][0] if company else ticker,
                buy_dollars=buy_dollars,
                sell_dollars=sell_dollars,
                buyers=buyers,
                sellers=tuple(sorted({t.member for t in sells})),
                option_buys=option_buys,
                latest_trade_date=max(t.trade_date for t in group),
                latest_filing_date=max(t.filing_date for t in group),
                trades=tuple(sorted(group, key=lambda t: t.trade_date, reverse=True)),
                score=_score(
                    net_buy_dollars=buy_dollars - sell_dollars,
                    buyer_count=len(buyers),
                    days_since_trade=(
                        (as_of - latest_buy).days if latest_buy else 0.0
                    ),
                    has_option_buys=option_buys > 0,
                ),
            )
        )

    signals.sort(key=lambda s: (s.score, s.net_dollars), reverse=True)
    return signals


def scan(lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> Scan:
    """Fetch and score every congressional trade filed in the lookback window."""
    trades, notes = fetch_congress_trades(lookback_days)
    as_of = date.today()
    return Scan(
        as_of=as_of,
        lookback_days=lookback_days,
        source_notes=tuple(notes),
        signals=tuple(aggregate(trades, as_of=as_of)),
    )


def render_evidence(scan_result: Scan, size: int = DEFAULT_SHORTLIST_SIZE) -> str:
    """The shortlist as markdown -- both the LLM's evidence and a readable table."""
    lines = [
        f"## Congressional buying, {scan_result.lookback_days} days to "
        f"{scan_result.as_of.isoformat()}",
        "",
        "Sources:",
    ]
    lines.extend(f"- {note}" for note in scan_result.source_notes)
    lines.append("")

    shortlist = scan_result.shortlist(size)
    if not shortlist:
        lines.append(
            "No net buying in any single stock was disclosed in this window."
        )
        return "\n".join(lines)

    lines.append("### Net-bought names, strongest disclosure signal first")
    lines.append("")
    for rank, signal in enumerate(shortlist, start=1):
        options = (
            f", {signal.option_buys} via options" if signal.option_buys else ""
        )
        lines.append(
            f"**{rank}. {signal.ticker} — {signal.company}** "
            f"(score {signal.score:.2f}) — net +${signal.net_dollars:,.0f} "
            f"from {len(signal.buyers)} buyer(s){options}; most recent trade "
            f"{signal.latest_trade_date.isoformat()}, filed "
            f"{signal.latest_filing_date.isoformat()}."
        )
        for trade in signal.trades[:EVIDENCE_TRADES_PER_TICKER]:
            lines.append(
                f"    - {trade.member} ({trade.constituency}) {trade.action} "
                f"{trade.asset_code} on {trade.trade_date.isoformat()}, "
                f"${trade.amount_low:,}-${trade.amount_high:,}, disclosed "
                f"{trade.disclosure_lag_days} days later"
            )
        lines.append("")

    selling = scan_result.selling[:5]
    if selling:
        lines.append("### Heaviest net selling, for context")
        lines.append("")
        lines.extend(
            f"- **{s.ticker} — {s.company}**: net -${abs(s.net_dollars):,.0f} "
            f"from {len(s.sellers)} seller(s)"
            for s in selling
        )
        lines.append("")

    return "\n".join(lines)


def render_picks(picks: CongressPicks) -> str:
    """Render the LLM verdict to the markdown the dashboard displays."""
    lines = ["## Top growth candidates from congressional disclosures", ""]
    for rank, pick in enumerate(picks.picks, start=1):
        lines.extend(
            [
                f"### {rank}. {pick.ticker} — {pick.company}",
                "",
                f"**Growth thesis.** {pick.growth_thesis}",
                "",
                f"**Disclosure evidence.** {pick.disclosure_evidence}",
                "",
                f"**Key risk.** {pick.key_risk}",
                "",
            ]
        )
    lines.extend(["### What would change this", "", picks.what_would_change_this])
    return "\n".join(lines)


_PROMPT = """You are screening US equities for growth potential, using the \
Periodic Transaction Reports that members of Congress must file under the \
STOCK Act. Today is {today}.

Pick the {count} names from the shortlist below with the strongest chance of a \
sustained move up from here, and argue for each. Rank them strongest first.

How to read the evidence. Each name is scored on how much net buying was \
disclosed, how many separate members bought it, how recently, and whether any \
used options. Breadth matters more than size: several members independently \
buying the same name is a stronger signal than one member buying a lot. \
Options carry more conviction than shares.

What this data cannot tell you, and what you must not pretend it does. \
Filings arrive up to 45 days after the trade, so every entry is old news and \
none of it is front-runnable. Sizes are disclosed as bands, never as amounts, \
so a "net dollar" figure is a midpoint estimate. Many trades are made by \
advisors or in blind trusts, so a filing is not necessarily the member's own \
view. A member buying a stock is not evidence of anything improper and you \
should not imply that it is.

Judge growth potential on the merits -- the company, its sector, its \
catalysts -- with the disclosure pattern as the reason these names are in \
front of you rather than as the thesis itself. Do not invent prices, earnings \
numbers, analyst targets, or any trade that is not listed below.

{evidence}{insider_evidence}"""

_INSIDER_PREAMBLE = """
---

The section below covers the same shortlist from a second, independent angle: \
open-market purchases by the companies' own officers and directors, filed on \
SEC Form 4 within two business days of the trade. Only purchases made with the \
insider's own cash are listed; grants, option exercises and tax withholding \
are excluded because they involve no decision to put money at risk.

Weight this heavily where it appears. A name bought by both elected officials \
and the executives who run the company is corroborated by two groups with \
entirely different information and motives, and several insiders at one \
company buying inside a fortnight is the single best-documented pattern in \
this kind of data. Purchases marked 10b5-1 were scheduled months in advance \
and say nothing about current conviction, so discount them. Insider selling is \
weak evidence of anything -- executives sell to diversify and to pay tax -- so \
do not read much into it either way.

A shortlist name absent from this section simply had no insider purchases in \
the window, which is the normal case and not a mark against it.

"""


def build_ranking_llm(config: dict | None = None):
    """The configured deep-thinking model, for the ranking call."""
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.llm_clients import create_llm_client

    config = config or DEFAULT_CONFIG
    return create_llm_client(
        provider=config["llm_provider"],
        model=config["deep_think_llm"],
        base_url=config.get("backend_url"),
    ).get_llm()


def rank_top3(
    scan_result: Scan,
    *,
    llm=None,
    size: int = DEFAULT_SHORTLIST_SIZE,
    config: dict | None = None,
    insider_evidence: str = "",
) -> Ranking:
    """Ask the LLM to pick and argue for the strongest few names.

    ``insider_evidence`` is optional Form 4 markdown covering the same
    shortlist, which lets the model see where company insiders bought the names
    Congress did. It only ever adds context: the candidates are still the
    congressional shortlist.

    Raises ``ValueError`` when the scan found no net-bought names, so the
    caller reports an empty window rather than inviting the model to invent
    candidates.

    Picks are intersected with the shortlist before being returned: a ticker
    the model invented would otherwise send a full, paid analysis run after a
    symbol nobody disclosed.
    """
    shortlist = scan_result.shortlist(size)
    if not shortlist:
        raise ValueError(
            "No net congressional buying in any single stock was disclosed in "
            "the lookback window, so there is nothing to rank. Widen the "
            "lookback and scan again."
        )

    llm = llm if llm is not None else build_ranking_llm(config)
    prompt = _PROMPT.format(
        today=scan_result.as_of.isoformat(),
        count=PICK_COUNT,
        evidence=render_evidence(scan_result, size),
        insider_evidence=(
            f"\n{_INSIDER_PREAMBLE}{insider_evidence}" if insider_evidence else ""
        ),
    )

    # The render callback is the only place the parsed result is available, and
    # it does not fire on the free-text fallback -- which is exactly when the
    # ticker list should come back empty.
    parsed: list[CongressPicks] = []

    def render(result: CongressPicks) -> str:
        parsed.append(result)
        return render_picks(result)

    markdown = invoke_structured_or_freetext(
        bind_structured(llm, CongressPicks, "Congress screener"),
        llm,
        prompt,
        render,
        "Congress screener",
    )

    offered = {s.ticker for s in shortlist}
    tickers: tuple[str, ...] = ()
    if parsed:
        picked = [p.ticker.strip().upper() for p in parsed[0].picks]
        unknown = [t for t in picked if t not in offered]
        if unknown:
            logger.warning(
                "Congress screener picked tickers absent from the shortlist, "
                "dropping them: %s", ", ".join(unknown),
            )
        tickers = tuple(t for t in picked if t in offered)[:PICK_COUNT]

    return Ranking(markdown=markdown, tickers=tickers)
