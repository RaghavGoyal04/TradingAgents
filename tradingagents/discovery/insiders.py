"""Rank growth candidates from SEC Form 4 corporate-insider filings.

Same two-stage shape as the congressional screener: a free, deterministic
``scan`` that aggregates and scores, then an optional LLM ranking.

The scoring encodes what practitioner and academic write-ups of Form 4 data
agree on, which is a narrower set of rules than the raw feed suggests:

* **Only open-market purchases count.** Transaction code ``P`` is the one where
  an insider chose to put their own cash at risk. Grants, option exercises,
  tax withholding and gifts make up most of the volume and mean nothing
  directionally. The vendor layer drops them before they reach here.
* **Cluster buying is the strongest signal in the dataset.** Three or more
  distinct insiders buying the same issuer inside a two-week window is harder
  to explain away than any single purchase, however large, and it is the
  pattern with the best-documented track record.
* **Rule 10b5-1 purchases are discounted heavily.** They were scheduled months
  in advance, so they carry no information about what the insider thinks now.
* **Who bought matters.** A CFO sees forward numbers first; directors buy for
  optics more often; a 10% holder topping up its own block is a different
  animal from an executive's judgement call.
* **Sales are kept but not scored.** Insiders sell to diversify, to pay tax
  and on schedule, so sales are far noisier than buys. They appear as context
  only.

What this cannot tell you: Form 4 says an insider bought, not that they were
right. Purchases cluster around price weakness, so a cluster is frequently a
bet on a company the market has already marked down.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta

from tradingagents.dataflows.sec_form4 import (
    InsiderTrade,
    fetch_company_trades,
    fetch_insider_trades,
)

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 14
DEFAULT_SHORTLIST_SIZE = 10

# Dollar floor a purchase is measured against. $100k is the level the
# literature consistently calls the start of a meaningful buy, so a $100k
# purchase scores 1.0 and every tenfold increase adds a point.
SIZE_REFERENCE_DOLLARS = 100_000.0

# Weight per additional distinct buyer. Higher than the congressional
# equivalent because co-workers buying the same stock in the same fortnight is
# a much tighter coincidence than two legislators doing so.
CLUSTER_WEIGHT = 0.75

# The window clusters are counted in, and the count at which one is declared.
CLUSTER_WINDOW_DAYS = 14
CLUSTER_THRESHOLD = 3
CLUSTER_BONUS = 0.5

# Form 4 lands within two business days, so staleness bites much faster here
# than on a congressional filing that was already six weeks old on arrival.
RECENCY_HALF_LIFE_DAYS = 14.0

# A purchase made under a pre-arranged Rule 10b5-1 plan still spends real
# money, so it is not discarded, but it was decided months ago.
SCHEDULED_WEIGHT = 0.25

# Relative informativeness of the buyer's seat.
ROLE_WEIGHTS = {
    "CFO": 1.30,
    "CEO": 1.25,
    "officer": 1.15,
    "director": 1.00,
    "insider": 1.00,
    "10% owner": 0.80,
}

EVIDENCE_TRADES_PER_TICKER = 6


@dataclass(frozen=True)
class InsiderSignal:
    """Every open-market insider trade in one ticker, aggregated."""

    ticker: str
    company: str
    buy_dollars: float
    sell_dollars: float
    conviction_dollars: float
    """Purchase dollars after discounting pre-scheduled 10b5-1 buying."""
    buyers: tuple[str, ...]
    sellers: tuple[str, ...]
    roles: tuple[str, ...]
    cluster_size: int
    """Most distinct buyers seen inside any single two-week window."""
    scheduled_buys: int
    latest_trade_date: date
    latest_filing_date: date
    trades: tuple[InsiderTrade, ...]
    score: float

    @property
    def net_dollars(self) -> float:
        return self.buy_dollars - self.sell_dollars

    @property
    def is_cluster(self) -> bool:
        return self.cluster_size >= CLUSTER_THRESHOLD


@dataclass(frozen=True)
class InsiderScan:
    """Result of one pass over EDGAR."""

    as_of: date
    lookback_days: int
    source_note: str
    signals: tuple[InsiderSignal, ...]

    @property
    def buying(self) -> tuple[InsiderSignal, ...]:
        return tuple(s for s in self.signals if s.score > 0)

    @property
    def selling(self) -> tuple[InsiderSignal, ...]:
        return tuple(
            sorted(
                (s for s in self.signals if s.net_dollars < 0),
                key=lambda s: s.net_dollars,
            )
        )

    def shortlist(self, size: int = DEFAULT_SHORTLIST_SIZE) -> tuple[InsiderSignal, ...]:
        return self.buying[:size]


def _cluster_size(buys: list[InsiderTrade]) -> int:
    """Most distinct insiders buying inside any ``CLUSTER_WINDOW_DAYS`` window.

    Counting distinct buyers across the whole lookback would call two purchases
    six weeks apart a cluster, which is precisely the coincidence the pattern
    is supposed to rule out. Windows are anchored on each purchase date, so a
    genuine burst is found wherever it sits in the range.
    """
    if not buys:
        return 0
    best = 0
    for anchor in buys:
        window_end = anchor.trade_date + timedelta(days=CLUSTER_WINDOW_DAYS)
        names = {
            t.owner
            for t in buys
            if anchor.trade_date <= t.trade_date <= window_end
        }
        best = max(best, len(names))
    return best


def _score(
    *,
    conviction_dollars: float,
    cluster_size: int,
    buyer_count: int,
    days_since_trade: float,
    role_weight: float,
) -> float:
    """Conviction-weighted score for insider buying in one ticker.

    Zero unless there was open-market buying: this ranks growth candidates, and
    a name insiders are only selling is not one.
    """
    if conviction_dollars <= 0 or buyer_count == 0:
        return 0.0

    size = max(0.0, math.log10(conviction_dollars / SIZE_REFERENCE_DOLLARS) + 1)
    breadth = 1 + CLUSTER_WEIGHT * (cluster_size - 1)
    if cluster_size >= CLUSTER_THRESHOLD:
        breadth += CLUSTER_BONUS
    recency = 0.5 ** (max(days_since_trade, 0) / RECENCY_HALF_LIFE_DAYS)
    return size * breadth * recency * role_weight


def aggregate(
    trades: list[InsiderTrade], *, as_of: date | None = None
) -> list[InsiderSignal]:
    """Group trades by ticker and score each, strongest signal first."""
    as_of = as_of or date.today()
    by_ticker: dict[str, list[InsiderTrade]] = {}
    for trade in trades:
        by_ticker.setdefault(trade.ticker, []).append(trade)

    signals = []
    for ticker, group in by_ticker.items():
        buys = [t for t in group if t.action == "buy"]
        sells = [t for t in group if t.action == "sell"]

        conviction = sum(
            t.value * (SCHEDULED_WEIGHT if t.scheduled_10b5_1 else 1.0) for t in buys
        )
        buyers = tuple(sorted({t.owner for t in buys}))
        roles = tuple(sorted({t.role for t in buys}))
        cluster = _cluster_size(buys)
        latest_buy = max((t.trade_date for t in buys), default=None)

        company = Counter(t.issuer for t in group if t.issuer).most_common(1)

        signals.append(
            InsiderSignal(
                ticker=ticker,
                company=company[0][0] if company else ticker,
                buy_dollars=sum(t.value for t in buys),
                sell_dollars=sum(t.value for t in sells),
                conviction_dollars=conviction,
                buyers=buyers,
                sellers=tuple(sorted({t.owner for t in sells})),
                roles=roles,
                cluster_size=cluster,
                scheduled_buys=sum(1 for t in buys if t.scheduled_10b5_1),
                latest_trade_date=max(t.trade_date for t in group),
                latest_filing_date=max(t.filing_date for t in group),
                trades=tuple(sorted(group, key=lambda t: t.trade_date, reverse=True)),
                score=_score(
                    conviction_dollars=conviction,
                    cluster_size=cluster,
                    buyer_count=len(buyers),
                    days_since_trade=(as_of - latest_buy).days if latest_buy else 0.0,
                    role_weight=max(
                        (ROLE_WEIGHTS.get(r, 1.0) for r in roles), default=1.0
                    ),
                ),
            )
        )

    signals.sort(key=lambda s: (s.score, s.conviction_dollars), reverse=True)
    return signals


def scan(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS, *, progress=None
) -> InsiderScan:
    """Fetch and score every open-market insider trade in the lookback window."""
    trades, note = fetch_insider_trades(lookback_days, progress=progress)
    as_of = date.today()
    return InsiderScan(
        as_of=as_of,
        lookback_days=lookback_days,
        source_note=note,
        signals=tuple(aggregate(trades, as_of=as_of)),
    )


def confirm(
    tickers: list[str],
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    *,
    progress=None,
) -> InsiderScan:
    """Score insider activity for named tickers only.

    The complement to ``scan``: instead of searching the market for insider
    buying, this asks whether the people running a handful of already-chosen
    companies have been buying too. Two insider groups arriving at the same
    name independently is the most interesting thing either dataset offers,
    and it costs a couple of requests per symbol rather than a full crawl.
    """
    trades, note = fetch_company_trades(tickers, lookback_days, progress=progress)
    as_of = date.today()
    return InsiderScan(
        as_of=as_of,
        lookback_days=lookback_days,
        source_note=note,
        signals=tuple(aggregate(trades, as_of=as_of)),
    )


def render_evidence(
    scan_result: InsiderScan, size: int = DEFAULT_SHORTLIST_SIZE
) -> str:
    """The shortlist as markdown, for an LLM prompt or for reading directly."""
    lines = [
        f"## Corporate insider buying (SEC Form 4), {scan_result.lookback_days} "
        f"days to {scan_result.as_of.isoformat()}",
        "",
        f"Source: {scan_result.source_note}",
        "",
    ]

    shortlist = scan_result.shortlist(size)
    if not shortlist:
        lines.append("No open-market insider purchases were filed in this window.")
        return "\n".join(lines)

    lines += ["### Names bought on the open market, strongest signal first", ""]
    for rank, signal in enumerate(shortlist, start=1):
        tags = []
        if signal.is_cluster:
            tags.append(f"CLUSTER of {signal.cluster_size} insiders")
        if signal.scheduled_buys:
            tags.append(f"{signal.scheduled_buys} pre-scheduled (10b5-1)")
        suffix = f" — {'; '.join(tags)}" if tags else ""

        lines.append(
            f"**{rank}. {signal.ticker} — {signal.company}** "
            f"(score {signal.score:.2f}) — ${signal.buy_dollars:,.0f} bought by "
            f"{len(signal.buyers)} insider(s) [{', '.join(signal.roles)}]; most "
            f"recent {signal.latest_trade_date.isoformat()}{suffix}."
        )
        for trade in signal.trades[:EVIDENCE_TRADES_PER_TICKER]:
            plan = " [10b5-1]" if trade.scheduled_10b5_1 else ""
            change = (
                f", {trade.position_change_pct:+.0%} position"
                if trade.position_change_pct is not None
                else ""
            )
            lines.append(
                f"    - {trade.owner} ({trade.role}"
                f"{': ' + trade.officer_title if trade.officer_title else ''}) "
                f"{trade.action} ${trade.value:,.0f} on "
                f"{trade.trade_date.isoformat()}{change}{plan}"
            )
        lines.append("")

    selling = scan_result.selling[:5]
    if selling:
        lines += ["### Heaviest insider selling, for context", ""]
        lines += [
            f"- **{s.ticker} — {s.company}**: ${abs(s.net_dollars):,.0f} net sold "
            f"by {len(s.sellers)} insider(s)"
            for s in selling
        ]
        lines.append("")

    return "\n".join(lines)
