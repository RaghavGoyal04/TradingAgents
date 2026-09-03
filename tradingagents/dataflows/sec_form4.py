"""SEC Form 4 insider-transaction vendor (Section 16 beneficial-ownership filings).

Officers, directors and 10%+ owners must file a Form 4 within **two business
days** of trading their own company's stock. That makes it the freshest
insider-flow dataset that exists in public -- three weeks fresher than a
congressional Periodic Transaction Report -- and it arrives as structured XML
rather than a scanned PDF.

Two ways in, because the cheap one covers most of what gets asked:

``fetch_company_trades`` takes tickers and reads only those issuers, via
``data.sec.gov/submissions/CIK*.json``. A couple of requests per symbol, fast
enough to run interactively. Use it whenever the names are already known --
checking whether company insiders bought the same stocks Congress did, or the
ones already held.

``fetch_insider_trades`` crawls the whole market from
``edgar/daily-index/{year}/QTR{q}/form.{yyyymmdd}.idx``, which is the only way
to *discover* a name nobody flagged. About 1,700 Form 4s are filed per day and
SEC's fair-access policy caps clients at 10 requests/second, so a day costs
several minutes and sustained crawling earns a temporary IP block -- both
verified the hard way. Each day is therefore parsed once and cached, and the
crawl is meant to be run from the command line rather than behind a button.

Note that the ``xslF345X06/`` path segment EDGAR links carry is SEC's
server-side XSLT rendering to HTML; dropping it yields the raw XML.

Only open-market purchases (transaction code ``P``) and sales (``S``) are kept
in the cache. Grants, option exercises, tax withholding and gifts are the bulk
of the volume and carry no directional information, so storing them would
inflate the cache for nothing. Changing that set means clearing the cache.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

from .config import get_config

logger = logging.getLogger(__name__)

ARCHIVES = "https://www.sec.gov/Archives"
DAILY_INDEX = ARCHIVES + "/edgar/daily-index/{year}/QTR{quarter}/form.{stamp}.idx"

REQUEST_TIMEOUT = 30

# SEC wants "Company Name contact@domain", and enforces it: a User-Agent
# carrying a URL, or one with no contact address at all, is refused with 403
# and their "undeclared automated tool" page. Both were verified against
# ``company_tickers.json``, and the URL form -- which most library defaults use
# -- fails every time. Set SEC_CONTACT to a real address you monitor.
# https://www.sec.gov/os/webmaster-faq#developers
SEC_CONTACT_DEFAULT = "contact@tradingagents.example.com"
USER_AGENT_NAME = "TradingAgents/0.3.1"

# Documented ceiling is 10 requests/second, but sustained bulk reading trips a
# temporary IP block well under that, so this stays at half the limit. The
# saving from going faster is minutes once, since every day is cached; the cost
# of overshooting is an hour of no access at all.
MAX_REQUESTS_PER_SECOND = 5.0
FETCH_WORKERS = 5

# SEC answers a burst with 429 rather than closing the connection, so a pause
# usually clears it. A block that survives all of these is an IP-level one that
# lasts minutes, and no amount of further retrying will help.
MAX_RETRIES = 4
RETRY_BASE_DELAY = 2.0

# EDGAR serves 403 for three unrelated situations, and only the body tells them
# apart: a path that does not exist (today's index before it is published), a
# User-Agent it refuses, and a client it is throttling. Rate limiting is worth
# waiting out; a rejected User-Agent is a configuration error that will fail
# identically forever, so it must not be retried or reported as congestion.
_RATE_LIMITED_BODY = re.compile(r"request rate|threshold exceeded", re.IGNORECASE)
_UNDECLARED_BODY = re.compile(r"undeclared automated tool", re.IGNORECASE)

# Transaction codes. Only "P" is a voluntary open-market purchase made with the
# insider's own cash, and it is the one code every practitioner guide agrees
# carries signal. "S" is an open-market sale: kept, but it is a far noisier
# indicator because insiders sell to diversify, to pay tax, and on schedule.
# Deliberately excluded: A (grant/award), M (option exercise), F (shares
# withheld for tax), G (gift), D (disposition to the issuer), J (other). None
# involve a discretionary decision to put money at risk.
BUY_CODE = "P"
SELL_CODE = "S"
SIGNAL_CODES = frozenset({BUY_CODE, SELL_CODE})

# Text that marks a trade as pre-scheduled under a Rule 10b5-1 plan. Such a
# trade was arranged months earlier and says nothing about today's view, so the
# scoring layer discounts it heavily. The document-level ``aff10b5One`` flag is
# authoritative when present; older filings only say so in a footnote.
_10B5_1_RE = re.compile(r"10b5[\s\-]?1", re.IGNORECASE)


class SourceUnavailable(RuntimeError):
    """EDGAR could not be reached or read this run."""


class UserAgentRejected(SourceUnavailable):
    """EDGAR refused the User-Agent.

    A configuration problem rather than a transient one: SEC requires a contact
    address and rejects any User-Agent carrying a URL, so retrying changes
    nothing until the header is fixed.
    """


class Throttled(SourceUnavailable):
    """SEC is rate-limiting this client.

    Separate from a plain failure because it applies to the whole host: once
    EDGAR starts refusing, continuing to the next day only deepens the block,
    so the fetch stops and keeps whatever it had already cached.
    """


@dataclass(frozen=True)
class InsiderTrade:
    """One open-market transaction by a corporate insider."""

    ticker: str
    issuer: str
    issuer_cik: str
    owner: str
    role: str
    officer_title: str
    action: str
    trade_date: date
    filing_date: date
    shares: float
    price: float
    shares_after: float
    scheduled_10b5_1: bool
    accession: str

    @property
    def value(self) -> float:
        """Dollar size of the trade. Exact, unlike a congressional band."""
        return self.shares * self.price

    @property
    def position_change_pct(self) -> float | None:
        """Signed change in the insider's holding, against what they held before.

        A director buying their first $50k is a different statement from one
        topping up a $40m holding, and for sales the fraction is what separates
        routine diversification from an exit. Negative for a sale, so the sign
        alone reads correctly. ``None`` when the filing does not report a
        post-transaction balance.
        """
        if not self.shares_after or self.shares <= 0:
            return None
        if self.action == "buy":
            before = self.shares_after - self.shares
            return self.shares / before if before > 0 else None
        before = self.shares_after + self.shares
        return -self.shares / before if before > 0 else None


# Once EDGAR blocks a client, further requests only extend the block, so the
# module stops sending them for a cooling-off period rather than retrying into
# it. Shared across threads and callers within the process.
BLOCK_COOLDOWN_SECONDS = 600.0
_blocked_until = 0.0
_block_lock = threading.Lock()


def _blocked_for() -> float:
    """Seconds left on the self-imposed cooling-off period, zero if clear."""
    with _block_lock:
        return max(0.0, _blocked_until - time.monotonic())


def _start_cooldown() -> None:
    global _blocked_until
    with _block_lock:
        _blocked_until = time.monotonic() + BLOCK_COOLDOWN_SECONDS


class _RateLimiter:
    """Paces requests across threads to stay under SEC's published ceiling."""

    def __init__(self, per_second: float):
        self._interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self._interval
        if wait > 0:
            time.sleep(wait)


def _session() -> requests.Session:
    import os

    contact = os.getenv("SEC_CONTACT", "").strip() or SEC_CONTACT_DEFAULT
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": f"{USER_AGENT_NAME} {contact}",
            "Accept-Encoding": "gzip, deflate",
        }
    )
    return session


def _get(session: requests.Session, url: str, limiter: _RateLimiter) -> bytes:
    """Fetch one EDGAR URL, backing off when throttled."""
    remaining = _blocked_for()
    if remaining:
        raise Throttled(
            f"SEC blocked this address; waiting {remaining / 60:.0f} more "
            f"minute(s) before trying again. Anything already cached still works."
        )

    for attempt in range(MAX_RETRIES):
        limiter.acquire()
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        body = response.text[:2000] if response.status_code == 403 else ""
        if _UNDECLARED_BODY.search(body):
            raise UserAgentRejected(
                f"EDGAR refused the User-Agent {session.headers['User-Agent']!r}. "
                f"It requires a contact address and rejects any User-Agent "
                f"containing a URL. Set SEC_CONTACT to an address you monitor."
            )
        if response.status_code == 429 or _RATE_LIMITED_BODY.search(body):
            time.sleep(RETRY_BASE_DELAY * (2**attempt))
            continue
        response.raise_for_status()
        return response.content

    _start_cooldown()
    raise Throttled(
        "SEC is rate-limiting this address, so further requests are paused for "
        f"{BLOCK_COOLDOWN_SECONDS / 60:.0f} minutes -- retrying into a block "
        "only extends it. Cached data is unaffected."
    )


def _cache_dir() -> Path:
    path = Path(get_config()["data_cache_dir"]) / "sec_form4"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _text(node: ET.Element | None, path: str) -> str:
    """Read a Form 4 field, which nests its content in a ``<value>`` child."""
    if node is None:
        return ""
    found = node.find(path)
    if found is None:
        return ""
    value = found.find("value")
    target = found if value is None else value
    return (target.text or "").strip()


def _float(raw: str) -> float:
    try:
        return float(raw.replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0


def _flag(raw: str) -> bool:
    """Read a Form 4 boolean.

    Filing agents emit these as either ``1``/``0`` or ``true``/``false`` and
    both appear in current filings, so accepting only one spelling silently
    misreads whole populations of filers -- including missing the 10b5-1 flag
    that marks a trade as pre-scheduled.
    """
    return raw.strip().lower() in {"1", "true"}


def _role(relationship: ET.Element | None, title: str) -> str:
    """Classify the filer, most informative role first.

    The ordering reflects a consistent finding in the practitioner literature:
    a CFO sees forward numbers earliest, operating officers next, while
    directors often buy for optics and a 10% holder topping up its own block is
    a different thing entirely from an insider's judgement call.
    """
    if relationship is None:
        return "insider"
    upper = title.upper()
    if _flag(_text(relationship, "isOfficer")):
        if "CFO" in upper or "CHIEF FINANCIAL" in upper or "FINANCE" in upper:
            return "CFO"
        if "CEO" in upper or "CHIEF EXECUTIVE" in upper or "PRESIDENT" in upper:
            return "CEO"
        return "officer"
    if _flag(_text(relationship, "isDirector")):
        return "director"
    if _flag(_text(relationship, "isTenPercentOwner")):
        return "10% owner"
    return "insider"


def parse_form4(
    xml_text: str, accession: str, filing_date: date
) -> list[InsiderTrade]:
    """Extract open-market purchases and sales from one Form 4 document.

    ``filing_date`` comes from the daily index the filing was listed in. The
    document's own ``periodOfReport`` is the earliest transaction date, not the
    date it was filed, so it cannot stand in for the disclosure lag.

    Only the non-derivative table is read. A derivative-table ``P`` is usually
    an option or warrant transaction whose economics depend on strike and
    expiry, so folding it in beside a share purchase at market would overstate
    the cash actually committed.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.debug("Form 4 %s is not parseable XML: %s", accession, exc)
        return []

    issuer = root.find("issuer")
    ticker = _text(issuer, "issuerTradingSymbol").upper()
    if not ticker or ticker in {"NONE", "N/A"}:
        return []

    owner = root.find("reportingOwner")
    relationship = None if owner is None else owner.find("reportingOwnerRelationship")
    officer_title = _text(relationship, "officerTitle")

    # Document-level flag first; fall back to scanning footnote prose, which is
    # how pre-2023 filings disclose a plan.
    flagged = _flag(root.findtext("aff10b5One", ""))
    footnotes = " ".join(node.text or "" for node in root.iter("footnote"))
    scheduled = flagged or bool(_10B5_1_RE.search(footnotes))

    trades = []
    for txn in root.iter("nonDerivativeTransaction"):
        coding = txn.find("transactionCoding")
        code = _text(coding, "transactionCode") or (
            coding.findtext("transactionCode", "").strip() if coding is not None else ""
        )
        if code not in SIGNAL_CODES:
            continue

        amounts = txn.find("transactionAmounts")
        shares = _float(_text(amounts, "transactionShares"))
        price = _float(_text(amounts, "transactionPricePerShare"))
        if shares <= 0 or price <= 0:
            # A zero price is a grant or an exercise mislabelled as P/S; without
            # a price there is no dollar conviction to measure.
            continue

        trade_date = _text(txn, "transactionDate")
        if not trade_date:
            continue

        trades.append(
            InsiderTrade(
                ticker=ticker,
                issuer=_text(issuer, "issuerName"),
                issuer_cik=_text(issuer, "issuerCik").lstrip("0"),
                owner=_text(
                    owner.find("reportingOwnerId") if owner is not None else None,
                    "rptOwnerName",
                )
                or "unknown",
                role=_role(relationship, officer_title),
                officer_title=officer_title,
                action="buy" if code == BUY_CODE else "sell",
                trade_date=date.fromisoformat(trade_date),
                filing_date=filing_date,
                shares=shares,
                price=price,
                shares_after=_float(
                    _text(
                        txn.find("postTransactionAmounts"),
                        "sharesOwnedFollowingTransaction",
                    )
                ),
                scheduled_10b5_1=scheduled,
                accession=accession,
            )
        )
    return trades


def _extract_xml(submission: str) -> str | None:
    """Pull the ``<ownershipDocument>`` out of a full submission text file."""
    start = submission.find("<ownershipDocument")
    if start < 0:
        return None
    end = submission.find("</ownershipDocument>", start)
    return None if end < 0 else submission[start : end + len("</ownershipDocument>")]


def _index_paths(
    day: date, session: requests.Session, limiter: _RateLimiter
) -> list[str] | None:
    """One submission path per Form 4 filed on ``day``.

    The index lists a filing once for every party to it -- the issuer and each
    reporting owner -- and each row carries that party's own CIK in the path,
    so a single filing routinely appears under eight different URLs that all
    serve the same document. Deduplicating on the path therefore does nothing;
    it has to be the accession number. Getting this wrong counts every trade
    once per party, inflating dollar totals severalfold, and doubles the
    fetching: a recent day held 1,686 rows but only 820 distinct filings.

    Returns ``None`` when no index exists for the day, which is different from
    an index listing no Form 4s: today's index is absent until EDGAR publishes
    it, and caching that as a genuinely empty day would freeze the gap in.
    """
    url = DAILY_INDEX.format(
        year=day.year, quarter=(day.month - 1) // 3 + 1, stamp=day.strftime("%Y%m%d")
    )
    try:
        raw = _get(session, url, limiter)
    except requests.HTTPError as exc:
        # Weekend, market holiday, or today before EDGAR publishes: absent
        # indexes come back as 403 as often as 404.
        if exc.response is not None and exc.response.status_code in (403, 404):
            return None
        raise

    by_accession: dict[str, str] = {}
    for line in raw.decode("latin-1").splitlines():
        if not line.startswith("4 "):
            continue
        # Company names contain spaces; the trailing three fields never do.
        fields = line.split()
        if len(fields) >= 3 and fields[-1].endswith(".txt"):
            path = fields[-1]
            by_accession.setdefault(path.rsplit("/", 1)[-1][: -len(".txt")], path)
    return sorted(by_accession.values())


def _fetch_day(
    day: date,
    session: requests.Session,
    limiter: _RateLimiter,
    progress: Callable[[str], None] | None,
) -> list[InsiderTrade]:
    """Every open-market insider trade filed on ``day``, cached on disk."""
    from concurrent.futures import ThreadPoolExecutor

    cached = _cache_dir() / f"{day.strftime('%Y%m%d')}.json"
    if cached.exists():
        rows = json.loads(cached.read_text(encoding="utf-8"))
        return [
            InsiderTrade(
                **{
                    **row,
                    "trade_date": date.fromisoformat(row["trade_date"]),
                    "filing_date": date.fromisoformat(row["filing_date"]),
                }
            )
            for row in rows
        ]

    paths = _index_paths(day, session, limiter)
    if paths is None:
        return []  # nothing published for this day yet, so nothing to cache
    if not paths:
        cached.write_text("[]", encoding="utf-8")
        return []

    if progress:
        progress(f"{day.isoformat()}: reading {len(paths)} Form 4 filings…")

    def read(path: str) -> list[InsiderTrade]:
        accession = Path(path).stem
        try:
            body = _get(session, f"{ARCHIVES}/{path}", limiter).decode(
                "utf-8", errors="replace"
            )
        except Throttled:
            raise  # host-wide, so abandon the day rather than logging 1,700 times
        except Exception as exc:
            # One withdrawn or malformed filing must not cost the whole day.
            logger.warning("Form 4 %s unreadable: %s", accession, exc)
            return []
        xml_text = _extract_xml(body)
        return parse_form4(xml_text, accession, day) if xml_text else []

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        trades = [t for batch in pool.map(read, paths) for t in batch]

    payload = []
    for trade in trades:
        row = asdict(trade)
        row["trade_date"] = trade.trade_date.isoformat()
        row["filing_date"] = trade.filing_date.isoformat()
        payload.append(row)
    cached.write_text(json.dumps(payload), encoding="utf-8")

    if progress:
        progress(
            f"{day.isoformat()}: {len(trades)} open-market trades from "
            f"{len(paths)} filings."
        )
    return trades


SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

# EDGAR's own links point at ``xslF345X06/form4.xml`` and similar. That path
# segment is a server-side XSLT rendering of the filing into HTML; removing it
# returns the raw XML the renderer was built from.
_XSL_DIR_RE = re.compile(r"^xsl[^/]*/")


def _ticker_map(session: requests.Session, limiter: _RateLimiter) -> dict[str, int]:
    """Ticker to CIK for every listed filer, refreshed daily.

    One 800 KB file covers the whole market, so there is no per-symbol lookup
    and no reason to hold a stale copy for more than a day.
    """
    cached = _cache_dir() / f"tickers_{date.today():%Y%m%d}.json"
    if cached.exists():
        return {k: int(v) for k, v in json.loads(cached.read_text()).items()}

    raw = json.loads(_get(session, TICKER_MAP_URL, limiter))
    mapping = {
        row["ticker"].upper(): int(row["cik_str"])
        for row in raw.values()
        if row.get("ticker")
    }
    cached.write_text(json.dumps(mapping), encoding="utf-8")
    return mapping


def _company_form4s(
    cik: int, cutoff: date, session: requests.Session, limiter: _RateLimiter
) -> list[tuple[str, str, date]]:
    """Accession, document path and filing date for one issuer's recent Form 4s."""
    payload = json.loads(_get(session, SUBMISSIONS.format(cik=cik), limiter))
    recent = payload.get("filings", {}).get("recent", {})
    out = []
    for form, filed, accession, document in zip(
        recent.get("form", []),
        recent.get("filingDate", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
        strict=False,
    ):
        if form != "4":
            continue
        filed_on = date.fromisoformat(filed)
        if filed_on < cutoff:
            # The list is newest-first, so the first old filing ends the window.
            break
        out.append((accession, _XSL_DIR_RE.sub("", document or ""), filed_on))
    return out


def fetch_company_trades(
    tickers: list[str],
    lookback_days: int = 14,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[InsiderTrade], str]:
    """Insider trades for named tickers only.

    Costs a couple of requests per symbol instead of the ~1,700 a day of
    whole-market crawling takes, so this is the path to use whenever the names
    of interest are already known -- checking whether company insiders have
    been buying the same stocks as Congress, or the ones already held.

    Unknown symbols are reported in the note rather than raising: a watchlist
    containing one delisted ticker should still return the rest.
    """
    session = _session()
    limiter = _RateLimiter(MAX_REQUESTS_PER_SECOND)
    cutoff = date.today() - timedelta(days=lookback_days)

    mapping = _ticker_map(session, limiter)
    wanted = [t.strip().upper() for t in tickers if t.strip()]
    unknown = [t for t in wanted if t not in mapping]

    trades: list[InsiderTrade] = []
    for ticker in wanted:
        cik = mapping.get(ticker)
        if cik is None:
            continue
        try:
            filings = _company_form4s(cik, cutoff, session, limiter)
        except (requests.RequestException, ValueError, KeyError) as exc:
            logger.warning("Form 4 index for %s unreadable: %s", ticker, exc)
            continue

        if progress and filings:
            progress(f"{ticker}: reading {len(filings)} Form 4 filings…")

        for accession, document, filed_on in filings:
            body = _cached_filing(cik, accession, document, session, limiter)
            if body is None:
                continue
            xml_text = body if body.lstrip().startswith("<") else _extract_xml(body)
            if not xml_text:
                continue
            # A company's own submission feed also carries the Form 4s it filed
            # as an insider of somebody else -- a bank that is a 10% holder of
            # dozens of funds returns those too. Only filings where this
            # company is the issuer describe trades in its own stock.
            trades.extend(
                t
                for t in parse_form4(xml_text, accession, filed_on)
                if t.issuer_cik == str(cik)
            )

    recent = [t for t in trades if t.trade_date >= cutoff]
    buys = sum(1 for t in recent if t.action == "buy")
    note = (
        f"SEC Form 4: {len(wanted) - len(unknown)} ticker(s) checked, "
        f"{len(recent)} open-market insider transactions ({buys} purchases) in "
        f"the last {lookback_days} days."
    )
    if unknown:
        note += f" Not found on EDGAR: {', '.join(unknown)}."
    return recent, note


def _cached_filing(
    cik: int,
    accession: str,
    document: str,
    session: requests.Session,
    limiter: _RateLimiter,
) -> str | None:
    """One filing's XML, cached permanently -- a filed Form 4 never changes."""
    cached = _cache_dir() / f"filing_{accession}.xml"
    if cached.exists():
        return cached.read_text(encoding="utf-8")

    stripped = accession.replace("-", "")
    urls = [f"{ARCHIVES}/edgar/data/{cik}/{stripped}/{document}"] if document else []
    # Older filings have no primary document recorded; the full submission
    # wrapper is always present and embeds the same XML.
    urls.append(f"{ARCHIVES}/edgar/data/{cik}/{accession}.txt")

    for url in urls:
        try:
            body = _get(session, url, limiter).decode("utf-8", errors="replace")
        except Throttled:
            raise
        except Exception:
            continue
        xml_text = body if body.lstrip().startswith("<") else _extract_xml(body)
        if xml_text:
            cached.write_text(xml_text, encoding="utf-8")
            return xml_text

    logger.warning("Form 4 %s could not be read from any known path", accession)
    return None


def cached_days() -> set[date]:
    """Business days already parsed, so a caller can size the work remaining."""
    days = set()
    for path in _cache_dir().glob("*.json"):
        try:
            days.add(datetime.strptime(path.stem, "%Y%m%d").date())
        except ValueError:
            continue
    return days


def fetch_insider_trades(
    lookback_days: int = 14,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[InsiderTrade], str]:
    """Open-market insider trades filed in the last ``lookback_days``.

    Returns the trades and a plain-language note on coverage. Days are fetched
    newest-first and cached individually, so an interrupted run keeps whatever
    it finished and a repeat run only pays for days it has not seen.
    """
    today = date.today()
    session = _session()
    limiter = _RateLimiter(MAX_REQUESTS_PER_SECOND)

    days = [today - timedelta(days=offset) for offset in range(lookback_days + 1)]
    # Weekends have no index; skipping them saves two wasted requests a week.
    days = [day for day in days if day.weekday() < 5]

    trades: list[InsiderTrade] = []
    read_days = 0
    failures = 0
    throttled = ""
    for day in days:
        try:
            trades.extend(_fetch_day(day, session, limiter, progress))
            read_days += 1
        except Throttled as exc:
            throttled = str(exc)
            logger.warning("EDGAR throttled at %s; stopping early", day)
            break
        except (requests.RequestException, SourceUnavailable) as exc:
            failures += 1
            logger.warning("EDGAR day %s unavailable: %s", day, exc)

    if not read_days:
        raise Throttled(throttled) if throttled else SourceUnavailable(
            f"EDGAR returned nothing for any of the last {lookback_days} days"
        )

    cutoff = today - timedelta(days=lookback_days)
    recent = [t for t in trades if t.trade_date >= cutoff]
    buys = sum(1 for t in recent if t.action == "buy")
    note = (
        f"SEC Form 4: {read_days} of {len(days)} trading days read, {len(recent)} "
        f"open-market insider transactions ({buys} purchases) in the last "
        f"{lookback_days} days."
    )
    if throttled:
        note += f" Stopped early -- {throttled}"
    elif failures:
        note += f" {failures} day(s) could not be read."
    return recent, note


def _main() -> int:
    """Warm the day cache from the command line.

    Reading a fortnight of filings costs about half an hour the first time, so
    it is worth doing outside a UI: ``python -m tradingagents.dataflows.sec_form4``
    """
    import argparse

    parser = argparse.ArgumentParser(description=_main.__doc__)
    parser.add_argument("--days", type=int, default=14)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    trades, note = fetch_insider_trades(args.days, progress=lambda m: print(m, flush=True))
    print(note)
    print(f"cached days: {len(cached_days())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
