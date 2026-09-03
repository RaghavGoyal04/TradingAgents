"""US Congress stock-disclosure vendor (STOCK Act periodic transaction reports).

Members of Congress must file a Periodic Transaction Report (PTR) within 45
days of a covered trade. Those filings are the only public record of what
elected officials actually bought and sold, and a cluster of buying in one
name is the signal this module surfaces.

Sources:

  House  -- the Clerk's official yearly archive
            (``financial-pdfs/{year}FD.ZIP``) indexes every filing; each PTR's
            transactions live in a generated, text-based PDF that ``pypdf``
            reads directly. Keyless and complete.
  Senate -- ``efdsearch.senate.gov``. Best-effort only: the site sits behind
            bot protection that rejects non-browser TLS clients (verified 403
            with full browser headers), so in practice this degrades to a note
            explaining the gap rather than returning data.

Two properties of the disclosure regime, not of this code, limit what the
output can mean: filings arrive up to 45 days after the trade, and sizes are
disclosed as bands ("$1,001 - $15,000") rather than exact amounts. Callers
must treat the result as a late, coarse signal.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

from .config import get_config

logger = logging.getLogger(__name__)

HOUSE_ARCHIVE = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.ZIP"
HOUSE_PTR_PDF = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"
SENATE_BASE = "https://efdsearch.senate.gov"

# Network timeout (seconds), consistent with the other vendors.
REQUEST_TIMEOUT = 30

USER_AGENT = "TradingAgents/0.3.1 (+https://github.com/TauricResearch/TradingAgents)"

# Max PTR PDFs fetched concurrently. Deliberately low: this is a .gov host
# serving every filing for free, and a run only needs a few dozen documents.
PDF_WORKERS = 6

# "P" in the index is a Periodic Transaction Report -- the only filing type
# that carries individual trades. Annual reports ("A"/"C"/"O") are holdings
# snapshots with no transaction dates.
PTR_FILING_TYPE = "P"

# House asset-type codes worth ranking. ST is common stock (including ADRs);
# OP is an option on one, the highest-conviction directional bet a member can
# disclose. Everything else (Treasuries, mutual funds, ETFs, corporate bonds)
# is either not a single name or not a growth bet.
# https://fd.house.gov/reference/asset-type-codes.aspx
TRADED_ASSET_CODES = frozenset({"ST", "OP"})

# "E" is an exchange: directionally ambiguous, so it is dropped rather than
# scored as either side.
_ACTIONS = {"P": "buy", "S": "sell", "S (partial)": "sell"}

# One transaction row, read off the whitespace-normalised page text. The ticker
# and its asset-type code always precede the action/dates/amount in pypdf's
# reading order, and the amount band may have wrapped a line in the PDF.
_RECORD_RE = re.compile(
    r"\(\s*(?P<ticker>[A-Z][A-Z0-9.\-]{0,5})\s*\)\s*"
    r"\[(?P<code>[A-Z0-9]{2})\]\s*"
    r"(?P<action>S \(partial\)|[PSE])\s+"
    r"(?P<traded>\d{2}/\d{2}/\d{4})\s+"
    r"(?P<filed>\d{2}/\d{2}/\d{4})\s+"
    r"(?:\$(?P<low>[\d,]+)\s*-\s*\$(?P<high>[\d,]+)|Over\s+\$(?P<over>[\d,]+))"
)

# Per-record trailing metadata ("Filing Status: New", "Sub-holding Owner: ...")
# and the page footer. Dropped before the page is flattened so an asset name
# can be read as the text between one record and the next. The label words
# themselves render as a single letter plus padding, hence the loose spacing.
_NOISE_RE = re.compile(r"^(?:F\s+S\s*:|S\s+O\s*:|Filing ID|\*)")

# Owner code sitting in front of an asset name: SP spouse, DC dependent child,
# JT joint.
_OWNER_PREFIX_RE = re.compile(r"^(?:SP|DC|JT)\s+")

# Everything a preceding record can leave in front of an asset name: a disclosed
# amount, an asset-type bracket, or a date. Splitting on all three and keeping
# the tail isolates the name even when the previous row was an untickered
# holding (a municipal bond, say) that no record pattern matched, or an option
# whose row carries a strike and an expiry ("6/17/27") in prose.
_ASSET_LEAD_RE = re.compile(r"\$[\d,]+|\]|\d{1,2}/\d{1,2}/\d{2,4}")

# The transaction table's header ends with this cell, and the asset-code
# reference line ends the table. Slicing between them drops the filer block and
# the certification text without needing to recognise either.
_TABLE_START = "$200?"
_TABLE_END = "* For the complete list"


class SourceUnavailable(RuntimeError):
    """A disclosure source could not be reached or read this run."""


@dataclass(frozen=True)
class CongressTrade:
    """One disclosed transaction by a member of Congress."""

    chamber: str
    member: str
    constituency: str
    ticker: str
    asset: str
    asset_code: str
    action: str
    trade_date: date
    filing_date: date
    amount_low: int
    amount_high: int
    doc_id: str

    @property
    def amount_mid(self) -> float:
        """Band midpoint. The only size estimate the disclosure supports."""
        return (self.amount_low + self.amount_high) / 2

    @property
    def disclosure_lag_days(self) -> int:
        return (self.filing_date - self.trade_date).days


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _cache_dir() -> Path:
    path = Path(get_config()["data_cache_dir"]) / "congress"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_amount(match: re.Match) -> tuple[int, int]:
    """Read a disclosed amount band into (low, high) dollars.

    The topmost band is open-ended ("Over $50,000,000"); it is pinned to its
    own floor so a single filing cannot dominate a ranking on an invented
    upper bound.
    """
    if match.group("over"):
        floor = int(match.group("over").replace(",", ""))
        return floor, floor
    low = int(match.group("low").replace(",", ""))
    high = int(match.group("high").replace(",", ""))
    return low, high


def _page_blob(page_text: str) -> str:
    """Flatten one PTR page to the transaction table's text, noise removed.

    The Clerk's PDFs draw the form's bold labels in a font whose padding
    glyphs extract as NUL bytes ("Filing Status" arrives as ``F\\x00\\x00 S...``),
    so those are turned back into spaces before anything tries to recognise a
    line.
    """
    body = (
        page_text.replace("\x00", " ")
        .split(_TABLE_START, 1)[-1]
        .split(_TABLE_END, 1)[0]
    )
    kept = [
        line for line in body.splitlines() if not _NOISE_RE.match(line.strip())
    ]
    return " ".join(" ".join(kept).split())


def _asset_name(window: str) -> str:
    """Read the asset description out of the text preceding a record.

    Punctuation is trimmed before the owner code is removed: an option row ends
    its prose with a full stop, and a leading ". " would otherwise stop the
    owner prefix from being recognised.
    """
    tail = _ASSET_LEAD_RE.split(window)[-1].strip(" -.,:%")
    return _OWNER_PREFIX_RE.sub("", tail).strip(" -.,:%")


def parse_ptr_text(
    pages: list[str],
    *,
    member: str,
    constituency: str,
    doc_id: str,
    chamber: str = "House",
) -> list[CongressTrade]:
    """Extract stock and option transactions from a PTR's page texts.

    Scanned paper filings extract as empty or unstructured text and simply
    yield nothing, which is the correct outcome: a filing we cannot read must
    not become a signal.
    """
    trades: list[CongressTrade] = []
    for page_text in pages:
        blob = _page_blob(page_text)
        cursor = 0
        for match in _RECORD_RE.finditer(blob):
            asset = _asset_name(blob[cursor : match.start()])
            cursor = match.end()

            code = match.group("code")
            action = _ACTIONS.get(match.group("action"))
            if code not in TRADED_ASSET_CODES or action is None:
                continue

            low, high = _parse_amount(match)
            trades.append(
                CongressTrade(
                    chamber=chamber,
                    member=member,
                    constituency=constituency,
                    ticker=match.group("ticker"),
                    asset=asset,
                    asset_code=code,
                    action=action,
                    trade_date=datetime.strptime(
                        match.group("traded"), "%m/%d/%Y"
                    ).date(),
                    filing_date=datetime.strptime(
                        match.group("filed"), "%m/%d/%Y"
                    ).date(),
                    amount_low=low,
                    amount_high=high,
                    doc_id=doc_id,
                )
            )
    return trades


def _house_index(year: int, session: requests.Session) -> list[dict]:
    """Every PTR filed in ``year``, newest first, from the Clerk's archive.

    Re-downloaded each run: the archive is rebuilt as filings arrive, so a
    cached copy would silently hide the newest trades.
    """
    response = session.get(HOUSE_ARCHIVE.format(year=year), timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".txt"))
        text = archive.read(name).decode("utf-8-sig")

    rows = [
        row
        for row in csv.DictReader(io.StringIO(text), delimiter="\t")
        if row.get("FilingType") == PTR_FILING_TYPE
    ]
    for row in rows:
        row["_filed"] = datetime.strptime(row["FilingDate"], "%m/%d/%Y").date()
    rows.sort(key=lambda r: r["_filed"], reverse=True)
    return rows


def _ptr_pages(year: int, doc_id: str, session: requests.Session) -> list[str]:
    """Page texts for one House PTR, cached on disk.

    A filed PTR never changes, so the extracted text is cached permanently and
    a re-run only pays for filings it has not seen.
    """
    if not doc_id.isdigit():
        raise SourceUnavailable(f"unexpected House document id {doc_id!r}")

    cached = _cache_dir() / f"house_{year}_{doc_id}.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8").split("\f")

    import pypdf

    response = session.get(
        HOUSE_PTR_PDF.format(year=year, doc_id=doc_id), timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()
    reader = pypdf.PdfReader(io.BytesIO(response.content))
    pages = [page.extract_text() or "" for page in reader.pages]
    cached.write_text("\f".join(pages), encoding="utf-8")
    return pages


def fetch_house_trades(lookback_days: int) -> tuple[list[CongressTrade], int]:
    """House stock and option transactions filed within ``lookback_days``.

    Returns the trades plus the number of PTRs scanned, so callers can report
    coverage (a filing that yields no rows is usually a scanned paper form).
    """
    today = date.today()
    session = _session()
    years = {today.year, (today - timedelta(days=lookback_days)).year}

    filings: list[dict] = []
    for year in sorted(years, reverse=True):
        try:
            filings.extend(
                row
                for row in _house_index(year, session)
                if (today - row["_filed"]).days <= lookback_days
            )
        except (requests.RequestException, zipfile.BadZipFile, StopIteration) as exc:
            raise SourceUnavailable(
                f"House Clerk archive for {year} could not be read ({exc})"
            ) from exc

    def read(row: dict) -> list[CongressTrade]:
        try:
            pages = _ptr_pages(int(row["Year"]), row["DocID"].strip(), session)
        except Exception as exc:
            # One unreadable filing (withdrawn document, malformed scan) must
            # not cost the run every other filing's signal.
            logger.warning("PTR %s unreadable: %s", row.get("DocID"), exc)
            return []
        return parse_ptr_text(
            pages,
            member=" ".join(
                part for part in (row.get("First"), row.get("Last")) if part
            ).strip(),
            constituency=(row.get("StateDst") or "").strip(),
            doc_id=row["DocID"].strip(),
        )

    with ThreadPoolExecutor(max_workers=PDF_WORKERS) as pool:
        trades = [trade for batch in pool.map(read, filings) for trade in batch]

    return trades, len(filings)


def fetch_senate_trades(lookback_days: int) -> tuple[list[CongressTrade], int]:
    """Senate transactions, best-effort.

    Implements the Electronic Financial Disclosure search flow (accept the
    search agreement, then query the report index). The host currently answers
    non-browser clients with HTTP 403 regardless of headers, so this raises
    ``SourceUnavailable`` in practice; it is kept so Senate coverage appears
    the moment the block lifts or the run happens from an allowed network.
    """
    session = _session()
    try:
        home = session.get(f"{SENATE_BASE}/search/home/", timeout=REQUEST_TIMEOUT)
        home.raise_for_status()
        token = _csrf_token(home.text)
        if not token:
            raise SourceUnavailable("no CSRF token in the Senate search page")

        session.post(
            f"{SENATE_BASE}/search/home/",
            data={"prohibition_agreement": "1", "csrfmiddlewaretoken": token},
            headers={"Referer": f"{SENATE_BASE}/search/home/"},
            timeout=REQUEST_TIMEOUT,
        ).raise_for_status()

        found = session.post(
            f"{SENATE_BASE}/search/report/data/",
            data={
                "start": "0",
                "length": "100",
                "report_types": "[11]",
                "filer_types": "[]",
                "submitted_start_date": (
                    date.today() - timedelta(days=lookback_days)
                ).strftime("%m/%d/%Y %H:%M:%S"),
                "submitted_end_date": "",
                "candidate_state": "",
                "senator_state": "",
                "office_id": "",
                "first_name": "",
                "last_name": "",
                "csrfmiddlewaretoken": token,
            },
            headers={"Referer": f"{SENATE_BASE}/search/home/"},
            timeout=REQUEST_TIMEOUT,
        )
        found.raise_for_status()
    except requests.RequestException as exc:
        raise SourceUnavailable(
            f"efdsearch.senate.gov rejected the request ({exc}); the site blocks "
            f"non-browser clients, so Senate filings are not covered this run"
        ) from exc

    # Each hit links an HTML periodic-transaction report; the per-report parse
    # is not implemented because the index above has never been reachable to
    # confirm its current shape against.
    hits = found.json().get("data", [])
    raise SourceUnavailable(
        f"Senate index returned {len(hits)} filings but per-report parsing is "
        f"not implemented (the search endpoint has never been reachable to "
        f"verify the report layout against)"
    )


def _csrf_token(html: str) -> str | None:
    match = re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"', html)
    return match.group(1) if match else None


def fetch_congress_trades(
    lookback_days: int = 60,
) -> tuple[list[CongressTrade], list[str]]:
    """Congressional trades *executed* in the last ``lookback_days``.

    Filings are gathered by filing date, because that is the only thing the
    indexes can be searched on, but the returned trades are filtered on the
    date the trade actually happened. Without that second filter a late or
    amended filing drags year-old trades into a window meant to describe
    current positioning, and each stale name inflates its ticker's buyer count.

    The trade-date window is necessarily incomplete at its recent end: a trade
    from last week may not be filed for another month.

    Returns the trades alongside one plain-language note per source, so a
    caller can always show which chamber the ranking actually saw. A source
    failing degrades to a note; only every source failing raises.
    """
    cutoff = date.today() - timedelta(days=lookback_days)
    trades: list[CongressTrade] = []
    notes: list[str] = []
    failures = 0

    for chamber, fetch in (
        ("House", fetch_house_trades),
        ("Senate", fetch_senate_trades),
    ):
        try:
            found, scanned = fetch(lookback_days)
        except SourceUnavailable as exc:
            failures += 1
            notes.append(f"{chamber}: unavailable -- {exc}")
            logger.warning("%s disclosures unavailable: %s", chamber, exc)
            continue
        recent = [t for t in found if t.trade_date >= cutoff]
        trades.extend(recent)
        notes.append(
            f"{chamber}: {scanned} periodic transaction reports scanned, "
            f"{len(found)} stock/option transactions read, {len(recent)} "
            f"traded within the last {lookback_days} days."
        )

    if failures == 2:
        raise SourceUnavailable("no congressional disclosure source was reachable")

    return trades, notes
