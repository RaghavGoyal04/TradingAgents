"""Reading SEC Form 4 insider filings, and scoring what they say.

The fixtures below are shaped like real ``<ownershipDocument>`` payloads,
including the detail that broke the first implementation: EDGAR emits the same
boolean as ``1``/``0`` from some filing agents and ``true``/``false`` from
others, and reading only one spelling silently misses the 10b5-1 flag that
marks a purchase as pre-scheduled.
"""

from datetime import date, timedelta

import pytest

from tradingagents.dataflows import sec_form4
from tradingagents.dataflows.sec_form4 import (
    InsiderTrade,
    Throttled,
    _extract_xml,
    _flag,
    parse_form4,
)
from tradingagents.discovery import insiders


@pytest.fixture(autouse=True)
def _clear_cooldown():
    """The block cooldown is process-global; no test may leak it to the next."""
    sec_form4._blocked_until = 0.0
    yield
    sec_form4._blocked_until = 0.0

FILED = date(2026, 9, 3)


def _doc(body: str, *, ticker: str = "TEST", flag_10b5: str = "0") -> str:
    return f"""<?xml version="1.0"?>
<ownershipDocument>
  <documentType>4</documentType>
  <periodOfReport>2026-09-01</periodOfReport>
  <issuer>
    <issuerCik>0000000123</issuerCik>
    <issuerName>Test Corp</issuerName>
    <issuerTradingSymbol>{ticker}</issuerTradingSymbol>
  </issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>Doe Jane</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isOfficer>1</isOfficer>
      <officerTitle>Chief Financial Officer</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <aff10b5One>{flag_10b5}</aff10b5One>
  <nonDerivativeTable>{body}</nonDerivativeTable>
</ownershipDocument>"""


def _txn(code: str, shares: str = "1000", price: str = "50.00", after: str = "11000"):
    return f"""
    <nonDerivativeTransaction>
      <securityTitle><value>Common Stock</value></securityTitle>
      <transactionDate><value>2026-09-01</value></transactionDate>
      <transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>{shares}</value></transactionShares>
        <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
      </transactionAmounts>
      <postTransactionAmounts>
        <sharesOwnedFollowingTransaction><value>{after}</value></sharesOwnedFollowingTransaction>
      </postTransactionAmounts>
    </nonDerivativeTransaction>"""


def test_only_open_market_codes_survive():
    """Grants, exercises, withholding and gifts are compensation, not conviction."""
    doc = _doc("".join(_txn(c) for c in ["P", "S", "A", "M", "F", "G", "D", "J"]))
    codes = {t.action for t in parse_form4(doc, "acc", FILED)}
    assert codes == {"buy", "sell"}
    assert len(parse_form4(doc, "acc", FILED)) == 2


def test_boolean_flags_accept_both_spellings():
    assert [_flag(v) for v in ("1", "true", "TRUE", "0", "false", "")] == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_10b5_1_flag_read_from_either_spelling():
    """A pre-scheduled purchase must never be mistaken for a live decision."""
    for spelling in ("1", "true"):
        trade = parse_form4(_doc(_txn("P"), flag_10b5=spelling), "acc", FILED)[0]
        assert trade.scheduled_10b5_1 is True
    assert parse_form4(_doc(_txn("P")), "acc", FILED)[0].scheduled_10b5_1 is False


def test_10b5_1_also_detected_from_footnote_prose():
    """Older filings disclose the plan only in a footnote."""
    doc = _doc(_txn("P")).replace(
        "</nonDerivativeTable>",
        "</nonDerivativeTable><footnotes><footnote id='F1'>Sold under a "
        "10b5-1 trading plan.</footnote></footnotes>",
    )
    assert parse_form4(doc, "acc", FILED)[0].scheduled_10b5_1 is True


def test_director_relationship_classified():
    doc = _doc(_txn("P")).replace(
        "<isOfficer>1</isOfficer>\n      <officerTitle>Chief Financial Officer</officerTitle>",
        "<isDirector>true</isDirector><isOfficer>false</isOfficer>",
    )
    assert parse_form4(doc, "acc", FILED)[0].role == "director"


def test_cfo_outranks_a_plain_officer_title():
    assert parse_form4(_doc(_txn("P")), "acc", FILED)[0].role == "CFO"


def test_zero_price_rows_are_dropped():
    """A priced-at-zero 'purchase' is a mislabelled grant, with no cash at risk."""
    assert parse_form4(_doc(_txn("P", price="0")), "acc", FILED) == []


def test_filing_date_comes_from_the_index_not_period_of_report():
    """periodOfReport is the earliest transaction date, not the filing date."""
    trade = parse_form4(_doc(_txn("P")), "acc", FILED)[0]
    assert trade.filing_date == FILED
    assert trade.trade_date == date(2026, 9, 1)


def test_untickered_issuers_are_skipped():
    assert parse_form4(_doc(_txn("P"), ticker=""), "acc", FILED) == []


def test_value_and_position_change():
    trade = parse_form4(_doc(_txn("P", shares="1000", price="50", after="11000")), "a", FILED)[0]
    assert trade.value == 50_000
    assert trade.position_change_pct == 0.1  # 1,000 added to a 10,000 base


def test_position_change_is_negative_for_a_sale():
    """A disposal shrinks the holding, so the sign has to say so."""
    trade = parse_form4(_doc(_txn("S", shares="1000", price="50", after="9000")), "a", FILED)[0]
    assert trade.position_change_pct == -0.1  # 1,000 out of the 10,000 held before


def test_extract_xml_from_full_submission():
    """The daily index points at a .txt wrapper, not the bare XML."""
    body = f"<SEC-DOCUMENT>header\n{_doc(_txn('P'))}\n</SEC-DOCUMENT>"
    assert _extract_xml(body).startswith("<ownershipDocument")
    assert _extract_xml("no xml here") is None


def _trade(owner, day, value=100_000.0, *, action="buy", role="officer", sched=False):
    return InsiderTrade(
        ticker="AAA",
        issuer="Alpha Inc",
        issuer_cik="1",
        owner=owner,
        role=role,
        officer_title="",
        action=action,
        trade_date=day,
        filing_date=day,
        shares=value / 10,
        price=10.0,
        shares_after=0.0,
        scheduled_10b5_1=sched,
        accession="acc",
    )


def test_cluster_counts_only_a_two_week_window():
    """Buyers months apart are a coincidence, not a cluster."""
    spread = [
        _trade("A", date(2026, 1, 1)),
        _trade("B", date(2026, 4, 1)),
        _trade("C", date(2026, 8, 1)),
    ]
    assert insiders._cluster_size(spread) == 1

    tight = [
        _trade("A", date(2026, 9, 1)),
        _trade("B", date(2026, 9, 3)),
        _trade("C", date(2026, 9, 9)),
    ]
    assert insiders._cluster_size(tight) == 3


def test_cluster_found_anywhere_in_the_range():
    """The burst need not sit at the start of the lookback."""
    trades = [
        _trade("A", date(2026, 7, 1)),
        _trade("B", date(2026, 8, 20)),
        _trade("C", date(2026, 8, 21)),
        _trade("D", date(2026, 8, 22)),
    ]
    assert insiders._cluster_size(trades) == 3


def test_one_buyer_repeating_is_not_a_cluster():
    """Cluster strength is distinct people, not distinct filings."""
    same = [_trade("A", date(2026, 9, day)) for day in (1, 2, 3, 4)]
    assert insiders._cluster_size(same) == 1


def test_cluster_outranks_a_larger_single_buy():
    """The documented signal is breadth, so three buyers beat one bigger cheque."""
    today = date(2026, 9, 3)
    cluster = insiders.aggregate(
        [_trade(n, today, 200_000.0) for n in ("A", "B", "C")], as_of=today
    )[0]
    solo = insiders.aggregate([_trade("A", today, 900_000.0)], as_of=today)[0]
    assert cluster.buy_dollars < solo.buy_dollars
    assert cluster.score > solo.score
    assert cluster.is_cluster and not solo.is_cluster


def test_scheduled_buying_is_discounted():
    """A 10b5-1 purchase was decided months ago and must score below a live one."""
    today = date(2026, 9, 3)
    live = insiders.aggregate([_trade("A", today, 500_000.0)], as_of=today)[0]
    planned = insiders.aggregate(
        [_trade("A", today, 500_000.0, sched=True)], as_of=today
    )[0]
    assert planned.conviction_dollars < live.conviction_dollars
    assert planned.score < live.score
    assert planned.buy_dollars == live.buy_dollars  # headline size is unchanged


def test_sell_only_names_score_zero():
    """This screens for growth candidates; a name only being sold is not one."""
    today = date(2026, 9, 3)
    signal = insiders.aggregate(
        [_trade("A", today, 500_000.0, action="sell")], as_of=today
    )[0]
    assert signal.score == 0.0
    assert signal.net_dollars < 0


def test_stale_buying_scores_below_fresh_buying():
    today = date(2026, 9, 3)
    fresh = insiders.aggregate([_trade("A", today, 300_000.0)], as_of=today)[0]
    stale = insiders.aggregate(
        [_trade("A", today - timedelta(days=28), 300_000.0)], as_of=today
    )[0]
    assert stale.score < fresh.score


def test_cfo_buying_outscores_a_ten_percent_owner():
    today = date(2026, 9, 3)
    cfo = insiders.aggregate([_trade("A", today, 300_000.0, role="CFO")], as_of=today)[0]
    holder = insiders.aggregate(
        [_trade("A", today, 300_000.0, role="10% owner")], as_of=today
    )[0]
    assert cfo.score > holder.score


def test_the_three_meanings_of_a_403_are_kept_apart():
    """EDGAR answers 403 for three unrelated things and only the body differs.

    Conflating them cost real time: a missing index (today's, before EDGAR
    publishes it) aborted the crawl on its first day, and a User-Agent EDGAR
    refuses was reported as congestion to be waited out.
    """
    rate = sec_form4._RATE_LIMITED_BODY
    bad_agent = sec_form4._UNDECLARED_BODY

    assert rate.search("SEC.gov | Request Rate Threshold Exceeded")
    assert not bad_agent.search("SEC.gov | Request Rate Threshold Exceeded")

    assert bad_agent.search("Your Request Originates from an Undeclared Automated Tool")
    assert not rate.search("Your Request Originates from an Undeclared Automated Tool")

    assert not rate.search("<h1>Page not found</h1>")
    assert not bad_agent.search("<h1>Page not found</h1>")


def test_user_agent_carries_a_contact_and_no_url():
    """EDGAR rejects any User-Agent containing a URL, or lacking a contact."""
    agent = sec_form4._session().headers["User-Agent"]
    assert "@" in agent
    assert "http" not in agent


def test_blocked_client_stops_sending_requests():
    """Retrying into an EDGAR block extends it, so the module must back off."""
    sent = []

    class Boom:
        headers: dict = {}

        def get(self, url, timeout=None):
            sent.append(url)
            raise AssertionError("must not reach the network during cooldown")

    sec_form4._start_cooldown()
    with pytest.raises(Throttled, match="more minute"):
        sec_form4._get(Boom(), "https://www.sec.gov/x", sec_form4._RateLimiter(100))
    assert sent == []


def test_index_deduplicates_on_accession_not_path(monkeypatch):
    """One filing is listed once per party, each row under that party's own CIK.

    All those URLs serve the same document, so deduplicating on the path keeps
    every copy -- which counted each trade once per party and doubled the
    fetching.
    """
    index = (
        "Form Type   Company Name    CIK    Date Filed   File Name\n"
        "-------------------------------------------------------\n"
        "4           ISSUER INC      20430  20260902     edgar/data/20430/0000020430-26-000040.txt\n"
        "4           OWNER ONE      914208  20260902     edgar/data/914208/0000020430-26-000040.txt\n"
        "4           OWNER TWO      869013  20260902     edgar/data/869013/0000020430-26-000040.txt\n"
        "4           OTHER CORP     824142  20260902     edgar/data/824142/0000824142-26-000063.txt\n"
        "8-K         NOISE CO        12345  20260902     edgar/data/12345/0000012345-26-000001.txt\n"
    ).encode("latin-1")

    monkeypatch.setattr(sec_form4, "_get", lambda *a, **k: index)
    paths = sec_form4._index_paths(
        date(2026, 9, 2), object(), sec_form4._RateLimiter(1000)
    )
    assert len(paths) == 2, "three rows for one filing must collapse to one fetch"
    assert {p.rsplit("/", 1)[-1] for p in paths} == {
        "0000020430-26-000040.txt",
        "0000824142-26-000063.txt",
    }


def test_xsl_prefix_stripped_to_reach_raw_xml():
    """EDGAR links point at an HTML rendering; the XML sits one level up."""
    assert sec_form4._XSL_DIR_RE.sub("", "xslF345X06/form4.xml") == "form4.xml"
    assert sec_form4._XSL_DIR_RE.sub("", "wf-form4_123.xml") == "wf-form4_123.xml"


def test_evidence_renders_cluster_and_plan_tags():
    today = date(2026, 9, 3)
    scan = insiders.InsiderScan(
        as_of=today,
        lookback_days=14,
        source_note="test",
        signals=tuple(
            insiders.aggregate(
                [_trade(n, today, 200_000.0) for n in ("A", "B", "C")]
                + [_trade("D", today, 100_000.0, sched=True)],
                as_of=today,
            )
        ),
    )
    text = insiders.render_evidence(scan)
    assert "CLUSTER of 4 insiders" in text
    assert "10b5-1" in text
