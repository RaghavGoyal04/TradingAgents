"""Reading congressional disclosures, and ranking what they say.

The PTR fixture below is shaped exactly like ``pypdf`` output for a real House
filing, including the two things that broke the first implementation: form
labels whose padding glyphs extract as NUL bytes, and an amount band that
wrapped a line in the PDF.
"""

from datetime import date

import pytest

from tradingagents.dataflows.congress_trades import parse_ptr_text
from tradingagents.discovery import congress

# "Filing Status" and "Sub-holding Owner" as the Clerk's PDFs render them.
_FS = "F\x00\x00\x00\x00\x00 S\x00\x00\x00\x00\x00: New"
_SO = "S\x00\x00\x00\x00\x00\x00\x00\x00\x00 O\x00: Family Trust"

PTR_PAGE = f"""P        T           R
Clerk of the House of Representatives
F     I
Name: Hon. Test Member
Status: Member
State/District: XX01
T
ID Owner Asset Transaction
Type
Date Notification
Date
Amount Cap.
Gains >
$200?
JT Adobe Inc. - Common Stock (ADBE)
[ST]
S (partial) 08/14/2026 08/27/2026 $15,001 -
$50,000
{_FS}
{_SO}
CANADIAN CNTY OKLA EDL FACS AUTH EDL 3.00000% 09/01/2027
[CS]
S 08/12/2026 08/27/2026 $1,001 - $15,000
{_FS}
Carrier Global Corporation Common Stock (CARR)
[ST]
P 08/14/2026 08/27/2026 $1,001 - $15,000
{_FS}
D\x00\x00: Purchased a call option with a strike price of $30.00 and an \
expiration date of 6/17/27.
SP Intel Corporation - Common Stock (INTC) [OP] P 07/24/2026 07/24/2026 \
$250,001 - $500,000
{_FS}
Mega Corp Common Stock (MEGA) [ST] P 08/01/2026 08/10/2026 Over $50,000,000
{_FS}
US Treas Bills Int Rate 0.000% MAT 2/18/2027 (TBILL) [GS] P 08/21/2026 \
08/31/2026 $250,001 - $500,000
* For the complete list of asset type abbreviations, please visit \
https://fd.house.gov/reference/asset-type-codes.aspx.
Digitally Signed: Hon. Test Member, 09/02/2026
"""


@pytest.fixture
def parsed():
    return parse_ptr_text(
        [PTR_PAGE], member="Test Member", constituency="XX01", doc_id="12345"
    )


@pytest.mark.unit
def test_only_stock_and_option_rows_are_read(parsed):
    """A municipal bond and a Treasury bill are not growth candidates."""
    assert [t.ticker for t in parsed] == ["ADBE", "CARR", "INTC", "MEGA"]


@pytest.mark.unit
def test_wrapped_amount_band_is_recovered(parsed):
    """The PDF broke "$15,001 - $50,000" across two lines."""
    adobe = parsed[0]
    assert (adobe.amount_low, adobe.amount_high) == (15_001, 50_000)
    assert adobe.action == "sell"
    assert adobe.trade_date == date(2026, 8, 14)
    assert adobe.filing_date == date(2026, 8, 27)
    assert adobe.disclosure_lag_days == 13


@pytest.mark.unit
def test_open_ended_top_band_is_pinned_to_its_floor(parsed):
    """"Over $50,000,000" has no upper bound to estimate from."""
    mega = parsed[3]
    assert (mega.amount_low, mega.amount_high) == (50_000_000, 50_000_000)
    assert mega.amount_mid == 50_000_000


@pytest.mark.unit
def test_asset_names_survive_neighbouring_rows(parsed):
    """Each name must be clean of the previous row's prose and the owner code.

    ``CARR`` follows an untickered municipal bond that matches no record
    pattern, and ``INTC`` follows an option description ending in a full stop
    plus an "SP" (spouse) owner code.
    """
    assert [t.asset for t in parsed] == [
        "Adobe Inc. - Common Stock",
        "Carrier Global Corporation Common Stock",
        "Intel Corporation - Common Stock",
        "Mega Corp Common Stock",
    ]


@pytest.mark.unit
def test_option_purchase_keeps_its_asset_code(parsed):
    intel = parsed[2]
    assert intel.asset_code == "OP"
    assert intel.action == "buy"


@pytest.mark.unit
def test_unreadable_filing_yields_nothing():
    """A scanned paper form must not become a signal."""
    assert parse_ptr_text(
        ["Hon. Someone\nhandwritten scan with no extractable table"],
        member="Someone",
        constituency="XX01",
        doc_id="1",
    ) == []


@pytest.mark.unit
def test_large_single_buyer_outranks_a_tiny_cluster():
    """Regression: log-of-dollars compressed size until $8k beat $1.1M.

    Breadth is a real signal, but not worth 140x in position size.
    """
    big = congress._score(
        net_buy_dollars=1_125_001,
        buyer_count=1,
        days_since_trade=41,
        has_option_buys=False,
    )
    small = congress._score(
        net_buy_dollars=8_000,
        buyer_count=2,
        days_since_trade=27,
        has_option_buys=False,
    )
    assert big > small


@pytest.mark.unit
def test_score_rewards_breadth_recency_and_options():
    base = {
        "net_buy_dollars": 100_000,
        "buyer_count": 1,
        "days_since_trade": 0,
        "has_option_buys": False,
    }
    assert congress._score(**{**base, "buyer_count": 3}) > congress._score(**base)
    assert congress._score(**base) > congress._score(
        **{**base, "days_since_trade": 30}
    )
    assert congress._score(**{**base, "has_option_buys": True}) > congress._score(
        **base
    )


@pytest.mark.unit
def test_net_selling_scores_zero():
    """This ranks growth candidates; a name being sold down is not one."""
    assert congress._score(
        net_buy_dollars=-50_000,
        buyer_count=1,
        days_since_trade=1,
        has_option_buys=False,
    ) == 0.0


@pytest.mark.unit
def test_aggregate_nets_buys_against_sells_per_ticker(parsed):
    signals = {s.ticker: s for s in congress.aggregate(parsed, as_of=date(2026, 9, 3))}

    adobe = signals["ADBE"]
    assert adobe.buy_dollars == 0
    assert adobe.sell_dollars == pytest.approx(32_500.5)
    assert adobe.sellers == ("Test Member",)
    assert adobe.score == 0.0  # sold, not bought

    intel = signals["INTC"]
    assert intel.option_buys == 1
    assert intel.net_dollars == pytest.approx(375_000.5)
    assert intel.score > 0


@pytest.mark.unit
def test_scan_splits_buying_from_selling(parsed):
    scan = congress.Scan(
        as_of=date(2026, 9, 3),
        lookback_days=60,
        source_notes=("House: test",),
        signals=tuple(congress.aggregate(parsed, as_of=date(2026, 9, 3))),
    )
    assert [s.ticker for s in scan.selling] == ["ADBE"]
    assert {s.ticker for s in scan.buying} == {"CARR", "INTC", "MEGA"}
    # Ordered strongest first, and the shortlist is a prefix of that order.
    assert [s.ticker for s in scan.shortlist(2)] == [
        s.ticker for s in scan.buying[:2]
    ]


@pytest.mark.unit
def test_ranking_refuses_an_empty_window():
    """With nothing bought there is nothing to rank, and no prompt to send."""
    empty = congress.Scan(
        as_of=date(2026, 9, 3),
        lookback_days=60,
        source_notes=("House: test",),
        signals=(),
    )
    with pytest.raises(ValueError, match="nothing to rank"):
        congress.rank_top3(empty)
