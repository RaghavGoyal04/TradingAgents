"""Trading212 snapshot builder: GET-only allowlist, mapping, universe."""

import pytest

from tradingagents.portfolio import t212


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    """Records every URL and refuses anything but allowlisted GETs."""

    def __init__(self, summary, positions):
        self.summary = summary
        self.positions = positions
        self.calls = []

    def get(self, url, *, auth, timeout):
        self.calls.append(url)
        assert auth == ("k", "s")
        if url.endswith("/account/summary"):
            return FakeResponse(self.summary)
        if url.endswith("/positions"):
            return FakeResponse(self.positions)
        raise AssertionError(f"unexpected URL {url}")


@pytest.fixture
def creds(monkeypatch):
    monkeypatch.setenv("TRADING212_API_KEY", "k")
    monkeypatch.setenv("TRADING212_API_SECRET", "s")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("broker", "yahoo"),
    [
        ("NVDA_US_EQ", "NVDA"),
        ("BRK.B_US_EQ", "BRK-B"),
        ("RRl_EQ", "RR.L"),
        ("SIEd_EQ", "SIE.DE"),
        ("SUp_EQ", "SU.PA"),
        ("IPXX_US_EQ", "USAR"),
    ],
)
def test_symbol_mapping(broker, yahoo):
    assert t212.trading212_to_yahoo_symbol(broker) == yahoo


@pytest.mark.unit
def test_unknown_suffix_is_unmapped():
    assert t212.trading212_to_yahoo_symbol("WEIRD_ZZ_EQ") is None


@pytest.mark.unit
def test_get_guard_rejects_non_allowlisted_endpoint():
    with pytest.raises(ValueError, match="non-allowlisted"):
        t212._get("/equity/orders", ("k", "s"), session=FakeSession({}, []))


@pytest.mark.unit
def test_fetch_snapshot_maps_aggregates_and_flags_unmapped(creds):
    session = FakeSession(
        {"totalValue": 1000.0, "currency": "GBP"},
        [
            {"instrument": {"ticker": "NVDA_US_EQ"}, "quantity": 2, "walletImpact": {"currentValue": 300.0}},
            {"instrument": {"ticker": "NVDA_US_EQ"}, "quantity": 1, "walletImpact": {"currentValue": 150.0}},
            {"instrument": {"ticker": "WEIRD_ZZ_EQ"}, "walletImpact": {"currentValue": 50.0}},
        ],
    )
    snap = t212.fetch_portfolio_snapshot(watchlist=["AAPL"], session=session)

    assert snap["source"] == "trading212"
    assert snap["base_currency"] == "GBP"
    nvda = next(p for p in snap["positions"] if p["symbol"] == "NVDA")
    assert nvda["value"] == 450.0  # aggregated
    assert nvda["quantity"] == 3.0
    assert "WEIRD_ZZ_EQ" in snap["unmapped"]
    # Cash = total - sum(position values).
    assert snap["cash"] == pytest.approx(1000.0 - 500.0)
    watch = [p for p in snap["positions"] if p.get("watch_only")]
    assert [p["symbol"] for p in watch] == ["AAPL"]


@pytest.mark.unit
def test_analyzed_symbols_excludes_zero_and_unmapped_but_keeps_watch(creds):
    snapshot = {
        "positions": [
            {"symbol": "NVDA", "value": 100, "mapping_status": "mapped", "watch_only": False},
            {"symbol": "ZERO", "value": 0.0, "mapping_status": "mapped", "watch_only": False},
            {"symbol": None, "value": 20, "mapping_status": "unmapped", "watch_only": False},
            {"symbol": "AAPL", "value": 0.0, "mapping_status": "mapped", "watch_only": True},
        ]
    }
    assert t212.analyzed_symbols(snapshot) == ["NVDA", "AAPL"]


@pytest.mark.unit
def test_snapshot_from_holdings_offline():
    snap = t212.snapshot_from_holdings(
        {"NVDA": 600.0, "GOOG": 400.0}, account_value=1200.0, currency="USD"
    )
    assert snap["source"] == "holdings_file"
    assert snap["cash"] == pytest.approx(200.0)
    assert {p["symbol"] for p in snap["positions"]} == {"NVDA", "GOOG"}


@pytest.mark.unit
def test_missing_credentials_raise(monkeypatch):
    monkeypatch.delenv("TRADING212_API_KEY", raising=False)
    monkeypatch.delenv("TRADING212_API_SECRET", raising=False)
    # Point .env lookup at an empty dir so nothing is loaded.
    monkeypatch.setattr(t212, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(t212.Trading212ConfigError):
        t212.fetch_portfolio_snapshot(session=FakeSession({}, []))
