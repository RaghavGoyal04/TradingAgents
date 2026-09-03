"""FX conversion and base-currency panel construction."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.portfolio.fx import FxConverter, currency_for_symbol
from tradingagents.portfolio.prices import to_base_currency_panel


@pytest.mark.unit
@pytest.mark.parametrize(
    ("symbol", "currency"),
    [("NVDA", "USD"), ("RR.L", "GBP"), ("SIE.DE", "EUR"), ("SU.PA", "EUR"), ("7203.T", "JPY")],
)
def test_currency_for_symbol(symbol, currency):
    assert currency_for_symbol(symbol) == currency


@pytest.mark.unit
def test_converter_uses_source_and_caches():
    calls = []

    def source(base, quote):
        calls.append((base, quote))
        return 0.8  # 1 USD -> 0.8 GBP

    fx = FxConverter("GBP", rate_source=source)
    assert fx.to_base(100.0, "USD") == pytest.approx(80.0)
    assert fx.to_base(50.0, "USD") == pytest.approx(40.0)
    assert fx.rate("GBP", "GBP") == 1.0
    assert calls == [("USD", "GBP")]  # cached after first lookup


@pytest.mark.unit
def test_converter_falls_back_to_inverse_pair():
    def source(base, quote):
        return 1.25 if (base, quote) == ("GBP", "USD") else None

    fx = FxConverter("GBP", rate_source=source)
    assert fx.rate("USD", "GBP") == pytest.approx(0.8)


@pytest.mark.unit
def test_base_currency_panel_applies_fx_series():
    idx = pd.bdate_range("2026-01-01", periods=5)
    panel = pd.DataFrame({"RR.L": np.arange(1.0, 6.0)}, index=idx)

    def fx_loader(local, base, index):
        assert (local, base) == ("GBP", "USD")
        return pd.Series(2.0, index=index)  # 1 GBP -> 2 USD

    base_panel, warnings = to_base_currency_panel(panel, "USD", fx_series_loader=fx_loader)
    assert warnings == []
    assert base_panel["RR.L"].tolist() == [2.0, 4.0, 6.0, 8.0, 10.0]


@pytest.mark.unit
def test_base_currency_panel_warns_when_fx_missing():
    idx = pd.bdate_range("2026-01-01", periods=3)
    panel = pd.DataFrame({"SIE.DE": [1.0, 2.0, 3.0]}, index=idx)
    base_panel, warnings = to_base_currency_panel(
        panel, "USD", fx_series_loader=lambda *a: None
    )
    assert base_panel["SIE.DE"].tolist() == [1.0, 2.0, 3.0]  # kept local
    assert any("unavailable" in w for w in warnings)
