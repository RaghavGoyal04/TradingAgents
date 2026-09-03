"""Mocked end-to-end: real forecast engine driven through the orchestrator.

No network: prices come from a synthetic loader, agents are skipped. Verifies the
full ingest -> forecast lifecycle writes a valid ``forecasts.json``.
"""

import json
from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tradingagents.forecast.run import generate_forecasts
from tradingagents.portfolio import orchestrator, sleep_guard
from tradingagents.portfolio.contracts import FORECASTS, read_json


@pytest.fixture(autouse=True)
def _no_caffeinate(monkeypatch):
    monkeypatch.setattr(sleep_guard, "is_supported", lambda: False)


def _synthetic_loader(symbol, curr_date):
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    dates = pd.bdate_range("2023-01-01", periods=400)
    rets = rng.normal(0.0003, 0.02, size=len(dates))
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame({"Date": dates, "Close": close})


@pytest.mark.unit
def test_orchestrated_forecast_writes_portfolio_block(tmp_path):
    run_dir = tmp_path / "run"
    holdings = tmp_path / "holdings.json"
    holdings.write_text(json.dumps({"NVDA": 600.0, "GOOG": 400.0}), encoding="utf-8")

    args = SimpleNamespace(
        run_dir=str(run_dir), trading212=False, holdings=str(holdings), watchlist=[],
        capital=1000.0, currency="USD", analysis_date="2024-06-01",
        quick_model="q", deep_model="d", aws_profile="p", aws_region="r",
        resume=False, skip_agents=True, skip_forecast=False,
    )

    forecast_phase = partial(generate_forecasts, price_loader=_synthetic_loader, n_paths=1000)
    code = orchestrator.orchestrate(args, forecast_phase=forecast_phase)

    assert code == 0
    payload = read_json(run_dir / FORECASTS)
    assert payload["schema_version"] == "1.0.0"
    portfolio = payload["portfolio"]
    assert portfolio["currency"] == "USD"
    assert set(portfolio["symbols"]) == {"NVDA", "GOOG"}
    assert 0.0 <= portfolio["loss_probability"] <= 1.0
    assert sum(portfolio["risk_contribution"].values()) == pytest.approx(1.0, abs=1e-6)
    assert set(payload["evaluation"]) == {"1", "5", "20"}
