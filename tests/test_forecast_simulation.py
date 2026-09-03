"""Block-bootstrap simulation invariants and risk metrics."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.forecast import simulation


def _returns(seed=0, n=300):
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0, 0.01, size=(n, 1))
    calm = market + rng.normal(0, 0.002, size=(n, 1))
    wild = 3 * market + rng.normal(0, 0.02, size=(n, 1))
    data = np.hstack([calm, wild])
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(data, columns=["CALM", "WILD"], index=idx)


@pytest.mark.unit
def test_risk_contributions_sum_to_one():
    returns = _returns()
    result = simulation.simulate_portfolio(
        returns, {"CALM": 500, "WILD": 500}, horizon=20, n_paths=2000, seed=1
    )
    assert sum(result["risk_contribution"].values()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_riskier_asset_contributes_more_risk_at_equal_weight():
    returns = _returns()
    result = simulation.simulate_portfolio(
        returns, {"CALM": 500, "WILD": 500}, horizon=20, n_paths=2000, seed=1
    )
    rc = result["risk_contribution"]
    assert rc["WILD"] > rc["CALM"]


@pytest.mark.unit
def test_pnl_scales_with_position_size():
    returns = _returns()
    small = simulation.simulate_portfolio(
        returns, {"CALM": 100, "WILD": 100}, horizon=20, n_paths=3000, seed=7
    )
    big = simulation.simulate_portfolio(
        returns, {"CALM": 1000, "WILD": 1000}, horizon=20, n_paths=3000, seed=7
    )
    # 10x capital -> ~10x the P5 loss magnitude.
    assert big["pnl"]["p5"] == pytest.approx(small["pnl"]["p5"] * 10, rel=0.05)


@pytest.mark.unit
def test_determinism_with_seed():
    returns = _returns()
    a = simulation.simulate_portfolio(returns, {"CALM": 500, "WILD": 500}, horizon=20, n_paths=1000, seed=42)
    b = simulation.simulate_portfolio(returns, {"CALM": 500, "WILD": 500}, horizon=20, n_paths=1000, seed=42)
    assert a["pnl"] == b["pnl"]
    assert a["var_95"] == b["var_95"]


@pytest.mark.unit
def test_loss_probability_and_drawdown_in_valid_ranges():
    returns = _returns()
    result = simulation.simulate_portfolio(
        returns, {"CALM": 500, "WILD": 500}, horizon=20, n_paths=2000, seed=3
    )
    assert 0.0 <= result["loss_probability"] <= 1.0
    assert result["max_drawdown_p50"] <= 0.0  # drawdown is non-positive


@pytest.mark.unit
def test_empty_or_no_overlap_raises():
    returns = _returns()
    with pytest.raises(ValueError):
        simulation.simulate_portfolio(returns, {"OTHER": 100}, horizon=20)


@pytest.mark.unit
def test_sample_shape_matches_request():
    returns = _returns().to_numpy()
    rng = np.random.default_rng(0)
    sample = simulation.sample_daily_returns(returns, horizon=20, n_paths=50, block_size=5, rng=rng)
    assert sample.shape == (50, 20, 2)
