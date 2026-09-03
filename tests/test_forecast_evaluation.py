"""Leakage-safe walk-forward evaluation and the TimesFM promotion gate."""

import numpy as np
import pytest

from tradingagents.forecast import evaluation as ev


@pytest.mark.unit
def test_pinball_loss_is_zero_for_perfect_prediction():
    assert ev.pinball_loss(0.5, {0.5: 0.5}) == pytest.approx(0.0)


@pytest.mark.unit
def test_pinball_loss_penalizes_asymmetrically():
    # High quantile under-predicting the outcome is penalized by the level.
    under = ev.pinball_loss(1.0, {0.9: 0.0})
    over = ev.pinball_loss(-1.0, {0.9: 0.0})
    assert under == pytest.approx(0.9)
    assert over == pytest.approx(0.1)


@pytest.mark.unit
def test_rolling_origin_never_leaks_future_data():
    series = np.arange(400, dtype=float)
    seen_lengths = []

    def spy(train, horizon, levels):
        seen_lengths.append(len(train))
        # If this ever saw >= its own origin index, it would be leakage.
        return dict.fromkeys(levels, 0.0)

    ev.rolling_origin_score(series, horizon=5, forecaster=spy, min_train=252)
    # Every training slice must end before the scored horizon window: the
    # largest train length is at most len(series) - horizon.
    assert max(seen_lengths) <= len(series) - 5


@pytest.mark.unit
def test_rolling_origin_returns_none_without_enough_history():
    assert ev.rolling_origin_score(np.zeros(100), horizon=5, forecaster=ev.naive_forecaster) is None


@pytest.mark.unit
def test_promotion_requires_margin_on_enough_holdings():
    # Two holdings: both improved by >5% -> promoted.
    per_symbol = {
        "A": {"eligible": True, "best_baseline": 1.0, "scores": {"bootstrap": 1.0, "timesfm": 0.90}},
        "B": {"eligible": True, "best_baseline": 1.0, "scores": {"bootstrap": 1.0, "timesfm": 0.80}},
    }
    decision = ev.promotion_decision(per_symbol, horizon=20)
    assert decision["promoted"] is True
    assert decision["winner"] == "timesfm"


@pytest.mark.unit
def test_promotion_rejected_when_margin_too_small():
    per_symbol = {
        "A": {"eligible": True, "best_baseline": 1.0, "scores": {"bootstrap": 1.0, "timesfm": 0.99}},
        "B": {"eligible": True, "best_baseline": 1.0, "scores": {"bootstrap": 1.0, "timesfm": 0.80}},
    }
    decision = ev.promotion_decision(per_symbol, horizon=20)
    # Only 1 of 2 improved by >=5% -> below the 80% fraction.
    assert decision["promoted"] is False
    assert decision["winner"] == "bootstrap"


@pytest.mark.unit
def test_promotion_no_candidate_selects_baseline():
    per_symbol = {
        "A": {"eligible": True, "best_baseline": 1.0, "scores": {"bootstrap": 1.0, "naive": 1.2}},
    }
    decision = ev.promotion_decision(per_symbol, horizon=5)
    assert decision["promoted"] is False
    assert decision["winner"] == "bootstrap"


@pytest.mark.unit
def test_evaluate_symbol_marks_short_history_ineligible():
    result = ev.evaluate_symbol(np.zeros(50), horizon=5, candidate=None)
    assert result["eligible"] is False
