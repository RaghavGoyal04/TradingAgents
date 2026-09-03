"""Next-day advisory action rules and pie rebalance plan."""

import pytest

from tradingagents.portfolio.actions import (
    decision_table,
    next_day_actions,
    rebalance_plan,
)


def _pie_snapshot():
    return {
        "base_currency": "GBP",
        "positions": [
            {"symbol": "NVDA", "name": "Nvidia", "broker_ticker": "NVDA_US_EQ",
             "value": 625.0, "weight": 0.1, "watch_only": False,
             "mapping_status": "mapped"},
        ],
        "unmapped": [],
        "pies": [
            {
                "id": 1,
                "name": "Diversified pie",
                "instruments": [
                    # 13% actual vs 9% target => 4pp overweight, trim
                    {"broker_ticker": "NVDA_US_EQ", "target_share": 0.09,
                     "current_share": 0.13, "value": 625.0},
                    # 2% actual vs 6% target => 4pp underweight, buy
                    {"broker_ticker": "PANW_US_EQ", "target_share": 0.06,
                     "current_share": 0.02, "value": 290.0},
                    # bang on target => hold
                    {"broker_ticker": "GOOG_US_EQ", "target_share": 0.07,
                     "current_share": 0.070, "value": 436.0},
                ],
            }
        ],
    }


@pytest.mark.unit
def test_rebalance_flags_overweight_as_trim_and_underweight_as_buy():
    plan = {r["broker_ticker"]: r for r in rebalance_plan(_pie_snapshot())}
    assert plan["NVDA_US_EQ"]["action"] == "TRIM"
    assert plan["NVDA_US_EQ"]["amount"] < 0
    assert plan["PANW_US_EQ"]["action"] == "BUY"
    assert plan["PANW_US_EQ"]["amount"] > 0


@pytest.mark.unit
def test_rebalance_holds_when_on_target():
    plan = {r["broker_ticker"]: r for r in rebalance_plan(_pie_snapshot())}
    assert plan["GOOG_US_EQ"]["action"] == "HOLD"


@pytest.mark.unit
def test_rebalance_amount_is_drift_times_pie_value():
    plan = {r["broker_ticker"]: r for r in rebalance_plan(_pie_snapshot())}
    pie_value = 625.0 + 290.0 + 436.0
    assert plan["NVDA_US_EQ"]["amount"] == pytest.approx(-(0.13 - 0.09) * pie_value)


@pytest.mark.unit
def test_rebalance_uses_holding_name_not_broker_ticker():
    plan = {r["broker_ticker"]: r for r in rebalance_plan(_pie_snapshot())}
    assert plan["NVDA_US_EQ"]["name"] == "Nvidia"


@pytest.mark.unit
def test_rebalance_sorted_by_largest_drift():
    drifts = [abs(r["drift"]) for r in rebalance_plan(_pie_snapshot())]
    assert drifts == sorted(drifts, reverse=True)


@pytest.mark.unit
def test_tiny_drift_below_min_trade_value_is_hold():
    snapshot = {
        "base_currency": "GBP",
        "positions": [],
        "pies": [
            {
                "id": 1,
                "name": "Small pie",
                "instruments": [
                    # 5pp drift, but the pie is tiny so the trade is ~5 GBP.
                    {"broker_ticker": "A_EQ", "target_share": 0.5,
                     "current_share": 0.55, "value": 55.0},
                    {"broker_ticker": "B_EQ", "target_share": 0.5,
                     "current_share": 0.45, "value": 45.0},
                ],
            }
        ],
    }
    assert all(r["action"] == "HOLD" for r in rebalance_plan(snapshot))


@pytest.mark.unit
def test_rebalance_is_not_duplicated_into_the_advisory_list():
    """Rebalancing has its own section; repeating it buries the risk notes."""
    titles = " ".join(a["title"] for a in next_day_actions(_pie_snapshot(), {}, {}))
    assert "Diversified pie" not in titles


@pytest.mark.unit
def test_no_pies_yields_empty_plan():
    assert rebalance_plan({"positions": [], "pies": []}) == []
    assert rebalance_plan(None) == []


def _snapshot(**overrides):
    base = {
        "base_currency": "GBP",
        "positions": [
            {"symbol": "NVDA", "name": "Nvidia", "value": 600.0, "weight": 0.30,
             "watch_only": False, "mapping_status": "mapped"},
            {"symbol": "GOOG", "name": "Alphabet", "value": 200.0, "weight": 0.10,
             "watch_only": False, "mapping_status": "mapped"},
        ],
        "unmapped": [],
    }
    base.update(overrides)
    return base


@pytest.mark.unit
def test_buy_recommendation_becomes_a_ranked_decision():
    recs = {
        "results": [
            {"ticker": "NVDA", "status": "success", "action": "BUY",
             "trade_value": 500, "executive_summary": "Momentum strong."},
        ]
    }
    rows = decision_table(_snapshot(), recs)
    assert len(rows) == 1
    assert rows[0]["action"] == "BUY"
    assert rows[0]["name"] == "Nvidia"
    assert rows[0]["amount"] == 500
    assert rows[0]["plan"] == "Momentum strong."


@pytest.mark.unit
def test_hold_only_recommendations_report_no_trades():
    recs = {"results": [{"ticker": "NVDA", "status": "success", "action": "HOLD"}]}
    actions = next_day_actions(_snapshot(), {}, recs)
    assert any("No buy or sell calls" in a["title"] for a in actions)


@pytest.mark.unit
def test_missing_agent_run_explains_how_to_get_recommendations():
    actions = next_day_actions(_snapshot(), {}, {})
    assert any("No agent analysis" in a["title"] for a in actions)
    assert any("Full analysis" in a["detail"] for a in actions)


@pytest.mark.unit
def test_single_position_over_limit_flagged_by_name():
    actions = next_day_actions(_snapshot(), {}, {})
    titles = " ".join(a["title"] for a in actions)
    assert "Nvidia alone is 30%" in titles


@pytest.mark.unit
def test_concentration_flagged_when_top5_above_limit():
    snapshot = _snapshot(
        positions=[
            {"symbol": f"S{i}", "name": f"Name{i}", "value": 100.0, "weight": 0.12,
             "watch_only": False, "mapping_status": "mapped"}
            for i in range(6)
        ]
    )
    actions = next_day_actions(snapshot, {}, {})
    assert any("top 5 holdings" in a["title"] for a in actions)


@pytest.mark.unit
def test_unmapped_positions_are_high_priority():
    actions = next_day_actions(_snapshot(unmapped=["WEIRD_ZZ_EQ"]), {}, {})
    flagged = [a for a in actions if "could not be priced" in a["title"]]
    assert flagged and flagged[0]["priority"] == "high"


@pytest.mark.unit
def test_high_loss_probability_flagged():
    forecasts = {"portfolio": {"loss_probability": 0.62, "var_95": 800}}
    actions = next_day_actions(_snapshot(), forecasts, {})
    assert any("62%" in a["title"] for a in actions)
    assert any("bad month" in a["title"] for a in actions)


@pytest.mark.unit
def test_actions_sorted_high_priority_first():
    recs = {"results": [{"ticker": "NVDA", "action": "SELL"}]}
    actions = next_day_actions(_snapshot(unmapped=["X_EQ"]), {}, recs)
    priorities = [a["priority"] for a in actions]
    assert priorities == sorted(priorities, key=lambda p: {"high": 0, "medium": 1, "info": 2}[p])


@pytest.mark.unit
def test_handles_completely_empty_inputs():
    assert next_day_actions(None, None, None)  # returns guidance, does not crash


def _decision_snapshot():
    return {
        "base_currency": "GBP",
        "positions": [
            {"symbol": "MSFT", "name": "Microsoft", "current_price": 510.0,
             "value": 211.0, "weight": 0.01, "watch_only": False,
             "mapping_status": "mapped"},
            {"symbol": "AVGO", "name": "Broadcom", "current_price": 400.0,
             "value": 330.0, "weight": 0.015, "watch_only": False,
             "mapping_status": "mapped"},
            {"symbol": "CNX1.L", "name": "iShares NASDAQ 100",
             "current_price": 125000.0,  # GBX pence
             "value": 2170.0, "weight": 0.1, "watch_only": False,
             "mapping_status": "mapped"},
        ],
        "unmapped": [],
    }


def _decision_recs():
    return {
        "results": [
            {"ticker": "MSFT", "status": "success", "action": "BUY",
             "trade_value": 927.13, "price_target": "540.0",
             "time_horizon": "6-12 months", "rating": "Overweight",
             "executive_summary": "Scale in at 510, 495-500, stop below 448."},
            {"ticker": "AVGO", "status": "success", "action": "SELL",
             "trade_value": 165.32, "price_target": "340.0",
             "time_horizon": "3-6 months", "rating": "Underweight",
             "executive_summary": "Trim half."},
            {"ticker": "CNX1.L", "status": "success", "action": "BUY",
             "trade_value": 500.0, "price_target": "1300.0",  # GBP vs GBX
             "time_horizon": "12 months", "rating": "Overweight",
             "executive_summary": "Add."},
            {"ticker": "AMZN", "status": "success", "action": "HOLD",
             "trade_value": 0.0, "price_target": "263.0"},
            {"ticker": "IPXX", "status": "error", "error": "NoMarketDataError"},
        ]
    }


@pytest.mark.unit
def test_conviction_outranks_size():
    """A Medium call must sit above a Low one even when it trades far less."""
    rows = decision_table(_decision_snapshot(), _decision_recs())
    order = [r["ticker"] for r in rows]
    # AVGO is Medium on GBP 165; MSFT is Low on GBP 927.
    assert order.index("AVGO") < order.index("MSFT")


@pytest.mark.unit
def test_decisions_exclude_holds_and_errors():
    tickers = {r["ticker"] for r in decision_table(_decision_snapshot(), _decision_recs())}
    assert "AMZN" not in tickers  # HOLD is not an action
    assert "IPXX" not in tickers  # failed ticker never becomes a decision


@pytest.mark.unit
def test_upside_computed_from_price_target():
    rows = {r["ticker"]: r for r in decision_table(_decision_snapshot(), _decision_recs())}
    assert rows["MSFT"]["upside"] == pytest.approx(540 / 510 - 1)
    assert rows["MSFT"]["conviction"] == "Low"  # ~5.9% move


@pytest.mark.unit
def test_high_conviction_when_target_far_away():
    rows = {r["ticker"]: r for r in decision_table(_decision_snapshot(), _decision_recs())}
    # AVGO target 340 vs price 400 => -15% => Medium
    assert rows["AVGO"]["conviction"] == "Medium"


@pytest.mark.unit
def test_currency_unit_mismatch_suppresses_upside():
    """GBP target against a GBX (pence) price must not show a -99% call."""
    rows = {r["ticker"]: r for r in decision_table(_decision_snapshot(), _decision_recs())}
    assert rows["CNX1.L"]["upside"] is None
    assert rows["CNX1.L"]["conviction"] == "Unknown"


@pytest.mark.unit
def test_decision_carries_entry_plan_and_name():
    rows = {r["ticker"]: r for r in decision_table(_decision_snapshot(), _decision_recs())}
    assert rows["MSFT"]["name"] == "Microsoft"
    assert "Scale in" in rows["MSFT"]["plan"]
    assert rows["MSFT"]["horizon"] == "6-12 months"


@pytest.mark.unit
def test_trades_no_longer_duplicated_in_housekeeping():
    """Decisions live in the table; the advisory list must stay housekeeping."""
    actions = next_day_actions(_decision_snapshot(), {}, _decision_recs())
    titles = " ".join(a["title"] for a in actions)
    assert "BUY Microsoft" not in titles


@pytest.mark.unit
def test_decision_table_empty_without_recommendations():
    assert decision_table(_decision_snapshot(), {}) == []
    assert decision_table(None, None) == []


@pytest.mark.unit
def test_decisions_sorted_high_conviction_first():
    rows = decision_table(_decision_snapshot(), _decision_recs())
    ranks = [{"High": 0, "Medium": 1, "Low": 2, "Unknown": 3}[r["conviction"]]
             for r in rows]
    assert ranks == sorted(ranks)


@pytest.mark.unit
def test_money_at_stake_breaks_ties_within_a_band():
    snapshot = {
        "base_currency": "GBP",
        "positions": [
            {"symbol": "A", "name": "Alpha", "current_price": 100.0},
            {"symbol": "B", "name": "Bravo", "current_price": 100.0},
        ],
    }
    recs = {"results": [
        # Both 30% upside => both High; the bigger trade must lead.
        {"ticker": "A", "status": "success", "action": "BUY",
         "trade_value": 100, "price_target": "130"},
        {"ticker": "B", "status": "success", "action": "BUY",
         "trade_value": 900, "price_target": "130"},
    ]}
    rows = decision_table(snapshot, recs)
    assert [r["ticker"] for r in rows] == ["B", "A"]


@pytest.mark.unit
def test_conviction_score_is_sortable_number_matching_its_band():
    rows = decision_table(_decision_snapshot(), _decision_recs())
    for r in rows:
        score = r["conviction_score"]
        if score is None:
            assert r["conviction"] == "Unknown"
            continue
        assert 0 <= score <= 100
        expected = "High" if score >= 50 else "Medium" if score >= 25 else "Low"
        assert r["conviction"] == expected


@pytest.mark.unit
def test_conviction_score_scales_with_distance_to_target():
    snapshot = {"positions": [{"symbol": "X", "name": "X", "current_price": 100.0}]}

    def score(target):
        recs = {"results": [{"ticker": "X", "status": "success", "action": "BUY",
                             "trade_value": 10, "price_target": str(target)}]}
        return decision_table(snapshot, recs)[0]["conviction_score"]

    assert score(110) == 25   # 10% move -> Medium floor
    assert score(120) == 50   # 20% move -> High floor
    assert score(140) == 100  # 40% move -> full scale
    assert score(180) == 100  # capped, never above 100


@pytest.mark.unit
def test_sell_with_target_above_price_is_not_scored():
    """A sell's target is often an exit level above spot; that is not conviction."""
    snapshot = {"positions": [{"symbol": "IREN", "name": "IREN",
                               "current_price": 41.3}]}
    recs = {"results": [{"ticker": "IREN", "status": "success", "action": "SELL",
                         "trade_value": 45, "price_target": "46.2"}]}
    row = decision_table(snapshot, recs)[0]
    assert row["conviction"] == "Unknown"
    assert row["conviction_score"] is None
    assert row["upside"] == pytest.approx(46.2 / 41.3 - 1)  # still reported


@pytest.mark.unit
def test_sell_with_target_below_price_is_scored():
    snapshot = {"positions": [{"symbol": "AVGO", "name": "Broadcom",
                               "current_price": 400.0}]}
    recs = {"results": [{"ticker": "AVGO", "status": "success", "action": "SELL",
                         "trade_value": 165, "price_target": "300"}]}
    row = decision_table(snapshot, recs)[0]
    assert row["conviction"] == "High"  # 25% downside
    assert row["conviction_score"] == 62  # 0.25 / 0.40 * 100, rounded


@pytest.mark.unit
def test_buy_with_target_below_price_is_not_scored():
    snapshot = {"positions": [{"symbol": "X", "name": "X", "current_price": 100.0}]}
    recs = {"results": [{"ticker": "X", "status": "success", "action": "BUY",
                         "trade_value": 50, "price_target": "90"}]}
    assert decision_table(snapshot, recs)[0]["conviction"] == "Unknown"
