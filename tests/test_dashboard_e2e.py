"""End-to-end dashboard scenarios driven through Streamlit AppTest.

AppTest executes the real ``dashboard/app.py`` and simulates the actual widget
interactions (radio, text inputs, buttons, selectbox), so these are true UI-level
scenario tests. ``subprocess.Popen`` is patched so clicking Run never spawns a
real orchestrator. Skipped entirely when Streamlit is not installed.
"""

import json
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


@pytest.fixture
def no_spawn(monkeypatch):
    """Patch Popen so Run/Resume never launches a real process; record calls.

    ``launcher.launch`` does ``import subprocess`` internally, so patch the
    stdlib module's ``Popen`` (that is what it resolves at call time).
    """
    import subprocess

    calls = []

    class FakePopen:
        def __init__(self, command, **kwargs):
            calls.append(command)
            self.pid = 4242

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    return calls


def _seed_run(base):
    from tradingagents.portfolio import contracts, manifest as manifest_mod

    run_dir = base / "portfolio" / "2026-09-03"
    run_dir.mkdir(parents=True)
    snapshot = {
        "schema_version": "1.0.0", "base_currency": "GBP", "account_value": 1000.0,
        "cash": 100.0,
        "positions": [
            {"symbol": "NVDA", "name": "Nvidia", "value": 500.0, "weight": 0.5, "quantity": 3,
             "mapping_status": "mapped", "watch_only": False},
            {"symbol": "GOOG", "name": "Alphabet", "value": 400.0, "weight": 0.4, "quantity": 2,
             "mapping_status": "mapped", "watch_only": False},
            {"symbol": "AAPL", "name": "Apple", "value": 0.0, "weight": 0.0, "quantity": None,
             "mapping_status": "mapped", "watch_only": True},
        ],
        "unmapped": ["WEIRD_ZZ_EQ"], "watchlist": ["AAPL"],
    }
    contracts.atomic_write_json(run_dir / contracts.PORTFOLIO_SNAPSHOT, snapshot)
    manifest_mod.new_manifest(
        run_dir, analysis_date="2026-09-03", snapshot=snapshot,
        settings={"quick_model": "q", "deep_model": "d", "base_currency": "GBP"},
    )
    (run_dir / contracts.FORECASTS).write_text(json.dumps({
        "schema_version": "1.0.0", "model_used": "bootstrap_v1", "data_cutoff": "2026-09-03",
        "assets": {
            "NVDA": {"history_tail": [1, 2, 3], "quantiles": {"20": {"p10": -0.1, "p50": 0.0, "p90": 0.1}}},
            "GOOG": {"history_tail": [4, 5, 6], "quantiles": {"20": {"p10": -0.2, "p50": 0.0, "p90": 0.2}}},
        },
        "portfolio": {"pnl": {"p5": -50, "p50": 1, "p95": 60}, "loss_probability": 0.3,
                       "var_95": 50, "cvar_95": 60, "max_drawdown_p50": -0.1,
                       "risk_contribution": {"NVDA": 0.6, "GOOG": 0.4}, "currency": "GBP",
                       "warnings": ["insufficient history dropped: SPCX"]},
        "evaluation": {"20": {"promoted": False, "winner": "bootstrap_v1"}},
    }), encoding="utf-8")
    # A recommendations artifact for the Analysis tab.
    agents = run_dir / "agents"
    agents.mkdir()
    (agents / "recommendations.json").write_text(json.dumps({
        "results": [
            {"ticker": "NVDA", "status": "success", "rating": "Buy",
             "action": "BUY", "executive_summary": "Add on strength.", "sizing": "buy 2%"},
            {"ticker": "GOOG", "status": "success", "rating": "Hold",
             "executive_summary": "Hold."},
        ]
    }), encoding="utf-8")
    return base


# --- Scenario 1: empty state -------------------------------------------------

@pytest.mark.unit
def test_scenario_empty_state(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_RESULTS_DIR", str(tmp_path))
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception
    assert app.error == []
    assert any("No runs" in i.value for i in app.info)


# --- Scenario 2: full render of a seeded run --------------------------------

@pytest.mark.unit
def test_scenario_full_run_renders_all_tabs(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_RESULTS_DIR", str(tmp_path))
    _seed_run(tmp_path)
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert not app.exception

    labels = " | ".join(m.label for m in app.metric)
    # Currency is in the LABEL (not the value) so values never truncate.
    assert "Account value (GBP)" in labels
    assert "Cash (GBP)" in labels
    assert "Holdings" in labels
    # Overview intentionally surfaces the unmapped-position error badge.
    assert any("WEIRD_ZZ_EQ" in str(e.value) for e in app.error)
    # Risk warnings from the forecast surface.
    assert any("insufficient history" in w.value for w in app.warning)
    # Company names are shown, not just tickers.
    body = " ".join(str(m.value) for m in app.markdown)
    assert "Nvidia" in body
    # The "what to do next" panel is present.
    assert any("What to do next" in str(s.value) for s in app.subheader)


# --- Scenario 3: forecasts asset switch -------------------------------------

@pytest.mark.unit
def test_scenario_switch_forecast_asset(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_RESULTS_DIR", str(tmp_path))
    _seed_run(tmp_path)
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    app.selectbox(key="fc_asset").select("GOOG").run()
    assert not app.exception
    assert app.selectbox(key="fc_asset").value == "GOOG"


# --- Scenario 4: Run click with invalid holdings config surfaces an error ---

@pytest.mark.unit
def test_scenario_run_holdings_without_path_errors(tmp_path, monkeypatch, no_spawn):
    monkeypatch.setenv("TRADINGAGENTS_RESULTS_DIR", str(tmp_path))
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    app.radio(key="source").set_value("holdings").run()
    app.button(key="run_btn").click().run()
    assert not app.exception
    assert any("holdings" in e.value.lower() for e in app.error)
    assert no_spawn == []  # nothing spawned


# --- Scenario 5: Run click with trading212 spawns the orchestrator ----------

@pytest.mark.unit
def test_scenario_run_trading212_launches_orchestrator(tmp_path, monkeypatch, no_spawn):
    monkeypatch.setenv("TRADINGAGENTS_RESULTS_DIR", str(tmp_path))
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    app.radio(key="run_mode").set_value("Forecast only (fast)").run()
    app.button(key="run_btn").click().run()
    assert not app.exception
    assert len(no_spawn) == 1
    command = no_spawn[0]
    assert "--trading212" in command
    assert "--skip-agents" in command
    assert "tradingagents.portfolio.orchestrator" in command
    assert any("Started orchestrator" in s.value for s in app.success)


# --- Scenario 6: Resume disabled for a brand-new run ------------------------

@pytest.mark.unit
def test_scenario_resume_disabled_for_new_run(tmp_path, monkeypatch, no_spawn):
    monkeypatch.setenv("TRADINGAGENTS_RESULTS_DIR", str(tmp_path))
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    assert app.button(key="resume_btn").disabled is True


@pytest.mark.unit
def test_scenario_insider_confirmation_of_the_congress_shortlist(tmp_path, monkeypatch):
    """The Form 4 step reads EDGAR, so both sources are stubbed.

    What this pins down is the wiring: scanning disclosures produces a
    shortlist, the Form 4 button checks exactly those tickers, and a name both
    groups bought is called out rather than left for the reader to spot.
    """
    from datetime import date

    from tradingagents.dataflows.sec_form4 import InsiderTrade
    from tradingagents.discovery import congress, insiders

    today = date(2026, 9, 3)
    shortlisted = congress.TickerSignal(
        ticker="PFE", company="Pfizer Inc", buy_dollars=100_000.0, sell_dollars=0.0,
        buyers=("Hon. Test Member",), sellers=(), option_buys=0,
        latest_trade_date=today, latest_filing_date=today, trades=(), score=2.0,
    )
    monkeypatch.setattr(
        congress, "scan",
        lambda *a, **k: congress.Scan(
            as_of=today, lookback_days=60,
            source_notes=("House: 1 filing read.",), signals=(shortlisted,),
        ),
    )

    checked = []

    def fake_confirm(tickers, lookback_days, **kwargs):
        checked.append(list(tickers))
        buys = [
            InsiderTrade(
                ticker="PFE", issuer="Pfizer Inc", issuer_cik="78003", owner=owner,
                role="CEO", officer_title="", action="buy", trade_date=today,
                filing_date=today, shares=1000.0, price=100.0, shares_after=0.0,
                scheduled_10b5_1=False, accession="acc",
            )
            for owner in ("Bourla A", "Blaylock R", "Buckley M")
        ]
        return insiders.InsiderScan(
            as_of=today, lookback_days=lookback_days, source_note="1 ticker checked.",
            signals=tuple(insiders.aggregate(buys, as_of=today)),
        )

    monkeypatch.setattr(insiders, "confirm", fake_confirm)
    monkeypatch.setenv("TRADINGAGENTS_RESULTS_DIR", str(tmp_path))

    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    app.button(key="congress_scan_btn").click().run()
    app.button(key="insider_check_btn").click().run()

    assert app.exception == []
    assert checked == [["PFE"]], "Form 4 must be read for the shortlist, nothing else"
    assert any("PFE" in s.value for s in app.success)
    assert app.session_state["insider_scan"].signals[0].is_cluster


@pytest.mark.unit
def test_model_pickers_hidden_when_no_agents_run(tmp_path, monkeypatch):
    """Forecast-only makes no LLM calls; a live model picker there would imply
    a choice that cannot affect the output."""
    monkeypatch.setenv("TRADINGAGENTS_RESULTS_DIR", str(tmp_path))
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    model_labels = {"Provider", "Quick model", "Deep model", "Reasoning effort"}
    assert model_labels & {s.label for s in app.get("selectbox")}

    app.radio(key="run_mode").set_value("Forecast only (fast)").run()
    assert not (model_labels & {s.label for s in app.get("selectbox")})
    assert any("no LLM is called" in c.value for c in app.caption)
    assert app.exception == []
