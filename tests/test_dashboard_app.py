"""Streamlit dashboard smoke test via AppTest (skipped if streamlit absent)."""

import json
import os
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def _seed_run(base, contracts, manifest_mod):
    run_dir = base / "portfolio" / "2026-09-03"
    run_dir.mkdir(parents=True)
    snapshot = {
        "schema_version": "1.0.0",
        "base_currency": "GBP",
        "account_value": 1000.0,
        "cash": 100.0,
        "positions": [
            {"symbol": "NVDA", "name": "Nvidia", "value": 500.0, "weight": 0.5, "quantity": 3,
             "mapping_status": "mapped", "watch_only": False},
            {"symbol": "GOOG", "name": "Alphabet", "value": 400.0, "weight": 0.4, "quantity": 2,
             "mapping_status": "mapped", "watch_only": False},
        ],
        "unmapped": [],
        "watchlist": [],
    }
    contracts.atomic_write_json(run_dir / contracts.PORTFOLIO_SNAPSHOT, snapshot)
    manifest_mod.new_manifest(
        run_dir, analysis_date="2026-09-03", snapshot=snapshot,
        settings={"quick_model": "q", "deep_model": "d", "base_currency": "GBP"},
    )
    (run_dir / contracts.FORECASTS).write_text(json.dumps({
        "schema_version": "1.0.0", "model_used": "bootstrap_v1", "data_cutoff": "2026-09-03",
        "assets": {"NVDA": {"history_tail": [1, 2, 3], "quantiles": {"20": {"p10": -0.1, "p50": 0.0, "p90": 0.1}}}},
        "portfolio": {"pnl": {"p5": -50, "p50": 1, "p95": 60}, "loss_probability": 0.3,
                       "var_95": 50, "cvar_95": 60, "max_drawdown_p50": -0.1,
                       "risk_contribution": {"NVDA": 0.6, "GOOG": 0.4}, "currency": "GBP"},
        "evaluation": {"20": {"promoted": False, "winner": "bootstrap_v1"}},
    }), encoding="utf-8")
    return base


@pytest.mark.unit
def test_dashboard_renders_seeded_run(tmp_path, monkeypatch):
    from tradingagents.portfolio import contracts, manifest as manifest_mod

    monkeypatch.setenv("TRADINGAGENTS_RESULTS_DIR", str(tmp_path))
    _seed_run(tmp_path, contracts, manifest_mod)

    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    assert not app.exception
    # The title and tab labels render.
    assert any("Portfolio Intelligence" in t.value for t in app.title)


@pytest.mark.unit
def test_dashboard_handles_empty_state(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_RESULTS_DIR", str(tmp_path))
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()
    assert not app.exception


@pytest.mark.unit
def test_dashboard_imports_with_only_its_own_directory_on_the_path():
    """Regression: the app imported `scripts.run_watchlist` for one constant.

    Streamlit runs the script with dashboard/ as sys.path[0] and never adds the
    repo root, so scripts/ is not importable and the whole page died with
    ModuleNotFoundError. Tests missed it because pytest puts the root on
    sys.path. Import the app in a subprocess with that same restricted path.
    """
    import subprocess
    import sys

    root = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, "-c", "import app; print(app.MAX_CONCURRENCY)"],
        cwd=root / "dashboard",
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": ""},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-800:]
