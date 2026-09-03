"""End-to-end orchestrator lifecycle: phases, resume, unmapped, locking.

Uses injected agents/forecast phases so the full lifecycle runs with no network
or AWS. The sleep guard is disabled so no ``caffeinate`` is spawned.
"""

import json
import os
from types import SimpleNamespace

import pytest

from tradingagents.portfolio import orchestrator, sleep_guard, t212
from tradingagents.portfolio.contracts import (
    FORECASTS,
    PORTFOLIO_SNAPSHOT,
    RUN_MANIFEST,
    read_json,
)


@pytest.fixture(autouse=True)
def _no_caffeinate(monkeypatch):
    monkeypatch.setattr(sleep_guard, "is_supported", lambda: False)


def _args(run_dir, holdings_path, **overrides):
    base = {
        "run_dir": str(run_dir),
        "trading212": False,
        "holdings": str(holdings_path),
        "watchlist": [],
        "capital": 1200.0,
        "currency": "USD",
        "analysis_date": "2026-09-03",
        "quick_model": "q",
        "deep_model": "d",
        "aws_profile": "prof",
        "aws_region": "us-east-1",
        "resume": False,
        "skip_agents": False,
        "skip_forecast": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def holdings_file(tmp_path):
    path = tmp_path / "holdings.json"
    path.write_text(json.dumps({"NVDA": 600.0, "GOOG": 400.0}), encoding="utf-8")
    return path


def _make_phases():
    calls = {"agents": 0, "forecast": 0}

    def agents_phase(run_dir, manifest, snapshot, args):
        calls["agents"] += 1
        agents_dir = run_dir / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        results = [
            {"ticker": s, "status": "success", "rating": "Hold"}
            for s in t212.analyzed_symbols(snapshot)
        ]
        (agents_dir / "recommendations.json").write_text(
            json.dumps({"results": results}), encoding="utf-8"
        )
        return 0

    def forecast_phase(snapshot, run_dir, *, settings, use_timesfm=False):
        calls["forecast"] += 1
        (run_dir / FORECASTS).write_text(json.dumps({"portfolio": {}}), encoding="utf-8")
        return {}

    return calls, agents_phase, forecast_phase


@pytest.mark.unit
def test_full_run_completes_and_writes_artifacts(tmp_path, holdings_file):
    run_dir = tmp_path / "run"
    calls, agents_phase, forecast_phase = _make_phases()
    code = orchestrator.orchestrate(
        _args(run_dir, holdings_file),
        agents_phase=agents_phase,
        forecast_phase=forecast_phase,
    )
    assert code == 0
    manifest = read_json(run_dir / RUN_MANIFEST)
    assert manifest["status"] == "complete"
    assert all(p["status"] == "complete" for p in manifest["phases"].values())
    assert (run_dir / PORTFOLIO_SNAPSHOT).exists()
    assert (run_dir / FORECASTS).exists()
    assert calls == {"agents": 1, "forecast": 1}
    assert manifest["phases"]["agents"]["tickers"] == {"NVDA": "success", "GOOG": "success"}


@pytest.mark.unit
def test_resume_skips_completed_agents_phase(tmp_path, holdings_file):
    run_dir = tmp_path / "run"
    calls, agents_phase, forecast_phase = _make_phases()
    orchestrator.orchestrate(_args(run_dir, holdings_file), agents_phase=agents_phase, forecast_phase=forecast_phase)
    assert calls["agents"] == 1

    # Resume with identical portfolio + settings: agents must not run again.
    orchestrator.orchestrate(
        _args(run_dir, holdings_file, resume=True),
        agents_phase=agents_phase,
        forecast_phase=forecast_phase,
    )
    assert calls["agents"] == 1  # reused
    assert calls["forecast"] == 2  # forecast always refreshes


@pytest.mark.unit
def test_interrupted_agents_phase_reruns_on_resume(tmp_path, holdings_file):
    run_dir = tmp_path / "run"
    calls = {"n": 0}

    def failing_agents(run_dir, manifest, snapshot, args):
        calls["n"] += 1
        return 1  # simulate interruption / failure

    _, good_agents, forecast_phase = _make_phases()
    orchestrator.orchestrate(_args(run_dir, holdings_file), agents_phase=failing_agents, forecast_phase=forecast_phase)
    manifest = read_json(run_dir / RUN_MANIFEST)
    assert manifest["phases"]["agents"]["status"] == "failed"

    orchestrator.orchestrate(
        _args(run_dir, holdings_file, resume=True),
        agents_phase=good_agents,
        forecast_phase=forecast_phase,
    )
    manifest = read_json(run_dir / RUN_MANIFEST)
    assert manifest["phases"]["agents"]["status"] == "complete"


@pytest.mark.unit
def test_unmapped_nonzero_position_is_skipped_not_fatal(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"

    def fake_snapshot(*, watchlist, session=None):
        return {
            "schema_version": "1.0.0",
            "base_currency": "GBP",
            "account_value": 200.0,
            "cash": 0.0,
            "positions": [
                {"broker_ticker": "NVDA_US_EQ", "symbol": "NVDA", "value": 100.0,
                 "mapping_status": "mapped", "watch_only": False, "weight": 0.5},
                {"broker_ticker": "WEIRD_ZZ_EQ", "symbol": None, "value": 100.0,
                 "mapping_status": "unmapped", "watch_only": False, "weight": 0.5},
            ],
            "unmapped": ["WEIRD_ZZ_EQ"],
            "watchlist": [],
        }

    monkeypatch.setattr(t212, "fetch_portfolio_snapshot", fake_snapshot)
    _, agents_phase, forecast_phase = _make_phases()
    # The unmappable instrument must not abort the run.
    orchestrator.orchestrate(
        _args(run_dir, tmp_path / "unused.json", trading212=True, holdings=None),
        agents_phase=agents_phase,
        forecast_phase=forecast_phase,
    )
    manifest = read_json(run_dir / RUN_MANIFEST)
    assert manifest["phases"]["ingest"]["status"] == "complete"
    assert manifest["status"] == "complete"
    # Only the mapped position was analyzed; the unmapped one is skipped.
    assert manifest["phases"]["agents"]["tickers"] == {"NVDA": "success"}
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert "unmapped_skipped" in events


@pytest.mark.unit
def test_exclusive_lock_rejects_active_owner(tmp_path, holdings_file):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / ".orchestrator.lock").write_text(str(os.getpid()), encoding="utf-8")
    _, agents_phase, forecast_phase = _make_phases()
    with pytest.raises(RuntimeError, match="another portfolio run"):
        orchestrator.orchestrate(_args(run_dir, holdings_file), agents_phase=agents_phase, forecast_phase=forecast_phase)


@pytest.mark.unit
def test_skip_agents_only_runs_forecast(tmp_path, holdings_file):
    run_dir = tmp_path / "run"
    calls, agents_phase, forecast_phase = _make_phases()
    orchestrator.orchestrate(
        _args(run_dir, holdings_file, skip_agents=True),
        agents_phase=agents_phase,
        forecast_phase=forecast_phase,
    )
    assert calls["agents"] == 0
    manifest = read_json(run_dir / RUN_MANIFEST)
    assert manifest["phases"]["agents"]["status"] == "complete"
    assert manifest["phases"]["forecast"]["status"] == "complete"


@pytest.mark.unit
def test_one_dead_ticker_does_not_fail_the_agents_phase(tmp_path, holdings_file):
    """37 good tickers + 1 delisted must still complete (real IPXX scenario)."""
    run_dir = tmp_path / "run"

    def partial_agents(run_dir, manifest, snapshot, args):
        agents_dir = run_dir / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        (agents_dir / "recommendations.json").write_text(
            json.dumps({"results": [
                {"ticker": "NVDA", "status": "success", "action": "BUY"},
                {"ticker": "GOOG", "status": "error",
                 "error": "NoMarketDataError: no rows"},
            ]}),
            encoding="utf-8",
        )
        return 1  # non-zero because one ticker errored

    _, _, forecast_phase = _make_phases()
    orchestrator.orchestrate(
        _args(run_dir, holdings_file),
        agents_phase=partial_agents,
        forecast_phase=forecast_phase,
    )
    manifest = read_json(run_dir / RUN_MANIFEST)
    assert manifest["phases"]["agents"]["status"] == "complete"
    assert manifest["status"] == "complete"
    # The individual failure is still visible, not swallowed.
    assert manifest["phases"]["agents"]["tickers"]["GOOG"] == "error"


@pytest.mark.unit
def test_agents_phase_fails_when_no_ticker_succeeds(tmp_path, holdings_file):
    """Total failure (e.g. expired credentials) must still be reported."""
    run_dir = tmp_path / "run"

    def all_failed(run_dir, manifest, snapshot, args):
        agents_dir = run_dir / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        (agents_dir / "recommendations.json").write_text(
            json.dumps({"results": [
                {"ticker": "NVDA", "status": "error", "error": "ExpiredToken"},
                {"ticker": "GOOG", "status": "error", "error": "ExpiredToken"},
            ]}),
            encoding="utf-8",
        )
        return 1

    _, _, forecast_phase = _make_phases()
    orchestrator.orchestrate(
        _args(run_dir, holdings_file),
        agents_phase=all_failed,
        forecast_phase=forecast_phase,
    )
    manifest = read_json(run_dir / RUN_MANIFEST)
    assert manifest["phases"]["agents"]["status"] == "failed"
    assert manifest["status"] == "failed"


@pytest.mark.unit
def test_crashed_agents_phase_does_not_inherit_a_previous_runs_successes(tmp_path):
    """Regression: a phase that died on startup reported itself complete.

    The tally was read from recommendations.json without checking when it was
    written, so 37 successes from an earlier run made a run that never analysed
    anything look finished.
    """
    import time as _time

    from tradingagents.portfolio import orchestrator

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    (agents_dir / "recommendations.json").write_text(
        json.dumps({"results": [{"ticker": "AAPL", "status": "success"}]}),
        encoding="utf-8",
    )
    manifest = {"tickers": {}, "phases": {"agents": {"tickers": {}}}}
    symbols = ["AAPL"]

    # Stale file: written before the phase began.
    stale = orchestrator._sync_ticker_statuses(
        tmp_path, manifest, agents_dir, symbols, since=_time.time() + 1
    )
    assert stale.get("success", 0) == 0, "stale results must not count as successes"

    # Fresh file: written after the phase began.
    fresh = orchestrator._sync_ticker_statuses(
        tmp_path, manifest, agents_dir, symbols, since=0
    )
    assert fresh.get("success") == 1


@pytest.mark.unit
def test_ui_concurrency_ceiling_matches_the_runner_validation():
    """The slider offered 38 while the runner rejected anything over 8, so a
    valid-looking choice killed the agents phase the moment it launched."""
    import dashboard.app as app
    from tradingagents.portfolio.launcher import MAX_CONCURRENCY

    assert app.MAX_CONCURRENCY == MAX_CONCURRENCY
