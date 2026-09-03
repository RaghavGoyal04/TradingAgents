"""Run manifest lifecycle and resume-invalidation guards."""

import pytest

from tradingagents.portfolio import manifest as m
from tradingagents.portfolio.contracts import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_RUNNING,
)


def _snapshot():
    return {
        "base_currency": "GBP",
        "positions": [
            {"symbol": "NVDA", "mapping_status": "mapped", "watch_only": False, "value": 600},
            {"symbol": "GOOG", "mapping_status": "mapped", "watch_only": False, "value": 400},
        ],
        "watchlist": [],
    }


def _settings():
    return {"quick_model": "q", "deep_model": "d", "base_currency": "GBP"}


@pytest.mark.unit
def test_new_manifest_has_all_phases_pending(tmp_path):
    man = m.new_manifest(
        tmp_path, analysis_date="2026-09-03", snapshot=_snapshot(), settings=_settings()
    )
    assert man["schema_version"] == "1.0.0"
    assert set(man["phases"]) == {"ingest", "agents", "forecast"}
    assert all(p["status"] == "pending" for p in man["phases"].values())
    assert m.load_manifest(tmp_path)["run_id"] == man["run_id"]


@pytest.mark.unit
def test_phase_and_run_status_transitions(tmp_path):
    man = m.new_manifest(
        tmp_path, analysis_date="2026-09-03", snapshot=_snapshot(), settings=_settings()
    )
    m.set_phase(tmp_path, man, "ingest", STATUS_COMPLETE)
    m.set_phase(tmp_path, man, "agents", STATUS_RUNNING)
    assert man["status"] == STATUS_RUNNING
    m.set_phase(tmp_path, man, "agents", STATUS_COMPLETE)
    m.set_phase(tmp_path, man, "forecast", STATUS_COMPLETE)
    assert man["status"] == STATUS_COMPLETE


@pytest.mark.unit
def test_resume_invalidated_by_portfolio_or_settings_change(tmp_path):
    man = m.new_manifest(
        tmp_path, analysis_date="2026-09-03", snapshot=_snapshot(), settings=_settings()
    )
    assert m.is_resumable(man, snapshot=_snapshot(), settings=_settings())

    changed_portfolio = _snapshot()
    changed_portfolio["positions"].append(
        {"symbol": "MSFT", "mapping_status": "mapped", "watch_only": False, "value": 100}
    )
    assert not m.is_resumable(man, snapshot=changed_portfolio, settings=_settings())

    changed_settings = _settings() | {"deep_model": "other"}
    assert not m.is_resumable(man, snapshot=_snapshot(), settings=changed_settings)


@pytest.mark.unit
def test_portfolio_hash_ignores_position_value_changes(tmp_path):
    snap = _snapshot()
    high = _snapshot()
    high["positions"][0]["value"] = 999999
    assert m.portfolio_hash(snap) == m.portfolio_hash(high)


@pytest.mark.unit
def test_append_event_writes_jsonl(tmp_path):
    m.append_event(tmp_path, {"event": "a"})
    m.append_event(tmp_path, {"event": "b"})
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert '"event": "a"' in lines[0]


@pytest.mark.unit
def test_ticker_status_recorded(tmp_path):
    man = m.new_manifest(
        tmp_path, analysis_date="2026-09-03", snapshot=_snapshot(), settings=_settings()
    )
    m.set_ticker_status(tmp_path, man, "NVDA", "success")
    assert m.load_manifest(tmp_path)["phases"]["agents"]["tickers"]["NVDA"] == "success"


@pytest.mark.unit
def test_stopped_run_with_failed_phase_reports_failed_not_running(tmp_path):
    """A finished run must never be left reporting 'running' forever."""
    man = m.new_manifest(
        tmp_path, analysis_date="2026-09-03", snapshot=_snapshot(), settings=_settings()
    )
    m.set_phase(tmp_path, man, "ingest", STATUS_COMPLETE)
    m.set_phase(tmp_path, man, "agents", STATUS_FAILED)
    m.set_phase(tmp_path, man, "forecast", STATUS_COMPLETE)
    assert man["status"] == STATUS_FAILED


@pytest.mark.unit
def test_partially_complete_run_still_reports_running(tmp_path):
    """An interrupted run with no failure is resumable, so stays 'running'."""
    man = m.new_manifest(
        tmp_path, analysis_date="2026-09-03", snapshot=_snapshot(), settings=_settings()
    )
    m.set_phase(tmp_path, man, "ingest", STATUS_COMPLETE)
    assert man["status"] == STATUS_RUNNING
