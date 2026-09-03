"""Data-quality classification for Yahoo vs Alpha Vantage reconciliation."""

import pytest

from tradingagents.dataflows import price_crosscheck as pc


@pytest.mark.unit
def test_ok_when_within_tolerance():
    status, diff = pc.classify(100.0, "2026-09-03", 100.5, "2026-09-03")
    assert status == pc.STATUS_OK
    assert diff == pytest.approx(0.005)


@pytest.mark.unit
def test_diverged_beyond_tolerance():
    status, diff = pc.classify(100.0, "2026-09-03", 105.0, "2026-09-03")
    assert status == pc.STATUS_DIVERGED
    assert diff == pytest.approx(0.05)


@pytest.mark.unit
def test_stale_yahoo_date():
    status, _ = pc.classify(100.0, "2026-08-01", 100.0, "2026-09-03")
    assert status == pc.STATUS_STALE


@pytest.mark.unit
def test_yahoo_only_when_av_missing():
    status, diff = pc.classify(100.0, "2026-09-03", None, "2026-09-03")
    assert status == pc.STATUS_YAHOO_ONLY
    assert diff is None


@pytest.mark.unit
def test_unavailable_when_yahoo_missing():
    status, _ = pc.classify(None, None, 100.0, "2026-09-03")
    assert status == pc.STATUS_UNAVAILABLE
