"""Live holdings drive the analyzed universe (design-review blocker fix)."""

from types import SimpleNamespace

import pytest

from scripts.run_watchlist import DEFAULT_WATCHLIST, derive_watchlist


@pytest.mark.unit
def test_trading212_holdings_drive_universe():
    args = SimpleNamespace(tickers=None, trading212=True)
    holdings = {"NVDA": 100.0, "GOOG": 50.0}
    watchlist = derive_watchlist(args, holdings, dict(DEFAULT_WATCHLIST))
    assert {t for t, _ in watchlist} == {"NVDA", "GOOG"}


@pytest.mark.unit
def test_explicit_tickers_override_holdings():
    args = SimpleNamespace(tickers=["aapl", "msft"], trading212=True)
    watchlist = derive_watchlist(args, {"NVDA": 100.0}, dict(DEFAULT_WATCHLIST))
    assert {t for t, _ in watchlist} == {"AAPL", "MSFT"}


@pytest.mark.unit
def test_default_watchlist_when_no_source():
    args = SimpleNamespace(tickers=None, trading212=False)
    watchlist = derive_watchlist(args, {}, dict(DEFAULT_WATCHLIST))
    assert watchlist == list(DEFAULT_WATCHLIST)


@pytest.mark.unit
def test_depth_profiles_increase_effort_monotonically():
    """Deeper must never mean fewer analysts or fewer debate rounds."""
    from scripts.run_watchlist import DEPTH_PROFILES

    order = ["shallow", "medium", "deep"]
    analysts = [len(DEPTH_PROFILES[d]["analysts"]) for d in order]
    debates = [DEPTH_PROFILES[d]["max_debate_rounds"] for d in order]
    assert analysts == sorted(analysts) and analysts[0] < analysts[-1]
    assert debates == sorted(debates) and debates[0] < debates[-1]
    # Every profile must name only real analyst types.
    valid = {"market", "news", "fundamentals", "social"}
    for d in order:
        assert set(DEPTH_PROFILES[d]["analysts"]) <= valid


@pytest.mark.unit
def test_worker_command_carries_depth_and_provider():
    """Regression: the worker re-parses argv, so omitted flags fall back to
    argparse defaults — which read TRADINGAGENTS_* from .env. A stray
    provider there silently hijacked every ticker while preflight passed."""
    import argparse
    from pathlib import Path

    from scripts.run_watchlist import worker_command

    args = argparse.Namespace(
        aws_profile="p", aws_region="r", quick_model="q", deep_model="d",
        llm_max_retries=1, depth="deep", llm_provider="openai_compatible",
        llm_base_url="https://example.com/serving-endpoints", effort="xhigh",
    )
    cmd = worker_command(
        args, ticker="AAPL", company="Apple", analysis_date="2026-09-02",
        output_dir=Path("/tmp/o"), result_path=Path("/tmp/o/r.json"),
    )
    assert cmd[cmd.index("--depth") + 1] == "deep"
    assert cmd[cmd.index("--llm-provider") + 1] == "openai_compatible"
    assert cmd[cmd.index("--effort") + 1] == "xhigh"
    assert "serving-endpoints" in cmd[cmd.index("--llm-base-url") + 1]


@pytest.mark.unit
def test_worker_command_omits_unset_optional_flags():
    """Bedrock runs must not gain empty --effort/--llm-base-url arguments."""
    import argparse
    from pathlib import Path

    from scripts.run_watchlist import worker_command

    args = argparse.Namespace(
        aws_profile="p", aws_region="r", quick_model="q", deep_model="d",
        llm_max_retries=1, depth="shallow", llm_provider=None,
        llm_base_url=None, effort=None,
    )
    cmd = worker_command(
        args, ticker="AAPL", company="Apple", analysis_date="2026-09-02",
        output_dir=Path("/tmp/o"), result_path=Path("/tmp/o/r.json"),
    )
    assert "--effort" not in cmd and "--llm-base-url" not in cmd
    assert cmd[cmd.index("--llm-provider") + 1] == "bedrock"
