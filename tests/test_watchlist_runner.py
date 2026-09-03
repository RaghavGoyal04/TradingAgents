import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_watchlist as watchlist_runner
from scripts.run_watchlist import (
    DEFAULT_WATCHLIST,
    exclusive_run_lock,
    extract_field,
    is_authentication_error,
    is_retryable_error,
    load_trading212_portfolio,
    next_weekday,
    run_batch,
    run_worker_once,
    size_recommendations,
    trading212_to_yahoo_symbol,
    validate_args,
)


def _result(ticker: str, rating: str) -> dict:
    return {
        "ticker": ticker,
        "company": ticker,
        "status": "success",
        "rating": rating,
    }


def test_default_watchlist_contains_requested_symbols():
    assert len(DEFAULT_WATCHLIST) == 21
    assert [ticker for ticker, _ in DEFAULT_WATCHLIST][:4] == [
        "NVDA",
        "GOOG",
        "PLTR",
        "RR.L",
    ]
    assert DEFAULT_WATCHLIST[-1] == ("ASML", "ASML")


def test_extract_field_reads_rendered_portfolio_decision():
    decision = (
        "**Rating**: Buy\n\n"
        "**Executive Summary**: Build a 5% position tomorrow.\n\n"
        "**Investment Thesis**: Momentum is positive.\n\n"
        "**Time Horizon**: 3-6 months"
    )

    assert extract_field(decision, "Executive Summary") == "Build a 5% position tomorrow."
    assert extract_field(decision, "Time Horizon") == "3-6 months"


def test_next_weekday_skips_weekend():
    assert next_weekday("2026-09-02") == "2026-09-03"
    assert next_weekday("2026-09-04") == "2026-09-07"


@pytest.mark.parametrize(
    ("broker_ticker", "yahoo_ticker"),
    [
        ("NVDA_US_EQ", "NVDA"),
        ("BRK.B_US_EQ", "BRK-B"),
        ("RRl_EQ", "RR.L"),
        ("SIEd_EQ", "SIE.DE"),
        ("SUp_EQ", "SU.PA"),
    ],
)
def test_trading212_to_yahoo_symbol(broker_ticker, yahoo_ticker):
    assert trading212_to_yahoo_symbol(broker_ticker) == yahoo_ticker


def test_load_trading212_portfolio_fetches_live_values(monkeypatch):
    monkeypatch.setenv("TRADING212_API_KEY", "test-key")
    monkeypatch.setenv("TRADING212_API_SECRET", "test-secret")
    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, *, auth, timeout):
        calls.append((url, auth, timeout))
        if url.endswith("/account/summary"):
            return Response({"totalValue": 12_500, "currency": "GBP"})
        return Response(
            [
                {
                    "instrument": {"ticker": "NVDA_US_EQ"},
                    "walletImpact": {"currentValue": 1_250},
                },
                {
                    "instrument": {"ticker": "RRl_EQ"},
                    "walletImpact": {"currentValue": 750},
                },
            ]
        )

    monkeypatch.setattr(watchlist_runner.requests, "get", fake_get)

    holdings, capital, currency = load_trading212_portfolio()

    assert holdings == {"NVDA": 1_250, "RR.L": 750}
    assert capital == 12_500
    assert currency == "GBP"
    assert len(calls) == 2
    assert all(call[1] == ("test-key", "test-secret") for call in calls)


def test_size_recommendations_maps_ratings_and_holdings():
    results = [
        _result("BUY", "Buy"),
        _result("OVER", "Overweight"),
        _result("HOLD", "Hold"),
        _result("UNDER", "Underweight"),
        _result("SELL", "Sell"),
        _result("FULL", "Overweight"),
    ]

    sized = size_recommendations(
        results,
        capital=100_000,
        max_position_pct=5,
        holdings={"BUY": 1_000, "UNDER": 4_000, "SELL": 3_000, "FULL": 6_000},
        currency="USD",
    )
    by_ticker = {item["ticker"]: item for item in sized}

    assert by_ticker["BUY"]["action"] == "BUY"
    assert by_ticker["BUY"]["trade_value"] == 4_000
    assert by_ticker["OVER"]["target_allocation_pct"] == 5
    assert by_ticker["HOLD"]["action"] == "HOLD"
    assert by_ticker["UNDER"]["trade_value"] == 2_000
    assert by_ticker["SELL"]["trade_value"] == 3_000
    assert by_ticker["FULL"]["action"] == "HOLD"
    assert by_ticker["FULL"]["current_allocation_pct"] == 6


def test_retry_classifier_only_retries_transient_failures():
    assert is_retryable_error("ReadTimeoutError: request timed out")
    assert is_retryable_error("ThrottlingException: rate exceeded")
    assert not is_retryable_error("AccessDeniedException: not authorized")
    assert not is_retryable_error("ValidationException: invalid model")


def test_authentication_classifier_detects_expired_sso():
    assert is_authentication_error("UnauthorizedSSOTokenError: session expired")
    assert not is_authentication_error("ReadTimeoutError: request timed out")


def test_run_lock_rejects_an_active_owner(tmp_path):
    (tmp_path / ".run.lock").write_text(str(os.getpid()), encoding="utf-8")

    with (
        pytest.raises(RuntimeError, match="another watchlist run"),
        exclusive_run_lock(tmp_path),
    ):
        pass


def test_validate_args_rejects_unsafe_concurrency():
    # Pinned to the declared ceiling rather than a literal, so raising the cap
    # cannot leave the runner and its guard rail disagreeing.
    from scripts.run_watchlist import MAX_CONCURRENCY

    args = SimpleNamespace(
        capital=None,
        max_position_pct=5,
        concurrency=MAX_CONCURRENCY + 1,
        ticker_timeout=1200,
        ticker_retries=2,
        llm_max_retries=2,
        retry_base_delay=5,
    )

    with pytest.raises(ValueError, match="concurrency"):
        validate_args(args)


def test_run_batch_honors_concurrency_and_returns_watchlist_order(monkeypatch, tmp_path):
    active = 0
    max_active = 0

    async def fake_run(_args, _semaphore, *, ticker, company, **_kwargs):
        nonlocal active, max_active
        async with _semaphore:
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
        return _result(ticker, "Hold") | {"company": company}

    monkeypatch.setattr(watchlist_runner, "run_ticker_with_retries", fake_run)
    monkeypatch.setattr(watchlist_runner, "load_previous_results", lambda *_args: {})
    monkeypatch.setattr(watchlist_runner, "write_outputs", lambda *_args: None)
    args = SimpleNamespace(
        concurrency=2,
        quick_model="quick",
        deep_model="deep",
        capital=None,
        currency="USD",
        max_position_pct=5,
    )
    watchlist = [("AAA", "A"), ("BBB", "B"), ("CCC", "C")]

    results = asyncio.run(run_batch(args, watchlist, "2026-09-02", tmp_path, {}))

    assert max_active == 2
    assert [result["ticker"] for result in results] == ["AAA", "BBB", "CCC"]


def test_worker_timeout_terminates_process_and_is_retryable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        watchlist_runner,
        "worker_command",
        lambda *_args, **_kwargs: [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
    )
    args = SimpleNamespace(ticker_timeout=0.05)

    result, retryable = asyncio.run(
        run_worker_once(
            args,
            ticker="AAA",
            company="A",
            analysis_date="2026-09-02",
            output_dir=tmp_path,
            attempt=1,
        )
    )

    assert result["status"] == "error"
    assert result["error"].startswith("TickerTimeout:")
    assert retryable


def test_reuse_is_invalidated_by_every_setting_that_changes_the_answer(tmp_path):
    """Regression: only the model names invalidated reuse.

    Re-running with a different depth, effort or provider replayed the previous
    run's results in a few seconds, so changed settings appeared to have been
    applied when nothing had been re-analysed.
    """
    import json as _json

    from scripts.run_watchlist import load_previous_results

    settings = {
        "quick_model": "q",
        "deep_model": "d",
        "research_depth": "shallow",
        "llm_provider": "openai_compatible",
        "effort": "high",
    }
    (tmp_path / "recommendations.json").write_text(
        _json.dumps({
            "analysis_date": "2026-09-03",
            "settings": settings,
            "results": [{"ticker": "AAPL", "status": "success"}],
        }),
        encoding="utf-8",
    )

    def _args(**over):
        base = {
            "force": False,
            "quick_model": "q",
            "deep_model": "d",
            "depth": "shallow",
            "llm_provider": "openai_compatible",
            "effort": "high",
        }
        return SimpleNamespace(**{**base, **over})

    unchanged = load_previous_results(tmp_path, "2026-09-03", _args())
    assert "AAPL" in unchanged, "identical settings should still reuse"

    for changed in (
        {"depth": "deep"},
        {"effort": "low"},
        {"llm_provider": "bedrock"},
        {"deep_model": "other"},
        {"force": True},
    ):
        assert load_previous_results(tmp_path, "2026-09-03", _args(**changed)) == {}, (
            f"{changed} must force a fresh analysis"
        )


def test_rate_limit_errors_are_told_apart_from_ordinary_transient_faults():
    from scripts.run_watchlist import is_rate_limit_error, is_retryable_error

    quota = (
        "OpenAIRateLimitError: Error code: 429 - {'error_code': "
        "'REQUEST_LIMIT_EXCEEDED', 'message': 'Exceeded workspace output tokens "
        "per minute rate limit for databricks-claude-sonnet-4-6.'}"
    )
    assert is_rate_limit_error(quota)
    assert is_retryable_error(quota)

    blip = "ConnectionError: connection reset by peer"
    assert is_retryable_error(blip), "still retryable"
    assert not is_rate_limit_error(blip), "must not get the 60s quota wait"


def test_quota_waits_do_not_spend_the_retry_budget(monkeypatch):
    """Regression: 17 of 38 holdings died to 429s.

    Every attempt was spent inside the same exhausted minute because a quota
    rejection was backed off like a network blip. A holding must be able to
    wait for the window and still have its retries left for real failures.
    """
    import asyncio

    from scripts import run_watchlist as rw

    attempts: list[int] = []
    slept: list[float] = []
    quota_error = {
        "status": "error",
        "error": "Error code: 429 - REQUEST_LIMIT_EXCEEDED: rate limit",
    }

    async def fake_worker(args, *, ticker, company, analysis_date, output_dir, attempt):
        attempts.append(attempt)
        # Rate limited twice, then the window refills and the work succeeds.
        if attempt <= 2:
            return dict(quota_error), True
        return {"status": "success", "ticker": ticker}, False

    monkeypatch.setattr(rw, "run_worker_once", fake_worker)

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(rw.asyncio, "sleep", fake_sleep)

    args = SimpleNamespace(ticker_retries=2, retry_base_delay=5.0)
    result = asyncio.run(
        rw.run_ticker_with_retries(
            args,
            asyncio.Semaphore(1),
            ticker="NVDA",
            company="Nvidia",
            analysis_date="2026-09-03",
            output_dir=Path("/tmp"),
        )
    )

    assert result["status"] == "success"
    assert attempts == [1, 2, 3]
    assert len(slept) == 2
    assert all(s >= rw.QUOTA_WINDOW_SECONDS for s in slept), (
        f"quota waits must span the refill window, got {slept}"
    )


def test_quota_waits_are_bounded(monkeypatch):
    """Waiting must not become an infinite loop when the quota never returns."""
    import asyncio

    from scripts import run_watchlist as rw

    async def always_limited(args, *, ticker, company, analysis_date, output_dir, attempt):
        return {"status": "error", "error": "429 rate limit"}, True

    async def no_wait(seconds):
        return None

    monkeypatch.setattr(rw, "run_worker_once", always_limited)
    monkeypatch.setattr(rw.asyncio, "sleep", no_wait)

    args = SimpleNamespace(ticker_retries=2, retry_base_delay=5.0)
    result = asyncio.run(
        rw.run_ticker_with_retries(
            args,
            asyncio.Semaphore(1),
            ticker="NVDA",
            company="Nvidia",
            analysis_date="2026-09-03",
            output_dir=Path("/tmp"),
        )
    )
    assert result["status"] == "error"
    assert result["attempts"] <= rw.MAX_QUOTA_WAITS + args.ticker_retries + 1
