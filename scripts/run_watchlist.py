"""Run TradingAgents over a stock watchlist and consolidate the decisions.

The default run uses today's date, English, only the Market Analyst, shallow
research, and the Bedrock models configured below. Tickers run in bounded,
killable subprocesses so timeouts and failures remain isolated.

Examples:
    uv run --extra bedrock python scripts/run_watchlist.py --smoke-test
    uv run --extra bedrock python scripts/run_watchlist.py --capital 100000 --concurrency 3
    uv run --extra bedrock python scripts/run_watchlist.py --holdings holdings.json
    uv run --extra bedrock python scripts/run_watchlist.py --trading212

``holdings.json`` contains current position values in the same currency as
``--capital``, for example: ``{"NVDA": 12000, "GOOG": 8000}``.
``--trading212`` reads the live account summary and positions using the
``TRADING212_API_KEY`` and ``TRADING212_API_SECRET`` environment variables.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import random
import re
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

# Canonical Trading212 symbol mapping lives in the portfolio package so the CLI
# and the dashboard share one implementation (avoids the duplicate mapping the
# design review flagged).
from tradingagents.portfolio.launcher import MAX_CONCURRENCY
from tradingagents.portfolio.t212 import trading212_to_yahoo_symbol

DEFAULT_WATCHLIST = [
    ("NVDA", "Nvidia"),
    ("GOOG", "Alphabet (Class C)"),
    ("PLTR", "Palantir"),
    ("RR.L", "Rolls-Royce"),
    ("META", "Meta Platforms"),
    ("PANW", "Palo Alto Networks"),
    ("AMD", "Advanced Micro Devices"),
    ("TSM", "Taiwan Semiconductor Manufacturing"),
    ("AVGO", "Broadcom"),
    ("SIE.DE", "Siemens"),
    ("CEG", "Constellation Energy"),
    ("CRWD", "Crowdstrike"),
    ("MU", "Micron Technology"),
    ("BA.L", "BAE Systems"),
    ("RHM.DE", "Rheinmetall"),
    ("MSFT", "Microsoft"),
    ("AMZN", "Amazon"),
    ("TSLA", "Tesla"),
    ("005930.KS", "Samsung Electronics"),
    ("SU.PA", "Schneider Electric"),
    ("ASML", "ASML"),
]

DEFAULT_PROFILE = "wbd-boltcloud-data-processing-dev.GtDcpDca"
# DEFAULT_QUICK_MODEL = "openai.gpt-oss-20b-1:0"
# DEFAULT_DEEP_MODEL = "openai.gpt-oss-120b-1:0"
DEFAULT_QUICK_MODEL = "us.anthropic.claude-sonnet-4-6"
DEFAULT_DEEP_MODEL = "us.anthropic.claude-opus-4-7"

# Effort levels. Each adds analysts (more evidence per ticker) and debate rounds
# (more challenge to the conclusion), so cost and runtime rise with depth.
DEPTH_PROFILES = {
    "shallow": {
        "analysts": ("market",),
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
    },
    "medium": {
        "analysts": ("market", "news"),
        "max_debate_rounds": 2,
        "max_risk_discuss_rounds": 1,
    },
    "deep": {
        "analysts": ("market", "news", "fundamentals", "social"),
        "max_debate_rounds": 3,
        "max_risk_discuss_rounds": 2,
    },
}
DEFAULT_DEPTH = "shallow"
DEFAULT_CONCURRENCY = 6
DEFAULT_TICKER_TIMEOUT = 20 * 60
DEFAULT_TICKER_RETRIES = 2
# Databricks meters tokens per minute across the workspace, so a rejection
# clears on the next window rather than after a few seconds.
QUOTA_WINDOW_SECONDS = 60.0
MAX_QUOTA_WAITS = 5
DEFAULT_LLM_RETRIES = 2
TRADING212_BASE_URL = "https://live.trading212.com/api/v0"
TRADING212_TIMEOUT = 20


class AuthenticationExpiredError(RuntimeError):
    """Raised when a worker detects credentials that require user login."""


def extract_field(markdown: str, label: str) -> str | None:
    """Extract one ``**Label**: value`` section from a rendered decision."""
    match = re.search(
        rf"\*\*{re.escape(label)}\*\*:\s*(.*?)(?=\n\s*\n\*\*[^*]+\*\*:|\Z)",
        markdown,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def next_weekday(date_string: str) -> str:
    """Return the next weekday; exchange-specific holidays are not checked."""
    day = datetime.strptime(date_string, "%Y-%m-%d").date() + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day.isoformat()


def derive_watchlist(
    args: argparse.Namespace,
    holdings: dict[str, float],
    names: dict[str, str],
) -> list[tuple[str, str]]:
    """Choose the analyzed universe (deduplicated ``(ticker, company)`` pairs).

    Precedence: an explicit ``--tickers`` wins; otherwise live Trading 212
    holdings drive the universe (analyze what you own, the design-review
    blocker); otherwise the default watchlist. Company names fall back to the
    ticker itself.
    """
    if args.tickers:
        requested = [
            (ticker.upper(), names.get(ticker.upper(), ticker.upper()))
            for ticker in args.tickers
        ]
    elif getattr(args, "trading212", False):
        requested = [(ticker, names.get(ticker, ticker)) for ticker in sorted(holdings)]
    else:
        requested = list(DEFAULT_WATCHLIST)
    return list(dict.fromkeys(requested))


def load_holdings(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("holdings must be a JSON object mapping ticker to position value")
    return {str(ticker).upper(): float(value) for ticker, value in values.items()}


def load_trading212_portfolio() -> tuple[dict[str, float], float, str]:
    """Return live holdings, total account value, and primary currency."""
    if not os.getenv("TRADING212_API_KEY") or not os.getenv("TRADING212_API_SECRET"):
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    api_key = os.getenv("TRADING212_API_KEY")
    api_secret = os.getenv("TRADING212_API_SECRET")
    if not api_key or not api_secret:
        raise ValueError(
            "TRADING212_API_KEY and TRADING212_API_SECRET must be set for --trading212"
        )

    auth = (api_key, api_secret)
    summary_response = requests.get(
        f"{TRADING212_BASE_URL}/equity/account/summary",
        auth=auth,
        timeout=TRADING212_TIMEOUT,
    )
    summary_response.raise_for_status()
    positions_response = requests.get(
        f"{TRADING212_BASE_URL}/equity/positions",
        auth=auth,
        timeout=TRADING212_TIMEOUT,
    )
    positions_response.raise_for_status()

    summary = summary_response.json()
    holdings: dict[str, float] = {}
    skipped = []
    for position in positions_response.json():
        broker_ticker = position["instrument"]["ticker"]
        symbol = trading212_to_yahoo_symbol(broker_ticker)
        if symbol is None:
            skipped.append(broker_ticker)
            continue
        current_value = float(position["walletImpact"]["currentValue"])
        holdings[symbol] = holdings.get(symbol, 0.0) + current_value

    if skipped:
        print(
            "Trading 212 positions skipped because their ticker format is unsupported: "
            + ", ".join(skipped),
            file=sys.stderr,
        )
    return holdings, float(summary["totalValue"]), str(summary["currency"])


def size_recommendations(
    results: list[dict],
    *,
    capital: float | None,
    max_position_pct: float,
    holdings: dict[str, float],
    currency: str,
) -> list[dict]:
    """Map five-tier ratings to conservative, deterministic trade sizes."""
    successful = [item for item in results if item["status"] == "success"]
    raw_targets = {
        item["ticker"]: (
            max_position_pct
            if item["rating"] in {"Buy", "Overweight"}
            else 0.0
        )
        for item in successful
    }
    scale = min(1.0, 100.0 / sum(raw_targets.values())) if sum(raw_targets.values()) else 1.0

    sized = []
    for item in results:
        item = dict(item)
        if item["status"] != "success":
            sized.append(item)
            continue

        rating = item["rating"]
        current_value = holdings.get(item["ticker"])
        current_pct = (
            current_value / capital * 100
            if current_value is not None and capital is not None
            else None
        )
        if rating in {"Buy", "Overweight"}:
            target_pct = raw_targets[item["ticker"]] * scale
            target_value = capital * target_pct / 100 if capital is not None else None
            trade_value = (
                max(target_value - (current_value or 0.0), 0.0)
                if target_value is not None
                else None
            )
            should_buy = trade_value is None or trade_value > 0.005
            item.update(
                action="BUY" if should_buy else "HOLD",
                current_allocation_pct=(
                    round(current_pct, 2) if current_pct is not None else None
                ),
                target_allocation_pct=round(target_pct, 2),
                trade_value=round(trade_value, 2) if trade_value is not None else None,
                sizing=(
                    f"Buy {currency} {trade_value:,.2f}; target {target_pct:.2f}%"
                    if trade_value is not None and should_buy
                    else (
                        f"No buy; current {current_pct:.2f}% already meets or exceeds "
                        f"the {target_pct:.2f}% target"
                    )
                    if current_pct is not None
                    else f"Build to {target_pct:.2f}% of the portfolio"
                ),
            )
        elif rating == "Underweight":
            trade_value = current_value * 0.5 if current_value is not None else None
            item.update(
                action="SELL",
                current_allocation_pct=(
                    round(current_pct, 2) if current_pct is not None else None
                ),
                target_allocation_pct=None,
                trade_value=round(trade_value, 2) if trade_value is not None else None,
                sizing=(
                    f"Sell {currency} {trade_value:,.2f} (50% of current position)"
                    if trade_value is not None
                    else "Sell 50% of the current position"
                ),
            )
        elif rating == "Sell":
            item.update(
                action="SELL",
                current_allocation_pct=(
                    round(current_pct, 2) if current_pct is not None else None
                ),
                target_allocation_pct=0.0,
                trade_value=round(current_value, 2) if current_value is not None else None,
                sizing=(
                    f"Sell {currency} {current_value:,.2f} (entire current position)"
                    if current_value is not None
                    else "Sell 100% of the current position"
                ),
            )
        else:
            item.update(
                action="HOLD",
                current_allocation_pct=(
                    round(current_pct, 2) if current_pct is not None else None
                ),
                target_allocation_pct=None,
                trade_value=0.0,
                sizing=(
                    f"No trade; retain {currency} {current_value:,.2f} "
                    f"({current_pct:.2f}% of portfolio)"
                    if current_value is not None and current_pct is not None
                    else "No trade; retain the current position"
                ),
            )
        sized.append(item)
    return sized


def render_summary(payload: dict) -> str:
    capital = payload["settings"]["capital"]
    sizing_basis = (
        f"{payload['settings']['currency']} {capital:,.2f} portfolio"
        if capital is not None
        else "portfolio percentages; pass --capital for monetary amounts"
    )
    lines = [
        f"# Watchlist recommendations for {payload['planned_execution_date']}",
        "",
        f"- Analysis date: {payload['analysis_date']}",
        f"- Sizing basis: {sizing_basis}",
        "- Configuration: English, Market Analyst only, shallow research",
        f"- Models: {payload['settings']['quick_model']} / {payload['settings']['deep_model']}",
        (
            "- Safeguards: "
            f"{payload['settings']['concurrency']} parallel workers, "
            f"{payload['settings']['ticker_timeout_seconds']}s ticker deadline, "
            f"{payload['settings']['ticker_retries']} ticker retries"
        ),
        "- Important: model-generated research, not personalized financial advice",
    ]

    for action in ("BUY", "HOLD", "SELL"):
        lines.extend(["", f"## {action}"])
        matches = [item for item in payload["results"] if item.get("action") == action]
        if not matches:
            lines.append("None.")
            continue
        for item in matches:
            lines.extend(
                [
                    "",
                    f"### {item['ticker']} — {item['company']}",
                    f"- Source rating: {item['rating']}",
                    f"- Size: {item['sizing']}",
                    f"- Horizon: {item.get('time_horizon') or 'Not specified'}",
                    f"- Rationale: {item.get('executive_summary') or 'See the full report.'}",
                    f"- Full report: `{item['report_path']}`",
                ]
            )

    failures = [item for item in payload["results"] if item["status"] == "error"]
    if failures:
        lines.extend(["", "## Failed analyses"])
        for item in failures:
            log_suffix = f" — log: `{item['log_path']}`" if item.get("log_path") else ""
            lines.append(f"- {item['ticker']}: {item['error']}{log_suffix}")

    lines.extend(
        [
            "",
            "Execution date is the next weekday only; verify each exchange is open and review "
            "the full reports before placing orders.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "recommendations.json"
    json_temp = output_dir / "recommendations.json.tmp"
    json_temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    json_temp.replace(json_path)

    markdown_path = output_dir / "recommendations.md"
    markdown_temp = output_dir / "recommendations.md.tmp"
    markdown_temp.write_text(render_summary(payload), encoding="utf-8")
    markdown_temp.replace(markdown_path)


def build_payload(
    analysis_date: str,
    args: argparse.Namespace,
    results: list[dict],
    holdings: dict[str, float],
) -> dict:
    return {
        "analysis_date": analysis_date,
        "planned_execution_date": next_weekday(analysis_date),
        "settings": {
            "analysts": list(
                DEPTH_PROFILES[getattr(args, "depth", DEFAULT_DEPTH)]["analysts"]
            ),
            "research_depth": getattr(args, "depth", DEFAULT_DEPTH),
            "language": "English",
            "quick_model": args.quick_model,
            "deep_model": args.deep_model,
            "llm_provider": getattr(args, "llm_provider", None) or "bedrock",
            "effort": getattr(args, "effort", None),
            "capital": args.capital,
            "currency": args.currency,
            "max_position_pct": args.max_position_pct,
            "concurrency": getattr(args, "concurrency", DEFAULT_CONCURRENCY),
            "ticker_timeout_seconds": getattr(
                args, "ticker_timeout", DEFAULT_TICKER_TIMEOUT
            ),
            "ticker_retries": getattr(args, "ticker_retries", DEFAULT_TICKER_RETRIES),
            "llm_max_retries": getattr(args, "llm_max_retries", DEFAULT_LLM_RETRIES),
        },
        "results": size_recommendations(
            results,
            capital=args.capital,
            max_position_pct=args.max_position_pct,
            holdings=holdings,
            currency=args.currency,
        ),
    }


def safe_component(value: str) -> str:
    """Return a filesystem-safe component without importing TradingAgents."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)[:32]


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextlib.contextmanager
def exclusive_run_lock(output_dir: Path):
    """Prevent two orchestrators from corrupting the same daily output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".run.lock"
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as lock:
                lock.write(str(os.getpid()))
            break
        except FileExistsError:
            try:
                owner = int(lock_path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                owner = -1
            if owner > 0 and is_process_running(owner):
                raise RuntimeError(
                    f"another watchlist run (PID {owner}) is using {output_dir}"
                ) from None
            lock_path.unlink(missing_ok=True)
    try:
        yield
    finally:
        try:
            if int(lock_path.read_text(encoding="utf-8").strip()) == os.getpid():
                lock_path.unlink(missing_ok=True)
        except (FileNotFoundError, OSError, ValueError):
            pass


def configure_aws_environment(args: argparse.Namespace, *, block_dotenv: bool) -> None:
    if block_dotenv:
        # TradingAgents loads .env at import; an existing empty value prevents
        # python-dotenv from introducing a stale bearer token.
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = ""
    else:
        os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
    os.environ["AWS_PROFILE"] = args.aws_profile
    os.environ["AWS_REGION"] = args.aws_region
    os.environ["AWS_DEFAULT_REGION"] = args.aws_region


def preflight(args: argparse.Namespace) -> None:
    """Fail once, before fan-out, on bad credentials or unavailable models.

    Dispatches on the provider: the Bedrock check speaks the Converse API and
    would reject a Databricks endpoint name as an invalid model identifier.
    """
    provider = getattr(args, "llm_provider", None) or "bedrock"
    if provider == "bedrock":
        preflight_aws(args)
    else:
        preflight_provider(args, provider)


def preflight_provider(args: argparse.Namespace, provider: str) -> None:
    """Check each distinct model answers on a non-Bedrock provider."""
    from tradingagents.llm_clients.factory import create_llm_client

    if args.skip_model_preflight:
        return
    for model in dict.fromkeys((args.quick_model, args.deep_model)):
        kwargs = {"max_tokens": 16}
        if getattr(args, "effort", None):
            kwargs["reasoning_effort"] = args.effort
        client = create_llm_client(
            provider, model, base_url=getattr(args, "llm_base_url", None), **kwargs
        )
        # The preflight is the first call of the run, so it lands on whatever
        # quota the previous run left behind. Failing here aborted everything
        # before a single holding was analysed.
        for wait in range(MAX_QUOTA_WAITS + 1):
            try:
                client.get_llm().invoke("Reply OK.")
                break
            except Exception as exc:
                if wait == MAX_QUOTA_WAITS or not is_rate_limit_error(str(exc)):
                    raise
                print(
                    f"Model preflight: {model} rate limited; waiting "
                    f"{QUOTA_WINDOW_SECONDS:.0f}s for the quota window",
                    flush=True,
                )
                time.sleep(QUOTA_WINDOW_SECONDS)
        print(f"Model preflight: {model} available on {provider}", flush=True)


def preflight_aws(args: argparse.Namespace) -> None:
    """Fail once, before fan-out, on expired SSO or unavailable models."""
    configure_aws_environment(args, block_dotenv=False)
    import boto3
    from botocore.config import Config

    session = boto3.Session(profile_name=args.aws_profile, region_name=args.aws_region)
    identity = session.client("sts").get_caller_identity()
    print(f"AWS preflight: account {identity['Account']}", flush=True)
    if args.skip_model_preflight:
        return

    runtime = session.client(
        "bedrock-runtime",
        config=Config(
            signature_version="v4",
            read_timeout=60,
            connect_timeout=10,
            retries={"max_attempts": 1},
        ),
    )
    for model in dict.fromkeys((args.quick_model, args.deep_model)):
        runtime.converse(
            modelId=model,
            messages=[{"role": "user", "content": [{"text": "Reply OK."}]}],
            inferenceConfig={"maxTokens": 32},
        )
        print(f"Model preflight: {model} available", flush=True)


def worker_result_path(output_dir: Path, ticker: str) -> Path:
    return output_dir / "workers" / f"{safe_component(ticker)}.json"


def worker_command(
    args: argparse.Namespace,
    *,
    ticker: str,
    company: str,
    analysis_date: str,
    output_dir: Path,
    result_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-ticker",
        ticker,
        "--worker-company",
        company,
        "--worker-result",
        str(result_path),
        "--analysis-date",
        analysis_date,
        "--output-dir",
        str(output_dir),
        "--aws-profile",
        args.aws_profile,
        "--aws-region",
        args.aws_region,
        "--quick-model",
        args.quick_model,
        "--deep-model",
        args.deep_model,
        "--llm-max-retries",
        str(args.llm_max_retries),
        # Depth and provider must be forwarded explicitly: the worker re-parses
        # argv from scratch, so anything omitted here silently falls back to the
        # argparse default (or to TRADINGAGENTS_* in .env) instead of the
        # settings the run was launched with.
        "--depth",
        getattr(args, "depth", DEFAULT_DEPTH),
        "--llm-provider",
        getattr(args, "llm_provider", None) or "bedrock",
    ] + (
        ["--llm-base-url", args.llm_base_url]
        if getattr(args, "llm_base_url", None)
        else []
    ) + (
        ["--effort", args.effort] if getattr(args, "effort", None) else []
    )


async def terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except TimeoutError:
        process.kill()
        await process.wait()


async def run_worker_once(
    args: argparse.Namespace,
    *,
    ticker: str,
    company: str,
    analysis_date: str,
    output_dir: Path,
    attempt: int,
) -> tuple[dict, bool]:
    """Run one killable ticker process and return (result, retryable)."""
    started = time.monotonic()
    result_path = worker_result_path(output_dir, ticker)
    result_path.unlink(missing_ok=True)
    log_path = output_dir / "logs" / f"{safe_component(ticker)}.attempt-{attempt}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = worker_command(
        args,
        ticker=ticker,
        company=company,
        analysis_date=analysis_date,
        output_dir=output_dir,
        result_path=result_path,
    )
    environment = os.environ.copy()
    environment.pop("AWS_BEARER_TOKEN_BEDROCK", None)

    with open(log_path, "w", encoding="utf-8") as log:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                env=environment,
            )
        except OSError as exc:
            return (
                {
                    "ticker": ticker,
                    "company": company,
                    "status": "error",
                    "error": f"WorkerStartError: {exc}",
                    "log_path": str(log_path),
                    "duration_seconds": round(time.monotonic() - started, 2),
                },
                True,
            )
        try:
            await asyncio.wait_for(process.wait(), timeout=args.ticker_timeout)
        except TimeoutError:
            await terminate_process(process)
            return (
                {
                    "ticker": ticker,
                    "company": company,
                    "status": "error",
                    "error": f"TickerTimeout: exceeded {args.ticker_timeout} seconds",
                    "log_path": str(log_path),
                    "duration_seconds": round(time.monotonic() - started, 2),
                },
                True,
            )
        except asyncio.CancelledError:
            await terminate_process(process)
            raise

    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["log_path"] = str(log_path)
        except (OSError, json.JSONDecodeError) as exc:
            result = {
                "ticker": ticker,
                "company": company,
                "status": "error",
                "error": f"WorkerResultError: {exc}",
                "log_path": str(log_path),
            }
    else:
        result = {
            "ticker": ticker,
            "company": company,
            "status": "error",
            "error": f"WorkerExit: process exited {process.returncode} without a result",
            "log_path": str(log_path),
        }
    result["duration_seconds"] = round(time.monotonic() - started, 2)
    return result, result["status"] == "error" and is_retryable_error(result["error"])


def is_rate_limit_error(error: str) -> bool:
    """True for a provider quota rejection, as opposed to any transient fault.

    These need their own handling: the quota refills on a fixed clock, so the
    only useful response is to wait out the window. The ordinary backoff of a
    few seconds just spends every attempt inside the same exhausted minute.
    """
    normalized = error.lower()
    return any(
        marker in normalized
        for marker in ("429", "rate limit", "too many requests", "request_limit_exceeded")
    )


def is_retryable_error(error: str) -> bool:
    normalized = error.lower()
    transient_markers = (
        "timeout",
        "throttl",
        "too many requests",
        "rate limit",
        "serviceunavailable",
        "internalserver",
        "connection",
        "temporarily unavailable",
        "workerexit",
        "429",
        "500",
        "502",
        "503",
        "504",
    )
    return any(marker in normalized for marker in transient_markers)


def is_authentication_error(error: str) -> bool:
    normalized = error.lower()
    auth_markers = (
        "unauthorizedssotoken",
        "expiredtoken",
        "invalidsso",
        "incompletesignature",
        "unable to locate credentials",
        "could not load credentials",
    )
    return any(marker in normalized for marker in auth_markers)


async def run_ticker_with_retries(
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    *,
    ticker: str,
    company: str,
    analysis_date: str,
    output_dir: Path,
) -> dict:
    total_attempts = args.ticker_retries + 1
    attempt = 0
    failures = 0  # only ordinary faults spend the retry budget
    quota_waits = 0
    while True:
        attempt += 1
        print(f"{ticker}: attempt {attempt}", flush=True)
        async with semaphore:
            result, retryable = await run_worker_once(
                args,
                ticker=ticker,
                company=company,
                analysis_date=analysis_date,
                output_dir=output_dir,
                attempt=attempt,
            )
        if result["status"] == "success" or not retryable:
            result["attempts"] = attempt
            return result

        # A quota rejection is not the holding's fault and says nothing about
        # whether it can be analysed, so waiting out the window is free of any
        # retry budget. Without this the whole portfolio failed together.
        if is_rate_limit_error(result.get("error") or "") and quota_waits < MAX_QUOTA_WAITS:
            quota_waits += 1
            delay = QUOTA_WINDOW_SECONDS + random.uniform(0, QUOTA_WINDOW_SECONDS / 2)
            print(
                f"{ticker}: provider quota exhausted; waiting {delay:.0f}s for the "
                f"window to refill (wait {quota_waits}/{MAX_QUOTA_WAITS})",
                flush=True,
            )
        else:
            failures += 1
            if failures >= total_attempts:
                result["attempts"] = attempt
                return result
            delay = args.retry_base_delay * (2 ** (failures - 1))
            delay += random.uniform(0, args.retry_base_delay)
            print(f"{ticker}: transient failure; retrying in {delay:.1f}s", flush=True)
        await asyncio.sleep(delay)


def load_previous_results(
    output_dir: Path,
    analysis_date: str,
    args: argparse.Namespace,
) -> dict[str, dict]:
    json_path = output_dir / "recommendations.json"
    if not json_path.exists() or args.force:
        return {}
    old_payload = json.loads(json_path.read_text(encoding="utf-8"))
    old_settings = old_payload.get("settings", {})
    # Every setting that changes what the agents produce must invalidate the
    # reuse, not just the model names. Re-running with a deeper depth or a
    # different provider used to silently return the previous run's answers.
    current = {
        "quick_model": args.quick_model,
        "deep_model": args.deep_model,
        "research_depth": getattr(args, "depth", DEFAULT_DEPTH),
        "llm_provider": getattr(args, "llm_provider", None) or "bedrock",
        "effort": getattr(args, "effort", None),
    }
    if old_payload.get("analysis_date") != analysis_date or any(
        old_settings.get(key) != value for key, value in current.items()
    ):
        return {}
    return {
        item["ticker"]: item
        for item in old_payload.get("results", [])
        if item.get("status") == "success"
    }


async def run_batch(
    args: argparse.Namespace,
    watchlist: list[tuple[str, str]],
    analysis_date: str,
    output_dir: Path,
    holdings: dict[str, float],
) -> list[dict]:
    previous = load_previous_results(output_dir, analysis_date, args)
    result_by_ticker = {
        ticker: previous[ticker]
        for ticker, _ in watchlist
        if ticker in previous
    }
    for ticker in result_by_ticker:
        print(f"{ticker}: reusing completed analysis", flush=True)

    pending = [(ticker, company) for ticker, company in watchlist if ticker not in previous]
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(
            run_ticker_with_retries(
                args,
                semaphore,
                ticker=ticker,
                company=company,
                analysis_date=analysis_date,
                output_dir=output_dir,
            )
        )
        for ticker, company in pending
    ]
    try:
        for completed in asyncio.as_completed(tasks):
            result = await completed
            result_by_ticker[result["ticker"]] = result
            print(
                f"{result['ticker']}: "
                f"{result.get('rating') if result['status'] == 'success' else result['error']}",
                flush=True,
            )
            ordered = [
                result_by_ticker[ticker]
                for ticker, _ in watchlist
                if ticker in result_by_ticker
            ]
            write_outputs(output_dir, build_payload(analysis_date, args, ordered, holdings))
            if result["status"] == "error" and is_authentication_error(result["error"]):
                raise AuthenticationExpiredError(
                    "AWS SSO session expired; run `aws sso login --profile "
                    f"{args.aws_profile}` and rerun. Completed tickers will be reused."
                )
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    return [result_by_ticker[ticker] for ticker, _ in watchlist]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", help="Override the default 21-stock watchlist")
    parser.add_argument("--analysis-date", help="YYYY-MM-DD; defaults to today")
    parser.add_argument("--smoke-test", action="store_true", help="Run only the first ticker")
    parser.add_argument("--capital", type=float, help="Portfolio value used for buy amounts")
    portfolio_source = parser.add_mutually_exclusive_group()
    portfolio_source.add_argument(
        "--holdings", help="JSON file mapping tickers to current position values"
    )
    portfolio_source.add_argument(
        "--trading212",
        action="store_true",
        help="Load live capital and holdings from the Trading 212 account",
    )
    parser.add_argument(
        "--currency",
        help="Capital/holdings currency label; defaults to Trading 212 currency or USD",
    )
    parser.add_argument("--max-position-pct", type=float, default=5.0)
    parser.add_argument("--output-dir", help="Output directory; defaults under results/watchlists")
    parser.add_argument("--force", action="store_true", help="Rerun completed tickers")
    parser.add_argument("--aws-profile", default=os.getenv("AWS_PROFILE", DEFAULT_PROFILE))
    parser.add_argument("--aws-region", default=os.getenv("AWS_REGION", "us-east-1"))
    parser.add_argument("--quick-model", default=DEFAULT_QUICK_MODEL)
    parser.add_argument("--deep-model", default=DEFAULT_DEEP_MODEL)
    parser.add_argument(
        "--depth",
        choices=sorted(DEPTH_PROFILES),
        default=DEFAULT_DEPTH,
        help="Analysis effort: more analysts and debate rounds as depth rises.",
    )
    parser.add_argument(
        "--llm-provider",
        default=os.getenv("TRADINGAGENTS_LLM_PROVIDER"),
        help="LLM provider (default bedrock). Use 'openai_compatible' with "
        "--llm-base-url for Databricks Model Serving or any OpenAI-style gateway.",
    )
    parser.add_argument(
        "--effort",
        default=None,
        help="Reasoning effort for models that expose one (low/medium/high/xhigh). "
        "Ignored by models without an effort dial.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=os.getenv("TRADINGAGENTS_LLM_BACKEND_URL"),
        help="Base URL for the provider, e.g. "
        "https://<workspace>.cloud.databricks.com/serving-endpoints",
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--ticker-timeout", type=int, default=DEFAULT_TICKER_TIMEOUT)
    parser.add_argument("--ticker-retries", type=int, default=DEFAULT_TICKER_RETRIES)
    parser.add_argument("--retry-base-delay", type=float, default=5.0)
    parser.add_argument("--llm-max-retries", type=int, default=DEFAULT_LLM_RETRIES)
    parser.add_argument("--skip-model-preflight", action="store_true")
    parser.add_argument("--worker-ticker", help=argparse.SUPPRESS)
    parser.add_argument("--worker-company", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", help=argparse.SUPPRESS)
    return parser.parse_args()


def run_worker(
    args: argparse.Namespace,
    *,
    analysis_date: str,
    output_dir: Path,
) -> int:
    """Execute one ticker in an isolated subprocess."""
    if not args.worker_ticker or not args.worker_company or not args.worker_result:
        raise ValueError("worker mode requires ticker, company, and result path")

    configure_aws_environment(args, block_dotenv=True)
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)

    ticker = args.worker_ticker
    company = args.worker_company
    safe_ticker = safe_component(ticker)
    profile = DEPTH_PROFILES[getattr(args, "depth", DEFAULT_DEPTH)]
    config = DEFAULT_CONFIG.copy()
    config.update(
        llm_provider=getattr(args, "llm_provider", None) or "bedrock",
        backend_url=getattr(args, "llm_base_url", None),
        openai_reasoning_effort=getattr(args, "effort", None),
        quick_think_llm=args.quick_model,
        deep_think_llm=args.deep_model,
        output_language="English",
        max_debate_rounds=profile["max_debate_rounds"],
        max_risk_discuss_rounds=profile["max_risk_discuss_rounds"],
        checkpoint_enabled=True,
        llm_max_retries=args.llm_max_retries,
        temperature=None,
        results_dir=str(output_dir / "runtime"),
        data_cache_dir=str(output_dir / "cache"),
        memory_log_path=str(output_dir / "memory" / f"{safe_ticker}.md"),
    )
    try:
        graph = TradingAgentsGraph(
            selected_analysts=profile["analysts"], debug=False, config=config
        )
        state, rating = graph.propagate(ticker, analysis_date)
        report_path = graph.save_reports(
            state,
            ticker,
            save_path=output_dir / "stocks" / safe_ticker,
        )
        decision = state["final_trade_decision"]
        result = {
            "ticker": ticker,
            "company": company,
            "status": "success",
            "rating": rating,
            "executive_summary": extract_field(decision, "Executive Summary"),
            "investment_thesis": extract_field(decision, "Investment Thesis"),
            "price_target": extract_field(decision, "Price Target"),
            "time_horizon": extract_field(decision, "Time Horizon"),
            "report_path": str(report_path),
        }
    except Exception as exc:
        result = {
            "ticker": ticker,
            "company": company,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    atomic_write_json(Path(args.worker_result), result)
    return int(result["status"] == "error")


def validate_args(args: argparse.Namespace) -> None:
    if args.capital is not None and args.capital <= 0:
        raise ValueError("--capital must be greater than zero")
    if not 0 < args.max_position_pct <= 100:
        raise ValueError("--max-position-pct must be between 0 and 100")
    if not 1 <= args.concurrency <= MAX_CONCURRENCY:
        raise ValueError(f"--concurrency must be between 1 and {MAX_CONCURRENCY}")
    if args.ticker_timeout < 60:
        raise ValueError("--ticker-timeout must be at least 60 seconds")
    if not 0 <= args.ticker_retries <= 5:
        raise ValueError("--ticker-retries must be between 0 and 5")
    if not 0 <= args.llm_max_retries <= 6:
        raise ValueError("--llm-max-retries must be between 0 and 6")
    if args.retry_base_delay < 0:
        raise ValueError("--retry-base-delay must be non-negative")


def main() -> int:
    args = parse_args()
    validate_args(args)
    analysis_date = args.analysis_date or datetime.now().astimezone().date().isoformat()
    datetime.strptime(analysis_date, "%Y-%m-%d")
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else Path(
            os.getenv("TRADINGAGENTS_RESULTS_DIR", "~/.tradingagents/logs")
        ).expanduser()
        / "watchlists"
        / analysis_date
    )
    if args.worker_ticker:
        return run_worker(args, analysis_date=analysis_date, output_dir=output_dir)

    names = dict(DEFAULT_WATCHLIST)

    if args.trading212:
        holdings, trading212_capital, trading212_currency = load_trading212_portfolio()
        if args.capital is None:
            args.capital = trading212_capital
        if args.currency is None:
            args.currency = trading212_currency
        print(
            f"Trading 212: loaded {len(holdings)} positions; "
            f"portfolio value {args.currency} {trading212_capital:,.2f}",
            flush=True,
        )
    else:
        holdings = load_holdings(args.holdings)
    if args.currency is None:
        args.currency = "USD"

    watchlist = derive_watchlist(args, holdings, names)
    if args.smoke_test:
        watchlist = watchlist[:1]
        args.concurrency = 1

    configure_aws_environment(args, block_dotenv=False)
    try:
        with exclusive_run_lock(output_dir):
            preflight(args)
            results = asyncio.run(
                run_batch(args, watchlist, analysis_date, output_dir, holdings)
            )
            write_outputs(output_dir, build_payload(analysis_date, args, results, holdings))
    except AuthenticationExpiredError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted; completed results and checkpoints were preserved.", file=sys.stderr)
        return 130

    print(f"\nConsolidated recommendations: {output_dir / 'recommendations.md'}")
    print(f"Machine-readable results:     {output_dir / 'recommendations.json'}")
    return int(any(item["status"] == "error" for item in results))


if __name__ == "__main__":
    raise SystemExit(main())
