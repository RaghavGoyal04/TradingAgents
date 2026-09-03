"""Streamlit entry point for the Portfolio Intelligence Dashboard.

Read-only: renders the versioned artifacts written by the orchestrator and
offers Run/Resume controls that spawn (never in-process) the orchestrator. Bind
to localhost:

    streamlit run dashboard/app.py --server.address 127.0.0.1

All heavy logic lives in ``tradingagents.portfolio`` / ``tradingagents.forecast``
so this file stays a thin presentation layer. Every number shown is explained in
plain language next to it -- the dashboard is for deciding, not for decoding.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import streamlit as st

from tradingagents.dataflows.congress_trades import (
    SourceUnavailable as CongressSourceUnavailable,
)
from tradingagents.dataflows.sec_form4 import (
    SourceUnavailable as InsiderSourceUnavailable,
)
from tradingagents.discovery import congress, insiders
from tradingagents.portfolio import launcher, llm_config
from tradingagents.portfolio.actions import (
    DRIFT_THRESHOLD,
    MIN_TRADE_VALUE,
    decision_table,
    next_day_actions,
    rebalance_plan,
)
from tradingagents.portfolio.contracts import (
    FORECASTS,
    PORTFOLIO_SNAPSHOT,
    RECOMMENDATIONS,
    read_json,
)
from tradingagents.portfolio.launcher import MAX_CONCURRENCY
from tradingagents.portfolio.manifest import load_manifest

st.set_page_config(page_title="Portfolio Intelligence", layout="wide")

STATUS_ICON = {
    "complete": "✅",
    "running": "🔄",
    "pending": "⏳",
    "failed": "❌",
    "interrupted": "⏸️",
    "success": "✅",
    "error": "❌",
}

PRIORITY_ICON = {"high": "🔴", "medium": "🟠", "info": "🔵"}

CONCURRENCY_HELP = (
    "How many holdings run in parallel. Each is a separate process that spends "
    "almost all its time waiting on the LLM, so this is bounded by the "
    "provider's rate limit rather than your CPU. Short probe calls survive 32 "
    "in parallel, but a real analysis sends far more tokens: 38 at once "
    "exhausted the workspace tokens-per-minute quota and every holding failed "
    "with a 429. Six is the setting known to complete. A worker needs ~155 MB."
)

DEPTH_HELP = (
    "How hard the analyst team works per holding. Deeper adds more analysts "
    "(more evidence) and more debate rounds (more challenge), so runtime and "
    "LLM cost rise roughly in proportion."
)
# Runtimes are extrapolated from the measured shallow run (38 holdings, ~40 min).
DEPTH_SUMMARY = {
    "shallow": "Market analyst only, 1 debate round. ~40 min for 38 holdings.",
    "medium": "Market + news, 2 debate rounds. Roughly 2x shallow.",
    "deep": "Market + news + fundamentals + social, 3 debate rounds. "
    "Roughly 4-5x shallow — start it and leave it.",
}

# Plain-language descriptions of each forecasting model, shown instead of raw
# model keys so the evaluation table is readable without domain knowledge.
MODEL_EXPLANATION = {
    "naive": "Simple random-walk: assumes no drift, spread set by recent volatility.",
    "bootstrap": "Replays real historical 5-day chunks of your holdings together, "
    "keeping crashes correlated.",
    "bootstrap_v1": "Replays real historical 5-day chunks of your holdings together, "
    "keeping crashes correlated.",
    "timesfm": "Google's TimesFM deep-learning time-series model.",
    "timesfm_v3": "Google's TimesFM deep-learning time-series model.",
}

# Long enough to cover the 45-day filing deadline, so most trades in the window
# have actually been disclosed by now.
CONGRESS_LOOKBACK_DAYS = 60

# Form 4 lands within two business days, so a much shorter window still sees
# everything. It also matches the fortnight over which a cluster of insiders
# buying the same name counts as one event.
INSIDER_LOOKBACK_DAYS = 30


class ModelChoice(NamedTuple):
    """The sidebar's model and run settings, for every tab that spawns work."""

    quick_model: str
    deep_model: str
    effort: str | None
    llm_provider: str
    llm_base_url: str | None
    run_env: dict[str, str]
    depth: str
    analysis_date: str
    runs_base: str


def _icon(status: str | None) -> str:
    return STATUS_ICON.get(status or "", "•")


def _md(text: str | None) -> str:
    """Escape agent prose for Streamlit markdown.

    Analyst text is full of dollar prices ("stop at $425, trim at $520-$540").
    Streamlit reads a pair of ``$`` as LaTeX math and renders the prices as
    mangled equations, so they must be escaped.
    """
    return (text or "").replace("$", r"\$")


def _money(value: float | None, currency: str = "") -> str:
    """Compact currency string that fits a narrow metric column."""
    if value is None:
        return "n/a"
    prefix = "-" if value < 0 else ""
    magnitude = abs(value)
    # Only abbreviate at millions; whole pounds still fit a narrow column and
    # keeping them exact avoids the "22.8k" loss of precision.
    if magnitude >= 1_000_000:
        return f"{prefix}{magnitude / 1_000_000:,.2f}M"
    return f"{prefix}{magnitude:,.0f}"


def _label(position: dict) -> str:
    """Human-first label: company name, ticker in parentheses."""
    symbol = position.get("symbol") or position.get("broker_ticker") or "?"
    name = position.get("name")
    return f"{name} ({symbol})" if name and name != symbol else str(symbol)


def _model_controls(
    *, agents_run: bool
) -> tuple[str, str, str | None, str, str | None, dict[str, str]]:
    """Provider, model and reasoning-effort pickers driven by config/models.yaml.

    Returns the runner arguments plus the environment carrying any captured
    token, which the caller merges into the child process rather than the
    command line.

    When no agents will run the whole block is hidden: a forecast makes no LLM
    calls, so showing a live model picker there implies a choice that has no
    effect on the output.
    """
    catalog = llm_config.load_catalog()
    names = list(catalog)
    if not agents_run:
        st.sidebar.caption(
            "Forecast only is pure price statistics — no LLM is called, so "
            "there is no model to choose."
        )
        fallback = catalog[names[0]]
        return (
            fallback.default_quick or fallback.models[0].id,
            fallback.default_deep or fallback.models[-1].id,
            None,
            fallback.llm_provider,
            None,
            {},
        )
    provider = catalog[
        st.sidebar.selectbox(
            "Provider",
            names,
            format_func=lambda n: catalog[n].label,
            key="provider",
            help="Credentials come from your existing AWS or Databricks profile; "
            "nothing to paste here.",
        )
    ]

    def pick(label: str, key: str, index: int) -> str:
        ids = [m.id for m in provider.models]
        return st.sidebar.selectbox(
            label,
            ids,
            index=min(index, len(ids) - 1),
            format_func=lambda i: provider.model(i).label,
            # Keys are provider-scoped: a shared key would keep the previous
            # provider's model id in widget state and crash on the next render,
            # because that id is not among the new provider's options.
            key=f"{key}_{provider.name}",
        )

    quick_model = pick("Quick model", "quick_model", provider.index_of(provider.default_quick))
    deep_model = pick("Deep model", "deep_model", provider.index_of(provider.default_deep))

    # TradingAgentsGraph builds both clients from one shared kwargs dict, so a
    # single effort reaches the quick model as well as the deep one. Offer the
    # control when either model can use it, and show which will.
    deep_efforts = provider.efforts_for(deep_model)
    quick_efforts = provider.efforts_for(quick_model)
    efforts = deep_efforts or quick_efforts
    effort = None
    if efforts:
        applies = [
            provider.model(m).label
            for m, e in ((quick_model, quick_efforts), (deep_model, deep_efforts))
            if e
        ]
        effort = st.sidebar.selectbox(
            "Reasoning effort",
            efforts,
            index=len(efforts) - 1,
            key=f"effort_{quick_model}_{deep_model}",
            help="One setting, applied to every model that accepts it — the "
            "pipeline builds both clients from the same options. The quick "
            "model makes far more calls than the deep one, so raising this "
            "while a reasoning model sits in the quick slot is what gets "
            "expensive.",
        )
        st.sidebar.caption(f"Applies to: {', '.join(applies)}.")
    else:
        st.sidebar.caption("Neither model has a reasoning-effort dial.")

    base_url, run_env = None, {}
    try:
        base_url, run_env = llm_config.resolve(provider)
        if run_env:
            st.sidebar.caption(f"Token picked up from `{provider.profile}` profile.")
    except llm_config.CredentialError as exc:
        st.sidebar.error(str(exc))
    return quick_model, deep_model, effort, provider.llm_provider, base_url, run_env


def sidebar_controls() -> tuple[Path | None, ModelChoice]:
    st.sidebar.header("Run")
    base = st.sidebar.text_input(
        "Runs directory", value=str(launcher.runs_root()), key="runs_dir"
    )
    run_dirs = launcher.list_run_dirs(base)
    labels = [p.name for p in run_dirs]
    selected = None
    if labels:
        # Default to the newest run (list_run_dirs is newest-first). Defaulting
        # to "(new run)" made the page report "No runs yet" while a finished
        # run sat one dropdown away.
        choice = st.sidebar.selectbox(
            "Existing run", ["(new run)"] + labels, index=1, key="existing_run"
        )
        if choice != "(new run)":
            selected = run_dirs[labels.index(choice)]

    st.sidebar.divider()
    # Positive framing: the mode says what you GET, not what you skip.
    mode = st.sidebar.radio(
        "What to run",
        ["Full analysis (agents + forecast)", "Forecast only (fast)"],
        key="run_mode",
        help=(
            "Full analysis runs the AI analyst team per holding and produces "
            "buy/sell calls (slower, uses your LLM credits). Forecast only "
            "refreshes prices, risk and rebalancing in seconds."
        ),
    )
    skip_agents = mode.startswith("Forecast only")
    depth, concurrency = "shallow", 6
    if not skip_agents:
        depth = st.sidebar.select_slider(
            "Analysis effort",
            options=["shallow", "medium", "deep"],
            value="shallow",
            key="depth",
            help=DEPTH_HELP,
        )
        st.sidebar.caption(DEPTH_SUMMARY[depth])
        concurrency = st.sidebar.slider(
            "Holdings analysed at once",
            min_value=1,
            max_value=MAX_CONCURRENCY,
            value=6,
            key="concurrency",
            help=CONCURRENCY_HELP,
        )
        st.sidebar.caption(
            f"{concurrency} at a time. The limit is tokens per minute across "
            "the whole workspace, not requests, so a full analysis at 38 was "
            "rate-limited into failing every holding. Raise this in small steps."
        )

        force = st.sidebar.checkbox(
            "Re-analyse everything",
            value=False,
            key="force",
            help=(
                "By default a re-run reuses tickers that already succeeded "
                "today, which is fast but returns the earlier answers. Tick "
                "this to analyse every holding again from scratch."
            ),
        )

    else:
        force = False

    use_timesfm = st.sidebar.checkbox(
        "Back-test TimesFM",
        value=False,
        key="use_timesfm",
        help=(
            "Adds Google's TimesFM model to the forecast bake-off. It is only "
            "used if it beats the simple models on your own holdings. Adds "
            "several minutes on CPU."
        ),
    )

    source = st.sidebar.radio(
        "Portfolio source", ["trading212", "holdings"], key="source"
    )
    holdings_path = None
    if source == "holdings":
        holdings_path = st.sidebar.text_input("Holdings JSON path", key="holdings_path")
    watchlist_raw = st.sidebar.text_input(
        "Optional watchlist (comma-separated)", key="watchlist"
    )
    watchlist = [s.strip().upper() for s in watchlist_raw.split(",") if s.strip()]
    quick_model, deep_model, effort, llm_provider, llm_base_url, run_env = (
        _model_controls(agents_run=not skip_agents)
    )
    analysis_date = st.sidebar.text_input(
        "Analysis date", datetime.now().date().isoformat(), key="analysis_date"
    )

    run_dir = selected or launcher.new_run_dir(analysis_date, base)
    active = launcher.is_run_active(run_dir)
    if active:
        st.sidebar.info(f"A run is active in {run_dir.name}.")

    run_label = "Run forecast" if skip_agents else "Run full analysis"
    col_run, col_resume = st.sidebar.columns(2)
    try:
        if col_run.button(run_label, disabled=active, key="run_btn", type="primary"):
            command = launcher.build_orchestrator_command(
                run_dir,
                source=source,
                holdings_path=holdings_path,
                watchlist=watchlist,
                analysis_date=analysis_date,
                quick_model=quick_model,
                deep_model=deep_model,
                skip_agents=skip_agents,
                use_timesfm=use_timesfm,
                force=force,
                depth=depth,
                effort=effort,
                llm_provider=llm_provider,
                llm_base_url=llm_base_url,
                concurrency=concurrency,
            )
            pid = launcher.launch(command, run_dir, env=run_env)
            st.sidebar.success(f"Started orchestrator (PID {pid}).")
        if col_resume.button(
            "Resume", disabled=active or selected is None, key="resume_btn"
        ):
            command = launcher.build_orchestrator_command(
                run_dir,
                source=source,
                holdings_path=holdings_path,
                watchlist=watchlist,
                analysis_date=analysis_date,
                quick_model=quick_model,
                deep_model=deep_model,
                skip_agents=skip_agents,
                use_timesfm=use_timesfm,
                force=force,
                depth=depth,
                effort=effort,
                llm_provider=llm_provider,
                llm_base_url=llm_base_url,
                concurrency=concurrency,
                resume=True,
            )
            pid = launcher.launch(command, run_dir, env=run_env)
            st.sidebar.success(f"Resuming orchestrator (PID {pid}).")
    except (ValueError, RuntimeError) as exc:
        st.sidebar.error(str(exc))

    st.sidebar.caption("Read-only. Recommendations are advisory; execute manually.")
    return (
        run_dir if run_dir.exists() else None,
        ModelChoice(
            quick_model=quick_model,
            deep_model=deep_model,
            effort=effort,
            llm_provider=llm_provider,
            llm_base_url=llm_base_url,
            run_env=run_env,
            depth=depth,
            analysis_date=analysis_date,
            runs_base=base,
        ),
    )


def next_actions_section(snapshot: dict, forecasts: dict, recommendations: dict):
    """Ranked decisions first, supporting housekeeping tucked away below."""
    st.subheader("What to do next")
    decisions = decision_table(snapshot, recommendations)
    currency = snapshot.get("base_currency", "")

    if decisions:
        execute_on = recommendations.get("planned_execution_date")
        st.caption(
            "Strongest conviction first, biggest trade first within each band — "
            f"act top-down. "
            f"{'Planned for ' + execute_on + '. ' if execute_on else ''}"
            "Click any column header to re-sort. Advisory only: place any trade "
            "yourself in Trading 212."
        )
        frame = pd.DataFrame(
            [
                {
                    "#": i,
                    "Action": d["action"],
                    "Holding": d["name"],
                    f"Amount ({currency})": round(d["amount"], 2),
                    "Conviction %": d["conviction_score"],
                    "Conviction": d["conviction"],
                    "Upside to target": d["upside"],
                    "Target price": d["price_target"],
                    "Now": d["current_price"],
                    "Horizon": d["horizon"],
                }
                for i, d in enumerate(decisions, start=1)
            ]
        )
        st.dataframe(
            frame,
            width="stretch",
            hide_index=True,
            column_config={
                "Upside to target": st.column_config.NumberColumn(
                    format="percent",
                    help="Move from today's price to the analysts' price "
                    "target. Blank when the two are quoted in different units.",
                ),
                "Conviction %": st.column_config.ProgressColumn(
                    min_value=0,
                    max_value=100,
                    format="%d",
                    help="How far the analysts think the price can travel, "
                    "rescaled 0-100 (a 40% move to target scores 100). This is "
                    "NOT a probability of being right. Sortable.",
                ),
                "Conviction": st.column_config.TextColumn(
                    help="Band of the score: High = 50+ (target 20%+ away), "
                    "Medium = 25-49 (10-20%), Low = under 25. 'Unknown' means "
                    "the price target does not point the same way as the trade "
                    "(often an exit level on a sell), so it cannot be scored.",
                ),
                f"Amount ({currency})": st.column_config.NumberColumn(
                    format="%.2f", help="How much to trade. Drives the ranking."
                ),
                "Target price": st.column_config.NumberColumn(
                    format="%.2f", help="In the share's own listing currency."
                ),
                "Now": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        st.markdown("**Entry plan — when to actually buy or sell**")
        st.caption(
            "Each plan below gives the scale-in levels, stop and trigger the "
            "analysts set. Open the ones you intend to act on."
        )
        for i, d in enumerate(decisions[:8], start=1):
            upside = f" · {d['upside']:+.1%} to target" if d["upside"] else ""
            with st.expander(
                f"{i}. {d['action']} {d['name']} — {currency} "
                f"{d['amount']:,.0f} · {d['conviction']} conviction{upside}"
            ):
                st.markdown(f"**Entry & exit plan:** {_md(d['plan'])}")
                if d["sizing"]:
                    st.caption(
                        f"Sizing: {_md(d['sizing'])}  ·  Horizon: {d['horizon']}"
                    )
                if d["thesis"]:
                    st.markdown("**Why:**")
                    st.markdown(_md(d["thesis"]))
    else:
        st.caption("Advisory only. Nothing here is executed for you.")

    actions = next_day_actions(snapshot, forecasts, recommendations)
    if not actions:
        return
    with st.expander(f"Housekeeping & risk notes ({len(actions)})"):
        for item in actions:
            icon = PRIORITY_ICON.get(item["priority"], "•")
            st.markdown(f"{icon} **{item['title']}**")
            st.caption(item["detail"])


def rebalance_section(snapshot: dict):
    """Buy/hold/trim per holding, measured against your own pie targets."""
    st.subheader("Buy / hold / trim plan")
    plan = rebalance_plan(snapshot)
    if not plan:
        st.info(
            "No pie targets found for this account, so there is nothing to "
            "rebalance against. This plan uses the target weights you set in "
            "your Trading 212 pies."
        )
        return
    currency = snapshot.get("base_currency", "")
    st.caption(
        "This compares each holding against the target weight **you** set in "
        "your Trading 212 pie — it is not a market prediction. 'BUY' means you "
        "are below your own target, 'TRIM' means above it. Drifts under "
        f"{DRIFT_THRESHOLD:.0%} or {currency} {MIN_TRADE_VALUE:,.0f} are left "
        "as HOLD because the trade is not worth the cost."
    )

    pies = sorted({row["pie"] for row in plan})
    chosen = st.selectbox("Pie", ["All pies"] + pies, key="rebalance_pie")
    rows = plan if chosen == "All pies" else [r for r in plan if r["pie"] == chosen]

    actionable = [r for r in rows if r["action"] != "HOLD"]
    buys = sum(r["amount"] for r in actionable if r["amount"] > 0)
    trims = -sum(r["amount"] for r in actionable if r["amount"] < 0)
    cols = st.columns(3)
    cols[0].metric("To buy", f"{currency} {buys:,.0f}")
    cols[1].metric("To trim", f"{currency} {trims:,.0f}")
    cols[2].metric("Holdings off target", str(len(actionable)))

    frame = pd.DataFrame(
        [
            {
                "Action": r["action"],
                "Holding": r["name"],
                "Pie": r["pie"],
                f"Amount ({currency})": round(r["amount"], 2),
                "Target": r["target_share"],
                "Actual": r["current_share"],
                "Off by": r["drift"],
            }
            for r in rows
        ]
    )
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={
            "Target": st.column_config.NumberColumn(
                format="percent", help="Weight you set for this holding in the pie."
            ),
            "Actual": st.column_config.NumberColumn(
                format="percent", help="What it actually is right now."
            ),
            "Off by": st.column_config.NumberColumn(
                format="percent", help="Positive means overweight versus your target."
            ),
            f"Amount ({currency})": st.column_config.NumberColumn(
                format="%.2f",
                help="Positive = buy this much, negative = trim this much.",
            ),
        },
    )


def overview_tab(run_dir: Path, manifest: dict, snapshot: dict, forecasts: dict):
    if not snapshot:
        st.warning("No portfolio snapshot yet. Configure a source and press Run.")
        return
    currency = snapshot.get("base_currency", "")
    positions = [p for p in snapshot["positions"] if not p.get("watch_only")]
    weights = sorted((p.get("weight") or 0.0) for p in positions)[::-1]

    recommendations = read_json(run_dir / RECOMMENDATIONS) or {}
    next_actions_section(snapshot, forecasts, recommendations)

    st.divider()
    rebalance_section(snapshot)

    st.divider()
    st.subheader("Account")
    # Currency lives in the label so the value never truncates on narrow screens.
    cols = st.columns(4)
    cols[0].metric(
        f"Account value ({currency})",
        _money(snapshot.get("account_value")),
        help=f"Total account value reported by Trading 212, in {currency}.",
    )
    cols[1].metric(
        f"Cash ({currency})",
        _money(snapshot.get("cash")),
        help="Uninvested cash: account value minus the value of all positions.",
    )
    cols[2].metric(
        "Holdings",
        str(len(positions)),
        help="Number of distinct instruments you hold (watchlist items excluded).",
    )
    cols[3].metric(
        "Top-5 share",
        f"{sum(weights[:5]):.0%}",
        help="Share of the account held in your five largest positions. "
        "Higher means more concentrated, so single-stock news matters more.",
    )

    if manifest:
        st.write(
            f"**Latest run:** {_icon(manifest.get('status'))} "
            f"{manifest.get('status')} · {manifest.get('analysis_date', '')}"
        )
        phase_cols = st.columns(len(manifest["phases"]))
        for col, (name, phase) in zip(
            phase_cols, manifest["phases"].items(), strict=False
        ):
            col.write(f"{_icon(phase['status'])} {name}: {phase['status']}")

    if snapshot.get("unmapped"):
        st.error(
            "These holdings could not be matched to a market symbol and are "
            "excluded from forecasts/risk: " + ", ".join(snapshot["unmapped"])
        )


def positions_tab(snapshot: dict):
    st.subheader("Positions")
    if not snapshot:
        st.info("No snapshot yet.")
        return
    currency = snapshot.get("base_currency", "")
    st.caption(
        f"Values are in your account currency ({currency}). 'Gain' is the move "
        "since your average buy price. 'watch' rows are watchlist-only, not owned."
    )
    group_by_pie = st.toggle(
        "Group by pie", value=True, key="group_by_pie",
        help="Split holdings into the Trading 212 pies they belong to.",
    )

    def _frame(positions: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Name": p.get("name") or p.get("symbol"),
                    "Ticker": p.get("symbol") or p.get("broker_ticker"),
                    "Type": "watch" if p.get("watch_only") else "holding",
                    f"Value ({currency})": p.get("value"),
                    "Weight": p.get("weight"),
                    "Gain": p.get("unrealized_pct"),
                    "Units": p.get("quantity"),
                    "Listed in": p.get("instrument_currency"),
                    "Mapping": p.get("mapping_status"),
                }
                for p in positions
            ]
        ).sort_values(f"Value ({currency})", ascending=False)

    config = {
        "Weight": st.column_config.NumberColumn(
            format="percent", help="Share of your total account value."
        ),
        "Gain": st.column_config.NumberColumn(
            format="percent", help="Change since your average purchase price."
        ),
        f"Value ({currency})": st.column_config.NumberColumn(format="%.2f"),
    }

    if not group_by_pie:
        st.dataframe(
            _frame(snapshot["positions"]),
            width="stretch", hide_index=True, column_config=config,
        )
        return

    groups: dict[str, list[dict]] = {}
    for p in snapshot["positions"]:
        groups.setdefault(p.get("pie") or "Not in a pie", []).append(p)
    # Largest pie first; the catch-all buckets sink to the bottom.
    ordering = sorted(
        groups,
        key=lambda name: (
            name in ("Not in a pie", "Watchlist"),
            -sum(p.get("value") or 0.0 for p in groups[name]),
        ),
    )
    for name in ordering:
        members = groups[name]
        total = sum(p.get("value") or 0.0 for p in members)
        st.markdown(f"**{name}** — {len(members)} holdings · {currency} {total:,.0f}")
        st.dataframe(
            _frame(members), width="stretch", hide_index=True, column_config=config
        )


def forecasts_tab(snapshot: dict, forecasts: dict):
    st.subheader("Forecasts")
    if not forecasts or not forecasts.get("assets"):
        st.info("No forecasts yet.")
        return

    names = {
        p.get("symbol"): p.get("name") or p.get("symbol")
        for p in (snapshot or {}).get("positions", [])
    }
    st.caption(
        "How to read this: we simulate thousands of plausible price paths for "
        "each holding. The bands below are percentage moves from today's price "
        f"over the horizon, using data up to {forecasts.get('data_cutoff')}."
    )

    options = list(forecasts["assets"].keys())
    symbol = st.selectbox(
        "Asset",
        options,
        key="fc_asset",
        format_func=lambda s: f"{names.get(s, s)} ({s})",
    )
    asset = forecasts["assets"][symbol]

    hist = asset.get("history_tail", [])
    if hist:
        st.line_chart(
            pd.DataFrame({"price": hist}),
            height=220,
        )
        st.caption("Recent closing prices (most recent 60 trading days).")

    for horizon, bands in (asset.get("quantiles") or {}).items():
        p10, p50, p90 = bands.get("p10"), bands.get("p50"), bands.get("p90")
        if None in (p10, p50, p90):
            continue
        st.markdown(f"**Next {horizon} trading days — {names.get(symbol, symbol)}**")
        band_cols = st.columns(3)
        band_cols[0].metric(
            "Pessimistic", f"{p10:+.1%}",
            help="Only 1 simulation in 10 was worse than this.",
        )
        band_cols[1].metric(
            "Middle", f"{p50:+.1%}",
            help="Half the simulations were better, half worse.",
        )
        band_cols[2].metric(
            "Optimistic", f"{p90:+.1%}",
            help="Only 1 simulation in 10 was better than this.",
        )
        st.caption(
            f"In plain terms: over the next {horizon} trading days, "
            f"{names.get(symbol, symbol)} most likely moves about {p50:+.1%}, and "
            f"in 8 cases out of 10 it lands between {p10:+.1%} and {p90:+.1%}. "
            "This is a range of outcomes, not a prediction."
        )

    _model_choice_section(forecasts)


def _model_choice_section(forecasts: dict):
    """Explain which forecasting model was chosen and why, without raw JSON."""
    evaluation = forecasts.get("evaluation") or {}
    if not evaluation:
        return
    st.divider()
    st.markdown("**Which forecasting model is being used?**")
    st.caption(
        "Before trusting a model we back-test it on your own holdings: it only "
        "ever sees past data, then we score how well its predicted ranges matched "
        "what actually happened (lower score = better). The winner is used."
    )
    rows = []
    for horizon, entry in sorted(evaluation.items(), key=lambda kv: int(kv[0])):
        winner = entry.get("winner", "?")
        scores = entry.get("baseline_mean_scores") or {}
        rows.append(
            {
                "Horizon": f"{horizon} day(s)",
                "Model used": winner,
                "What it does": MODEL_EXPLANATION.get(winner, "—"),
                "Score (lower is better)": scores.get(winner),
                "Holdings tested": entry.get("eligible_count"),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if not any(e.get("promoted") for e in evaluation.values()):
        tested = any((e.get("counted") or 0) > 0 for e in evaluation.values())
        state = (forecasts.get("timesfm") or {}).get("state")
        detail = (forecasts.get("timesfm") or {}).get("detail", "")
        if tested:
            st.info(
                "TimesFM was tested but did not beat the simple models by enough "
                "to be trusted, so the winning baseline above is used."
            )
        elif state == "load_failed":
            st.warning(
                f"TimesFM is installed but could not run. {detail} "
                "The baselines above are being used in the meantime."
            )
        elif state == "not_installed":
            st.info(
                "TimesFM (the deep-learning candidate) is not installed, so it "
                'was not evaluated. Install it with: uv pip install "timesfm'
                '[torch]". The baselines above are being used.'
            )
        else:
            st.info(
                "TimesFM was not evaluated in this run. The baselines above are "
                "being used."
            )
    with st.expander("Show raw evaluation data"):
        st.json(evaluation)


def risk_tab(snapshot: dict, forecasts: dict):
    st.subheader("Portfolio risk")
    portfolio = (forecasts or {}).get("portfolio") or {}
    if not portfolio:
        st.info("No portfolio simulation yet.")
        return

    currency = portfolio.get("currency", "")
    horizon = portfolio.get("horizon", 20)
    paths = portfolio.get("n_paths", 0)
    st.caption(
        f"What this is: we replayed {paths:,} plausible versions of the next "
        f"{horizon} trading days using your actual holdings and how they have "
        "historically moved together. Each number below is the result across all "
        f"those simulated futures, in {currency}."
    )

    pnl = portfolio["pnl"]
    loss = portfolio.get("loss_probability", 0)
    var = portfolio.get("var_95")
    cvar = portfolio.get("cvar_95")

    cols = st.columns(4)
    cols[0].metric(
        f"Typical outcome ({currency})",
        _money(pnl["p50"]),
        help="The middle result: half the simulations did better, half worse.",
    )
    cols[1].metric(
        f"Bad case ({currency})",
        _money(pnl["p5"]),
        help="The 5th percentile: only 1 simulation in 20 was worse than this.",
    )
    cols[2].metric(
        f"Good case ({currency})",
        _money(pnl["p95"]),
        help="The 95th percentile: only 1 simulation in 20 was better than this.",
    )
    cols[3].metric(
        "Chance of a loss",
        f"{loss:.0%}",
        help=f"Share of simulations that ended below today's value after "
        f"{horizon} trading days.",
    )

    st.markdown(
        f"**In one sentence:** over the next {horizon} trading days your "
        f"portfolio most likely changes by about **{currency} {pnl['p50']:,.0f}**; "
        f"there is a **{loss:.0%}** chance of ending down, and in the worst 5% of "
        f"cases you would be down at least **{currency} {abs(var or 0):,.0f}**."
    )

    with st.expander("What do these risk terms mean?"):
        st.markdown(
            f"""
- **Value at Risk (VaR 95): {currency} {abs(var or 0):,.0f}** — on the worst 1
  day in 20, you lose *at least* this much over the period.
- **Expected shortfall (CVaR 95): {currency} {abs(cvar or 0):,.0f}** — when
  you are in that worst 5%, this is the *average* loss. Always worse than VaR.
- **Median max drawdown: {portfolio.get('max_drawdown_p50', 0):.1%}** — the
  typical peak-to-trough dip *along the way*, even in runs that end up fine.
  This is the number that tests your nerve.
- **Risk contribution** — how much of the portfolio's total wobble comes from
  each holding. A holding can be small but still dominate risk if it is volatile.
"""
        )

    rc = portfolio.get("risk_contribution", {})
    if rc:
        names = {
            p.get("symbol"): p.get("name") or p.get("symbol")
            for p in (snapshot or {}).get("positions", [])
        }
        st.markdown("**Where your risk actually comes from**")
        frame = (
            pd.DataFrame(
                {
                    "Holding": [names.get(s, s) for s in rc],
                    "Share of portfolio risk": list(rc.values()),
                }
            )
            .sort_values("Share of portfolio risk", ascending=False)
            .set_index("Holding")
        )
        st.bar_chart(frame, height=260)
        st.caption(
            "Bars sum to 100% of portfolio variance. The tallest bars are what "
            "actually drive your day-to-day swings."
        )

    if portfolio.get("warnings"):
        st.warning("\n\n".join(portfolio["warnings"]))


def analysis_tab(run_dir: Path, snapshot: dict):
    st.subheader("Analysis & runs")
    recs = read_json(run_dir / RECOMMENDATIONS) or {}
    results = recs.get("results", [])
    names = {
        p.get("symbol"): p.get("name") or p.get("symbol")
        for p in (snapshot or {}).get("positions", [])
    }
    if not results:
        st.info(
            "No agent analysis in this run. Untick 'Skip agents (forecast only)' "
            "in the sidebar and press Run to generate per-holding recommendations."
        )
    for item in results:
        ticker = item.get("ticker", "?")
        label = names.get(ticker, ticker)
        with st.expander(
            f"{_icon(item.get('status'))} {label} ({ticker}) — "
            f"{item.get('rating', 'n/a')}"
        ):
            st.markdown(_md(item.get("executive_summary") or "No summary."))
            if item.get("report_path"):
                st.caption(f"Report: {item['report_path']}")
    events = run_dir / "events.jsonl"
    if events.exists():
        with st.expander("Run event log"):
            st.text(events.read_text(encoding="utf-8"))


def congress_tab(models: ModelChoice):
    """Screen the newest congressional disclosures for growth candidates.

    Three explicit steps, because each costs more than the last: reading the
    filings is free, ranking them is one LLM call, and analysing the picks is a
    full analyst run per ticker.
    """
    st.subheader("Growth candidates from congressional trades")
    st.caption(
        "Members of Congress must file a Periodic Transaction Report within 45 "
        "days of trading a stock. This reads those filings straight from the "
        "House Clerk's official archive, ranks the names elected officials have "
        "been buying, and asks the deep model to pick three with the most room "
        "to run."
    )
    st.warning(
        "Filings arrive up to 45 days after the trade and disclose a size band, "
        "never an amount. So this shows where buying has clustered recently — "
        "it is not what anyone is buying today, none of it is front-runnable, "
        "and a ranking is not a forecast."
    )

    lookback = st.slider(
        "Include trades from the last … days",
        min_value=30,
        max_value=180,
        value=CONGRESS_LOOKBACK_DAYS,
        step=15,
        key="congress_lookback",
        help="Measured on the trade date, not the filing date. Windows shorter "
        "than the 45-day filing deadline are necessarily incomplete.",
    )
    if st.button("Scan official filings", key="congress_scan_btn", type="primary"):
        with st.spinner("Reading House disclosure filings…"):
            try:
                st.session_state["congress_scan"] = congress.scan(lookback)
                st.session_state.pop("congress_ranking", None)
            except CongressSourceUnavailable as exc:
                st.error(f"No disclosure source could be reached: {exc}")

    scan = st.session_state.get("congress_scan")
    if scan is None:
        st.info("Press **Scan official filings** to pull the newest disclosures.")
        return

    for note in scan.source_notes:
        if "unavailable" in note:
            st.info(note)
        else:
            st.caption(note)

    shortlist = scan.shortlist()
    if not shortlist:
        st.info(
            "No net buying in any single stock was disclosed in this window. "
            "Widen the lookback and scan again."
        )
        return

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Ticker": s.ticker,
                    "Company": s.company,
                    "Score": round(s.score, 2),
                    "Net bought": _money(s.net_dollars, ""),
                    "Buyers": len(s.buyers),
                    "Option buys": s.option_buys,
                    "Latest trade": s.latest_trade_date.isoformat(),
                    "Who": ", ".join(s.buyers),
                }
                for s in shortlist
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Score rises with how much was bought, how many separate members "
        "bought it, how recently, and whether any of them used options. "
        "Breadth counts for more than size: several members independently "
        "buying one name is harder to explain away than one member buying a lot."
    )

    if scan.selling:
        with st.expander("Heaviest net selling, for context"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Ticker": s.ticker,
                            "Company": s.company,
                            "Net sold": _money(abs(s.net_dollars), ""),
                            "Sellers": len(s.sellers),
                            "Who": ", ".join(s.sellers),
                        }
                        for s in scan.selling[:10]
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    st.divider()
    _insider_confirmation_section(shortlist)

    st.divider()
    if st.button(
        f"Rank the top {congress.PICK_COUNT} with {models.deep_model}",
        key="congress_rank_btn",
    ):
        with st.spinner("Asking the deep model to pick and argue…"):
            try:
                st.session_state["congress_ranking"] = _rank_with_sidebar_model(
                    scan, models
                )
            except Exception as exc:  # provider/credential errors belong on screen
                st.error(f"Ranking failed: {exc}")

    ranking = st.session_state.get("congress_ranking")
    if ranking is None:
        return

    st.markdown(_md(ranking.markdown))
    st.divider()
    _congress_analysis_section(ranking, models)


def _insider_confirmation_section(shortlist):
    """Check whether the companies' own officers bought the shortlisted names.

    A second, independent read on the same tickers: Congress files 45 days
    after the fact in size bands, while an officer files an exact dollar amount
    within two business days. Only the shortlist is checked, which costs a
    couple of requests per symbol rather than the whole-market crawl that
    discovering new names from Form 4 would take.
    """
    st.markdown("**Do the companies' own insiders agree?**")
    st.caption(
        "Officers, directors and 10% owners must file SEC Form 4 within two "
        "business days of trading their own company's stock. This checks the "
        "shortlist above for open-market purchases — bought with their own "
        "cash, excluding grants, option exercises and tax withholding, which "
        "carry no view either way."
    )

    tickers = [s.ticker for s in shortlist]
    if st.button("Check SEC Form 4 filings", key="insider_check_btn"):
        with st.spinner(f"Reading EDGAR for {len(tickers)} tickers…"):
            try:
                st.session_state["insider_scan"] = insiders.confirm(
                    tickers, INSIDER_LOOKBACK_DAYS
                )
            except InsiderSourceUnavailable as exc:
                st.error(f"EDGAR could not be read: {exc}")

    scan = st.session_state.get("insider_scan")
    if scan is None:
        return

    st.caption(scan.source_note)
    buying = scan.buying
    if not buying:
        st.info(
            "No insider bought any shortlisted name on the open market in the "
            "last "
            f"{INSIDER_LOOKBACK_DAYS} days. That is the normal case — most "
            "companies go months without one — not a mark against the names."
        )
        return

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Ticker": s.ticker,
                    "Company": s.company,
                    "Score": round(s.score, 2),
                    "Bought": _money(s.buy_dollars, ""),
                    "Insiders": len(s.buyers),
                    "Cluster": "yes" if s.is_cluster else "",
                    "Roles": ", ".join(s.roles),
                    "Pre-scheduled": s.scheduled_buys,
                    "Latest": s.latest_trade_date.isoformat(),
                }
                for s in buying
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.success(
        "Cross-confirmed: "
        + ", ".join(s.ticker for s in buying)
        + " — bought by both elected officials and the companies' own insiders."
    )
    st.caption(
        "Three or more insiders at one company buying inside a fortnight is "
        "the strongest pattern in this data. Purchases marked pre-scheduled "
        "were arranged months ahead under a Rule 10b5-1 plan, so they are "
        "discounted in the score."
    )


def _rank_with_sidebar_model(scan, models: ModelChoice):
    """Run the ranking call with the sidebar's provider, model and token.

    Any captured provider token lives in the sidebar's env mapping rather than
    the process environment, so it is applied around this one in-process call
    and removed again -- the spawned runs get it through their child env.
    """
    insider_scan = st.session_state.get("insider_scan")
    with _temporary_env(models.run_env):
        return congress.rank_top3(
            scan,
            config={
                "llm_provider": models.llm_provider,
                "deep_think_llm": models.deep_model,
                "backend_url": models.llm_base_url,
            },
            insider_evidence=(
                insiders.render_evidence(insider_scan) if insider_scan else ""
            ),
        )


@contextmanager
def _temporary_env(values: dict[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _congress_analysis_section(ranking, models: ModelChoice):
    """Offer, launch and report the full analyst run over the picked tickers."""
    if not ranking.tickers:
        st.info(
            "The model answered as free text rather than a structured pick, so "
            "there is no ticker list to analyse. Rank again to retry."
        )
        return

    output_dir = (
        Path(models.runs_base).expanduser() / "congress" / models.analysis_date
    )
    st.markdown(
        f"**Run the full analyst team on {', '.join(ranking.tickers)}?** "
        f"That is one complete debate per ticker at *{models.depth}* effort — "
        f"{len(ranking.tickers)}x the cost and runtime of a single `analyze`. "
        f"Change the effort and models in the sidebar first if you want."
    )
    if st.button("Run full analysis on these picks", key="congress_analyze_btn"):
        try:
            pid = launcher.launch(
                launcher.build_watchlist_command(
                    list(ranking.tickers),
                    output_dir=output_dir,
                    analysis_date=models.analysis_date,
                    quick_model=models.quick_model,
                    deep_model=models.deep_model,
                    depth=models.depth,
                    effort=models.effort,
                    llm_provider=models.llm_provider,
                    llm_base_url=models.llm_base_url,
                ),
                output_dir,
                env=models.run_env,
                log_name="congress_screener.log",
            )
            st.success(f"Started the analyst run (PID {pid}). Reload to see results.")
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))

    results = (read_json(output_dir / "recommendations.json") or {}).get("results", [])
    for item in results:
        with st.expander(
            f"{_icon(item.get('status'))} {item.get('ticker', '?')} — "
            f"{item.get('rating', 'n/a')}"
        ):
            st.markdown(_md(item.get("executive_summary") or "No summary."))
            if item.get("report_path"):
                st.caption(f"Report: {item['report_path']}")

    log = output_dir / "congress_screener.log"
    if log.exists():
        with st.expander("Analyst run log"):
            st.text(log.read_text(encoding="utf-8")[-8000:])


def main():
    st.title("Portfolio Intelligence Dashboard")
    run_dir, models = sidebar_controls()
    if run_dir is None:
        st.info("No runs yet. Configure a source and click Run.")
        # The screener reads public filings, not your portfolio, so it is the
        # one thing worth offering before any run exists.
        congress_tab(models)
        return
    manifest = load_manifest(run_dir) or {}
    snapshot = read_json(run_dir / PORTFOLIO_SNAPSHOT) or {}
    forecasts = read_json(run_dir / FORECASTS) or {}

    tabs = st.tabs(
        [
            "Overview",
            "Positions",
            "Forecasts",
            "Portfolio risk",
            "Analysis & runs",
            "Congress screener",
        ]
    )
    with tabs[0]:
        overview_tab(run_dir, manifest, snapshot, forecasts)
    with tabs[1]:
        positions_tab(snapshot)
    with tabs[2]:
        forecasts_tab(snapshot, forecasts)
    with tabs[3]:
        risk_tab(snapshot, forecasts)
    with tabs[4]:
        analysis_tab(run_dir, snapshot)
    with tabs[5]:
        congress_tab(models)


if __name__ == "__main__":
    main()
