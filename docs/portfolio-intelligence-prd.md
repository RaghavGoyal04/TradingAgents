# Portfolio Intelligence Dashboard - PRD

Status: implementation reference. This document is the reviewed product/technical
spec for the local, read-only Trading212 portfolio intelligence dashboard. It
mirrors the accepted plan and the staff design review, and defines the versioned
artifact contracts the code and tests depend on.

## 1. Goal

Give a single local, read-only view of the live Trading212 portfolio that:

- shows current positions, cash, and account-currency allocation;
- reconciles Yahoo Finance (primary) with Alpha Vantage (validation/fallback);
- runs the existing TradingAgents analysis per holding and shows the rationale;
- produces portfolio forecasts and correlated scenarios (loss probability,
  VaR/CVaR, drawdown, per-position risk contribution) in the account currency;
- resumes safely after interruption and keeps the machine awake while running.

Non-goals: order placement, any Trading212 write endpoint, cloud deployment,
multi-user auth, and treating model output as executed trades.

## 2. Users and runtime

- Single user, personal machine (macOS), localhost only.
- LLM analysis uses the already-configured AWS Bedrock profile; no AWS hosting.
- Market data: Yahoo Finance primary, Alpha Vantage for cross-checks/fallback.
  Alpha Vantage free tier is capped (~25 requests/day), so it is used sparingly
  and cached once per day.

## 3. Blocking constraints (from staff review)

1. Live positions drive the analyzed universe. A `--trading212` run must analyze
   the actual non-zero holdings plus a clearly separate optional watchlist, not a
   hardcoded list.
2. One canonical, versioned run contract coordinates all artifacts and the three
   resume layers (orchestrator, per-ticker LangGraph checkpoint, UI).
3. Every portfolio-level number is in the account currency, includes cash, and
   converts local-listing prices through dated FX rates.

## 4. Behavior

### 4.1 Ingestion
- Fetch account summary and positions via GET-only Trading212 endpoints.
- Build `portfolio_snapshot.json` with, per position: broker ticker, resolved
  Yahoo symbol, quantity (when available), current value, currency, and mapping
  status (`mapped`, `unmapped`).
- Include cash (`totalValue - sum(position values)`), account currency, and the
  optional watchlist entries (flagged `watch_only`).
- If a non-zero position cannot be mapped to a market symbol, it is skipped from
  analysis (never guessed), kept visible in the snapshot's `unmapped` list, and
  surfaced as a prominent warning in the UI. It still counts toward account
  value/cash. A wrong symbol is never substituted. Zero-value positions are
  excluded from analysis unless explicitly added to the watchlist.

### 4.2 Market data reconciliation
- Historical adjusted OHLCV comes from Yahoo via the existing `load_ohlcv`.
- For each analyzed symbol, record `yahoo_close`, `alpha_vantage_close`,
  `pct_diff`, a `status` (`ok`, `diverged`, `stale`, `yahoo_only`,
  `unavailable`), and source timestamps. Divergence beyond tolerance or staleness
  is surfaced in the UI and never silently substituted.
- Alpha Vantage failures (missing key, rate limit) degrade gracefully to
  `yahoo_only` with a visible flag.

### 4.3 FX / base currency
- Base currency is the Trading212 account currency.
- Convert each listing's local-currency price series to base currency using a
  dated FX series (Yahoo `<PAIR>=X`). Record the FX pair, rate, and as-of date.
- All risk/forecast aggregation happens in base currency after conversion.

### 4.4 Agents
- Reuse the existing subprocess worker and per-ticker SQLite checkpoints.
- Per-ticker artifact exposes rating, executive summary, report path, model IDs,
  and duration.

### 4.5 Forecast and risk
- Baseline (always available): correlated block bootstrap over the aligned,
  FX-adjusted return panel. Preserves joint market shocks (correlated crashes).
- Candidate: TimesFM, loaded once per portfolio run in a single process, never
  per ticker worker.
- Promotion gate: TimesFM outputs are only promoted to "actionable" per horizon
  when leakage-safe rolling-origin evaluation shows the primary score (pinball /
  quantile loss) improves by >= 5% on >= 80% of eligible holdings, each with
  >= 252 valid observations. Otherwise TimesFM is labeled experimental and the
  winning baseline drives next steps.
- Horizons: 1, 5, and 20 trading days. Scenario count: 10,000 for the 20-day
  portfolio distribution.
- Outputs recorded to `forecasts.json`: model used, data cutoff, metrics,
  warnings, per-asset quantiles, and portfolio P&L distribution.

### 4.6 Run lifecycle and resume
- A single orchestrator process owns a run directory and writes
  `run_manifest.json` (schema-versioned) plus an append-only `events.jsonl`.
- Phases: `ingest`, `agents`, `forecast`. Each has a status and timestamps.
- An exclusive PID lock prevents two orchestrators from writing the same run.
- Resume reuses completed ticker results and valid checkpoints; it never starts
  duplicate workers.
- Interruption model: `caffeinate` prevents idle sleep while a run is active.
  Closing the lid still stops execution unless macOS clamshell conditions are
  met; wake-time resume is the guaranteed behavior.

### 4.7 UI (Streamlit, localhost only)
- Overview: account value, cash, allocation, concentration, latest run status,
  data-health badges, prioritized next actions.
- Positions: holdings + watchlist table with weights, T212/Yahoo/Alpha marks,
  timestamps, and discrepancy/mapping warnings.
- Forecasts: per-asset history with P10/P50/P90 bands, horizon/model selector,
  backtest score, and the evidence source (baseline vs TimesFM).
- Portfolio risk: P&L fan/distribution, loss probabilities, VaR/CVaR, drawdown,
  correlation, and risk contribution.
- Analysis and runs: agent rationale/full reports, phase/ticker progress, logs,
  a Resume control, and explicit partial/stale states.
- The UI only reads artifacts and launches/resumes the orchestrator. It never
  runs analysis in the Streamlit process.

## 5. Security and side effects
- Trading212 client enforces a GET-only endpoint allowlist; no order path exists.
- Secrets come from environment variables; never written into artifacts or logs.
- Streamlit binds to `127.0.0.1`.
- Portfolio artifacts and holdings snapshots are git-ignored.

## 6. Artifact contracts (schema v1)

All artifacts include `schema_version`. Writers use atomic temp-file replace.

### 6.1 `portfolio_snapshot.json`
```
{
  "schema_version": "1.0.0",
  "captured_at": "<iso8601>",
  "source": "trading212" | "holdings_file",
  "base_currency": "GBP",
  "account_value": 6060.51,
  "cash": 123.45,
  "positions": [
    {
      "broker_ticker": "NVDA_US_EQ",
      "symbol": "NVDA",
      "quantity": 3.0 | null,
      "value": 597.82,
      "currency": "GBP",
      "weight": 0.0986,
      "mapping_status": "mapped" | "unmapped",
      "watch_only": false
    }
  ],
  "unmapped": ["<broker_ticker>", ...],
  "watchlist": ["AAPL", ...]
}
```

### 6.2 `run_manifest.json`
```
{
  "schema_version": "1.0.0",
  "run_id": "<uuid>",
  "created_at": "<iso8601>",
  "updated_at": "<iso8601>",
  "analysis_date": "YYYY-MM-DD",
  "portfolio_hash": "<sha256-16>",
  "settings_fingerprint": "<sha256-16>",
  "status": "pending|running|complete|failed|interrupted",
  "phases": {
    "ingest":   {"status": "...", "started_at": "...", "updated_at": "..."},
    "agents":   {"status": "...", "started_at": "...", "updated_at": "...",
                  "tickers": {"NVDA": "success", "GOOG": "pending", ...}},
    "forecast": {"status": "...", "started_at": "...", "updated_at": "..."}
  },
  "artifacts": {
    "portfolio_snapshot": "portfolio_snapshot.json",
    "recommendations": "agents/recommendations.json",
    "forecasts": "forecasts.json"
  },
  "settings": { "quick_model": "...", "deep_model": "...", "base_currency": "..." }
}
```

### 6.3 `forecasts.json`
```
{
  "schema_version": "1.0.0",
  "generated_at": "<iso8601>",
  "base_currency": "GBP",
  "data_cutoff": "YYYY-MM-DD",
  "horizons": [1, 5, 20],
  "model_used": "bootstrap_v1" | "timesfm_v3",
  "evaluation": {
    "1":  {"promoted": false, "winner": "bootstrap_v1", "metrics": {...}},
    "5":  {...}, "20": {...}
  },
  "assets": {
    "NVDA": {"history_tail": [...], "quantiles": {"20": {"p10": ..., "p50": ..., "p90": ...}}}
  },
  "portfolio": {
    "horizon": 20, "n_paths": 10000, "currency": "GBP",
    "pnl": {"p5": ..., "p50": ..., "p95": ...},
    "loss_probability": 0.34,
    "var_95": ..., "cvar_95": ..., "max_drawdown_p50": ...,
    "risk_contribution": {"NVDA": 0.21, ...},
    "warnings": [...]
  }
}
```

## 6.4 Running it

Install the optional UI (and, if wanted, the forecast candidate):

```
pip install -e ".[dashboard]"          # Streamlit UI + charts
pip install -e ".[dashboard,forecast]" # also install TimesFM (heavy)
```

Set `TRADING212_API_KEY`, `TRADING212_API_SECRET`, and optionally
`ALPHA_VANTAGE_API_KEY` in `.env` (see `.env.example`).

Launch the dashboard (localhost only) and drive runs from its sidebar:

```
streamlit run dashboard/app.py --server.address 127.0.0.1
```

Or run/resume headless (the same process the UI spawns; `caffeinate` is applied
automatically on macOS):

```
python -m tradingagents.portfolio.orchestrator --run-dir <dir> --trading212
python -m tradingagents.portfolio.orchestrator --run-dir <dir> --trading212 --resume
```

`--holdings holdings.json` substitutes a static holdings file for live
Trading212; `--skip-agents` runs forecasts only.

## 7. Acceptance criteria
- One Run starts exactly one orchestrator; a second launch is rejected while the
  lock is held. Killing mid-run and resuming reuses completed ticker results and
  valid checkpoints without duplicate workers.
- Every non-zero position is either mapped and analyzed, or skipped-and-visible
  with a warning (never analyzed under a guessed symbol). Zero-value positions
  appear only via the watchlist.
- Every portfolio metric is in account currency and includes cash. Stale,
  rate-limited, or conflicting market data is visible, never silently replaced.
- Tests prove: no look-ahead leakage in evaluation, atomic artifact writes,
  portfolio weight/scenario invariants, TimesFM graceful fallback, and a strict
  Trading212 GET-endpoint allowlist.
- `pytest`, Ruff, a mocked end-to-end run, and a manual macOS interrupt/resume
  smoke test pass before the dashboard is called ready.
