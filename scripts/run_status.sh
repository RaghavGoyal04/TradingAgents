#!/usr/bin/env bash
# Compact progress view for an in-flight portfolio orchestrator run.
# Usage: scripts/run_status.sh [run_dir]
set -uo pipefail
RUN="${1:-$HOME/.tradingagents/logs/portfolio/$(date +%F)}"

echo "run: $RUN"
if pgrep -f "portfolio.orchestrator" >/dev/null; then
  echo "orchestrator: RUNNING (elapsed $(ps -o etime= -p "$(pgrep -f 'portfolio.orchestrator' | head -1)" | tr -d ' '))"
else
  echo "orchestrator: NOT RUNNING"
fi

python3 - "$RUN" <<'PY'
import json, os, sys, pathlib
run = pathlib.Path(sys.argv[1])
m = run / "run_manifest.json"
if m.exists():
    d = json.loads(m.read_text())
    print("status:", d.get("status"))
    print("phases:", ", ".join(f"{k}={v['status']}" for k, v in d.get("phases", {}).items()))
    tick = d.get("tickers") or {}
    if tick:
        from collections import Counter
        print("tickers:", dict(Counter(tick.values())))
recs = run / "agents" / "recommendations.json"
if recs.exists():
    r = json.loads(recs.read_text()).get("results", [])
    from collections import Counter
    print(f"recommendations: {len(r)} ->", dict(Counter(x.get("action") or x.get("rating") for x in r)))
PY

LOGS="$RUN/agents/logs"
if [ -d "$LOGS" ]; then
  total=$(ls "$LOGS"/*.log 2>/dev/null | wc -l | tr -d ' ')
  done_n=$(grep -lc "" "$LOGS"/*.log 2>/dev/null | wc -l | tr -d ' ')
  nonempty=$(find "$LOGS" -name '*.log' -size +0 2>/dev/null | wc -l | tr -d ' ')
  echo "ticker logs: $total started, $nonempty with output"
fi
echo "errors so far: $(grep -ric "throttl\|accessdenied\|expiredtoken\|traceback" "$RUN/orchestrator.log" 2>/dev/null || echo 0)"
tail -3 "$RUN/orchestrator.log" 2>/dev/null
