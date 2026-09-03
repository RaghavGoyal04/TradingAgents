"""Portfolio intelligence: Trading212 ingestion, run manifest, FX, orchestration.

This package powers the local, read-only portfolio dashboard. It reuses the
existing TradingAgents data and agent pipeline and adds a versioned artifact
contract so the orchestrator, per-ticker checkpoints, and the UI share one
source of truth. See docs/portfolio-intelligence-prd.md.
"""

SCHEMA_VERSION = "1.0.0"
