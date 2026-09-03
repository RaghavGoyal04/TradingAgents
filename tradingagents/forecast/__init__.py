"""Portfolio forecasting: correlated block-bootstrap scenarios, an optional
TimesFM candidate, and a leakage-safe walk-forward promotion gate.

The baseline (block bootstrap) is always available and preserves joint market
shocks. TimesFM is only ever promoted to "actionable" per horizon when it beats
the baselines in rolling-origin evaluation; otherwise its output is labeled
experimental. See docs/portfolio-intelligence-prd.md.
"""
