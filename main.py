"""Example entrypoint: loads ``.env`` and runs ``TradingAgentsGraph``.

If ``DATABRICKS_TOKEN``, ``DEEP_THINK_LLM``, and ``QUICK_THINK_LLM`` are set
(endpoint names require ``DATABRICKS_HOST``; full URLs embed the workspace),
Databricks serving is used; otherwise the fallback OpenAI-style block applies.
"""

import os

from dotenv import load_dotenv

from tradingagents.env_utils import clean_env_value
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

load_dotenv()

config = DEFAULT_CONFIG.copy()

if (
    clean_env_value(os.environ.get("DATABRICKS_TOKEN"))
    and clean_env_value(os.environ.get("DEEP_THINK_LLM"))
    and clean_env_value(os.environ.get("QUICK_THINK_LLM"))
):
    from tradingagents.databricks_connecting import build_tradingagents_config_from_env

    config.update(build_tradingagents_config_from_env())
else:
    config["deep_think_llm"] = "gpt-5.4-mini"
    config["quick_think_llm"] = "gpt-5.4-mini"
    config["llm_provider"] = "openai"

config["max_debate_rounds"] = 1

config["data_vendors"] = {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}

ta = TradingAgentsGraph(debug=True, config=config)

_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns

