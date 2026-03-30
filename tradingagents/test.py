"""Smoke test Databricks serving endpoints using the same .env as TradingAgents.

Set in ``.env`` (see ``.env.example``):

- ``DATABRICKS_TOKEN``
- ``DATABRICKS_HOST`` — workspace URL (required if ``DEEP_THINK_LLM`` / ``QUICK_THINK_LLM``
  are endpoint names only)
- ``DEEP_THINK_LLM`` — full URL *or* endpoint name, e.g.
  ``https://wbd-dcp-odp-dev.cloud.databricks.com/serving-endpoints/databricks-claude-opus-4-6/invocations``
  or ``databricks-claude-opus-4-6``
- ``QUICK_THINK_LLM`` — same pattern for the quick model

Run: ``python tradingagents/test.py`` (from repo root, with venv active).
"""

from __future__ import annotations

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

from tradingagents.databricks_connecting import (
    resolve_endpoint_spec,
    serving_invocation_url,
)
from tradingagents.env_utils import clean_env_value


def _invocation_urls_from_env() -> tuple[str, str, str]:
    token = clean_env_value(os.environ.get("DATABRICKS_TOKEN"))
    if not token:
        print("Set DATABRICKS_TOKEN in .env", file=sys.stderr)
        sys.exit(2)

    dh = clean_env_value(os.environ.get("DATABRICKS_HOST"))
    deep_spec = clean_env_value(os.environ.get("DEEP_THINK_LLM"))
    quick_spec = clean_env_value(os.environ.get("QUICK_THINK_LLM"))
    if not deep_spec or not quick_spec:
        print("Set DEEP_THINK_LLM and QUICK_THINK_LLM in .env", file=sys.stderr)
        sys.exit(2)

    ws_d, deep_name = resolve_endpoint_spec(deep_spec, dh)
    ws_q, quick_name = resolve_endpoint_spec(quick_spec, dh)
    if ws_d.rstrip("/") != ws_q.rstrip("/"):
        print(
            "DEEP_THINK_LLM and QUICK_THINK_LLM must use the same workspace",
            file=sys.stderr,
        )
        sys.exit(2)

    workspace = ws_d
    deep_url = serving_invocation_url(workspace, deep_name)
    quick_url = serving_invocation_url(workspace, quick_name)
    return token, deep_url, quick_url


def main() -> None:
    token, deep_url, quick_url = _invocation_urls_from_env()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    data = {"messages": [{"role": "user", "content": "Hello, model!"}]}

    print("Quick think POST:", quick_url)
    rq = requests.post(quick_url, headers=headers, json=data, timeout=120)
    print("quick status:", rq.status_code)
    print("quick body:", rq.json())

    print()
    print("Deep think POST:", deep_url)
    rd = requests.post(deep_url, headers=headers, json=data, timeout=200)
    print("deep status:", rd.status_code)
    print("deep body:", rd.json())


if __name__ == "__main__":
    main()
