"""Databricks notebook / job helpers for TradingAgents + Foundation Model serving.

Databricks exposes OpenAI-compatible chat completions at::

    {workspace_url}/serving-endpoints

Authentication uses a personal access token (PAT) or token from the notebook
context. The *model* names you pass to TradingAgents must be your **serving
endpoint names** (or FM API endpoint identifiers as shown in the workspace).

Typical notebook usage::

    from tradingagents.databricks_connecting import (
        build_tradingagents_config,
        configure_databricks_llm_environment,
    )
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG

    configure_databricks_llm_environment()  # uses dbutils from the notebook
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(
        build_tradingagents_config(
            quick_think_endpoint="my-quick-endpoint",
            deep_think_endpoint="my-deep-endpoint",
        )
    )
    ta = TradingAgentsGraph(config=cfg)
    _, decision = ta.propagate("SPY", "2026-03-23")

Jobs (no dbutils): set ``DATABRICKS_TOKEN`` and ``DATABRICKS_HOST`` (workspace URL,
e.g. ``https://dbc-xxxxx.cloud.databricks.com``), then call
``build_tradingagents_config(...)`` without ``configure_*`` first — it will read
from the environment.

If you already have a PAT, pass it explicitly (never commit it to git)::

    configure_databricks_llm_environment(
        token="dapixxxx",
        workspace_url="https://dbc-xxxxx.cloud.databricks.com",
    )

**Environment variables (local / Jobs):** use ``export`` so child processes (e.g. ``python``)
see them::

    export DATABRICKS_TOKEN='dapixxxxxxxxx'
    export DATABRICKS_HOST='https://dbc-xxxxx.cloud.databricks.com'

**CLI smoke test** (after ``export``, or a ``.env`` file if ``python-dotenv`` is installed)::

    python tradingagents/databricks_connecting.py --env-check
    python tradingagents/databricks_connecting.py --endpoint your-serving-endpoint-name

**Config from ``.env``** (``DEEP_THINK_LLM`` / ``QUICK_THINK_LLM`` as full URLs or endpoint names)::

    from tradingagents.databricks_connecting import build_tradingagents_config_from_env
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tradingagents.default_config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG.copy()
    cfg.update(build_tradingagents_config_from_env())
    ta = TradingAgentsGraph(config=cfg)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from tradingagents.env_utils import clean_env_value


def normalize_workspace_url(url: str) -> str:
    """Return ``https://...`` workspace URL with no trailing slash."""
    u = (url or "").strip()
    if not u:
        raise ValueError("workspace URL is empty")
    if not u.startswith("http"):
        u = "https://" + u.lstrip("/")
    return u.rstrip("/")


def serving_endpoints_openai_base_url(workspace_url: str) -> str:
    """OpenAI-compatible base URL for Databricks model serving (chat completions)."""
    return f"{normalize_workspace_url(workspace_url)}/serving-endpoints"


def parse_databricks_serving_url(url: str) -> Tuple[str, str]:
    """Parse workspace URL and endpoint name from a full serving URL.

    Accepts URLs like
    ``https://host/serving-endpoints/<endpoint_name>/invocations`` or
    ``https://host/serving-endpoints/<endpoint_name>``.
    """
    from urllib.parse import urlparse

    p = urlparse((url or "").strip())
    if not p.scheme or not p.netloc:
        raise ValueError(f"Invalid URL: {url!r}")
    workspace = f"{p.scheme}://{p.netloc}"
    parts = [seg for seg in p.path.strip("/").split("/") if seg]
    try:
        i = parts.index("serving-endpoints")
        name = parts[i + 1]
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"Expected path .../serving-endpoints/<endpoint_name>/..., got: {url!r}"
        ) from e
    return normalize_workspace_url(workspace), name


def serving_invocation_url(workspace_url: str, endpoint_name: str) -> str:
    """REST URL for ``POST .../invocations`` (e.g. ``requests`` smoke tests)."""
    return (
        f"{normalize_workspace_url(workspace_url)}/serving-endpoints/"
        f"{endpoint_name.strip()}/invocations"
    )


def resolve_endpoint_spec(spec: str, fallback_workspace_url: str) -> Tuple[str, str]:
    """Return ``(workspace_url, endpoint_name)``.

    If ``spec`` is an ``http`` URL, it is parsed. Otherwise ``spec`` is treated
    as an endpoint name and ``fallback_workspace_url`` (``DATABRICKS_HOST``) must
    be set.
    """
    s = (spec or "").strip()
    if not s:
        raise ValueError("endpoint spec is empty")
    if s.startswith("http"):
        return parse_databricks_serving_url(s)
    if not (fallback_workspace_url or "").strip():
        raise ValueError(
            "DATABRICKS_HOST is required when DEEP_THINK_LLM / QUICK_THINK_LLM "
            "are endpoint names only (not full URLs)."
        )
    return normalize_workspace_url(fallback_workspace_url), s


def build_tradingagents_config_from_env() -> Dict[str, Any]:
    """Build ``TradingAgentsGraph`` config from environment / ``.env``.

    Required:

    - ``DATABRICKS_TOKEN``
    - ``DEEP_THINK_LLM`` — full serving URL **or** endpoint name
    - ``QUICK_THINK_LLM`` — full serving URL **or** endpoint name

    ``DATABRICKS_HOST`` (workspace URL) is required when the think vars are plain
    endpoint names; optional if both are full URLs (workspace must still match).

    Sets ``DATABRICKS_TOKEN`` and ``DATABRICKS_HOST`` in ``os.environ``.
    """
    _maybe_load_dotenv()
    token = clean_env_value(os.environ.get("DATABRICKS_TOKEN"))
    if not token:
        raise ValueError("DATABRICKS_TOKEN is required")

    dh = clean_env_value(os.environ.get("DATABRICKS_HOST"))
    deep_spec = clean_env_value(os.environ.get("DEEP_THINK_LLM"))
    quick_spec = clean_env_value(os.environ.get("QUICK_THINK_LLM"))
    if not deep_spec or not quick_spec:
        raise ValueError("DEEP_THINK_LLM and QUICK_THINK_LLM are required")

    ws_deep, deep_name = resolve_endpoint_spec(deep_spec, dh)
    ws_quick, quick_name = resolve_endpoint_spec(quick_spec, dh)
    if normalize_workspace_url(ws_deep) != normalize_workspace_url(ws_quick):
        raise ValueError(
            "DEEP_THINK_LLM and QUICK_THINK_LLM must refer to the same Databricks workspace."
        )

    ws = normalize_workspace_url(ws_deep)
    os.environ["DATABRICKS_TOKEN"] = token
    os.environ["DATABRICKS_HOST"] = ws
    return {
        "llm_provider": "databricks",
        "backend_url": serving_endpoints_openai_base_url(ws),
        "deep_think_llm": deep_name,
        "quick_think_llm": quick_name,
    }


def get_notebook_dbutils() -> Any:
    """Resolve ``dbutils`` when this module is imported from a Databricks notebook."""
    try:
        from IPython import get_ipython  # type: ignore[import-untyped]

        shell = get_ipython()
        if shell is None:
            return None
        return shell.user_ns.get("dbutils")
    except Exception:
        return None


def get_pat_and_workspace_url(
    dbutils: Any = None,
    *,
    token: Optional[str] = None,
    workspace_url: Optional[str] = None,
) -> Tuple[str, str]:
    """Return ``(pat, workspace_url)``.

    Resolution order:

    1. Explicit ``token`` and ``workspace_url`` (use your PAT + workspace hostname).
    2. ``dbutils`` notebook context.
    3. Environment variables ``DATABRICKS_TOKEN`` and ``DATABRICKS_HOST``.
    """
    if token is not None and workspace_url is not None:
        if not (token and str(token).strip()):
            raise ValueError("token is empty")
        return str(token).strip(), str(workspace_url).strip()

    if dbutils is None:
        dbutils = get_notebook_dbutils()

    if dbutils is not None:
        ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
        token = ctx.apiToken().get()
        workspace_url = ctx.workspaceUrl().get()
        if not token or not workspace_url:
            raise RuntimeError(
                "Notebook context returned empty api token or workspace URL."
            )
        return str(token), str(workspace_url)

    token = clean_env_value(os.environ.get("DATABRICKS_TOKEN"))
    host = clean_env_value(os.environ.get("DATABRICKS_HOST"))
    if token and host:
        return token, host

    raise RuntimeError(
        "Could not resolve Databricks credentials: pass token= and workspace_url=, "
        "or dbutils= from the notebook, or set DATABRICKS_TOKEN and DATABRICKS_HOST."
    )


def configure_databricks_llm_environment(
    dbutils: Any = None,
    *,
    token: Optional[str] = None,
    workspace_url: Optional[str] = None,
) -> Dict[str, str]:
    """Set ``DATABRICKS_TOKEN`` and ``DATABRICKS_HOST`` for the current process.

    Call this before constructing ``TradingAgentsGraph`` with ``llm_provider``
    ``\"databricks\"`` so the OpenAI-compatible client can authenticate.

    Parameters
    ----------
    token, workspace_url
        Optional. Your Databricks PAT and workspace URL (e.g. from the browser
        address bar). When both are set, ``dbutils`` and env vars are ignored.

    Returns a dict with ``token``, ``workspace_url``, and ``openai_base_url`` for
    inspection (do not log the token).
    """
    token, workspace_url = get_pat_and_workspace_url(
        dbutils=dbutils, token=token, workspace_url=workspace_url
    )
    ws = normalize_workspace_url(workspace_url)
    os.environ["DATABRICKS_TOKEN"] = token
    os.environ["DATABRICKS_HOST"] = ws
    return {
        "token": token,
        "workspace_url": ws,
        "openai_base_url": serving_endpoints_openai_base_url(ws),
    }


def build_tradingagents_config(
    quick_think_endpoint: str,
    deep_think_endpoint: str,
    dbutils: Any = None,
    *,
    token: Optional[str] = None,
    workspace_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Build config keys for ``TradingAgentsGraph`` (merge into ``DEFAULT_CONFIG``).

    Parameters
    ----------
    quick_think_endpoint, deep_think_endpoint
        Serving endpoint **names** as shown in Databricks (Serving UI).
    dbutils
        Optional Databricks ``dbutils`` handle.
    token, workspace_url
        Optional PAT and workspace URL if you are not using ``dbutils`` or env vars.
    """
    info = configure_databricks_llm_environment(
        dbutils=dbutils, token=token, workspace_url=workspace_url
    )
    return {
        "llm_provider": "databricks",
        "backend_url": info["openai_base_url"],
        "quick_think_llm": quick_think_endpoint,
        "deep_think_llm": deep_think_endpoint,
    }


def smoke_test_llm(
    endpoint_name: str,
    dbutils: Any = None,
    *,
    token: Optional[str] = None,
    workspace_url: Optional[str] = None,
    prompt: str = "Reply with exactly one word: OK",
) -> str:
    """Send one chat turn via the same stack TradingAgents uses; return text reply.

    Run this in a Databricks notebook after ``%pip install`` / attaching the package::

        from tradingagents.databricks_connecting import smoke_test_llm
        print(smoke_test_llm("your-serving-endpoint-name"))

    On success you get a non-empty model string. On failure you get an exception
    (HTTP 401/403/404, timeout, etc.) with a traceback you can share for debugging.

    Parameters
    ----------
    endpoint_name
        Serving endpoint name in the workspace (same as ``quick_think_llm``).
    dbutils
        Optional; resolved from the notebook if omitted.
    token, workspace_url
        Optional PAT and workspace URL if you already have them.
    prompt
        User message for the smoke test.
    """
    from langchain_core.messages import HumanMessage

    from tradingagents.llm_clients.openai_client import OpenAIClient

    info = configure_databricks_llm_environment(
        dbutils=dbutils, token=token, workspace_url=workspace_url
    )
    client = OpenAIClient(
        model=endpoint_name,
        base_url=info["openai_base_url"],
        provider="databricks",
    )
    llm = client.get_llm()
    out = llm.invoke([HumanMessage(content=prompt)])
    content = getattr(out, "content", None)
    if isinstance(content, str):
        return content.strip()
    return str(content or out)


def check_env_credentials_loaded() -> bool:
    """Return True if ``DATABRICKS_TOKEN`` and ``DATABRICKS_HOST`` are non-empty."""
    return bool(clean_env_value(os.environ.get("DATABRICKS_TOKEN"))) and bool(
        clean_env_value(os.environ.get("DATABRICKS_HOST"))
    )


def _maybe_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _main_cli() -> None:
    import argparse
    import sys

    _maybe_load_dotenv()

    p = argparse.ArgumentParser(
        description="Test Databricks LLM connectivity (uses DATABRICKS_TOKEN + DATABRICKS_HOST)."
    )
    p.add_argument(
        "--env-check",
        action="store_true",
        help="Only verify credentials resolve and print workspace/base URL (no LLM call).",
    )
    p.add_argument(
        "--endpoint",
        type=str,
        default=None,
        metavar="NAME",
        help="Serving endpoint name for smoke_test_llm (required unless --env-check).",
    )
    args = p.parse_args()

    if not check_env_credentials_loaded():
        print(
            "Missing DATABRICKS_TOKEN or DATABRICKS_HOST in the environment.",
            file=sys.stderr,
        )
        print(
            "Export them in the same shell before running Python, e.g.:",
            file=sys.stderr,
        )
        print(
            "  export DATABRICKS_TOKEN='dapixxxxxxxxx'",
            file=sys.stderr,
        )
        print(
            "  export DATABRICKS_HOST='https://dbc-xxxxx.cloud.databricks.com'",
            file=sys.stderr,
        )
        print(
            "Or add them to a .env file in the project root (see python-dotenv).",
            file=sys.stderr,
        )
        sys.exit(2)

    info = configure_databricks_llm_environment()
    if args.env_check:
        print("Databricks connectivity (credentials only):")
        print("  workspace_url:", info["workspace_url"])
        print("  openai_base_url:", info["openai_base_url"])
        print("OK (token not printed).")
        return

    if not args.endpoint:
        p.error("--endpoint NAME is required for a chat smoke test (or use --env-check).")

    try:
        reply = smoke_test_llm(args.endpoint)
    except Exception as e:
        print("LLM smoke test failed:", e, file=sys.stderr)
        sys.exit(1)
    print("Smoke test reply:")
    print(reply)


if __name__ == "__main__":
    _main_cli()