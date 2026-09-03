"""Provider catalog loading and credential capture."""

from __future__ import annotations

import contextlib

import pytest

from tradingagents.portfolio import llm_config


@pytest.fixture
def catalog():
    return llm_config.load_catalog()


@pytest.mark.unit
def test_shipped_catalog_maps_onto_real_tradingagents_providers(catalog):
    """A typo here would only surface as a failed run, so assert the wiring."""
    from tradingagents.llm_clients.factory import create_llm_client  # noqa: F401
    from tradingagents.llm_clients.openai_client import is_openai_compatible

    assert set(catalog) == {"bedrock", "databricks"}
    assert catalog["bedrock"].llm_provider == "bedrock"
    # Databricks is reached through the generic OpenAI-compatible client.
    assert is_openai_compatible(catalog["databricks"].llm_provider)
    for provider in catalog.values():
        assert provider.models, f"{provider.name} lists no models"


@pytest.mark.unit
def test_declared_efforts_are_only_on_models_that_accept_them(catalog):
    """The client silently drops effort for non-reasoning models.

    Declaring an effort the client would discard makes the UI lie, so the
    catalog and the client's own gate must agree.
    """
    from tradingagents.llm_clients.openai_client import _supports_reasoning_effort

    for model in catalog["databricks"].models:
        assert bool(model.efforts) == _supports_reasoning_effort(model.id), model.id
    # Bedrock's Converse client has no effort passthrough at all.
    assert not any(m.efforts for m in catalog["bedrock"].models)


@pytest.mark.unit
def test_databricks_prefixed_reasoning_models_keep_their_effort():
    """Regression: the bare regex dropped effort for gateway-prefixed names."""
    from tradingagents.llm_clients.openai_client import _supports_reasoning_effort

    assert _supports_reasoning_effort("databricks-gpt-5-6-sol")
    assert _supports_reasoning_effort("gpt-5.5")
    assert not _supports_reasoning_effort("databricks-claude-sonnet-4-6")
    assert not _supports_reasoning_effort("gpt-4o")


@pytest.mark.unit
def test_reasoning_effort_reaches_openai_compatible_providers():
    """Databricks-served GPT-5 must get the effort the user picked."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    graph = TradingAgentsGraph.__new__(TradingAgentsGraph)
    graph.config = {
        "llm_provider": "openai_compatible",
        "openai_reasoning_effort": "xhigh",
    }
    assert graph._get_provider_kwargs().get("reasoning_effort") == "xhigh"


@pytest.mark.unit
def test_token_is_read_from_the_databricks_profile(tmp_path, monkeypatch):
    cfg = tmp_path / ".databrickscfg"
    # configparser treats [DEFAULT] as magic; that bug hid the token once.
    cfg.write_text("[DEFAULT]\nhost = https://example.databricks.com/\ntoken = pat-123\n")
    monkeypatch.setattr(llm_config, "DATABRICKS_CONFIG", cfg)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

    host, token = llm_config.databricks_credentials("DEFAULT")
    assert (host, token) == ("https://example.databricks.com", "pat-123")
    assert llm_config.base_url(host) == "https://example.databricks.com/serving-endpoints"


@pytest.mark.unit
def test_environment_overrides_the_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_config, "DATABRICKS_CONFIG", tmp_path / "absent")
    monkeypatch.setenv("DATABRICKS_HOST", "https://env.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "env-token")
    assert llm_config.databricks_credentials()[1] == "env-token"


@pytest.mark.unit
def test_missing_credentials_say_how_to_fix_them(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_config, "DATABRICKS_CONFIG", tmp_path / "absent")
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.setattr(llm_config, "_oauth_token", lambda profile: "")
    with pytest.raises(llm_config.CredentialError, match="databricks auth login"):
        llm_config.databricks_credentials("nope")


@pytest.mark.unit
def test_bedrock_needs_no_captured_secret(catalog):
    """Bedrock rides the AWS credential chain the runner already resolves."""
    assert llm_config.resolve(catalog["bedrock"]) == (None, {})


@pytest.mark.unit
def test_one_effort_setting_reaches_both_clients():
    """The graph builds quick and deep from one shared kwargs dict.

    The UI must not imply the effort is deep-only: a reasoning model in the
    quick slot receives it too, and that slot makes far more calls.
    """
    from unittest.mock import MagicMock, patch

    from tradingagents.default_config import DEFAULT_CONFIG

    seen = []
    config = DEFAULT_CONFIG.copy()
    config.update(
        llm_provider="openai_compatible",
        backend_url="https://example/serving-endpoints",
        quick_think_llm="databricks-gpt-5-5",
        deep_think_llm="databricks-gpt-5-6-sol",
        openai_reasoning_effort="xhigh",
    )
    with patch(
        "tradingagents.graph.trading_graph.create_llm_client",
        side_effect=lambda provider, model, base_url=None, **kw: (
            seen.append((model, kw.get("reasoning_effort"))) or MagicMock()
        ),
    ):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        # Graph wiring past client creation is irrelevant here.
        with contextlib.suppress(Exception):
            TradingAgentsGraph(selected_analysts=("market",), debug=False, config=config)

    assert dict(seen) == {
        "databricks-gpt-5-6-sol": "xhigh",
        "databricks-gpt-5-5": "xhigh",
    }


@pytest.mark.unit
def test_databricks_models_avoid_function_calling_for_structured_output():
    """Databricks chat-completions rejects the function-calling shape.

    GPT-5.x answers "Function tools with reasoning_effort are not supported"
    and the Anthropic endpoints reject parallel_tool_calls. Both 400s were
    swallowed by the free-text fallback, which silently emptied every parsed
    field (price target, executive summary, thesis, horizon) and left the
    dashboard with no conviction to score.
    """
    from tradingagents.llm_clients.capabilities import get_capabilities

    for model in (
        "databricks-gpt-5-6-sol",
        "databricks-claude-sonnet-4-6",
        "databricks-claude-opus-5",
    ):
        caps = get_capabilities(model)
        assert caps.preferred_structured_method == "json_schema", model
        assert not caps.supports_tool_choice, model
    # Non-Databricks models keep the default function-calling path.
    assert get_capabilities("gpt-5.5").preferred_structured_method == "function_calling"


@pytest.mark.unit
def test_rendered_decision_carries_the_labels_the_runner_parses():
    """render_pm_decision output must satisfy run_watchlist.extract_field."""
    from scripts.run_watchlist import extract_field
    from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision

    markdown = render_pm_decision(
        PortfolioDecision(
            rating="Overweight",
            executive_summary="Buy the dip.",
            investment_thesis="Cloud growth intact.",
            price_target=550.0,
            time_horizon="1-3 months",
        )
    )
    assert extract_field(markdown, "Price Target") == "550.0"
    assert extract_field(markdown, "Executive Summary") == "Buy the dip."
    assert extract_field(markdown, "Investment Thesis") == "Cloud growth intact."
    assert extract_field(markdown, "Time Horizon") == "1-3 months"
