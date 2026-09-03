"""Dashboard launch helpers: command building and active-run detection."""

import os

import pytest

from tradingagents.portfolio import launcher


@pytest.mark.unit
def test_build_command_trading212():
    cmd = launcher.build_orchestrator_command(
        "/runs/today",
        source="trading212",
        watchlist=["AAPL", "MSFT"],
        analysis_date="2026-09-03",
        quick_model="q",
        deep_model="d",
    )
    assert "-m" in cmd and "tradingagents.portfolio.orchestrator" in cmd
    assert "--trading212" in cmd
    assert cmd[cmd.index("--watchlist") + 1 : cmd.index("--watchlist") + 3] == ["AAPL", "MSFT"]


@pytest.mark.unit
def test_build_command_holdings_requires_path():
    with pytest.raises(ValueError, match="holdings"):
        launcher.build_orchestrator_command(
            "/runs/x", source="holdings", quick_model="q", deep_model="d"
        )


@pytest.mark.unit
def test_build_command_rejects_unknown_source():
    with pytest.raises(ValueError, match="unknown source"):
        launcher.build_orchestrator_command(
            "/runs/x", source="bogus", quick_model="q", deep_model="d"
        )


@pytest.mark.unit
def test_resume_flag_added():
    cmd = launcher.build_orchestrator_command(
        "/runs/x", source="trading212", quick_model="q", deep_model="d", resume=True
    )
    assert "--resume" in cmd


@pytest.mark.unit
def test_is_run_active_true_for_live_pid(tmp_path):
    (tmp_path / ".orchestrator.lock").write_text(str(os.getpid()), encoding="utf-8")
    assert launcher.is_run_active(tmp_path) is True


@pytest.mark.unit
def test_is_run_active_false_for_dead_pid(tmp_path):
    (tmp_path / ".orchestrator.lock").write_text("999999", encoding="utf-8")
    assert launcher.is_run_active(tmp_path) is False


@pytest.mark.unit
def test_is_run_active_false_without_lock(tmp_path):
    assert launcher.is_run_active(tmp_path) is False


@pytest.mark.unit
def test_launch_rejected_when_active(tmp_path):
    (tmp_path / ".orchestrator.lock").write_text(str(os.getpid()), encoding="utf-8")
    with pytest.raises(RuntimeError, match="already active"):
        launcher.launch(["echo"], tmp_path)


@pytest.mark.unit
def test_timesfm_flag_is_opt_in(tmp_path):
    """TimesFM is expensive, so it must only appear when explicitly requested."""
    common = {"source": "trading212", "quick_model": "q", "deep_model": "d"}
    assert "--timesfm" not in launcher.build_orchestrator_command(
        tmp_path / "run", **common
    )
    assert "--timesfm" in launcher.build_orchestrator_command(
        tmp_path / "run", use_timesfm=True, **common
    )


@pytest.mark.unit
def test_depth_is_forwarded_to_the_runner(tmp_path):
    common = {"source": "trading212", "quick_model": "q", "deep_model": "d"}
    cmd = launcher.build_orchestrator_command(tmp_path / "r", depth="deep", **common)
    assert cmd[cmd.index("--depth") + 1] == "deep"
    # Defaults to the cheapest profile.
    assert "shallow" in launcher.build_orchestrator_command(tmp_path / "r", **common)


@pytest.mark.unit
def test_alternate_llm_provider_is_opt_in(tmp_path):
    """Databricks / any OpenAI-compatible gateway without touching the code."""
    common = {"source": "trading212", "quick_model": "q", "deep_model": "d"}
    assert "--llm-provider" not in launcher.build_orchestrator_command(
        tmp_path / "r", **common
    )
    cmd = launcher.build_orchestrator_command(
        tmp_path / "r",
        llm_provider="openai_compatible",
        llm_base_url="https://example.cloud.databricks.com/serving-endpoints",
        **common,
    )
    assert cmd[cmd.index("--llm-provider") + 1] == "openai_compatible"
    assert "databricks" in cmd[cmd.index("--llm-base-url") + 1]


@pytest.mark.unit
def test_watchlist_command_carries_an_explicit_ticker_list(tmp_path):
    """Screener candidates come from filings, not a portfolio, so no source."""
    cmd = launcher.build_watchlist_command(
        ["BE", "INTC", "FWONK"],
        output_dir=tmp_path / "congress",
        quick_model="q",
        deep_model="d",
        analysis_date="2026-09-03",
        depth="medium",
    )
    assert cmd[1].endswith("run_watchlist.py")
    assert cmd[cmd.index("--tickers") + 1 : cmd.index("--tickers") + 4] == [
        "BE",
        "INTC",
        "FWONK",
    ]
    assert cmd[cmd.index("--depth") + 1] == "medium"
    assert "--trading212" not in cmd and "--holdings" not in cmd


@pytest.mark.unit
def test_watchlist_command_needs_at_least_one_ticker(tmp_path):
    with pytest.raises(ValueError, match="at least one ticker"):
        launcher.build_watchlist_command(
            [], output_dir=tmp_path, quick_model="q", deep_model="d"
        )


@pytest.mark.unit
def test_launch_log_name_is_overridable(tmp_path):
    """Two launchers write to one run directory; their logs must not collide."""

    class FakePopen:
        def __init__(self, command, **kwargs):
            self.pid = 7
            kwargs["stdout"].write("hello")

    launcher.launch(
        ["echo"], tmp_path, popen=FakePopen, log_name="congress_screener.log"
    )
    assert (tmp_path / "congress_screener.log").exists()
    assert not (tmp_path / "orchestrator.log").exists()


@pytest.mark.unit
def test_provider_token_never_appears_in_argv(tmp_path):
    """A token in the command line is readable by any process via ps."""
    captured = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["env"] = kwargs.get("env")
            self.pid = 4242

    cmd = launcher.build_orchestrator_command(
        tmp_path / "r",
        source="trading212",
        quick_model="databricks-gpt-5-5",
        deep_model="databricks-gpt-5-6-sol",
        llm_provider="openai_compatible",
        llm_base_url="https://example.databricks.com/serving-endpoints",
        effort="xhigh",
    )
    launcher.launch(
        cmd, tmp_path / "r", env={"OPENAI_COMPATIBLE_API_KEY": "sekrit"}, popen=FakePopen
    )
    assert "sekrit" not in " ".join(captured["command"])
    assert captured["env"]["OPENAI_COMPATIBLE_API_KEY"] == "sekrit"
    # The child still inherits the ambient environment.
    assert "PATH" in captured["env"]
    assert cmd[cmd.index("--effort") + 1] == "xhigh"


@pytest.mark.unit
def test_concurrency_is_forwarded_to_the_runner(tmp_path):
    """It was never forwarded, so the runner always used its default of 6 and
    the dashboard had no way to change it."""
    common = {"source": "trading212", "quick_model": "q", "deep_model": "d"}
    cmd = launcher.build_orchestrator_command(tmp_path / "r", concurrency=19, **common)
    assert cmd[cmd.index("--concurrency") + 1] == "19"
    default = launcher.build_orchestrator_command(tmp_path / "r", **common)
    assert default[default.index("--concurrency") + 1] == "6"


def test_force_flag_is_opt_in():
    """Without --force a re-run silently replays today's completed tickers."""
    default = launcher.build_orchestrator_command(
        "/tmp/run", source="trading212", quick_model="q", deep_model="d"
    )
    assert "--force" not in default

    forced = launcher.build_orchestrator_command(
        "/tmp/run", source="trading212", quick_model="q", deep_model="d", force=True
    )
    assert "--force" in forced
