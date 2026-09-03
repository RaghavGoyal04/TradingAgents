"""GitHub Copilot SDK provider wiring and LangChain adapter tests."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.validators import validate_model


class _FakeToolResult:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeTool:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeSession:
    def __init__(self, options):
        self.options = options
        self.prompt = None
        self.timeout = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def send_and_wait(self, prompt, *, timeout):
        self.prompt = prompt
        self.timeout = timeout
        content = "copilot response"
        if self.options.get("tools"):
            invocation = SimpleNamespace(arguments={"value": 4})
            result = await self.options["tools"][0].handler(invocation)
            content = result.text_result_for_llm
        return SimpleNamespace(data=SimpleNamespace(content=content))


class _FakeCopilotClient:
    instances = []

    def __init__(self, **kwargs):
        self.options = kwargs
        self.session = None
        self.__class__.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def create_session(self, **kwargs):
        if self.options.get("mode") == "empty" and "available_tools" not in kwargs:
            raise ValueError("empty mode requires available_tools")
        self.session = _FakeSession(kwargs)
        return self.session


@pytest.fixture
def fake_sdk(monkeypatch):
    import tradingagents.llm_clients.copilot_client as copilot_client

    _FakeCopilotClient.instances.clear()
    monkeypatch.setattr(
        copilot_client,
        "_load_copilot_sdk",
        lambda: (_FakeCopilotClient, _FakeTool, _FakeToolResult),
    )
    return copilot_client


@pytest.mark.unit
def test_factory_key_and_model_wiring():
    client = create_llm_client("copilot", "auto")
    assert type(client).__name__ == "GitHubCopilotClient"
    assert get_api_key_env("copilot") == "COPILOT_GITHUB_TOKEN"
    assert validate_model("copilot", "account-specific-model") is True


@pytest.mark.unit
def test_token_is_required(monkeypatch, fake_sdk):
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    with pytest.raises(ValueError, match="COPILOT_GITHUB_TOKEN"):
        create_llm_client("copilot", "auto").get_llm()


@pytest.mark.unit
def test_chat_adapter_uses_isolated_copilot_session(monkeypatch, fake_sdk):
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "github_pat_test")
    llm = create_llm_client("copilot", "auto").get_llm()

    response = llm.invoke(
        [
            SystemMessage(content="Analyze markets only."),
            HumanMessage(content="Analyze AAPL."),
        ]
    )

    client = _FakeCopilotClient.instances[-1]
    assert response.content == "copilot response"
    assert client.options["github_token"] == "github_pat_test"
    assert client.options["use_logged_in_user"] is False
    assert client.options["mode"] == "empty"
    assert client.session.options["system_message"] == {
        "mode": "replace",
        "content": "Analyze markets only.",
    }
    assert client.session.options["available_tools"] == []
    assert client.session.prompt == "Human: Analyze AAPL."


@pytest.mark.unit
def test_bound_langchain_tool_executes_inside_copilot(monkeypatch, fake_sdk):
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "github_pat_test")

    @tool
    def double(value: int) -> int:
        """Double an integer."""
        return value * 2

    llm = create_llm_client("copilot", "auto").get_llm().bind_tools([double])
    response = llm.invoke("Use the double tool.")

    client = _FakeCopilotClient.instances[-1]
    sdk_tool = client.session.options["tools"][0]
    assert response.content == "8"
    assert sdk_tool.name == "double"
    assert sdk_tool.skip_permission is True
    assert sdk_tool.defer == "never"
    assert client.session.options["available_tools"] == ["custom:double"]


@pytest.mark.unit
def test_structured_output_falls_back_to_existing_free_text_path(monkeypatch, fake_sdk):
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "github_pat_test")
    llm = create_llm_client("copilot", "auto").get_llm()
    with pytest.raises(NotImplementedError, match="structured output"):
        llm.with_structured_output(dict)


@pytest.mark.unit
def test_copilot_permission_error_is_actionable(monkeypatch, fake_sdk):
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "github_pat_test")

    async def unauthorized(*args, **kwargs):
        raise RuntimeError("403 unauthorized: not authorized to use this Copilot feature")

    monkeypatch.setattr(_FakeSession, "send_and_wait", unauthorized)
    llm = create_llm_client("copilot", "auto").get_llm()

    with pytest.raises(PermissionError, match="Copilot Requests"):
        llm.invoke("Analyze AAPL.")


@pytest.mark.unit
def test_copilot_policy_error_is_actionable(monkeypatch, fake_sdk):
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "github_pat_test")

    async def policy_blocked(*args, **kwargs):
        raise RuntimeError(
            "You are not authorized to use this Copilot feature, it requires "
            "an enterprise or organization policy to be enabled."
        )

    monkeypatch.setattr(_FakeSession, "send_and_wait", policy_blocked)
    llm = create_llm_client("copilot", "auto").get_llm()

    with pytest.raises(PermissionError, match="enable Copilot CLI"):
        llm.invoke("Analyze AAPL.")
