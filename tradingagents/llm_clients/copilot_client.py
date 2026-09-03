"""GitHub Copilot SDK adapter for TradingAgents' LangChain-based agents."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, get_buffer_string
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, SecretStr

from .base_client import BaseLLMClient


def _load_copilot_sdk():
    try:
        from copilot import CopilotClient
        from copilot.tools import Tool, ToolResult
    except ImportError as exc:
        raise ImportError(
            "GitHub Copilot support requires Python 3.11+ and the optional SDK. "
            'Install it with: pip install "tradingagents[copilot]"'
        ) from exc
    return CopilotClient, Tool, ToolResult


def _message_parts(messages: list[BaseMessage]) -> tuple[str, str]:
    system = "\n\n".join(
        str(message.content) for message in messages if isinstance(message, SystemMessage)
    )
    conversation = [
        message for message in messages if not isinstance(message, SystemMessage)
    ]
    prompt = get_buffer_string(conversation) if conversation else "Continue."
    return system or "You are a helpful assistant.", prompt


async def _invoke_tool(tool: Any, arguments: dict[str, Any]) -> Any:
    if isinstance(tool, BaseTool):
        return await tool.ainvoke(arguments)
    if not callable(tool):
        raise TypeError("Copilot tools must be LangChain tools or callables")

    result = tool(**arguments)
    if inspect.isawaitable(result):
        return await result
    return result


def _tool_result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "model_dump_json"):
        return result.model_dump_json()
    return json.dumps(result, default=str)


def _copilot_tools(
    tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
    tool_class,
    result_class,
) -> list[Any]:
    converted = []
    for source_tool in tools:
        specification = convert_to_openai_tool(source_tool)["function"]

        async def handler(invocation, bound_tool=source_tool):
            try:
                result = await _invoke_tool(bound_tool, invocation.arguments or {})
                return result_class(
                    text_result_for_llm=_tool_result_text(result),
                    result_type="success",
                )
            except Exception as exc:
                return result_class(
                    text_result_for_llm="Invoking this tool produced an error.",
                    result_type="failure",
                    error=str(exc),
                )

        converted.append(
            tool_class(
                name=specification["name"],
                description=specification.get("description", ""),
                parameters=specification.get("parameters"),
                handler=handler,
                skip_permission=True,
                defer="never",
            )
        )
    return converted


class ChatGitHubCopilot(BaseChatModel):
    """Expose GitHub Copilot as the chat-model contract TradingAgents expects.

    Copilot runs bound LangChain tools inside its own agent loop and returns the
    final assistant response to LangGraph.
    """

    model_name: str
    github_token: SecretStr = Field(repr=False)
    request_timeout: float = 300.0
    working_directory: str = Field(default_factory=os.getcwd)
    base_directory: str = Field(
        default_factory=lambda: os.path.join(
            os.path.expanduser("~"), ".tradingagents", "copilot"
        )
    )

    @property
    def _llm_type(self) -> str:
        return "github-copilot"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        if tool_choice not in (None, "auto"):
            raise NotImplementedError("GitHub Copilot controls tool selection")
        return self.bind(tools=list(tools), **kwargs)

    def with_structured_output(self, schema: Any, **kwargs: Any):
        raise NotImplementedError(
            "GitHub Copilot SDK does not expose LangChain structured output"
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        copilot_client_class, tool_class, result_class = _load_copilot_sdk()
        system_message, prompt = _message_parts(messages)
        tools = _copilot_tools(kwargs.get("tools", []), tool_class, result_class)

        try:
            async with copilot_client_class(
                github_token=self.github_token.get_secret_value(),
                use_logged_in_user=False,
                working_directory=self.working_directory,
                base_directory=self.base_directory,
                mode="empty",
            ) as client, await client.create_session(
                model=self.model_name,
                tools=tools or None,
                available_tools=[f"custom:{tool.name}" for tool in tools],
                system_message={"mode": "replace", "content": system_message},
                working_directory=self.working_directory,
                enable_session_store=False,
                infinite_sessions={"enabled": False},
            ) as session:
                response = await session.send_and_wait(
                    prompt,
                    timeout=self.request_timeout,
                )
        except Exception as exc:
            error = str(exc)
            if "requires an enterprise or organization policy to be enabled" in error:
                raise PermissionError(
                    "GitHub Copilot SDK access is disabled by your organization or "
                    "enterprise policy. Ask an administrator to enable Copilot CLI "
                    "for your account."
                ) from exc
            if "not authorized to use this Copilot feature" in error:
                raise PermissionError(
                    "GitHub rejected this token. Create a personal-account "
                    "fine-grained PAT with the 'Copilot Requests' account permission "
                    "and ensure that account has Copilot access."
                ) from exc
            raise

        if response is None or not hasattr(response.data, "content"):
            raise RuntimeError("GitHub Copilot returned no assistant response")

        message = AIMessage(content=response.data.content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        return asyncio.run(
            self._agenerate(
                messages,
                stop=stop,
                run_manager=None,
                **kwargs,
            )
        )


class GitHubCopilotClient(BaseLLMClient):
    """Create a LangChain-compatible model backed by GitHub Copilot."""

    provider = "copilot"

    def get_llm(self) -> ChatGitHubCopilot:
        _load_copilot_sdk()
        token = os.environ.get("COPILOT_GITHUB_TOKEN")
        if not token:
            raise ValueError(
                "GitHub Copilot token is not set. Add "
                "COPILOT_GITHUB_TOKEN=github_pat_... to your .env file."
            )

        chat_kwargs = {
            key: self.kwargs[key]
            for key in ("callbacks", "tags", "metadata", "verbose")
            if key in self.kwargs
        }
        return ChatGitHubCopilot(
            model_name=self.model,
            github_token=token,
            request_timeout=float(self.kwargs.get("timeout", 300.0)),
            **chat_kwargs,
        )

    def validate_model(self) -> bool:
        return True
