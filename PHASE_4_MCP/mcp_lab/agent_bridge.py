"""Bridge MCP tools into the Phase 3 agent loop (notebooks 19–21).

Phase 3's :class:`agents_lab.Agent` runs a think→act→observe loop over a
:class:`agents_lab.ToolRegistry`. This module turns the tools discovered from one
or more MCP servers into exactly that registry, so the *same* agent can now act
through MCP. Because ``Agent`` uses the provider's normalized tool-calling
(``ChatResponse.tool_calls``), the bridge is **provider-agnostic**: the same code
runs on Ollama / OpenAI / DeepSeek / Anthropic / oxalpha (notebook 20).

    with build_agent(["filesystem", "terminal"], provider="ollama") as agent:
        print(agent.run("Find every TODO in the repository.").answer)
"""

from __future__ import annotations

from typing import Any

from agents_lab import Agent, Tool, ToolRegistry

from mcp_lab.client import MCPClient, lab_servers


def registry_from_mcp(client: MCPClient) -> ToolRegistry:
    """Wrap every tool exposed by ``client`` as an agents_lab ``Tool``."""
    registry = ToolRegistry()
    for spec in client.list_tools():
        name = spec["name"]

        # Bind the namespaced name so each closure calls its own tool.
        def make(qualified: str):
            def call(**arguments: Any) -> str:
                return client.call_tool(qualified, arguments)

            return call

        registry.add(
            Tool(
                name=name,
                description=spec.get("description", ""),
                func=make(name),
                parameters=spec.get("inputSchema", {"type": "object", "properties": {}}),
            )
        )
    return registry


class MCPAgent:
    """An :class:`agents_lab.Agent` wired to MCP servers, with owned lifecycle.

    Use as a context manager so the underlying server subprocesses are always
    shut down. Delegates :meth:`run` to the wrapped agent.
    """

    def __init__(
        self,
        servers: list[str] | dict[str, list[str]] | None = None,
        provider: str = "ollama",
        model: str | None = None,
        **agent_kwargs: Any,
    ) -> None:
        specs = servers if isinstance(servers, dict) else lab_servers(servers)
        self.client = MCPClient(specs)
        self._provider = provider
        self._model = model
        self._agent_kwargs = agent_kwargs
        self.agent: Agent | None = None

    def start(self) -> "MCPAgent":
        self.client.start()
        self.agent = Agent(
            provider=self._provider,
            model=self._model,
            tools=registry_from_mcp(self.client),
            **self._agent_kwargs,
        )
        return self

    def run(self, task: str, **kwargs: Any):
        assert self.agent is not None, "call start() first"
        return self.agent.run(task, **kwargs)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "MCPAgent":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()


def build_agent(
    servers: list[str] | dict[str, list[str]] | None = None,
    provider: str = "ollama",
    model: str | None = None,
    **agent_kwargs: Any,
) -> MCPAgent:
    """Convenience factory: an :class:`MCPAgent` ready to ``start()`` (or ``with``)."""
    return MCPAgent(servers=servers, provider=provider, model=model, **agent_kwargs)
