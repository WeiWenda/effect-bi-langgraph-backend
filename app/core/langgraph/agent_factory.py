"""Factory and registry for LangGraph workflow agents."""

from typing import (
    Dict,
    Optional,
    Type,
)

from app.core.langgraph.agent_interface import LangGraphAgentInterface
from app.core.langgraph.find_resource_agent import FindResourceLangGraphAgent
from app.core.langgraph.graph import LangGraphAgent
from app.core.logging import logger

DEFAULT_AGENT_NAME = "default"

_AGENT_REGISTRY: Dict[str, Type[LangGraphAgentInterface]] = {
    LangGraphAgent.agent_name: LangGraphAgent,
    FindResourceLangGraphAgent.agent_name: FindResourceLangGraphAgent,
}

_agents: Dict[str, LangGraphAgentInterface] = {}


def register_agent(agent_cls: Type[LangGraphAgentInterface]) -> Type[LangGraphAgentInterface]:
    """Register a LangGraph agent class for lookup by ``agent_name``."""
    _AGENT_REGISTRY[agent_cls.agent_name] = agent_cls
    return agent_cls


def resolve_agent_name(pre_defined_workflow: Optional[str] = None) -> str:
    """Resolve workflow name; empty or missing values map to the default agent."""
    if pre_defined_workflow is None:
        return DEFAULT_AGENT_NAME
    name = pre_defined_workflow.strip()
    return name if name else DEFAULT_AGENT_NAME


async def get_default_agent() -> LangGraphAgentInterface:
    """FastAPI dependency that returns the default LangGraph agent."""
    return await get_agent()


async def get_agent(pre_defined_workflow: Optional[str] = None) -> LangGraphAgentInterface:
    """Get or create a LangGraph agent instance for the given workflow.

    Args:
        pre_defined_workflow: Workflow name. When empty or omitted, returns the default agent.

    Returns:
        LangGraphAgentInterface: The agent for the requested workflow.

    Raises:
        ValueError: If the workflow name is not registered.
    """
    agent_name = resolve_agent_name(pre_defined_workflow)
    if agent_name not in _AGENT_REGISTRY:
        registered = ", ".join(sorted(_AGENT_REGISTRY))
        raise ValueError(f"Unknown pre_defined_workflow: {agent_name}. Available: {registered}")

    if agent_name not in _agents:
        agent_cls = _AGENT_REGISTRY[agent_name]
        _agents[agent_name] = agent_cls()
        logger.info("langgraph_agent_created", agent_name=agent_name)

    return _agents[agent_name]


async def close_all_agents() -> None:
    """Close all cached agent instances."""
    for agent_name, agent in list(_agents.items()):
        try:
            await agent.close()
            logger.info("langgraph_agent_closed", agent_name=agent_name)
        except Exception as e:
            logger.error("langgraph_agent_close_failed", agent_name=agent_name, error=str(e))
    _agents.clear()
