"""Interface for LangGraph workflow agents."""

from abc import ABC, abstractmethod
from typing import (
    AsyncGenerator,
    List,
    Optional,
)

from app.schemas import Message, StreamChunk


class LangGraphAgentInterface(ABC):
    """Interface for LangGraph workflow agents.

    Subclasses must define ``agent_name`` identifying the workflow they implement.
    """

    agent_name: str

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "agent_name", None):
            raise TypeError(f"{cls.__name__} must define agent_name")

    @abstractmethod
    async def close(self) -> None:
        """Release resources held by the agent."""

    @abstractmethod
    async def create_graph(self):
        """Create and configure the LangGraph workflow."""

    @abstractmethod
    async def get_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        """Get a response from the LLM."""

    @abstractmethod
    async def get_stream_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Get a stream response from the LLM."""

    @abstractmethod
    async def get_chat_history(self, session_id: str) -> List[Message]:
        """Get the chat history for a given session."""

    @abstractmethod
    async def clear_chat_history(self, session_id: str) -> None:
        """Clear all chat history for a given session."""
