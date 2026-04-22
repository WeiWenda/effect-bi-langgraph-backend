"""This file contains the graph schema for the application."""

from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from pydantic import (
    BaseModel,
    Field,
)


class GraphState(BaseModel):
    """State definition for the LangGraph Agent/Workflow."""

    messages: Annotated[list, add_messages] = Field(
        default_factory=list, description="The messages in the conversation"
    )
    long_term_memory: str = Field(default="", description="The long term memory of the conversation")
    custom_system_prompt: Optional[str] = Field(
        default=None, description="Custom system prompt to override the default"
    )
