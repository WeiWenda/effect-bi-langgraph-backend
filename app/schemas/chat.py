"""This file contains the chat schema for the application."""

import re
from typing import (
    List,
    Literal,
    Optional,
)

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.base import BaseResponse


class Message(BaseModel):
    """Message model for chat endpoint.

    Attributes:
        role: The role of the message sender (user or assistant).
        content: The content of the message.
    """

    model_config = {"extra": "ignore"}

    role: Literal["user", "assistant", "system"] = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message", min_length=1, max_length=3000)

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate the message content.

        Args:
            v: The content to validate

        Returns:
            str: The validated content

        Raises:
            ValueError: If the content contains disallowed patterns
        """
        # Check for potentially harmful content
        if re.search(r"<script.*?>.*?</script>", v, re.IGNORECASE | re.DOTALL):
            raise ValueError("Content contains potentially harmful script tags")

        # Check for null bytes
        if "\0" in v:
            raise ValueError("Content contains null bytes")

        return v


class ChatRequest(BaseModel):
    """Request model for chat endpoint.

    Attributes:
        messages: List of messages in the conversation.
        system_prompt: Optional custom system prompt to override the default.
        pre_defined_workflow: Optional predefined workflow name for routing to a specific agent.
    """

    messages: List[Message] = Field(
        ...,
        description="List of messages in the conversation",
        min_length=1,
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt. If provided, overrides the default system prompt.",
    )
    pre_defined_workflow: Optional[str] = Field(
        default=None,
        description="Predefined workflow name. Routes to the matching LangGraph agent when set.",
    )

    @model_validator(mode="after")
    def clear_workflow_when_system_prompt_set(self) -> "ChatRequest":
        """Custom system prompt requests must use the default agent."""
        if self.system_prompt and self.pre_defined_workflow:
            self.pre_defined_workflow = None
        return self


class ChatResponse(BaseResponse):
    """Response model for chat endpoint.

    Attributes:
        messages: List of messages in the conversation.
    """

    messages: List[Message] = Field(..., description="List of messages in the conversation")


class StreamChunk(BaseModel):
    """A single chunk from agent streaming."""

    content: str = Field(default="", description="The content of the current chunk")


class StreamResponse(BaseResponse):
    """Response model for streaming chat endpoint.

    Attributes:
        content: The content of the current chunk.
        done: Whether the stream is complete.
    """

    content: str = Field(default="", description="The content of the current chunk")
    done: bool = Field(default=False, description="Whether the stream is complete")
