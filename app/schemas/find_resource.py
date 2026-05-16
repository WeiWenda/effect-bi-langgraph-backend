"""State schema for the find-resource LangGraph workflow."""

from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class FindResourceGraphState(BaseModel):
    """State for the find-resource multi-step workflow."""

    messages: Annotated[list, add_messages] = Field(
        default_factory=list,
        description="Conversation messages including user input and final assistant reply",
    )
    user_question: str = Field(default="", description="Raw question from the user")
    preprocessed_question: str = Field(default="", description="Clarified question after preprocessing")
    knowledge_context: str = Field(default="", description="Retrieved knowledge for answering")
    reasoning_answer: str = Field(default="", description="Draft answer from reasoning step")
    quality_passed: bool = Field(default=False, description="Whether answer quality meets the bar")
    planning_notes: str = Field(default="", description="Notes from reasoning planning for the next retrieval")
    formatted_output: str = Field(default="", description="Final formatted response for the user")
    iteration_count: int = Field(default=0, description="Number of retrieve-reason loops completed")
    max_iterations: int = Field(default=3, description="Maximum retrieve-reason loops before forcing format")
    custom_system_prompt: Optional[str] = Field(
        default=None,
        description="Optional override (unused by node prompts; reserved for compatibility)",
    )
