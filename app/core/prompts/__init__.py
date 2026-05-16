"""This file contains the prompts for the agent."""

import os
from datetime import datetime
from typing import Optional

from app.core.config import settings

# Read template once at module load — no file I/O per request
with open(os.path.join(os.path.dirname(__file__), "system.md"), "r") as _f:
    _SYSTEM_PROMPT_TEMPLATE = _f.read()


def load_system_prompt(username: Optional[str] = None, **kwargs):
    """Load the system prompt from the cached template."""
    user_context = f"# User\nYou are talking to {username}.\n" if username else ""
    return _SYSTEM_PROMPT_TEMPLATE.format(
        agent_name=settings.PROJECT_NAME + " Agent",
        current_date_and_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        user_context=user_context,
        **kwargs,
    )


_FIND_RESOURCE_PROMPT_CACHE: dict[str, str] = {}
_FIND_RESOURCE_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "find_resource")


def load_find_resource_prompt(node: str, username: Optional[str] = None, **kwargs) -> str:
    """Load a find-resource workflow node prompt from ``prompts/find_resource/{node}.md``."""
    if node not in _FIND_RESOURCE_PROMPT_CACHE:
        path = os.path.join(_FIND_RESOURCE_PROMPTS_DIR, f"{node}.md")
        with open(path, "r", encoding="utf-8") as f:
            _FIND_RESOURCE_PROMPT_CACHE[node] = f.read()
    user_context = f"# User\n与 {username} 对话。\n" if username else ""
    return _FIND_RESOURCE_PROMPT_CACHE[node].format(
        current_date_and_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        user_context=user_context,
        **kwargs,
    )
