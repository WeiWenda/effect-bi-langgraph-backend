"""This file contains the prompts for the agent."""

import os
from datetime import datetime

from app.core.config import settings

# Read template once at module load — no file I/O per request
with open(os.path.join(os.path.dirname(__file__), "system.md"), "r") as _f:
    _SYSTEM_PROMPT_TEMPLATE = _f.read()


def load_system_prompt(**kwargs):
    """Load the system prompt from the cached template."""
    return _SYSTEM_PROMPT_TEMPLATE.format(
        agent_name=settings.PROJECT_NAME + " Agent",
        current_date_and_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **kwargs,
    )
