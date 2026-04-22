"""LangGraph tools for enhanced language model capabilities.

This package contains custom tools that can be used with LangGraph to extend
the capabilities of language models. Currently includes tools for web search,
CLAWHUB skill management, and other external integrations.
"""

from langchain_core.tools.base import BaseTool

from .ask_human import ask_human
from .clawhub_skills import (
    install_clawhub_skill,
    list_installed_clawhub_skills,
    load_clawhub_skill,
    search_clawhub_skills,
)
from .duckduckgo_search import duckduckgo_search_tool
from .env_check import check_env_key
from .skill_exec import skill_exec_bash_cmd

tools: list[BaseTool] = [
    # duckduckgo_search_tool,
    ask_human,
    check_env_key,
    skill_exec_bash_cmd,
    search_clawhub_skills,
    install_clawhub_skill,
    load_clawhub_skill,
    list_installed_clawhub_skills,
]
