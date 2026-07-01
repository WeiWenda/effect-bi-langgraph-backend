"""News mode LangGraph agent backed by the yt-graphrag MCP server."""

import asyncio
import os
import re
from typing import (
    AsyncGenerator,
    List,
    Optional,
)
from urllib.parse import quote_plus

from fastmcp import Client
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, StateGraph
from langgraph.graph.state import Command, CompiledStateGraph
from langgraph.types import RunnableConfig, StateSnapshot
from psycopg_pool import AsyncConnectionPool

from app.core.config import Environment, settings
from app.core.langgraph.agent_interface import LangGraphAgentInterface
from app.core.langgraph.tools import ask_human
from app.core.logging import logger
from app.core.metrics import llm_inference_duration_seconds
from app.core.observability import langfuse_callback_handler
from app.schemas import GraphState, Message, StreamChunk
from app.services.llm import LLMService
from app.utils import dump_messages, prepare_messages, process_llm_response
from app.utils.langchain_message import ai_message_content_to_str

_GRAPHRAG_MCP_URL = os.getenv("GRAPHRAG_MCP_URL", "http://localhost:50425/mcp")
_GRAPHRAG_SKILL_PATH = os.getenv(
    "GRAPHRAG_SKILL_PATH",
    "/Users/weiwenda/workspace/yt-graphrag/skills/yt-graphrag-query/SKILL.md",
)

_NEWS_SYSTEM_PROMPT_CACHE: Optional[str] = None


def _load_news_system_prompt() -> str:
    """Load and cache the yt-graphrag SKILL.md as the news agent system prompt."""
    global _NEWS_SYSTEM_PROMPT_CACHE
    if _NEWS_SYSTEM_PROMPT_CACHE is not None:
        return _NEWS_SYSTEM_PROMPT_CACHE
    try:
        with open(_GRAPHRAG_SKILL_PATH, "r", encoding="utf-8") as f:
            raw = f.read()
        # Strip YAML frontmatter (--- ... ---) if present
        content = re.sub(r"\A---\n.*?\n---\n*", "", raw, flags=re.DOTALL).strip()
        _NEWS_SYSTEM_PROMPT_CACHE = content
        logger.info("news_skill_prompt_loaded", path=_GRAPHRAG_SKILL_PATH, length=len(content))
    except Exception as e:
        logger.error("news_skill_prompt_load_failed", path=_GRAPHRAG_SKILL_PATH, error=str(e))
        _NEWS_SYSTEM_PROMPT_CACHE = "You are a news assistant with access to a knowledge graph."
    return _NEWS_SYSTEM_PROMPT_CACHE


class NewsLangGraphAgent(LangGraphAgentInterface):
    """News mode agent that queries the yt-graphrag knowledge graph via MCP tools."""

    agent_name = "news"

    def __init__(self) -> None:
        self.llm_service = LLMService()
        self._local_tools = [ask_human]
        self._mcp_client: Optional[Client] = None
        self._mcp_tools: List = []
        self.tools_by_name: dict = {}
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._graph: Optional[CompiledStateGraph] = None
        asyncio.create_task(self._init_mcp_and_bind_tools())

    async def close(self) -> None:
        if self._mcp_client:
            try:
                await self._mcp_client.close()
                logger.info("news_mcp_client_closed")
            except Exception as e:
                logger.error("news_mcp_client_close_failed", error=str(e))
        if self._connection_pool:
            try:
                await self._connection_pool.close()
                logger.info("news_connection_pool_closed")
            except Exception as e:
                logger.error("news_connection_pool_close_failed", error=str(e))

    def _convert_mcp_tools_to_langchain(self, mcp_tools):
        """Convert MCP tools to LangChain StructuredTool objects."""
        langchain_tools = []
        for tool in mcp_tools:
            def create_func(mcp_client, tool_name):
                async def mcp_tool_func(**kwargs):
                    try:
                        async with mcp_client:
                            result = await mcp_client.call_tool(tool_name, kwargs)
                            if hasattr(result, "content") and result.content:
                                if isinstance(result.content, list):
                                    first_item = result.content[0]
                                    if hasattr(first_item, "text"):
                                        return first_item.text
                                    elif isinstance(first_item, dict):
                                        return first_item.get("text", "")
                                    return str(first_item)
                                return str(result.content)
                            return str(result)
                    except Exception as e:
                        logger.error("news_mcp_tool_call_failed", tool_name=tool_name, error=str(e))
                        return f"Error calling tool {tool_name}: {str(e)}"
                return mcp_tool_func

            mcp_tool_func = create_func(self._mcp_client, tool.name)
            structured_tool = StructuredTool.from_function(
                func=mcp_tool_func,
                coroutine=mcp_tool_func,
                name=tool.name,
                description=tool.description or "",
                args_schema=tool.inputSchema or {},
            )
            langchain_tools.append(structured_tool)
        return langchain_tools

    async def _init_mcp_and_bind_tools(self):
        """Initialize MCP client and bind all tools to the LLM."""
        try:
            self._mcp_client = Client(_GRAPHRAG_MCP_URL)
            async with self._mcp_client:
                tools = await self._mcp_client.list_tools()
                self._mcp_tools = self._convert_mcp_tools_to_langchain(tools) if tools else []
                logger.info("news_mcp_tools_loaded", tool_count=len(self._mcp_tools), url=_GRAPHRAG_MCP_URL)
        except Exception as e:
            logger.error("news_mcp_client_connection_failed", url=_GRAPHRAG_MCP_URL, error=str(e))
            self._mcp_client = None
            self._mcp_tools = []

        all_tools = self._local_tools.copy()
        all_tools.extend(self._mcp_tools)
        self.tools_by_name = {}
        for tool in all_tools:
            name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
            if name:
                self.tools_by_name[name] = tool
        self.llm_service.bind_tools(all_tools)
        logger.info(
            "news_all_tools_bound",
            total=len(all_tools),
            local=len(self._local_tools),
            mcp=len(self._mcp_tools),
        )

    async def _get_connection_pool(self) -> AsyncConnectionPool:
        if self._connection_pool is None:
            connection_url = (
                "postgresql://"
                f"{quote_plus(settings.POSTGRES_USER)}:{quote_plus(settings.POSTGRES_PASSWORD)}"
                f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
            )
            self._connection_pool = AsyncConnectionPool(
                connection_url,
                open=False,
                max_size=settings.POSTGRES_POOL_SIZE,
                kwargs={"autocommit": True, "connect_timeout": 5, "prepare_threshold": None},
            )
            await self._connection_pool.open()
            logger.info("news_connection_pool_created")
        return self._connection_pool

    async def _chat(self, state: GraphState, config: RunnableConfig) -> Command:
        current_llm = self.llm_service.get_llm()
        model_name = (
            current_llm.model_name
            if current_llm and hasattr(current_llm, "model_name")
            else settings.DEFAULT_LLM_MODEL
        )
        system_prompt = _load_news_system_prompt()
        username = config.get("metadata", {}).get("username")
        if username:
            system_prompt = f"# User\nYou are talking to {username}.\n\n{system_prompt}"
        messages = prepare_messages(state.messages, system_prompt)
        try:
            with llm_inference_duration_seconds.labels(model=model_name).time():
                response_message = await self.llm_service.call(messages)
            response_message = process_llm_response(response_message)
            logger.info(
                "news_llm_response_generated",
                session_id=config["configurable"]["thread_id"],
                model=model_name,
            )
            goto = "tool_call" if response_message.tool_calls else END
            return Command(update={"messages": [response_message]}, goto=goto)
        except Exception as e:
            logger.exception("news_llm_call_failed", session_id=config["configurable"]["thread_id"], error=str(e))
            raise

    async def _tool_call(self, state: GraphState) -> Command:
        tool_calls = state.messages[-1].tool_calls

        async def _execute_tool(tool_call: dict) -> ToolMessage:
            tool = self.tools_by_name.get(tool_call["name"])
            if tool is None:
                return ToolMessage(
                    content=f"Tool {tool_call['name']} not found.",
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            tool_result = await tool.ainvoke(tool_call["args"])
            return ToolMessage(
                content=str(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )

        if len(tool_calls) == 1:
            outputs = [await _execute_tool(tool_calls[0])]
        else:
            outputs = list(await asyncio.gather(*[_execute_tool(tc) for tc in tool_calls]))
        return Command(update={"messages": outputs}, goto="chat")

    async def create_graph(self) -> Optional[CompiledStateGraph]:
        if self._graph is None:
            graph_builder = StateGraph(GraphState)
            graph_builder.add_node("chat", self._chat, ends=["tool_call", END])
            graph_builder.add_node("tool_call", self._tool_call, ends=["chat"])
            graph_builder.set_entry_point("chat")
            graph_builder.set_finish_point("chat")

            connection_pool = await self._get_connection_pool()
            checkpointer = None
            if connection_pool:
                checkpointer = AsyncPostgresSaver(connection_pool)
                await checkpointer.setup()
            elif settings.ENVIRONMENT != Environment.PRODUCTION:
                raise Exception("Connection pool initialization failed for news agent")

            self._graph = graph_builder.compile(
                checkpointer=checkpointer,
                name=f"{settings.PROJECT_NAME} News ({settings.ENVIRONMENT.value})",
            )
            logger.info("news_graph_created", has_checkpointer=checkpointer is not None)
        return self._graph

    def _build_run_config(self, session_id: str, user_id: Optional[str], username: Optional[str]) -> dict:
        callbacks = [langfuse_callback_handler] if settings.LANGFUSE_TRACING_ENABLED else []
        return {
            "configurable": {"thread_id": session_id},
            "callbacks": callbacks,
            "metadata": {
                "user_id": user_id,
                "username": username,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "workflow": self.agent_name,
                "debug": settings.DEBUG,
            },
        }

    async def get_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        if self._graph is None:
            await self.create_graph()
        config = self._build_run_config(session_id, user_id, username)
        try:
            state = await self._graph.aget_state(config)
            if state.next:
                response = await self._graph.ainvoke(
                    Command(resume=messages[-1].content), config=config
                )
            else:
                response = await self._graph.ainvoke(
                    input={"messages": dump_messages(messages), "long_term_memory": "", "custom_system_prompt": None},
                    config=config,
                )
            state = await self._graph.aget_state(config)
            if state.next:
                interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
                return [Message(role="assistant", content=str(interrupt_value))]
            return self._process_messages(response["messages"])
        except GraphInterrupt:
            state = await self._graph.aget_state(config)
            interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else None
            return [Message(role="assistant", content=str(interrupt_value))]
        except Exception as e:
            logger.exception("news_get_response_failed", session_id=session_id, error=str(e))
            raise

    async def get_stream_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        if self._graph is None:
            await self.create_graph()
        config = self._build_run_config(session_id, user_id, username)
        try:
            state = await self._graph.aget_state(config)
            if state.next:
                graph_input = Command(resume=messages[-1].content)
            else:
                graph_input = {
                    "messages": dump_messages(messages),
                    "long_term_memory": "",
                    "custom_system_prompt": None,
                }

            async for output in self._graph.astream(graph_input, config):
                for _node_name, node_state in output.items():
                    if isinstance(node_state, dict) and "messages" in node_state:
                        node_messages = node_state["messages"]
                        if node_messages:
                            last_message = node_messages[-1]
                            if isinstance(last_message, AIMessage):
                                content = ai_message_content_to_str(last_message.content)
                                if content:
                                    yield StreamChunk(content=content)

            state = await self._graph.aget_state(config)
            if state.next:
                interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else None
                yield StreamChunk(content=str(interrupt_value))
        except GraphInterrupt:
            state = await self._graph.aget_state(config)
            interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else None
            yield StreamChunk(content=str(interrupt_value))
        except Exception as e:
            logger.exception("news_stream_failed", session_id=session_id, error=str(e))
            raise

    async def get_chat_history(self, session_id: str) -> list[Message]:
        if self._graph is None:
            await self.create_graph()
        state: StateSnapshot = await self._graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )
        return self._process_messages(state.values["messages"]) if state.values else []

    def _process_messages(self, messages: list[BaseMessage]) -> list[Message]:
        openai_style_messages = convert_to_openai_messages(messages)
        return [
            Message(role=message["role"], content=str(message["content"]))
            for message in openai_style_messages
            if message["role"] in ["assistant", "user"] and message["content"]
        ]

    async def clear_chat_history(self, session_id: str) -> None:
        conn_pool = await self._get_connection_pool()
        async with conn_pool.connection() as conn:
            async with conn.pipeline():
                for table in settings.CHECKPOINT_TABLES:
                    await conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (session_id,))
        logger.info("news_checkpoint_cleared", session_id=session_id)
