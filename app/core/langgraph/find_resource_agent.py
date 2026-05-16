"""Find-resource LangGraph agent with multi-step LLM workflow."""

import asyncio
import re
from typing import (
    AsyncGenerator,
    List,
    Optional,
)
from urllib.parse import quote_plus

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, StateGraph
from langgraph.graph.state import Command, CompiledStateGraph
from langgraph.types import RunnableConfig, StateSnapshot
from psycopg_pool import AsyncConnectionPool

from app.core.config import Environment, settings
from app.core.langgraph.agent_interface import LangGraphAgentInterface
from app.core.langgraph.tools.ask_human import ask_human
from app.core.langgraph.tools.query_related_document import query_related_document
from app.core.logging import logger
from app.core.metrics import llm_inference_duration_seconds
from app.core.observability import langfuse_callback_handler
from app.core.prompts import load_find_resource_prompt
from app.schemas import Message, StreamChunk
from app.utils.interrupt import extract_interrupt_text
from app.utils.langchain_message import ai_message_content_to_str
from app.schemas.find_resource import FindResourceGraphState
from app.services.llm import LLMRegistry, llm_service
from app.utils import dump_messages, process_llm_response

# Nodes whose LLM output may be streamed to the client
_STREAM_YIELD_NODES = frozenset({"format_result", "stream_output"})


def _extract_last_human_content(messages: list) -> str:
    """Return content of the last human message in state."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


_VALID_WAREHOUSE_LAYERS = frozenset({"原始数据", "明细", "宽表", "报表", "不限"})
_VALID_BUSINESS_DOMAINS = frozenset({"商品域", "交易域", "用户域", "店铺域", "不限"})
_PREPROCESS_VAGUE_MARKERS = ("未指定", "不明确", "不清楚", "待澄清", "待定", "未知", "？")


def _extract_preprocess_field(content: str, label: str) -> Optional[str]:
    match = re.search(rf"{re.escape(label)}[：:]\s*([^\n]+)", content)
    return match.group(1).strip() if match else None


def _parse_preprocessed_result(content: str) -> Optional[dict[str, str]]:
    """Return parsed fields only when preprocessing output is valid for downstream nodes."""
    if not content or "【预处理结果】" not in content:
        return None

    clarified = _extract_preprocess_field(content, "澄清后的问题")
    layer = _extract_preprocess_field(content, "数仓分层")
    domain = _extract_preprocess_field(content, "业务域")
    keywords = _extract_preprocess_field(content, "检索关键词") or ""

    if not clarified or len(clarified) < 4:
        return None
    if any(marker in clarified for marker in _PREPROCESS_VAGUE_MARKERS):
        return None
    if layer not in _VALID_WAREHOUSE_LAYERS or domain not in _VALID_BUSINESS_DOMAINS:
        return None

    return {
        "clarified": clarified,
        "layer": layer,
        "domain": domain,
        "keywords": keywords,
        "raw": content,
    }


def _build_preprocess_clarification_question(user_question: str, invalid_content: str) -> str:
    """Build ask_human prompt when preprocessing is not yet valid."""
    intro = ""
    q = user_question.strip()
    workflow_hints = ("怎么用", "如何使用", "什么是", "帮助", "不会用", "找表模式")
    vague_intent = len(q) < 8 or q in ("找表", "查表", "有没有表") or not any(
        c.isalnum() or "\u4e00" <= c <= "\u9fff" for c in q
    )

    if vague_intent or any(h in q for h in workflow_hints):
        intro = (
            "【找表模式说明】本工作流根据您选择的数仓分层和业务域，检索可能相关的数据表。"
            "请用一句话说明想找什么数据（例如：交易域里记录订单支付金额的明细表）。\n\n"
        )

    missing: list[str] = []
    parsed = _parse_preprocessed_result(invalid_content)
    if not parsed:
        missing.append("请明确找表意图（想找什么业务对象/指标相关的表）")
        missing.append(
            "请选择数仓分层：原始数据 / 明细 / 宽表 / 报表 / 不限"
        )
        missing.append(
            "请选择业务域：商品域 / 交易域 / 用户域 / 店铺域 / 不限"
        )
    else:
        if parsed["layer"] not in _VALID_WAREHOUSE_LAYERS:
            missing.append("请选择数仓分层：原始数据 / 明细 / 宽表 / 报表 / 不限")
        if parsed["domain"] not in _VALID_BUSINESS_DOMAINS:
            missing.append("请选择业务域：商品域 / 交易域 / 用户域 / 店铺域 / 不限")

    return intro + "\n".join(f"- {item}" for item in missing)


def _parse_quality_passed(text: str) -> bool:
    """Parse quality-check LLM output (expects Y or N)."""
    normalized = text.strip().upper()
    if normalized.startswith("Y"):
        return True
    if normalized.startswith("N"):
        return False
    if re.search(r"\bYES\b|\bPASS\b|达标", normalized):
        return True
    return False


class FindResourceLangGraphAgent(LangGraphAgentInterface):
    """Multi-step workflow for locating data / resource tables."""

    agent_name = "findResource"

    def __init__(self) -> None:
        self.llm_service = llm_service
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._graph: Optional[CompiledStateGraph] = None
        self.tools_by_name = {
            ask_human.name: ask_human,
            query_related_document.name: query_related_document,
        }

    async def close(self) -> None:
        if self._connection_pool:
            try:
                await self._connection_pool.close()
                logger.info("find_resource_connection_pool_closed")
            except Exception as e:
                logger.error("find_resource_connection_pool_close_failed", error=str(e))

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
                kwargs={
                    "autocommit": True,
                    "connect_timeout": 5,
                    "prepare_threshold": None,
                },
            )
            await self._connection_pool.open()
        return self._connection_pool

    async def _invoke_llm(
        self,
        node: str,
        user_payload: str,
        config: RunnableConfig,
        *,
        tools: Optional[list] = None,
        history: Optional[list] = None,
    ) -> AIMessage:
        """Call LLM with a node-specific system prompt and no cross-node prompt mixing."""
        username = config.get("metadata", {}).get("username")
        system_prompt = load_find_resource_prompt(node, username=username)
        llm = LLMRegistry.get(settings.DEFAULT_LLM_MODEL)
        if tools:
            llm = llm.bind_tools(tools)
        messages = history or [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_payload),
        ]
        if history is None and messages and not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt), *messages]
        model_name = settings.DEFAULT_LLM_MODEL
        with llm_inference_duration_seconds.labels(model=model_name).time():
            response = await llm.ainvoke(messages)
        return process_llm_response(response)

    async def _run_tool_loop(
        self,
        node: str,
        user_payload: str,
        config: RunnableConfig,
        *,
        tools: list,
        allowed_tool_names: set[str],
        max_rounds: int = 6,
    ) -> tuple[AIMessage, list[BaseMessage], str]:
        """Run LLM + inline tool execution until no tool calls remain."""
        username = config.get("metadata", {}).get("username")
        system_prompt = load_find_resource_prompt(node, username=username)
        messages: list = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_payload),
        ]
        trace: list[BaseMessage] = []
        knowledge_accumulated = ""

        for _ in range(max_rounds):
            response = await self._invoke_llm(
                node,
                user_payload,
                config,
                tools=tools,
                history=messages,
            )
            trace.append(response)
            if not response.tool_calls:
                break

            messages.append(response)
            for tool_call in response.tool_calls:
                name = tool_call["name"]
                if name not in allowed_tool_names:
                    tool_message = ToolMessage(
                        content=f"Tool {name} is not allowed in this step.",
                        name=name,
                        tool_call_id=tool_call["id"],
                    )
                else:
                    tool_result = await self.tools_by_name[name].ainvoke(tool_call["args"])
                    result_text = str(tool_result)
                    knowledge_accumulated += result_text + "\n"
                    tool_message = ToolMessage(
                        content=result_text,
                        name=name,
                        tool_call_id=tool_call["id"],
                    )
                messages.append(tool_message)
                trace.append(tool_message)

        final = trace[-1] if trace else response
        return final, trace, knowledge_accumulated.strip()

    async def _receive_question(self, state: FindResourceGraphState, config: RunnableConfig) -> Command:
        question = _extract_last_human_content(state.messages) or state.user_question
        logger.info(
            "find_resource_receive_question",
            session_id=config["configurable"]["thread_id"],
            question_length=len(question),
        )
        return Command(update={"user_question": question}, goto="question_preprocess")

    def _build_question_preprocess_payload(self, state: FindResourceGraphState) -> str:
        latest_human = _extract_last_human_content(state.messages)
        payload = f"# 用户原始问题\n{state.user_question}"
        if latest_human and latest_human != state.user_question:
            payload += f"\n\n# 用户后续补充\n{latest_human}"
        return payload

    async def _question_preprocess(self, state: FindResourceGraphState, config: RunnableConfig) -> Command:
        payload = self._build_question_preprocess_payload(state)
        final, trace, _ = await self._run_tool_loop(
            "question_preprocess",
            payload,
            config,
            tools=[ask_human],
            allowed_tool_names={ask_human.name},
        )
        content = str(final.content) if final.content else ""

        parsed = _parse_preprocessed_result(content)
        if parsed:
            logger.info(
                "find_resource_preprocess_valid",
                session_id=config["configurable"]["thread_id"],
                layer=parsed["layer"],
                domain=parsed["domain"],
            )
            return Command(
                update={
                    "messages": trace,
                    "preprocessed_question": parsed["raw"],
                },
                goto="knowledge_retrieval",
            )

        logger.info(
            "find_resource_preprocess_invalid",
            session_id=config["configurable"]["thread_id"],
            has_preprocess_block="【预处理结果】" in content,
        )
        clarify_question = _build_preprocess_clarification_question(state.user_question, content)
        await ask_human.ainvoke({"question": clarify_question})
        return Command(
            update={"messages": trace},
            goto="question_preprocess",
        )

    async def _knowledge_retrieval(self, state: FindResourceGraphState, config: RunnableConfig) -> Command:
        question = state.preprocessed_question or state.user_question
        planning = f"\n\n# 检索规划（上一轮）\n{state.planning_notes}" if state.planning_notes else ""
        payload = (
            f"# 用户问题\n{question}\n\n"
            f"# 预处理结果\n{state.preprocessed_question}{planning}"
        )
        final, trace, tool_context = await self._run_tool_loop(
            "knowledge_retrieval",
            payload,
            config,
            tools=[query_related_document],
            allowed_tool_names={query_related_document.name},
        )
        knowledge_context = tool_context or str(final.content)
        if not tool_context:
            logger.warning(
                "find_resource_knowledge_retrieval_no_tool_calls",
                session_id=config["configurable"]["thread_id"],
            )
        # TODO: 恢复 reason_answer 节点后改回 goto="reason_answer"
        return Command(
            update={
                "knowledge_context": knowledge_context,
                "reasoning_answer": knowledge_context,
                "messages": trace,
            },
            goto="quality_check",
        )

    # async def _reason_answer(self, state: FindResourceGraphState, config: RunnableConfig) -> Command:
    #     """推理答案（暂时下线，流程跑通后再启用）。"""
    #     payload = (
    #         f"# 用户问题\n{state.preprocessed_question or state.user_question}\n\n"
    #         f"# 知识检索结果\n{state.knowledge_context}"
    #     )
    #     response = await self._invoke_llm("reason_answer", payload, config)
    #     return Command(
    #         update={"reasoning_answer": str(response.content), "messages": [response]},
    #         goto="quality_check",
    #     )

    async def _quality_check(self, state: FindResourceGraphState, config: RunnableConfig) -> Command:
        retrieval_result = state.knowledge_context or state.reasoning_answer
        payload = (
            f"# 用户原始问题\n{state.user_question}\n\n"
            f"# 预处理结果（含数仓分层、业务域）\n{state.preprocessed_question}\n\n"
            f"# 检索结果\n{retrieval_result}"
        )
        response = await self._invoke_llm("quality_check", payload, config)
        passed = _parse_quality_passed(str(response.content))
        return Command(update={"quality_passed": passed, "messages": [response]})

    def _route_after_quality(self, state: FindResourceGraphState) -> str:
        if state.quality_passed:
            return "format_result"
        if state.iteration_count >= state.max_iterations:
            logger.warning(
                "find_resource_max_iterations_reached",
                iteration_count=state.iteration_count,
                max_iterations=state.max_iterations,
            )
            return "format_result"
        return "reasoning_planning"

    async def _reasoning_planning(self, state: FindResourceGraphState, config: RunnableConfig) -> Command:
        payload = (
            f"# 用户问题\n{state.preprocessed_question or state.user_question}\n\n"
            f"# 知识检索结果\n{state.knowledge_context}\n\n"
            f"# 推理答案\n{state.reasoning_answer}"
        )
        response = await self._invoke_llm("reasoning_planning", payload, config)
        return Command(
            update={
                "planning_notes": str(response.content),
                "iteration_count": state.iteration_count + 1,
                "messages": [response],
            },
            goto="knowledge_retrieval",
        )

    async def _format_result(self, state: FindResourceGraphState, config: RunnableConfig) -> Command:
        payload = (
            f"# 用户原始问题\n{state.user_question}\n\n"
            f"# 预处理结果\n{state.preprocessed_question}\n\n"
            f"# 检索结果\n{state.knowledge_context}"
        )
        response = await self._invoke_llm("format_result", payload, config)
        return Command(
            update={"formatted_output": str(response.content), "messages": [response]},
            goto="stream_output",
        )

    async def _stream_output(self, state: FindResourceGraphState, config: RunnableConfig) -> Command:
        final = state.formatted_output
        if not final and state.messages:
            last = state.messages[-1]
            if isinstance(last, AIMessage):
                final = str(last.content)
        ai_msg = AIMessage(content=final)
        logger.info(
            "find_resource_stream_output",
            session_id=config["configurable"]["thread_id"],
            output_length=len(final),
        )
        return Command(update={"messages": [ai_msg]}, goto=END)

    async def create_graph(self) -> Optional[CompiledStateGraph]:
        if self._graph is not None:
            return self._graph

        graph_builder = StateGraph(FindResourceGraphState)
        graph_builder.add_node("receive_question", self._receive_question, ends=["question_preprocess"])
        graph_builder.add_node(
            "question_preprocess",
            self._question_preprocess,
            ends=["knowledge_retrieval", "question_preprocess"],
        )
        graph_builder.add_node("knowledge_retrieval", self._knowledge_retrieval, ends=["quality_check"])
        # graph_builder.add_node("reason_answer", self._reason_answer, ends=["quality_check"])
        graph_builder.add_node("quality_check", self._quality_check)
        graph_builder.add_node("reasoning_planning", self._reasoning_planning, ends=["knowledge_retrieval"])
        graph_builder.add_node("format_result", self._format_result, ends=["stream_output"])
        graph_builder.add_node("stream_output", self._stream_output, ends=[END])

        graph_builder.set_entry_point("receive_question")
        graph_builder.add_conditional_edges(
            "quality_check",
            self._route_after_quality,
            {"format_result": "format_result", "reasoning_planning": "reasoning_planning"},
        )
        graph_builder.set_finish_point("stream_output")

        connection_pool = await self._get_connection_pool()
        checkpointer = None
        if connection_pool:
            checkpointer = AsyncPostgresSaver(connection_pool)
            await checkpointer.setup()
        elif settings.ENVIRONMENT != Environment.PRODUCTION:
            raise Exception("Connection pool initialization failed for find-resource agent")

        self._graph = graph_builder.compile(
            checkpointer=checkpointer,
            name=f"{settings.PROJECT_NAME} FindResource ({settings.ENVIRONMENT.value})",
        )
        logger.info("find_resource_graph_created", has_checkpointer=checkpointer is not None)
        return self._graph

    def _build_run_config(
        self,
        session_id: str,
        user_id: Optional[str],
        username: Optional[str],
    ) -> dict:
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

    def _initial_graph_input(self, messages: list[Message]) -> dict:
        dumped = dump_messages(messages)
        question = messages[-1].content if messages else ""
        return {
            "messages": dumped,
            "user_question": question,
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
                    Command(resume=messages[-1].content),
                    config=config,
                )
            else:
                response = await self._graph.ainvoke(
                    self._initial_graph_input(messages),
                    config=config,
                )

            state = await self._graph.aget_state(config)
            if state.next:
                interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
                return [Message(role="assistant", content=extract_interrupt_text(interrupt_value))]

            return self._process_messages(response["messages"])
        except GraphInterrupt:
            state = await self._graph.aget_state(config)
            interrupt_value = extract_interrupt_text(
                state.tasks[0].interrupts[0].value if state.tasks else None
            )
            return [Message(role="assistant", content=interrupt_value)]
        except Exception as e:
            logger.exception("find_resource_get_response_failed", session_id=session_id, error=str(e))
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
                graph_input = self._initial_graph_input(messages)

            async for output in self._graph.astream(graph_input, config):
                for node_name, node_state in output.items():
                    if node_name not in _STREAM_YIELD_NODES:
                        continue
                    if not isinstance(node_state, dict) or "messages" not in node_state:
                        continue
                    node_messages = node_state["messages"]
                    if not node_messages:
                        continue
                    last_message = node_messages[-1]
                    if isinstance(last_message, AIMessage):
                        content = ai_message_content_to_str(last_message.content)
                        if content:
                            yield StreamChunk(content=content)

            state = await self._graph.aget_state(config)
            if state.next:
                interrupt_value = extract_interrupt_text(
                    state.tasks[0].interrupts[0].value if state.tasks else None
                )
                yield StreamChunk(content=interrupt_value)
        except GraphInterrupt:
            state = await self._graph.aget_state(config)
            interrupt_value = extract_interrupt_text(
                state.tasks[0].interrupts[0].value if state.tasks else None
            )
            yield StreamChunk(content=interrupt_value)
        except Exception as e:
            logger.exception("find_resource_stream_failed", session_id=session_id, error=str(e))
            raise

    async def get_chat_history(self, session_id: str) -> list[Message]:
        if self._graph is None:
            await self.create_graph()
        state: StateSnapshot = await self._graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )
        if not state.values or "messages" not in state.values:
            return []
        return self._process_messages(state.values["messages"])

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
        logger.info("find_resource_checkpoint_cleared", session_id=session_id)
