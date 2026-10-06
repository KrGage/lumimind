"""
Agent Framework — LangChain 兼容 LLM 包装器
=============================================

将 core_ai 多层级 AI 推理引擎包装为 LangChain 兼容的 .invoke() 接口，
供 LangGraph Agent 使用。

用法:
    from agent_framework.llm_wrapper import CoreAIWrapper
    llm = CoreAIWrapper()
    response = llm.invoke([SystemMessage(content="..."), HumanMessage(content="...")])
"""

import asyncio
import logging
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

AI_GENERATION_FAILURE_MESSAGE = "AI is temporarily unavailable. Please try again shortly."


class CoreAIWrapper:
    """
    LangChain 兼容的 LLM 包装器，内部委托给 core_ai 多层级推理引擎。

    自动处理：
      - LangChain 消息列表 → core_ai.chat 结构化消息
      - 同步/异步上下文兼容
      - 三层降级（Ollama → Gemini → Cloud）
    """

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """
        LangGraph 风格同步调用。

        优先用 core_ai.chat 结构化消息接口（system + 对话历史），
        避免 generate 裸文本拼接导致部分模型回显 prompt；
        无对话消息时降级 generate。

        Args:
            messages: LangChain 消息列表（SystemMessage, HumanMessage, AIMessage）

        Returns:
            AIMessage 包含 AI 回复
        """
        system_parts = [m.content for m in messages if isinstance(m, SystemMessage)]
        chat_messages = [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in messages
            if isinstance(m, (HumanMessage, AIMessage))
        ]

        try:
            from .core_ai import chat as ai_chat, generate

            if chat_messages:
                system = "\n\n".join(system_parts) or None
                coro = (
                    ai_chat(messages=chat_messages, system=system)
                    if system
                    else ai_chat(messages=chat_messages)
                )
            else:
                # 全是 SystemMessage 等无对话内容的退化情况
                coro = generate("\n\n".join(system_parts))

            result = self._run_async(coro)

            if result:
                return AIMessage(content=result)
            return AIMessage(content=AI_GENERATION_FAILURE_MESSAGE)

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Quota" in err_str:
                return AIMessage(
                    content="⚠️ **Quota Exceeded.** Please wait a moment or configure a local Ollama model "
                            "for unlimited free inference."
                )
            logger.error("AI generation failed: %s", e)
            return AIMessage(content=AI_GENERATION_FAILURE_MESSAGE)

    @staticmethod
    def _run_async(coro):
        """在同步上下文运行协程，兼容已有事件循环的线程池场景。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)
