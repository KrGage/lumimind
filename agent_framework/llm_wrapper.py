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
      - 消息列表 → 文本拼接
      - 同步/异步上下文兼容
      - 三层降级（Ollama → Gemini → Cloud）
    """

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """
        LangChain 风格同步调用。

        Args:
            messages: LangChain 消息列表（SystemMessage, HumanMessage, AIMessage）

        Returns:
            AIMessage 包含 AI 回复
        """
        full_prompt = ""
        for msg in messages:
            role = (
                "User" if isinstance(msg, HumanMessage)
                else "System" if isinstance(msg, SystemMessage)
                else "AI"
            )
            full_prompt += f"{role}: {msg.content}\n\n"

        try:
            loop = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

            if loop and loop.is_running():
                import concurrent.futures
                from .core_ai import generate
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(asyncio.run, generate(full_prompt)).result()
            else:
                from .core_ai import generate
                result = asyncio.run(generate(full_prompt))

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
