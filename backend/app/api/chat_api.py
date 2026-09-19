"""
Chat API — AI 智慧问答接口
===========================

提供前端 AI 问答页面的后端支持，集成 RAG 知识库管线。

流程:
  1. 接收用户问题
  2. 调用 rag_pipeline.rag_chat() → 知识库检索 + LLM 生成
  3. 返回回答 + 引用来源
"""

import logging
from typing import Optional, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """聊天请求"""
    messages: list[dict] = Field(
        default_factory=list,
        description="对话历史，每项包含 role (user/assistant) 和 content",
    )
    system: str = Field(
        default="",
        description="系统提示词，可选",
    )


class CitationResponse(BaseModel):
    """引用来源"""
    index: int = Field(description="引用编号")
    source: str = Field(description="来源名称")
    type: str = Field(description="来源类型")
    relevance: float = Field(description="相关性分数")
    excerpt: str = Field(description="摘录内容")


class ChatResponse(BaseModel):
    """聊天响应"""
    reply: str = Field(description="AI 回复内容")
    model: Optional[str] = Field(default=None, description="实际使用的模型名称")
    citations: List[CitationResponse] = Field(
        default_factory=list,
        description="引用来源列表",
    )
    knowledge_used: bool = Field(
        default=False,
        description="是否使用了知识库",
    )


# ── 默认系统提示词（从 prompts 统一管理模块获取） ─────────────

import sys
from pathlib import Path
_prompts_dir = Path(__file__).resolve().parent.parent.parent.parent / "prompts"
if str(_prompts_dir.parent) not in sys.path:
    sys.path.insert(0, str(_prompts_dir.parent))

from prompts.prompt_texts import LUMIMIND_SYSTEM_PROMPT

DEFAULT_SYSTEM_PROMPT = LUMIMIND_SYSTEM_PROMPT


@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    AI 智慧问答接口。

    接收多轮对话历史，通过 RAG 管线检索知识库并生成回答。
    """
    try:
        from agent_framework.rag_pipeline import rag_chat

        result = await rag_chat(
            messages=request.messages,
            system=request.system or DEFAULT_SYSTEM_PROMPT,
        )

        return ChatResponse(
            reply=result["reply"],
            model=result.get("model"),
            citations=[CitationResponse(**c) for c in result.get("citations", [])],
            knowledge_used=result.get("knowledge_used", False),
        )
    except ImportError as e:
        logger.warning("agent_framework not available for chat: %s", e)
        return ChatResponse(
            reply="AI 推理引擎暂不可用，请检查 agent_framework 模块是否已正确安装。",
            model=None,
        )
    except Exception as e:
        logger.error("Chat API error: %s", e)
        return ChatResponse(
            reply=f"AI 服务暂时不可用，请稍后重试。错误: {str(e)}",
            model=None,
        )


@router.get("/health")
async def chat_health():
    """
    检查 AI 推理引擎是否可用。
    供前端判断后端连接状态。
    """
    try:
        from agent_framework.core_ai import is_available as ai_available, get_ollama_models
        from agent_framework.rag import get_vector_store

        available = await ai_available()
        ollama_models = await get_ollama_models()
        model_name = ollama_models[0] if ollama_models else None

        # 知识库状态
        store = get_vector_store()
        kb_count = store.count()

        return {
            "status": "ok",
            "available": available,
            "model": model_name or "qwen/cloud",
            "knowledge_base": {
                "documents": kb_count,
                "status": "ready" if kb_count > 0 else "empty",
            },
        }
    except ImportError:
        return {"status": "unavailable", "available": False, "model": None}
    except Exception as e:
        return {"status": "error", "available": False, "model": None, "error": str(e)}
"""
Chat API — AI 智慧问答接口
===========================

提供前端 AI 问答页面的后端支持，委托给 agent_framework.core_ai.chat()。
"""

import logging
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """聊天请求"""
    messages: list[dict] = Field(
        default_factory=list,
        description="对话历史，每项包含 role (user/assistant) 和 content",
    )
    system: str = Field(
        default="",
        description="系统提示词，可选",
    )


class ChatResponse(BaseModel):
    """聊天响应"""
    reply: str = Field(description="AI 回复内容")
    model: Optional[str] = Field(default=None, description="实际使用的模型名称")


@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    AI 智慧问答接口。

    接收多轮对话历史，返回 AI 回复。
    内部通过 agent_framework.core_ai.chat() 进行三层降级推理。
    """
    try:
        from agent_framework.core_ai import chat as ai_chat

        reply = await ai_chat(
            messages=request.messages,
            system=request.system or (
                "你是 Lumimind 光环境智能助手，专精于通过光照调节人的情绪和生理状态。"
                "请用中文回答，提供具体、科学的光环境建议。"
            ),
        )

        return ChatResponse(reply=reply, model="auto")
    except ImportError:
        logger.warning("agent_framework not available for chat")
        return ChatResponse(
            reply="AI 推理引擎暂不可用，请检查 agent_framework 模块是否已正确安装。",
            model=None,
        )
    except Exception as e:
        logger.error("Chat API error: %s", e)
        return ChatResponse(
            reply=f"AI 服务暂时不可用，请稍后重试。错误: {str(e)}",
            model=None,
        )


@router.get("/health")
async def chat_health():
    """
    检查 AI 推理引擎是否可用。
    供前端判断后端连接状态。
    """
    try:
        from agent_framework.core_ai import is_available as ai_available, get_ollama_models

        available = await ai_available()
        ollama_models = await get_ollama_models()
        model_name = ollama_models[0] if ollama_models else None

        return {
            "status": "ok",
            "available": available,
            "model": model_name or "gemini/cloud",
        }
    except ImportError:
        return {"status": "unavailable", "available": False, "model": None}
    except Exception as e:
        return {"status": "error", "available": False, "model": None, "error": str(e)}
