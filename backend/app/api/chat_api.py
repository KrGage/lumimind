"""
Chat API — AI 智慧问答接口
===========================

提供前端 AI 问答页面的后端支持，集成 RAG 知识库管线。

两种调用模式:
  1. 会话记忆模式（推荐）: 只传 session_id + message，历史由后端
     SqliteSaver checkpointer（data/checkpoints.db）按 thread_id 维护，
     前端不再回传完整 history。
  2. 兼容模式: 传完整 messages（老前端行为）。

流程:
  1. 接收用户问题（+可选 session_id）
  2. rag_pipeline 检索知识库 + LLM 生成回答
  3. 返回回答 + 引用来源
"""

import logging
from collections import deque
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── 对话调试日志（内存环形缓冲，最近 50 条） ───────────────────
_CHAT_DEBUG_LOG: deque = deque(maxlen=50)

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    """聊天请求（两种模式二选一）"""
    session_id: Optional[str] = Field(
        default=None,
        description="会话 ID（推荐）。传入后后端通过 checkpointer 管理多轮历史，只需配合 message 传新消息",
    )
    message: Optional[str] = Field(
        default=None,
        description="本轮新消息（配合 session_id 使用）",
    )
    messages: List[dict] = Field(
        default_factory=list,
        description="兼容模式：完整对话历史，每项包含 role (user/assistant) 和 content",
    )
    system: str = Field(
        default="",
        description="系统提示词，可选",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="用户 ID。传入后启用长期记忆（生理/光偏好/问答要点检索与记录）",
    )
    model: Optional[str] = Field(
        default=None,
        description="指定模型提供商（deepseek/qwen/glm/gemini/ollama/auto）。auto 或不传则自动降级",
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
    user_memory_used: bool = Field(
        default=False,
        description="是否注入了用户长期记忆（生理/光偏好/问答要点）",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="本次请求关联的会话 ID（记忆模式时回显）",
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

    - 记忆模式: session_id + message → checkpointer 维护多轮上下文
    - 兼容模式: messages 完整历史（旧前端）
    """
    use_memory = bool(request.session_id and request.message and request.message.strip())

    if not use_memory and not request.messages:
        return ChatResponse(
            reply="请求需包含 (session_id + message) 或非空 messages。",
            model=None,
        )

    system = request.system or DEFAULT_SYSTEM_PROMPT
    t0 = datetime.now(timezone.utc).isoformat()
    user_msg = (request.message or "").strip() or (request.messages[-1]["content"] if request.messages else "")

    # 解析模型选择：auto 或空视为自动降级
    _model_raw = (request.model or "").strip().lower()
    selected_provider = _model_raw if _model_raw and _model_raw != "auto" else None

    try:
        if use_memory:
            from agent_framework.rag_pipeline import chat_with_memory

            result = await chat_with_memory(
                session_id=request.session_id.strip(),
                message=request.message.strip(),
                system=system,
                user_id=(request.user_id or "").strip() or None,
                api_provider=selected_provider,
            )
            result["session_id"] = request.session_id.strip()
        else:
            from agent_framework.rag_pipeline import rag_chat

            result = await rag_chat(
                messages=request.messages, system=system,
                api_provider=selected_provider,
            )

        response = ChatResponse(
            reply=result["reply"],
            model=result.get("model"),
            citations=[CitationResponse(**c) for c in result.get("citations", [])],
            knowledge_used=result.get("knowledge_used", False),
            user_memory_used=result.get("user_memory_used", False),
            session_id=result.get("session_id"),
        )

        # 记录到调试日志
        _CHAT_DEBUG_LOG.append({
            "timestamp": t0,
            "session_id": request.session_id or "",
            "user_id": request.user_id or "",
            "mode": "memory" if use_memory else "full_history",
            "user_message": user_msg[:500],
            "ai_reply": response.reply[:1000],
            "model": response.model,
            "knowledge_used": response.knowledge_used,
            "user_memory_used": response.user_memory_used,
            "citations_count": len(response.citations),
            "system_prompt_preview": system[:200] + "..." if len(system) > 200 else system,
        })

        return response
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


@router.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """删除指定会话的短期记忆（checkpoint 线程）。"""
    try:
        from agent_framework.checkpointer import clear_history

        clear_history(session_id)
        return {"status": "success", "session_id": session_id}
    except ImportError:
        return {"status": "unavailable", "detail": "agent_framework 未安装"}
    except Exception as e:
        logger.error("Clear history error: %s", e)
        return {"status": "error", "detail": str(e)}


@router.get("/history/{session_id}")
async def get_chat_history(session_id: str):
    """读取指定会话的后端记忆历史（供前端刷新页面时恢复对话）。"""
    try:
        from agent_framework.checkpointer import get_history, is_persistent

        return {
            "session_id": session_id,
            "persistent": is_persistent(),
            "messages": get_history(session_id),
        }
    except ImportError:
        return {"session_id": session_id, "persistent": False, "messages": []}
    except Exception as e:
        logger.error("Get history error: %s", e)
        return {"session_id": session_id, "persistent": False, "messages": []}


@router.get("/memories")
async def list_long_term_memories(user_id: str, mem_type: Optional[str] = None):
    """查看某用户的长期记忆（生理/光偏好/问答要点）。"""
    try:
        from agent_framework.long_term_memory import list_user_memories

        return {
            "user_id": user_id,
            "memories": list_user_memories(user_id, mem_type=mem_type),
        }
    except ImportError:
        return {"user_id": user_id, "memories": []}
    except Exception as e:
        logger.error("List memories error: %s", e)
        return {"user_id": user_id, "memories": [], "error": str(e)}


@router.delete("/memories")
async def clear_long_term_memories(user_id: str, mem_type: Optional[str] = None):
    """清空某用户的长期记忆（可按类型过滤）。"""
    try:
        from agent_framework.long_term_memory import clear_user_memories

        deleted = clear_user_memories(user_id, mem_type=mem_type)
        return {"status": "success", "user_id": user_id, "deleted": deleted}
    except ImportError:
        return {"status": "unavailable", "detail": "agent_framework 未安装"}
    except Exception as e:
        logger.error("Clear memories error: %s", e)
        return {"status": "error", "detail": str(e)}


@router.get("/debug")
async def chat_debug_log(limit: int = 20):
    """
    对话调试接口：查看最近的 N 条对话记录（用户输入 + AI 回复 + 系统提示词摘要）。
    用于实时排查 prompt/模型/知识库相关问题。

    访问: http://127.0.0.1:8000/api/chat/debug?limit=20
    """
    logs = list(_CHAT_DEBUG_LOG)
    return {
        "total": len(logs),
        "limit": limit,
        "entries": logs[-limit:][::-1],  # 最新的在前
    }


@router.delete("/debug")
async def chat_debug_clear():
    """清空对话调试日志。"""
    _CHAT_DEBUG_LOG.clear()
    return {"status": "cleared"}


@router.get("/models")
async def list_available_models():
    """
    列出所有可用模型提供商及其状态。
    前端用于填充模型选择下拉框。
    """
    import os

    def _has_key(env_name: str) -> bool:
        v = os.getenv(env_name, "").strip()
        return bool(v) and not v.startswith("your_")

    models = []

    # DeepSeek
    if _has_key("DEEPSEEK_API_KEY"):
        models.append({
            "id": "deepseek",
            "name": f"DeepSeek ({os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')})",
            "available": True,
            "priority": 1,
        })

    # Qwen
    if _has_key("QWEN_API_KEY"):
        models.append({
            "id": "qwen",
            "name": f"Qwen ({os.getenv('QWEN_MODEL', 'qwen-plus')})",
            "available": True,
            "priority": 2,
        })

    # GLM
    if _has_key("GLM_API_KEY"):
        models.append({
            "id": "glm",
            "name": f"GLM 智谱 ({os.getenv('GLM_MODEL', 'glm-4-flash')})",
            "available": True,
            "priority": 3,
        })

    # Gemini
    if _has_key("GOOGLE_API_KEY"):
        models.append({
            "id": "gemini",
            "name": f"Gemini ({os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')})",
            "available": True,
            "priority": 4,
        })

    # Ollama
    try:
        from agent_framework.core_ai import is_ollama_running
        ollama_ok = await is_ollama_running()
    except ImportError:
        ollama_ok = False

    if ollama_ok:
        models.append({
            "id": "ollama",
            "name": f"Ollama ({os.getenv('OLLAMA_MODEL', 'llama3.2')})",
            "available": True,
            "priority": 10,
        })

    # 按优先级排序
    models.sort(key=lambda m: m.get("priority", 99))

    # 当前默认模型
    default_id = models[0]["id"] if models else None

    return {
        "models": models,
        "default": default_id,
        "auto": {
            "id": "auto",
            "name": "自动选择" + (f" ({models[0]['name']})" if models else " (无可用模型)"),
        },
    }


@router.get("/health")
async def chat_health():
    """
    检查 AI 推理引擎是否可用。
    供前端判断后端连接状态。
    """
    try:
        from agent_framework.core_ai import is_available as ai_available
        from agent_framework.rag import get_vector_store
        from agent_framework.checkpointer import is_persistent
        from agent_framework.rag_pipeline import _detect_model_name

        available = await ai_available()
        model_name = _detect_model_name()

        # 知识库状态
        store = get_vector_store()
        kb_count = store.count()

        return {
            "status": "ok",
            "available": available,
            "model": model_name,
            "knowledge_base": {
                "documents": kb_count,
                "status": "ready" if kb_count > 0 else "empty",
            },
            "memory": {
                "persistent": is_persistent(),
                "backend": "sqlite" if is_persistent() else "memory",
            },
        }
    except ImportError:
        return {"status": "unavailable", "available": False, "model": None}
    except Exception as e:
        return {"status": "error", "available": False, "model": None, "error": str(e)}
