"""
Agent Framework — RAG 问答管线
================================

将本地知识库检索与 LLM 生成结合，实现基于知识的问答。

流程:
  1. 用户提问 → 本地知识库语义搜索
  2. 检索结果评分 → 判断是否足够
  3. 若不足 → 预留联网搜索扩展点（暂未实现）
  4. 组装上下文 → 注入系统提示词 → LLM 生成回答
  5. 返回回答 + 引用来源

用法:
    from agent_framework.rag_pipeline import rag_chat

    result = await rag_chat(
        messages=[{"role": "user", "content": "光照对睡眠有什么影响？"}],
        system="你是医学健康助手...",
    )
    # result = {"reply": "...", "citations": [...], "model": "qwen-plus"}

    # 多轮记忆模式：后端通过 SqliteSaver checkpoint 维护历史，前端只需传新消息
    result = await chat_with_memory(
        session_id="conv_123",
        message="那对褪黑素呢？",
        system="你是医学健康助手...",
    )
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── 配置常量 ─────────────────────────────────────────────────────

# 知识库检索阈值：高于此分数认为本地知识足够
RELEVANCE_THRESHOLD = 0.15

# 检索返回数量
DEFAULT_TOP_K = 5

# 上下文 token 预算
CONTEXT_TOKEN_BUDGET = 2000


# ── 核心管线函数 ─────────────────────────────────────────────────

async def chat_with_memory(
    session_id: str,
    message: str,
    system: str = "",
    user_id: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    带短期记忆 + 用户长期记忆的问答入口。

    历史消息由 SqliteSaver checkpointer（data/checkpoints.db）按
    thread_id = session_id 维护，调用方只需传新消息；回答完成后
    本轮 user/assistant 消息自动追加到检查点。

    长期记忆：传入 user_id 且问题命中健康/自身状况关键词时，
    从用户记忆库（生理情绪/光偏好/问答要点）检索上下文注入回答。

    Args:
        session_id: 会话 ID（复用 session_api 的 session/对话概念）
        message: 用户本轮新消息
        system: 基础系统提示词
        user_id: 用户 ID（用于长期记忆检索，可空）
        **kwargs: 透传给 rag_chat（top_k/threshold/token_budget）

    Returns:
        与 rag_chat 相同结构的结果字典，额外包含 user_memory_used
    """
    from .checkpointer import get_history, save_messages

    history = get_history(session_id)
    messages = history + [{"role": "user", "content": message}]

    # 长期记忆：健康/自身状况关键词命中时检索用户生理记忆
    extra_context = ""
    user_memory_used = False
    if user_id and _hits_personal_health_keywords(message):
        try:
            from .long_term_memory import build_physio_context

            extra_context = build_physio_context(user_id, query=message)
            user_memory_used = bool(extra_context)
            if user_memory_used:
                logger.info("User memory injected for user %s", user_id)
        except Exception as e:
            logger.error("Failed to build user memory context: %s", e)

    result = await rag_chat(
        messages=messages, system=system, extra_context=extra_context, **kwargs
    )
    result["user_memory_used"] = user_memory_used

    # 仅在本轮回答成功时持久化，避免把错误回复写进记忆
    if result.get("reply") and not result["reply"].startswith("AI 推理暂时不可用"):
        save_messages(
            session_id,
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": result["reply"]},
            ],
        )
        # 长期记忆：记录本次问答要点
        if user_id:
            try:
                from .long_term_memory import record_qa_digest

                record_qa_digest(user_id, message, result["reply"], session_id=session_id)
            except Exception as e:
                logger.error("Failed to record qa digest: %s", e)

    return result


# 触发用户长期记忆检索的关键词（自身健康/生理/情绪状况类问题）
PERSONAL_HEALTH_KEYWORDS = [
    "身体状况", "健康状况", "健康状态", "我的健康", "最近状态", "我的状态",
    "我的情绪", "情绪状态", "最近情绪", "心理状态", "压力状况", "我的压力",
    "生理", "生理数据", "生理状况", "心率", "hrv", "心电", "ecg",
    "脑电", "eeg", "睡眠状况", "我的睡眠", "最近怎样", "最近怎么样",
]


def _hits_personal_health_keywords(message: str) -> bool:
    """判断用户问题是否与自身健康/生理/情绪状况相关。"""
    text = message.lower()
    return any(kw in text for kw in PERSONAL_HEALTH_KEYWORDS)



async def rag_chat(
    messages: List[Dict[str, str]],
    system: str = "",
    top_k: int = DEFAULT_TOP_K,
    threshold: float = RELEVANCE_THRESHOLD,
    token_budget: int = CONTEXT_TOKEN_BUDGET,
    extra_context: str = "",
) -> Dict[str, Any]:
    """
    RAG 问答主入口。

    Args:
        messages: 对话历史 [{"role": "user", "content": "..."}, ...]
        system: 基础系统提示词
        top_k: 检索返回数量
        threshold: 相关性阈值
        token_budget: 上下文 token 预算
        extra_context: 额外注入的用户长期记忆上下文（可空）

    Returns:
        {
            "reply": str,           # LLM 生成的回答
            "citations": [...],     # 引用来源列表
            "model": str,           # 使用的模型名
            "knowledge_used": bool, # 是否使用了知识库
        }
    """
    from .rag import get_vector_store, assemble_context, RetrievedChunk
    from .core_ai import chat as ai_chat

    # 1. 提取用户最新问题
    user_query = _extract_last_user_query(messages)
    if not user_query:
        return {
            "reply": "未检测到有效问题。",
            "citations": [],
            "model": None,
            "knowledge_used": False,
        }

    # 2. 本地知识库检索
    store = get_vector_store()
    if store.count() == 0:
        logger.warning("Knowledge base is empty, skipping RAG")
        return await _chat_without_rag(messages, system, extra_context)

    search_results = store.search_with_scores(user_query, k=top_k)
    logger.info(f"RAG search for '{user_query[:30]}...' → {len(search_results)} results")

    # 3. 过滤低分结果
    filtered = [r for r in search_results if r.get("score", 0) >= threshold]
    
    if not filtered:
        logger.info(f"No results above threshold {threshold}, falling back to LLM only")
        # TODO: 未来在此处触发联网搜索 skill
        return await _chat_without_rag(messages, system, extra_context)

    # 4. 组装上下文
    chunks = [
        RetrievedChunk(
            record_type=r.get("metadata", {}).get("source_type", "document"),
            record_id=r.get("id", "unknown"),
            text=r.get("text", ""),
            similarity=r.get("score", 0),
            metadata=r.get("metadata", {}),
        )
        for r in filtered
    ]
    context_str, tokens_used, selected_chunks = assemble_context(
        chunks, token_budget=token_budget
    )

    logger.info(f"Assembled context: {tokens_used} tokens, {len(selected_chunks)} chunks")

    # 5. 构建增强系统提示词（知识库上下文 + 用户长期记忆）
    enhanced_system = _build_rag_system_prompt(system, context_str)
    if extra_context:
        enhanced_system = _append_user_memory(enhanced_system, extra_context)

    # 6. 调用 LLM
    try:
        reply = await ai_chat(messages=messages, system=enhanced_system)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        reply = f"AI 推理暂时不可用：{str(e)}"

    # 7. 构建引用列表
    citations = _build_citations(selected_chunks)

    return {
        "reply": reply,
        "citations": citations,
        "model": _detect_model_name(),
        "knowledge_used": True,
    }


# ── 辅助函数 ─────────────────────────────────────────────────────

def _extract_last_user_query(messages: List[Dict[str, str]]) -> Optional[str]:
    """从对话历史中提取最后一条用户消息。"""
    for msg in reversed(messages):
        if msg.get("role") == "user" and msg.get("content"):
            return msg["content"].strip()
    return None


async def _chat_without_rag(
    messages: List[Dict[str, str]], 
    system: str,
    extra_context: str = "",
) -> Dict[str, Any]:
    """无知识库时的降级处理：直接调用 LLM（仍可注入用户长期记忆）。"""
    from .core_ai import chat as ai_chat

    if extra_context:
        system = _append_user_memory(system, extra_context)

    try:
        reply = await ai_chat(messages=messages, system=system)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        reply = f"AI 推理暂时不可用：{str(e)}"

    return {
        "reply": reply,
        "citations": [],
        "model": _detect_model_name(),
        "knowledge_used": False,
    }


def _build_rag_system_prompt(base_system: str, context: str) -> str:
    """构建包含知识库上下文的增强系统提示词。"""
    import sys
    from pathlib import Path
    _prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    if str(_prompts_dir.parent) not in sys.path:
        sys.path.insert(0, str(_prompts_dir.parent))
    from prompts.prompt_texts import build_rag_instruction
    return build_rag_instruction(base_system=base_system, context=context)


def _append_user_memory(system: str, user_context: str) -> str:
    """将用户长期记忆上下文追加到系统提示词。"""
    import sys
    from pathlib import Path
    _prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
    if str(_prompts_dir.parent) not in sys.path:
        sys.path.insert(0, str(_prompts_dir.parent))
    from prompts.prompt_texts import build_user_memory_instruction
    return build_user_memory_instruction(base_system=system, user_context=user_context)


def _build_citations(chunks: List) -> List[Dict[str, Any]]:
    """从检索结果构建引用列表。"""
    citations = []
    seen_ids = set()
    
    for i, chunk in enumerate(chunks, 1):
        # 去重
        if chunk.record_id in seen_ids:
            continue
        seen_ids.add(chunk.record_id)
        
        # 提取来源信息
        source = chunk.metadata.get("source", "未知来源")
        source_type = chunk.metadata.get("source_type", "document")
        
        citations.append({
            "index": i,
            "source": source,
            "type": source_type,
            "relevance": round(chunk.similarity, 3),
            "excerpt": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
        })
    
    return citations


def _detect_model_name() -> str:
    """检测当前使用的模型名称。"""
    import os
    # 按优先级检测
    if os.getenv("QWEN_API_KEY", "").strip() and not os.getenv("QWEN_API_KEY", "").startswith("your_"):
        return f"qwen ({os.getenv('QWEN_MODEL', 'qwen-plus')})"
    if os.getenv("DEEPSEEK_API_KEY", "").strip() and not os.getenv("DEEPSEEK_API_KEY", "").startswith("your_"):
        return f"deepseek ({os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')})"
    if os.getenv("GOOGLE_API_KEY", "").strip() and not os.getenv("GOOGLE_API_KEY", "").startswith("your_"):
        return f"gemini ({os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')})"
    return "ollama (local)"
