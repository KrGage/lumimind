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

async def rag_chat(
    messages: List[Dict[str, str]],
    system: str = "",
    top_k: int = DEFAULT_TOP_K,
    threshold: float = RELEVANCE_THRESHOLD,
    token_budget: int = CONTEXT_TOKEN_BUDGET,
) -> Dict[str, Any]:
    """
    RAG 问答主入口。

    Args:
        messages: 对话历史 [{"role": "user", "content": "..."}, ...]
        system: 基础系统提示词
        top_k: 检索返回数量
        threshold: 相关性阈值
        token_budget: 上下文 token 预算

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
        return await _chat_without_rag(messages, system)

    search_results = store.search_with_scores(user_query, k=top_k)
    logger.info(f"RAG search for '{user_query[:30]}...' → {len(search_results)} results")

    # 3. 过滤低分结果
    filtered = [r for r in search_results if r.get("score", 0) >= threshold]
    
    if not filtered:
        logger.info(f"No results above threshold {threshold}, falling back to LLM only")
        # TODO: 未来在此处触发联网搜索 skill
        return await _chat_without_rag(messages, system)

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

    # 5. 构建增强系统提示词
    enhanced_system = _build_rag_system_prompt(system, context_str)

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
    system: str
) -> Dict[str, Any]:
    """无知识库时的降级处理：直接调用 LLM。"""
    from .core_ai import chat as ai_chat

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
