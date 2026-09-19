"""
提示词管理中心
==============
所有 RAG、LLM、Agent 相关的系统提示词统一在此管理。

用法:
    from prompts import (
        LUMIMIND_SYSTEM_PROMPT,
        build_rag_instruction,
    )
"""

from .prompt_texts import (
    LUMIMIND_SYSTEM_PROMPT,
    RAG_INSTRUCTION_TEMPLATE,
    LIGHTING_DESIGN_PROMPT,
    EMOTION_ANALYSIS_PROMPT,
    HEALTH_ADVICE_PROMPT,
    build_rag_instruction,
)
from .registry_init import register_all_prompts

__all__ = [
    # ── 提示词常量 ──
    "LUMIMIND_SYSTEM_PROMPT",
    "RAG_INSTRUCTION_TEMPLATE",
    "LIGHTING_DESIGN_PROMPT",
    "EMOTION_ANALYSIS_PROMPT",
    "HEALTH_ADVICE_PROMPT",
    # ── 构造函数 ──
    "build_rag_instruction",
    # ── 注册中心 ──
    "register_all_prompts",
]
