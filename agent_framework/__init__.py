"""
Agent Framework — 通用 AI Agent 开发框架
===========================================

基于 LangChain + LangGraph 的可复用 Agent 框架，从 AI-Healthcare-System 抽取而来。

核心能力：
  - 三层 AI 推理引擎（Ollama → Gemini → 云端 API）
  - LangGraph 工作流编排（可自定义节点和路由）
  - 提示词版本管理中心
  - 向量知识库（RAG）
  - 工具注册与调用
  - 语义缓存 + 安全护栏

快速开始：
    from agent_framework import (
        PromptRegistry,           # 提示词管理
        AgentWorkflow,            # LangGraph 工作流
        CoreAIWrapper,            # LangChain 兼容 LLM 包装器
        generate, chat, chat_stream,  # 直接 AI 推理
        ToolRegistry,             # 工具注册
        SimpleVectorStore,        # 向量知识库
        SemanticCache,            # 语义缓存
    )
"""

from .core_ai import (
    generate,
    chat,
    chat_stream,
    embed_text,
    is_available,
    get_ollama_models,
    is_ollama_running,
)
from .prompt_registry import (
    PromptRegistry,
    PromptVersion,
    get_prompt_registry,
    get_prompt,
    register_prompt,
)
from .llm_wrapper import CoreAIWrapper
from .workflow import AgentWorkflow, BaseNode, BaseRouter
from .tools import ToolRegistry, tavily_search
from .rag import (
    SimpleVectorStore,
    LocalitySensitiveHash,
    get_vector_store,
    search_similar,
    add_document,
    delete_document,
    RetrievedChunk,
    Citation,
    RAGResult,
)
from .rag_pipeline import rag_chat
from .guardrails import is_prompt_injection, redact_pii_from_text
from .semantic_cache import SemanticCache

__all__ = [
    # AI Engine
    "generate", "chat", "chat_stream", "embed_text", "is_available",
    "get_ollama_models", "is_ollama_running",
    # Prompt
    "PromptRegistry", "PromptVersion", "get_prompt_registry", "get_prompt", "register_prompt",
    # Workflow
    "AgentWorkflow", "BaseNode", "BaseRouter", "CoreAIWrapper",
    # Tools
    "ToolRegistry", "tavily_search",
    # RAG
    "SimpleVectorStore", "LocalitySensitiveHash", "get_vector_store",
    "search_similar", "add_document", "delete_document",
    "RetrievedChunk", "Citation", "RAGResult",
    # RAG Pipeline
    "rag_chat",
    # Safety
    "is_prompt_injection", "redact_pii_from_text",
    # Cache
    "SemanticCache",
]
