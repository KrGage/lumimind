"""
Agent Framework — LangGraph Checkpointer（短期记忆持久化）
==========================================================

基于 SqliteSaver 将 LangGraph 对话状态持久化到 data/checkpoints.db，
以 thread_id = session_id 隔离不同会话，复用 session_api 的会话概念。

职责:
  - get_checkpointer(): SqliteSaver 单例（未安装 langgraph-checkpoint-sqlite 时降级 MemorySaver）
  - thread_config(session_id): 构造 LangGraph config
  - save_messages(session_id, msgs): 将新消息追加到指定线程的检查点
  - get_history(session_id): 读取线程历史，转为 [{"role","content"}] 供 RAG 管线使用

用法:
    from agent_framework.checkpointer import save_messages, get_history

    save_messages("conv_123", [{"role": "user", "content": "你好"}])
    history = get_history("conv_123")
"""

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)

# 检查点数据库路径: ai_agent/data/checkpoints.db
_CHECKPOINT_DB = Path(__file__).resolve().parent.parent / "data" / "checkpoints.db"

# 历史消息读取上限（防止上下文无限膨胀）
MAX_HISTORY_MESSAGES = 40

_lock = threading.Lock()
_checkpointer_instance: Optional[Any] = None
_memory_graph: Optional[Any] = None


def get_checkpointer() -> Any:
    """获取 checkpointer 单例（线程安全）。

    优先使用 SqliteSaver 持久化到磁盘；
    未安装 langgraph-checkpoint-sqlite 时降级为内存 MemorySaver（重启丢失）。
    """
    global _checkpointer_instance
    if _checkpointer_instance is not None:
        return _checkpointer_instance

    with _lock:
        if _checkpointer_instance is not None:
            return _checkpointer_instance

        try:
            import sqlite3

            from langgraph.checkpoint.sqlite import SqliteSaver

            os.makedirs(_CHECKPOINT_DB.parent, exist_ok=True)
            # 注意：SqliteSaver.from_conn_string() 返回上下文管理器，未 __enter__ 时
            # 连接锁未初始化会死锁，因此直接用 sqlite3 连接构造单例
            conn = sqlite3.connect(
                str(_CHECKPOINT_DB),
                check_same_thread=False,  # FastAPI 线程池中共享
            )
            _checkpointer_instance = SqliteSaver(conn)
            logger.info("SqliteSaver checkpointer ready: %s", _CHECKPOINT_DB)
        except ImportError:
            from langgraph.checkpoint.memory import MemorySaver

            logger.warning(
                "langgraph-checkpoint-sqlite 未安装，短期记忆降级为 MemorySaver（重启后丢失）。"
                "安装: pip install langgraph-checkpoint-sqlite"
            )
            _checkpointer_instance = MemorySaver()

    return _checkpointer_instance


def thread_config(session_id: str) -> Dict[str, Any]:
    """构造 LangGraph 调用 config，thread_id 即会话 ID。"""
    return {"configurable": {"thread_id": session_id}}


def _get_memory_graph() -> Any:
    """
    获取"记忆图"单例：一个单空节点的 StateGraph，仅用于通过
    checkpointer 按线程追加/读取 messages（operator.add 累加语义）。
    """
    global _memory_graph
    if _memory_graph is not None:
        return _memory_graph

    from .workflow import AgentState

    # 先在锁外获取 checkpointer，避免与 get_checkpointer() 的锁形成死锁
    cp = get_checkpointer()

    with _lock:
        if _memory_graph is not None:
            return _memory_graph

        graph = StateGraph(AgentState)
        graph.add_node("_noop", lambda state: {})
        graph.set_entry_point("_noop")
        graph.add_edge("_noop", END)
        _memory_graph = graph.compile(checkpointer=cp)

    return _memory_graph


def _to_langchain_message(role: str, content: str) -> BaseMessage:
    return HumanMessage(content=content) if role == "user" else AIMessage(content=content)


def save_messages(session_id: str, messages: List[Dict[str, str]]) -> None:
    """将新消息追加到指定会话线程的检查点（不覆盖历史）。"""
    if not session_id or not messages:
        return

    graph = _get_memory_graph()
    lc_messages = [
        _to_langchain_message(m.get("role", "user"), str(m.get("content", "")))
        for m in messages
        if m.get("content")
    ]
    if not lc_messages:
        return

    try:
        graph.invoke({"messages": lc_messages}, config=thread_config(session_id))
    except Exception as e:
        logger.error("Failed to save checkpoint messages for session %s: %s", session_id, e)


def get_history(session_id: str, limit: int = MAX_HISTORY_MESSAGES) -> List[Dict[str, str]]:
    """读取指定会话线程的历史消息，返回 [{"role","content"}]（最旧→最新）。

    自动跳过 SystemMessage；仅保留最近 limit 条。
    """
    if not session_id:
        return []

    graph = _get_memory_graph()
    try:
        state = graph.get_state(thread_config(session_id))
    except Exception as e:
        logger.error("Failed to read checkpoint state for session %s: %s", session_id, e)
        return []

    values = getattr(state, "values", None) or {}
    raw: List[BaseMessage] = values.get("messages", [])

    history = []
    for msg in raw:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})
        # SystemMessage / 其他类型不进入对话历史

    return history[-limit:]


def clear_history(session_id: str) -> None:
    """清空指定会话线程的检查点（删除线程）。"""
    if not session_id:
        return

    graph = _get_memory_graph()
    try:
        # delete_thread 接收 thread_id 字符串而非 config
        graph.checkpointer.delete_thread(session_id)
        # 内存图按线程缓存了 state，需一并重建
        _reset_memory_graph_cache(session_id)
    except Exception as e:
        logger.error("Failed to delete thread for session %s: %s", session_id, e)


def _reset_memory_graph_cache(session_id: str) -> None:
    """删除线程后使内存图缓存失效，确保下次读取落盘状态。"""
    global _memory_graph
    with _lock:
        _memory_graph = None


def is_persistent() -> bool:
    """当前 checkpointer 是否为磁盘持久化（SqliteSaver）。"""
    return type(get_checkpointer()).__name__ == "SqliteSaver"
