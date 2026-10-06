"""
Agent Framework — 用户长期记忆库
================================

将用户相关的长期信息向量化存入独立记忆库（与医学知识库隔离），
在 AI 问答时按 user_id 过滤检索，为"我最近的身体状况怎么样"类
问题提供个性化上下文。

记忆类型 (mem_type):
  - light_preference: 用户偏爱的办公环境光（色温/模式/备注）
  - qa_digest:        问答中得到的关键回答要点
  - physio_emotion:   生理信号 + 情绪状态记录（含时间信息）

检索策略:
  1. 语义检索: 用户问题 → 嵌入 → 余弦相似度 + 关键词混合评分
  2. 降级: 嵌入不可用（离线）时按时间倒序返回该用户近期记录

用法:
    from agent_framework.long_term_memory import (
        record_light_preference, record_physio_emotion,
        record_qa_digest, search_user_memory, recent_user_summary,
    )
"""

import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 用户记忆库独立于医学知识库（models/vector_store.json）
USER_MEMORY_DB_FILE = os.path.join(
    os.path.dirname(__file__), "..", "models", "user_memory_store.json"
)

# 记忆类型常量
MEM_LIGHT_PREFERENCE = "light_preference"
MEM_QA_DIGEST = "qa_digest"
MEM_PHYSIO_EMOTION = "physio_emotion"

# 近期摘要默认时间窗口（天）
DEFAULT_RECENT_DAYS = 7

# 单用户记忆容量上限（超出时淘汰最旧的 physio 记录）
MAX_MEMORIES_PER_USER = 500


# ── 存储单例 ─────────────────────────────────────────────────────

_memory_store = None


def get_memory_store():
    """获取用户记忆向量库单例（独立文件，避免污染医学知识库）。"""
    global _memory_store
    if _memory_store is None:
        from .rag import SimpleVectorStore

        _memory_store = SimpleVectorStore(filename=USER_MEMORY_DB_FILE)
    return _memory_store


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── 写入接口 ─────────────────────────────────────────────────────

def add_user_memory(
    user_id: str,
    text: str,
    mem_type: str,
    metadata: Optional[Dict[str, Any]] = None,
    record_id: Optional[str] = None,
) -> str:
    """
    写入一条用户长期记忆（文本会被向量化）。

    Args:
        user_id: 用户 ID
        text:    记忆正文（建议是自包含的自然语言摘要，便于语义检索）
        mem_type: light_preference / qa_digest / physio_emotion
        metadata: 额外元数据（timestamp, session_id, emotion_label 等）
        record_id: 可选指定 ID，缺省自动生成

    Returns:
        record_id
    """
    user_id = str(user_id)
    meta = {"user_id": user_id, "mem_type": mem_type, "timestamp": _now_iso()}
    if metadata:
        meta.update(metadata)

    rid = record_id or f"mem_{user_id}_{mem_type}_{uuid.uuid4().hex[:12]}"

    try:
        store = get_memory_store()
        store.add(text, meta, rid)
        _evict_if_needed(user_id)
    except Exception as e:
        # 记忆写入失败不影响主流程（生理落库 / 问答继续）
        logger.error("Failed to write user memory (%s/%s): %s", user_id, mem_type, e)
    return rid


def record_light_preference(
    user_id: str,
    light_mode: str,
    operator_note: Optional[str] = None,
    session_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> str:
    """记录用户偏爱的办公环境光（切换灯光模式时调用）。"""
    ts = timestamp or datetime.now(timezone.utc)
    note = (operator_note or "").strip()
    text = (
        f"{ts.strftime('%Y-%m-%d %H:%M')}（UTC）用户切换办公环境光为 {light_mode}"
        + (f"，备注：{note}" if note else "")
        + "。这反映了用户对该光环境模式的偏好。"
    )
    return add_user_memory(
        user_id,
        text,
        MEM_LIGHT_PREFERENCE,
        metadata={"session_id": session_id, "light_mode": light_mode, "ts": ts.isoformat()},
    )


def record_physio_emotion(
    user_id: str,
    emotion_label: str,
    emotion_score: Optional[float] = None,
    confidence: float = 1.0,
    physio: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    face_detected: bool = True,
) -> str:
    """
    记录一条生理/情绪状态（含时间信息）。

    Args:
        emotion_label: positive / neutral / negative
        emotion_score: 情绪分数 0~1（可空）
        confidence:    判定置信度
        physio:        生理信号摘要 dict，如 {"hrv_lf_hf": 3.2, "heart_rate": 78}
        session_id:    关联实验会话
        timestamp:     信号时间（缺省当前时间）
    """
    ts = timestamp or datetime.now(timezone.utc)
    label_zh = {"positive": "积极", "neutral": "平静", "negative": "消极"}.get(
        emotion_label, emotion_label
    )

    parts = [
        f"{ts.strftime('%Y-%m-%d %H:%M')}（UTC）生理情绪检测：情绪状态为{label_zh}"
        f"（{emotion_label}）"
    ]
    if emotion_score is not None:
        parts.append(f"情绪分数 {emotion_score:.2f}")
    parts.append(f"置信度 {confidence:.2f}")
    if not face_detected:
        parts.append("（未检测到人脸，信号可能不可靠）")
    if physio:
        sig = "，".join(f"{k}={v}" for k, v in physio.items())
        parts.append(f"生理信号：{sig}")
    text = "，".join(parts) + "。"

    return add_user_memory(
        user_id,
        text,
        MEM_PHYSIO_EMOTION,
        metadata={
            "session_id": session_id,
            "emotion_label": emotion_label,
            "ts": ts.isoformat(),
            **(physio or {}),
        },
    )


def record_qa_digest(
    user_id: str,
    question: str,
    answer: str,
    session_id: Optional[str] = None,
) -> str:
    """
    记录一次问答的关键内容（问题 + 回答摘要）。

    摘要策略：取回答前 500 字符，问答多轮后可升级为 LLM 摘要。
    """
    q = question.strip()[:200]
    digest = " ".join(answer.strip().split())[:500]
    ts = _now_iso()
    text = f"用户曾询问「{q}」，得到的关键回答要点：{digest}"

    return add_user_memory(
        user_id,
        text,
        MEM_QA_DIGEST,
        metadata={"session_id": session_id, "question": q, "ts": ts},
    )


# ── 检索接口 ─────────────────────────────────────────────────────

def search_user_memory(
    user_id: str,
    query: str,
    k: int = 5,
    mem_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    检索某用户的长期记忆（语义检索，嵌入不可用时降级为近期记录）。

    Returns:
        [{"text", "metadata", "id", "score"}]，按相关度/时间降序
    """
    user_id = str(user_id)
    store = get_memory_store()

    try:
        filter_meta = {"user_id": user_id}
        if mem_type:
            filter_meta["mem_type"] = mem_type
        results = store.search_with_scores(query, filter_meta=filter_meta, k=k)
        if results:
            return results
    except Exception as e:
        logger.error("User memory search failed: %s", e)

    # 降级：嵌入不可用（零向量/相似度全 0）→ 返回近期记录
    return recent_user_records(user_id, k=k, mem_type=mem_type)


def recent_user_records(
    user_id: str,
    k: int = 5,
    mem_type: Optional[str] = None,
    days: int = DEFAULT_RECENT_DAYS,
) -> List[Dict[str, Any]]:
    """按时间倒序返回某用户最近 N 天的记录（不依赖嵌入，确定性）。"""
    user_id = str(user_id)
    store = get_memory_store()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    items = []
    for rid, text, meta in zip(store.ids, store.documents, store.metadatas):
        if meta.get("user_id") != user_id:
            continue
        if mem_type and meta.get("mem_type") != mem_type:
            continue
        try:
            ts = datetime.fromisoformat(str(meta.get("ts", "")))
        except ValueError:
            ts = None
        if ts and ts < cutoff:
            continue
        items.append({"text": text, "metadata": meta, "id": rid, "score": 0.0, "ts": ts})

    items.sort(key=lambda x: x["ts"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return items[:k]


def build_physio_context(
    user_id: str,
    query: str = "",
    k: int = 6,
    days: int = DEFAULT_RECENT_DAYS,
) -> str:
    """
    为健康类问题构建用户生理记忆上下文：
    语义检索的历史生理记录 + 近期确定性趋势摘要。
    """
    user_id = str(user_id)

    semantic = search_user_memory(user_id, query, k=k, mem_type=MEM_PHYSIO_EMOTION)
    recent = recent_user_records(user_id, k=k, mem_type=MEM_PHYSIO_EMOTION, days=days)

    # 合并去重（语义命中优先）
    seen, merged = set(), []
    for item in semantic + recent:
        if item["id"] not in seen:
            seen.add(item["id"])
            merged.append(item)
    merged = merged[: k + 2]

    if not merged:
        return ""

    # 确定性趋势：统计近期情绪分布
    labels = [r["metadata"].get("emotion_label", "") for r in recent]
    if labels:
        n = len(labels)
        dist = ", ".join(
            f"{lbl} {labels.count(lbl)} 次" for lbl in dict.fromkeys(labels) if lbl
        )
        trend = f"近 {days} 天共 {n} 条生理情绪记录（{dist}）"
    else:
        trend = f"近 {days} 天无生理记录"

    lines = [f"[用户近{days}天生理情绪概况] {trend}"]
    lines += [f"- {item['text']}" for item in merged]
    return "\n".join(lines)


# ── 管理接口 ─────────────────────────────────────────────────────

def list_user_memories(user_id: str, mem_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """列出某用户全部记忆（前端管理页面用）。"""
    user_id = str(user_id)
    store = get_memory_store()
    items = []
    for rid, text, meta in zip(store.ids, store.documents, store.metadatas):
        if meta.get("user_id") != user_id:
            continue
        if mem_type and meta.get("mem_type") != mem_type:
            continue
        items.append({"id": rid, "text": text, "metadata": meta})
    return items


def clear_user_memories(user_id: str, mem_type: Optional[str] = None) -> int:
    """清空某用户记忆（可按类型），返回删除条数。"""
    user_id = str(user_id)
    store = get_memory_store()
    targets = [
        rid
        for rid, meta in zip(store.ids, store.metadatas)
        if meta.get("user_id") == user_id
        and (mem_type is None or meta.get("mem_type") == mem_type)
    ]
    for rid in targets:
        store.delete(rid)
    return len(targets)


def _evict_if_needed(user_id: str) -> None:
    """单用户记忆超出上限时淘汰最旧的 physio 记录。"""
    user_id = str(user_id)
    store = get_memory_store()
    idxs = [
        i
        for i, meta in enumerate(store.metadatas)
        if meta.get("user_id") == user_id
    ]
    if len(idxs) <= MAX_MEMORIES_PER_USER:
        return

    physio = [
        i
        for i in idxs
        if store.metadatas[i].get("mem_type") == MEM_PHYSIO_EMOTION
    ]
    # metadatas 中 ts 越小越旧
    physio.sort(key=lambda i: str(store.metadatas[i].get("ts", "")))
    n_evict = len(idxs) - MAX_MEMORIES_PER_USER
    for i in physio[:n_evict]:
        store.delete(store.ids[i])
    logger.info("Evicted %d old physio memories for user %s", n_evict, user_id)
