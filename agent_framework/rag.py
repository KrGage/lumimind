"""
Agent Framework — 向量知识库 (RAG)
===================================

基于 LSH（局部敏感哈希）+ 混合评分（余弦相似度 + 关键词匹配）的向量知识库。

存储后端：
  1. QdrantVectorStore（如果配置了 QDRANT_HOST）
  2. SimpleVectorStore（本地 JSON + scikit-learn 余弦相似度，默认）

用法:
    from agent_framework.rag import (
        get_vector_store, search_similar, add_document,
    )

    store = get_vector_store()
    store.add("这是一段知识文本", {"source": "论文A"}, "doc_001")

    results = store.search("知识查询")
    scores = store.search_with_scores("知识查询", k=5)

从 AI-Healthcare-System 抽取而来，已移除医疗领域特化逻辑和微服务路由。
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────

DEFAULT_CONTEXT_TOKEN_BUDGET = 3000
DEFAULT_MAX_CHUNKS = 10
DB_FILE = os.path.join(os.path.dirname(__file__), "..", "models", "vector_store.json")


# ── Embedding 生成（通过 core_ai） ──────────────────────────────────

def _get_embedding(text: str, task_type: str = "retrieval_document") -> List[float]:
    """通过 core_ai 生成嵌入向量。"""
    try:
        from .core_ai import embed_text
        return embed_text(text, task_type=task_type)
    except Exception:
        logger.warning("core_ai embedding unavailable, using zero vector fallback.")
        return [0.0] * 768


def _get_query_embedding(text: str) -> List[float]:
    return _get_embedding(text, task_type="retrieval_query")


# ── RAG 数据类 ─────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """检索到的单个上下文块。"""
    record_type: str
    record_id: str
    text: str
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def citation_key(self) -> str:
        return f"{self.record_type}:{self.record_id}"

    def estimated_tokens(self) -> int:
        """粗略 token 估算（4 字符 ≈ 1 token）。"""
        return max(1, len(self.text) // 4)


@dataclass
class Citation:
    """将生成文本链接回来源记录的引用。"""
    record_type: str
    record_id: str
    record_name: str
    relevance: float
    excerpt: str = ""


@dataclass
class RAGResult:
    """RAG 流水线结果（含引用）。"""
    answer: str
    citations: List[Citation] = field(default_factory=list)
    context_chunks_used: int = 0
    total_context_tokens: int = 0
    model_used: str = ""
    grounded: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [
                {
                    "record_type": c.record_type,
                    "record_id": c.record_id,
                    "record_name": c.record_name,
                    "relevance": round(c.relevance, 3),
                    "excerpt": c.excerpt,
                }
                for c in self.citations
            ],
            "metadata": {
                "context_chunks_used": self.context_chunks_used,
                "total_context_tokens": self.total_context_tokens,
                "model_used": self.model_used,
                "grounded": self.grounded,
            },
        }


def assemble_context(
    chunks: List[RetrievedChunk],
    token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> Tuple[str, int, List[RetrievedChunk]]:
    """
    将检索块组装为上下文字符串，控制在 token 预算内。

    Returns:
        (context_string, total_tokens_used, selected_chunks)
    """
    selected = []
    total_tokens = 0

    for chunk in chunks[:max_chunks]:
        chunk_tokens = chunk.estimated_tokens()
        if total_tokens + chunk_tokens > token_budget:
            break
        selected.append(chunk)
        total_tokens += chunk_tokens

    context_parts = []
    for i, chunk in enumerate(selected, 1):
        source = f"[{chunk.record_type.title()} #{chunk.record_id}]"
        context_parts.append(f"{i}. {source} {chunk.text}")

    return "\n".join(context_parts), total_tokens, selected


# ── 辅助函数 ──────────────────────────────────────────────────────

def _normalize_str(value: Any) -> str:
    return str(value).strip()


def _metadata_matches(metadata: Dict[str, Any], filter_meta: Dict[str, Any]) -> bool:
    """检查元数据是否匹配过滤条件。"""
    for key, expected in filter_meta.items():
        if key not in metadata:
            return False
        if _normalize_str(metadata[key]) != _normalize_str(expected):
            return False
    return True


# ── 局部敏感哈希 (LSH) ────────────────────────────────────────────

class LocalitySensitiveHash:
    """
    用于快速近似最近邻 (ANN) 搜索的局部敏感哈希。

    使用随机投影超平面将高维空间分区到哈希桶中。
    """

    def __init__(self, num_tables: int = 5, hash_size: int = 6):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.dim: Optional[int] = None
        self.tables: List[Dict[str, Any]] = []

    def _init_tables(self, dim: int) -> None:
        self.dim = dim
        self.tables = []
        from collections import defaultdict

        rng = np.random.RandomState(42)
        for _ in range(self.num_tables):
            planes = rng.normal(0, 1, (self.hash_size, dim))
            self.tables.append({"planes": planes, "buckets": defaultdict(list)})

    def _hash(self, planes: np.ndarray, vector: np.ndarray) -> int:
        projection = np.dot(planes, vector)
        bits = projection > 0
        val = 0
        for b in bits:
            val = (val << 1) | int(b)
        return val

    def index(self, record_id: str, vector: np.ndarray) -> None:
        if self.dim is None:
            self._init_tables(len(vector))
        for table in self.tables:
            hash_val = self._hash(table["planes"], vector)
            if record_id not in table["buckets"][hash_val]:
                table["buckets"][hash_val].append(record_id)

    def query(self, query_vector: np.ndarray) -> set:
        if self.dim is None:
            return set()
        candidates = set()
        for table in self.tables:
            hash_val = self._hash(table["planes"], query_vector)
            candidates.update(table["buckets"][hash_val])
        return candidates

    def clear(self) -> None:
        self.dim = None
        self.tables = []


# ── 简单向量存储（默认后端） ──────────────────────────────────────

class SimpleVectorStore:
    """
    基于 JSON + scikit-learn 余弦相似度的持久化向量存储。

    嵌入通过 core_ai 生成。支持 LSH 加速的 ANN 搜索和混合评分。
    """

    def __init__(self, filename: Optional[str] = None):
        self.filename = filename or DB_FILE
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.vectors: List[List[float]] = []
        self.ids: List[str] = []
        self.id_to_idx: Dict[str, int] = {}
        self.lsh = LocalitySensitiveHash()
        self.load()

    def load(self) -> None:
        """从 JSON 文件加载（避免 pickle 反序列化风险）。"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                self.documents = data.get("documents", []) or []
                self.metadatas = data.get("metadatas", []) or []
                self.vectors = data.get("vectors", []) or []
                self.ids = data.get("ids", []) or []
                self.id_to_idx = {rid: i for i, rid in enumerate(self.ids)}

                # 重建 LSH 索引
                self.lsh.clear()
                for record_id, vec in zip(self.ids, self.vectors):
                    self.lsh.index(record_id, np.array(vec))

                logger.info("Loaded Vector Store: %d records, LSH indexed.", len(self.ids))
                return
            except Exception:
                logger.error("Failed to load vector store JSON")

        # 尝试从旧格式迁移（可选）
        legacy_pkl = os.path.splitext(self.filename)[0] + ".pkl"
        if os.path.exists(legacy_pkl) and os.getenv("ALLOW_PICKLE_MIGRATION", "").strip().lower() in {
            "1", "true", "yes", "on",
        }:
            try:
                import pickle
                with open(legacy_pkl, "rb") as f:
                    data = pickle.load(f) or {}
                self.documents = data.get("documents", []) or []
                self.metadatas = data.get("metadatas", []) or []
                self.vectors = data.get("vectors", []) or []
                self.ids = data.get("ids", []) or []
                self.id_to_idx = {rid: i for i, rid in enumerate(self.ids)}
                self.save()
                self.lsh.clear()
                for record_id, vec in zip(self.ids, self.vectors):
                    self.lsh.index(record_id, np.array(vec))
                logger.warning("Migrated legacy pickle store to JSON. Disable ALLOW_PICKLE_MIGRATION.")
            except Exception:
                logger.error("Failed to migrate legacy pickle store")

    def save(self) -> None:
        """持久化到 JSON 文件（原子写入）。"""
        try:
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            tmp_path = self.filename + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "documents": self.documents,
                        "metadatas": self.metadatas,
                        "vectors": self.vectors,
                        "ids": self.ids,
                    },
                    f,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            os.replace(tmp_path, self.filename)
        except Exception:
            logger.error("Failed to save vector store", exc_info=True)

    def add(self, text: str, metadata: Dict[str, Any], record_id: str) -> None:
        """添加或更新文档。"""
        vector = _get_embedding(text)

        if record_id in self.id_to_idx:
            idx = self.id_to_idx[record_id]
            self.documents[idx] = text
            self.metadatas[idx] = metadata
            self.vectors[idx] = vector
        else:
            idx = len(self.ids)
            self.documents.append(text)
            self.metadatas.append(metadata)
            self.vectors.append(vector)
            self.ids.append(record_id)
            self.id_to_idx[record_id] = idx

        self.lsh.index(record_id, np.array(vector))
        self.save()

    def delete(self, record_id: str) -> bool:
        """按 ID 删除。"""
        if record_id in self.id_to_idx:
            idx = self.id_to_idx[record_id]
            self.documents.pop(idx)
            self.metadatas.pop(idx)
            self.vectors.pop(idx)
            self.ids.pop(idx)

            self.id_to_idx = {rid: i for i, rid in enumerate(self.ids)}

            self.lsh.clear()
            for rid, vec in zip(self.ids, self.vectors):
                self.lsh.index(rid, np.array(vec))

            self.save()
            return True
        return False

    def _hybrid_score(self, query: str, document_text: str, similarity_score: float) -> float:
        """计算结合向量相似度和关键词匹配的混合评分。"""
        query_words = set(query.lower().split())
        doc_lower = document_text.lower()

        stop_words = {
            "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
            "to", "of", "in", "on", "at", "for", "的", "了", "是", "在", "和",
        }
        filtered = {w for w in query_words if w not in stop_words and len(w) > 2}
        if not filtered:
            return similarity_score

        matches = 0
        for word in filtered:
            if (
                f" {word} " in f" {doc_lower} "
                or doc_lower.startswith(f"{word} ")
                or doc_lower.endswith(f" {word}")
            ):
                matches += 1

        boost = min(0.05 * matches, 0.20)
        return similarity_score + boost

    def search(
        self,
        query: str,
        filter_meta: Optional[Dict[str, Any]] = None,
        k: int = 3,
    ) -> List[str]:
        """语义搜索，返回文档文本列表。"""
        results = self.search_with_scores(query, filter_meta, k)
        return [r["text"] for r in results]

    def search_with_scores(
        self,
        query: str,
        filter_meta: Optional[Dict[str, Any]] = None,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """语义搜索，返回带评分和元数据的结果。"""
        if not self.vectors:
            return []

        # 修复 ID 不一致
        if len(self.ids) != len(self.vectors):
            self.ids = [f"auto_id_{i}" for i in range(len(self.vectors))]
            self.id_to_idx = {rid: i for i, rid in enumerate(self.ids)}
            self.lsh.clear()
            for record_id, vec in zip(self.ids, self.vectors):
                self.lsh.index(record_id, np.array(vec))

        query_vector = _get_query_embedding(query)
        q_vec = np.array([query_vector])

        # LSH 候选剪枝
        use_lsh = len(self.ids) > 10
        candidates = set()
        if use_lsh:
            candidates = self.lsh.query(np.array(query_vector))

        if use_lsh and candidates:
            indices_to_scan = [self.id_to_idx[cid] for cid in candidates if cid in self.id_to_idx]
        else:
            indices_to_scan = list(range(len(self.ids)))

        if not indices_to_scan:
            return []

        candidate_vectors = [self.vectors[idx] for idx in indices_to_scan]
        vec_matrix = np.array(candidate_vectors)

        from sklearn.metrics.pairwise import cosine_similarity

        sim_scores = cosine_similarity(q_vec, vec_matrix)[0]

        # 混合评分
        hybrid_scores = []
        for idx, score in enumerate(sim_scores):
            orig_idx = indices_to_scan[idx]
            h_score = self._hybrid_score(query, self.documents[orig_idx], score)
            hybrid_scores.append(h_score)
        hybrid_scores = np.array(hybrid_scores)

        sorted_indices = hybrid_scores.argsort()[::-1]

        results = []
        count = 0

        for idx in sorted_indices:
            original_idx = indices_to_scan[idx]
            if hybrid_scores[idx] <= 0.0:
                break
            if filter_meta and not _metadata_matches(self.metadatas[original_idx], filter_meta):
                continue
            results.append({
                "text": self.documents[original_idx],
                "metadata": self.metadatas[original_idx],
                "id": self.ids[original_idx],
                "score": float(hybrid_scores[idx]),
            })
            count += 1
            if count >= k:
                break

        return results

    def count(self) -> int:
        """返回文档总数。"""
        return len(self.ids)


# ── 全局单例与快捷函数 ───────────────────────────────────────────

_store: Optional[SimpleVectorStore] = None


def get_vector_store(filename: Optional[str] = None) -> SimpleVectorStore:
    """
    获取向量存储单例。

    优先级:
      1. QdrantVectorStore（如果配置了 QDRANT_HOST）
      2. SimpleVectorStore（默认）
    """
    global _store
    if _store is None:
        # 本地优先模式（EU AI Act 合规）
        local_safety = os.environ.get("LOCAL_FIRST_SAFETY", "").strip().lower() in {"1", "true", "yes", "on"}
        if local_safety:
            _store = SimpleVectorStore(filename=filename)
            return _store

        # 尝试 Qdrant
        qdrant_host = os.environ.get("QDRANT_HOST")
        if qdrant_host:
            try:
                from .qdrant_store import QdrantVectorStore
                _store = QdrantVectorStore()
                return _store
            except Exception as e:
                logger.warning("Qdrant init failed, falling back: %s", e)

        # 默认
        _store = SimpleVectorStore(filename=filename)
    return _store


def search_similar(
    query: str,
    filter_meta: Optional[Dict[str, Any]] = None,
    k: int = 3,
) -> List[str]:
    """快捷语义搜索。"""
    return get_vector_store().search(query, filter_meta, k)


def add_document(text: str, metadata: Dict[str, Any], record_id: str) -> None:
    """快捷添加文档。"""
    get_vector_store().add(text, metadata, record_id)


def delete_document(record_id: str) -> bool:
    """快捷删除文档。"""
    return get_vector_store().delete(record_id)
