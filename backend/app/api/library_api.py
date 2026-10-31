"""Local management of the two JSON stores used by RAG and user memory."""

import ipaddress
import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from agent_framework.rag import SimpleVectorStore, _get_embedding, get_vector_store
from agent_framework.long_term_memory import get_memory_store
from ingest_knowledge_base import PARSERS, chunk_text

router = APIRouter(prefix="/api/library", tags=["library"])
_lock = RLock()
_uploads = Path(__file__).resolve().parents[3] / "models" / "uploads"
_cdata_catalog = Path(__file__).resolve().parents[3] / "models" / "cdata_panas_records.json"


def _local(request: Request):
    peer = request.client.host if request.client else ""
    try:
        if not ipaddress.ip_address(peer).is_loopback:
            raise ValueError()
    except ValueError:
        raise HTTPException(403, "知识库管理仅允许本机访问")
    origin = request.headers.get("origin")
    if origin and origin != str(request.base_url).rstrip("/"):
        raise HTTPException(403, "请从本机知识库页面操作")


def _store(memory=False) -> SimpleVectorStore:
    store = get_memory_store() if memory else get_vector_store()
    if not isinstance(store, SimpleVectorStore):
        raise HTTPException(503, "当前使用的向量存储不是本地 JSON，管理页面暂不支持")
    return store


def _write(store, rows):
    """Persist a complete snapshot once, then refresh the live search index."""
    ids, docs, metas, vecs = zip(*rows) if rows else ([], [], [], [])
    payload = {"ids": list(ids), "documents": list(docs), "metadatas": list(metas), "vectors": list(vecs)}
    target = Path(store.filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + "." + uuid4().hex + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as out:
            json.dump(payload, out, ensure_ascii=False, separators=(",", ":"))
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    store.ids, store.documents, store.metadatas, store.vectors = map(list, (ids, docs, metas, vecs))
    store.id_to_idx = {rid: i for i, rid in enumerate(store.ids)}
    store.lsh.clear()
    for rid, vec in zip(store.ids, store.vectors):
        store.lsh.index(rid, np.asarray(vec))


def _rows(store):
    return list(zip(store.ids, store.documents, store.metadatas, store.vectors))


def _source(meta, rid):
    return str(meta.get("source") or meta.get("filename") or rid.split("#chunk")[0])


@router.get("/documents")
def documents(request: Request):
    _local(request)
    with _lock:
        groups = {}
        for rid, text, meta, _ in _rows(_store()):
            source = _source(meta, rid)
            item = groups.setdefault(source, {"source": source, "filename": meta.get("filename") or Path(source).name,
                                              "chunks": 0, "characters": 0})
            item["chunks"] += 1
            item["characters"] += len(text)
        return {"documents": sorted(groups.values(), key=lambda x: x["filename"].lower()),
                "total_chunks": sum(x["chunks"] for x in groups.values())}


@router.get("/document")
def document(request: Request, source: str):
    _local(request)
    with _lock:
        chunks = [{"id": rid, "text": text, "metadata": meta} for rid, text, meta, _ in _rows(_store())
                  if _source(meta, rid) == source]
        if not chunks:
            raise HTTPException(404, "文献来源不存在")
        chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
        return {"source": source, "chunks": chunks}


def _index_text(source: str, filename: str, text: str, replace=False):
    if not text.strip():
        raise HTTPException(422, "文档没有可索引的文本")
    pieces = chunk_text(text, 800, 100)
    if len(pieces) > 500:
        raise HTTPException(413, "文档过大，最多 500 个片段")
    with _lock:
        store = _store()
        old = [row for row in _rows(store) if _source(row[2], row[0]) == source]
        if replace and not old:
            raise HTTPException(404, "文献来源不存在")
        if not replace and old:
            raise HTTPException(409, "同名来源已存在")
        rows = [row for row in _rows(store) if _source(row[2], row[0]) != source]
        for i, piece in enumerate(pieces):
            meta = {"source": source, "filename": filename, "chunk_index": i,
                    "total_chunks": len(pieces), "char_count": len(piece), "managed": True}
            rows.append((f"{source}#chunk{i}", piece, meta, _get_embedding(piece)))
        _write(store, rows)
    return {"source": source, "chunks": len(pieces)}


@router.post("/documents")
async def upload_document(request: Request, filename: str):
    _local(request)
    name = Path(filename.replace("\\", "/")).name
    ext = Path(name).suffix.lower()
    if not name or ext not in PARSERS:
        raise HTTPException(422, "仅支持 PDF、DOCX、TXT、MD")
    data = await request.body()
    if not data or len(data) > 20 * 1024 * 1024:
        raise HTTPException(413, "文件须小于 20 MB")
    _uploads.mkdir(parents=True, exist_ok=True)
    source = "上传/" + uuid4().hex[:12] + "_" + name
    path = _uploads / (uuid4().hex + ext)
    try:
        path.write_bytes(data)
        text = PARSERS[ext](str(path))
        result = _index_text(source, name, text or "")
        return result
    finally:
        path.unlink(missing_ok=True)


class TextUpdate(BaseModel):
    text: str = Field(min_length=1, max_length=500000)


@router.put("/document")
def update_document(request: Request, source: str, payload: TextUpdate):
    _local(request)
    return _index_text(source, Path(source).name, payload.text, replace=True)


@router.delete("/document")
def delete_document(request: Request, source: str):
    _local(request)
    with _lock:
        store = _store()
        rows = _rows(store)
        kept = [row for row in rows if _source(row[2], row[0]) != source]
        if len(kept) == len(rows):
            raise HTTPException(404, "文献来源不存在")
        _write(store, kept)
        return {"deleted_chunks": len(rows) - len(kept), "source": source}


@router.get("/memories")
def memories(request: Request, user_id: str = "local_user"):
    _local(request)
    with _lock:
        items = [{"id": rid, "text": text, "metadata": meta}
                 for rid, text, meta, _ in _rows(_store(True)) if meta.get("user_id") == user_id]
        items.sort(key=lambda x: x["metadata"].get("timestamp", ""), reverse=True)
        return {"user_id": user_id, "memories": items}


@router.get("/experiments")
def experiment_records(request: Request):
    """Read real C-data observations without conflating subjects with the live user."""
    _local(request)
    if not _cdata_catalog.is_file():
        raise HTTPException(404, "C-data 记录尚未生成；运行 scripts/import_cdata_panas.py")
    return json.loads(_cdata_catalog.read_text(encoding="utf-8"))


class MemoryInput(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=10000)
    mem_type: str = Field(default="note", max_length=64)


@router.post("/memories")
def add_memory(request: Request, payload: MemoryInput):
    _local(request)
    from datetime import datetime, timezone
    rid = "mem_manual_" + uuid4().hex
    meta = {"user_id": payload.user_id, "mem_type": payload.mem_type,
            "timestamp": datetime.now(timezone.utc).isoformat(), "manual": True}
    with _lock:
        store = _store(True)
        _write(store, _rows(store) + [(rid, payload.text, meta, _get_embedding(payload.text))])
    return {"id": rid}


@router.put("/memory")
def update_memory(request: Request, memory_id: str, payload: TextUpdate):
    _local(request)
    with _lock:
        store = _store(True)
        rows = _rows(store)
        if memory_id not in store.id_to_idx:
            raise HTTPException(404, "记忆不存在")
        i = store.id_to_idx[memory_id]
        rid, _, meta, _ = rows[i]
        rows[i] = (rid, payload.text, meta, _get_embedding(payload.text))
        _write(store, rows)
    return {"id": memory_id}


@router.delete("/memory")
def delete_memory(request: Request, memory_id: str):
    _local(request)
    with _lock:
        store = _store(True)
        rows = _rows(store)
        kept = [row for row in rows if row[0] != memory_id]
        if len(kept) == len(rows):
            raise HTTPException(404, "记忆不存在")
        _write(store, kept)
    return {"id": memory_id}
