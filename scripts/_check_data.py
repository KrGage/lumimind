"""检查记忆库和知识库数据状态"""
import json, os

mem_path = os.path.join(os.path.dirname(__file__), "..", "ai_agent", "models", "user_memory_store.json")
vs_path = os.path.join(os.path.dirname(__file__), "..", "ai_agent", "models", "vector_store.json")

print("=== User Memory Store ===")
if os.path.exists(mem_path):
    with open(mem_path) as f:
        d = json.load(f)
    ids = d.get("ids", [])
    docs = d.get("documents", [])
    metas = d.get("metadatas", [])
    print(f"Total records: {len(ids)}")
    for i in range(min(len(ids), 20)):
        meta = metas[i] if i < len(metas) else {}
        print(f"  [{meta.get('mem_type','?')}] {docs[i][:90]}")
    if len(ids) > 20:
        print(f"  ... and {len(ids)-20} more")
else:
    print("NOT FOUND - needs seeding")

print("\n=== RAG Knowledge Base ===")
if os.path.exists(vs_path):
    with open(vs_path) as f:
        d = json.load(f)
    print(f"Total documents: {len(d.get('ids', []))}")
    for i in range(min(len(d.get("ids",[])), 5)):
        print(f"  {d['documents'][i][:80]}")
else:
    print("NOT FOUND - needs ingest")
