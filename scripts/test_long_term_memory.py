import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ai_agent"))

from agent_framework import long_term_memory as ltm

UID = "test_user_ltm"

# 0. 清场
ltm.clear_user_memories(UID)

# 1. 写入三类长期记忆
ltm.record_light_preference(UID, "warm_focus", operator_note="傍晚阅读", session_id="s1")
for i, (label, score) in enumerate(
    [("negative", 0.22), ("negative", 0.30), ("neutral", 0.55), ("positive", 0.80)]
):
    ltm.record_physio_emotion(
        UID,
        emotion_label=label,
        emotion_score=score,
        confidence=0.88,
        physio={"hrv_lf_hf": 2.1 + i * 0.3, "heart_rate": 76 - i},
        session_id="s1",
        timestamp=datetime(2026, 10, 4, 9 + i, 0, tzinfo=timezone.utc),
    )
ltm.record_qa_digest(UID, "什么色温适合晚间工作？", "建议 3000K 左右暖白光，减少蓝光暴露…", session_id="s1")

mems = ltm.list_user_memories(UID)
print(f"1. 写入记忆数: {len(mems)} (期望 6)")
assert len(mems) == 6
types = {m["metadata"]["mem_type"] for m in mems}
print("   类型:", types)
assert types == {"light_preference", "physio_emotion", "qa_digest"}

# 2. 生理记忆构建（应含趋势概况 + 时间信息）
ctx = ltm.build_physio_context(UID, query="我最近的身体状况怎么样")
print("2. 生理上下文片段:")
print(ctx[:400])
assert "生理情绪概况" in ctx and "negative 2 次" in ctx and "hrv" in ctx

# 3. 检索用户隔离（另一用户应为空）
assert ltm.list_user_memories("someone_else") == []
print("3. 用户隔离 OK")

# 4. 健康关键词路由
from agent_framework.rag_pipeline import _hits_personal_health_keywords

hits = ["我最近的身体状况如何", "我的心率怎么样", "看看我的情绪状态", "我的睡眠状况"]
misses = ["什么是黑视素等效照度", "推荐一个阅读灯"]
assert all(_hits_personal_health_keywords(q) for q in hits)
assert not any(_hits_personal_health_keywords(q) for q in misses)
print("4. 关键词路由 OK")

# 5. chat_with_memory 集成（注入 + qa_digest 记录；LLM 失败时仍验证流程）
from agent_framework.rag_pipeline import chat_with_memory


async def main():
    ltm.clear_user_memories(UID, mem_type="qa_digest")
    result = await chat_with_memory(
        session_id="ltm_test_conv",
        message="我最近的身体健康状况怎么样？",
        user_id=UID,
    )
    print("5. chat_with_memory:", {k: result.get(k) for k in ("reply", "user_memory_used")})
    assert result.get("user_memory_used") is True
    digests = ltm.list_user_memories(UID, mem_type="qa_digest")
    assert len(digests) == 1, "qa_digest 应记录 1 条"
    print("   qa_digest 已记录:", digests[0]["text"][:80])

    # 清理
    ltm.clear_user_memories(UID)
    from agent_framework.checkpointer import clear_history
    clear_history("ltm_test_conv")
    print("LONG-TERM MEMORY TEST OK")


asyncio.run(main())
