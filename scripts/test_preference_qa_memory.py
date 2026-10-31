"""
验证：光偏好记忆 + 历史问答要点记忆（全部经 LangGraph 问答验证）
================================================================

1. 写入多条用户偏好光记录（模拟用户多次切换灯光模式）
2. 模拟历史问答（记录 qa_digest 要点）
3. 通过 LangGraph 工作流提问验证：
   Q1: "我喜欢什么样的办公环境光？"      → 应命中偏好光记录
   Q2: "我之前都问过你什么问题？"        → 应命中历史问答要点
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai_agent"))

UID = "local_user"                # 与生产侧 _memory_user_id() 一致
SESSION = "pref_qa_verify_conv"   # LangGraph thread_id

from agent_framework import long_term_memory as ltm  # noqa: E402


# ── 第 1 步：写入偏好光 + 历史问答记忆 ────────────────────────────

def seed_memories():
    ltm.clear_user_memories(UID, mem_type="light_preference")
    ltm.clear_user_memories(UID, mem_type="qa_digest")

    now = datetime.now(timezone.utc)

    # 模拟用户多次切换灯光（生产中由 crud.update_light_mode 自动写入）
    prefs = [
        ("warm_reading", "傍晚阅读，喜欢暖光不刺眼", now - timedelta(days=3, hours=4)),
        ("warm_reading", "晚上加班也调回了暖光模式", now - timedelta(days=2, hours=2)),
        ("warm_reading", "睡前用 2700K 暖光放松", now - timedelta(days=1, hours=1)),
        ("cool_focus", "上午需要专注时用冷白光", now - timedelta(days=2, hours=10)),
    ]
    for mode, note, ts in prefs:
        ltm.record_light_preference(UID, mode, operator_note=note, session_id="pref_demo", timestamp=ts)

    # 模拟历史问答要点（生产中由 chat_with_memory 自动写入）
    qas = [
        ("什么色温适合晚上工作？", "建议 3000K 左右暖白光，减少蓝光暴露，保护褪黑素分泌。"),
        ("蓝光对睡眠有什么影响？", "夜间蓝光会抑制褪黑素，建议睡前 2 小时切换暖光并降低照度。"),
        ("mEDI 是什么指标？", "黑视素等效照度，衡量光照对非视觉效应（昼夜节律）影响的指标。"),
    ]
    for q, a in qas:
        ltm.record_qa_digest(UID, q, a, session_id="pref_demo")

    print(f"偏好光记录 {len(ltm.list_user_memories(UID, mem_type='light_preference'))} 条, "
          f"问答要点 {len(ltm.list_user_memories(UID, mem_type='qa_digest'))} 条")


# ── 第 2 步：LangGraph 工作流（Analyst 检索偏好+问答记忆） ────────

def build_workflow():
    from agent_framework.llm_wrapper import CoreAIWrapper
    from agent_framework.workflow import AgentWorkflow, AgentState, BaseNode

    class PersonalMemoryAnalystNode(BaseNode):
        """检索用户偏好光 + 历史问答要点，注入 analysis_results。"""

        def run(self, state: AgentState) -> dict:
            msgs = state.get("messages", [])
            query = msgs[-1].content if msgs else ""

            wants_light = any(k in query for k in ("光", "灯", "色温", "偏好", "喜欢"))
            wants_qa = any(k in query for k in ("问", "之前", "上次", "历史"))

            lines = []
            if wants_light:
                lights = ltm.recent_user_records(UID, k=5, mem_type="light_preference", days=30)
                if lights:
                    lines.append("[用户偏好光记录]")
                    lines += [f"- {r['text']}" for r in lights]
            if wants_qa:
                qas = ltm.recent_user_records(UID, k=5, mem_type="qa_digest", days=30)
                if qas:
                    lines.append("[用户历史问答要点]")
                    lines += [f"- {r['text']}" for r in qas]
            if not lines:
                for r in ltm.recent_user_records(UID, k=4, days=30):
                    lines.append(f"- {r['text']}")
            return {"analysis_results": "\n".join(lines) or "无用户记录"}

    return AgentWorkflow(
        llm=CoreAIWrapper(),
        system_prompt=(
            "你是 LumiMind 光环境智能助手。回答必须优先依据 ANALYSIS RESULTS 中的"
            "用户个人记录（偏好光/历史问答），结合记录给出个性化回答；如无相关记录则如实说明。"
            "请用中文回答。"
        ),
        enable_search=False,
        enable_guardrail=False,
        enable_analysis=True,
        analysis_node=PersonalMemoryAnalystNode(),
        supervisor_kwargs={
            "analyze_keywords": ["偏好", "喜欢", "之前", "问过", "历史", "光", "灯", "色温", "推荐", "怎么样", "最近"],
        },
        use_checkpointer=True,
    )


def ask(workflow, question: str) -> str:
    from agent_framework.checkpointer import thread_config, clear_history
    from langchain_core.messages import HumanMessage

    clear_history(SESSION)
    result = workflow.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=thread_config(SESSION),
    )
    return result["messages"][-1].content


def main():
    print("== 1. 写入偏好光 + 历史问答记忆 ==")
    seed_memories()

    wf = build_workflow()

    print("\n== 2. LangGraph 验证：偏好光记忆 ==")
    q1 = "我喜欢什么样的办公环境光？帮我总结一下我的用光偏好。"
    print(f"用户: {q1}")
    a1 = ask(wf, q1)
    print(f"LumiMind: {a1}\n")
    ok1 = ("暖" in a1 or "warm" in a1.lower()) and ("阅读" in a1 or "专注" in a1 or "冷白" in a1)
    print(f"验证(偏好光): 提及暖光偏好+场景={ok1}")
    assert ok1, a1

    print("== 3. LangGraph 验证：历史问答记忆 ==")
    q2 = "我之前都问过你哪些问题？"
    print(f"用户: {q2}")
    a2 = ask(wf, q2)
    print(f"LumiMind: {a2}\n")
    ok2 = ("色温" in a2 or "蓝光" in a2 or "mEDI" in a2 or "睡眠" in a2)
    print(f"验证(历史问答): 复现了之前的问题={ok2}")
    assert ok2, a2

    print("\n== 4. 用户隔离验证 ==")
    assert ltm.list_user_memories("stranger", mem_type="light_preference") == []
    print("他人无法检索到该用户记忆 OK")

    # 清理本次验证写入（保留 DEAP 生理演示数据）
    ltm.clear_user_memories(UID, mem_type="light_preference")
    ltm.clear_user_memories(UID, mem_type="qa_digest")
    print("\nPREFERENCE + QA MEMORY VERIFICATION OK")


if __name__ == "__main__":
    main()
