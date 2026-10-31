"""
演示：用 DEAP 真实数据填充长期记忆，并通过 LangGraph 工作流完成问答
==================================================================

流程（全部走真实管线，无人工编撰答案）:
  1. 读取 DEAP s01 被试数据，选取 valence 最低（情绪低落）与中性的 trial
  2. 从真实 EEG/外周信号计算特征（alpha/beta 比值、心率估计、GSR）
  3. 以 mock 时间（10月3日 17:00 等）写入用户长期记忆库
  4. 用 LangGraph AgentWorkflow（Supervisor → Analyst(记忆检索) → Generate）
     持 SqliteSaver checkpointer 回答用户问题
"""

import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai_agent"))

DEAP_FILE = ROOT / "DEAP" / "data_preprocessed_python" / "s01.dat"
UID = "local_user"          # 与 crud._memory_user_id() 缺省一致
SESSION = "deap_demo_conv"  # LangGraph thread_id
FS = 128                    # DEAP 采样率


# ── 第 1 步：从 DEAP 真实数据计算特征 ────────────────────────────

def load_deap():
    with open(DEAP_FILE, "rb") as f:
        return pickle.load(f, encoding="latin1")


def band_power(signal: np.ndarray, lo: float, hi: float) -> float:
    """单通道平均功率谱密度（Welch 简化版：分段 FFT）。"""
    seg = 4 * FS
    n_seg = len(signal) // seg
    powers = []
    win = np.hanning(seg)
    for i in range(n_seg):
        chunk = signal[i * seg:(i + 1) * seg] * win
        spec = np.abs(np.fft.rfft(chunk)) ** 2
        freqs = np.fft.rfftfreq(seg, 1 / FS)
        powers.append(spec[(freqs >= lo) & (freqs < hi)].mean())
    return float(np.mean(powers))


def extract_features(trial_data: np.ndarray) -> dict:
    """
    trial_data: (40, 8064) —— 前 32 通道 EEG，后 8 通道外周生理。
    外周通道序（DEAP 官方）: 32 hEOG, 33 vEOG, 34 zEMG, 35 tEMG,
                            36 GSR, 37 Respiration, 38 Pletthysmo(BVP), 39 Temp
    """
    eeg = trial_data[:32]

    # EEG：全通道 alpha(8-13Hz)/beta(13-30Hz) 平均功率比（情绪低落时 alpha 相对增强）
    ab_ratios = []
    for ch in eeg:
        alpha = band_power(ch, 8, 13)
        beta = band_power(ch, 13, 30)
        if beta > 0:
            ab_ratios.append(alpha / beta)
    alpha_beta_ratio = float(np.mean(ab_ratios))

    # 额叶 alpha 不对称（F3/F4，抑郁/负性情绪相关指标；DEAP 通道 3≈F3? 用前额通道近似）
    frontal_asym = float(np.mean([band_power(eeg[c], 8, 13) for c in range(4)]))

    # 外周：GSR 均值（压力指标）、BVP 通道估计心率
    gsr = float(np.mean(trial_data[36]))
    bvp = trial_data[38]
    bvp_f = bvp - np.mean(bvp)
    # 0.5~4Hz 带通近似（脉搏主频约 1~2Hz）
    spec = np.abs(np.fft.rfft(bvp_f * np.hanning(len(bvp_f))))
    freqs = np.fft.rfftfreq(len(bvp_f), 1 / FS)
    band = spec[(freqs >= 0.7) & (freqs <= 3.5)]
    peak_hz = freqs[(freqs >= 0.7) & (freqs <= 3.5)][np.argmax(band)]
    heart_rate = float(peak_hz * 60)

    return {
        "alpha_beta_ratio": round(alpha_beta_ratio, 3),
        "frontal_alpha": round(frontal_asym, 2),
        "gsr_mean_uS": round(gsr, 4),
        "heart_rate_bpm": round(heart_rate, 1),
    }


def valence_to_label(v: float) -> str:
    """DEAP 1~9 效价：低效价=消极。常规阈值 4.5/6.5。"""
    if v < 4.5:
        return "negative"
    if v > 6.5:
        return "positive"
    return "neutral"


# ── 第 2 步：写入长期记忆（真实特征 + mock 时间） ─────────────────

def seed_memories():
    from agent_framework import long_term_memory as ltm

    ltm.clear_user_memories(UID, mem_type="physio_emotion")

    data = load_deap()
    labels, signals = data["labels"], data["data"]
    valences = labels[:, 0]

    # 情绪低落 trial（valence 最低）→ 10月3日 17:00
    low_idx = int(np.argmin(valences))
    # 对照：中性 / 积极 trial → 10月3日 其他时段
    order = np.argsort(np.abs(valences - 5.5))  # 最接近 5.5 的中性段
    neutral_idx = int(order[0]) if order[0] != low_idx else int(order[1])
    high_idx = int(np.argmax(valences))

    schedule = [
        (low_idx, datetime(2026, 10, 3, 17, 0, tzinfo=timezone.utc)),
        (neutral_idx, datetime(2026, 10, 3, 10, 0, tzinfo=timezone.utc)),
        (high_idx, datetime(2026, 10, 3, 20, 30, tzinfo=timezone.utc)),
    ]

    print(f"DEAP s01 选中 trial: 低落=#{low_idx}(valence={valences[low_idx]:.2f}) "
          f"中性=#{neutral_idx}(valence={valences[neutral_idx]:.2f}) "
          f"积极=#{high_idx}(valence={valences[high_idx]:.2f})")

    for trial_idx, ts in schedule:
        v, a = valences[trial_idx], labels[trial_idx, 1]
        feats = extract_features(signals[trial_idx])
        label = valence_to_label(v)
        ltm.record_physio_emotion(
            user_id=UID,
            emotion_label=label,
            emotion_score=round((v - 1) / 8, 2),   # 1~9 → 0~1
            confidence=0.85,
            physio={
                "arousal": round(float(a), 2),
                "eeg_alpha_beta_ratio": feats["alpha_beta_ratio"],
                "gsr_mean_uS": feats["gsr_mean_uS"],
                "heart_rate_bpm": feats["heart_rate_bpm"],
                "deap_trial": trial_idx,
            },
            session_id="deap_demo",
            timestamp=ts,
        )
        print(f"  写入 {ts:%m-%d %H:%M} {label}: {feats}")


# ── 第 3 步：LangGraph 工作流问答 ────────────────────────────────

def build_langgraph_workflow():
    """Supervisor → Analyst(检索用户生理记忆) → Generate(LLM) 的 LangGraph 图。"""
    from agent_framework.llm_wrapper import CoreAIWrapper
    from agent_framework.long_term_memory import build_physio_context
    from agent_framework.workflow import AgentWorkflow, BaseNode, AgentState

    class PhysioMemoryAnalystNode(BaseNode):
        """分析节点：检索用户长期生理记忆，注入 analysis_results。"""

        def run(self, state: AgentState) -> dict:
            ctx = build_physio_context(UID, query=self._last_query(state))
            return {"analysis_results": ctx or "无用户生理记录"}

        @staticmethod
        def _last_query(state) -> str:
            msgs = state.get("messages", [])
            return msgs[-1].content if msgs else ""

    return AgentWorkflow(
        llm=CoreAIWrapper(),
        system_prompt=(
            "你是 LumiMind 医学健康智能助手。回答用户关于自身健康状况的问题时，"
            "必须优先依据 ANALYSIS RESULTS 中的用户生理记录（来自 DEAP 数据集的真实信号特征），"
            "结合时间信息解读情绪/生理趋势，给出光环境建议，"
            "并提醒这些数据仅供参考、持续异常应咨询医生。请用中文回答。"
        ),
        enable_search=False,
        enable_guardrail=False,
        enable_analysis=True,
        analysis_node=PhysioMemoryAnalystNode(),
        supervisor_kwargs={
            "analyze_keywords": [
                "身体状况", "健康状况", "健康状态", "情绪", "心率", "生理",
                "怎么样", "最近", "低落", "压力", "睡眠",
            ],
        },
        use_checkpointer=True,   # SqliteSaver → data/checkpoints.db
    )


def ask(workflow, question: str) -> str:
    """通过 LangGraph 图提问（thread_id 复用会话记忆）。"""
    from agent_framework.checkpointer import thread_config
    from langchain_core.messages import HumanMessage

    result = workflow.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=thread_config(SESSION),
    )
    return result["messages"][-1].content


def main():
    print("== 1. 用 DEAP 真实数据写入长期记忆 ==")
    seed_memories()

    print("\n== 2. 通过 LangGraph 工作流问答 ==")
    wf = build_langgraph_workflow()
    question = "10月3日下午五点左右我是不是有一段时间情绪低落？结合我的生理数据说说最近的身体状况怎么样？"
    print(f"用户: {question}\n")
    answer = ask(wf, question)
    print("LumiMind:", answer)

    # 验证：回答应提到情绪低落时段及真实生理特征
    checks = ["低落", "消极"]
    physio_hits = ["心率", "alpha", "gsr", "GSR", "压"]
    found_low = any(c in answer for c in checks)
    found_physio = any(c in answer for c in physio_hits)
    print(f"\n验证: 提及情绪低落={found_low}, 引用生理指标={found_physio}")
    assert found_low, "回答未提及情绪低落"
    assert found_physio, "回答未引用生理指标"
    print("DEAP + LANGGRAPH DEMO OK")


if __name__ == "__main__":
    main()
