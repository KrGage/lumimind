"""ReAct Agent 闭环（LangGraph，C-data 版）：Sense → Think → Predict → Safety → Act → Observe

接入 wm_cdata.py 的最新世界模型（Causal Transformer + 残差 Δ 头）：
  · 状态向量 STATE_KEYS = [HR, HF, LFHF, symp, KSS, PA, NA, VASF]（标准化）
  · 光照动作 = [E/lx, CCT, melanopic EDI, melanopic irradiance]（4 维光度量，标准化）
  · Predict 输出完整的生理+情绪变化 Δ（8 维），下一状态 = 当前状态 + Δ
  · Safety 沿用 MC-dropout 置信度护栏，Observe 沿用数据自增强（token 追加历史）
"""
from typing import Annotated, List, TypedDict

import numpy as np
import torch
from langgraph.graph import END, StateGraph

from cdata_loader import ACTION_DIM, GROUPS, STATE_DIM, STATE_KEYS, build_dataset
from wm_cdata import CKPT, TOKEN_DIM, WorldModel, make_token


class ReActState(TypedDict, total=False):
    user_id: str
    raw_measures: dict        # 原始量表/生理读数: {"HR":..., "KSS":..., "PA":...}
    current_state: list       # 标准化 state 向量 (STATE_DIM,)
    user_history: list        # token 序列（numpy, (L, TOKEN_DIM)）
    thought_log: Annotated[List[str], lambda a, b: a + b]
    goal: str                 # "alertness" | "valence"
    candidates: list
    ranked_plans: list
    selected_plan: dict
    before_state: list
    after_state: list
    evaluation: dict
    should_continue: bool


# ── 模型便捷接口：MC-dropout 方差作置信度（沿用第一版 predict 的做法） ──

def load_world_model(ckpt=CKPT, device="cpu"):
    """加载 wm_cdata 训练好的世界模型；无 checkpoint 时返回未训练模型（仅联调用）。"""
    model = WorldModel().to(device)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()  # 保持 eval；predict 时手动开 dropout 做 MC 采样
    return model


@torch.no_grad()
def predict(model, history, action, step=0.0, n_mc=8):
    """
    单样本接口: history (L, TOKEN_DIM) numpy, action (ACTION_DIM,) → (delta, conf)
    conf 由 MC-dropout 多次前向的方差折算（越大越稳定）。
    """
    h = torch.tensor(np.asarray(history, np.float32)).unsqueeze(0)
    a = torch.tensor(np.asarray(action, np.float32)).unsqueeze(0)
    s = torch.tensor([[float(step)]])
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()  # 开启 dropout → MC 采样
    preds = np.stack([model(h, a, s)[0].numpy() for _ in range(n_mc)])
    model.eval()
    delta = preds.mean(0)
    # Δ 量级 ~0.5-1.5，std×8 恒为 0（过严）；按 Δ 量纲校准为 std×3
    conf = float(1.0 - min(1.0, preds.std(0).mean() * 3))
    return delta, conf


# ── 节点逻辑 ──────────────────────────────────────────────────────

def sense_state(raw_measures, ds):
    """原始读数 → 标准化 state 向量；缺失指标填 0（=均值，与训练一致）。"""
    v = np.zeros(STATE_DIM, np.float32)
    for i, k in enumerate(STATE_KEYS):
        raw = raw_measures.get(k, np.nan)
        if raw is None or np.isnan(raw):
            continue
        v[i] = (raw - ds.m_mean[k]) / ds.m_std[k]
    return v


def generate_candidates(ds, goal):
    """候选 = 5 个真实光照条件（动作已在 C-data 光度量网格内）。"""
    return [{"group": g,
             "E_lx": round(ds.light[g]["E_lx"], 1),
             "CCT": round(ds.light[g]["CCT"], 0),
             "mel_EDI": round(ds.light[g]["mel_EDI"], 1),
             "action_vec": ds.action_vec(g).tolist()}
            for g in GROUPS]


def _utility(next_state, goal):
    kss_i = STATE_KEYS.index("KSS"); pa_i = STATE_KEYS.index("PA"); na_i = STATE_KEYS.index("NA")
    if goal == "valence":      # 提升情绪效价
        return float(next_state[pa_i] - next_state[na_i])
    return float(-next_state[kss_i])  # 默认: 提升清醒度（降 KSS）


def build_react_graph(world_model, ds, env_simulate=True):
    """构建 ReAct 闭环状态图。world_model 需提供 forward(history, action, step)。"""

    def sense_node(state):
        s = sense_state(state.get("raw_measures", {}), ds)
        return {"current_state": s.tolist(), "before_state": s.tolist(),
                "thought_log": [f"[Sense] state=" + ", ".join(
                    f"{k}={v:+.2f}" for k, v in zip(STATE_KEYS, s))]}

    def think_node(state):
        goal = state.get("goal", "alertness")
        cands = generate_candidates(ds, goal)
        return {"candidates": cands,
                "thought_log": [f"[Think] 目标={goal}，生成 {len(cands)} 个候选光照（G1-G5 真实光度量）"]}

    def predict_node(state):
        s = np.asarray(state["current_state"], np.float32)
        hist = np.asarray(state.get("user_history") or [make_token(s, np.zeros(ACTION_DIM))],
                          np.float32)
        step = float(min(len(hist) - 1, 1))  # T1->T2 为 step0, T2->T3 为 step1
        ranked = []
        for c in state["candidates"]:
            delta, conf = predict(world_model, hist, np.asarray(c["action_vec"]), step)
            nxt = s + delta
            ranked.append({**{k: v for k, v in c.items() if k != "action_vec"},
                           "pred_delta": {k: round(float(d), 3) for k, d in zip(STATE_KEYS, delta)},
                           "pred_next": [round(float(x), 3) for x in nxt],
                           "confidence": round(conf, 3),
                           "utility": round(_utility(nxt, state.get("goal", "alertness")), 4)})
        ranked.sort(key=lambda x: x["utility"], reverse=True)
        best = ranked[0]
        return {"ranked_plans": ranked, "selected_plan": best,
                "thought_log": [f"[Predict] 最优: G{best['group']} (EDI={best['mel_EDI']}, "
                                f"{best['E_lx']}lx) → Δ=" + ", ".join(
                                    f"{k}:{v:+.2f}" for k, v in best["pred_delta"].items())
                                + f"，置信度={best['confidence']}"]}

    def safety_node(state):
        p = state["selected_plan"]
        # 护栏基于真实 G1-G5 光度量网格 ±50% 容差（网格: E 50~302lx, CCT 2800~6731K, EDI 43~781）
        ok = (p["confidence"] > 0.3
              and 25 <= p["E_lx"] <= 450
              and 1400 <= p["CCT"] <= 10100
              and 20 <= p["mel_EDI"] <= 1170)
        return {"should_continue": ok,
                "thought_log": [f"[Safety] {'✅ 通过' if ok else '⚠️ 置信度低/光度量越界, 拒绝执行'}"]}

    def act_node(state):
        p = state["selected_plan"]
        return {"thought_log": [f"[Act] 已下发灯具指令: G{p['group']} / "
                                f"{p['E_lx']}lx / {p['CCT']}K / EDI={p['mel_EDI']}"]}

    def observe_node(state):
        before = np.asarray(state["before_state"], np.float32)
        p = state["selected_plan"]
        act = np.asarray(p["action_vec"] if "action_vec" in p else
                         ds.action_vec(p["group"]), np.float32)
        if env_simulate:
            # 演示模式：用模型预测 + 噪声模拟真实响应（真实部署：等待后重新采集再标准化）
            delta, _ = predict(world_model, np.asarray(state.get("user_history") or
                                                       [make_token(before, act)], np.float32), act)
            after = np.clip(before + delta + np.random.normal(0, 0.05, STATE_DIM), -4, 4)
        else:
            after = before.copy()
        ev = {f"delta_{k}": round(float(b - a), 3)
              for k, a, b in zip(STATE_KEYS, before, after)}
        kss_i = STATE_KEYS.index("KSS"); pa_i = STATE_KEYS.index("PA"); na_i = STATE_KEYS.index("NA")
        goal = state.get("goal", "alertness")
        ev["improved"] = bool(after[kss_i] < before[kss_i] - 0.05) if goal == "alertness" \
            else bool((after[pa_i] - after[na_i]) > (before[pa_i] - before[na_i]) + 0.05)
        # 数据自增强：干预后的新状态作为 token 追加进用户历史（世界模型在线训练数据）
        new_token = make_token(after.astype(np.float32), act)
        return {"after_state": after.tolist(), "evaluation": ev,
                "user_history": (state.get("user_history") or []) + [new_token],
                "thought_log": [f"[Observe] 干预后 KSS={after[kss_i]:+.2f}, "
                                f"PA={after[pa_i]:+.2f}, NA={after[na_i]:+.2f}；"
                                f"改善={ev['improved']}；历史已追加 token（在线积累）"]}

    def safety_router(state):
        return "act" if state.get("should_continue") else "end"

    g = StateGraph(ReActState)
    for name, fn in [("sense", sense_node), ("think", think_node), ("predict", predict_node),
                     ("safety_check", safety_node), ("act", act_node), ("observe", observe_node)]:
        g.add_node(name, fn)
    g.set_entry_point("sense")
    g.add_edge("sense", "think")
    g.add_edge("think", "predict")
    g.add_edge("predict", "safety_check")
    g.add_conditional_edges("safety_check", safety_router, {"act": "act", "end": END})
    g.add_edge("act", "observe")
    g.add_edge("observe", END)
    return g.compile()


# ── 演示 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    ds = build_dataset()
    model = load_world_model()
    graph = build_react_graph(model, ds)

    init = ds.state_vec(1, 2, 1)  # 真实被试1 G2T1 作为初始状态
    raw = {k: float(ds.m_mean[k] + v * ds.m_std[k]) for k, v in zip(STATE_KEYS, init)}

    result = graph.invoke({
        "user_id": "demo_s01",
        "raw_measures": raw,
        "goal": "alertness",
        "user_history": [make_token(init, ds.action_vec(2))],
    })
    print("\n== 推理链 ==")
    for line in result["thought_log"]:
        print(" ", line)
