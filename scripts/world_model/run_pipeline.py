"""
世界模型 + ReAct 闭环主管线
============================
Step 1 构建混合数据（论文先验模拟 + DEAP 真实转移）
Step 2 训练 Causal Transformer 世界模型
Step 3 评估（单步 R²/MAE + 双基线对照 + Rollout 趋势一致性）
Step 4 LangGraph ReAct 闭环运行 + 闭环质量统计
Step 5 输出质量报告 JSON
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from wm_data import load_deap_transitions, simulate_emotion_response, simulate_sessions
from wm_evaluate import evaluate_single_step, rollout_consistency
from wm_model import CausalTransformerWorldModel, train
from wm_react import build_react_graph

OUT = Path(__file__).resolve().parent / "wm_report.json"
CKPT = Path(__file__).resolve().parent / "wm_model.pt"


def main():
    torch.manual_seed(0)
    report = {"pipeline": "world_model + react (langgraph)"}

    # ── Step 1 数据 ──
    print("== Step 1: 构建混合数据 ==")
    t0 = time.time()
    sim = simulate_sessions(n_users=120, sessions_per_user=4, steps=20)
    rng = np.random.RandomState(0)
    rng.shuffle(sim)
    n_train = int(len(sim) * 0.8)
    train_samples = sim[:n_train]
    test_sim = sim[n_train:n_train + 1500]  # 留出模拟测试集
    deap = load_deap_transitions(n_subjects=8)
    print(f"  模拟样本 {len(sim)}（训练 {n_train} / 测试 {len(test_sim)}），DEAP 真实转移 {len(deap)}")
    mixed = train_samples + deap  # 混合训练
    report["data"] = {"sim_total": len(sim), "deap_transitions": len(deap)}

    # ── Step 2 训练 ──
    print("== Step 2: 训练 Causal Transformer 世界模型 ==")
    model = CausalTransformerWorldModel(d_model=64, n_heads=4, n_layers=2)
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    train(model, mixed, epochs=10, batch=128, lr=5e-4)
    torch.save(model.state_dict(), CKPT)
    report["model"] = {"params": sum(p.numel() for p in model.parameters()), "epochs": 10}

    # ── Step 3 评估 ──
    print("== Step 3: 评估 ==")
    eval_sim = evaluate_single_step(model, test_sim)
    eval_deap = evaluate_single_step(model, deap)
    print(f"  [模拟测试集] world_model: {eval_sim['world_model']}")
    print(f"               persistence: {eval_sim['baseline_persistence']}")
    print(f"  [DEAP 真实集] world_model: {eval_deap['world_model']}")
    print(f"               persistence: {eval_deap['baseline_persistence']}")
    roll = rollout_consistency(model, simulate_emotion_response, n_trials=20, steps=5)
    print(f"  [Rollout] {roll}")

    def better(wm, base):
        return (wm["r2_valence"] > base["r2_valence"] and wm["r2_arousal"] >= base["r2_arousal"]) \
            or (wm["mae_valence"] + wm["mae_arousal"]) < (base["mae_valence"] + base["mae_arousal"])

    report["evaluation"] = {
        "sim_test": eval_sim, "deap_test": eval_deap, "rollout": roll,
        "beats_persistence_sim": better(eval_sim["world_model"], eval_sim["baseline_persistence"]),
        "beats_persistence_deap": better(eval_deap["world_model"], eval_deap["baseline_persistence"]),
    }

    # ── Step 4 ReAct 闭环 ──
    print("== Step 4: LangGraph ReAct 闭环 ==")
    agent = build_react_graph(model)
    scenarios = [
        ("anxious_user", {"hrv_lf_hf": 4.2, "eeg_alpha_power": 0.25}, "焦虑(高唤醒低效价)"),
        ("sleepy_user", {"hrv_lf_hf": 0.8, "eeg_alpha_power": 0.75}, "困倦(低唤醒)"),
        ("mild_low_user", {"hrv_lf_hf": 2.5, "eeg_alpha_power": 0.62}, "轻度低效价(亚健康)"),
    ]
    closed_results = []
    for uid, signals, label in scenarios:
        result = agent.invoke({"user_id": uid, "physio_signals": signals, "user_history": []})
        print(f"\n  ── {label} ({uid}) ──")
        for log in result["thought_log"]:
            print(f"    {log}")
        ev = result["evaluation"]
        closed_results.append({"user": uid, "scenario": label,
                               "plan": {k: result["selected_plan"][k] for k in ("cct", "lux", "duration")},
                               "confidence": result["selected_plan"]["confidence"],
                               "predicted": [result["selected_plan"]["predicted_valence"], result["selected_plan"]["predicted_arousal"]],
                               "actual_after": result["after_emotion"], **ev})
        assert result["selected_plan"]["confidence"] > 0.3, "置信度过低"
        assert result["evaluation"]["improved"], f"{uid} 干预未改善"
    improve_rate = sum(r["improved"] for r in closed_results) / len(closed_results)
    print(f"\n  闭环改善率: {improve_rate:.0%}")
    report["react_closed_loop"] = {"scenarios": closed_results, "improvement_rate": improve_rate}

    # ── Step 5 报告 ──
    report["runtime_seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n== 完成: 报告已写入 {OUT.name}，模型已保存 {CKPT.name} ==")


if __name__ == "__main__":
    main()
