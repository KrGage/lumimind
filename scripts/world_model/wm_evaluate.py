"""世界模型评估：单步 R²/MAE、基线对照、Rollout 趋势一致性"""
import numpy as np
from sklearn.metrics import r2_score

import torch

from wm_model import CausalTransformerWorldModel


def batch_predict(model, samples, max_len=20):
    preds, tgts, prevs = [], [], []
    with torch.no_grad():
        model.eval()
        for hist, act, tgt in samples:
            h = torch.tensor(hist[-max_len:], dtype=torch.float32).unsqueeze(0)
            a = torch.tensor(act, dtype=torch.float32).unsqueeze(0)
            preds.append(model(h, a)[0][0].numpy())
            tgts.append(tgt)
            prevs.append(hist[-1][32 + 3:32 + 5])  # 上一真实情绪（persistence 基线）
    return np.array(preds), np.array(tgts), np.array(prevs)


def evaluate_single_step(model, samples, name="test"):
    """单步预测精度 + 两个基线对照（persistence / 全局均值）。"""
    preds, tgts, prevs = batch_predict(model, samples)
    mean = tgts.mean(0)

    def _m(x):
        return {
            "r2_valence": round(float(r2_score(tgts[:, 0], x[:, 0])), 3),
            "r2_arousal": round(float(r2_score(tgts[:, 1], x[:, 1])), 3),
            "mae_valence": round(float(np.abs(tgts[:, 0] - x[:, 0]).mean()), 3),
            "mae_arousal": round(float(np.abs(tgts[:, 1] - x[:, 1]).mean()), 3),
        }

    return {
        "n": len(samples),
        "world_model": _m(preds),
        "baseline_persistence": _m(prevs),
        "baseline_global_mean": _m(np.tile(mean, (len(samples), 1))),
    }


def rollout_consistency(model, simulator, n_trials=30, steps=5, seed=123):
    """
    AMRS 式 Rollout 评估：自回归模拟 5 步，比较世界模型预测与
    模拟器真实轨迹的情绪变化方向（趋势）一致性。
    """
    rng = np.random.RandomState(seed)
    agree, total = 0, 0
    for _ in range(n_trials):
        v, a = rng.uniform(0.3, 0.7, 2)
        history = []
        for _s in range(steps):
            # 模型选最优动作（贪心 utility）
            best, best_u = None, -1
            for cct in [2700, 3500, 4000, 5000, 6500]:
                for lux in [100, 300, 500, 800]:
                    act = np.array([(cct - 2000) / 4500, np.log(lux / 50) / np.log(20), 20 / 60], np.float32)
                    pv, pa, _ = model.predict(np.array(history or [np.zeros(40, np.float32)]), act)
                    u = pv * 0.5 + pa * 0.5
                    if u > best_u:
                        best_u, best = u, (cct, lux, 20.0)
            # 真实环境（模拟器）执行
            v_next, a_next = simulator(best[0], best[1], best[2], v, a, rng)
            from wm_data import make_token
            physio = np.zeros(32, np.float32) + np.array([v - 0.5, a - 0.5] * 16) * 0.5
            history.append(make_token(physio, best, (v, a)))
            # 趋势一致性：ΔV、ΔA 方向是否同号
            dv_m, da_m, _ = model.predict(np.array(history), np.array(
                [(best[0] - 2000) / 4500, np.log(best[1] / 50) / np.log(20), 1 / 3], np.float32))
            # 与真实一步转移比较方向
            agree += int(np.sign(v_next - v) == np.sign(dv_m - v)) + int(np.sign(a_next - a) == np.sign(da_m - a))
            total += 2
            v, a = v_next, a_next
    return {"rollout_trend_consistency": round(agree / max(total, 1), 3), "n_transitions": total}
