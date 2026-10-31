"""世界模型数据构建：论文先验模拟 + DEAP 真实序列（弱标注动作）"""
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent  # 项目主体
DEAP_DIR = ROOT / "DEAP" / "data_preprocessed_python"

# 动作空间（候选光照网格）
CCT_GRID = np.array([2700, 3500, 4000, 5000, 6500], dtype=np.float32)
LUX_GRID = np.array([100, 300, 500, 800], dtype=np.float32)
DUR_DEFAULT = 20.0

D_PHYSIO = 32  # 生理嵌入维度（实验用紧凑表示）


# ── 论文先验模拟器（Smolders 2012 / Ru 2019 / Lan 2021 规律） ──────

def simulate_emotion_response(cct, lux, duration, cur_v, cur_a, rng=None):
    rng = rng or np.random
    cct_effect = (cct - 4000) / 4000                      # 高色温提神
    lux_effect = np.log(max(lux, 50) / 300) * 0.1         # 照度对数效应
    dur = 1 - np.exp(-duration / 20)                      # 时间边际递减
    d_a = (cct_effect * 0.3 + lux_effect * 0.4) * dur + rng.normal(0, 0.03)
    d_v = -abs(cct_effect) * 0.2 + 0.08 + rng.normal(0, 0.03)
    return float(np.clip(cur_v + d_v, 0, 1)), float(np.clip(cur_a + d_a, 0, 1))


def simulate_sessions(n_users=120, sessions_per_user=4, steps=20, seed=42):
    """生成模拟 (历史 token 序列, 动作, 下一情绪) 样本。"""
    rng = np.random.RandomState(seed)
    samples = []
    for u in range(n_users):
        # 个体基线与敏感度（世界模型需学会的个体差异）
        v0 = rng.uniform(0.3, 0.7)
        a0 = rng.uniform(0.3, 0.7)
        sens = rng.uniform(0.7, 1.3)
        user_emb = rng.randn(D_PHYSIO) * 0.3
        for s in range(sessions_per_user):
            v, a = v0 + rng.normal(0, 0.05), a0 + rng.normal(0, 0.05)
            history = []
            for _ in range(steps):
                action = (CCT_GRID[rng.randint(5)], LUX_GRID[rng.randint(4)], DUR_DEFAULT)
                physio = user_emb + np.array([v - 0.5, a - 0.5] * (D_PHYSIO // 2)) * 0.5
                history.append(make_token(physio, action, (v, a)))
                dv, da = simulate_emotion_response(*action[:2], action[2], v, a, rng)
                v, a = v + (dv - v) * sens * 0.1 + rng.normal(0, 0.02), a + (da - a) * sens * 0.1 + rng.normal(0, 0.02)
                v, a = float(np.clip(v, 0, 1)), float(np.clip(a, 0, 1))
                samples.append((np.array(history, np.float32), norm_action(action), (v, a)))
    return samples


# ── Token 构造 ────────────────────────────────────────────────────

def make_token(physio, action, va):
    """token = [physio(d) ⊕ light(3) ⊕ feedback(2) ⊕ mask(3)]"""
    cct, lux, dur = action
    token = np.zeros(D_PHYSIO + 8, np.float32)
    token[:D_PHYSIO] = physio
    token[D_PHYSIO:D_PHYSIO + 3] = norm_action(action)
    token[D_PHYSIO + 3:D_PHYSIO + 5] = va
    token[D_PHYSIO + 5:D_PHYSIO + 8] = [1, 1, 1]  # has_feedback 掩码
    return token


def norm_action(action):
    cct, lux, dur = action
    return np.array([(cct - 2000) / 4500, np.log(max(lux, 50) / 50) / np.log(20), dur / 60], np.float32)


def denorm_action(vec):
    return (float(vec[0] * 4500 + 2000), float(np.exp(vec[1] * np.log(20)) * 50), float(vec[2] * 60))


# ── DEAP 真实转移（弱标注动作） ───────────────────────────────────

def load_deap_transitions(n_subjects=8, use_features=True):
    """
    用相邻 trial 的 valence/arousal 构造真实状态转移；
    动作采用弱标注：从候选网格中选取"模拟器先验下最能解释该转移"的光照参数。
    """
    rng = np.random.RandomState(7)
    samples = []
    files = sorted(DEAP_DIR.glob("s*.dat"))[:n_subjects]
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        labels = d["labels"][:, :2] / 9.0  # valence/arousal → [0,1]
        if use_features:
            # 从 EEG alpha/beta 比与 BVP 通道提取紧凑生理特征（复用 demo 逻辑的简化版）
            feats = _subject_features(d["data"])
        else:
            feats = rng.randn(len(labels), D_PHYSIO) * 0.3

        history = []
        for i in range(len(labels) - 1):
            v, a = float(labels[i, 0]), float(labels[i, 1])
            nv, na = float(labels[i + 1, 0]), float(labels[i + 1, 1])
            # 弱标注动作：网格中使模拟器预测最接近真实下一状态的候选
            best, best_err = None, 1e9
            for cct in CCT_GRID:
                for lux in LUX_GRID:
                    pv, pa = simulate_emotion_response(cct, lux, DUR_DEFAULT, v, a, rng)
                    err = (pv - nv) ** 2 + (pa - na) ** 2
                    if err < best_err:
                        best_err, best = err, (cct, lux, DUR_DEFAULT)
            physio = feats[i] + np.array([v - 0.5, a - 0.5] * (D_PHYSIO // 2)) * 0.5
            history.append(make_token(physio, best, (v, a)))
            samples.append((np.array(history, np.float32), norm_action(best), (nv, na)))
        history = history[10:]  # 截断历史，控制序列长度
    return samples


def _subject_features(data):
    """每 trial 的 EEG alpha/beta 比（前 8 通道）+ GSR 均值 + BVP 主频 → D_PHYSIO 维。"""
    out = np.zeros((data.shape[0], D_PHYSIO), np.float32)
    for t in range(data.shape[0]):
        seg = data[t, :8, ::4]  # 降采样加速
        spec = np.abs(np.fft.rfft(seg * np.hanning(seg.shape[1]), axis=1)) ** 2
        freqs = np.fft.rfftfreq(seg.shape[1], 4 / 128)
        ab = spec[:, (freqs >= 8) & (freqs < 13)].mean(1) / (spec[:, (freqs >= 13) & (freqs < 30)].mean(1) + 1e-6)
        gsr = data[t, 36].mean()
        row = np.tile(np.concatenate([ab, [gsr]]), D_PHYSIO // 9 + 1)[:D_PHYSIO]
        out[t] = (row - row.mean()) / (row.std() + 1e-6)
    return out
