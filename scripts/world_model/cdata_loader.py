"""
C-data 世界模型数据加载器
=========================
把真实实验数据 (10 被试 × 5 光照条件 G1-G5 × 3 时间点 T1/T2/T3) 解析成
世界模型可用的 (state, action) 结构：

- state  : 低维、可解释的生理+情绪特征向量（标准化后）
- action : 光照条件的光度/黑视素量 [E/lx, CCT, melanopic EDI, melanopic irradiance]（标准化后）
- 转移   : 同一被试、同一光照条件内 T1->T2->T3

设计遵循前次结论：输出保持低维（预测 Δ），输入用可解释生理特征而非原始波形，
并用真实 melanopic EDI 作为光照动作的生理正确度量。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parent.parent.parent  # 项目主体
CDATA = ROOT / "ai_agent" / "Oupu_Lighting" / "data" / "C-data"

GCOLS = [f"G{g}T{t}" for g in range(1, 6) for t in range(1, 4)]  # 15 列
GROUPS = [1, 2, 3, 4, 5]
TIMES = [1, 2, 3]

# 指标 -> (sav 相对路径, 说明)；HRV 的 HF 只有 G1-G3
MEASURE_SAV = {
    "HR":      "HR/HR.sav",
    "HF":      "HRV/HF.sav",                 # 高频(副交感)
    "LFHF":    "HRV/交感迷走平衡.sav",         # LF/HF 交感迷走平衡
    "symp":    "HRV/交感神经.sav",             # 交感神经活性
    "balance": "Balance/Balance.sav",
    "abs":     "ABS/ABS.sav",
    "KSS":     "KSS/KSS.sav",                 # 嗜睡量表 1-9
    "PA":      "PANAS/PA.sav",                # 正性情绪 10-50
    "NA":      "PANAS/NA.sav",                # 负性情绪 10-50
    "VASF":    "VASF/VASF.sav",               # Visual Analogue Scale for Fatigue，疲劳量表
}

# 世界模型 state 向量使用的指标（顺序固定）
STATE_KEYS = ["HR", "HF", "LFHF", "symp", "KSS", "PA", "NA", "VASF"]
STATE_DIM = len(STATE_KEYS)
ACTION_DIM = 4  # E/lx, CCT, melanopic EDI, melanopic irradiance


def _read_sav_measure(rel: str, root: str | Path | None = None) -> dict[int, dict[str, float]]:
    """读取单个 sav，返回 {被试者id: {G1T1: val, ...}}，自动丢弃 NaN 填充行。
    root 为空时读真实 C-data；传入模拟数据目录则可把模拟集当 C-data 读。"""
    base = Path(root) if root is not None else CDATA
    df, _ = pyreadstat.read_sav(str(base / rel))
    present = [c for c in GCOLS if c in df.columns]  # HF 等只有 G1-G3
    df = df.dropna(subset=["被试者"])
    df = df.dropna(subset=present, how="all")  # 丢掉整行无数据的填充
    out: dict[int, dict[str, float]] = {}
    for _, row in df.iterrows():
        sid = int(round(float(row["被试者"])))
        if sid in out:  # 去重，保留首次出现的完整行
            continue
        vals = {c: (float(row[c]) if c in present and pd.notna(row[c]) else np.nan) for c in GCOLS}
        if all(np.isnan(v) for v in vals.values()):
            continue
        out[sid] = vals
    return out


def load_light_actions(root: str | Path | None = None) -> dict[int, dict[str, float]]:
    """从 光谱对应表.xlsx 提取 G1-G5 的光照动作 [E/lx, CCT, melanopic EDI, melanopic irr]。"""
    base = Path(root) if root is not None else CDATA
    df = pd.read_excel(base / "光谱对应表.xlsx", header=None)

    def num(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    actions = {}
    for g in GROUPS:
        col = (g - 1) * 2 + 1  # 数据列: 1,3,5,7,9
        e_lx = num(df.iloc[1, col])
        cct = num(df.iloc[2, col])
        edi = num(df.iloc[3, col])
        irr = num(df.iloc[4, col])
        actions[g] = {"E_lx": e_lx, "CCT": cct, "mel_EDI": edi, "mel_irr": irr}
    return actions


@dataclass
class CDataDataset:
    subjects: list[int]
    measures: dict[str, dict[int, dict[str, float]]]   # key -> {sid -> {Gcol -> val}}
    light: dict[int, dict[str, float]]                  # group -> action dict
    # 标准化统计（按指标）
    m_mean: dict[str, float]
    m_std: dict[str, float]
    a_mean: np.ndarray
    a_std: np.ndarray

    def state_vec(self, sid: int, g: int, t: int) -> np.ndarray:
        """返回标准化后的 state 向量 (STATE_DIM,)，缺失指标填 0（=均值）。"""
        col = f"G{g}T{t}"
        v = np.zeros(STATE_DIM, np.float32)
        for i, k in enumerate(STATE_KEYS):
            raw = self.measures.get(k, {}).get(sid, {}).get(col, np.nan)
            if raw is None or np.isnan(raw):
                v[i] = 0.0
            else:
                v[i] = (raw - self.m_mean[k]) / self.m_std[k]
        return v

    def state_raw(self, sid: int, g: int, t: int) -> np.ndarray:
        col = f"G{g}T{t}"
        v = np.full(STATE_DIM, np.nan, np.float64)
        for i, k in enumerate(STATE_KEYS):
            v[i] = self.measures.get(k, {}).get(sid, {}).get(col, np.nan)
        return v

    def action_vec(self, g: int) -> np.ndarray:
        a = self.light[g]
        raw = np.array([a["E_lx"], a["CCT"], a["mel_EDI"], a["mel_irr"]], np.float32)
        return ((raw - self.a_mean) / self.a_std).astype(np.float32)


def build_dataset(cct_fill: float = 2800.0, root: str | Path | None = None) -> CDataDataset:
    measures = {k: _read_sav_measure(v, root) for k, v in MEASURE_SAV.items()}
    light = load_light_actions(root)
    # CCT 缺失(暖光/未定)用 cct_fill 兜底，避免 NaN 污染标准化
    for g in GROUPS:
        if np.isnan(light[g]["CCT"]):
            light[g]["CCT"] = cct_fill

    # 所有出现在任一指标里的被试
    subjects = sorted({sid for m in measures.values() for sid in m.keys()})

    # 指标级标准化统计（跨所有被试/组/时间点）
    m_mean, m_std = {}, {}
    for k in STATE_KEYS:
        allv = [val for sid in measures.get(k, {}) for val in measures[k][sid].values()
                if val is not None and not np.isnan(val)]
        arr = np.array(allv, np.float64) if allv else np.array([0.0])
        m_mean[k] = float(np.nanmean(arr)) if len(arr) else 0.0
        sd = float(np.nanstd(arr)) if len(arr) else 1.0
        m_std[k] = sd if sd > 1e-6 else 1.0

    # 动作级标准化统计（跨 G1-G5）
    A = np.array([[light[g]["E_lx"], light[g]["CCT"], light[g]["mel_EDI"], light[g]["mel_irr"]]
                  for g in GROUPS], np.float64)
    a_mean = A.mean(0)
    a_std = np.where(A.std(0) > 1e-6, A.std(0), 1.0)

    return CDataDataset(subjects, measures, light, m_mean, m_std,
                        a_mean.astype(np.float32), a_std.astype(np.float32))


def build_transitions(ds: CDataDataset):
    """
    构造真实转移样本。
    返回 list of dict:
      {sid, g, t_from, t_to, s_from (STATE_DIM,), s_to, action (ACTION_DIM,), delta}
    仅保留 s_from / s_to 都非全缺失的转移。
    """
    samples = []
    for sid in ds.subjects:
        for g in GROUPS:
            act = ds.action_vec(g)
            for t in (1, 2):  # T1->T2, T2->T3
                sf = ds.state_vec(sid, g, t)
                st = ds.state_vec(sid, g, t + 1)
                rawf = ds.state_raw(sid, g, t)
                rawt = ds.state_raw(sid, g, t + 1)
                # 至少有一项状态指标在转移两端各有观测才纳入
                if np.all(np.isnan(rawf)) or np.all(np.isnan(rawt)):
                    continue
                samples.append({
                    "sid": sid, "g": g, "t_from": t, "t_to": t + 1,
                    "s_from": sf, "s_to": st, "action": act,
                    "raw_from": rawf, "raw_to": rawt,
                })
    return samples


if __name__ == "__main__":
    ds = build_dataset()
    print("被试:", ds.subjects)
    print("\n光照动作 (原始):")
    for g in GROUPS:
        a = ds.light[g]
        print(f"  G{g}: E/lx={a['E_lx']:.1f}  CCT={a['CCT']:.0f}  melEDI={a['mel_EDI']:.1f}  mel_irr={a['mel_irr']:.3f}")
    print(f"\n指标标准化均值: { {k: round(v,2) for k,v in ds.m_mean.items()} }")
    print(f"指标标准化std : { {k: round(v,2) for k,v in ds.m_std.items()} }")

    tr = build_transitions(ds)
    print(f"\n真实转移样本数: {len(tr)}  (期望 ~10被试×5组×2转移=100)")

    # 统计每种光照下 T1->T3 的平均变化（原始尺度），看光是否真的引起响应
    print("\n各光照条件 T1->T3 的平均原始变化 (Δ):")
    print(f"  {'组':<4}" + "".join(f"{k:>9}" for k in STATE_KEYS))
    for g in GROUPS:
        deltas = []
        for sid in ds.subjects:
            r1 = ds.state_raw(sid, g, 1)
            r3 = ds.state_raw(sid, g, 3)
            d = r3 - r1
            if not np.all(np.isnan(d)):
                deltas.append(d)
        if deltas:
            md = np.nanmean(np.array(deltas), axis=0)
            print(f"  G{g:<3}" + "".join(f"{x:>9.2f}" for x in md))
