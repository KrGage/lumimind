"""诊断：真实数据里 melanopic EDI → KSS 的信号到底有多强？Transformer 是否学反？"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdata_loader import (STATE_KEYS, GROUPS, build_dataset, build_transitions)
from wm_cdata import fit_linear_dynamics

ds = build_dataset()
tr = build_transitions(ds)
ki = STATE_KEYS.index("KSS")
pai = STATE_KEYS.index("PA")

# 真实转移层面：EDI(标准化 action dim2) 与 KSS Δ 的相关
edi_n = np.array([s["action"][2] for s in tr])       # 标准化 EDI
dkss = np.array([(s["s_to"] - s["s_from"])[ki] for s in tr])
dpa = np.array([(s["s_to"] - s["s_from"])[pai] for s in tr])
print(f"[转移层面 n={len(tr)}] corr(EDI, ΔKSS)={np.corrcoef(edi_n, dkss)[0,1]:.3f}  (期望<0)")
print(f"                    corr(EDI, ΔPA )={np.corrcoef(edi_n, dpa)[0,1]:.3f}  (期望>0)")

# 组均值层面（每被试 T1->T3）
print("\n[组均值层面] 每光照条件的真实 ΔKSS / ΔPA（标准化）与 EDI:")
print(f"  {'组':<4}{'EDI':>8}{'ΔKSS':>9}{'ΔPA':>9}{'n':>5}")
gedi, gkss = [], []
for g in GROUPS:
    ds_ = []
    for s in tr:
        if s["g"] == g and s["t_from"] == 1:  # T1->T2 单步
            ds_.append((s["s_to"] - s["s_from"]))
    arr = np.array(ds_)
    mk = arr[:, ki].mean(); mp = arr[:, pai].mean()
    e = ds.light[g]["mel_EDI"]
    gedi.append(e); gkss.append(mk)
    print(f"  G{g:<3}{e:>8.1f}{mk:>9.3f}{mp:>9.3f}{len(arr):>5}")
print(f"  → corr(组EDI, 组ΔKSS)={np.corrcoef(gedi, gkss)[0,1]:.3f}  (期望<0)")

# 岭回归系数：KSS 维对 EDI 的系数
coef, rstd = fit_linear_dynamics(tr)
# 特征顺序: [state(8), action(4: E_lx,CCT,EDI,irr), step, bias]
print(f"\n[岭回归 ΔKSS 系数] action 部分: E_lx={coef[ki,8]:.3f} CCT={coef[ki,9]:.3f} "
      f"EDI={coef[ki,10]:.3f} irr={coef[ki,11]:.3f}  (EDI 期望<0)")
print(f"[岭回归 ΔPA  系数] action 部分: E_lx={coef[pai,8]:.3f} CCT={coef[pai,9]:.3f} "
      f"EDI={coef[pai,10]:.3f} irr={coef[pai,11]:.3f}")
print(f"[岭回归 ΔKSS 系数] state 部分(KSS自身)={coef[ki,ki]:.3f}  step={coef[ki,12]:.3f}")

# G2 与 G5 是否为重复光照条件（动作相同但结果不同 → 不可辨识噪声）
a2 = ds.action_vec(2); a5 = ds.action_vec(5)
print(f"\n[G2 vs G5 动作向量] 是否相同: {np.allclose(a2, a5)}  a2={np.round(a2,3)} a5={np.round(a5,3)}")
d2 = np.array([(s['s_to']-s['s_from'])[ki] for s in tr if s['g']==2 and s['t_from']==1])
d5 = np.array([(s['s_to']-s['s_from'])[ki] for s in tr if s['g']==5 and s['t_from']==1])
print(f"  但真实 ΔKSS(T1->T2): G2 均值={d2.mean():.3f}, G5 均值={d5.mean():.3f} → 同光不同果(顺序/疲劳混淆)")
