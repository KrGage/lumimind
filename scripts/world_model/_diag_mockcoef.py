"""
诊断：为什么清醒度反事实方向学反了？
=====================================
核心怀疑：mock 数据是从 fit_linear_dynamics 的系数生成的。
若光照维（尤其 mel_EDI）对 ΔKSS 的拟合系数因共线/顺序混淆而符号错误，
mock 会以 6x 数据量把这个"错误但确定"的方向洗进模型。

本脚本检查：
  1. 真实组均值：EDI vs ΔKSS 的方向（T1->T3）
  2. 线性拟合给各光照维分配的系数（标准化尺度），及 EDI 系数的符号/显著性
  3. 光照维之间的共线性（CCT/EDI/mel_irr/E_lx 相关）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdata_loader import (ACTION_DIM, GROUPS, STATE_DIM, STATE_KEYS,
                          build_dataset, build_transitions)
from wm_cdata import fit_linear_dynamics, transitions_to_samples

ACT_NAMES = ["E_lx", "CCT", "mel_EDI", "mel_irr"]


def main():
    ds = build_dataset()
    tr = build_transitions(ds)

    print("=" * 70)
    print("① 真实组均值：每种光照 T1->T3 的原始 ΔKSS / ΔPA / ΔNA")
    print("=" * 70)
    ki, pai, nai = STATE_KEYS.index("KSS"), STATE_KEYS.index("PA"), STATE_KEYS.index("NA")
    rows = []
    for g in GROUPS:
        dk, dp, dn = [], [], []
        for sid in ds.subjects:
            r1 = ds.state_raw(sid, g, 1)
            r3 = ds.state_raw(sid, g, 3)
            d = r3 - r1
            if not np.isnan(d[ki]):
                dk.append(d[ki])
            if not np.isnan(d[pai]):
                dp.append(d[pai])
            if not np.isnan(d[nai]):
                dn.append(d[nai])
        edi = ds.light[g]["mel_EDI"]
        elx = ds.light[g]["E_lx"]
        rows.append((g, edi, elx,
                     np.mean(dk) if dk else np.nan,
                     np.mean(dp) if dp else np.nan,
                     np.mean(dn) if dn else np.nan))
    print(f"  {'组':<4}{'EDI':>9}{'E_lx':>9}{'ΔKSS':>9}{'ΔPA':>9}{'ΔNA':>9}")
    for g, edi, elx, dk, dp, dn in rows:
        print(f"  G{g:<3}{edi:>9.1f}{elx:>9.1f}{dk:>9.2f}{dp:>9.2f}{dn:>9.2f}")

    edi_arr = np.array([r[1] for r in rows])
    dkss_arr = np.array([r[3] for r in rows])
    print(f"\n  组层面 corr(EDI, ΔKSS) = {np.corrcoef(edi_arr, dkss_arr)[0,1]:+.3f} "
          f"(昼夜节律期望<0：高黑视素→更清醒→ΔKSS更小)")
    # 排除暗环境 G4（照度极低本身影响，属混淆）后的方向
    mask = edi_arr > 0
    print(f"  注：G4 是暗环境(E_lx={rows[3][2]:.0f}lx)，其低 ΔKSS 可能由'暗→放松'而非黑视素驱动")

    print("\n" + "=" * 70)
    print("② 线性拟合的光照系数（标准化尺度，Δ ~ [state(8), action(4), step, bias]）")
    print("=" * 70)
    coef, rstd = fit_linear_dynamics(tr)
    # coef 形状 (STATE_DIM, 8+4+1+1)；action 占索引 8..11
    a0 = STATE_DIM
    print(f"  {'维度':<6}" + "".join(f"{n:>10}" for n in ACT_NAMES) + f"{'step':>10}{'resid_std':>11}")
    for d, k in enumerate(STATE_KEYS):
        act_coef = coef[d, a0:a0 + ACTION_DIM]
        step_c = coef[d, a0 + ACTION_DIM]
        print(f"  {k:<6}" + "".join(f"{c:>+10.3f}" for c in act_coef)
              + f"{step_c:>+10.3f}{rstd[d]:>11.3f}")

    edi_c_kss = coef[ki, a0 + 2]
    print(f"\n  >>> ΔKSS 对 mel_EDI 的拟合系数 = {edi_c_kss:+.4f}")
    print(f"      期望符号：<0（高EDI→更清醒）。实际：{'<0 ✅' if edi_c_kss < 0 else '>0 ⚠️ 符号错误'}")
    print(f"      mock 会把该系数当作确定性规律放大 6x → 若符号错，模型必学反")

    print("\n" + "=" * 70)
    print("③ 光照动作维之间的共线性（只有5个不同光照向量）")
    print("=" * 70)
    A = np.array([[ds.light[g]["E_lx"], ds.light[g]["CCT"], ds.light[g]["mel_EDI"],
                   ds.light[g]["mel_irr"]] for g in GROUPS], np.float64)
    A = (A - A.mean(0)) / np.where(A.std(0) > 1e-9, A.std(0), 1)
    C = np.corrcoef(A.T)
    print(f"  {'':<10}" + "".join(f"{n:>10}" for n in ACT_NAMES))
    for i, n in enumerate(ACT_NAMES):
        print(f"  {n:<10}" + "".join(f"{C[i,j]:>+10.2f}" for j in range(ACTION_DIM)))
    print(f"\n  EDI 与 mel_irr 近乎完全共线(|r|>0.99) → 回归系数在两者间任意劈分，单个不可解释")
    print(f"  仅 5 个光照向量、且实验按 G1→G5 固定顺序 → 光照效应与时间/疲劳顺序效应纠缠，本质不可辨识")


if __name__ == "__main__":
    main()
