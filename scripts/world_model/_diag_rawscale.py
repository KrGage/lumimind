"""
真实 C-data 原始量纲画像
========================
为"仿 C-data 结构生成完整模拟数据集"提供基准：
  · 每个指标的原始量纲 mean/std/min/max（用于反标准化）
  · 逐 group×time 的原始均值（真实轨迹形态）
  · 被试基线（T1）的跨被试变异（用于生成有个体差异的合成被试）
  · .sav / xlsx 的列结构与取值范围
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdata_loader import (CDATA, GROUPS, MEASURE_SAV, STATE_KEYS, TIMES,
                          _read_sav_measure, build_dataset, load_light_actions)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def main():
    ds = build_dataset()
    sids = ds.subjects

    print("=" * 78)
    print("① 每个 state 指标的原始量纲分布（反标准化基准）")
    print("=" * 78)
    print(f"  {'指标':<6}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}   标准化用(mean/std)")
    for k in STATE_KEYS:
        vals = []
        for sid in sids:
            for g in GROUPS:
                for t in TIMES:
                    col = f"G{g}T{t}"
                    v = ds.measures.get(k, {}).get(sid, {}).get(col, np.nan)
                    if v is not None and not np.isnan(v):
                        vals.append(v)
        a = np.array(vals, float)
        print(f"  {k:<6}{a.mean():>10.3f}{a.std():>10.3f}{a.min():>10.3f}{a.max():>10.3f}"
              f"   ({ds.m_mean[k]:.3f}/{ds.m_std[k]:.3f})  n={len(a)}")

    print("\n" + "=" * 78)
    print("② 逐 group×time 的原始均值（真实轨迹形态，看光照/时间趋势）")
    print("=" * 78)
    for k in STATE_KEYS:
        print(f"\n  --- {k} ---")
        print(f"    {'':<6}" + "".join(f"{'T'+str(t):>9}" for t in TIMES))
        for g in GROUPS:
            row = []
            for t in TIMES:
                vv = []
                for sid in sids:
                    v = ds.measures.get(k, {}).get(sid, {}).get(f"G{g}T{t}", np.nan)
                    if v is not None and not np.isnan(v):
                        vv.append(v)
                row.append(np.mean(vv) if vv else np.nan)
            print(f"    G{g:<5}" + "".join(f"{x:>9.2f}" for x in row))

    print("\n" + "=" * 78)
    print("③ 被试基线变异：每被试在 G*T1 的均值/跨被试std（合成被试个体差异基准）")
    print("=" * 78)
    print(f"  {'指标':<6}{'跨被试基线std':>16}{'被试内(组间)std均值':>22}")
    for k in STATE_KEYS:
        subj_means = []
        within_stds = []
        for sid in sids:
            base = []
            for g in GROUPS:
                v = ds.measures.get(k, {}).get(sid, {}).get(f"G{g}T1", np.nan)
                if v is not None and not np.isnan(v):
                    base.append(v)
            if base:
                subj_means.append(np.mean(base))
                within_stds.append(np.std(base))
        bs = np.std(subj_means) if len(subj_means) > 1 else 0.0
        ws = np.mean(within_stds) if within_stds else 0.0
        print(f"  {k:<6}{bs:>16.3f}{ws:>22.3f}")

    print("\n" + "=" * 78)
    print("④ 缺失情况：每指标有多少 (被试×组×时间) 单元有真值 / 共 %d" % (len(sids)*5*3))
    print("=" * 78)
    for k in STATE_KEYS:
        cnt = sum(1 for sid in sids for g in GROUPS for t in TIMES
                  if not np.isnan(ds.measures.get(k, {}).get(sid, {}).get(f"G{g}T{t}", np.nan)))
        present_groups = sorted({g for sid in sids for g in GROUPS
                                 if not np.isnan(ds.measures.get(k, {}).get(sid, {}).get(f"G{g}T1", np.nan))})
        print(f"  {k:<6} 有值单元={cnt:>4}   覆盖的组={present_groups}")

    print("\n" + "=" * 78)
    print("⑤ 光照表 光谱对应表.xlsx（前 6 行 × 前 12 列）")
    print("=" * 78)
    df = pd.read_excel(CDATA / "光谱对应表.xlsx", header=None)
    print(f"  shape={df.shape}")
    print(df.iloc[:6, :12].to_string())

    print("\n" + "=" * 78)
    print("⑥ 单个 .sav 的列结构（以 KSS.sav 为例）")
    print("=" * 78)
    import pyreadstat
    d2, meta = pyreadstat.read_sav(str(CDATA / "KSS" / "KSS.sav"))
    print(f"  columns={list(d2.columns)}")
    print(f"  labels={meta.column_names_to_labels}")
    print(d2.head(12).to_string())


if __name__ == "__main__":
    main()
