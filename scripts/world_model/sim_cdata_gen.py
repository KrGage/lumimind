"""
仿 C-data 结构生成"完整模拟数据集"
====================================
把世界模型训练用的 mock（原本只是内存里的标准化 Δ 转移样本）物化成一份
与真实 C-data 同构、同量纲、同缺失模式的完整数据集，可作为 C-data 的
"drop-in 替身"喂进同一条加载/训练管线。

生成模型（混合效应，全部参数从真实 C-data 估计）：
    value[s,g,t] = F[g,t]            固定效应：真实 group×time 单元均值（天然编码光照效应+时间趋势）
                 + b_s               随机被试截距   ~ N(0, σ_b²)   个体基线差异
                 + c_{s,g}           随机被试×光照   ~ N(0, σ_c²)   个体对光照的反应差异
                 + d_{s,t}           随机被试×时间   ~ N(0, σ_d²)   个体轨迹形状差异
                 + e_{s,g,t}         残差           ~ N(0, σ_e²)   测量噪声
    → 裁剪到真实 [min,max]，按真实精度取整/保留小数

与真实数据一致性由 construction 保证：模拟集的 group×time 均值 = 真实均值（随机效应均值为0），
因此组层面光照效应、corr(EDI,Δ)、边际分布都会对齐真实（验证见 sim_cdata_validate.py）。

运行:  python scripts/world_model/sim_cdata_gen.py [--n 60] [--seed 7] [--out sim_C-data]
输出:  scripts/world_model/sim_C-data/{HR,HRV,KSS,PANAS,VASF}/*.sav + 光谱对应表.xlsx
       + All_Data.xlsx（多表合一，便于人工查看）+ sim_meta.json（生成参数）
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdata_loader import (CDATA, GCOLS, GROUPS, MEASURE_SAV, TIMES,
                          build_dataset)

HERE = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════
# 1. 从真实数据估计生成参数
# ══════════════════════════════════════════════════════════════
def _real_columns(rel: str) -> list[str]:
    """真实 sav 的列（HF 只有 G1-G3），用于镜像结构。"""
    df, _ = pyreadstat.read_sav(str(CDATA / rel))
    return [c for c in df.columns]


def estimate_params(ds, measures_keys: list[str]) -> dict:
    """对每个指标做方差分解，得到 F[g,t] + 4 个方差分量 + 量纲范围/精度/列结构。"""
    params = {}
    for k in measures_keys:
        rel = MEASURE_SAV[k]
        meas = ds.measures.get(k, {})
        cols = _real_columns(rel)
        present = [c for c in GCOLS if c in cols]

        # 收集真实原始值 raw[(s,g,t)]
        raw = {}
        for sid in ds.subjects:
            for g in GROUPS:
                for t in TIMES:
                    col = f"G{g}T{t}"
                    if col not in present:
                        continue
                    v = meas.get(sid, {}).get(col, np.nan)
                    if v is not None and not np.isnan(v):
                        raw[(sid, g, t)] = float(v)
        if len(raw) < 5:
            continue

        # 固定效应 F[g,t] = 单元均值
        cells = {}
        for (s, g, t), v in raw.items():
            cells.setdefault((g, t), []).append(v)
        F = {gt: float(np.mean(vs)) for gt, vs in cells.items()}
        covered_groups = sorted({g for (g, t) in F})

        # 残差 r = raw - F
        r = {key: v - F[(key[1], key[2])] for key, v in raw.items()}

        # b_s：被试截距
        b = {}
        for s in ds.subjects:
            vs = [r[key] for key in r if key[0] == s]
            if vs:
                b[s] = float(np.mean(vs))
        r1 = {key: v - b.get(key[0], 0.0) for key, v in r.items()}

        # c_{s,g}：被试×光照
        c = {}
        for s in ds.subjects:
            for g in covered_groups:
                vs = [r1[key] for key in r1 if key[0] == s and key[1] == g]
                if vs:
                    c[(s, g)] = float(np.mean(vs))
        r2 = {key: v - c.get((key[0], key[1]), 0.0) for key, v in r1.items()}

        # d_{s,t}：被试×时间
        d = {}
        for s in ds.subjects:
            for t in TIMES:
                vs = [r2[key] for key in r2 if key[0] == s and key[2] == t]
                if vs:
                    d[(s, t)] = float(np.mean(vs))
        e = {key: v - d.get((key[0], key[2]), 0.0) for key, v in r2.items()}

        def _std(dic):
            a = np.array(list(dic.values()), float)
            return float(a.std()) if len(a) > 1 else 0.0

        allv = np.array(list(raw.values()), float)
        is_int = bool(np.allclose(allv, np.round(allv)))
        # 小数位：整数=0，否则看真实值的最小分辨率
        if is_int:
            decimals = 0
        else:
            frac = np.abs(allv - np.round(allv, 1))
            decimals = 1 if np.allclose(frac, 0, atol=1e-6) else 2

        params[k] = {
            "rel": rel, "sav_columns": cols, "present_gcols": present,
            "covered_groups": covered_groups,
            "F": {f"{g}_{t}": F[(g, t)] for (g, t) in F},
            "sigma_b": _std(b), "sigma_c": _std(c),
            "sigma_d": _std(d), "sigma_e": _std(e),
            "vmin": float(allv.min()), "vmax": float(allv.max()),
            "is_int": is_int, "decimals": decimals,
            "real_mean": float(allv.mean()), "real_std": float(allv.std()),
        }
    return params


# ══════════════════════════════════════════════════════════════
# 2. 生成合成被试（原始量纲）
# ══════════════════════════════════════════════════════════════
def generate(params: dict, n_subj: int, seed: int) -> dict:
    """返回 {measure: {sid: {Gcol: value}}}。"""
    rng = np.random.RandomState(seed)
    out = {k: {} for k in params}
    for i in range(n_subj):
        sid = i + 1
        for k, p in params.items():
            bs = rng.normal(0, p["sigma_b"])                     # 被试截距
            row = {}
            # 被试×时间效应（跨组共享）
            dsub = {t: rng.normal(0, p["sigma_d"]) for t in TIMES}
            for g in p["covered_groups"]:
                cg = rng.normal(0, p["sigma_c"])                 # 被试×光照
                for t in TIMES:
                    key = f"{g}_{t}"
                    if key not in p["F"]:
                        continue
                    val = p["F"][key] + bs + cg + dsub[t] + rng.normal(0, p["sigma_e"])
                    val = float(np.clip(val, p["vmin"], p["vmax"]))
                    val = round(val) if p["is_int"] else round(val, p["decimals"])
                    row[f"G{g}T{t}"] = val
            out[k][sid] = row
    return out


# ══════════════════════════════════════════════════════════════
# 3. 落盘：镜像 C-data 结构
# ══════════════════════════════════════════════════════════════
def _measure_df(gen_k: dict, p: dict) -> pd.DataFrame:
    cols = p["sav_columns"]
    rows = []
    for sid in sorted(gen_k):
        row = {"被试者": float(sid)}
        for c in cols:
            if c == "被试者":
                continue
            row[c] = gen_k[sid].get(c, np.nan)
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def write_dataset(gen: dict, params: dict, out_root: Path) -> list[Path]:
    written = []
    out_root.mkdir(parents=True, exist_ok=True)
    for k, p in params.items():
        rel = Path(p["rel"])
        path = out_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        df = _measure_df(gen[k], p)
        pyreadstat.write_sav(df, str(path))
        written.append(path)
    # 光照表直接复制真实（同一套 G1-G5 光照条件）
    for name in ["光谱对应表.xlsx", "定标数据-有EDI和相对光谱.xlsx"]:
        src = CDATA / name
        if src.exists():
            shutil.copy(src, out_root / name)
            written.append(out_root / name)
    # 合并工作簿，便于人工查看
    with pd.ExcelWriter(out_root / "All_Data.xlsx") as xw:
        for k, p in params.items():
            _measure_df(gen[k], p).to_excel(xw, sheet_name=k, index=False)
    written.append(out_root / "All_Data.xlsx")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="合成被试数（默认60，对应600条转移）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="sim_C-data")
    args = ap.parse_args()

    out_root = HERE / args.out
    print("== Step 1: 读真实 C-data 并估计生成参数（方差分解）==")
    ds = build_dataset()
    keys = list(MEASURE_SAV.keys())
    params = estimate_params(ds, keys)
    print(f"  指标 {len(params)} 个；被试 {len(ds.subjects)} 人；光照条件 {GROUPS}")
    print(f"  {'指标':<8}{'σ_b被试':>9}{'σ_c光照':>9}{'σ_d时间':>9}{'σ_e残差':>9}"
          f"{'范围':>16}{'精度':>6}")
    for k, p in params.items():
        prec = "整数" if p["is_int"] else f"{p['decimals']}位小数"
        print(f"  {k:<8}{p['sigma_b']:>9.3f}{p['sigma_c']:>9.3f}{p['sigma_d']:>9.3f}"
              f"{p['sigma_e']:>9.3f}   [{p['vmin']:.1f},{p['vmax']:.1f}]{prec:>8}"
              f"  组{p['covered_groups']}")

    print(f"\n== Step 2: 生成 {args.n} 名合成被试（seed={args.seed}）==")
    gen = generate(params, args.n, args.seed)
    n_cells = sum(len(v) for m in gen.values() for v in m.values())
    print(f"  生成单元数 {n_cells}（= 被试×指标×有效group×time）")

    print(f"\n== Step 3: 落盘到 {out_root.relative_to(HERE.parent.parent)} ==")
    written = write_dataset(gen, params, out_root)
    for w in written:
        print(f"    ✓ {w.relative_to(out_root)}")

    # 保存生成参数（可复现）
    meta = {"n_subjects": args.n, "seed": args.seed, "groups": GROUPS, "times": TIMES,
            "generative_model": "value[s,g,t]=F[g,t]+b_s+c_{s,g}+d_{s,t}+e_{s,g,t}",
            "params": {k: {kk: vv for kk, vv in p.items() if kk != "sav_columns"}
                       for k, p in params.items()}}
    (out_root / "sim_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n== 完成：模拟数据集 → {out_root} ==")


if __name__ == "__main__":
    main()
