"""
验证模拟数据集对真实 C-data 的忠实度
=====================================
五个维度对比"真实 vs 模拟"：
  ① 边际分布：每指标 mean/std/min/max
  ② 轨迹形态：group×time 单元均值的最大偏差（标准化尺度）
  ③ 光照效应：组层面 ΔKSS/ΔPA/ΔNA + corr(EDI,Δ)（世界模型最关心的因果方向）
  ④ 结构一致性：被试数、转移数、HF 缺失模式
  ⑤ 回灌管线：把模拟集当 C-data 喂进世界模型训练，看能否学到并超过 persistence、
     反事实方向是否正确 —— 最强证据：模拟集是真实集的"可用替身"

运行:  python scripts/world_model/sim_cdata_validate.py [--sim sim_C-data]
输出:  scripts/world_model/sim_cdata_validation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdata_loader import (GROUPS, STATE_KEYS, TIMES, build_dataset,
                          build_transitions)
import wm_cdata as W

HERE = Path(__file__).resolve().parent


def raw_cell_means(ds, k):
    """真实/模拟的 group×time 原始单元均值。"""
    out = {}
    for g in GROUPS:
        for t in TIMES:
            vs = [ds.measures.get(k, {}).get(sid, {}).get(f"G{g}T{t}", np.nan)
                  for sid in ds.subjects]
            vs = [v for v in vs if v is not None and not np.isnan(v)]
            if vs:
                out[(g, t)] = float(np.mean(vs))
    return out


def group_delta(ds, k):
    """每个光照条件 T1->T3 的原始平均变化。"""
    out = {}
    for g in GROUPS:
        ds_ = []
        for sid in ds.subjects:
            a = ds.measures.get(k, {}).get(sid, {}).get(f"G{g}T1", np.nan)
            b = ds.measures.get(k, {}).get(sid, {}).get(f"G{g}T3", np.nan)
            if not np.isnan(a) and not np.isnan(b):
                ds_.append(b - a)
        if ds_:
            out[g] = float(np.mean(ds_))
    return out


def marginals(ds, k):
    vs = [ds.measures.get(k, {}).get(sid, {}).get(f"G{g}T{t}", np.nan)
          for sid in ds.subjects for g in GROUPS for t in TIMES]
    a = np.array([v for v in vs if v is not None and not np.isnan(v)], float)
    return {"mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()), "n": int(len(a))}


def corr_edi_delta(ds, k, transform=None):
    edi = np.array([ds.light[g]["mel_EDI"] for g in GROUPS])
    gd = group_delta(ds, k)
    gs = sorted(gd)
    x = np.array([ds.light[g]["mel_EDI"] for g in gs])
    y = np.array([transform(gd[g]) if transform else gd[g] for g in gs])
    return float(np.corrcoef(x, y)[0, 1]) if len(gs) > 2 else float("nan")


def pipeline_on(ds, epochs=80, k_folds=5, device="cpu"):
    """把数据集喂进世界模型：K 折被试划分 CV（防泄漏）+ 反事实方向。
    用 K 折而非 LOSO：模拟集被试多，LOSO 过慢；K 折同样按被试划分、不泄漏。"""
    tr = build_transitions(ds)
    samples = W.transitions_to_samples(tr)
    light_std = [ds.action_vec(g) for g in GROUPS]

    def mock_gen(train_tr):
        c, r = W.fit_linear_dynamics(train_tr)
        c_cal, _ = W.calibrate_light_group_level(train_tr, c, light_std)
        return W.mock_subjects(ds, c_cal, r, n_subj=60, seed=1)

    # K 折被试划分
    sids = sorted({s["sid"] for s in samples})
    rng = np.random.RandomState(0); rng.shuffle(sids)
    folds = [sids[i::k_folds] for i in range(k_folds)]
    all_eval = []
    for held in folds:
        held_set = set(held)
        tr_tr = [t for t in tr if t["sid"] not in held_set]
        tr_s = [s for s in samples if s["sid"] not in held_set] + mock_gen(tr_tr)
        te = [s for s in samples if s["sid"] in held_set]
        m = W.train_model(tr_s, epochs=epochs, device=device)
        all_eval.extend(W._tag_pred(m, te, device))
    loso = W._summarize(all_eval)

    final = W.train_model(samples + mock_gen(tr), epochs=150, device=device)
    init = ds.state_vec(ds.subjects[0], 2, 1)
    plan_a = W.plan_best_light(final, ds, init, "alertness", device)
    plan_v = W.plan_best_light(final, ds, init, "valence", device)
    edi = np.array([ds.light[g]["mel_EDI"] for g in GROUPS])
    kss_pred = np.array([r["pred_KSS"] for r in sorted(plan_a, key=lambda x: x["group"])])
    val_pred = np.array([r["utility"] for r in sorted(plan_v, key=lambda x: x["group"])])
    return {
        "n_transitions": len(samples), "cv": f"{k_folds}-fold subject-split",
        "loso_overall": loso["overall"],
        "counterfactual_alertness_corr": float(np.corrcoef(edi, kss_pred)[0, 1]),
        "counterfactual_valence_corr": float(np.corrcoef(edi, val_pred)[0, 1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=str, default="sim_C-data")
    args = ap.parse_args()
    sim_root = HERE / args.sim

    print("== 加载 真实 C-data 与 模拟数据集 ==")
    real = build_dataset()
    sim = build_dataset(root=sim_root)
    print(f"  真实: {len(real.subjects)} 被试 | 模拟: {len(sim.subjects)} 被试")

    report = {"real_subjects": len(real.subjects), "sim_subjects": len(sim.subjects)}
    checks = []

    print("\n" + "=" * 82)
    print("① 边际分布对齐（真实 vs 模拟，原始量纲）")
    print("=" * 82)
    print(f"  {'指标':<6}{'真实mean±std':>18}{'模拟mean±std':>18}{'mean相对误差':>14}{'std相对误差':>12}")
    marg = {}
    for k in STATE_KEYS:
        rm, sm = marginals(real, k), marginals(sim, k)
        marg[k] = {"real": rm, "sim": sm}
        me = abs(sm["mean"] - rm["mean"]) / max(abs(rm["mean"]), 1e-6)
        se = abs(sm["std"] - rm["std"]) / max(rm["std"], 1e-6)
        ok = me < 0.10 and se < 0.25
        checks.append((f"marginal_{k}", ok))
        print(f"  {k:<6}{rm['mean']:>9.2f}±{rm['std']:<7.2f}{sm['mean']:>9.2f}±{sm['std']:<7.2f}"
              f"{me*100:>12.1f}%{se*100:>11.1f}%  {'✅' if ok else '⚠️'}")
    report["marginals"] = marg

    print("\n" + "=" * 82)
    print("② 轨迹形态对齐（group×time 单元均值最大偏差，以真实std为单位）")
    print("=" * 82)
    cellmax = {}
    for k in STATE_KEYS:
        rc, sc = raw_cell_means(real, k), raw_cell_means(sim, k)
        sd = max(marginals(real, k)["std"], 1e-6)
        diffs = [abs(rc[gt] - sc[gt]) / sd for gt in rc if gt in sc]
        mx = float(np.max(diffs)) if diffs else 0.0
        cellmax[k] = round(mx, 3)
        ok = mx < 0.35
        checks.append((f"cellmean_{k}", ok))
        print(f"  {k:<6} 最大偏差 = {mx:.3f} ×std  {'✅' if ok else '⚠️'}")
    report["cellmean_maxdev_in_std"] = cellmax

    print("\n" + "=" * 82)
    print("③ 光照效应对齐（世界模型最关心的因果方向）")
    print("=" * 82)
    edi = [real.light[g]["mel_EDI"] for g in GROUPS]
    eff = {}
    for k in ["KSS", "PA", "NA"]:
        rg, sg = group_delta(real, k), group_delta(sim, k)
        eff[k] = {"real": {str(g): round(rg.get(g, float('nan')), 2) for g in GROUPS},
                  "sim": {str(g): round(sg.get(g, float('nan')), 2) for g in GROUPS}}
        print(f"  {k} 组均值Δ(T1→T3):  真实 {[round(rg.get(g,0),2) for g in GROUPS]}")
        print(f"  {'':<16}模拟 {[round(sg.get(g,0),2) for g in GROUPS]}   (EDI={[round(e) for e in edi]})")
    rk = corr_edi_delta(real, "KSS"); sk = corr_edi_delta(sim, "KSS")
    # 效价 = PA - NA
    rgp, sgp = group_delta(real, "PA"), group_delta(sim, "PA")
    rgn, sgn = group_delta(real, "NA"), group_delta(sim, "NA")
    gsv = sorted(set(rgp) & set(rgn))
    rv = np.array([rgp[g] - rgn[g] for g in gsv]); sv = np.array([sgp[g] - sgn[g] for g in gsv])
    ev = np.array([real.light[g]["mel_EDI"] for g in gsv])
    rval = float(np.corrcoef(ev, rv)[0, 1]); sval = float(np.corrcoef(ev, sv)[0, 1])
    eff["corr_EDI_deltaKSS"] = {"real": round(rk, 3), "sim": round(sk, 3)}
    eff["corr_EDI_deltaValence"] = {"real": round(rval, 3), "sim": round(sval, 3)}
    ok_k = np.sign(rk) == np.sign(sk) and abs(sk) > 0.4
    ok_v = np.sign(rval) == np.sign(sval) and abs(sval) > 0.4
    checks.append(("corr_EDI_KSS_direction", ok_k))
    checks.append(("corr_EDI_valence_direction", ok_v))
    print(f"\n  corr(EDI, ΔKSS)   : 真实 {rk:+.3f}  模拟 {sk:+.3f}  (期望<0) {'✅' if ok_k else '⚠️'}")
    print(f"  corr(EDI, Δ效价)  : 真实 {rval:+.3f}  模拟 {sval:+.3f}  (期望>0) {'✅' if ok_v else '⚠️'}")
    report["light_effects"] = eff

    print("\n" + "=" * 82)
    print("④ 结构一致性")
    print("=" * 82)
    rtr = build_transitions(real); str_ = build_transitions(sim)
    hf_real = sorted({g for sid in real.subjects for g in GROUPS
                      if not np.isnan(real.measures.get("HF", {}).get(sid, {}).get(f"G{g}T1", np.nan))})
    hf_sim = sorted({g for sid in sim.subjects for g in GROUPS
                     if not np.isnan(sim.measures.get("HF", {}).get(sid, {}).get(f"G{g}T1", np.nan))})
    struct = {"real_transitions": len(rtr), "sim_transitions": len(str_),
              "HF_covered_groups_real": hf_real, "HF_covered_groups_sim": hf_sim}
    ok_struct = hf_real == hf_sim
    checks.append(("HF_missingness_pattern", ok_struct))
    print(f"  转移样本数: 真实 {len(rtr)} | 模拟 {len(str_)}")
    print(f"  HF 覆盖组: 真实 {hf_real} | 模拟 {hf_sim}  {'✅ 缺失模式一致' if ok_struct else '⚠️'}")
    report["structure"] = struct

    print("\n" + "=" * 82)
    print("⑤ 回灌世界模型管线（模拟集当 C-data 用）")
    print("=" * 82)
    pipe = pipeline_on(sim)
    o = pipe["loso_overall"]
    ok_pipe = o["beats_persistence"]
    ok_cfa = pipe["counterfactual_alertness_corr"] < 0
    ok_cfv = pipe["counterfactual_valence_corr"] > 0
    checks.append(("sim_beats_persistence", ok_pipe))
    checks.append(("sim_counterfactual_alertness", ok_cfa))
    checks.append(("sim_counterfactual_valence", ok_cfv))
    print(f"  {pipe['cv']}(真实+逐折重生mock,防泄漏): WM_MAE={o['next_state_mae_worldmodel']:.4f} "
          f"vs persistence={o['next_state_mae_persistence']:.4f} → "
          f"{'✅ 超' if ok_pipe else '⚠️ 未超'} {o['improvement_vs_persistence_%']:+.1f}%")
    print(f"  反事实(清醒度) corr(EDI,预测KSS)={pipe['counterfactual_alertness_corr']:+.3f} (期望<0) {'✅' if ok_cfa else '⚠️'}")
    print(f"  反事实(情绪效价) corr(EDI,预测效价)={pipe['counterfactual_valence_corr']:+.3f} (期望>0) {'✅' if ok_cfv else '⚠️'}")
    report["pipeline_on_sim"] = pipe

    n_pass = sum(1 for _, ok in checks if ok)
    report["checks"] = [{"name": n, "passed": bool(ok)} for n, ok in checks]
    report["summary"] = {"n_checks": len(checks), "n_passed": n_pass,
                         "all_passed": bool(n_pass == len(checks))}
    print("\n" + "=" * 82)
    print(f"验证汇总: {n_pass}/{len(checks)} 项通过")
    print("=" * 82)
    for n, ok in checks:
        print(f"  {'✅' if ok else '⚠️'} {n}")
    (HERE / "sim_cdata_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n== 验证报告 → sim_cdata_validation.json ==")


if __name__ == "__main__":
    main()
