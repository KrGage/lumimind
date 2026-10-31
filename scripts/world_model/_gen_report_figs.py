# -*- coding: utf-8 -*-
"""从世界模型真实运行产物(JSON报告)生成报告图表, 并内嵌到 HTML 模板."""
import base64
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WM = os.path.join(ROOT, "scripts", "world_model")
DOCS = os.path.join(ROOT, "docs")
FIGDIR = os.path.join(WM, "_figs_wm")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

BLUE, ORANGE, GRAY, GREEN, RED, TEAL = "#2B6CB0", "#C05621", "#a0aec0", "#2f855a", "#c53030", "#319795"

with open(os.path.join(WM, "wm_report.json"), encoding="utf-8") as f:
    R1 = json.load(f)
with open(os.path.join(WM, "wm_cdata_report.json"), encoding="utf-8") as f:
    R2 = json.load(f)
with open(os.path.join(WM, "sim_cdata_validation.json"), encoding="utf-8") as f:
    R3 = json.load(f)


def save(fig, name):
    p = os.path.join(FIGDIR, name)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", p)


# ---------- 图1: 管线1 双数据集评估 (R2) ----------
ev = R1["evaluation"]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
for ax, key, title in zip(axes, ["sim_test", "deap_test"],
                          ["模拟测试集 (n=1500)", "DEAP 真实转移 (n=312)"]):
    d = ev[key]
    labels = ["Valence", "Arousal"]
    wm = [d["world_model"]["r2_valence"], d["world_model"]["r2_arousal"]]
    per = [d["baseline_persistence"]["r2_valence"], d["baseline_persistence"]["r2_arousal"]]
    gm = [0.0, 0.0]
    x = np.arange(2); w = 0.26
    b1 = ax.bar(x - w, wm, w, label="世界模型", color=BLUE)
    b2 = ax.bar(x, per, w, label="持久性基线", color=GRAY)
    b3 = ax.bar(x + w, gm, w, label="全局均值基线", color=ORANGE)
    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.axhline(0, color="#4a5568", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("R²")
    ax.set_ylim(min(min(per), 0) - 0.12, 1.12)
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("管线1: Causal Transformer 在模拟集与 DEAP 真实集上的预测精度", fontsize=12)
save(fig, "wm_fig1_sim_deap.png")

# ---------- 图2: C-data LOSO 总体 MAE ----------
ro, rm = R2["loso_real_only"]["overall"], R2["loso_real_plus_mock"]["overall"]
fig, ax = plt.subplots(figsize=(8.6, 4.0))
labels = ["仅真实数据训练\n(诚实评估)", "真实+mock 训练\n(逐折重生,防泄漏)"]
wm = [ro["next_state_mae_worldmodel"], rm["next_state_mae_worldmodel"]]
per = [ro["next_state_mae_persistence"], rm["next_state_mae_persistence"]]
gm = [ro["next_state_mae_globalmean_delta"], rm["next_state_mae_globalmean_delta"]]
x = np.arange(2); w = 0.26
b1 = ax.bar(x - w, wm, w, label="世界模型", color=BLUE)
b2 = ax.bar(x, per, w, label="持久性基线", color=GRAY)
b3 = ax.bar(x + w, gm, w, label="全局Δ均值基线", color=ORANGE)
for bars in (b1, b2, b3):
    ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=2)
ax.annotate("击败持久性基线 [√]\n改进 +5.0%", xy=(1 - w, wm[1]), xytext=(1.28, 0.60),
            fontsize=10, color=GREEN, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=GREEN))
ax.annotate("未击败基线 [×]\n-3.2%", xy=(0 - w, wm[0]), xytext=(-0.42, 0.83),
            fontsize=10, color=RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED))
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("下一状态预测 MAE (z 分数, 越低越好)")
ax.set_ylim(0.55, 0.92)
ax.set_title("管线2: 真实 C-data LOSO 交叉验证 — mock 数据注入前后的对比 (n=100)", fontsize=12)
ax.legend(fontsize=9)
save(fig, "wm_fig2_loso.png")

# ---------- 图3: real+mock 逐维 delta_R2 ----------
pd = R2["loso_real_plus_mock"]["per_dim"]
dims = list(pd.keys())
dr2 = [pd[k]["delta_r2_vs_mean"] for k in dims]
beat = [pd[k]["beats_persistence"] for k in dims]
colors = [GREEN if b else RED for b in beat]
fig, ax = plt.subplots(figsize=(8.6, 4.2))
y = np.arange(len(dims))[::-1]
bars = ax.barh(y, dr2, color=colors, alpha=0.88)
ax.axvline(0, color="#4a5568", lw=0.8)
for yi, v, b in zip(y, dr2, beat):
    ax.text(v + (0.008 if v >= 0 else -0.008), yi, f"{v:+.3f}" + ("  √" if b else "  ×"),
            va="center", ha="left" if v >= 0 else "right", fontsize=9,
            color="#1a202c")
ax.set_yticks(y); ax.set_yticklabels(dims)
ax.set_xlabel("ΔR² (相对全局Δ均值基线)")
ax.set_xlim(-0.34, 0.40)
ax.set_title("管线2: 真实+mock 逐维预测增益 (绿=优于持久性基线, 6/8 维达标)", fontsize=12)
save(fig, "wm_fig3_perdim.png")

# ---------- 图4: 规划演示 ----------
plan = R2["planning"]
al = plan["alertness_ranking"]; va = plan["valence_ranking"]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
# 左: 清醒度目标 → 预测KSS (越低越好)
gs = [f"G{r['group']}" for r in al]
edi = [r["mel_EDI"] for r in al]
kss = [r["pred_KSS"] for r in al]
order = np.argsort(kss)
gs = [gs[i] for i in order]; edi = [edi[i] for i in order]; kss = [kss[i] for i in order]
bars = axes[0].bar(np.arange(len(gs)), kss,
                   color=[GREEN if k == min(kss) else (ORANGE if k == max(kss) else BLUE) for k in kss])
for i, (k, e) in enumerate(zip(kss, edi)):
    axes[0].text(i, k + 0.015, f"{k:.3f}\nEDI={e:.0f}", ha="center", fontsize=8.5)
axes[0].set_xticks(np.arange(len(gs))); axes[0].set_xticklabels(gs)
axes[0].set_ylabel("预测 KSS (z, 越低越清醒)")
axes[0].set_ylim(0, 1.12)
axes[0].set_title("目标=提升清醒度: 5 种光照排序\n(G4 最低照度最优, G3 最差)", fontsize=10.5)
# 右: 效价目标 → utility (越高越好)
gs2 = [f"G{r['group']}" for r in va]
ut = [r["utility"] for r in va]
order2 = np.argsort(ut)[::-1]
gs2 = [gs2[i] for i in order2]; ut = [ut[i] for i in order2]
axes[1].bar(np.arange(len(gs2)), ut,
            color=[GREEN if u == max(ut) else (ORANGE if u == min(ut) else BLUE) for u in ut])
for i, u in enumerate(ut):
    axes[1].text(i, u + (0.012 if u >= 0 else -0.03), f"{u:+.3f}", ha="center", fontsize=9)
axes[1].axhline(0, color="#4a5568", lw=0.8)
axes[1].set_xticks(np.arange(len(gs2))); axes[1].set_xticklabels(gs2)
axes[1].set_ylabel("效价效用 U = PA−NA (越高越好)")
axes[1].set_ylim(-0.68, 0.56)
axes[1].set_title("目标=提升情绪效价: 5 种光照排序\n(G2/G5 高 EDI 冷白光最优)", fontsize=10.5)
fig.suptitle("规划演示: 用训练好的世界模型对 5 种真实定标光照做反事实排序 (初始状态=被试1)", fontsize=12)
save(fig, "wm_fig4_planning.png")

# ---------- 图5: 反事实一致性检验 ----------
cf = R2["counterfactual_check"]
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
for ax, key, title in zip(axes, ["alertness", "valence"],
                          ["清醒度: corr(EDI, 预测KSS) 期望 < 0", "效价: corr(EDI, 预测效价) 期望 > 0"]):
    c = cf[key]
    if key == "alertness":
        vals = [c["corr_EDI_vs_predKSS"], c["real_corr_transition_level"], c["real_corr_group_level"]]
    else:
        vals = [c["corr_EDI_vs_predValenceUtility"], c["real_corr_transition_level"], c["real_corr_group_level"]]
    names = ["世界模型\n(反事实)", "真实数据\n(转移层面)", "真实数据\n(组层面)"]
    cols = [BLUE, GRAY, GREEN]
    bars = ax.bar(names, vals, color=cols, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.075),
                f"{v:+.3f}", ha="center", fontsize=9.5)
    ax.axhline(0, color="#4a5568", lw=0.8)
    ax.set_title(title + ("  [通过]" if c["passed"] else "  [未通过]"), fontsize=10.5)
    ax.set_ylim(-1.0, 1.15)
fig.suptitle("反事实方向检验: 模型学到的光照效应方向与真实数据组层面一致", fontsize=12)
save(fig, "wm_fig5_counterfactual.png")

# ---------- 图6: 模拟数据集保真度校验 (光照效应) ----------
le = R3["light_effects"]
groups = ["G1", "G2", "G3", "G4", "G5"]
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9))
for ax, key, title, ylab in zip(
        axes, ["KSS", "PA"],
        ["光照组 → ΔKSS (困倦度变化)", "光照组 → ΔPA (正性情绪变化)"],
        ["ΔKSS (真实组内变化)", "ΔPA (真实组内变化)"]):
    real = [le[key]["real"][str(g)] for g in range(1, 6)]
    sim = [le[key]["sim"][str(g)] for g in range(1, 6)]
    x = np.arange(5); w = 0.36
    b1 = ax.bar(x - w / 2, real, w, label="真实 C-data", color=BLUE)
    b2 = ax.bar(x + w / 2, sim, w, label="模拟数据 (60 虚拟被试)", color=TEAL)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", fontsize=8.5, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.axhline(0, color="#4a5568", lw=0.8)
    ax.set_title(title, fontsize=10.5)
    ax.set_ylabel(ylab)
    ax.legend(fontsize=8.5)
fig.suptitle("模拟 C-data 保真度: 逐光照组效应复现 (ΔKSS 组级相关 r=-0.751→-0.829; 22/22 项校验全部通过)", fontsize=11.5)
save(fig, "wm_fig6_simcheck.png")

# ---------- 图7: ReAct 闭环 ----------
sc = R1["react_closed_loop"]["scenarios"]
names = ["焦虑用户\n(高唤醒低效价)", "困倦用户\n(低唤醒)", "亚健康用户\n(轻度低效价)"]
dv = [s["delta_valence"] for s in sc]
da = [s["delta_arousal"] for s in sc]
plans = [f"{s['plan']['cct']}K / {s['plan']['lux']}lux / {s['plan']['duration']}min" for s in sc]
fig, ax = plt.subplots(figsize=(9.2, 4.0))
x = np.arange(3); w = 0.35
b1 = ax.bar(x - w / 2, dv, w, label="Δ Valence (期望↑)", color=GREEN)
b2 = ax.bar(x + w / 2, da, w, label="Δ Arousal (期望↓)", color=BLUE)
for bars in (b1, b2):
    ax.bar_label(bars, fmt="%+.3f", fontsize=9.5, padding=3)
for i, p in enumerate(plans):
    ax.text(i, max(dv[i], 0) + 0.012, f"处方: {p}", ha="center", fontsize=8.5, color="#4a5568")
ax.axhline(0, color="#4a5568", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("干预前后情绪变化")
ax.set_ylim(-0.08, 0.10)
ax.set_title(f"ReAct 闭环: 世界模型选光 → 模拟执行 → 效果评估 (3/3 场景改善, 改善率 {R1['react_closed_loop']['improvement_rate']:.0%})", fontsize=12)
ax.legend(fontsize=9)
save(fig, "wm_fig7_react.png")

# ---------- 内嵌到 HTML ----------
def to_data_uri(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

tmpl_path = os.path.join(DOCS, "_wm_report_template.html")
out_path = os.path.join(DOCS, "世界模型实现结果报告.html")
with open(tmpl_path, encoding="utf-8") as f:
    html = f.read()
fig_names = {
    1: "wm_fig1_sim_deap.png",
    2: "wm_fig2_loso.png",
    3: "wm_fig3_perdim.png",
    4: "wm_fig4_planning.png",
    5: "wm_fig5_counterfactual.png",
    6: "wm_fig6_simcheck.png",
    7: "wm_fig7_react.png",
}
for i in range(1, 8):
    html = html.replace("{{FIG%d}}" % i, to_data_uri(os.path.join(FIGDIR, fig_names[i])))
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
os.remove(tmpl_path)
print("HTML written:", out_path, "size =", os.path.getsize(out_path) // 1024, "KB")
