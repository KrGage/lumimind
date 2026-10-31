"""
ecg.pth 模型在 DEAP/SEED 框架下的性能可视化图
参考 PhysioWave (NeurIPS 2025) 论文的多模态融合架构
"""
import sys, os
sys.path.insert(0, r"d:\研究生\项目内容\杂事\光源智能体设计大赛\项目主体\reference\example\PhysioWave-main")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.gridspec as gridspec

# ============================================================
# 全局绘图风格
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

OUTPUT_DIR = r"d:\研究生\项目内容\杂事\光源智能体设计大赛\项目主体\scripts"

# ============================================================
# Figure 1: 综合性能 + 架构图
# ============================================================
def figure1_performance_and_architecture():
    """综合性能图 + DEAP 多模态架构图"""
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('PhysioWave ecg.pth 模型验证 — DEAP/SEED 框架性能分析',
                 fontsize=18, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35,
                           left=0.06, right=0.96, top=0.92, bottom=0.06)

    # ---- Panel (a): PTB-XL 验证性能 ----
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Accuracy', 'Balanced\nAcc', 'Weighted\nF1', 'AUROC']
    values = [86.40, 77.98, 85.82, 94.66]
    colors_ptbxl = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    bars = ax1.bar(metrics, values, color=colors_ptbxl, width=0.6,
                   edgecolor='white', linewidth=1.5, zorder=3)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.set_ylabel('性能 (%)', fontsize=12)
    ax1.set_title('(a) ecg.pth PTB-XL 验证性能', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, zorder=0)
    ax1.set_axisbelow(True)

    # ---- Panel (b): 模型参数分布 ----
    ax2 = fig.add_subplot(gs[0, 1])
    module_names = ['Transformer\nEncoder', 'Task Heads\n(Class+Recon)', 'Patch\nEmbedding', 'Wavelet\nDecomp']
    module_params = [14.20, 0.57, 0.03, 0.03]
    module_colors = ['#1565C0', '#43A047', '#FB8C00', '#8E24AA']
    wedges, texts, autotexts = ax2.pie(
        module_params, labels=module_names, colors=module_colors,
        autopct=lambda p: f'{p:.1f}%' if p > 2 else '',
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))
    for t in texts:
        t.set_fontsize(9)
    ax2.set_title('(b) 模型参数分布 (总计 14.83M)', fontsize=13, fontweight='bold')

    # ---- Panel (c): 小波分解可视化 ----
    ax3 = fig.add_subplot(gs[0, 2])
    t = np.linspace(0, 4.096, 2048)
    # 模拟一个 ECG 信号
    ecg = np.zeros(2048)
    for beat in np.arange(0, 2048, 415):
        # QRS
        for dt, amp, w in [(-12, -0.15, 6), (0, 1.0, 8), (10, -0.2, 6)]:
            idx = int(beat + dt)
            if 0 <= idx < 2048:
                for i in range(-w, w+1):
                    if 0 <= idx+i < 2048:
                        ecg[idx+i] += amp * np.exp(-0.5*(i/w*3)**2)
    ecg += np.random.randn(2048) * 0.02
    ecg = ecg / (np.abs(ecg).max() + 1e-8)
    ax3.plot(t, ecg, color='#1565C0', linewidth=1.2, label='Lead I')
    ax3.fill_between(t, ecg, alpha=0.15, color='#1565C0')
    ax3.set_xlabel('时间 (秒)', fontsize=11)
    ax3.set_ylabel('归一化幅度', fontsize=11)
    ax3.set_title('(c) 12 导联 ECG 信号示例 (Lead I)', fontsize=13, fontweight='bold')
    ax3.set_xlim(0, 4.096)
    ax3.grid(alpha=0.3)
    ax3.legend(loc='upper right')

    # ---- Panel (d): PhysioWave 多模态 DEAP 架构图 ----
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_xlim(0, 20)
    ax4.set_ylim(0, 8)
    ax4.axis('off')
    ax4.set_title('(d) PhysioWave 多模态融合架构 — DEAP 情感识别框架', fontsize=13, fontweight='bold')

    # 绘制方块辅助函数
    def draw_box(ax, x, y, w, h, text, color, fontsize=10, text_color='white', alpha=0.85):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor='white', linewidth=2, alpha=alpha, zorder=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=text_color, fontweight='bold', zorder=3)

    def draw_arrow(ax, x1, y1, x2, y2, color='#666', style='->', lw=2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw), zorder=1)

    # --- 输入层 ---
    draw_box(ax4, 0.3, 5.5, 2.8, 1.8, 'DEAP\nEEG\n(32 ch, 128Hz)', '#42A5F5', fontsize=10)
    draw_box(ax4, 0.3, 3.0, 2.8, 1.8, 'DEAP\nEMG\n(2 ch, 128Hz)', '#66BB6A', fontsize=10)

    # --- 编码器层 ---
    draw_box(ax4, 5.0, 5.5, 3.2, 1.8, 'EEG Encoder\n(EEGPT / LaBraM)\n预训练 EEG 模型', '#1565C0', fontsize=9)
    draw_box(ax4, 5.0, 3.0, 3.2, 1.8, 'EMG Encoder\nPhysioWave-emg\n(5M params)', '#2E7D32', fontsize=9)

    # ecg.pth 参考 (虚线框)
    rect = FancyBboxPatch((5.0, 0.5), 3.2, 1.8, boxstyle="round,pad=0.15",
                          facecolor='#FFF3E0', edgecolor='#E65100', linewidth=2.5,
                          linestyle='--', alpha=0.9, zorder=2)
    ax4.add_patch(rect)
    ax4.text(6.6, 1.4, 'ECG Encoder\nPhysioWave-ecg\n(ecg.pth, 14.8M)\n[!] DEAP 无 12 导联 ECG', ha='center', va='center',
             fontsize=9, color='#E65100', fontweight='bold', zorder=3)

    # --- 特征提取 ---
    draw_box(ax4, 9.5, 5.5, 2.5, 1.8, 'EEG Embedding\n(384-d)', '#1565C0', fontsize=10, alpha=0.7)
    draw_box(ax4, 9.5, 3.0, 2.5, 1.8, 'EMG Embedding\n(384-d)', '#2E7D32', fontsize=10, alpha=0.7)

    # --- Channel-wise Average ---
    draw_box(ax4, 13.0, 4.2, 2.2, 1.8, 'Channel\nAverage', '#78909C', fontsize=10)

    # --- 融合层 ---
    draw_box(ax4, 16.0, 4.2, 2.0, 1.8, 'Concat\nFusion', '#FF6F00', fontsize=11)

    # --- 分类头 ---
    draw_box(ax4, 18.5, 4.2, 1.2, 1.8, 'Classifier\nHead', '#C62828', fontsize=10)

    # --- 输出 ---
    draw_box(ax4, 17.0, 7.0, 2.5, 0.7, 'Valence / Arousal / Dominance', '#B71C1C', fontsize=9)

    # --- 箭头 ---
    draw_arrow(ax4, 3.1, 6.4, 5.0, 6.4)
    draw_arrow(ax4, 3.1, 3.9, 5.0, 3.9)
    draw_arrow(ax4, 8.2, 6.4, 9.5, 6.4)
    draw_arrow(ax4, 8.2, 3.9, 9.5, 3.9)
    draw_arrow(ax4, 12.0, 6.4, 13.5, 5.5, color='#1565C0')
    draw_arrow(ax4, 12.0, 3.9, 13.5, 4.5, color='#2E7D32')
    draw_arrow(ax4, 15.2, 5.1, 16.0, 5.1)
    draw_arrow(ax4, 18.0, 5.1, 18.5, 5.1)
    draw_arrow(ax4, 19.1, 6.0, 18.5, 7.0)

    # 虚线箭头表示 ecg.pth 不直接使用
    ax4.annotate('', xy=(5.0, 1.4), xytext=(3.1, 3.0),
                 arrowprops=dict(arrowstyle='->', color='#E65100', lw=2, linestyle='dashed'))
    ax4.text(3.7, 1.7, '(DEAP 无 ECG)', fontsize=9, color='#E65100', fontstyle='italic')

    # 图例
    legend_elements = [
        mpatches.Patch(facecolor='#1565C0', edgecolor='white', label='EEG 分支'),
        mpatches.Patch(facecolor='#2E7D32', edgecolor='white', label='EMG 分支'),
        mpatches.Patch(facecolor='#FFF3E0', edgecolor='#E65100', label='ECG 分支 (ecg.pth, 需 ECG 数据)', linestyle='--'),
        mpatches.Patch(facecolor='#FF6F00', edgecolor='white', label='融合层'),
    ]
    ax4.legend(handles=legend_elements, loc='lower right', fontsize=10, framealpha=0.9)

    fig.savefig(os.path.join(OUTPUT_DIR, 'ecg_deap_framework.png'))
    plt.close(fig)
    print("[OK] figure1 saved: ecg_deap_framework.png")


# ============================================================
# Figure 2: DEAP 论文性能对比 + SEED 验证策略路线图
# ============================================================
def figure2_deap_benchmark_and_seed_strategy():
    """DEAP 性能对比 + SEED 验证策略"""
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('PhysioWave 多模态框架性能对比 & SEED 验证策略',
                 fontsize=18, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3,
                           left=0.06, right=0.96, top=0.90, bottom=0.08)

    # ---- 左: DEAP 性能对比 ----
    ax1 = fig.add_subplot(gs[0, 0])

    # PhysioWave 论文 Figure 5 报告的数据
    methods = ['EEGPT\n(EEG only)', 'Bi-LSTM\n+Attention', 'Bayesian\nPipeline', 'TPRO-Net', 'PhysioWave\n(Multimodal)']
    valence = [79.1, 63.0, 72.0, 84.8, 85.2]
    arousal = [81.3, 58.0, 66.0, 84.0, 88.6]

    x = np.arange(len(methods))
    w = 0.35
    bars1 = ax1.bar(x - w/2, valence, w, label='Valence', color='#1E88E5', edgecolor='white', linewidth=1.2)
    bars2 = ax1.bar(x + w/2, arousal, w, label='Arousal', color='#E53935', edgecolor='white', linewidth=1.2)

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.8, f'{h:.1f}%', ha='center', fontsize=9, fontweight='bold', color='#1E88E5')
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.8, f'{h:.1f}%', ha='center', fontsize=9, fontweight='bold', color='#E53935')

    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('(a) DEAP 情感识别性能对比 (PhysioWave 论文数据)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_axisbelow(True)

    # 标注 PhysioWave 多模态提升
    ax1.annotate('+6.1%\nValence', xy=(4 - w/2, 85.2), xytext=(3.2, 92),
                 fontsize=10, color='#1E88E5', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#1E88E5', lw=1.5))
    ax1.annotate('+7.3%\nArousal', xy=(4 + w/2, 88.6), xytext=(4.3, 95),
                 fontsize=10, color='#E53935', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.5))

    # ---- 右: SEED 验证策略路线图 ----
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('(b) SEED 数据集验证策略路线图', fontsize=13, fontweight='bold')

    def draw_rounded_box(ax, x, y, w, h, text, color, fontsize=9, tc='white'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                             facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9, zorder=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, color=tc, fontweight='bold', zorder=3, linespacing=1.4)

    def arrow(ax, x1, y1, x2, y2, c='#555'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.8), zorder=1)

    # 标题行
    ax2.text(5, 9.5, 'SEED 数据集 (EEG 62ch + 眼动)', ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#1565C0', lw=2))

    # --- 策略 A ---
    draw_rounded_box(ax2, 0.2, 7.5, 3, 1.3,
                     '策略A: EEG 编码器替换\n使用 EEGPT/LaBraM\n(推荐, 直接适用)', '#1565C0', fontsize=8)
    arrow(ax2, 1.7, 8.8, 1.7, 7.5+1.3, c='#1565C0')

    draw_rounded_box(ax2, 0.2, 5.5, 3, 1.3,
                     '步骤1: 加载 SEED EEG\n62ch → 选32ch / 重采样\n匹配 EEGPT 输入规格', '#42A5F5', fontsize=8)
    arrow(ax2, 1.7, 7.5, 1.7, 5.5+1.3, c='#42A5F5')

    draw_rounded_box(ax2, 0.2, 3.5, 3, 1.3,
                     '步骤2: 冻结 EEG 编码器\n提取 embedding\n3 类情感分类微调', '#90CAF9', fontsize=8, tc='#1565C0')
    arrow(ax2, 1.7, 5.5, 1.7, 3.5+1.3, c='#90CAF9')

    draw_rounded_box(ax2, 0.2, 1.8, 3, 1.0,
                     '输出: Valence 三分类\nAccuracy / F1 / AUROC', '#E3F2FD', fontsize=8, tc='#1565C0')
    arrow(ax2, 1.7, 3.5, 1.7, 1.8+1.0, c='#E3F2FD')

    # --- 策略 B ---
    draw_rounded_box(ax2, 3.7, 7.5, 3, 1.3,
                     '策略B: 跨模态迁移 (实验性)\necg.pth 编码器 → EEG\n需要修改 in_channels', '#E65100', fontsize=8)
    arrow(ax2, 5.2, 8.8, 5.2, 7.5+1.3, c='#E65100')

    draw_rounded_box(ax2, 3.7, 5.5, 3, 1.3,
                     '步骤1: 修改输入通道\n62ch → pad 或 select 12ch\n重采样 200Hz → 500Hz', '#FF8A65', fontsize=8)
    arrow(ax2, 5.2, 7.5, 5.2, 5.5+1.3, c='#FF8A65')

    draw_rounded_box(ax2, 3.7, 3.5, 3, 1.3,
                     '步骤2: 替换分类头\n5 类 → 3 类 (pos/neu/neg)\nLinear Probing 或 微调', '#FFCCBC', fontsize=8, tc='#E65100')
    arrow(ax2, 5.2, 5.5, 5.2, 3.5+1.3, c='#FFCCBC')

    draw_rounded_box(ax2, 3.7, 1.8, 3, 1.0,
                     '输出: 3 类情感 Accuracy\n(对比策略A, 验证迁移效果)', '#FFF3E0', fontsize=8, tc='#E65100')
    arrow(ax2, 5.2, 3.5, 5.2, 1.8+1.0, c='#FFCCBC')

    # --- 策略 C ---
    draw_rounded_box(ax2, 7.2, 7.5, 2.6, 1.3,
                     '策略C: 特征级验证\necg.pth 的 DE 特征\n对比 SEED DE 特征', '#2E7D32', fontsize=8)
    arrow(ax2, 8.5, 8.8, 8.5, 7.5+1.3, c='#2E7D32')

    draw_rounded_box(ax2, 7.2, 5.5, 2.6, 1.3,
                     '步骤1: 提取小波特征\necg.pth wavelet_decomp\n对 SEED EEG 各频段', '#66BB6A', fontsize=8)
    arrow(ax2, 8.5, 7.5, 8.5, 5.5+1.3, c='#66BB6A')

    draw_rounded_box(ax2, 7.2, 3.5, 2.6, 1.3,
                     '步骤2: 对比分析\nDE/DASM/RASM 特征\nt-SNE 可视化聚类', '#A5D6A7', fontsize=8, tc='#2E7D32')
    arrow(ax2, 8.5, 5.5, 8.5, 3.5+1.3, c='#A5D6A7')

    draw_rounded_box(ax2, 7.2, 1.8, 2.6, 1.0,
                     '输出: 特征质量评估\n可分性 / 聚类指标', '#E8F5E9', fontsize=8, tc='#2E7D32')
    arrow(ax2, 8.5, 3.5, 8.5, 1.8+1.0, c='#A5D6A7')

    # 推荐标记
    ax2.text(1.7, 7.3, '★ 推荐', ha='center', fontsize=10, color='#FFD600', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#1B5E20', edgecolor='#FFD600', lw=1.5))
    ax2.text(5.2, 7.3, '实验性', ha='center', fontsize=9, color='white', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#BF360C', edgecolor='white', lw=1))
    ax2.text(8.5, 7.3, '辅助验证', ha='center', fontsize=9, color='white', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#1B5E20', edgecolor='white', lw=1))

    fig.savefig(os.path.join(OUTPUT_DIR, 'ecg_deap_seed_strategy.png'))
    plt.close(fig)
    print("[OK] figure2 saved: ecg_deap_seed_strategy.png")


# ============================================================
# Figure 3: 信号规格对比 & 适配方案图
# ============================================================
def figure3_signal_compatibility():
    """DEAP/SEED/DREAMER 信号规格对比 + ecg.pth 适配方案"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('数据集信号规格对比 & ecg.pth 适配性分析', fontsize=16, fontweight='bold', y=1.02)

    # ---- 左: 通道数 & 采样率对比 ----
    ax1 = axes[0]
    datasets = ['DEAP\n(EEG)', 'DEAP\n(EMG)', 'SEED\n(EEG)', 'DREAMER\n(EEG)', 'DREAMER\n(ECG)', 'PTB-XL\n(ECG)', 'ecg.pth\n需求']
    channels = [32, 2, 62, 14, 2, 12, 12]
    sample_rates = [128, 128, 200, 128, 256, 500, 500]

    x = np.arange(len(datasets))
    colors_ch = ['#42A5F5', '#66BB6A', '#42A5F5', '#42A5F5', '#66BB6A', '#FFA726', '#EF5350']
    colors_sr = ['#1E88E5', '#43A047', '#1E88E5', '#1E88E5', '#43A047', '#F57C00', '#C62828']

    ax1_twin = ax1.twinx()
    w = 0.35
    bars_ch = ax1.bar(x - w/2, channels, w, label='通道数', color=colors_ch, edgecolor='white', linewidth=1.2, alpha=0.85)
    bars_sr = ax1_twin.bar(x + w/2, sample_rates, w, label='采样率 (Hz)', color=colors_sr, edgecolor='white', linewidth=1.2, alpha=0.6, hatch='//')

    for bar, ch in zip(bars_ch, channels):
        ax1.text(bar.get_x() + bar.get_width()/2, ch + 1, str(ch), ha='center', fontsize=9, fontweight='bold')
    for bar, sr in zip(bars_sr, sample_rates):
        ax1_twin.text(bar.get_x() + bar.get_width()/2, sr + 10, str(sr), ha='center', fontsize=9, color='#C62828')

    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=9)
    ax1.set_ylabel('通道数', fontsize=11, color='#1E88E5')
    ax1_twin.set_ylabel('采样率 (Hz)', fontsize=11, color='#C62828')
    ax1.set_title('(a) 各数据集信号规格对比', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 75)
    ax1_twin.set_ylim(0, 600)

    # 画 ecg.pth 需求线
    ax1.axhline(y=12, color='#EF5350', linestyle='--', alpha=0.5, lw=1.5)
    ax1.text(-0.5, 12.5, 'ecg.pth: 12ch', fontsize=8, color='#EF5350')
    ax1_twin.axhline(y=500, color='#C62828', linestyle='--', alpha=0.5, lw=1.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    # ---- 右: 适配性评估矩阵 ----
    ax2 = axes[1]
    ax2.axis('off')
    ax2.set_title('(b) ecg.pth 在各数据集上的适配性评估', fontsize=13, fontweight='bold')

    # 表格数据
    col_labels = ['数据集', '信号类型', '通道匹配', '采样率匹配', '情感标签', '总体适配性']
    table_data = [
        ['DEAP', 'EEG 32ch', '[X] 32!=12', '[X] 128!=500', 'V/A/D 连续', '[X] 不适用'],
        ['DEAP', 'EMG 2ch', '[X] 2!=12', '[X] 128!=500', 'V/A/D 连续', '[X] 不适用'],
        ['SEED', 'EEG 62ch', '[~] 可裁剪', '[~] 200->500', '3 类离散', '[~] 实验性'],
        ['DREAMER', 'EEG 14ch', '[~] 近似', '[X] 128!=500', 'V/A/D 连续', '[~] 需适配'],
        ['DREAMER', 'ECG 2ch', '[X] 2!=12', '[~] 256->500', 'V/A/D 连续', '[~] 需填充'],
        ['PTB-XL', 'ECG 12ch', '[OK] 精确匹配', '[OK] 500Hz', '5 类心律失常', '[OK] 原生适用'],
    ]

    colors_cell = []
    for row in table_data:
        row_colors = []
        for cell in row:
            if '[OK]' in cell:
                row_colors.append('#C8E6C9')
            elif '[X]' in cell:
                row_colors.append('#FFCDD2')
            elif '[~]' in cell:
                row_colors.append('#FFF9C4')
            else:
                row_colors.append('white')
        colors_cell.append(row_colors)

    table = ax2.table(cellText=table_data, colLabels=col_labels,
                      cellColours=colors_cell,
                      colColours=['#E3F2FD']*6,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_linewidth(1.5)
        if row == 0:
            cell.set_text_props(fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'ecg_compatibility_matrix.png'))
    plt.close(fig)
    print("[OK] figure3 saved: ecg_compatibility_matrix.png")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("正在生成可视化图表...")
    figure1_performance_and_architecture()
    figure2_deap_benchmark_and_seed_strategy()
    figure3_signal_compatibility()
    print(f"\n所有图表已保存到: {OUTPUT_DIR}")
