"""
工作室光谱数据批量可视化
读取 工作室/ 目录下调光前、调光后的 CSV 光谱数据
按点位分组绘制调光前/后的光谱对比图
输出到 outputs/ 目录
"""

import os
import re
import glob

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ─── 路径配置 ───────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.join(SCRIPT_DIR, '..', '..', '..', '工作室')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs', '工作室光谱')

# 可见光范围
WL_MIN, WL_MAX = 380, 780

# ─── 方向名排序 ───────────────────────────────────────────────
DIRECTION_ORDER = ['上', '下', '左', '右', '前', '后']


# ─── 颜色工具 ────────────────────────────────────────────────
def wavelength_to_rgb(wavelength):
    """将可见光波长(nm)转换为 RGB 颜色（近似 CIE 算法）"""
    if wavelength < 380 or wavelength > 780:
        return (0.5, 0.5, 0.5)
    if wavelength < 440:
        r, g, b = (440 - wavelength) / 60, 0, 1
    elif wavelength < 490:
        r, g, b = 0, (wavelength - 440) / 50, 1
    elif wavelength < 510:
        r, g, b = 0, 1, (510 - wavelength) / 20
    elif wavelength < 580:
        r, g, b = (wavelength - 510) / 70, 1, 0
    elif wavelength < 645:
        r, g, b = 1, (645 - wavelength) / 65, 0
    else:
        r, g, b = 1, 0, 0

    if wavelength < 420:
        f = 0.3 + 0.7 * (wavelength - 380) / 40
    elif wavelength > 700:
        f = 0.3 + 0.7 * (780 - wavelength) / 80
    else:
        f = 1.0
    return (r * f, g * f, b * f)


def plot_spectrum_fill(ax, wavelengths, values_norm, alpha=0.35):
    """在曲线下方绘制渐变光谱填充"""
    unique_wls = np.unique(wavelengths)
    for i in range(len(unique_wls) - 1):
        wl_s, wl_e = unique_wls[i], unique_wls[i + 1]
        rgb = wavelength_to_rgb((wl_s + wl_e) / 2)
        mask = (wavelengths >= wl_s) & (wavelengths <= wl_e)
        wl_range = wavelengths[mask]
        val_range = values_norm[mask]
        if len(wl_range) > 0:
            ax.fill_between(wl_range, 0, val_range,
                            color=rgb, alpha=alpha, linewidth=0)
    return ax


# ─── 数据读取 ────────────────────────────────────────────────
def load_csv_spectrum(filepath):
    """
    读取单个 CSV 光谱文件（波长,强度 格式）
    返回: (wavelengths, intensities) 两个 numpy 数组，已过滤到 380-780nm
    """
    data = np.loadtxt(filepath, delimiter=',')
    wavelengths = data[:, 0]
    intensities = data[:, 1]
    mask = (wavelengths >= WL_MIN) & (wavelengths <= WL_MAX)
    return wavelengths[mask], intensities[mask]


def parse_filename(filename):
    """
    从文件名中提取编号和方向
    例如: 'M020（上）.csv' -> ('M020', '上')
    """
    base = os.path.splitext(filename)[0]
    m = re.match(r'(M\d+)[（(](.+?)[）)]', base)
    if m:
        return m.group(1), m.group(2)
    return base, ''


def scan_workspace(workspace_dir):
    """
    扫描工作室目录，返回结构化数据:
    {
        'before': {方向: (编号, 文件路径)},
        'after':  {点位: {方向: (编号, 文件路径)}}
    }
    """
    result = {'before': {}, 'after': {}}

    # 调光前
    before_dir = os.path.join(workspace_dir, '调光前')
    if os.path.isdir(before_dir):
        for f in os.listdir(before_dir):
            if f.lower().endswith('.csv'):
                code, direction = parse_filename(f)
                result['before'][direction] = (code, os.path.join(before_dir, f))

    # 调光后
    after_dir = os.path.join(workspace_dir, '调光后')
    if os.path.isdir(after_dir):
        for point_dir in sorted(os.listdir(after_dir)):
            point_path = os.path.join(after_dir, point_dir)
            if os.path.isdir(point_path):
                point_data = {}
                for f in os.listdir(point_path):
                    if f.lower().endswith('.csv'):
                        code, direction = parse_filename(f)
                        point_data[direction] = (code, os.path.join(point_path, f))
                if point_data:
                    result['after'][point_dir] = point_data

    return result


# ─── 绘图函数 ────────────────────────────────────────────────
def normalize(values):
    """归一化到 [0, 1]"""
    vmin, vmax = values.min(), values.max()
    if vmax > vmin:
        return (values - vmin) / (vmax - vmin)
    return np.zeros_like(values)


def _setup_axes(ax, title):
    """统一设置坐标轴样式"""
    ax.set_xlim(WL_MIN, WL_MAX)
    ax.set_ylim(0, None)
    ax.set_xlabel('波长 (nm)', fontsize=10)
    ax.set_ylabel('相对强度 (归一化)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(range(400, 800, 50))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')


def plot_single_direction_comparison(direction, before_data, after_data,
                                     point_name, save_path=None):
    """
    绘制单个方向的调光前/后对比图（左右子图）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- 调光前 ---
    wl_b, val_b = before_data
    val_b_norm = normalize(val_b)
    plot_spectrum_fill(ax1, wl_b, val_b_norm, alpha=0.35)
    ax1.plot(wl_b, val_b_norm, color='black', linewidth=2)
    _setup_axes(ax1, f'调光前 · {direction}')

    # --- 调光后 ---
    wl_a, val_a = after_data
    val_a_norm = normalize(val_a)
    plot_spectrum_fill(ax2, wl_a, val_a_norm, alpha=0.35)
    ax2.plot(wl_a, val_a_norm, color='black', linewidth=2)
    _setup_axes(ax2, f'调光后({point_name}) · {direction}')

    # 统一 Y 轴上限为 1.1
    ax1.set_ylim(0, 1.1)
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return fig


def plot_overlay_comparison(direction, before_data, after_data,
                            before_code, after_code, point_name, save_path=None):
    """
    绘制调光前/后叠加对比图（同一坐标系）
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    wl_b, val_b = before_data
    wl_a, val_a = after_data
    val_b_norm = normalize(val_b)
    val_a_norm = normalize(val_a)

    # 调光前 - 渐变填充 + 蓝色曲线
    plot_spectrum_fill(ax, wl_b, val_b_norm, alpha=0.2)
    ax.plot(wl_b, val_b_norm, color='#1f77b4', linewidth=2,
            label=f'调光前 {before_code}')

    # 调光后 - 红色曲线
    ax.plot(wl_a, val_a_norm, color='#d62728', linewidth=2, linestyle='--',
            label=f'调光后({point_name}) {after_code}')

    _setup_axes(ax, f'调光前/后光谱对比 · {direction}方向')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return fig


def plot_point_grid(before_all, after_all, point_name, save_path=None):
    """
    为某个点位绘制 6 方向 2x3 网格对比图
    before_all: {方向: (wl, val)}
    after_all:  {方向: (wl, val)}
    """
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    axes_flat = axes.flatten()

    for idx, direction in enumerate(DIRECTION_ORDER):
        ax = axes_flat[idx]
        if direction not in before_all or direction not in after_all:
            ax.set_visible(False)
            continue

        wl_b, val_b = before_all[direction]
        wl_a, val_a = after_all[direction]
        val_b_norm = normalize(val_b)
        val_a_norm = normalize(val_a)

        # 调光前渐变填充（浅色）
        plot_spectrum_fill(ax, wl_b, val_b_norm, alpha=0.15)
        ax.plot(wl_b, val_b_norm, color='#1f77b4', linewidth=1.5,
                label='调光前')
        ax.plot(wl_a, val_a_norm, color='#d62728', linewidth=1.5,
                linestyle='--', label='调光后')

        ax.set_xlim(WL_MIN, WL_MAX)
        ax.set_ylim(0, 1.1)
        ax.set_title(f'{direction}方向', fontsize=11, fontweight='bold')
        ax.set_xlabel('波长 (nm)', fontsize=9)
        ax.set_ylabel('相对强度', fontsize=9)
        ax.set_xticks(range(400, 800, 100))
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')
        ax.legend(fontsize=8, loc='upper right')

    fig.suptitle(f'{point_name} · 各方向调光前/后光谱对比',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return fig


def plot_before_summary(before_data_dict, save_path=None):
    """
    绘制调光前 6 方向总览图（2x3 网格）
    before_data_dict: {方向: (wl, val)}
    """
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    axes_flat = axes.flatten()

    for idx, direction in enumerate(DIRECTION_ORDER):
        ax = axes_flat[idx]
        if direction not in before_data_dict:
            ax.set_visible(False)
            continue

        wl, val = before_data_dict[direction]
        val_norm = normalize(val)
        plot_spectrum_fill(ax, wl, val_norm, alpha=0.35)
        ax.plot(wl, val_norm, color='black', linewidth=1.8)
        ax.set_xlim(WL_MIN, WL_MAX)
        ax.set_ylim(0, 1.1)
        ax.set_title(f'{direction}方向', fontsize=11, fontweight='bold')
        ax.set_xlabel('波长 (nm)', fontsize=9)
        ax.set_ylabel('相对强度', fontsize=9)
        ax.set_xticks(range(400, 800, 100))
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')

    fig.suptitle('调光前 · 各方向光谱分布', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return fig


# ─── 主流程 ──────────────────────────────────────────────────
def main():
    workspace_dir = os.path.normpath(WORKSPACE_DIR)
    output_dir = os.path.normpath(OUTPUT_DIR)

    print(f"工作室目录: {workspace_dir}")
    print(f"输出目录:   {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 扫描目录
    data = scan_workspace(workspace_dir)
    before = data['before']
    after_groups = data['after']

    if not before:
        print("[错误] 未找到调光前数据")
        return
    print(f"调光前: {len(before)} 个方向")
    print(f"调光后: {len(after_groups)} 个点位")

    # ── 1. 调光前总览 ──
    print("\n[1/4] 绘制调光前总览...")
    before_loaded = {}
    for direction in DIRECTION_ORDER:
        if direction in before:
            code, fpath = before[direction]
            before_loaded[direction] = load_csv_spectrum(fpath)
    plot_before_summary(before_loaded,
                        os.path.join(output_dir, '调光前_总览.png'))
    print("  -> 调光前_总览.png")

    # ── 2. 每个点位: 叠加对比图 ──
    print("[2/4] 绘制各点位叠加对比图...")
    for point_name, point_files in sorted(after_groups.items()):
        for direction in DIRECTION_ORDER:
            if direction not in before or direction not in point_files:
                continue
            b_code, b_path = before[direction]
            a_code, a_path = point_files[direction]
            b_data = load_csv_spectrum(b_path)
            a_data = load_csv_spectrum(a_path)

            save_name = f'{point_name}_{direction}方向_对比.png'
            plot_overlay_comparison(
                direction, b_data, a_data,
                b_code, a_code, point_name,
                os.path.join(output_dir, point_name, save_name))
        print(f"  -> {point_name} 完成")

    # ── 3. 每个点位: 网格总览 ──
    print("[3/4] 绘制各点位网格总览...")
    for point_name, point_files in sorted(after_groups.items()):
        after_loaded = {}
        for direction in DIRECTION_ORDER:
            if direction in point_files:
                _, fpath = point_files[direction]
                after_loaded[direction] = load_csv_spectrum(fpath)
        plot_point_grid(
            before_loaded, after_loaded, point_name,
            os.path.join(output_dir, point_name, f'{point_name}_总览.png'))
        print(f"  -> {point_name}_总览.png")

    # ── 4. 每个点位: 左右对比子图 ──
    print("[4/4] 绘制各方向左右对比图...")
    for point_name, point_files in sorted(after_groups.items()):
        for direction in DIRECTION_ORDER:
            if direction not in before or direction not in point_files:
                continue
            b_data = load_csv_spectrum(before[direction][1])
            a_data = load_csv_spectrum(point_files[direction][1])
            save_name = f'{point_name}_{direction}方向_左右对比.png'
            plot_single_direction_comparison(
                direction, b_data, a_data, point_name,
                os.path.join(output_dir, point_name, save_name))
        print(f"  -> {point_name} 完成")

    # 统计
    total = 1  # 调光前总览
    for point_name, point_files in after_groups.items():
        matched = sum(1 for d in DIRECTION_ORDER
                      if d in before and d in point_files)
        total += matched * 2 + 1  # 叠加 + 左右 + 网格
    png_count = sum(1 for _ in glob.glob(os.path.join(output_dir, '**/*.png'),
                                          recursive=True))
    print(f"\n全部完成! 共生成 {png_count} 张图片 -> {output_dir}")


if __name__ == '__main__':
    main()
