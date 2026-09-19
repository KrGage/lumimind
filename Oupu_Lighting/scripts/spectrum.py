"""
光谱数据可视化
读取 C-data/光谱对应表.xlsx 中的光谱数据
绘制 380nm-780nm 可见光范围内的光谱曲线图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def wavelength_to_rgb(wavelength):
    """
    将可见光波长(nm)转换为RGB颜色
    基于CIE颜色匹配函数的近似算法
    """
    if wavelength < 380 or wavelength > 780:
        return (0.5, 0.5, 0.5)  # 不可见区域返回灰色
    
    if wavelength < 440:
        # 紫色到蓝色
        r = (440 - wavelength) / (440 - 380)
        g = 0
        b = 1
    elif wavelength < 490:
        # 蓝色到青色
        r = 0
        g = (wavelength - 440) / (490 - 440)
        b = 1
    elif wavelength < 510:
        # 青色到绿色
        r = 0
        g = 1
        b = (510 - wavelength) / (510 - 490)
    elif wavelength < 580:
        # 绿色到黄色
        r = (wavelength - 510) / (580 - 510)
        g = 1
        b = 0
    elif wavelength < 645:
        # 黄色到橙色到红色
        r = 1
        g = (645 - wavelength) / (645 - 580)
        b = 0
    else:
        # 红色
        r = 1
        g = 0
        b = 0
    
    # 强度调整（边缘波长较暗）
    if wavelength < 420:
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif wavelength > 700:
        factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
    else:
        factor = 1.0
    
    return (r * factor, g * factor, b * factor)


def plot_spectrum_fill(ax, wavelengths, values_norm):
    """
    在曲线下方绘制渐变光谱填充（蓝色到红色）
    """
    # 创建波长到颜色的映射
    unique_wls = np.unique(wavelengths)
    colors_list = []
    for wl in unique_wls:
        rgb = wavelength_to_rgb(wl)
        colors_list.append(rgb)
    
    # 使用 fill_between 并添加渐变效果
    # 将渐变分解为多个段
    for i in range(len(unique_wls) - 1):
        wl_start = unique_wls[i]
        wl_end = unique_wls[i + 1]
        rgb = wavelength_to_rgb((wl_start + wl_end) / 2)
        
        # 获取对应波长范围内的数据
        mask = (wavelengths >= wl_start) & (wavelengths <= wl_end)
        wl_range = wavelengths[mask]
        val_range = values_norm[mask]
        
        if len(wl_range) > 0:
            # 绘制填充区域
            ax.fill_between(wl_range, 0, val_range, 
                           color=rgb, alpha=0.4, linewidth=0)
    return ax


def load_spectrum_data(filepath):
    """
    读取Excel中的光谱数据
    返回：字典 {光谱名称: (波长数组, 强度数组)}
    """
    df = pd.read_excel(filepath, header=None)
    
    spectra = {}
    num_spectra = df.shape[1] // 2  # 每2列为一组
    
    for i in range(num_spectra):
        col_wl = i * 2
        col_val = i * 2 + 1
        
        # 从第6行开始（第5行是表头）
        wavelengths = df.iloc[6:, col_wl].astype(float).values
        values = df.iloc[6:, col_val].astype(float).values
        
        # 过滤380-780nm范围
        mask = (wavelengths >= 380) & (wavelengths <= 780)
        wavelengths = wavelengths[mask]
        values = values[mask]
        
        if len(wavelengths) > 0:
            spectra[f'光谱 {i+1}'] = (wavelengths, values)
    
    return spectra


def plot_spectra(filepath, save_path=None):
    """
    绘制光谱数据图
    """
    # 读取数据
    spectra = load_spectrum_data(filepath)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 只绘制第一条光谱数据
    spectra_list = list(spectra.items())
    if len(spectra_list) == 0:
        print("未找到光谱数据")
        return None
    
    name, (wavelengths, values) = spectra_list[0]
    
    # 归一化处理
    if values.max() > values.min():
        values_norm = (values - values.min()) / (values.max() - values.min())
    else:
        values_norm = values
    
    # 绘制渐变填充
    plot_spectrum_fill(ax, wavelengths, values_norm)
    
    # 绘制曲线
    ax.plot(wavelengths, values_norm, color='black', linewidth=2.5)
    
    # 设置坐标轴
    ax.set_xlim(380, 780)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('波长 (nm)', fontsize=12)
    ax.set_ylabel('相对强度 (归一化)', fontsize=12)
    ax.set_title('可见光光谱分布图 (380nm - 780nm)', fontsize=14, fontweight='bold')
    
    ax.set_xticks(range(400, 800, 50))
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"光谱图已保存至: {save_path}")
    
    plt.show()
    return fig


if __name__ == '__main__':
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'C-data', '光谱对应表.xlsx')
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'spectrum.png')
    
    print("正在读取光谱数据...")
    plot_spectra(filepath, save_path)