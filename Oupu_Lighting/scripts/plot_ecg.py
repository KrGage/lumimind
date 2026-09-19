"""
心电图波形可视化
使用matplotlib绘制单个.acq文件的心电波动图像
"""

import matplotlib.pyplot as plt
import os
import data_ECG


def plot_single_ecg():
    """绘制单个心电图波形"""
    # 读取第一个被试者的第一个心电图文件
    all_ecg = data_ECG.load_all_ecg_data(save_pickle=False)
    
    # 获取第一个被试者
    subject = list(all_ecg.keys())[0]
    # 获取该被试者的第一个文件
    file_key = list(all_ecg[subject].keys())[0]
    ecg_data = all_ecg[subject][file_key]
    
    # 获取数据
    data = ecg_data['data']
    rate = ecg_data['rate']
    samples = ecg_data['samples']
    
    # 计算时间轴 (秒)
    time = [i / rate for i in range(samples)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time, data, linewidth=0.5, color='darkgreen')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Voltage', fontsize=12)
    ax.set_title(f'{subject} - {file_key}.acq ECG\nRate: {rate} Hz, Duration: {samples/rate:.2f} sec', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, min(10, samples/rate)])  # Only show first 10 seconds
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'ecg_waveform.png'), dpi=150)
    plt.show()
    print(f'\nFigure saved to: ecg_waveform.png')


if __name__ == '__main__':
    plot_single_ecg()