"""
心电图原始数据读取工具
使用bioread读取AcqKnowledge软件生成的.acq文件

数据结构：
{
    '1孙宇轩': {
        'Untitled1': {'data': numpy.array, 'samples': int, 'rate': float},
        'Untitled2': {...},
        ...
    },
    '2张戈': {...},
    ...
}
"""

import os
import bioread
import numpy as np
import pickle


# 心电图目录路径
ECG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "心电图for4.2")


def load_acq_file(filepath):
    """
    读取单个.acq文件

    Args:
        filepath: .acq文件完整路径

    Returns:
        dict: 包含data, samples, rate的字典
    """
    try:
        data = bioread.read_file(filepath)
        if len(data.channels) > 0:
            ch = data.channels[0]
            return {
                'data': ch.data,                          # 心电信号数据
                'samples': ch.point_count,               # 采样点数
                'rate': ch.samples_per_second,            # 采样率 (Hz)
                'name': ch.name                           # 通道名称
            }
        return None
    except Exception as e:
        print(f"  读取失败 {filepath}: {e}")
        return None


def load_subject_ecg(subject_folder, base_dir=ECG_DIR):
    """
    读取单个被试者的所有心电图数据

    Args:
        subject_folder: 被试者文件夹名 (如 "1孙宇轩")
        base_dir: 心电图目录

    Returns:
        dict: 该被试者的所有心电图数据
    """
    subject_path = os.path.join(base_dir, subject_folder)
    if not os.path.isdir(subject_path):
        return {}

    ecg_data = {}
    acq_files = sorted([f for f in os.listdir(subject_path)
                        if f.lower().endswith('.acq')])

    print(f"  正在读取 {subject_folder}: {len(acq_files)} 个文件")

    for acq_file in acq_files:
        filepath = os.path.join(subject_path, acq_file)
        file_key = os.path.splitext(acq_file)[0]  # 去掉扩展名作为key
        ecg_data[file_key] = load_acq_file(filepath)

    # 移除读取失败的数据
    ecg_data = {k: v for k, v in ecg_data.items() if v is not None}

    return ecg_data


def load_all_ecg_data(base_dir=ECG_DIR, save_pickle=True):
    """
    读取所有被试者的心电图数据

    Args:
        base_dir: 心电图目录
        save_pickle: 是否保存为pickle文件

    Returns:
        dict: 所有被试者的心电图数据
    """
    print("=" * 60)
    print("心电图原始数据读取")
    print("=" * 60)

    all_ecg = {}

    # 获取所有被试者文件夹
    subject_folders = sorted([d for d in os.listdir(base_dir)
                            if os.path.isdir(os.path.join(base_dir, d))],
                           key=lambda x: (len(x), x))

    print(f"\n找到 {len(subject_folders)} 个被试者文件夹")

    for subject_folder in subject_folders:
        ecg_data = load_subject_ecg(subject_folder, base_dir)
        if ecg_data:
            all_ecg[subject_folder] = ecg_data

    # 统计信息
    print("\n" + "=" * 60)
    print("读取完成统计")
    print("=" * 60)
    print(f"成功读取 {len(all_ecg)} 个被试者数据")

    total_files = sum(len(v) for v in all_ecg.values())
    print(f"总文件数: {total_files}")

    # 保存为pickle文件
    if save_pickle:
        pickle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ecg_data.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(all_ecg, f)
        print(f"\n数据已保存到: {pickle_path}")

    return all_ecg


def get_ecg_info(all_ecg):
    """
    获取心电图数据的基本信息

    Args:
        all_ecg: 所有心电图数据字典

    Returns:
        DataFrame: 信息汇总表
    """
    info_list = []

    for subject, ecg_dict in all_ecg.items():
        for file_key, data in ecg_dict.items():
            info_list.append({
                '被试者': subject,
                '文件': file_key,
                '采样点数': data['samples'],
                '采样率(Hz)': data['rate'],
                '时长(s)': data['samples'] / data['rate'] if data['rate'] > 0 else 0,
                '通道名': data['name']
            })

    return pd.DataFrame(info_list)


# 为了兼容性保留函数名
def load_ecg_data(base_dir=ECG_DIR, save_pickle=True):
    """兼容性别名"""
    return load_all_ecg_data(base_dir, save_pickle)


if __name__ == "__main__":
    import pandas as pd

    # 读取所有数据
    all_ecg = load_all_ecg_data()

    # 显示数据概览
    print("\n" + "=" * 60)
    print("数据概览")
    print("=" * 60)

    info_df = get_ecg_info(all_ecg)
    print(f"\n数据维度: {info_df.shape}")
    print("\n前10条记录:")
    print(info_df.head(10).to_string())

    # 保存信息表
    info_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ecg_info.xlsx")
    info_df.to_excel(info_path, index=False)
    print(f"\n信息表已保存到: {info_path}")