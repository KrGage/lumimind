"""
将心电图(.acq)和脑电图(.edf)数据导出为JSON，供前端滚动波形展示使用。
输出:
  - fontend/data/ecg_wave.json   (心电数据)
  - fontend/data/eeg_wave.json   (脑电数据)
"""

import os, json, math
import numpy as np
import bioread
import pyedflib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ECG_DIR = os.path.join(BASE_DIR, "..", "..", "dataset", "心电图", "3耿文俊")
EDF_FILE = os.path.join(BASE_DIR, "..", "..", "dataset",
                        "Flex_BM01AS_380102_20250418124850_9802812e92df2710.edf")
OUT_DIR = os.path.join(BASE_DIR, "..", "fontend", "data")

# 前端展示目标采样率 (每秒钟保留多少个点)
DISPLAY_RATE = 100  # 100 points/sec → 很流畅


def downsample(data, orig_rate, target_rate):
    """将信号从 orig_rate 降采样到 target_rate"""
    if orig_rate <= target_rate:
        return data, orig_rate
    step = orig_rate / target_rate
    n = int(len(data) / step)
    indices = (np.arange(n) * step).astype(int)
    indices = np.clip(indices, 0, len(data) - 1)
    return data[indices], target_rate


def export_ecg():
    """读取3耿文俊的前10个.acq文件，拼接并降采样后导出"""
    acq_files = sorted(
        [f for f in os.listdir(ECG_DIR) if f.lower().endswith(".acq")],
        key=lambda x: int("".join(filter(str.isdigit, x)) or 0),
    )

    # 选取前10个文件拼接 (~30min)，太多会很大
    selected = acq_files[:10]
    print(f"[ECG] 读取 {len(selected)} 个文件 ...")

    all_data = []
    orig_rate = None
    for fname in selected:
        fp = os.path.join(ECG_DIR, fname)
        d = bioread.read_file(fp)
        if d.channels:
            ch = d.channels[0]
            all_data.append(ch.data.astype(np.float64))
            orig_rate = ch.samples_per_second
            print(f"  {fname}: {ch.point_count} samples @ {ch.samples_per_second} Hz")

    if not all_data:
        print("[ECG] 无数据!")
        return

    raw = np.concatenate(all_data)
    print(f"[ECG] 拼接后总采样点: {len(raw)}, 时长: {len(raw)/orig_rate:.1f}s")

    ds, ds_rate = downsample(raw, orig_rate, DISPLAY_RATE)
    # 归一化到 [-1, 1] 方便前端绘制
    vmin, vmax = float(ds.min()), float(ds.max())
    vrange = max(vmax - vmin, 1e-9)
    norm = ((ds - vmin) / vrange * 2 - 1).tolist()
    # 四舍五入减小文件体积
    norm = [round(v, 4) for v in norm]

    out = {
        "label": "ECG (心电)",
        "subject": "3耿文俊",
        "unit": "mV (归一化)",
        "sampleRate": ds_rate,
        "originalRate": orig_rate,
        "totalSamples": len(norm),
        "durationSec": round(len(norm) / ds_rate, 1),
        "data": norm,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "ecg_wave.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[ECG] 已导出 → {out_path}  ({os.path.getsize(out_path)/1024/1024:.1f} MB)")


def export_eeg():
    """读取EDF文件的 EEG FP1-FP2 通道，降采样后导出"""
    print(f"[EEG] 读取 {os.path.basename(EDF_FILE)} ...")
    reader = pyedflib.EdfReader(EDF_FILE)

    labels = reader.getSignalLabels()
    # 找到 EEG FP1-FP2 通道
    ch_idx = None
    for i, lbl in enumerate(labels):
        if "EEG" in lbl.upper():
            ch_idx = i
            break
    if ch_idx is None:
        ch_idx = 0  # fallback

    orig_rate = reader.getSampleFrequency(ch_idx)
    raw = reader.readSignal(ch_idx).astype(np.float64)
    reader.close()
    print(f"[EEG] 通道: {labels[ch_idx]}, {len(raw)} samples @ {orig_rate} Hz, "
          f"时长: {len(raw)/orig_rate:.1f}s")

    ds, ds_rate = downsample(raw, orig_rate, DISPLAY_RATE)
    vmin, vmax = float(ds.min()), float(ds.max())
    vrange = max(vmax - vmin, 1e-9)
    norm = ((ds - vmin) / vrange * 2 - 1).tolist()
    norm = [round(v, 4) for v in norm]

    out = {
        "label": "EEG (脑电)",
        "channel": labels[ch_idx],
        "unit": "μV (归一化)",
        "sampleRate": ds_rate,
        "originalRate": orig_rate,
        "totalSamples": len(norm),
        "durationSec": round(len(norm) / ds_rate, 1),
        "data": norm,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "eeg_wave.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[EEG] 已导出 → {out_path}  ({os.path.getsize(out_path)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    export_ecg()
    export_eeg()
    print("\n✅ 数据导出完成！")
