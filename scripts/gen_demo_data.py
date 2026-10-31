"""
从真实数据集预提取演示数据 → fontend/demo_data.js
===================================================
- ECG：受试者 1 孙宇轩，BIOPAC AcqKnowledge (.acq) Lead II 心电图
    - 平静段：Untitled37 @ 55s  HR ≈ 87 bpm
    - 紧张段：Untitled6  @ 56s  HR ≈ 111 bpm
- EEG：DEAP s01，通道 0（Fp1）
    - 平静段：trial#20（valence=9.0）
    - 低落段：trial#28（valence=1.36）
- 频段功率与心率均从真实数据计算
"""
import json
import pickle
from pathlib import Path

import bioread
import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "ai_agent" / "fontend" / "demo_data.js"
ECG_DIR = ROOT / "dataset" / "心电图" / "1孙宇轩"
FS_OUT = 50
SEG_SECONDS = 8

# ═══════════════ 1. 真实 ECG（BIOPAC .acq）═══════════════
ECG_CALM   = ("Untitled37.acq", 55)   # (文件名, 起始秒)
ECG_STRESS = ("Untitled6.acq",  56)


def extract_ecg(filename: str, offset_s: int):
    """从 .acq 文件中提取 8s ECG 片段，降采样到 50Hz 并归一化。"""
    raw = bioread.read(str(ECG_DIR / filename))
    ch = raw.channels[0]
    fs = int(ch.samples_per_second)          # 1000 Hz
    sig = np.asarray(ch.data, dtype=np.float64)

    # 取 8s 片段
    start = offset_s * fs
    seg = sig[start : start + SEG_SECONDS * fs]

    # R-peak 检测 → 心率
    peaks, _ = find_peaks(seg, height=np.percentile(seg, 70), distance=int(fs * 0.4))
    if len(peaks) >= 4:
        intervals = np.diff(peaks) / fs
        hr = round(float(60 / np.median(intervals)), 1)
    else:
        hr = 72.0

    # 降采样 1000→50 Hz
    n_out = SEG_SECONDS * FS_OUT
    idx = np.linspace(0, len(seg) - 1, n_out).astype(int)
    out = seg[idx]
    wave = np.round((out - out.mean()) / (out.std() + 1e-9), 3).tolist()
    return wave, hr


print("[1/3] 提取真实 ECG ...")
ecg_c, hr_c = extract_ecg(*ECG_CALM)
ecg_s, hr_s = extract_ecg(*ECG_STRESS)
print(f"  calm  : {ECG_CALM[0]} @ {ECG_CALM[1]}s  HR={hr_c} bpm")
print(f"  stress: {ECG_STRESS[0]} @ {ECG_STRESS[1]}s  HR={hr_s} bpm")

# ═══════════════ 2. EEG + 频段功率（DEAP s01）═══════════════
FS_DEAP = 128

print("[2/3] 提取 DEAP EEG + 频段功率 ...")
with open(ROOT / "DEAP" / "data_preprocessed_python" / "s01.dat", "rb") as f:
    d = pickle.load(f, encoding="latin1")

labels, data = d["labels"], d["data"]
calm_idx  = int(np.argmax(labels[:, 0]))   # valence 最高 → 平静
stress_idx = int(np.argmin(labels[:, 0]))   # valence 最低 → 低落


def extract_eeg(trial_idx, offset_seconds=30):
    """取 DEAP 指定 trial 的 EEG（Fp1）片段与频段功率。"""
    seg = slice(offset_seconds * FS_DEAP, (offset_seconds + SEG_SECONDS) * FS_DEAP)

    eeg = data[trial_idx, 0]                # Fp1
    eeg_seg = eeg[seg]
    n_out = SEG_SECONDS * FS_OUT
    idx = np.linspace(0, len(eeg_seg) - 1, n_out).astype(int)
    wave = np.round((eeg_seg[idx] - eeg_seg[idx].mean()) / (eeg_seg[idx].std() + 1e-9), 3).tolist()

    # 频段功率（32 通道平均）
    n_seg = SEG_SECONDS * FS_DEAP
    powers = {"theta": [], "alpha": [], "beta": []}
    for ch_i in range(32):
        sig = data[trial_idx, ch_i, seg]
        spec = np.abs(np.fft.rfft(sig * np.hanning(n_seg))) ** 2
        freqs = np.fft.rfftfreq(n_seg, 1 / FS_DEAP)
        for name, lo, hi in [("theta", 4, 8), ("alpha", 8, 13), ("beta", 13, 30)]:
            powers[name].append(spec[(freqs >= lo) & (freqs < hi)].mean())
    bands = {k: round(float(np.mean(v)), 3) for k, v in powers.items()}
    return wave, bands


eeg_c, bands_c = extract_eeg(calm_idx)
eeg_s, bands_s = extract_eeg(stress_idx)
print(f"  calm  : trial#{calm_idx}  valence={labels[calm_idx,0]:.2f}")
print(f"  stress: trial#{stress_idx}  valence={labels[stress_idx,0]:.2f}")

# ═══════════════ 3. 写出 demo_data.js ═══════════════
print("[3/3] 生成 demo_data.js ...")
payload = {
    "source": "ECG: BIOPAC 1孙宇轩 / EEG: DEAP s01",
    "trials": {"calm": calm_idx, "stress": stress_idx},
    "valence": {"calm": round(float(labels[calm_idx, 0]), 2),
                "stress": round(float(labels[stress_idx, 0]), 2)},
    "fs": FS_OUT,
    "ecg": {"calm": ecg_c, "stress": ecg_s},        # 真实 ECG Lead II
    "eeg": {"calm": eeg_c, "stress": eeg_s},
    "bands": {"calm": bands_c, "stress": bands_s},
    "hr":  {"calm": hr_c, "stress": hr_s},
}

OUT.write_text(
    "window.DEMO_DATA = " + json.dumps(payload, ensure_ascii=False) + ";\n",
    encoding="utf-8",
)
print(f"bands:  calm={bands_c}  stress={bands_s}")
print(f"hr:     calm={hr_c}  stress={hr_s}")
print(f"OK -> {OUT}")
