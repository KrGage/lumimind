"""
填充用户长期记忆数据，确保 AI 问答可以测试到三种记忆类型：
  1. physio_emotion - 生理情绪记录（已有 3 条，补充更多近期数据）
  2. light_preference - 光环境偏好记录
  3. qa_digest - 历史问答要点记录
"""
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai_agent"))

UID = "local_user"

def seed():
    from agent_framework import long_term_memory as ltm

    # ── 1. 补充更多近期生理情绪记录（用 DEAP 数据）──
    import pickle
    import numpy as np
    DEAP_FILE = ROOT / "DEAP" / "data_preprocessed_python" / "s01.dat"
    with open(DEAP_FILE, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    labels, signals = data["labels"], data["data"]
    valences = labels[:, 0]
    FS = 128

    def band_power(signal, lo, hi):
        seg = 4 * FS
        n_seg = len(signal) // seg
        powers = []
        win = np.hanning(seg)
        for i in range(n_seg):
            chunk = signal[i*seg:(i+1)*seg] * win
            spec = np.abs(np.fft.rfft(chunk)) ** 2
            freqs = np.fft.rfftfreq(seg, 1/FS)
            powers.append(spec[(freqs >= lo) & (freqs < hi)].mean())
        return float(np.mean(powers))

    def extract_feats(trial_data):
        eeg = trial_data[:32]
        ab = []
        for ch in eeg:
            a = band_power(ch, 8, 13)
            b = band_power(ch, 13, 30)
            if b > 0: ab.append(a/b)
        gsr = float(np.mean(trial_data[36]))
        bvp = trial_data[38] - np.mean(trial_data[38])
        spec = np.abs(np.fft.rfft(bvp * np.hanning(len(bvp))))
        freqs = np.fft.rfftfreq(len(bvp), 1/FS)
        band = spec[(freqs >= 0.7) & (freqs <= 3.5)]
        peak_hz = freqs[(freqs >= 0.7) & (freqs <= 3.5)][np.argmax(band)]
        return {
            "eeg_alpha_beta_ratio": round(float(np.mean(ab)), 3),
            "gsr_mean_uS": round(gsr, 4),
            "heart_rate_bpm": round(float(peak_hz * 60), 1),
        }

    def valence_to_label(v):
        if v < 4.5: return "negative"
        if v > 6.5: return "positive"
        return "neutral"

    # 补充 5 条近 2 天的记录（今天是 10月7日）
    now = datetime.now(timezone.utc)
    extra_trials = [5, 12, 20, 28, 35]  # 选几个不同情绪状态的 trial
    print("== 补充近期生理情绪记录 ==")
    for i, tidx in enumerate(extra_trials):
        if tidx >= len(labels): continue
        v, a = valences[tidx], labels[tidx, 1]
        feats = extract_feats(signals[tidx])
        label = valence_to_label(v)
        # 时间分布在近 2 天
        ts = now - timedelta(hours=36 - i*8)
        ltm.record_physio_emotion(
            user_id=UID,
            emotion_label=label,
            emotion_score=round((v-1)/8, 2),
            confidence=0.82 + i*0.01,
            physio={
                "arousal": round(float(a), 2),
                **feats,
            },
            session_id="auto_seed",
            timestamp=ts,
        )
        print(f"  {ts:%m-%d %H:%M} {label} (V={v:.1f}, A={a:.1f}) HR={feats['heart_rate_bpm']}")

    # ── 2. 光环境偏好记录 ──
    print("\n== 写入光环境偏好记录 ==")
    light_records = [
        (now - timedelta(days=3, hours=2), "focus", "下午写论文时需要高色温照明"),
        (now - timedelta(days=2, hours=5), "relax", "晚上阅读，偏好暖色调低照度"),
        (now - timedelta(days=1, hours=1), "sleep_prep", "睡前 1 小时切换到极低照度暖光"),
    ]
    for ts, mode, note in light_records:
        ltm.record_light_preference(
            user_id=UID,
            light_mode=mode,
            operator_note=note,
            session_id="auto_seed",
            timestamp=ts,
        )
        print(f"  {ts:%m-%d %H:%M} {mode}: {note}")

    # ── 3. 历史问答要点记录 ──
    print("\n== 写入历史问答要点 ==")
    qa_records = [
        ("光照对睡眠有什么影响？",
         "高色温(>4000K)光照在夜间会抑制褪黑素分泌，延迟入睡时间。建议睡前2小时切换到低色温(<3000K)、低照度(<100lux)的暖光环境，有助于维持昼夜节律。研究表明，即使是中等照度(200lux)的蓝光成分也会影响褪黑素水平。"),
        ("什么样的光环境适合专注工作？",
         "冷白光(5000-6500K)配合中高照度(500-750lux)被证明能提升警觉性和注意力。动态光照方案（从早晨的5000K逐渐过渡到下午的4000K）可以在维持专注力的同时减少视觉疲劳。建议每90分钟进行一次5分钟的暖光休息。"),
        ("我最近压力比较大，有什么光疗建议？",
         "基于您的生理数据（GSR偏高、HRV偏低），建议采用渐进式光疗方案：早晨使用5000K/500lux冷白光30分钟提升皮质醇觉醒反应，下午切换至3500K/300lux中性光，晚间使用2700K/100lux暖光辅助放松。配合深呼吸练习效果更佳。"),
    ]
    for q, a in qa_records:
        ltm.record_qa_digest(UID, q, a, session_id="auto_seed")
        print(f"  Q: {q[:40]}...")
        print(f"  A: {a[:60]}...")

    # ── 统计 ──
    print("\n== 记忆数据统计 ==")
    memories = ltm.list_user_memories(UID)
    by_type = {}
    for m in memories:
        mt = m["metadata"].get("mem_type", "?")
        by_type[mt] = by_type.get(mt, 0) + 1
    for mt, cnt in by_type.items():
        print(f"  {mt}: {cnt} records")
    print(f"  Total: {len(memories)} records")
    print("\nDone! Memory data ready for testing.")


if __name__ == "__main__":
    seed()
