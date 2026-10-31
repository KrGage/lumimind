"""Build a local, reproducible C-data PANAS/HRV observation catalog.

This catalog is deliberately separate from the live user's memory vector store.
PANAS provides PA/NA scores, not directly measured valence/arousal labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
from uuid import uuid4

import pyreadstat

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dataset" / "C-data"
OUTPUT = ROOT / "ai_agent" / "models" / "cdata_panas_records.json"
MEASURES = {
    "pa": "PANAS/PA.sav",
    "na": "PANAS/NA.sav",
    "heart_rate": "HR/HR.sav",
    "hf_hrv": "HRV/HF.sav",
    "lf_hf_balance": "HRV/交感迷走平衡.sav",
    "sympathetic": "HRV/交感神经.sav",
}
PAPER_SOURCE = "论文\\周莉论文-光照的情绪调节功能研究与验证-V1.00227-标记.pdf"
PAPER_CONTEXT_CHUNKS = [95, 98, 99]


def read_measure(rel: str) -> dict[tuple[int, int, int], float]:
    frame, _ = pyreadstat.read_sav(str(SOURCE / rel))
    out = {}
    for _, row in frame.dropna(subset=["被试者"]).iterrows():
        subject = int(row["被试者"])
        for group in range(1, 6):
            for time in range(1, 4):
                col = f"G{group}T{time}"
                if col not in frame.columns:
                    continue
                value = row[col]
                if value is not None and math.isfinite(float(value)):
                    out[(subject, group, time)] = float(value)
    return out


def build_records() -> dict:
    data = {key: read_measure(rel) for key, rel in MEASURES.items()}
    rag_file = ROOT / "ai_agent" / "models" / "vector_store.json"
    rag = json.loads(rag_file.read_text(encoding="utf-8"))
    expected_chunks = {f"{PAPER_SOURCE}#chunk{i}" for i in PAPER_CONTEXT_CHUNKS}
    if not expected_chunks.issubset(set(rag["ids"])):
        raise ValueError("The referenced dissertation chunks are absent from the local RAG index")
    keys = sorted(set(data["pa"]) & set(data["na"]))
    pa_mid = median(data["pa"][key] for key in keys)
    na_mid = median(data["na"][key] for key in keys)
    records = []
    for subject, group, time in keys:
        key = (subject, group, time)
        column = f"G{group}T{time}"
        pa, na = data["pa"][key], data["na"][key]
        high_pa, high_na = pa >= pa_mid, na >= na_mid
        category = ("positive_dominant" if high_pa and not high_na else
                    "negative_dominant" if high_na and not high_pa else
                    "mixed_high" if high_pa else "low_affect")
        records.append({
            "id": f"cdata_s{subject:02d}_g{group}_t{time}",
            "subject_id": f"C{subject:02d}", "group": f"G{group}", "timepoint": f"T{time}",
            "pa": pa, "na": na, "emotion_category": category,
            "valence_proxy_0_1": round((pa - na + 40) / 80, 4),
            "activation_proxy_0_1": round((pa + na - 20) / 80, 4),
            "heart_rate": data["heart_rate"].get(key),
            "hf_hrv": data["hf_hrv"].get(key),
            "lf_hf_balance": data["lf_hf_balance"].get(key),
            "sympathetic": data["sympathetic"].get(key),
            "source_cells": {
                measure: {"file": "dataset/C-data/" + rel, "subject_column": "被试者",
                          "subject_value": subject, "value_column": column}
                for measure, rel in MEASURES.items() if key in data[measure]
            },
            "related_literature": {
                "source": PAPER_SOURCE,
                "chunk_ids": [f"{PAPER_SOURCE}#chunk{i}" for i in PAPER_CONTEXT_CHUNKS],
                "relation": "实验方法与 PANAS/HRV 分析的文献背景；单条观测值以 .sav 单元格为准",
            },
        })
    return {
        "schema_version": 1,
        "source_root": "dataset/C-data",
        "source_sha256": {key: hashlib.sha256((SOURCE / rel).read_bytes()).hexdigest()
                          for key, rel in MEASURES.items()},
        "literature_context": {"source": PAPER_SOURCE, "chunk_ids": sorted(expected_chunks)},
        "method": {
            "pa_na": "PANAS PA and NA are observed 10–50 scale totals.",
            "valence_proxy": "(PA - NA + 40) / 80; derived proxy, not a measured V score.",
            "activation_proxy": "(PA + NA - 20) / 80; exploratory activation proxy, not a measured arousal score.",
            "category": f"PA/NA median split for display only; PA median={pa_mid:g}, NA median={na_mid:g}.",
            "missing": "Missing HRV measurements remain null; HF is absent for G4/G5.",
        },
        "count": len(records), "records": records,
    }


def main() -> None:
    catalog = build_records()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_name(OUTPUT.name + "." + uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, OUTPUT)
    finally:
        temp.unlink(missing_ok=True)
    print(f"Saved {catalog['count']} C-data observations to {OUTPUT}")


if __name__ == "__main__":
    main()
