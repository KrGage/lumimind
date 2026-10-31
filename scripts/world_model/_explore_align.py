"""核对 All_Data 各 sheet 的被试编号对齐情况 + 透视成 (被试,组,时间点) 结构。"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "ai_agent" / "Oupu_Lighting" / "data"
ALL = DATA / "All_Data.xlsx"

xl = pd.ExcelFile(ALL)
gcols = [f"G{g}T{t}" for g in range(1, 6) for t in range(1, 4)]

print("############ 各 sheet 的被试编号(被试者)分布 ############")
for sh in xl.sheet_names:
    df = pd.read_excel(ALL, sheet_name=sh)
    if "被试者" not in df.columns:
        print(f"[{sh}] 无 被试者 列, columns={list(df.columns)[:6]}")
        continue
    subs = df["被试者"].dropna().unique()
    subs_sorted = sorted(subs.tolist())
    sav = df["_sav_file"].dropna().unique().tolist() if "_sav_file" in df.columns else []
    print(f"[{sh}] rows={len(df)}  被试者={subs_sorted[:12]}{'...' if len(subs_sorted)>12 else ''} "
          f"(n_unique={len(subs_sorted)}, min={min(subs_sorted)}, max={max(subs_sorted)})")
    print(f"      _sav_file={sav}  有值的组列: "
          f"{ {c: int(df[c].notna().sum()) for c in gcols} }")

# PANAS 特殊：区分 PA / NA
print("\n############ PANAS 明细（区分 PA/NA） ############")
pn = pd.read_excel(ALL, sheet_name="PANAS")
print("  _sav_file 取值:", pn["_sav_file"].astype(str).unique().tolist())
print("  按 _sav_file 分组的被试者范围:")
for tag, sub in pn.groupby(pn["_sav_file"].astype(str)):
    s = sorted(sub["被试者"].dropna().unique().tolist())
    print(f"    '{tag}': n={len(s)}, range=[{min(s) if s else '-'},{max(s) if s else '-'}]")
