"""C-data 深入探查：All_Data 全量 + sav 变量标签 + 光谱/定标条件映射。"""
from pathlib import Path

import pandas as pd
import pyreadstat

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 240)
pd.set_option("display.max_rows", 300)

ROOT = Path(__file__).resolve().parent.parent.parent
CDATA = ROOT / "ai_agent" / "Oupu_Lighting" / "data" / "C-data"
DATA = ROOT / "ai_agent" / "Oupu_Lighting" / "data"

print("\n############ 1) All_Data.xlsx 各 sheet 的 _sav_file 取值分布 ############")
xl = pd.ExcelFile(DATA / "All_Data.xlsx")
for sh in xl.sheet_names:
    df = pd.read_excel(DATA / "All_Data.xlsx", sheet_name=sh)
    if "_sav_file" in df.columns:
        vc = df["_sav_file"].value_counts(dropna=False).to_dict()
        print(f"[{sh}] rows={len(df)}  _sav_file 分布: {vc}")
        # 每个 _sav_file 对应哪些 ID
        print(f"     ID 范围: {sorted(df['ID'].dropna().unique().tolist())}")
    else:
        print(f"[{sh}] rows={len(df)}  (无 _sav_file 列)")

print("\n############ 2) 各 .sav 文件的变量标签 / 值标签 ############")
for sav in sorted(CDATA.rglob("*.sav")):
    try:
        df, meta = pyreadstat.read_sav(str(sav))
        print("\n" + "-" * 80)
        print(f"[SAV] {sav.relative_to(CDATA)}  shape={df.shape}")
        print(f"  columns: {list(df.columns)}")
        lbl = {k: v for k, v in meta.column_names_to_labels.items() if v}
        if lbl:
            print(f"  labels: {lbl}")
        if meta.variable_value_labels:
            print(f"  value_labels: {meta.variable_value_labels}")
        print(df.head(4).to_string())
    except Exception as e:
        print(f"[SAV] {sav.name} ERROR: {type(e).__name__}: {e}")

print("\n############ 3) 光谱对应表：6 个光照条件的光度/黑视素量 ############")
sp = pd.read_excel(CDATA / "光谱对应表.xlsx", header=None)
# 前 5 行是 E/lx, CCT, melanopic EDI, melanopic irradiance, spectrum 头
print(sp.iloc[0:6, :].to_string())

print("\n############ 4) 定标数据-有EDI和相对光谱：表头结构 ############")
cal = pd.ExcelFile(CDATA / "定标数据-有EDI和相对光谱.xlsx")
print(f"  sheets: {cal.sheet_names}")
for sh in cal.sheet_names[:3]:
    dfc = pd.read_excel(CDATA / "定标数据-有EDI和相对光谱.xlsx", sheet_name=sh, header=None, nrows=8)
    print(f"  -- sheet '{sh}' 前8行 x 前10列:")
    print(dfc.iloc[:, :10].to_string())
