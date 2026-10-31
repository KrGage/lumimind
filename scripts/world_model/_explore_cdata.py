"""C-data 数据探查：dump 所有 xlsx / sav 文件的结构，搞清楚变量与光照条件。"""
import sys
from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 200)

ROOT = Path(__file__).resolve().parent.parent.parent  # 项目主体
CDATA = ROOT / "ai_agent" / "Oupu_Lighting" / "data" / "C-data"
DATA = ROOT / "ai_agent" / "Oupu_Lighting" / "data"


def dump_xlsx(path: Path, max_rows=8):
    print("\n" + "=" * 90)
    print(f"[XLSX] {path.relative_to(ROOT)}")
    try:
        xl = pd.ExcelFile(path)
        print(f"  sheets: {xl.sheet_names}")
        for sh in xl.sheet_names:
            df = pd.read_excel(path, sheet_name=sh)
            print(f"  -- sheet '{sh}': shape={df.shape}")
            print(f"     columns({len(df.columns)}): {list(df.columns)[:40]}")
            print(df.head(max_rows).to_string())
    except Exception as e:
        print(f"  ERROR: {e}")


def dump_sav(path: Path, max_rows=8):
    print("\n" + "=" * 90)
    print(f"[SAV] {path.relative_to(ROOT)}")
    try:
        import pyreadstat
        df, meta = pyreadstat.read_sav(str(path))
        print(f"  shape={df.shape}")
        print(f"  columns({len(df.columns)}): {list(df.columns)[:40]}")
        print(f"  column_labels: {meta.column_names_to_labels}")
        print(df.head(max_rows).to_string())
    except ImportError:
        print("  pyreadstat not installed yet")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "xlsx"

    xlsx_files = sorted(DATA.rglob("*.xlsx"))
    sav_files = sorted(CDATA.rglob("*.sav"))

    if which in ("xlsx", "all"):
        for f in xlsx_files:
            dump_xlsx(f)
    if which in ("sav", "all"):
        for f in sav_files:
            dump_sav(f)


if __name__ == "__main__":
    main()
