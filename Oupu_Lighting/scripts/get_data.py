import os
import pandas as pd
from pandas.compat.pyarrow import pa
import pyreadstat

C_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "C-data")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "output.xlsx")
ALL_DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "All_Data.xlsx")
ALL_MOOD = ["high_press", "high_mood", "normal_mood", "low_mood"]

def read_sav(filepath):
    df, meta = pyreadstat.read_sav(filepath)
    return df


def savs_to_sheet(filepath, fname):
    key = os.path.splitext(fname)[0]
    df = read_sav(filepath)
    df.insert(0, "_sav_file", key)
    if "ID" not in df.columns:
        df.insert(1, "ID", range(1, len(df) + 1))
    return df


with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    for folder in sorted(os.listdir(C_DATA_DIR)):
        folder_path = os.path.join(C_DATA_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        sav_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".sav")])
        if not sav_files:
            continue

        if len(sav_files) == 1:
            fpath = os.path.join(folder_path, sav_files[0])
            df = savs_to_sheet(fpath, sav_files[0])
            sheet_name = folder[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            dfs = []
            for sf in sav_files:
                fpath = os.path.join(folder_path, sf)
                dfs.append(savs_to_sheet(fpath, sf))

            start_rows = []
            total = 0
            for df in dfs:
                start_rows.append(total)
                total += df.shape[0] + 2

            sheet_name = folder[:31]
            for i, df in enumerate(dfs):
                df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_rows[i])

print(f"Done. Output: {OUTPUT_FILE}")


# ============================================
# 情绪状态分类
# ============================================

def classify_mood_states(input_file=ALL_DATA_FILE, output_file=None):
    """
    根据PA和NA值将被试者分类到4种情绪状态
    
    分类规则:
    - high_press: PA >= 中位 且 NA >= 中位
    - high_mood: PA >= 中位 且 NA < 中位
    - normal_mood: PA < 中位 且 NA < 中位
    - low_mood: PA < 中位 且 NA >= 中位
    """
    # 读取PANAS表
    df = pd.read_excel(input_file, sheet_name='PANAS')
    
    # 分离NA和PA数据
    df_na = df[df['_sav_file'].astype(str).str.upper().str.contains('NA', na=False)].copy()
    df_pa = df[df['_sav_file'].astype(str).str.upper().str.contains('PA', na=False)].copy()
    
    # G1-G5各时间点的列
    g_cols = [f'G{g}T{t}' for g in range(1, 6) for t in range(1, 4)]
    
    # 计算NA和PA的总体中位数
    na_all = pd.concat([df_na[col].dropna() for col in g_cols if col in df_na.columns])
    pa_all = pd.concat([df_pa[col].dropna() for col in g_cols if col in df_pa.columns])
    na_median = na_all.median()
    na_max = na_all.max()
    pa_median = pa_all.median()
    pa_max = pa_all.max()
    
    print("=" * 60)
    print("情绪状态分类分析")
    print("=" * 60)
    print(f"NA 中位数: {na_median:.2f}")
    print(f"PA 中位数: {pa_median:.2f}")
    print()
    
    # 过滤被试者1-10
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    df_na_filtered = df_na[df_na['被试者'].isin(subjects)].copy()
    df_pa_filtered = df_pa[df_pa['被试者'].isin(subjects)].copy()
    
    # 存储结果
    results = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    # 存储每个数据点的距离（情绪激烈程度）
    distances = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: []}
    distance_max_0 = int(((pa_max-pa_median) ** 2 + (na_max-na_median) ** 2) ** 0.5)
    distance_max_1 = int(((pa_max-pa_median) ** 2 + (0-na_median) ** 2) ** 0.5)
    distance_max_2 = int(((0-pa_median) ** 2 + (0-na_median) ** 2) ** 0.5)
    distance_max_3 = int(((0-pa_median) ** 2 + (na_max-na_median) ** 2) ** 0.5)
    
    # 对每个被试者每个时间点进行分类
    for _, row_na in df_na_filtered.iterrows():
        subject_id = int(row_na['被试者'])
        row_pa = df_pa_filtered[df_pa_filtered['被试者'] == subject_id]
        
        for col in g_cols:
            if row_pa.shape[0] > 0:
                pa_val = row_pa[col].values[0]
            else:
                continue
            na_val = row_na[col]
            
            if pd.isna(pa_val) or pd.isna(na_val):
                continue
            
            # 计算与原点的欧几里得距离（情绪激烈程度）
            # 距离越大表示情绪越强烈

            
            # 根据分类规则确定情绪状态
            if pa_val >= pa_median and na_val >= na_median:
                pa_val = pa_val - pa_median
                na_val = na_val - na_median
                distance = (pa_val ** 2 + na_val ** 2) ** 0.5 / distance_max_0
                mood_state = 0
            elif pa_val >= pa_median and na_val < na_median:
                pa_val = pa_val - pa_median
                na_val = na_val - na_median
                distance = (pa_val ** 2 + na_val ** 2) ** 0.5 / distance_max_1
                mood_state = 1
            elif pa_val < pa_median and na_val < na_median:
                pa_val = pa_val - pa_median
                na_val = na_val - na_median
                distance = (pa_val ** 2 + na_val ** 2) ** 0.5 / distance_max_2
                mood_state = 2
            else:
                pa_val = pa_val - pa_median
                na_val = na_val - na_median
                distance = (pa_val ** 2 + na_val ** 2) ** 0.5 / distance_max_3
                mood_state = 3
            distances[subject_id].append(distance)
            results[subject_id].append(mood_state)
    
    
    return results, distances


if __name__ == "__main__":
    all_person_moods, all_person_distances = classify_mood_states()
    
    print("\n情绪激烈程度（与原点距离）:")
    print("=" * 60)
    for subject_id in sorted(all_person_distances.keys()):
        dist_list = all_person_distances[subject_id]
        if dist_list:
            print(f"被试者 {subject_id}: 平均距离={sum(dist_list)/len(dist_list):.2f}, "
                  f"范围=[{min(dist_list):.2f}, {max(dist_list):.2f}]")
        else:
            print(f"被试者 {subject_id}: 无数据")