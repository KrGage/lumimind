"""
情绪分类模型测试脚本
直接测试RandomForestClassifier模型
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from get_data import classify_mood_states

# 文件路径
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "All_Data.xlsx")

# 情绪分类名称
MOOD_NAMES = ['high_press', 'high_mood', 'normal_mood', 'low_mood']
G_COLS = [f'G{g}T{t}' for g in range(1, 6) for t in range(1, 4)]


def load_panas_data(data_file, subjects=None):
    """加载PANAS表的NA和PA数据"""
    df = pd.read_excel(data_file, sheet_name='PANAS')
    
    df_na = df[df['_sav_file'].astype(str).str.upper().str.contains('NA', na=False)].copy()
    df_pa = df[df['_sav_file'].astype(str).str.upper().str.contains('PA', na=False)].copy()
    
    if subjects is None:
        subjects = list(range(1, 11))
    df_na = df_na[df_na['被试者'].isin(subjects)].copy()
    df_pa = df_pa[df_pa['被试者'].isin(subjects)].copy()
    
    return df_na, df_pa


def calculate_medians(df_na, df_pa):
    """计算NA和PA的中位数"""
    na_all = pd.concat([df_na[col].dropna() for col in G_COLS])
    pa_all = pd.concat([df_pa[col].dropna() for col in G_COLS])
    return na_all.median(), pa_all.median()

def prepare_dataset():
    """准备分类数据集"""
    # 加载HR和HRV数据
    df_hr = pd.read_excel(DATA_FILE, sheet_name='HR')
    df_hrv = pd.read_excel(DATA_FILE, sheet_name='HRV')
    
    subjects = list(range(1, 11))
    df_hr = df_hr[df_hr['被试者'].isin(subjects)].copy()
    df_hrv = df_hrv[df_hrv['被试者'].isin(subjects)].copy()
    
    
    X_list= []
    
    for _, row_hr in df_hr.iterrows():
        subject_id = int(row_hr['被试者'])
        hr_features = [row_hr[col] for col in G_COLS]
        
        row_hrv = df_hrv[df_hrv['被试者'] == subject_id]
        if row_hrv.shape[0] == 0:
            continue
        hrv_features = [row_hrv[col].values[0] for col in G_COLS]
        hf_features = [row_hrv[col].values[1] for col in G_COLS]
        jiaogan_features = [row_hrv[col].values[2] for col in G_COLS]
        mizou_features = [row_hrv[col].values[3] for col in G_COLS]
        features = []
        for i in range(len(hrv_features)):
            temp = []
            temp.append(hrv_features[i])
            temp.append(hf_features[i])
            temp.append(jiaogan_features[i])
            temp.append(mizou_features[i])
            temp.append(hr_features[i])
            features.append(temp)
        X_list.extend(features)
    
    return np.array(X_list)


def main():
    print("=" * 60)
    print("情绪分类模型 (RandomForestClassifier)")
    print("=" * 60)
    
    # 准备数据
    X= prepare_dataset()
    y, _ = classify_mood_states(DATA_FILE)
    y_new = []
    for key, value in y.items():
        y_new.extend(value)
    y = np.array(y_new)
    print(f"\n数据集: {X.shape[0]}样本, {X.shape[1]}特征")
    print(f"类别分布: {np.bincount(y)}")
    print(f"类别说明: 0=high_press, 1=high_mood, 2=normal_mood, 3=low_mood")
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n训练集: {len(y_train)}样本")
    print(f"测试集: {len(y_test)}样本")
    
    # 训练
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 评估
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'=' * 60}")
    print(f"测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"{'=' * 60}")
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=MOOD_NAMES))
    
    # 特征重要性
    print("\n特征重要性 (Top 10):")
    feature_names = [f'HR_{c}' for c in G_COLS] + [f'HRV_{c}' for c in G_COLS]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")


if __name__ == "__main__":
    main()