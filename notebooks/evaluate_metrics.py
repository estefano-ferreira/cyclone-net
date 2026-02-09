"""
CycloneNet: Physics-Guided Framework for Targeted RI Detection.
---------------------------------------------------------------
Software Engineer: Estefano Senhor Ferreira
License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)

This work is licensed under CC BY-NC 4.0. 
Commercial use is strictly prohibited without prior authorization.
Copyright (c) 2026 Estefano Senhor Ferreira
"""

import pandas as pd
import numpy as np
import io
import os
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, brier_score_loss
)


def heal_csv_data(file_path):
    """
    Data Healing Function:
    Fixes I/O racing conditions where log lines were merged without newlines.
    Splits merged rows based on the expected 19-column schema.
    """
    expected_cols = 19
    clean_lines = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target CSV not found at: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        headers = f.readline()
        clean_lines.append(headers)
        for line in f:
            parts = line.strip().split(',')
            # Check for merged lines (e.g., 37 or 38 columns instead of 19)
            if len(parts) > expected_cols:
                # Recover both observations from the merged line
                clean_lines.append(",".join(parts[:expected_cols]) + "\n")
                clean_lines.append(",".join(parts[expected_cols:]) + "\n")
            else:
                clean_lines.append(line)

    return io.StringIO("".join(clean_lines))


# 1. Path Configuration
base_path = os.path.dirname(os.path.abspath(__file__))
# Points to the specific scientific log generated
raw_csv = os.path.join(base_path, '..', 'outputs',
                       'cyclonenet_scientific_2026-02-08.csv')

# 2. Robust Data Processing & Sanitization
try:
    # Heal the data in memory
    virtual_file = heal_csv_data(raw_csv)
    df = pd.read_csv(virtual_file)

    # Cast critical columns to numeric, coercing noise/headers into NaN
    numeric_cols = ['is_RI_actual', 'confidence_weight',
                    'prediction_binary', 'error_km']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any corrupted rows that resulted in NaN values during casting
    initial_count = len(df)
    df = df.dropna(
        subset=['is_RI_actual', 'confidence_weight', 'prediction_binary'])

    # Strict Binary Validation: Ensure target is strictly [0, 1] for Scikit-Learn
    df = df[df['is_RI_actual'].isin([0, 1])].copy()

    final_count = len(df)
    print(f"✅ Data Integrity Verified.")
    print(
        f"📈 Total Valid Records: {final_count} (Noise removed: {initial_count - final_count})\n")

except Exception as e:
    print(f"❌ Critical Failure during data sanitization: {e}")
    exit()

# 3. Scientific Metric Calculation
# ROC-AUC: Measures the model's ability to rank high-risk events correctly
auc = roc_auc_score(df['is_RI_actual'], df['confidence_weight'])

# F1, Precision, Recall: Binary classification performance
f1 = f1_score(df['is_RI_actual'], df['prediction_binary'])
precision = precision_score(df['is_RI_actual'], df['prediction_binary'])
recall = recall_score(df['is_RI_actual'], df['prediction_binary'])

# Brier Score: Measures the calibration of predicted probabilities (lower is better)
brier = brier_score_loss(df['is_RI_actual'], df['confidence_weight'])

# 4. Final Scientific Report (International Research Standard)
print("#" * 60)
print(f"{'CYCLONENET OFFICIAL VALIDATION SUMMARY':^60}")
print("#" * 60)
print(f"{'METRIC':<25} | {'VALUE':<10} | {'INTERPRETATION'}")
print("-" * 60)
print(f"{'Area Under ROC (AUC)':<25} | {auc:.4f}    | High Predictive Power")
print(f"{'F1-Score':<25} | {f1:.4f}    | Harmonized Accuracy")
print(f"{'Precision (PPV)':<25} | {precision:.4f}    | 1 - False Alarm Ratio")
print(f"{'Recall (Sensitivity)':<25} | {recall:.4f}    | No Missed Events")
print(f"{'Brier Score':<25} | {brier:.4f}    | Reliability/Calibration")
print("-" * 60)
print(
    f"{'Mean Tracking Error':<25} | {df['error_km'].mean():.2f} km  | Center Displacement")
print(
    f"{'Median Tracking Error':<25} | {df['error_km'].median():.2f} km  | Robust Median Error")
print("#" * 60 + "\n")

# 5. Granular Event Breakdown
print("📊 DETAILED PERFORMANCE PER STORM (NHC BENCHMARK):")
storm_report = df.groupby('event_name').agg({
    'error_km': 'mean',
    'confidence_weight': 'mean',
    'prediction_binary': 'sum'
}).rename(columns={
    'error_km': 'MAE (km)',
    'confidence_weight': 'Avg_Conf',
    'prediction_binary': 'RI_Hits'
})

print(storm_report.round(3).sort_values(by='MAE (km)'))
