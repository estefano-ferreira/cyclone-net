"""
CycloneNet: Physics-Guided Framework for Targeted RI Detection.
---------------------------------------------------------------
Software Engineer: Estefano Senhor Ferreira
License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)

This work is licensed under CC BY-NC 4.0. 
Commercial use is strictly prohibited without prior authorization.
Copyright (c) 2026 Estefano Senhor Ferreira
"""

import os
import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, brier_score_loss, accuracy_score
)

# --- 1. BOOTSTRAP: Environment Setup ---
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.utils.config import PATHS
except ModuleNotFoundError:
    print(f"Error: Could not find 'src' folder in: {PROJECT_ROOT}")
    sys.exit(1)


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
            if len(parts) > expected_cols:
                # Recover multiple observations from a single corrupted line
                clean_lines.append(",".join(parts[:expected_cols]) + "\n")
                clean_lines.append(",".join(parts[expected_cols:]) + "\n")
            else:
                clean_lines.append(line)

    return io.StringIO("".join(clean_lines))


# --- 2. Path Configuration ---
prediction_path = PATHS['output_predictions']
raw_csv = os.path.join(prediction_path, 'cyclonenet_scientific.csv')
report_txt = os.path.join(prediction_path, 'validation_report.txt')

# --- 3. Data Loading & Sanitization ---
try:
    virtual_file = heal_csv_data(raw_csv)
    df = pd.read_csv(virtual_file)

    # Cast metrics to numeric, handling potential noise
    numeric_cols = ['is_RI_actual', 'confidence_weight',
                    'prediction_binary', 'error_km']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    initial_count = len(df)
    df = df.dropna(
        subset=['is_RI_actual', 'confidence_weight', 'prediction_binary'])
    df = df[df['is_RI_actual'].isin([0, 1])].copy()

    final_count = len(df)
    print(f"✅ Data Integrity Verified. Records: {final_count}")

except Exception as e:
    print(f"❌ Critical Failure during sanitization: {e}")
    sys.exit(1)

# --- 4. Metric Calculations ---
auc = roc_auc_score(df['is_RI_actual'], df['confidence_weight'])
f1 = f1_score(df['is_RI_actual'], df['prediction_binary'])
prec = precision_score(df['is_RI_actual'], df['prediction_binary'])
rec = recall_score(df['is_RI_actual'], df['prediction_binary'])
acc = accuracy_score(df['is_RI_actual'], df['prediction_binary'])
brier = brier_score_loss(df['is_RI_actual'], df['confidence_weight'])

# --- 5. Report Generation ---
report_content = []
report_content.append("#" * 65)
report_content.append(f"{'CYCLONENET OFFICIAL SCIENTIFIC REPORT':^65}")
report_content.append("#" * 65)
report_content.append(f"{'METRIC':<25} | {'VALUE':<10} | {'INTERPRETATION'}")
report_content.append("-" * 65)
report_content.append(
    f"{'ROC-AUC Score':<25} | {auc:.4f}     | Discriminative Power")
report_content.append(
    f"{'Brier Score':<25} | {brier:.4f}     | Calibration Quality")
report_content.append(
    f"{'F1-Score':<25} | {f1:.4f}     | Harmonic Performance")
report_content.append(
    f"{'Precision (PPV)':<25} | {prec:.4f}     | False Alarm Resistance")
report_content.append(
    f"{'Recall (Sensitivity)':<25} | {rec:.4f}     | Event Detection Rate")
report_content.append(
    f"{'Overall Accuracy':<25} | {acc:.4f}     | Global Success Rate")
report_content.append("-" * 65)
report_content.append(
    f"{'Mean Tracking Error':<25} | {df['error_km'].mean():.2f} km  | Center Displacement")
report_content.append(
    f"{'Median Tracking Error':<25} | {df['error_km'].median():.2f} km  | Robust Displacement")
report_content.append("#" * 65 + "\n")

# Event Breakdown
report_content.append("📊 STORM-LEVEL PERFORMANCE (NHC BENCHMARK):")
storm_report = df.groupby('event_name').agg({
    'error_km': 'mean',
    'confidence_weight': 'mean',
    'prediction_binary': 'sum',
    'is_RI_actual': 'sum'
}).rename(columns={
    'error_km': 'MAE(km)',
    'confidence_weight': 'AvgConf',
    'prediction_binary': 'Hits',
    'is_RI_actual': 'Actual_RI'
})

report_content.append(storm_report.round(
    3).sort_values(by='MAE(km)').to_string())

# --- 6. Execution & File Output ---
final_output = "\n".join(report_content)
print(final_output)

with open(report_txt, 'w', encoding='utf-8') as f:
    f.write(final_output)

print(f"\n💾 Scientific report saved to: {report_txt}")

# 1. Calculate the Confusion Matrix
tn, fp, fn, tp = confusion_matrix(
    df['is_RI_actual'], df['prediction_binary']).ravel()
cm = [[tn, fp], [fn, tp]]

# 2. Setup the Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No RI', 'RI (Predicted)'],
            yticklabels=['No RI', 'RI (Actual)'])

plt.title(f'CycloneNet Confusion Matrix\n(AUC: {auc:.4f} | F1: {f1:.4f})')
plt.ylabel('Ground Truth (NHC)')
plt.xlabel('Model Prediction')

# 3. Save the visualization
chart_path = os.path.join(prediction_path, 'confusion_matrix.png')
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
print(f"📈 Confusion Matrix chart saved to: {chart_path}")
plt.show()
