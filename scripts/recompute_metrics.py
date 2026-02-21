import pandas as pd
import numpy as np
import json
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, brier_score_loss
)
from geopy.distance import geodesic

df = pd.read_csv("outputs/results/test_set_samples.csv")
y_true = df["y_true"].values
y_score = df["y_score"].values
threshold = 0.1536

y_pred = (y_score >= threshold).astype(int)

roc_auc = roc_auc_score(y_true, y_score)
pr_auc = average_precision_score(y_true, y_score)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
brier = brier_score_loss(y_true, y_score)

errors = df["error_km"].dropna()
spatial_mean = errors.mean()
spatial_median = errors.median()

results = {
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "brier": brier,
    "spatial_error_km_mean": spatial_mean,
    "spatial_error_km_median": spatial_median,
    "threshold_used": threshold
}

with open("outputs/results/test_set_final.json", "w") as f:
    json.dump(results, f, indent=2)

print("Métricas finais:")
for k, v in results.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
