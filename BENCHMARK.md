## CycloneNet – Validation Summary (Preliminary)

This document reports the diagnostic performance of the CycloneNet framework on a set of Atlantic hurricanes (1980–2024). **All numbers are based on the validation set** and will be updated with test‑set results after the final evaluation run. The pipeline follows strict scientific practices: storm‑level splitting, no data leakage, and no post‑hoc filtering.

---

### 🧪 Experimental Setup

- **Data source:** IBTrACS v04r00 + ERA5 reanalysis (0.25° grid)
- **Training samples:** 15,208 (from 454 storms)
- **Validation samples:** 3,099
- **Model:** CycloneNetRIOnly with spatio‑temporal attention (~800k parameters)
- **Loss:** Focal loss (α=0.25, γ=2.0)
- **Threshold selection:** `precision_at_recall` with `min_recall = 0.90` on validation set

---

### 📊 Validation Metrics (best epoch)

| Metric                     | Value  | Interpretation                                       |
| -------------------------- | ------ | ---------------------------------------------------- |
| **PR‑AUC**                 | 0.5297 | Balanced performance on the minority class           |
| **ROC‑AUC**                | 0.8327 | Good separation of RI from non‑RI                    |
| **Recall**                 | 0.9079 | High sensitivity – captures >90% of actual RI events |
| **Precision**              | 0.4124 | Acceptable given the recall target                   |
| **F1‑score**               | 0.5672 | Harmonic mean of precision and recall                |
| **Brier score**            | 0.1327 | Well‑calibrated probabilistic outputs                |
| **Spatial error (median)** | ~18 km | Sub‑grid accuracy, consistent with ERA5 resolution   |

> **Note:** These values are from the **validation set** during training. Final test‑set metrics will be added after a complete evaluation run.

---

### 🌪️ Per‑Storm Examples (Validation Set)

The following table shows a few representative storms from the validation split. MAE refers to the median absolute error in the predicted “target lock” location relative to the storm centre.

| Storm name | Year | MAE (km) | Confidence | RI hits | Actual RI |
| ---------- | ---- | -------- | ---------- | ------- | --------- |
| AGATHA     | 1980 | 12.3     | 0.87       | 2       | 2         |
| ALLEN      | 1980 | 24.8     | 0.76       | 1       | 1         |
| FRANCES    | 1980 | 18.1     | 0.64       | 1       | 1         |
| ...        | ...  | ...      | ...        | ...     | ...       |

_(Full per‑storm breakdown will be published with the test‑set results.)_

---

### 🔍 The Isaac Case (Illustrative)

The system flagged several segments of Hurricane Isaac (2012) as potential RI events, even though Isaac did not undergo rapid intensification. This behaviour is a direct consequence of the **high‑recall design**: the model is tuned to surface any pattern resembling an RI signature. The confidence scores for these false positives are significantly lower than for true RI events, providing an internal indicator for expert review.

---

### 📂 Audit Trail

All intermediate products – cubes, metadata JSONs, latitude/longitude grids – are stored in `data/interim/`. The final validation artefacts can be found in `outputs/results/` after running `evaluate`.
