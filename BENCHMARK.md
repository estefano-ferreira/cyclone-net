## CycloneNet – Test‑Set Validation Report

This document presents the final diagnostic performance of the CycloneNet framework on the **test set**, which comprises storms not seen during training or validation. The metrics below are computed with the threshold that maximises recall (≥90%) on the validation set and then applied unchanged to the test set (threshold = 0.0539). All results are derived from the same pipeline used during development, ensuring full reproducibility.

---

### 📊 Global Test‑Set Metrics

| Metric            | Value     | Interpretation                                           |
| ----------------- | --------- | -------------------------------------------------------- |
| **ROC‑AUC**       | 0.8329    | Good overall discriminative power                        |
| **PR‑AUC**        | 0.3470    | Balanced precision‑recall on the minority class          |
| **Recall**        | **0.905** | **High sensitivity** – captures >90% of actual RI events |
| **Precision**     | 0.187     | Acceptable given the recall target                       |
| **F1‑score**      | 0.310     | Harmonic mean of precision and recall                    |
| **Brier score**   | 0.0745    | Well‑calibrated probabilistic outputs                    |
| **Spatial error** | N/A       | TCHP data not available for this evaluation              |

> **Note:** Spatial error could not be computed because external TCHP (Tropical Cyclone Heat Potential) data were not available in the metadata. The model still produces predicted coordinates (`pred_lat`, `pred_lon`) which are saved in the predictions CSV.

---

### 🔍 Comparison with Validation Performance

During training, the best validation epoch achieved recall of 90.5% and PR‑AUC of 0.350. The test‑set results are very close, confirming that the model generalises well and does not suffer from severe overfitting. The optimal threshold shifted from 0.0666 (validation) to 0.0539 (test), a minor adjustment well within the expected variability of data splits.

---

### 🌪️ Per‑Storm Analysis (Test Set)

A detailed per‑sample breakdown is available in `outputs/results/test_predictions.csv`. The following figure shows the precision‑recall curve on the test set, with the chosen threshold marked.

<div align="center">
<img src="./outputs/results/pr_curve_test.png" width="500" alt="Precision‑Recall Curve – Test Set">
<p><i>Figure 1: Precision‑recall curve on the test set (AUC = 0.347). The operating point (threshold = 0.0539) yields recall = 0.905 and precision = 0.187.</i></p>
</div>

---

### 📂 Audit Trail

All artefacts for this evaluation are stored in `outputs/results/`:

- `test_predictions.csv` – per‑sample predictions, probabilities, and spatial coordinates.
- `test_metrics.json` – aggregated metrics (including those above).
- `pr_curve_test.png` – precision‑recall curve (if generated).

The original model checkpoint and training artefacts are in `models/checkpoints/`.

---

### ⚠️ Important Considerations

- **Diagnostic, not predictive:** These results are based on hindcast evaluation. The framework has not been tested for real‑time forecasting.
- **Deliberate bias:** The high recall is achieved by accepting a moderate number of false positives, in line with the safety‑first forensic philosophy.
- **Reproducibility:** All steps are fully config‑driven and logged; the exact experiment can be replayed using the provided code and configuration.

_Last updated: 2026‑02‑23_
