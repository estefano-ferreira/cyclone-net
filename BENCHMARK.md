## CycloneNet – Test‑Set Validation Report

> ⚠️ **Superseded — pending regeneration.** The metrics in this report were produced by an
> earlier pipeline revision in which the physics‑guided losses were inactive and the
> validation threshold was selected by max‑F1 (not the forensic recall target). The current
> code activates the physics losses by default and uses `precision_at_recall`. All numbers
> below will be regenerated after re‑training; treat them as a historical baseline only.

This document presents the final diagnostic performance of the CycloneNet framework on the **test set**, which comprises storms not seen during training or validation. The metrics below are computed using the threshold that maximises recall (≥90%) on the validation set (**threshold = 0.0666**) and then applied unchanged to the test set. All results are derived from the same pipeline used during development, ensuring full reproducibility.

---

### 📊 Global Test‑Set Metrics

| Metric            | Value     | Interpretation                                           |
| ----------------- | --------- | -------------------------------------------------------- |
| **ROC‑AUC**       | 0.8329    | Good overall discriminative power                        |
| **PR‑AUC**        | 0.3470    | Balanced precision‑recall on the minority class          |
| **Recall**        | **0.905** | **High sensitivity** – captures >90% of actual RI events |
| **Precision**     | 0.187     | Acceptable given the recall target                       |
| **F1‑score**      | 0.310     | Harmonic mean of precision and recall                    |
| **Brier score**   | 0.0741    | Well‑calibrated probabilistic outputs                    |
| **Spatial error** | N/A       | TCHP data not available for this evaluation              |

> **Note:** Spatial error could not be computed because external TCHP (Tropical Cyclone Heat Potential) data were not used in this benchmark. The model does produce predicted coordinates (`pred_lat`, `pred_lon`) which are saved in the predictions CSV, but without an independent ground truth (such as TCHP maxima) we cannot quantify the localisation accuracy. Future releases will integrate TCHP data to validate the model’s geospatial attribution.

---

### 🔍 Comparison with Validation Performance

During training, the best validation epoch achieved recall of 90.5% and PR‑AUC of 0.350, using a threshold of 0.0666. The test‑set results are very close, confirming that the model generalises well and does not suffer from severe overfitting. The threshold remained unchanged at 0.0666, demonstrating that the operating point is stable across different data splits.

---

### 🌪️ Per‑Storm Analysis (Test Set)

A detailed per‑sample breakdown is available in [`outputs/results/test_predictions.csv`](./outputs/results/test_predictions.csv). This file contains, for each event:

- `event_id`: unique identifier (e.g., `era5_2005_08_27_0600`)
- `y_true`: ground truth RI label (0 or 1)
- `y_score`: model probability (after sigmoid)
- `pred_lat`, `pred_lon`: predicted coordinates of the thermodynamic hotspot (when available)

Below we highlight a few illustrative cases from the test set that reflect the model’s forensic behaviour:

| Event                  | y_true | y_score | Notes                                                                                                                                                                                   |
| ---------------------- | ------ | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `era5_2005_08_27_0600` | 1      | 0.951   | True positive with very high confidence (Hurricane Katrina).                                                                                                                            |
| `era5_2018_10_10_1200` | 1      | 0.858   | True positive (Hurricane Michael).                                                                                                                                                      |
| `era5_2012_08_27_1200` | 0      | 0.322   | False positive (Hurricane Isaac) – the model detected thermodynamic patterns similar to RI, but no actual intensification occurred. This exemplifies the intentional safety‑first bias. |
| `era5_2017_09_20_0000` | 0      | 0.033   | True negative with low score, correctly classified.                                                                                                                                     |

These examples illustrate the trade‑off: high recall comes at the cost of false positives in cases where environmental conditions resemble RI precursors.

---

### 📈 Precision‑Recall Curve

The precision‑recall curve on the test set (not shown here) has an area under the curve (PR‑AUC) of **0.347**. At the chosen threshold of 0.0666, recall reaches 90.5% while precision is 18.7%. This operating point is deliberately chosen to prioritise sensitivity, in accordance with the forensic audit mandate.

---

### 📂 Audit Trail

All artefacts for this evaluation are stored in `outputs/results/`:

- `test_predictions.csv` – per‑sample predictions, probabilities, and spatial coordinates.
- `test_metrics.json` – aggregated metrics (including those above).
- `train_history.json` – training and validation loss curves.

The original model checkpoint and training artefacts are in `models/checkpoints/`.

---

### ⚠️ Important Considerations

- **Diagnostic, not predictive:** These results are based on hindcast evaluation. The framework has not been tested for real‑time forecasting.
- **Deliberate bias:** The high recall is achieved by accepting a moderate number of false positives, in line with the safety‑first forensic philosophy.
- **Spatial validation pending:** Localisation accuracy has not yet been quantified due to the absence of external validation data (TCHP). This is a priority for future releases.
- **Reproducibility:** All steps are fully config‑driven and logged; the exact experiment can be replayed using the provided code and configuration.

_Last updated: 2026‑02‑23_
