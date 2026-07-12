## CycloneNet – Test-Set Validation Report

**Dataset:** full 1980–2023 North Atlantic sector archive — 16,780 valid
events / 802 RI positives / 992 storms. Storm-level splits are
hash-deterministic by SID (leakage-free; adding storms never reassigns
existing ones). Test split: **2,679 events / 115 RI positives / 153 storms**,
never used during development. These numbers are reproducible from the
public repository and dataset; the correction history that led here is
documented in [ERRATA.md](./ERRATA.md).

The operating threshold (**0.0097**) is selected on the validation split
(2,951 events / 142 positives) by `precision_at_recall` at the forensic
recall target (≥ 0.85 achieved), then applied unchanged to the test set.

---

### 📊 Global Test-Set Metrics

| Metric | Value | Interpretation |
| --- | --- | --- |
| **ROC-AUC** | **0.796** [95% CI 0.753–0.837] | CI entirely above chance (0.5) |
| **PR-AUC** | **0.251** [95% CI 0.179–0.331] | 5.8× the 0.043 prevalence; CI entirely above chance |
| **Recall** | **0.852** | High sensitivity at the forensic operating point |
| Precision | 0.070 | The accepted cost of the recall-first mandate at 4.3% prevalence |
| F1-score | 0.129 | Dominated by the deliberate recall bias |
| Brier score | 0.0372 | Probabilistic calibration (ECE 0.011) |

Confidence intervals: bootstrap, 10,000 resamples, seed 42.

**Honest trajectory.** The same architecture went from
indistinguishable-from-chance to demonstrable skill purely through data
scale — the diagnostic verdict ("the bottleneck is sample size") confirmed
by intervention:

| Dataset generation | Test positives | PR-AUC ÷ chance | ROC-AUC [CI] |
| --- | --- | --- | --- |
| 2020–2023 (original) | 9 | 2.1× | 0.590 [0.334–0.842] — spans chance |
| 2020–2023 + SID-fixed IDs | 13 | 3.6× | 0.614 [0.453–0.785] — spans chance |
| **1980–2023 (current)** | **115** | **5.8×** | **0.796 [0.753–0.837] — above chance** |

---

### 🗺️ Spatial attribution (FuelMap): validated NEGATIVE

The model's FuelMap coordinate was validated against audited TCHP peaks
(n = 226 eligible test events, TCHP publicly gridded 2022+): it beats a
random-point null (p = 0.0003) but does **not** localize the storm's energy
source beyond storm position — median 539 km to the TCHP peak vs 561 km for
the naive storm-centre baseline (closer in only 46% of events, p = 0.30).
A dynamic displacement test with a pure-physics-prior control shows the
FuelMap's collapse-toward-centre during RI is arithmetic of the
enthalpy-flux prior, not learned skill. Do not interpret `pred_lat`/
`pred_lon` as energy-source localization.

---

### 🌪️ Per-Event Examples (Test Set)

Per-sample predictions: [`outputs/results/test_predictions.csv`](./outputs/results/test_predictions.csv).
Event identifiers carry the storm ID: `era5_{YYYY_MM_DD_HHMM}_{SID}`
(concurrent storms never collide).

| Event | Storm | y_true | score | Notes |
| --- | --- | --- | --- | --- |
| `era5_2015_08_28_1200_2015238N10255` | JIMENA (2015) | 1 | 0.469 | Highest-confidence true positive |
| `era5_1998_09_19_1200_1998259N10335` | GEORGES (1998) | 0 | 0.437 | High-scoring false positive — RI-like environment without labelled RI; the accepted recall-first trade-off |
| `era5_2016_10_17_0600_2016278N23300` | NICOLE (2016) | 0 | 0.000 | Confident true negative |

---

### 📂 Audit Trail

- `outputs/results/test_predictions.csv` — per-sample predictions.
- `outputs/results/test_metrics.json` — aggregated metrics **with embedded
  dataset provenance** (event/positive counts, git commit, timestamp).
- `outputs/results/training_history.json` — loss/AUC curves.
- `outputs/provenance/window_*.json` — 22 per-window manifests of the
  download→extract→verify→discard dataset build (checksums included).
- `models/checkpoints/` — checkpoint + `dataset_provenance.json`. Training
  is deterministic: the committed checkpoint was reproduced digit-for-digit
  from scratch.

---

### ⚠️ Important Considerations

- **Diagnostic, not predictive:** hindcast evaluation only; no real-time
  forecasting claim, and no comparison against operational baselines
  (SHIPS-RII) yet.
- **Deliberate bias:** recall-first thresholding accepts many false
  positives by design.
- **Spatial attribution is unsupported** (see above) — the classification
  skill does not extend to energy-source localization.

_Last updated: 2026-07-12_
