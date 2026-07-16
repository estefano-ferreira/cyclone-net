## CycloneNet – Validation Report

> **⚠️ TWO PROTOCOLS, TWO SECTIONS — DO NOT COMPARE ACROSS THEM.**
> This report contains numbers from two different measurement protocols:
> **(1)** the frozen test set (single read, 2,679 events) and **(2)**
> dev-fold out-of-fold measurement (SID-grouped 3-fold CV, 3 seeds,
> 14,101 events). The test-set PR-AUC 0.251 and the dev-OOF PR-AUC 0.249
> are **not comparable** — different events, different protocol, different
> prevalence. Every table below is labeled with its protocol.

**Reference-model status (author decision, 2026-07-16):** the project
reports **no single reference model**. The CNN has a frozen test-set
number (PR-AUC 0.251 [0.179–0.331]) but was **retired by H9/V2** (see the
dev-fold section). The GBM_SF is the empirical reference **on the dev
folds** (0.249 pooled OOF) and has **never** been evaluated on the frozen
test set — promoting it would require a second test-set read, which is
not justified: the released contribution is the **dataset**, not a model.
Both are reported below with their protocols explicitly separated. The
CNN's test-set metrics stand as historical record of a retired
architecture.

**Dataset:** full 1980–2023 **two-basin archive (East Pacific + North
Atlantic)** — 16,780 valid events / 802 RI positives / 992 storms (578 EP /
414 NA by genesis basin, i.e. the first point of each storm's IBTrACS
record; the extraction bounding box cuts the EP basin west of 140°W). Basin
is a per-point IBTrACS attribute: per-point the valid events split 8,888 EP
/ 7,892 NA, six storms in the archive genuinely cross between the basins,
and one valid storm (Joan/Miriam 1988) has valid events in both. An earlier
version of this line said "North Atlantic sector" — incorrect; the
basin-relabel correction is documented in [ERRATA.md](./ERRATA.md) item 7.
Storm-level splits are
hash-deterministic by SID (leakage-free; adding storms never reassigns
existing ones). Test split: **2,679 events / 115 RI positives / 153 storms**,
never used during development. These numbers are reproducible from the
public repository and dataset; the correction history that led here is
documented in [ERRATA.md](./ERRATA.md).

The operating threshold (**0.0097**) is selected on the validation split
(2,951 events / 142 positives) by `precision_at_recall` at the forensic
recall target of **0.90 on validation** (achieved: 0.9014 at selection —
recorded in `models/checkpoints/best_threshold.json`, committed before the
test-set documentation), then applied unchanged to the test set, where the
achieved recall is **0.852**. The target has been 0.90 throughout the
project's history (`eval_target_recall`, unchanged in git); an earlier
wording here ("≥ 0.85 achieved") conflated the validation target with the
test-set outcome.

---

## PROTOCOL 1 — Frozen test set (retired CNN; historical record)

> **Architecture retired 2026-07-16 by the pre-registered H9/V2 verdict**
> (see Protocol 2). These metrics are the single authorized read of the
> frozen test set and stand as the historical record of that model. They
> are NOT comparable to the dev-fold numbers below.

### 📊 Global Test-Set Metrics

| Metric | Value | Interpretation |
| --- | --- | --- |
| **ROC-AUC** | **0.796** [95% CI 0.753–0.837] | CI entirely above chance (0.5) |
| **PR-AUC** | **0.251** [95% CI 0.179–0.331] | 5.8× the 0.043 prevalence; CI entirely above chance |
| **Recall** | **0.852** | High sensitivity at the forensic operating point |
| Precision | 0.070 | The accepted cost of the recall-first mandate at 4.3% prevalence |
| F1-score | 0.129 | Dominated by the deliberate recall bias |
| Brier score | 0.0372 | Probabilistic calibration (ECE 0.009 quantile / 0.011 fixed-width) |

Confidence intervals: bootstrap, 10,000 resamples, seed 42.

---

### 🎯 Calibration and operating points (analysis of 2026-07-14)

Post-hoc analysis of the released test predictions
(`analysis/calibration_report.py`; artifacts in
`outputs/results/calibration/`). The scores are re-analyzed as saved — the
test set is not re-read.

- **Well calibrated:** ECE 0.0085 (quantile bins) / 0.0111 (fixed-width);
  largest bin gap 0.018; bin fraction-positive is monotone in the score.
  The tiny operating threshold (0.0097) reflects the 4.3% prevalence, not
  miscalibration.
- **Weak resolution (the honest reading):** Murphy decomposition of the
  Brier score — reliability 0.0001 (excellent), **resolution 0.0030**
  against uncertainty 0.0411. The model's probabilities are trustworthy
  but only weakly separate RI from non-RI: statistically real skill,
  operationally modest.
- **Arithmetic note (so the identity closes for any reader):** the three
  components above are computed over quantile BINS; for continuous scores
  the 3-term identity REL − RES + UNC (= 0.03819) differs from the direct
  Brier (0.03718) by the within-bin terms of the generalized decomposition
  (Stephenson et al. 2008): WBV − WBC = 0.00090 − 0.00192 = **−0.00101**.
  The 5-term identity closes to machine precision; all components are in
  `calibration_summary.json`.
- **Forensic trade-off, quantified** (threshold swept on the saved test
  scores):

| Recall target | Precision | Alerts per 100 events |
| --- | --- | --- |
| 0.90 | 0.063 | 62 |
| 0.80 | 0.074 | 46 |
| 0.50 | 0.169 | 13 |

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

## PROTOCOL 2 — Dev-fold measurement (SID-grouped 3-fold CV × 3 seeds; closed verdicts of 2026-07-16)

All numbers in this section are **pooled out-of-fold PR-AUC on the
PL-gated dev set** (14,101 events / 687 RI positives / 839 storms;
StratifiedGroupKFold by SID, k=3, seeds {42, 123, 456}, identical folds
for every model; 15-epoch budget for every CNN cell). **These numbers are
NOT comparable to the test-set numbers above — different protocol.** The
test set was never read by any of these experiments.

### The measurement ladder (cross-seed mean OOF PR-AUC)

| Model (information diet) | Pooled OOF PR-AUC (mean of 3 seeds) |
| --- | --- |
| GBM **S** — storm state only (Vmax, persistence, lat, season, basin) | **0.202** |
| GBM **F** — field aggregates only (mean/std/min/max of the 11 cube channels) | **0.170** |
| CNN (arm A) — full-resolution spatial fields (intensity-blind) | **0.171** |
| GBM **SF** — state + field aggregates (strongest tabular baseline) | **0.249** |
| Logistic regression on SF (linear reference) | 0.203 |

### Closed pre-registered verdicts (each CI read once)

- **H6 — shear/RH feature ablation: NULL.** ΔPR-AUC(B−A) = +0.0185, 95%
  CI **[−0.0070, +0.0431] includes zero**: the added channels give no
  detectable skill at this resolution/regime for this architecture.
  Not a weak positive (pre-registered reading).
- **H9/V1 — validity vs tabular baseline: NEGATIVE.** Δ₁ = PR-AUC(CNN) −
  PR-AUC(GBM_SF) = −0.0781, 95% CI **[−0.1162, −0.0422] < 0**: the
  tabular baseline **beats** the CNN. Pre-registered consequence: the CNN
  is not justified over a classical baseline.
- **H9/V2 — architecture justification: NULL.** Δ₂ = PR-AUC(CNN) −
  PR-AUC(GBM_F) = +0.0005, 95% CI **[−0.0285, +0.0316] includes zero**:
  at a fixed information diet, full 0.25° grids give this CNN nothing
  detectable beyond 44 aggregate scalars. Pre-registered consequence:
  **the architecture is not justified in its current form and was
  retired**; any redesign is a new pre-registered test.
- **H8 — FuelMap physics-loss ablation: CANCELLED** (2026-07-16). Its
  question became undecidable once H9/V2 retired the architecture those
  losses shape; ablating a component of a retired model decides nothing.
  Pre-registration and harness remain in the repo as record.

**Scope guard for V2 (fixed before the read):** a null/negative Δ₂
unjustifies THIS architecture; it does NOT establish that "spatial
structure carries no RI information" — a stronger spatial reader could
extract what this one cannot. The claim licensed is about the model, not
the physics.

### Ex-ante qualifications (all known before the read; none invented after)

1. **Global average pooling:** the CNN aggregates spatially before
   classifying — CNN ≈ GBM_F was expected *of this architecture* and
   licenses no claim about spatial signal in the data.
2. **Intensity-blind:** the CNN receives no Vmax/persistence; GBM_SF
   does. Part of the V1 gap is information diet, not model class
   (S − F ≈ +0.03 shows how far state alone carries).
3. **Basin:** the (then-mislabeled) basin one-hot carried the true
   two-basin partition — the GBM effectively used basin as a predictor;
   the CNN never sees it.
4. **15-epoch budget**, identical for every arm and every CNN cell.

Full protocol and fixed consequences:
`docs/tabular_baseline_preregistration.md` (H9, two co-primary verdicts),
`docs/ablation_preregistration.md` (H6). Verdict artifacts:
`outputs/results/feature_ablation_cnn/aggregate_20260716T120517Z.json`,
`outputs/results/tabular_baseline/compare_20260716T121803Z.json`.

---

### ⚠️ Important Considerations

- **Diagnostic, not predictive:** hindcast evaluation only; no real-time
  forecasting claim, and no comparison against operational baselines
  (SHIPS-RII) executed. Note that SHIPS-RII is fitted **per basin**; a
  direct comparison against this two-basin model is not possible without
  separating the basins. The **pre-registered classical-baseline
  comparison** (H9) was executed and read on 2026-07-16 — see Protocol 2
  above: the tabular baseline beats the CNN on the dev folds.
- **Two-basin heterogeneity:** the dataset mixes two basins (East Pacific +
  North Atlantic) with different RI climatologies (shear, TCHP,
  seasonality); basin is not used as a control in the current experiments,
  and per-basin analysis is under-powered (687 dev positives total). This
  heterogeneity is a **declared limitation, not a controlled variable**
  (see ERRATA.md item 7).
- **Model note:** the CNN receives only spatial fields — current intensity
  and persistence are NOT inputs. The H9 measurement (Protocol 2)
  quantified what that omission costs: V1 Δ = −0.078 against the
  state-aware baseline, with the ex-ante qualification that this is
  information diet as much as model class.
- **Deliberate bias:** recall-first thresholding accepts many false
  positives by design.
- **Spatial attribution is unsupported** (see above) — the classification
  skill does not extend to energy-source localization.
- **Pre-registered campaign CLOSED (2026-07-16):** H6 NULL, H9/V1
  negative, H9/V2 null (architecture retired), H8 cancelled — all
  verdicts and their fixed consequences in Protocol 2 above. The full
  hypothesis ledger — including refuted ones — is
  `docs/hypothesis_registry.md`.

_Last updated: 2026-07-16_
