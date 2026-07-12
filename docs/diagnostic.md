# Diagnostic Report — Why Is RI Classification Near Chance?

**Date:** 2026-07-11 · **Status: PRELIMINARY** — all findings below are read-only
observations of existing artifacts; nothing was retrained or tuned for this report.

## Question

The released 3D-CNN scores ROC-AUC ≈ 0.59 / PR-AUC ≈ 0.12 on the test split.
Is the bottleneck (a) data/class imbalance, (b) features, or (c) architecture/training?

## Evidence

### 1. Class balance per split (from `data/normalized/splits.csv` + `valid_events.csv`)

| Split | n | RI positives | RI negatives | Positive rate |
|---|---:|---:|---:|---:|
| train | 655 | 30 | 625 | 4.58% |
| val   | 162 | **3** | 159 | 1.85% |
| test  | 155 | 9 | 146 | 5.81% |
| **total** | **972** | **42** | **930** | **4.32%** |

The validation split contains **three** positive events. Any threshold or
checkpoint selection driven by validation AUC is dominated by noise: reranking
a single positive moves val AUC by tens of points.

### 2. Baseline vs 3D-CNN on the test split (n = 155, 9 positives)

Bootstrap CIs: 10,000 resamples, stratification-free, seed 42
(`average_precision_score` / `roc_auc_score`). Chance level for PR-AUC equals
prevalence (0.058); for ROC-AUC it is 0.500.

| Model | PR-AUC [95% CI] | ROC-AUC [95% CI] |
|---|---|---|
| Tabular logistic regression (36 aggregate features) | 0.140 [0.048, 0.372] | 0.651 [0.426, 0.860] |
| 3D-CNN (physics-guided, released checkpoint) | 0.121 [0.042, 0.290] | 0.590 [0.334, 0.842] |

Both confidence intervals **include chance level** on both metrics. The CNN's
point estimate does not exceed the linear baseline's. With 9 test positives,
neither model can be distinguished from chance, nor from each other.

### 3. Training history (`outputs/results/training_history.json`, 40 epochs)

- Train classification loss falls monotonically 0.259 → 0.051 (the network fits
  the training set).
- Validation classification loss reaches its minimum at epoch 29 (0.125) and
  **worsens afterwards** — classic overfitting with 30 training positives.
- Validation AUC oscillates between 0.77 and 0.94 with no stable trend —
  expected behavior of an AUC computed on 3 positives, and the reason
  checkpoint selection ("best val AUC" = 0.943 @ epoch 27) is unreliable.
- **Provenance caveat:** `train_phys_loss` and `val_phys_loss` are exactly 0.0
  in every epoch — this history predates the physics-loss reconnection
  (see `ERRATA.md`). The released checkpoint was trained with physics terms
  inactive, so these numbers do not measure the physics-guided configuration.

## Verdict

**The dominant bottleneck is SAMPLE SIZE / CLASS IMBALANCE, not architecture.**

Reasoning, following the pre-agreed decision rule:

1. The linear baseline does not separate reliably either (PR-AUC 0.140, CI
   spanning chance). If aggregate features carried a strong signal, a
   well-regularized LR on 36 features would show it. It does not — but with
   9 test positives we also cannot *rule out* a moderate signal (the CI
   reaches 0.37).
2. The CNN does not outperform the baseline (0.121 vs 0.140 point estimates,
   heavily overlapping CIs). There is therefore **no evidence** that the 3D
   architecture extracts more than the aggregate features already provide —
   and no statistical basis for blaming the architecture.
3. With 42 positives overall, 30 for training, 3 for validation, and 9 for
   testing, **every stage of the pipeline is starved**: training overfits,
   validation cannot select models, and the test set cannot resolve
   differences between models.

Secondary (actionable) findings, in order of expected impact:

- **The val split is unusable for model selection** (3 positives). Any future
  run should re-stratify splits by RI label within the storm-level split
  constraint, or use k-fold CV over storms.
- Overfitting after ~epoch 29 suggests earlier stopping criteria tied to val
  *loss* (less noisy than val AUC at these sample sizes).
- Released metrics come from a checkpoint trained with physics losses inactive;
  post-fix retraining has not been benchmarked yet.

## What this report does NOT claim

- It does not claim the features are informative (underpowered to show it).
- It does not claim the architecture is adequate (untestable at this n).
- It does not compare against operational RI baselines (SHIPS-RII); PR-AUC
  values here are not comparable to literature using different event bases.

**Every number above is conditional on n_test = 155 with 9 positives and
should be treated as preliminary.**
