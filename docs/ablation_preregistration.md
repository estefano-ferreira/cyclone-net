# Pre-registration — Feature ablation (shear_850_200_mps + rh_mid)

> **Provenance note (2026-07-14):** faithful English translation of the
> Portuguese original frozen at commit `eaa8ae8` on 2026-07-13, BEFORE any
> training result existed. No substantive changes; the original wording is
> preserved in git history.

**Registered on 2026-07-13, BEFORE any training result.** This document
fixes the metric, the verdict criterion, and the reading discipline. It
does not change after the numbers exist.

## Metric (fixed)

ΔPR-AUC = PR-AUC(arm B: 9ch + shear_850_200_mps + rh_mid) − PR-AUC(arm A:
current 9ch), out-of-fold mean, with 95% CI via SID-cluster bootstrap.
PR-AUC is threshold-independent — no threshold will be chosen.

## Verdict (read the CI, not the sign of the point estimate)

- **CI excludes zero, positive** → shear/rh ADD skill (quantified
  increment). Report the delta and the CI. No overselling: "they add X,
  CI [a, b]" — not "crucial features".
- **CI includes zero** → **NULL**: indistinguishable from zero. Report as
  "the core predictors add no detectable skill at this resolution/regime" —
  reinforcing that the bottleneck is data, not features. Do NOT reinterpret
  as a weak positive.
- **CI excludes zero, negative** → investigate (do not report raw; may be
  an artifact).

## Discipline (anti-rationalization)

- Read the CI, once. Do not move the goalposts after seeing the number.
- No mining folds/seeds/thresholds that favor one arm.
- No re-running with different parameters hoping for a better result. One
  well-powered run, one verdict, accepted.
- Report the honest number with the CI, whatever it is — expected positive
  or informative null. Both are valid results.

## Success criterion for the experiment

A CLEAR verdict: a CI tight enough to fall into one of the branches above
without ambiguity. The only outcome to avoid is inconclusiveness from lack
of power — hence seeds ≥ 3 (captures initialization variability).

## Run parameters

**FIXED on 2026-07-13 ~17:30, before any training result:**

- **k = 3 folds, 15 epochs per training.**
- **Seeds = {42, 123, 456}**, PHASED execution: one seed per night
  (42 on 2026-07-13; 123 and 456 on the following nights). 6 training
  cells per seed (3 folds × 2 arms); 18 in total.
- **Device: CPU** (CUDA unavailable on this machine). Measured cost in the
  2026-07-13 calibration: ~5.61 min/epoch → ~84.5 min/cell → ~8.5 h/seed.
- **ADT: faithful to production** — ADT channel present in both arms
  (10/12-input-channel model), raw values where covered (1,257 events),
  zeros where not; symmetric between arms A and B, identical to the current
  production model behavior (10-channel checkpoint, stats without
  adt_mean).
- **Verdict**: mean ΔPR-AUC (B−A) across the 3 seeds with 95% SID-cluster
  bootstrap CI, computed via `--aggregate` over the 3 saved
  `oof_predictions.csv`. Read ONCE through the 3 branches above.
  **No verdict before the 3 seeds are aggregated** — per-seed results are
  intermediate and will not be interpreted in isolation.

- Set: census-validated dev set (14,101 events, 687 positives, 839 SIDs;
  full PL coverage, gate PASS of 2026-07-13).
- Folds: StratifiedGroupKFold grouped by SID, identical across arms.
- Normalization stats: train-only per (seed, fold, arm), scoped to the run
  dir; global stats untouched.
- Training: real src.training.trainer.train(); the test split is never
  read.
