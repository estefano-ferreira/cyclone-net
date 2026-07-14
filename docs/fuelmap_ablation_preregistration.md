# Pre-registration — FuelMap physics-loss ablation (H8)

**Registered on 2026-07-14, BEFORE any result of this experiment exists.**
This document fixes the hypothesis, the metric, the verdict criterion and
the decision consequences. It does not change after the numbers exist.

**Scheduling constraint (fixed):** this experiment runs ONLY AFTER the
shear/RH feature ablation (H6) completes all 3 seeds and its verdict is
read. It shares the H6 infrastructure and (in the pre-registered design)
reuses H6's arm-A training cells.

## Context (why this test exists)

H1 — "the FuelMap recovers the storm's real energy source" — was
**REFUTED** (TCHP validation n=226, p=0.30 vs the storm-center baseline;
dynamic test negative; prior-control showed prior arithmetic, not learned
structure). Yet the production model still trains with 4 active
FuelMap-related loss terms (`lambda_prior_align=1.0`, `lambda_forward=0.5`,
`lambda_tv=0.001`, `lambda_l1=0.0001`; `lambda_consistency=0.0` already
off). The refutation killed the *interpretive* claim, but it does NOT
answer whether these losses help or hurt the *classification* objective as
regularizers / weak supervision. That is an empirical question — this
ablation answers it instead of assuming either way.

## Hypothesis (H8 in `hypothesis_registry.md`)

Do the FuelMap physics-guided losses improve RI classification (PR-AUC)
relative to an identical model trained without them?

This is a GENUINE (non-confirmatory) question: the literature does not
predict the answer, and the refutation of the FuelMap's semantics (H1)
cuts both ways — the losses could be harmless regularization, useful
inductive bias, or an actively harmful distortion of the loss landscape.

## Arms

- **Arm A — `A_physics_on`:** production behavior. Channels as configured
  (`model.input_channels_names` + ADT), physics lambdas as in
  `config.yaml` (1.0 / 0.5 / 0.001 / 0.0001, consistency 0.0).
- **Arm B — `B_physics_off`:** IDENTICAL architecture and channels; all 5
  physics lambdas set to 0.0 → plain 3D-CNN classifier (the trainer
  docstring documents this equivalence). The FuelMap branch remains in the
  architecture but receives no supervision — the comparison isolates the
  effect of the LOSSES, not of branch capacity (~1.1k parameters, unused
  in B).
- `lambda_ri` / `lambda_dv12` / `lambda_dv24` identical in both arms (the
  multi-task intensity heads are not under test).

## Metric (fixed)

ΔPR-AUC = PR-AUC(arm B: physics off) − PR-AUC(arm A: physics on), per-seed
pooled out-of-fold, mean across seeds, with 95% SID-cluster bootstrap CI
(`--aggregate`, same machinery as H6). PR-AUC is threshold-independent —
no threshold will be chosen.

**Sign convention: Δ > 0 means the physics losses HURT classification;
Δ < 0 means they HELP.**

## Verdict and pre-registered consequences (read the CI once)

- **CI excludes zero, positive (Δ > 0)** → the FuelMap losses HURT RI
  classification. **Action: remove them from the production model** (V3
  trains as a plain 3D-CNN); report as an honest negative for the
  "physics-guided" framing.
- **CI includes zero** → **NULL**: no detectable effect. **Action: remove
  for parsimony.** A component with refuted semantics (H1) AND no
  measurable benefit has no justification to remain; V3 documents both
  facts. Do NOT reinterpret a null as "harmless, so keep it".
- **CI excludes zero, negative (Δ < 0)** → the losses HELP classification.
  **Action: keep them, reframed in V3 as regularization / weak
  supervision** — explicitly NOT as validated physics (the H1 refutation
  stands regardless of this outcome).

## Discipline (anti-rationalization)

- Read the CI once, after all seeds are aggregated. No per-seed verdicts.
- No mining folds/seeds/lambda settings that favor one arm.
- No re-running with different weights hoping for a better result.
- The decision branches above were fixed before any number existed; the
  action taken must be the branch's action, whatever the number is.

## Run parameters

**FIXED on 2026-07-14, before any result:**

- **k = 3 folds, 15 epochs, seeds {42, 123, 456}** — identical to H6.
- **Dev set:** the same PL-gated census-validated dev set as H6 (14,101
  events / 687 positives / 839 SIDs; PL gate PASS of 2026-07-13). The PL
  restriction is inherited deliberately so arm A can be reused from H6 and
  results stay comparable across the two ablations.
- **Arm A reused from H6 (pre-registered design):** arm `A_current` of the
  H6 runs (physics on, production channels) IS this experiment's arm A —
  same dev set, same deterministic folds per seed, same trainer, same
  epochs. Per seed, `prob_A` is taken from the H6 run's saved
  `oof_predictions.csv` (seed 42: run `20260713T232126Z`; seeds 123/456:
  their respective nightly runs). The harness
  (`analysis/fuelmap_ablation_cnn.py`) validates fold-identity against
  H6's saved per-fold `splits.csv` and aborts on any mismatch; if reuse
  fails validation, the fallback is training both arms fresh (6
  cells/seed) — declared here as the contingency, not a free choice.
- **Trainings: 3 cells/seed** (arm B only) → ~5.5 h/seed on CPU
  (~110 min/cell). Phased execution one seed per night, same detached-run
  protocol as H6 (PROJECT_STATE §6).
- Folds: StratifiedGroupKFold grouped by SID, identical across arms
  (validated, not assumed).
- Normalization stats: train-only per (seed, fold, arm), scoped to the run
  dir; global stats untouched.
- Training: real `src.training.trainer.train()`; the test split is never
  read.
- **Verdict:** mean ΔPR-AUC (B−A) across the 3 seeds with 95% SID-cluster
  bootstrap CI via `--aggregate`. Read ONCE through the 3 branches above.
  No verdict before the 3 seeds are aggregated.
