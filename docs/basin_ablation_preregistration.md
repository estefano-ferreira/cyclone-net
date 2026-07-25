# Pre-registration — Basin contribution to the state-only rung (H10)

**Registered on 2026-07-24, BEFORE any result of this experiment exists.**
This document fixes the arms, the metric, the CI procedure, the minimum
detectable effect, the decision branches and the reading discipline. It
does not change after the numbers exist.

## Context (why this test exists)

ERRATA item 7 / paper ex-ante qualification 3 (§6.5 after the 2026-07-24
MDE-subsection insertion) disclose that the basin
one-hot arrived mislabeled (`basin_EP` / `basin_` with an empty-string
suffix, a `keep_default_na` parsing bug) and that the two mislabeled
columns carried the true two-basin partition at 99.9% event-level
correspondence. The GBM therefore effectively used basin membership as a
predictor; the CNN never received basin input. The measurement ladder's
`GBM_S` rung ("storm state only", pooled OOF PR-AUC 0.202) is confounded
by basin, and the magnitude of that confound is unquantified. This test
converts the disclosed limitation into a quantified one. The purpose is
to **quantify, not to exonerate**: basin is a genuine RI predictor (East
Pacific and North Atlantic have distinct RI climatologies; SHIPS-RII is
fitted separately per basin), so a non-zero contribution is physically
plausible and would not be a defect. Either outcome is publishable; both
numbers go in the paper.

> **AMENDMENT 1 (2026-07-25, BEFORE any paired statistic existed):** the
> original design reused the published S-arm OOF (run `20260715T123221Z`,
> 14,101 dev events) as the with-basin arm. That run predates the
> 2026-07-16 v2 label correction; the corrected pipeline EXCLUDES the 13
> dev events whose v2 label is NULL (`null_no_partner`; 11 train + 2 val
> of the 19 valid→NULL events), so the current dev set is 14,088 events /
> 687 positives / 839 storms and the published dev set is no longer
> reproducible — restoring it would resurrect the corrected NULL-handling
> bug. Pairing published-vs-new on the 14,088-event intersection would
> confound the basin intervention with a 13-event training-set difference,
> defeating the design's "only the basin columns change" requirement.
> The with-basin arm is therefore ALSO a new run of the identical harness
> (no `--drop-basin`, current pipeline, v2 labels), and the paired delta
> compares the two new runs on the identical 14,088-event dev set with
> identical folds. The published OOF stays untouched as historical record;
> the ladder's published S rung (0.202 on 14,101/v1) is reported as such.
> Timeline evidence that no result was read before this amendment: the
> read harness (`analysis/basin_ablation_h10.py`) ABORTED at the
> event-alignment assertion (14,101 vs 14,088) before computing any delta
> or interval; the without-basin run's `summary.json` carries per-arm
> absolute PR-AUCs written by the training harness (as any run does), but
> the pre-registered paired statistic and its CI have never been computed.
> The declared proxy MDE (0.048, from the published S−SF replicate SD)
> stands as the design MDE; the realized MDE supersedes it per the
> degenerate-outcomes section if the proxy assumption fails.

## Hypothesis (H10 in `hypothesis_registry.md`)

How much of the `GBM_S` rung's pooled OOF PR-AUC (0.202) is attributable
to the basin one-hot columns?

## Design

- **Dev set:** the same PL-gated census-validated dev set as H6/H9
  (14,101 events / 687 positives / 839 SIDs). Test split never read.
- **Folds/seeds:** identical to the published H9 run —
  `StratifiedGroupKFold` by SID via `build_folds`, k=3, seeds
  {42, 123, 456}. `build_folds` is deterministic in (events, seed), so
  the folds are identical to run `20260715T123221Z` by construction.
- **Arm WITH basin (no retraining):** the published S-arm OOF
  predictions (`prob_gbm_state_only`) from run
  `outputs/results/tabular_baseline/20260715T123221Z/`, reused as-is.
- **Arm WITHOUT basin (the only new computation):** the identical
  harness (`analysis/tabular_baseline_kfold.py`) run with
  `--drop-basin`, which removes exactly the columns prefixed `basin_`
  from the feature table before the factorial split. Verified against
  the feature cache on 2026-07-24: the cache contains exactly two such
  columns, `basin_EP` and `basin_` (the mislabeled pair); nothing else
  matches the prefix. Default behaviour of the harness is unchanged
  (`--drop-basin` defaults to False); the published run remains
  reproducible.
- **Model (fixed, no tuning):** `HistGradientBoostingClassifier`
  (sklearn defaults, `random_state=seed`) — identical to the published
  arm.
- **Anti-leakage:** same PL gate, same anti-leakage assertion on
  `dv12/dv24` names, same `keep_default_na=False, na_values=[""]`
  reader discipline (basin code "NA" is North Atlantic, not missing).

## Metric and CI (fixed)

Paired delta, per seed on pooled OOF predictions, mean across the 3
seeds:

Δ_basin = PR-AUC(S with basin) − PR-AUC(S without basin)

95% CI by SID-cluster bootstrap, 10,000 replicates, ONE shared storm
resampling per replicate applied to both arms; storms drawn k times
appear k times (collapsing duplicates narrows the interval and
manufactures significance); `boot_seed=42`; the same
`paired_cross_seed_bootstrap` implementation used by the published H9
comparison.

## Minimum detectable effect (declared BEFORE the run)

Required field (registry rule of 2026-07-24; this registration is its
first use). The SE of Δ_basin cannot be computed before the run (it
needs the without-basin arm), so the design MDE is declared from a
**pre-specified conservative proxy**: the bootstrap replicate SD of the
paired delta S − SF from the PUBLISHED OOF predictions, under the
identical shared-resampling cluster bootstrap. The S-with-basin and
S-without-basin arms will be **more** correlated than S and SF are, so
the proxy overstates the SE and the true design is at least this
sensitive.

- Proxy replicate SE: 0.0172 PR-AUC (10,000/10,000 valid replicates,
  `boot_seed=42`; MDE report in `outputs/results/power_analysis/`,
  computed 2026-07-24 with the published H6/H9 CIs reproduced exactly
  as the fidelity gate)
- **Design MDE @ power 0.80 (α=0.05, two-sided): 0.048 PR-AUC**
- Design MDE @ power 0.90: 0.056 PR-AUC

The realized MDE (replicate SD of the actual Δ_basin bootstrap) is
computed and reported alongside the result, whatever it is.

## Decision branches (fixed in advance)

- **ci_low > 0** → basin contributes measurably to the S rung.
  Consequence: the measurement-ladder table reports the S rung BOTH
  ways (with and without basin), and the paper text states that "storm
  state only" includes basin membership.
- **ci_high < 0** → unexpected direction; report as-is, do not
  rationalize.
- **CI contains zero** → basin contribution is not detectable at this
  design's resolution. Consequence: the S rung stands as published,
  with the MDE stated so the null has scope.

## Degenerate outcomes (pre-declared, closing the branch gap)

The three decision branches above assume a non-degenerate bootstrap
distribution. Two degenerate outcomes are possible and their readings are
fixed now, before any number exists:

1. **Point-mass distribution (SE = 0).** If the with-basin and
   without-basin arms produce event-wise identical OOF predictions (the
   GBM never splits on the basin columns), every replicate delta is
   exactly 0: the CI is [0, 0] and the realized MDE is 0. This outcome is
   read as **"zero measurable contribution by construction"** — exact
   equality, NOT the "not detectable at this design's resolution" branch,
   which would misattribute a definite result to lack of power. Before
   applying this reading, prediction identity must be verified directly
   (column-wise comparison of the OOF CSVs), not inferred from the CI.
   More generally, if the replicate SD is smaller than 10% of the proxy
   SE (0.0172), the harness reports it and the result records that the
   realized precision far exceeds the declared design — reported as-is.

2. **Proxy failure (realized SE > declared proxy SE).** The declared MDE
   rests on the assumption that the ablation arms are more correlated
   than S and SF are. If the realized replicate SE exceeds 0.0172, that
   assumption FAILED: the result JSON and the registry entry record the
   failure explicitly, the realized MDE supersedes the declared design
   MDE, and the verdict branch is read against the realized MDE — no
   re-run, no re-thresholding, no rationalization of the discrepancy.

In both cases the interval is still read exactly once; the degenerate
readings above are classifications fixed ex ante, not licenses to
re-interpret.

## Discipline (anti-rationalization)

- The interval is read EXACTLY ONCE, after all 3 seeds complete. No
  re-runs, no re-thresholding, no partial-seed verdicts.
- This test re-reads NO published verdict: H6 NULL, H9/V1 NEGATIVE,
  H9/V2 NULL, H8 CANCELLED all stand untouched.
- The frozen test split (2,679 events) is not read, touched, or used.
- No feature mining: the intervention is exactly "remove the two
  `basin_*` columns", nothing else.
- Whatever the outcome, it is reported with its CI and MDE in the paper.

## Cost

Minutes of CPU: 3 seeds × 3 folds × 1 GBM arm on the cached feature
table (no cube reads, no CNN, no GPU). No interaction with any frozen
artifact.
