# Pre-registration — Tabular baseline vs CNN (H9)

**Registered on 2026-07-14, BEFORE any result of this experiment exists.**
This document fixes the features, the models, the metric, the verdict
criterion and the reading discipline. It does not change after the numbers
exist.

> **AMENDMENT (2026-07-14, same day, still before any training result):**
> the original single-arm design (one GBM on state+field-mean scalars vs
> the CNN) CONFOUNDS information content with model class: the two arms
> differ simultaneously in what they see (scalars vs spatial fields) and
> in what they are (GBM vs CNN) — and the "scalar" arm already contained
> field-derived means, so neither label was even accurate. The design is
> amended to a FACTORIAL tabular baseline with three arms (S / F / SF,
> below). The primary verdict comparison (CNN vs strongest tabular
> baseline = SF) and its branches are unchanged; the S and F arms are
> descriptive decomposition, pre-declared here. No training result existed
> when this amendment was made (the feature cache had been built; features
> reveal no outcome).

> **THIRD AMENDMENT (2026-07-15, implementation fix, still before any
> result):** the first `--execute` crashed at the very first
> LogisticRegression fit (`ValueError: Input X contains NaN`) — before any
> model finished training, so no outcome was observed. Cause: `cube_*`
> features carry NaN by design (paired with `*_missing` flags); GBM
> handles NaN natively, LogReg does not. Fix: `SimpleImputer(median)`
> prepended to the LogReg pipeline, fitted inside the fold pipeline
> (train-fold statistics only — no leakage). GBM arms (the primary
> models) are untouched.

## Context (why this test exists)

The project has never compared the 3D-CNN against a strong classical
baseline on the SAME data and splits. BENCHMARK.md records this gap
explicitly. Without it, the CNN's PR-AUC (0.251 on the frozen test set)
cannot be attributed to spatial information — it could be reachable by
SHIPS-like scalar predictors alone. This is the single largest remaining
validity gap, and it is also the first step of the project's candidate
scientific product: measuring how much RI predictability exists in coarse
surface data at all.

## Hypothesis (H9 in `hypothesis_registry.md`)

Does the CNN add PR-AUC skill beyond a gradient-boosting model on
SHIPS-like scalar predictors, on identical SID-grouped folds?

## Design

- **Dev set:** the same PL-gated census-validated dev set as H6/H8
  (14,101 events / 687 positives / 839 SIDs). Test split never read.
- **Folds/seeds:** the same fold recipe and seeds as H6
  (StratifiedGroupKFold by SID via `build_folds`, k=3, seeds
  {42, 123, 456}) — fold-identical to the CNN OOF predictions being
  compared against.
- **Tabular models (fixed, NO hyperparameter search):**
  `HistGradientBoostingClassifier` (sklearn defaults, random_state=seed)
  trained separately on each of the three factorial feature sets;
  `StandardScaler + LogisticRegression(class_weight="balanced",
  max_iter=2000)` on SF only as the linear reference (reported, not part
  of the verdict).
- **Factorial feature sets (fixed; t0 and past only):**
  - **S — state_only:** Vmax t0 (`wind_kt`), `pressure_mb`, persistence
    `dv_past_12h`/`dv_past_24h` (derived from the same SID's earlier
    events; missing → 0 + missing-indicator), latitude (raw + absolute),
    day-of-year (sin/cos), basin (one-hot). No field information.
  - **F — fields_only:** spatial-temporal mean/std/min/max of the 11 cube
    channels (the 9 production channels + shear_850_200_mps + rh_mid),
    with per-channel missing indicators. No state information — the
    tabular counterpart of the CNN's information diet.
  - **SF — state_plus_fields:** union of S and F. The PRIMARY baseline.
  **Anti-leakage rule:** metadata `dv12_kt`/`dv24_kt` are FUTURE targets
  and are forbidden as features (asserted in code).
- **Decomposition readings (descriptive, pre-declared, no verdict value):**
  SF−S isolates the value of field information within a fixed model class;
  S alone shows how far persistence/state carries — the quantity the
  intensity-blind CNN cannot see. (CNN−F was initially listed here; it is
  CO-PRIMARY with fixed consequences since the second amendment — see
  "Metrics and verdicts" below.)
- **CNN side of the comparison:** arm `A_current` OOF predictions
  (`prob_A`) from the H6 runs — the physics-on, production-channel CNN,
  fold-identical by construction (seed 42: run `20260713T232126Z`;
  seeds 123/456: their nightly runs).
- **Harness:** `analysis/tabular_baseline_kfold.py` (new file; frozen
  training code untouched).

## Metrics and verdicts (TWO co-primary comparisons, each CI read once)

> **Second amendment (2026-07-14, still before any result — author's
> requirement):** CNN−F was initially declared descriptive. That was
> wrong: it drives a decision (does the architecture justify itself?),
> and a comparison that drives a decision must have its consequence fixed
> before the number exists. It is hereby CO-PRIMARY with CNN−SF. Both
> endpoints are declared here, each with its own fixed branches; this is
> a two-co-primary design, not post-hoc multiplicity.

Both deltas: per-seed pooled OOF, mean across the 3 seeds, 95% SID-cluster
bootstrap CI with ONE shared storm resampling per replicate.

### V1 — validity: Δ₁ = PR-AUC(CNN) − PR-AUC(GBM on SF)

Does the CNN beat the strongest tabular baseline?

- **CI > 0** → the CNN adds quantified skill beyond everything the
  tabular route captures. Report the delta with its CI, no overselling.
- **CI includes 0** → **NULL**: the CNN is not currently justified over a
  classical baseline; the tabular model becomes the project's reference.
- **CI < 0** → the baseline BEATS the CNN; same consequence, stated
  stronger. Persistence/state (which the intensity-blind CNN cannot see)
  is the prime suspect — check V2 and the S arm to decompose.

### V2 — architecture justification: Δ₂ = PR-AUC(CNN) − PR-AUC(GBM on F)

At a fixed information diet (the same cube fields, full grids vs
mean/std/min/max aggregates): does THIS CNN extract anything beyond
aggregate statistics? This is the comparison that decides whether the
architecture earns its existence, and the most reusable measurement this
project can produce (the spatial-structure rung of the surface-information
ceiling).

- **CI > 0** → the CNN extracts spatial-structure signal beyond field
  aggregates — the architecture is justified on the field diet, and the
  delta quantifies the value of full spatial resolution at 0.25°.
- **CI includes 0** → **NULL — the architecture is NOT justified in its
  current form**: full grids give this CNN nothing detectable beyond 44
  aggregate scalars. Consequence: the V3 model line becomes the tabular
  reference (or a redesigned model, e.g. with a state branch, as a NEW
  pre-registered test); the CNN is reported as a documented negative.
- **CI < 0** → the aggregates BEAT the CNN; same consequence, stronger.

**Scope guard for V2 (fixed now):** a null/negative Δ₂ unjustifies THIS
architecture; it does NOT establish that "spatial structure carries no RI
information" — a stronger spatial reader could extract what this one
cannot. The claim licensed is about the model, not the physics.

### Joint reading (declared in advance)

- Δ₂ > 0 and Δ₁ ≤ 0: the CNN reads spatial structure but loses to state
  information it never receives — the indicated path is adding a state
  branch to the CNN (pre-registered separately before any such run).
- Δ₂ ≤ 0: architecture retired/redesigned regardless of Δ₁.

Secondary descriptive readings (no verdict value, reported once):
- Absolute OOF PR-AUC with 95% SID-cluster CI for every arm (S, F, SF,
  logistic reference, CNN) — the S arm alone shows how far
  persistence/state carries.
- SF−S (value of field information within a fixed model class).

## Discipline (anti-rationalization)

- The GBM may be trained before H6 completes (its OOF does not depend on
  the CNN), but the **Δ verdict is computed ONCE, only after all 3 H6
  seeds exist**. No partial-seed verdicts (the harness refuses unless
  explicitly overridden with `--allow-partial`, which prints INTERMEDIATE
  and has no verdict value).
- No feature mining: the feature list above is final for the primary
  verdict. Any added feature set is a NEW pre-registered comparison.
- No hyperparameter tuning on either side.
- Whatever the outcome, it is reported with its CI in the V3 manuscript.

## Cost

Feature extraction ~14k cube reads (one-time, cached); GBM training:
minutes of CPU. No GPU, no overnight run, no interaction with the phased
CNN ablation schedule.
