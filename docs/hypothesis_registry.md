# Hypothesis Registry

Living research agenda for CycloneNet. Every project hypothesis — tested or
not — lives here, in the spirit of pre-registration.

**Rules:**
- Register the hypothesis AND the test design BEFORE running (for all
  hypotheses added from 2026-07-14 onward).
- Record the HONEST verdict afterwards — positive, null, or refuted.
- A hypothesis without a verdict is UNTESTED (that means "unknown", not
  "likely").
- Refuted hypotheses are NOT removed — they are knowledge (we know they are
  false). Keep the history.
- Date everything. Version in git (every change is a traceable commit).
- A hypothesis without clear physical motivation: record that absence — it
  is a fragility signal.

**Convention (instrument integrity):**
- **Tested:** the real test date.
- **Registered:** the date the entry was added to this log (log created
  2026-07-14).
- Tests predating this log are marked **RETROACTIVE ENTRY — not
  pre-registered in this instrument; strict pre-registration applies only
  to hypotheses added from 2026-07-14 onward.**
- The feature ablation (H6) is the first GENUINELY pre-registered test.

Note: `analysis/ri_precursors.py` uses an internal "H1–H4" family naming;
the mapping to this registry is given in the notes of H4/H5 below.

---

### H1: The FuelMap recovers the storm's real energy source
- **Registered:** 2026-07-14 | **Tested:** 2026-06/07 (consolidated)
- **RETROACTIVE ENTRY** — not pre-registered in this instrument.
- **Question:** does the region highlighted by the FuelMap correspond to
  the real oceanic energy source sustaining intensification, beating the
  trivial storm-center baseline?
- **Physical motivation:** the cyclone's fuel is air–sea enthalpy flux fed
  by subsurface ocean heat content (TCHP) — established operational
  literature (AOML / SHIPS-RII).
- **Test design:** three-way comparison (storm center / pure physics /
  FuelMap) against TCHP peak; dynamic test; prior-control experiment.
- **Status:** TESTED
- **Verdict:** **REFUTED** (three converging angles): TCHP validation
  (n=226) shows no significant difference vs the storm-center baseline
  (p=0.30); dynamic test negative; prior control showed the spatial
  collapse was prior arithmetic, not learned structure.
- **Notes:** standing caveat from earlier work: causal ablation
  (`src/evaluation/causal_ablation.py`) proves only the model's INTERNAL
  causal dependence on the FuelMap, not correspondence with the physical
  source. Inviolable consequence: FuelMap = "hypothesis maps", never a
  validated energy source.

---

### H2: ADT as an input channel improves RI prediction
- **Registered:** 2026-07-14 | **Tested:** 2026-07
- **RETROACTIVE ENTRY** — not pre-registered in this instrument.
- **Question:** does adding ADT (surface proxy of the subsurface heat
  reservoir; rho=0.30 vs TCHP, replicated in 2022/2023) as an input channel
  improve RI skill?
- **Physical motivation:** TCHP is an operational RI predictor (SHIPS-RII);
  ADT is its altimetric surface signature with broad coverage.
- **Test design:** with/without-ADT ablation, same seed and protocol.
- **Status:** TESTED
- **Verdict:** **NULL/marginal** — exp_adt: 0.906 (with) vs 0.914
  (without); no improvement, slightly negative direction. Caveat: partial
  ADT coverage (neutral channel outside 2022–2023) and few positives in the
  covered subset → underpowered; NULL here does not definitively refute the
  physical signal (the ADT↔TCHP proxy remains valid).
- **Notes:** re-testable with multi-year SLA/ADT coverage.

---

### H3: The performance bottleneck is sample size, not architecture
- **Registered:** 2026-07-14 | **Tested:** 2026-07-12
- **RETROACTIVE ENTRY** — not pre-registered in this instrument.
- **Question:** does the low PR-AUC stem from the number of positives
  rather than model capacity?
- **Motivation (statistical):** with ~35 original positives, any AUC is
  noise-dominated; this is statistical power, not physics.
- **Test design:** direct intervention — expand the dataset (1980–2023) and
  observe the skill response with the architecture held fixed.
- **Status:** TESTED
- **Verdict:** **POSITIVE** — the 9→115 test-positive expansion (802 total
  positives) confirmed sample size was the bottleneck.
- **Notes:** basis for the current dataset (16,780 events / 802 positives).

---

### H4: Lower deep-layer shear PRECEDES RI onset (precursor, t-24h)
- **Registered:** 2026-07-14 | **Tested:** 2026-07-13
- **RETROACTIVE ENTRY** — the matched-pairs protocol and the frozen-pairs
  re-test were pre-declared in `analysis/ri_precursors.py` (internal "H2"),
  but not in this instrument.
- **Question:** in intensity-matched onset-vs-control pairs, is the shear
  level at t-24h lower for RI onsets?
- **Physical motivation:** deep-layer shear disrupts the warm core and
  suppresses RI.
- **Test design:** frozen matched pairs, sign-flip permutation null,
  primary = level at t-24h, Bonferroni ×4 (fixed H1–H4 family of the
  script).
- **Status:** TESTED
- **Verdict:** **POSITIVE** — n=394/394 pairs (100% PL coverage), paired
  Δ = −1.03 m/s, Cliff's δ = −0.13, p(Bonf ×4) = 1.2e-3,
  physics-consistent direction (commit `970a419`).
- **Notes:** **CONFIRMATORY positive** — shear as an RI predictor is
  established (SHIPS basis); this confirms it with real N, it does not
  discover it. **Statistical precedence ≠ incremental model gain — this
  does NOT prejudge the feature ablation (H6).** The pre-backfill version
  (n=5) was untestable; the valid verdict is the post-backfill re-test on
  the SAME frozen pairs (zero re-matching).

---

### H5: Higher mid-level humidity PRECEDES RI onset (precursor, t-24h)
- **Registered:** 2026-07-14 | **Tested:** 2026-07-13
- **RETROACTIVE ENTRY** — same standing as H4 (script-internal "H4").
- **Question:** as H4, for rh_mid at t-24h (higher for RI onsets?).
- **Physical motivation:** dry mid-level air suppresses the deep convection
  RI requires.
- **Test design:** identical to H4 (same Bonferroni ×4 family).
- **Status:** TESTED
- **Verdict:** **POSITIVE** — n=394/394, paired Δ = +2.59% RH,
  Cliff's δ = +0.12, p(Bonf ×4) = 8.8e-3, physics-consistent direction
  (commit `970a419`).
- **Notes:** same reading as H4: CONFIRMATORY, small and robust; does NOT
  prejudge the ablation (H6). The family's other two primaries (24h
  pressure fall, p=4.0e-4; SST, δ=+0.53, p=4.0e-4) were also POSITIVE in
  the same run — recorded here as family context, no separate entries.

---

### H6: Shear and mid-level RH add PR-AUC over the current channel baseline
- **Registered:** 2026-07-13 (**first genuinely pre-registered test**:
  `docs/ablation_preregistration.md`, commit `eaa8ae8`, fixed before any
  training result) | **Tested:** 2026-07-16
- **Question:** does adding shear_850_200_mps and rh_mid to the current 9
  input channels improve RI PR-AUC?
- **Physical motivation:** classic SHIPS predictors (see H4/H5); the open
  question is incremental gain over what the current channels already
  encode.
- **Test design:** arm A (9 channels) vs arm B (+shear/RH), k=3 folds, 15
  epochs, seeds {42, 123, 456} phased one per night; single verdict via
  mean cross-seed ΔPR-AUC with 95% SID-cluster bootstrap CI
  (`--aggregate`), through the 3 pre-registered decision branches. CI read
  ONCE; no mining, no re-runs.
- **Status:** TESTED
- **Verdict:** **NULL** — mean cross-seed ΔPR-AUC (B−A) = +0.0185,
  95% SID-cluster bootstrap CI **[−0.0070, +0.0431] includes zero**
  (10,000/10,000 valid resamples; per-seed Δ +0.033/+0.011/+0.011;
  14,101 events / 839 SIDs; report
  `outputs/results/feature_ablation_cnn/aggregate_20260716T120517Z.json`).
  Read once, 2026-07-16, through the pre-registered branches: shear/rh_mid
  add **no detectable skill at this resolution/regime for this
  architecture**. Per pre-registration: NOT to be reinterpreted as a weak
  positive.
- **Notes:** operational progress in `docs/ablation_progress.md`. Runs:
  seed 42 `20260713T232126Z` (`c608f19`), seed 123 `20260714T223910Z`
  (`c6a3b20`), seed 456 `20260715T223350Z` (`bb7adaa`); zero incidents,
  all detached. Claim discipline: the verdict is about THIS estimator
  (GAP-pooling 3D-CNN) on 0.25° surface fields — it does not assert the
  added channels carry no extractable signal for other estimators (cf.
  H9's GBM, where scalar shear/RH means participate in arm F). Verdict
  unlocks H8 (arm-A reuse) and the H9 `--compare-cnn` paired read.

---

### H7: A region shows residual RI beyond what known conditions explain ("anomaly hypothesis")
- **Registered:** 2026-07-14 | **Tested:** —
- **Question:** is there a region where RI occurs more than
  SST/TCHP/shear/RH explain (positive spatial residual after accounting for
  known conditions)?
- **Physical motivation (filled 2026-07-14 from the literature survey,
  `docs/literature_review.md`):** the residual is the field's central
  KNOWN problem, not an open anomaly — favorable environment is necessary
  but not sufficient for RI, and the unexplained variance is attributed to
  internal inner-core processes (convective bursts, eyewall dynamics) plus
  a stochastic component (Kowch & Emanuel 2015; Judt & Chen 2016;
  Hendricks et al. 2010). A spatial residual test on 0.25° surface
  reanalysis cannot observe those processes.
- **Test design:** pre-declared sketch retained (RI rate per spatial cell
  + residual vs a conditions model + spatial permutation null), but see
  status.
- **Status:** **DEFERRED — future work.** Re-classified 2026-07-14:
  answering this requires inner-core data (high-resolution satellite
  imagery / structure indices) beyond the current 0.25° surface-level
  dataset. Target: MS program (2027).
- **Verdict:** —
- **Notes:** high spatial-fishing risk stands; if ever run, the null and
  multiple-comparison correction must be fixed before the first plot. Any
  positive found with current data would most likely be an unmeasured
  inner-core/stochastic signal leaking through — interpret accordingly.

---

### H8: The FuelMap physics-guided losses improve RI classification
- **Registered:** 2026-07-14 (**pre-registered**:
  `docs/fuelmap_ablation_preregistration.md`, fixed before any result) |
  **Tested:** —
- **Question:** do the 4 active FuelMap loss terms (KL prior alignment,
  forward constraint, TV, L1) improve RI PR-AUC relative to an identical
  model trained with all physics lambdas = 0 (plain 3D-CNN)?
- **Physical motivation:** genuinely open — H1's refutation killed the
  FuelMap's interpretive claim but says nothing about the losses' value as
  regularization / weak supervision for the classification objective. The
  literature does not predict the answer.
- **Test design:** arm A (physics on, production config) vs arm B (all
  lambdas 0), identical architecture and channels; k=3, 15 epochs, seeds
  {42, 123, 456}; same PL-gated dev set and fold recipe as H6; arm A
  reused from H6's `A_current` cells (fold-identity validated); verdict =
  mean cross-seed ΔPR-AUC (B−A) with 95% SID-cluster bootstrap CI, read
  once through 3 pre-registered branches with FIXED consequences
  (hurt → remove; null → remove for parsimony; help → keep as
  regularization, never as validated physics).
- **Status:** **CANCELLED (2026-07-16).** H8 (FuelMap physics-loss
  ablation) is CANCELLED, not deferred. Its question — do the FuelMap
  physics losses help RI classification? — became undecidable when H9's
  V2 retired the architecture those losses shape (pre-registered joint
  reading, 2026-07-16: Δ₂ ≤ 0 → architecture retired/redesigned
  regardless of Δ₁). Ablating a component of a retired model decides
  nothing. The H8 pre-registration and harness
  (`analysis/fuelmap_ablation_cnn.py`,
  `docs/fuelmap_ablation_preregistration.md`) remain in the repo as
  record — they are not to be run.
- **Verdict:** —
- **Notes:** whatever the outcome would have been, the H1 refutation
  stands. **Consequence closed with the cancellation — the
  "physics-guided" label is retired:** H8 was the honesty test for the
  name; it is moot because the architecture carrying the four
  FuelMap-centred losses was itself retired by H9/V2. Independently of
  H8, the label was already only weakly supported: the KL
  prior-alignment term targets a heuristic prior whose semantics were
  REFUTED in H1; the only equation-consistency term is disabled
  (lambda_consistency = 0.0) and documented as near-degenerate; there
  are no conservation laws or imposed dynamics. Do not use
  "physics-guided" in V3 or in public descriptions. (Repo-wide label
  inventory: PROJECT_STATE §4 relabel item.)

---

### H9: The CNN adds skill beyond a SHIPS-like tabular baseline
- **Registered:** 2026-07-14 (**pre-registered**:
  `docs/tabular_baseline_preregistration.md`, fixed before any result) |
  **Tested:** 2026-07-16 (paired read; GBM side executed 2026-07-15)
- **Question:** does the 3D-CNN (H6 arm `A_current`) beat a
  gradient-boosting model on scalar predictors (Vmax, persistence,
  latitude, season, basin, shear/RH/SST cube means) on identical
  SID-grouped folds, in OOF PR-AUC?
- **Motivation (validity, then science):** never tested — the CNN's skill
  has no classical reference on the same data (BENCHMARK.md records the
  gap). Scientifically, this is the first rung of measuring the
  surface-data information ceiling for RI: scalars vs +spatial structure.
  Note the CNN is intensity-blind (no Vmax/persistence input), so a strong
  tabular showing is plausible, not a strawman.
- **Test design (amended 2026-07-14 pre-result — factorial, TWO co-primary
  verdicts):** GBM (sklearn defaults, no tuning) on three feature sets — S
  (state only), F (field aggregates only: the tabular counterpart of the
  CNN's diet), SF (union) — plus logistic reference on SF; same PL-gated
  dev set, same folds/seeds as H6. **V1 (validity):** Δ₁ = PR-AUC(CNN) −
  PR-AUC(GBM_SF). **V2 (architecture justification, promoted from
  descriptive to co-primary at the author's requirement, still
  pre-result):** Δ₂ = PR-AUC(CNN) − PR-AUC(GBM_F) — decides whether the
  architecture earns its existence at a fixed information diet; null or
  negative → architecture retired/redesigned regardless of V1. Scope
  guard: V2 speaks about THIS model, not about whether spatial structure
  carries information. Both CIs: mean cross-seed, 95% SID-cluster
  bootstrap (shared resampling), read ONCE after all 3 H6 seeds exist.
  SF−S and the S arm alone remain descriptive. Harness:
  `analysis/tabular_baseline_kfold.py`.
- **Status:** TESTED
- **Verdict (both CIs read once, 2026-07-16; report
  `outputs/results/tabular_baseline/compare_20260716T121803Z.json`;
  10,000/10,000 valid shared-resampling replicates):**
  - **V1 (validity): NEGATIVE — the tabular baseline BEATS the CNN.**
    Δ₁ = −0.0781, 95% CI **[−0.1162, −0.0422] < 0** (per-seed
    −0.079/−0.072/−0.083). Pre-registered consequence: the CNN is not
    justified over a classical baseline; **GBM_SF becomes the project's
    reference model**.
  - **V2 (architecture justification): NULL.** Δ₂ = +0.0005, 95% CI
    **[−0.0285, +0.0316] includes zero** (per-seed
    −0.007/+0.014/−0.005). Pre-registered consequence: **the
    architecture is NOT justified in its current form** — full 0.25°
    grids give this CNN nothing detectable beyond 44 aggregate scalars.
    The CNN is reported as a documented negative; any redesign (e.g.
    state branch) is a NEW pre-registered test.
  - **Joint reading (pre-declared):** Δ₂ ≤ 0 (null) → architecture
    retired/redesigned regardless of Δ₁. The Δ₂>0∧Δ₁≤0 "state branch"
    path did NOT trigger.
  - Absolute cross-seed mean OOF PR-AUC (descriptive): CNN_A 0.171 ·
    GBM_F 0.170 · GBM_S 0.202 · GBM_SF 0.249 · LogReg_SF 0.203
    (per-seed values in the report JSON).
- **Ex-ante qualifications (all known before the read; none invented
  after):**
  1. **GAP:** the CNN global-average-pools before classifying — CNN ≈
     GBM(F) was expected OF THIS ARCHITECTURE; per the scope guard, no
     claim about spatial-structure information in the data is licensed.
  2. **Intensity-blind:** the CNN receives no Vmax/persistence; GBM_SF
     does. Part of the V1 gap is information diet, not model class
     (S−F ≈ +0.03 shows how far state alone carries).
  3. **Basin:** the mislabeled one-hot carried the true two-basin
     partition (99.9%) — the GBM effectively used basin as a predictor;
     the CNN never sees it (see collateral finding below).
  4. **15-epoch budget**, identical for every arm and the CNN cells.
- **Notes:** NULL or negative here does not kill the project — it
  redirects it (the tabular model becomes the honest reference, and the
  intensity-blindness of the CNN becomes the first fix). Positive here is
  the first real evidence for spatial surface signal.
- **Collateral finding (2026-07-15, basin audit — relevant when reading
  the verdict):** the `basin` one-hot in the GBM feature table came out as
  `basin_EP`/`basin_` (empty string) due to the `"NA"`-parsing bug
  (ERRATA.md item 7), but the two mislabeled columns carry the TRUE
  two-basin partition almost perfectly (empty ≡ North Atlantic, 99.9%
  event-level correspondence). The GBM therefore effectively **used basin
  membership as a predictor**. Part of the S-arm's skill comes from
  knowing the basin — information the CNN never sees (no basin input).
  This is one more channel of the CNN's state-blindness that V1 captures
  without distinguishing; it does not invalidate any arm (same table in
  all three), but the verdict reading should not attribute all of the
  S/SF advantage to storm state alone.

---

## Future agenda — beyond the current data (MS program, 2027)

Physically valid directions, already established in the literature (see
`docs/literature_review.md`), that CANNOT be tested with the current 0.25°
surface-level reanalysis dataset. Recorded here so they are not mistaken
for open hypotheses of THIS project; each becomes an H[N] entry only when
the required data exists.

- **Inner-core convection / hot towers:** convective bursts and inner-core
  symmetry are the leading explanation for the RI residual (Judt & Chen
  2016; the DL frontier already works on satellite-derived structure).
  Requires high-resolution satellite imagery — outside current resolution.
- **Ocean currents / eddies:** warm-core eddies and current-driven TCHP
  anomalies modulate the fuel reservoir (operational TCHP literature).
  Partially reachable via TCHP/altimetry, but current coverage in this
  project is partial (see H2 caveats); a proper test needs multi-year
  eddy-resolving ocean fields.

Both are established physics, NOT discoveries waiting to be made here —
the honest framing is "known mechanisms our data cannot see".

---

_To add a new hypothesis: copy the template block, take the next H[N],
fill Question / Physical motivation / Test design BEFORE running any
analysis, and commit the registration before the result._
