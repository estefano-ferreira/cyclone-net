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
  training result) | **Tested:** —
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
- **Status:** TESTING
- **Verdict:** — (seed 42 complete 2026-07-14, pooled-OOF Δ +0.033 is
  INTERMEDIATE with no verdict value; seeds 123/456 pending)
- **Notes:** operational progress in `docs/ablation_progress.md`.

---

### H7: A region shows residual RI beyond what known conditions explain ("anomaly hypothesis")
- **Registered:** 2026-07-14 | **Tested:** —
- **Question:** is there a region where RI occurs more than
  SST/TCHP/shear/RH explain (positive spatial residual after accounting for
  known conditions)?
- **Physical motivation:** [to be filled by author]. Absence recorded per
  the rules — until filled, this is a fragility signal of the hypothesis.
- **Test design:** to be designed — pre-declared sketch: RI rate per
  spatial cell (not raw count) + residual vs a conditions model + spatial
  permutation null. Design BEFORE looking at any map.
- **Status:** UNTESTED
- **Verdict:** —
- **Notes:** high spatial-fishing risk; the null and multiple-comparison
  correction must be fixed before the first plot.

---

_To add a new hypothesis: copy the template block, take the next H[N],
fill Question / Physical motivation / Test design BEFORE running any
analysis, and commit the registration before the result._
