# CycloneNet: A Reproducible Physics-Guided Pipeline for Forensic Rapid-Intensification Analysis — and an Honest Audit of What It Can and Cannot Do

**Estefano Senhor Ferreira**
Software Engineer, Independent Researcher
estefano.senhor@gmail.com · https://github.com/estefano-ferreira/cyclone-net

*Revised, corrected manuscript (supersedes "CycloneNet V2: ... Atmospheric Singularity Mapping", Feb 2026; this revision, Jul 2026, adds the full 1980–2023 data release and results). See §8 for the correction record.*

---

## Abstract

CycloneNet is an open-source, configuration-driven pipeline for forensic (hindcast)
analysis of tropical-cyclone rapid intensification (RI) from ERA5 reanalysis. It couples
a 3D-CNN RI classifier with an interpretable spatial output ("FuelMap") and physics-guided
weak-constraint losses.

This paper is deliberately **not** a claim of novelty. The individual components
(3D-CNNs for RI, physics-informed/interpretable ML for tropical cyclones, ocean-heat
predictors of RI) are established in the literature, and the central physical question —
*where is the ocean energy that fuels a storm?* — is already addressed operationally by
Tropical Cyclone Heat Potential (TCHP)/ocean-heat-content products and by coupled
ocean–atmosphere models (HWRF, HAFS) that compute air–sea enthalpy flux directly.
CycloneNet does not advance the state of the art in RI skill, nor does it discover a new
physical mechanism.

What this paper contributes is twofold and modest: (1) a **transparent, auditable,
reproducible engineering artifact** — a hybrid physics+ML hindcast pipeline with
leakage-safe splits, train-only normalization, unit tests, and a counterfactual causal
test; and (2) an **honest audit** that yields cautionary, mostly **negative** results:

- The learned FuelMap does **not** localize the ocean energy proxy (TCHP) better than a
  trivial "predict the storm centre" baseline.
- Adding a surface-altimetry ocean channel (absolute dynamic topography, ADT) — itself a
  known TCHP proxy — does **not** measurably improve RI skill in this small-data regime.
- The "equation-consistency" physics loss is **near-degenerate** by construction.
- A prior release of this pipeline reported the model as "physics-guided" while the
  physics losses were, in fact, inactive (weight 0) in the released configuration.

The audit's central reproducibility failure — headline metrics (ROC-AUC 0.83) obtained
on an unreleased dataset — is resolved in this revision by releasing the full
**1980–2023** dataset (16,780 events, 802 RI positives, 992 storms) built by a windowed,
checksummed pipeline, and retraining deterministically on it. On the held-out test split
(2,679 events, 115 RI positives) the classifier achieves **ROC-AUC 0.796 [95% CI
0.753–0.837]** and **PR-AUC 0.251 [0.179–0.331]** at the test split's 4.3% prevalence — the first version
of this project whose skill is statistically distinguishable from chance, and the
confirmation-by-intervention of the audit's diagnostic that sample size, not
architecture, was the bottleneck. The negative results on spatial attribution stand and
are strengthened by two additional controls. We present this as a reproducible baseline
and as a caution against overclaiming in physics-guided cyclone ML.

---

## 1. Introduction

Tropical cyclones are heat engines: they convert heat stored in the upper ocean into
kinetic energy through air–sea enthalpy flux, with the rate of extraction governed by the
wind field and modulated by subsurface ocean heat content (Emanuel's potential-intensity
theory). Predicting rapid intensification (RI; a ≥30 kt increase in 24 h) is an active,
mature operational and research problem.

CycloneNet asks a *forensic* variant of this question: for a historical storm episode,
which environmental conditions in the reanalysis are most consistent with
intensification-supportive thermodynamics? This is a diagnostic/interpretive framing, not
a forecasting one. We stress at the outset that this framing, and the methods used to
pursue it, are **not new** (§2). The purpose of this paper is to document the system
honestly — including what an audit shows it cannot do — rather than to assert a
contribution it does not make.

## 2. Positioning relative to existing work (what already exists)

It is important to state plainly what CycloneNet does *not* originate:

- **Deep learning for RI** is established, including convolutional models over the North
  Atlantic and East Pacific and ensemble/transformer approaches that match or exceed the
  skill reported here.
- **Ocean heat content / altimetry for RI** is foundational operational oceanography:
  TCHP can be estimated from altimetry-derived sea-surface-height fields and is already a
  high-performing predictor in the operational SHIPS Rapid Intensification Index. The ADT
  ↔ TCHP relationship CycloneNet "rediscovers" (§5) is decades old.
- **Physics-informed and interpretable ML for TC intensity** (spatial attention,
  physics-informed networks, interpretable transformers) is an active subfield.
- **Identifying and quantifying the ocean energy source** of a storm is operational:
  TCHP/OHC products (NOAA/AOML, NESDIS, Navy NCODA) map the available fuel, and coupled
  models (HWRF since 2007; HAFS, official 2023) compute the air–sea enthalpy-flux field
  and the storm-induced cold wake every cycle.

CycloneNet occupies none of this as new ground. Its only non-trivial differentiator is
engineering discipline (auditability, reproducibility, testing), which is a matter of
execution quality, not scientific contribution.

## 3. Data

- **ERA5** (Copernicus CDS), 0.25°: SST, MSLP, 10 m winds; optionally 2 m temperature and
  dewpoint for bulk heat-flux diagnostics. The **public release covers 1980–2023**,
  season months over a single extraction sector (60°N–0°, 140°W–20°W) that **spans two
  basins: East Pacific and North Atlantic** (992 storms in the valid dataset: 578 EP /
  414 NA by genesis basin — the first point of each storm's IBTrACS record; basin is a
  per-point attribute and six storms genuinely cross between the basins; the bounding
  box cuts the EP basin west of 140°W).
  Earlier versions of this section said "North Atlantic sector" — incorrect; see the
  correction record (§8) and ERRATA.md item 7. Because the raw monthly files (~60 GB) exceed
  local storage, the dataset is built by a **windowed pipeline**
  (download → extract → verify → discard raw), with per-window provenance manifests
  recording SHA-256 checksums, per-event extraction outcomes, and the deletion record —
  every released cube is auditable back to its source file.
- **IBTrACS** best-track for storm centres, intensities, and RI labels (Δv ≥ 30 kt/24 h).
  Event identifiers embed the storm ID (`era5_{timestamp}_{SID}`), so concurrent storms
  at the same timestamp never collide.
- **TCHP** (NOAA/AOML ERDDAP) and **sea-level anomaly / ADT** (Copernicus DUACS) for
  external ocean validation. Note: the AOML gridded TCHP product covers 2022–present only;
  pre-2022 gridded TCHP is not publicly available, which limits validation to 2022–2023.

For each event, a 40×40 (≈10°×10°) spatio-temporal cube over five 6-hourly steps
(t0…t−24 h) is extracted with nine input channels (SST, MSLP, U10, V10, wind speed,
vorticity, divergence, |∇MSLP|, SST anomaly). Splits are storm-level and
**hash-deterministic**: each storm is assigned to train/val/test by the SHA-256 hash of
its identifier, so adding new storms never reassigns existing ones (the previous
seeded-shuffle scheme reassigned 136 of 155 test events when the dataset grew — a
leakage mechanism eliminated in this revision). Normalization statistics are computed on
the training split only. The full dataset yields **16,780 valid events from 992 storms
(802 RI positives, 4.8% global prevalence)**; train/val/test = 11,150/2,951/2,679 events with
545/142/115 positives (per-split prevalence 4.3–4.9%).

## 4. Methods

**Model.** A two-layer 3D-CNN produces (i) an RI logit and auxiliary 12/24 h intensity-
change regressions via global pooling, and (ii) a 2D FuelMap logit map via a 1×1 head,
from which a soft-argmax yields continuous predicted coordinates.

**Physics-guided losses (now correctly active).** Controlled by explicit weights in
`config.yaml`:
- *FuelMap–prior alignment* (KL): aligns the FuelMap distribution with a heuristic
  physical prior P = ReLU(SST_anom)·wind·(1+ReLU(−div)). The prior is explicitly heuristic.
- *Forward constraint*: an energy score from the overlap of FuelMap and prior is mapped to
  the 24 h intensity change, supervising "localized surface energy → intensification".
- *TV / L1* regularizers on the FuelMap.
- *Equation consistency* (off by default): we document that this term is **near-degenerate**
  — it compares vorticity/divergence recomputed from the wind field against stored channels
  derived from the same wind field; empirically, enabling it changes the loss by <0.001%.

**Optional ADT ocean channel.** ADT can be sampled onto each event grid and appended as an
extra input channel (with a missingness mask where ocean coverage is absent).

**Threshold.** Selected on validation by a configurable policy; the default targets the
project's high-recall forensic regime (highest-precision threshold meeting a recall target).

**Causal ablation (with control).** To test whether the model's RI prediction depends on
the region the FuelMap identifies, we occlude the top-k% FuelMap pixels and compare the
drop in predicted RI probability against occluding an equal-size *low-fuel control* region,
using a paired test.

**Spatial validation (with baseline).** Where TCHP is available, we compare the distance
from the FuelMap peak to the TCHP peak against the distance from the **storm centre** to
the TCHP peak — the model only demonstrates spatial skill if it beats this baseline.

## 5. Results

### 5.1 Classification (reproducible from public data)

On the held-out 1980–2023 test split (n = 2,679; **115 RI positives**), the
physics-guided model achieves **ROC-AUC 0.796 [95% CI 0.753–0.837]** and
**PR-AUC 0.251 [0.179–0.331]** against the test split's 4.3% prevalence (5.8× chance), with recall
0.852 and precision 0.070 at the validation-selected high-recall operating point
(Brier 0.037, ECE 0.011; bootstrap CIs, 10,000 resamples). Both confidence intervals
sit entirely above chance. Training is deterministic — the released checkpoint was
reproduced digit-for-digit from scratch — and every number in this section is
computable from the public repository and dataset, resolving the reproducibility
failure recorded in §8.

The trajectory across dataset generations is itself a result. The same architecture,
losses, and hyperparameters produce:

| Dataset | Test positives | ROC-AUC [95% CI] | PR-AUC ÷ prevalence |
| --- | --- | --- | --- |
| 2020–2023 (original release) | 9 | 0.590 [0.334–0.842] | 2.1× |
| 2020–2023, collision-fixed IDs | 13 | 0.614 [0.453–0.785] | 3.6× |
| 1980–2023 (this revision) | 115 | **0.796 [0.753–0.837]** | **5.8×** |

The first two rows have confidence intervals spanning chance; the audit's diagnostic —
that sample size, not architecture, was the binding constraint — is thereby confirmed
by intervention rather than argument. We still make no operational skill claim: no
comparison against SHIPS-RII or other operational baselines has been performed (§7).

### 5.2 Spatial localization — negative result, three independent angles

The negative result on FuelMap localization survives the dataset expansion and two
additional controls:

1. **Static TCHP validation** (n = 226 eligible test events, 2022–2023 coverage): the
   FuelMap peak beats a random-point null (p = 0.0003) — it tracks the storm — but does
   **not** locate the TCHP peak better than the naive storm-centre baseline (median
   539 km vs 561 km; closer in only 46% of events; sign-flip permutation p = 0.30).
2. **Dynamic displacement test:** the FuelMap's apparent collapse toward the storm
   centre during RI episodes, a candidate dynamic signal, was tested explicitly.
3. **Physics-prior control:** repeating the displacement analysis on the pure
   enthalpy-flux prior — no learned weights at all — reproduces the same dynamic
   behavior. The "signal" is arithmetic of the physics prior, not learned skill.

Naïve "peak-of-field-in-window" localization is further confounded by the basin-scale
ocean gradient (a majority of in-window TCHP maxima fall on the window edge). **We
therefore report FuelMap-based localization of the energy source as unsupported**, and
note that the model's demonstrable classification skill (§5.1) does not transfer to
spatial attribution.

### 5.3 ADT ocean input — null result

A controlled with/without-ADT ablation (same seed, data, epochs) shows ADT does **not**
improve RI classification — it is marginally worse on every split (full test ROC-AUC
0.599 → 0.578). With only 4–9 RI positives per evaluation subset, the result is
underpowered and inconclusive; the honest reading is that ADT's value cannot be
demonstrated in this data regime. Separately, ADT does track TCHP at the storm centre
(Spearman ρ ≈ 0.30, replicated for 2022 and 2023) — but this merely **reproduces known
altimetry-OHC relationships**, it does not establish new science. (This ablation
predates the 1980–2023 expansion; a paired feature ablation on the expanded dataset,
with ~800 positives of statistical power, is planned as follow-up work.)

### 5.4 Model-internal causal dependence — positive but limited

Occluding the FuelMap-identified region reduces predicted RI significantly more than
occluding a low-fuel control region (paired test, p ≈ 1e-6; ~93–96% of events). This shows
the *trained model* relies on that region — it does **not** show the region is the true
physical energy source. It is an interpretability check, not a physical attribution.

## 6. Discussion

The audit clarifies what CycloneNet is. Its FuelMap is a saliency/attention-like derived
representation whose localization validity, when tested against an independent ocean proxy,
is not supported. Its "physics guidance" is weak supervision toward a heuristic prior, not
a physical law (and one of its physics terms is degenerate). Its ocean-input extension
reproduces established altimetry-OHC relationships without improving skill. The framing of
"identifying the energy source feeding a hurricane" overstates the scope: the fuel is
extracted locally at the air–sea interface beneath the storm (not from a remote region),
and mapping it is already operational via TCHP/OHC and coupled models.

What remains genuinely useful is the artifact's transparency and the discipline of testing
each claim — including reporting the negative ones. We believe documenting these failure
modes has value: physics-guided and "interpretable" cyclone-ML papers frequently assert
that learned maps recover physical quantities; here we show, with an explicit baseline,
that a plausible such map does not.

## 7. Limitations

- **Surface-only input features.** The nine channels are surface/near-surface fields;
  the model sees neither the subsurface ocean reservoir that governs sustained RI nor
  the atmospheric column (e.g. deep-layer shear, mid-level humidity) that operational
  predictors exploit. Pressure-level extensions are implemented in the pipeline but not
  yet part of the released training set.
- **0.25° ERA5 resolution** cannot resolve inner-core structure; the model necessarily
  learns environmental favorability, not storm-scale dynamics.
- **No comparison against operational baselines (SHIPS-RII)** — required before any
  operational skill claim; §5.1's result is skill above chance, not skill above the
  state of practice. SHIPS-RII is additionally fitted **per basin**, so a direct
  comparison against this two-basin model is not possible without separating basins.
- **Two-basin heterogeneity.** The dataset mixes two basins (East Pacific + North
  Atlantic) with different RI climatologies (shear, TCHP, seasonality); basin is not
  used as a control in the current experiments, and per-basin analysis is
  under-powered (687 dev positives total). This heterogeneity is a declared
  limitation, not a controlled variable.
- TCHP validation limited to 2022–2023 (gridded pre-2022 TCHP unavailable publicly),
  i.e. n = 226 of 2,679 test events.
- The heuristic prior is not a physical guarantee; the equation-consistency term is degenerate.
- The recall-first operating point yields precision 0.070 — appropriate for the forensic
  screening framing, unusable as an alarm system.

## 8. Correction record

This manuscript corrects the prior version ("CycloneNet V2 … Atmospheric Singularity
Mapping"). Specifically: (1) the released code did not implement the physics-guided losses
it described (weights defaulted to 0); this is fixed. (2) The threshold methodology in the
code differed from the text; this is reconciled. (3) The headline metrics (2,193 test
samples, ROC-AUC 0.83) were not reproducible from the public repository and derived from a
larger, unreleased dataset; this is **resolved in this revision** by releasing the full
1980–2023 dataset and replacing the superseded numbers with the reproducible results of
§5.1 — the 0.83/0.905 figures should not be cited. (4) The title and
"uniqueness/innovation" framing overstated the scope relative to existing literature and
operational tools; the title is revised and the framing corrected throughout. A detailed
errata is provided in the repository (`ERRATA.md`).

## 9. Conclusion

CycloneNet is an honest, reproducible, well-engineered hybrid physics+ML pipeline for
forensic RI analysis. It is **not** a novel method, a new scientific finding, nor a tool
that identifies the physical energy source of hurricanes — that capability already exists
operationally. Its contribution is engineering transparency plus a two-sided audit: on
the positive side, RI classification skill that is statistically distinguishable from
chance and fully reproducible from public artifacts (ROC-AUC 0.796, PR-AUC 5.8× chance,
115 test positives), obtained by fixing the audit-identified bottleneck (sample size)
rather than the architecture; on the negative side, learned FuelMap localization does
not beat a trivial baseline — a result now robust across a static TCHP comparison, a
dynamic displacement test, and a physics-prior control. We release it as a reproducible
baseline and as a record of what, in this approach, does and does not work.

*Code, data instructions, tests, and the errata are available under CC BY-NC 4.0 at
https://github.com/estefano-ferreira/cyclone-net.*

## Acknowledgements

ERA5 (Copernicus C3S), IBTrACS (NOAA NCEI), TCHP (NOAA/AOML), and DUACS sea-level data
(Copernicus Marine). No external funding.
