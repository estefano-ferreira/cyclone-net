# Literature Review (initial)

> **Status: initial survey based on abstract-level search; to be deepened
> with full-text reading and domain-expert (meteorologist) review before
> publication. Not exhaustive.** This document exists to (a) position
> CycloneNet honestly relative to the established field and (b) seed the
> related-work section of the V3 manuscript. Claims below are drawn from
> abstracts and secondary sources unless marked otherwise; bibliographic
> details flagged "to verify" must be checked against the full text before
> citation in any manuscript.

## 1. RI prediction — the established field

Rapid intensification (RI) is conventionally defined as an increase in
maximum sustained wind of **ΔV24 ≥ 30 kt in 24 h**, the ~95th percentile of
24-h intensity changes (Kaplan & DeMaria 2003). The operational statistical
baseline is the **SHIPS Rapid Intensification Index (SHIPS-RII)**, in
operational use since the early 2000s (Kaplan & DeMaria 2003; revised in
Kaplan et al. 2010) — probabilistic RI guidance built on linear
discriminant analysis over environmental predictors.

The environmental predictors of RI are well established and replicated
across basins:

- low deep-layer (850–200 hPa) vertical wind shear;
- high sea-surface temperature and, more specifically, high **ocean heat
  content / TCHP** (integrated upper-ocean warmth, not skin SST alone);
- high mid-level relative humidity;
- upper-level divergence / favorable outflow.

(Kaplan & DeMaria 2003; Kaplan et al. 2015; Hendricks et al. 2010.)

**Key finding for our positioning:** favorable environmental conditions are
**necessary but not sufficient** for RI. The environments of RI storms and
of non-RI *intensifying* storms are statistically similar, and RI has been
described as a "weak function" of the environment, with a case for treating
part of RI occurrence as effectively **stochastic** (Kowch & Emanuel 2015).
The residual variance is attributed in the literature to **internal
inner-core processes** — convective bursts, inner-core symmetry, eyewall
mesovortices, vortex-scale dynamics — plus a stochastic component; Judt &
Chen (2016) show with high-resolution stochastic ensembles that RI timing
uncertainty arises from interactions between shear, the mean vortex, and
internal convective processes. This is the field's central open problem,
not a gap this project discovered.

## 2. Deep learning for RI — the active ML sub-field

Deep learning for RI prediction is an active sub-field with several
recurring architectures (abstract-level survey):

- **CNN on reanalysis/NWP fields** — e.g., DeepTC (Kim et al. 2024), a
  CNN with amplitude focal loss for 24-h RI in the Western Pacific,
  reporting skill above operational forecasts; hybrid CNNs combining
  satellite imagery with NWP predictors.
- **CNN + gradient boosting hybrids** — e.g., TCNET (Wei, Yang & Sun
  2023), combining CNN feature extraction on ERA-Interim fields with
  XGBoost over SHIPS-type scalar predictors.
- **Attention-based models on satellite imagery** (Bai, Chen & Lin 2020)
  and **temporal LSTM/RNN** approaches over predictor sequences.

Two trends matter for positioning:

1. **The frontier has moved to inner-core structure** observed via
   high-resolution satellite imagery (inner-core symmetry indices,
   convective-burst signatures) — consistent with §1's residual being
   inner-core-driven. CycloneNet does **not** have this data: 0.25°
   surface-level reanalysis cannot resolve inner-core structure.
2. **Reproducibility and benchmarking are a recognized contribution
   category** in this sub-field — e.g., the public satellite-image RI
   benchmark with released dataset of Bai, Chen & Lin (2020; ECML PKDD).
   Papers whose primary contribution is a documented, auditable,
   reproducible pipeline (rather than a new SOTA number) have an
   established niche.

**Direct precedent found 2026-07-15 (weakens any release-novelty claim):**
An & Jeong (2026, *JGR: Machine Learning and Computation*) predict WNP RI
with a tabular ML model (TabNet; SMOTE augmentation and loss-function
comparison as the stated contributions) over **ERA5-derived predictors**
(single + pressure levels, 1977–2021, JMA best track) and **release the
derived dataset + code publicly on Zenodo** (DOI 10.5281/zenodo.15833650).
Their feature construction differs from ours — distance-weighted averages
of the 4 grid points nearest the storm center (scalar features), not
spatial cubes — but this IS the precedent of "publicly released,
peer-reviewed, ERA5-derived RI dataset". See §5 for the positioning
consequence.

## 3. Where CycloneNet sits (honest positioning)

- **Not a physical discovery.** Our environmental-precursor findings
  (shear and mid-level RH differ before RI vs. non-RI; H4/H5 in
  `hypothesis_registry.md` — not to be confused with the internal
  "H1–H4" family of `ri_precursors.py`, a separate numbering the
  registry documents) are **confirmatory** of §1's established
  predictors, obtained on our own 1980–2023 reanalysis-derived dataset.
  They demonstrate the pipeline recovers known signal — a sanity check,
  not a contribution.
- **Our "anomaly hypothesis" (H7)** — that some RI residual exists beyond
  known environmental conditions — **is the field's central known
  problem**, already attributed to inner-core and stochastic processes
  (Kowch & Emanuel 2015; Judt & Chen 2016). Those processes are beyond the
  reach of 0.25° surface reanalysis, so this project cannot be expected to
  resolve it; any H7 test must be framed against that prior.
- **Our contribution is on a different axis:** reproducibility and
  auditability engineering (hash-deterministic SID splits, frozen test
  set, provenance manifests, pre-registered hypotheses and ablations), an
  **honest negative result** (the FuelMap hypothesis refuted with
  storm-center controls — see `fuelmap_validation.md`), and a fully
  documented, resumable pipeline over 46 years of data. This places the
  work in the **reproducibility / applied-ML / benchmark** category, not
  the physical-discovery category.
- **Appropriate venue:** reproducibility tracks, applied-ML venues, or
  workshops, with a domain (meteorology) coauthor validating the
  scientific framing before submission. **Added 2026-07-15:** data
  journals — *Earth System Science Data* (Copernicus) and *Scientific
  Data* (Nature) — publish **datasets as the primary, peer-reviewed
  contribution**, and both already carry ERA5-derived TC datasets (§4,
  entries 13–14). Given that this project's real contribution is the
  auditable dataset + pipeline rather than the model, a data journal is
  arguably the best-matched route for the dataset component (with the
  measurement/benchmark analysis published separately or as companion).

## 4. References (to complete)

> **Peer-review status verified per reference; preprints marked as such and
> not treated as established fact.** Tags: **[PEER-REVIEWED]** journal with
> review (verified); **[PREPRINT]** arXiv or similar without a confirmed
> peer-reviewed publication — cite explicitly as preprint; **[SECONDARY]**
> Wikipedia/ResearchGate-level sources — orientation only, never cited
> directly. Reading status per entry: [full-text] or [abstract-only].
> V3 rule: only [PEER-REVIEWED] work may be cited as "established".

Bibliographic details (authors/title/venue/DOI) were verified against
publisher pages via web search on 2026-07-14 (entries 1–11) and 2026-07-15
(entries 12–14). **Every entry below is [abstract-only]** — none read in
full yet (see §5).

**Foundational — RI climatology and operational prediction:**

1. **[PEER-REVIEWED]** [abstract-only] Kaplan, J., & DeMaria, M. (2003).
   Large-Scale Characteristics of Rapidly Intensifying Tropical Cyclones
   in the North Atlantic Basin. *Weather and Forecasting*, 18(6),
   1093–1108. DOI: 10.1175/1520-0434(2003)018<1093:LCORIT>2.0.CO;2 —
   defines RI (ΔV24 ≥ 30 kt ≈ 95th percentile), identifies large-scale
   environmental predictors; basis of SHIPS-RII (operational 2003).
2. **[PEER-REVIEWED]** [abstract-only] Kaplan, J., DeMaria, M., & Knaff,
   J. A. (2010). A Revised Tropical Cyclone Rapid Intensification Index
   for the Atlantic and Eastern North Pacific Basins. *Weather and
   Forecasting*, 25(1), 220–241. DOI: 10.1175/2009WAF2222280.1 — revised
   SHIPS-RII, multiple RI thresholds (25/30/35 kt), E. Pacific extension.
3. **[PEER-REVIEWED]** [abstract-only] Kaplan, J., et al. (2015).
   Evaluating Environmental Impacts on Tropical Cyclone Rapid
   Intensification Predictability Utilizing Statistical Models. *Weather
   and Forecasting*, 30(5), 1374–1396. DOI: 10.1175/WAF-D-15-0032.1 —
   multi-lead-time evaluation of statistical RI models; consensus skill.
4. **[PEER-REVIEWED]** [abstract-only] Hendricks, E. A., Peng, M. S., Fu,
   B., & Li, T. (2010). Quantifying Environmental Control on Tropical
   Cyclone Intensity Change. *Monthly Weather Review*, 138(8), 3243–3271.
   DOI: 10.1175/2010MWR3185.1 — partitions environmental vs. internal
   contributions; environments of RI and non-RI intensifiers are similar.
5. **[PEER-REVIEWED]** [abstract-only] Kowch, R., & Emanuel, K. (2015).
   Are Special Processes at Work in the Rapid Intensification of Tropical
   Cyclones? *Monthly Weather Review*, 143(3), 878–882.
   DOI: 10.1175/MWR-D-14-00360.1 — RI as the tail of a single smooth
   intensification distribution; case for a stochastic view.
6. **[PEER-REVIEWED]** [abstract-only] Judt, F., & Chen, S. S. (2016).
   Predictability and Dynamics of Tropical Cyclone Rapid Intensification
   Deduced from High-Resolution Stochastic Ensembles. *Monthly Weather
   Review*, 144(12), 4395–4420. DOI: 10.1175/MWR-D-15-0283.1 — RI timing
   uncertainty from shear–vortex–convection interactions; stochastic
   inner-core component.

**Deep learning for RI:**

7. **[PEER-REVIEWED]** [abstract-only] Kim, J.-H., Ham, Y.-G., Kim, D.,
   Li, T., & Ma, C. (2024). Improvement in Forecasting Short-Term
   Tropical Cyclone Intensity Change and Their Rapid Intensification
   Using Deep Learning. *Artificial Intelligence for the Earth Systems*,
   3(2), e230052. DOI: 10.1175/AIES-D-23-0052.1 — DeepTC: CNN with
   amplitude focal loss, W. Pacific; reports skill above operational
   forecasts. (Model name to re-confirm on full-text read.)
8. **[PEER-REVIEWED]** [abstract-only] Wei, Y., Yang, R., & Sun, D.
   (2023). Investigating Tropical Cyclone Rapid Intensification with an
   Advanced Artificial Intelligence System and Gridded Reanalysis Data.
   *Atmosphere*, 14(2), 195. DOI: 10.3390/atmos14020195 — TCNET:
   CNN + XGBoost on ERA-Interim + SHIPS predictors; reports POD/FAR gains
   over SHIPS-only. (MDPI journal — weigh accordingly.)
9. **[PEER-REVIEWED]** [abstract-only] Bai, C.-Y., Chen, B.-F., & Lin,
   H.-T. (2020). Benchmarking Tropical Cyclone Rapid Intensification with
   Satellite Images and Attention-Based Deep Models. *ECML PKDD 2020*,
   LNCS 12460, 497–512. Springer. (Preprint: arXiv:1909.11616, 2019 —
   later peer-reviewed, so cite the proceedings version.) — first
   satellite-image-only RI benchmark with public dataset; the
   reproducibility/benchmark precedent for this project's category.

**Reviews (entry points for deepening this survey):**

10. **[PEER-REVIEWED]** [abstract-only] Chen, R., Zhang, W., & Wang, X.
    (2020). Machine Learning in Tropical Cyclone Forecast Modeling: A
    Review. *Atmosphere*, 11(7), 676. DOI: 10.3390/atmos11070676.
11. **[PEER-REVIEWED]** [abstract-only] Wang, Z., et al. (2022). A Review
    on the Application of Machine Learning Methods in Tropical Cyclone
    Forecasting. *Frontiers in Earth Science*, 10, 902596.
    DOI: 10.3389/feart.2022.902596.

**ERA5-derived TC/RI datasets released publicly (added 2026-07-15):**

12. **[PEER-REVIEWED]** [abstract-only] An, S., & Jeong, J. (2026).
    Machine Learning Based Prediction of Tropical Cyclone Rapid
    Intensification in the Western North Pacific: Importance of Data
    Augmentation and Loss Function. *Journal of Geophysical Research:
    Machine Learning and Computation*, 3, e2025JH000876.
    DOI: 10.1029/2025JH000876. Derived data + code:
    Zenodo, DOI 10.5281/zenodo.15833650 (verified: notebook + CSVs with
    track and environmental data, WNP). — TabNet on ERA5-derived scalar
    predictors (single + pressure levels, 1977–2021, JMA best track;
    distance-weighted average of the 4 grid points nearest the center —
    NOT spatial cubes); contributions framed as SMOTE augmentation +
    loss-function choice. **The direct precedent of a publicly released,
    peer-reviewed, ERA5-derived RI dataset.** (Publisher article page is
    bot-blocked; bibliography verified via the publisher's search
    listing, the Zenodo record, and an institutional repository listing.
    Method details from abstract-level search — confirm on full read.)
13. **[PEER-REVIEWED]** [abstract-only] Xu, Z., Guo, J., Zhang, G., Ye,
    Y., Zhao, H., & Chen, H. (2024). Global tropical cyclone size and
    intensity reconstruction dataset for 1959–2022 based on IBTrACS and
    ERA5 data. *Earth System Science Data*, 16, 5753.
    DOI: 10.5194/essd-16-5753-2024 — ML reconstruction merging IBTrACS +
    ERA5; dataset as the primary peer-reviewed contribution (data
    journal).
14. **[PEER-REVIEWED]** [abstract-only] Liu, G., Jiang, S., Zheng, M.,
    Lin, S., Kong, Y., & Zhan, P. (2025). A Global ERA5-based Tropical
    Cyclone Wind Field Dataset Enhanced by Integrated Parametric
    Correction Methods. *Scientific Data*.
    DOI: 10.1038/s41597-025-05789-w — corrected ERA5 TC wind fields
    validated against SMAP/WindSat/SFMR; dataset as the primary
    peer-reviewed contribution (data journal).
15. **[PEER-REVIEWED]** [FULL-TEXT] Kapoor, S., & Narayanan, A. (2023).
    Leakage and the reproducibility crisis in machine-learning-based
    science. *Patterns*, 4(9), 100804.
    DOI: 10.1016/j.patter.2023.100804 — methodology reference (outside
    RI, cited for principle): taxonomy of data leakage.
    "Nonindependence between train and test samples" is the exact mode
    of storm-level leakage in event-based RI datasets (recommended
    mitigation: blocked/grouped CV); temporal leakage is classified
    separately — the vector that leave-one-year-out designs control and
    SID-grouped splits do not. Cited in MANUSCRIPT_V3 §1 and §8
    (limitations: temporal axis).

No [PREPRINT]-only or [SECONDARY]-only entries remain in this list: the
one arXiv item (#9) was confirmed as later peer-reviewed. Secondary
sources (ResearchGate/Wikipedia) were used only to locate primary pages
during verification and are not cited.

## 5. Gaps in this review (honest)

- No paper below has been read in full yet by the author; everything is
  abstract-level or secondary-source. Full-text reading is required before
  the V3 related-work section is written.
- Needs domain-expert (meteorologist) validation — both of the summary
  claims and of paper selection.
- Recent work (2023–2026) is under-covered; key papers may be missing
  entirely.
- Basin-specific literature (Atlantic vs. West/East Pacific differences in
  RI climatology and predictor importance) not systematically surveyed.
- Operational-forecasting literature (HAFS, consensus aids, DTOPS) not
  covered at all.
- **Positioning hypothesis, NOT VERIFIED at this review level:** we have
  not found a publication that measures the RI-predictability lower bound
  extractable from coarse (0.25°) surface reanalysis alone, under
  storm-level leakage control and without mixing satellite/NWP predictors.
  Novelty-by-absence cannot be claimed from an abstract-level,
  non-exhaustive review with 2023–2026 under-covered. Requires full-text
  reading plus domain review before any novelty claim in V3 — first item
  for the domain co-author. If the hypothesis falls, the V3 framing
  ("measuring the limit" as the contribution) must be downgraded to
  replication/benchmark.
  - **Update 2026-07-15 — one component REFUTED:** the search found a
    direct precedent of a **publicly released, peer-reviewed,
    ERA5-derived RI dataset** (An & Jeong 2026, entry 12; data + code on
    Zenodo). Any claim of pioneering "releasing a reanalysis-derived RI
    dataset" is refuted — do not make it. The STRICT hypothesis above
    (measuring the lower bound from coarse surface reanalysis ALONE,
    with storm-level leakage control, no satellite/NWP mixing) remains
    unverified either way: none of the precedents found does exactly
    that — but this is abstract-level search, not a review, so absence
    still cannot be claimed. The downgrade consequence stands unchanged.
