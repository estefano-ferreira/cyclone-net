# CycloneNet: A Reproducible Pipeline and Leakage-Safe Two-Basin Dataset for Tropical-Cyclone Rapid-Intensification Analysis

<!-- DRAFT (V3 Data Descriptor) — assembled 2026-07-16 from reviewed section drafts.
     Skeleton/spec: docs/manuscript_v3_skeleton.md. DOI slots resolved 2026-07-18 to the
     concept DOI 10.5281/zenodo.18571957 (single archived record since v3.0.1).
     Format: Scientific Data Data Descriptor (strict); venue decision open. -->


## Abstract

CycloneNet is an open-source, configuration-driven pipeline for retrospective analysis of tropical-cyclone rapid intensification (RI) from ERA5 reanalysis and IBTrACS best tracks, released together with a leakage-safe two-basin dataset spanning 1980–2023 (East Pacific and North Atlantic; 16,780 events from 992 storms: 799 RI positives, 15,962 negatives, 19 undefined labels under strict-temporal v2 labeling). The dataset is built through a windowed extraction pipeline with checksummed provenance manifests, employing storm-level hash-deterministic splits and train-only normalization to prevent label leakage. A full audit trail documents byte-reproducibility of the RI labeling chain from raw IBTrACS; the underlying ERA5 data are discarded after windowed extraction but cubes are reconstructible via CDS re-download. Complete v1→v2 correction records are documented: 148 of 32,989 rows misaligned (0.45%), zero valid-set label flips, and 19 events reclassified as undefined. Pre-registered evaluation results are frozen on the test split (2,679 events; 112 labeled RI positives, 6 undefined). The pipeline architecture excludes ocean heat-flux proxies from training inputs to prevent information leakage during model development. We release this as a reproducible baseline for RI research and as an honest record of a pre-registered campaign's negative results on spatial-attribution and architecture hypotheses.


## 1. Background & Summary

Tropical-cyclone rapid intensification (RI), defined as sustained wind increase ≥30 kt within 24 hours, remains one of the most challenging phenomena to forecast operationally. Understanding the environmental conditions that enable RI—and retrospectively diagnosing why a particular storm intensified rapidly—is essential for research and risk assessment. This work adopts a forensic (hindcast) framing: given a historical storm event and reanalysis-based environmental fields, which features are most consistent with the observed intensification? This is a diagnostic question, not a real-time forecast, motivating a dataset designed for reproducible, auditable analysis rather than operational deployment.

A persistent challenge in tropical-cyclone machine-learning studies is *label leakage*—unintended information flow from the target or related fields into the training set. The present dataset is designed to be leakage-safe through four architectural choices: (1) **SID-level assignment**—storms, not individual events, form the unit of train/validation/test partition, preventing the same storm's history from spanning training and held-out sets and avoiding information correlation across splits; (2) **deterministic and frozen test assignment**—the test split is assigned deterministically by SHA-256 hash of storm identifier, making the assignment invariant to dataset growth, and is further checked against a frozen map (data/normalized/frozen_splits.json) that preserves historical benchmark assignments; (3) **train-only normalization**—all feature standardization statistics (mean, variance) are computed on the training split only and applied identically to validation and test, preventing the held-out distribution from leaking into training-set preprocessing; (4) **heat-flux channel exclusion**—the derived surface heat-flux channels (latent, sensible, total) are stored in the cubes but deliberately excluded from model inputs, a conservative anti-leakage design. These design decisions are formally verified in the Technical Validation section (split integrity), where byte-reproducibility checks and per-channel audit trails establish their integrity.

CycloneNet's development proceeded through a pre-registered evaluation campaign (2026-07-16) testing hypotheses about spatial attribution, ocean-input utility, and model architecture. The campaign closed with null or negative results on all central claims: the learned FuelMap does not outperform a storm-centre baseline in localizing ocean energy when tested against independent TCHP observations (ERRATA item 4, docs/hypothesis_registry.md); added pressure-level channels showed no detectable contribution (H6: null result); and the three-dimensional convolutional architecture showed no advantage over a simpler tabular baseline when operating on equivalent information (H9: the architecture is not justified in its current form). Accordingly, this work designates no reference model and positions the validated contribution as the dataset and reproducible pipeline—released as a transparent, auditable baseline for RI research.

Existing RI datasets and prediction tools are well-established. Deep learning for RI, including convolutional and ensemble approaches, is an active research area with published work across the North Atlantic and East Pacific basins. Ocean heat content and sea-level anomaly (altimetry-derived absolute dynamic topography, ADT) are foundational oceanographic predictors of RI, already integrated into the operational SHIPS Rapid Intensification Index (SHIPS-RII). Physics-informed and interpretable machine-learning methods for tropical-cyclone intensity prediction—spatial attention mechanisms, physics-informed neural networks, interpretable transformers—constitute an established subfield. The present dataset does not originate these capabilities. Its contribution is engineering-focused: an openly available, two-basin RI dataset (East Pacific and North Atlantic, 1980–2023 hurricane seasons) with explicit leakage-mitigation architecture, full provenance documentation, complete audit trails of label corrections (v1→v2), and reproducible results from a pre-registered campaign. Direct comparison against SHIPS-RII is not performed; SHIPS-RII is fitted separately per basin, and a two-basin model requires basin-stratified evaluation to be meaningful. The dataset is released to support reproducible RI research.

## 2. Methods

### Data Sources

The CycloneNet v2 dataset integrates three primary data sources: ERA5 reanalysis for environmental fields, IBTrACS best-track for storm positions and intensities, and supplementary ocean-heat validation products. 

**ERA5 Reanalysis.** Environmental data are from Copernicus CDS. Single-level: Hersbach, H., Bell, B., Berrisford, P., et al. (2018): ERA5 hourly data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.adbb2d47. Pressure levels: Hersbach, H., et al. (2018): ERA5 hourly data on pressure levels from 1940 to present. DOI: 10.24381/cds.bd0915c6. Data span 1980–2023, 0.25° resolution, 6-hourly, over [60°N, 140°W, 0°N, 20°W] (East Pacific + North Atlantic). Contains modified Copernicus Climate Change Service information (1980-2023). Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains. Raw monthly files are downloaded, extracted, verified, and discarded by design.

**IBTrACS Best-Track.** Storm positions, intensities, and rapid-intensification labels derive from the International Best Track Archive for Climate Stewardship (IBTrACS), Version v04r00 (Knapp, K. R., M. C. Kruk, D. H. Levinson, H. J. Diamond, and C. J. Neumann, 2010: The International Best Track Archive for Climate Stewardship (IBTrACS): Unifying tropical cyclone best track data. Bull. Amer. Meteor. Soc., 91, 363-376; NOAA National Centers for Environmental Information, Dataset DOI: 10.25921/82ty-9e16). Each 6-hourly track point defines an event.

**Ocean-Heat Validation Supplements.** Tropical Cyclone Heat Potential (TCHP; NOAA/AOML) and absolute dynamic topography (ADT; Copernicus Marine Service) are used exclusively for external validation (see repository documentation) and do not serve as model inputs.

### Event Definition

An event is a single 6-hourly storm-track point at time t₀, together with its 24-hour preceding ERA5 history. For each event, a 40×40 grid window (approximately 10°×10° at the event's latitude) is extracted from ERA5 centered on the storm's IBTrACS position, capturing five 6-hourly time levels: t₀, t−₆ₕ, t−₁₂ₕ, t−₁₈ₕ, and t−₂₄ₕ. This yields a spatio-temporal cube of shape (40, 40, 5, C), where C is the channel count.

The dataset contains **16,780 valid events** from **992 tropical cyclones** (578 EP / 414 NA by genesis point). The per-point basin attribute is taken from IBTrACS; six storms cross basin boundaries during their track, recorded in the per-point basin field; genesis-basin attribution uses the first recorded point of each storm's IBTrACS record.

### Cube Channels (C = 14)

Fourteen channels are preserved: (1) nine model-input channels—SST, MSLP, U10/V10 wind, wind speed, vorticity, divergence, MSLP gradient, SST anomaly—derived from ERA5 single-level fields via finite differences; (2) three stored-only heat-flux channels (latent, sensible, total; W/m²), computed but deliberately excluded from model inputs to prevent leakage; and (3) two tested supplementary channels—shear (850–200 hPa) and mid-level relative humidity (700–600–500 hPa)—from ERA5 pressure levels. These were formally tested (H6 hypothesis) and found to contribute no detectable skill; they are retained for reproducibility.

### Label Semantics (v2, Strict-Temporal with Tri-State)

Rapid intensification is defined as a ≥30 knot increase in maximum sustained wind over exactly 24 hours (Kaplan & DeMaria 2003). For each event at time t₀, the label is determined by identifying the **exact temporal partner**: the IBTrACS record for the same storm at precisely t₀ + 24 hours (no tolerance window). If such a partner exists, `ri_label` = 1 if the 24 h wind change (dv24) ≥ 30 kt, else 0. If no exact partner exists—at storm dissipation, in reporting gaps, or at the dataset's temporal boundary—the label is `NULL` (undefined), **never coerced to 0**. This tri-state semantics (0, 1, NULL) prevents label artifacts from temporal discontinuities.

The dataset contains **799 RI-positive events (dv24 ≥ 30 kt), 15,962 negatives, and 19 undefined events** (out of 16,780 valid events). 12-hour deltas (dv12) are computed analogously for supplementary analysis.

**v1→v2 Correction.** The previous release (v1) computed dv24/dv12 by positional row shifts (−4/−2 rows, assuming perfect 6-h regularity in the track table). This method misaligned 148 of 32,989 raw track points (0.45%). Corrected v2 labels use exact 24 h temporal partners with no positional assumptions. The correction reclassified 19 valid events to NULL (storm end or reporting gaps); positives decreased from 802 to 799 (−3, all in the test split, via NULL reclassification). A full per-row provenance record (`label_diff_v1_v2.csv`) is distributed, mapping each event's v1 and v2 labels and the correction reason (unchanged, flip_misaligned, null_no_partner, dv_drift_only). The audit details are recorded in the repository's ERRATA and TECHNICAL_VALIDATION documents.

### Quality Control

Events are subject to per-cube quality-control checks:

- **Physical ranges:** SST ∈ [240, 330] K, MSLP ∈ [80000, 110000] Pa, wind speed ≤ 80 m/s.
- **NaN budgets:** events are rejected if more than 50% of any single-channel grid is NaN.
- **Temporal integrity:** all five time levels must correspond to distinct ERA5 records (no timestamp collapse).
- **Geospatial integrity:** the storm center (from IBTrACS) must lie within the extracted 40×40 window; events where the center falls outside are rejected.

Rejected events are catalogued in `rejected_events.csv` with their exclusion reason and are **not distributed** (their cubes are not packaged). This transparency allows users to audit the filtering threshold and verify that a desired event was either included or intentionally excluded.

### Normalization

Training-set statistics (channel-wise mean and standard deviation) are computed over the 70% training split **only**, using all spatial and temporal grid points in training events. These statistics are applied to all subsequent splits (validation and test) and distributed in `normalization_stats.json`. Per-channel statistics ensure that each feature is zero-centered and unit-scaled consistently. This train-only normalization prevents the test split statistics from influencing model training—a critical leakage safeguard.

### Splits: Hash-Deterministic with Frozen Override

Events are partitioned into train (70%), validation (15%), and test (15%) splits at the **storm level** using the SHA256 hash of each storm's identifier (SID). No storm appears in more than one split, preventing event-level leakage. The hash-deterministic scheme ensures that adding new storms to the dataset never reassigns existing storms to different splits.

A **frozen override map** (`frozen_splits.json`) is consulted first during split assignment, preserving historical benchmark assignments. The SHA256 hash rule assigns any storm not in the frozen map, ensuring deterministic composition invariance as new storms are added. Split composition: **train** 11,150 events; **validation** 2,951 events; **test** 2,679 events, comprising 112 RI-positive and 6 NULL labels. No label stratification is applied. The test split is frozen (never read during this project's model development); once published, external researchers may read it, but this project does not assert external benchmark-replicability (see Usage Notes).

### Windowed Processing and Provenance Manifests

Raw ERA5 monthly files are downloaded, extracted per-event, verified, and **discarded by design**—a choice balancing storage practicality with provenance rigor. Each processing window generates a manifest (JSON) recording source-file SHA256 hashes, per-event extraction outcomes, and deletion timestamps. These manifests form an immutable audit trail linking every released cube to its source file and enabling byte-level replication. Basin-repair and label-correction manifests document all transformations. The labeling chain (raw IBTrACS → event labels) is byte-reproducible via deterministic recomputation with an abort-on-mismatch replication gate.

### Summary

The v2 dataset is a leakage-safe, auditable compilation of 16,780 events from 992 storms across 1980–2023, spanning two tropical cyclone basins (EP/NA). Events are defined as 6-hourly track points with 24 h ERA5 reanalysis context, extracted as (40, 40, 5, 14) float32 cubes. Labels are strict-temporal, tri-state (0, 1, NULL), derived from exact 24 h partner identification with ≥30 kt threshold. Quality-control, train-only normalization, and hash-deterministic SID-level splits enforce leakage prevention. Windowed processing and per-window provenance manifests provide reproducibility without distributing raw reanalysis files. The dataset is distributed under CC BY 4.0 with verbatim source citations.

## 3. Data Records

### 3.1 Repository and access

The dataset is published in the project's single archived Zenodo record (concept DOI 10.5281/zenodo.18571957, resolves to the latest version) under the CC BY 4.0 license. See NOTICE for mandatory attributions; the analysis pipeline (github.com/estefano-ferreira/cyclone-net) is released under MIT. The released package contains 46 files totaling 6.03 GiB:
- 44 per-year cubic shards (organized in `cubes/` by year subdirectories)
- 1 metadata archive
- 1 checksums file

### 3.2 File inventory

| File pattern | Content | Quantity |
|---|---|---|
| `cubes/<year>/{event_id}.npy` | Per-event cube, shape (40, 40, 5, 14), float32, 448 KB each | 16,780 |
| `cubes/<year>/{event_id}.json` | Per-event metadata sidecar (§3.3) | 16,780 |
| `cubes/<year>/{event_id}_lats.npy` | Latitude grid (40-point array) for the window | 16,780 |
| `cubes/<year>/{event_id}_lons.npy` | Longitude grid (40-point array) for the window | 16,780 |
| `cubes/<year>/{event_id}_adt.npy` | ADT (Absolute Dynamic Topography) window; 2020–2023 only | 761 |
| `event_list_augmented.csv` | Complete labeled track-point list; superset of valid events | 32,989 rows |
| `valid_events.csv` | Summary: event_id, SID, storm_name, ri_label for the 16,780 released events | 1 |
| `splits.csv` | Train/val/test split assignment (70/15/15 by storm) | 1 |
| `frozen_splits.json` | Frozen override map ensuring split determinism and historical reproducibility | 1 |
| `normalization_stats.json` | Channel-wise statistics (mean, std, min, max) computed on train split only | 1 |
| `label_diff_v1_v2.csv` | v1→v2 label provenance and corrections for all 32,989 rows; reasons: unchanged, flip_misaligned, null_no_partner, dv_drift_only | 1 |
| `rejected_events.csv` | QC-excluded events (cubes not distributed); rejection-reason breakdown included | 1 |
| `provenance/*.json` | Processing-window, basin-repair, and label-correction manifests with MD5 checksums | 45 |
| `DATA_DICTIONARY.md`, `TECHNICAL_VALIDATION.md`, `NOTICE`, `LICENSE`, `CHECKSUMS.sha256`, `package_manifest.json` | Documentation and integrity records | — |

Event IDs follow the format `era5_YYYY_MM_DD_HHMM_<SID>` (UTC). The time axis T indexes as follows: index 0 = t₀ (event time), then −6 h, −12 h, −18 h, −24 h.

### 3.3 Cube schema and sidecar fields

Each valid event is stored as a float32 array of shape **(40, 40, 5, 14)** (spatial height, spatial width, time steps, channels). Accompanying each cube is a JSON sidecar containing the following fields:

| Field | Meaning |
|---|---|
| `event_id`, `sid`, `storm_name`, `basin` | Identity fields; `basin` is the per-point IBTrACS basin code (§3.5) |
| `timestamp` | Event time t₀ in UTC, format `YYYY-MM-DD HH:MM` |
| `center_lat`, `center_lon` | Storm center latitude and longitude at t₀ (from IBTrACS) |
| `wind_kt` | Best-track maximum sustained wind at t₀, in knots |
| `pressure_mb` | Central pressure if available; nullable |
| `dv12_kt`, `dv24_kt` | Intensity change (wind speed delta) at t₀+12h and t₀+24h, respectively; nullable if partner does not exist |
| `ri_label` | RI label ∈ {0, 1, null}; null denotes undefined (§3.6) |
| `channels`, `cube_shape`, `units` | Cube metadata: ordered channel names and their units (listed in §3.4) |
| `timestamps`, `era5_selected_times`, `era5_time_indices` | UTC timestamps and metadata for the 5 ERA5 records backing the T dimension |
| `qc_flags` | Quality control details: NaN fraction per channel (SST, MSL, u10, v10), SST range validation, MSL range validation, wind magnitude validation |
| `temporal_integrity_ok` | Boolean: all 5 timesteps are distinct ERA5 records (no duplicates or gaps) |
| `adt_saved`, `fuel_potential_saved` | Boolean flags indicating whether optional artifacts are present in this event's directory |
| `source_files` | ERA5 source file names from CDS (for provenance tracking) |

In the distributed package, `fuel_potential_saved` is `false` for all events. The underlying heuristic was unsupported and is not included; divergence from the local pipeline's internal sidecars is documented in `package_manifest.json`.

### 3.4 Channels (14 stored, variable consumption)

The cube's channel dimension (C = 14) is ordered as follows:

| # | Name | Unit | Source | Model input (historical)? |
|---|------|------|--------|---|
| 1 | `sst_K` | K | ERA5 single-level | yes |
| 2 | `mslp_Pa` | Pa | ERA5 single-level | yes |
| 3 | `u10_mps` | m s⁻¹ | ERA5 single-level | yes |
| 4 | `v10_mps` | m s⁻¹ | ERA5 single-level | yes |
| 5 | `wind_mps` | m s⁻¹ | Derived (√u10²+v10²) | yes |
| 6 | `vort_1ps` | s⁻¹ | Derived (∂v/∂x − ∂u/∂y) | yes |
| 7 | `div_1ps` | s⁻¹ | Derived (∂u/∂x + ∂v/∂y) | yes |
| 8 | `grad_mslp_Pa_per_m` | Pa m⁻¹ | Derived | yes |
| 9 | `sst_anom_K` | K | Derived (local mean removal) | yes |
| 10 | `latent_heat_flux_Wpm2` | W m⁻² | Derived (bulk formula) | **no** — stored only |
| 11 | `sensible_heat_flux_Wpm2` | W m⁻² | Derived (bulk formula) | **no** — stored only |
| 12 | `total_heat_flux_Wpm2` | W m⁻² | Derived | **no** (excluded via `exclude_total_heat_flux_from_input: true` to prevent leakage) |
| 13 | `shear_850_200_mps` | m s⁻¹ | ERA5 pressure levels [850, 200] hPa | tested; hypothesis H6: NULL |
| 14 | `rh_mid` | % | ERA5 pressure levels [700, 600, 500] hPa | tested; hypothesis H6: NULL |

Channels 1–9 were the channels used as model inputs in the project's experiments; ADT is an optional additional feature. The three heat-flux channels (10–12) are stored for completeness but deliberately excluded from model inputs (leakage prevention). Channels 13–14 were added during pressure-level backfill; hypothesis H6 formally tested their contribution and found no detectable skill gain at this resolution/regime (a documented null result, not an endorsement of their utility).

### 3.5 Basin semantics and the "NA" field

Each event's `basin` field contains the per-point IBTrACS basin code: either `EP` (East Pacific) or `NA` (North Atlantic). Six storms cross basin boundaries during their track; when attributing properties at the storm level (e.g., for storm-level grouping), use the genesis point. Of the 992 storms: 578 originated in the East Pacific, and 414 in the North Atlantic.

**CRITICAL READER NOTE:** The string `"NA"` (North Atlantic) is a categorical value, not a missing value. When reading any distributed CSV file with pandas, always specify `keep_default_na=False, na_values=[""]` to preserve it. Without this flag, `"NA"` entries are silently coerced to NaN, causing silent data loss. This bug pattern has occurred six times in this project's history (documented in TECHNICAL_VALIDATION).

### 3.6 Label semantics (v2: strict-temporal, tri-state)

The `ri_label` field is tri-state: 0, 1, or null.
- **ri_label = 1** if the 24-hour maximum sustained wind increase (ΔV₂₄) from t₀ to t₀+24h is ≥ 30 kt (Kaplan & DeMaria, 2003).
- **ri_label = 0** if ΔV₂₄ < 30 kt.
- **ri_label = null** if no exact temporal partner exists at t₀+24h (e.g., storm dissipated, reporting gap).

NULL is a label state, not a missing value; it denotes that the RI classification is undefined for that event. Readers must not coerce NULL to 0 and must exclude NULL events from RI-classification tasks (19 of 16,780 events).

Full label provenance is recorded in `label_diff_v1_v2.csv` (32,989 rows): per-row v1/v2 values, the correction applied, and the reason (unchanged, flip_misaligned, null_no_partner, or dv_drift_only). Complete audit, defect documentation, and correction statistics are in the Technical Validation section and the repository's ERRATA (items 6 and 8).

### 3.7 Population summary and splits

The distributed dataset comprises:
- **16,780 valid events** (6-hourly track points with a complete 24-hour ERA5 history window)
- **992 storms**: 578 EP genesis, 414 NA genesis
- **RI labels**: 799 positive (RI occurred), 15,962 negative (no RI), 19 NULL (undefined)
- **Event list**: 32,989 labeled track points (superset of valid events; excluded events are listed in `rejected_events.csv` with QC-failure reasons)
- **Temporal coverage**: 1980–2023, June–November (hurricane season only; no off-season tropical cyclones)
- **Spatial coverage**: 60°N–0°N latitude, 140°W–20°W longitude (East Pacific truncated west of 140°W; North Atlantic)

The frozen test split (never read during model development in this project; distributed for reproducibility) contains 2,679 events, of which 112 are RI-positive (v2 labels) and 6 are NULL. Train/val splits are assigned by storm (SID) via SHA256-deterministic hash into 70/15/15 proportions, with historical overrides applied via `frozen_splits.json`. No label stratification is applied; class proportions per split are approximate. Full split documentation and integrity tests are in the Technical Validation section (§4.3).

## 4. Technical Validation

### 4.1 Byte-reproducibility of the labeling pipeline

Reconstruction of the 32,989-row event list from the raw IBTrACS archive reproduces the shipped file **byte-for-byte** (32,989/32,989 rows with identical dv24 and ri_label values on every row). The labeling chain applies synoptic-hour filtering, per-storm intensity-change calculation, and target dropna. Script `analysis/dv24_impact_assessment_v5_raw_reference.py` performs exact replication and aborts unless byte-for-byte match is achieved — the project's permanent "replication gate" for all defect diagnoses.

### 4.2 Label correction from v1 to v2, fully audited

The v1 labels were computed using positional shifts (−4 rows for dv24, −2 rows for dv12), assuming perfect 6 h regularity in the best-track archive. When measured against raw IBTrACS on the full population, **148/32,989 rows (0.45%) show misalignment** where the positional partner is not at the canonical temporal offset. Critically, **zero label flips occurred in the released valid set**. Under strict-temporal semantics (partner = exact match at t0 + 12 h or t0 + 24 h), **19 valid events have no temporal partner and are reclassified as undefined (NULL)**; as a result, RI positives decreased from 802 to **799**. All corrections were made at the source in `src/processors/ri_labeling.py` (temporal join, nullable Int64) with per-file verification and full per-row provenance documented in `label_diff_v1_v2.csv`. The authoritative impact assessment (`outputs/results/dv24_impact/report_v5_20260716_152525.*`) supersedes earlier intermediate reports.

### 4.3 Split integrity and leakage safety (proof of title claim)

The title's "Leakage-Safe" claim is directly supported by this subsection. Splits are assigned at the storm (SID) level using SHA256-hash-determinism with a frozen override map; **no storm crosses between train, validation, or test splits**. Property-tested via `tests/test_splits_stability.py`, these assignments are invariant to dataset composition and preserve the historical benchmark. Events lacking a storm assignment fail loudly (no silent exclusion). The development set (all model work and hypothesis testing) comprises 14,101 events across 839 storms. By SID, the full dataset is divided 70/15/15 into train, validation, and test, yielding 2,679 test events (112 v2-positives, 6 NULL). **Train-only normalization** (coefficients computed on training split only; `normalization_stats.json`) prevents information leakage from scaling statistics. **Stored-only channels** (heat-flux fields, per DATA_DICTIONARY §4) are never fed to supervised models, providing structural proof against leakage through feature engineering. Together, SID-deterministic splits with frozen override, train-only normalization, and stored-only channels constitute the complete leakage-safety architecture, closing all three classical leakage vectors: (1) storm information across splits, (2) test-time distribution in training statistics, and (3) test-time information in feature derivation.

### 4.4 Pressure-level completeness census

Following the 1980–2019 pressure-level backfill campaign (21,662 events processed through 20 windowed extraction stages, zero failures, with per-window provenance manifests), an independent census confirmed **100% coverage of the development set: 14,101/14,101 events** carry both pressure-level channels (`shear_850_200_mps` and `rh_mid`, DATA_DICTIONARY §4). The result (`outputs/results/pl_gate_census.json`) gates every downstream experiment.

### 4.5 Processing provenance and basin metadata repair

Raw ERA5 monthly files (≈57 GB in total, measured from the per-window provenance manifests: ≈24 GB of single-level fields over 22 windows, 1980–2023, and ≈33 GB of pressure-level fields over 20 windows, 1980–2019) are not retained in the release; they are discarded exclusively through a windowed extraction process in which every compressed field is verified before deletion, with a provenance manifest per window recording row counts, field checksums, and source-file sizes. Basin metadata was audited and repaired in July 2026 following discovery of a pandas NA-parsing bug that had silently converted the literal "NA" (North Atlantic) basin code to empty string in all intermediate artifacts. Post-repair verification confirms **8,888 East Pacific / 7,892 North Atlantic over 16,780 valid events** (per point); by storm genesis basin, **578 EP / 414 NA over 992 storms**. The same pandas pitfall pattern appeared in seven additional readers; a class-wide remediation sweep (all readers now use `keep_default_na=False, na_values=[""]` to preserve "NA" as a literal string) closed the bug class and added regression tests (`tests/test_na_handling_readers.py`).

### 4.6 Quality control and rejected events

Per-event QC occurs at extraction, testing physical ranges (SST ∈ [240, 330] K, MSLP ∈ [80000, 110000] Pa, wind ≤ 80 m/s), NaN budgets (allowable sparsity per channel), and temporal/geospatial integrity (monotonic timestamps, bounding box compliance). QC flags are stored in each event sidecar JSON. Events failing QC are listed in `rejected_events.csv` and excluded from all cube arrays. Normalization statistics (`normalization_stats.json`) are computed exclusively on the training split, preventing inadvertent test-time information leakage through scaling coefficients.

### 4.7 Usability evidence

A gradient-boosted baseline trained on tabular features derived from the data cubes (grouped k-fold by SID over the development set, pooled out-of-fold) attains **PR-AUC ≈ 0.25** against a positive base rate of **4.8%**, demonstrating that the labels support supervised learning despite the modestly imbalanced class distribution. The retired convolutional neural network architecture, historically evaluated on a frozen test set under v1 labels (2,679 events, 115 RI positives; 112 positives under the v2 recount), attained **ROC-AUC 0.796 [95% CI 0.753–0.837], PR-AUC 0.251 [0.179–0.331], recall 0.852** — reported here exclusively as a historical benchmark record and proof-of-concept that the labels admit neural learning; this architecture is retired and its evaluation protocol is separate from current development. Model comparisons, hypothesis verdicts, ablation studies, and operational guidance reside in `BENCHMARK.md` and `hypothesis_registry.md`, not in this data descriptor.

### 4.8 Retraction discipline: the wrong-reference lesson and permanent replication gate

An intermediate assessment (2026-07-16) diagnosed "cross-SID label leakage with 84 phantom positives, 12 storms losing all positives, and 3,062 undefined labels (18.2% of the valid set)" — **the diagnosed defect does not exist**. Reconstruction from raw IBTrACS refuted it entirely. The error arose because the diagnosis operated on a derived artifact (`data/event_list_augmented.csv`) from which the builder's dropna operation had already removed the partner rows used in the original label computation. On that file, the states "partner row never existed" and "partner row was created and then dropped" are indistinguishable, causing legitimate trailing-row labels to appear as cross-storm leakage. The supporting "global shift" evidence (959/5,829 matches, 16% correspondence) was coincidental collision of 5 kt-quantized deltas, not systematic leakage (genuine leakage would show ~100% match rate). **The lesson:** arithmetic consistency across analytical scopes does **not** detect wrong-reference errors; numbers can agree with each other while measuring the wrong thing. **Consequence:** the **raw-replication gate** (reconstruct all shipped artifacts byte-exactly from raw sources; abort on any mismatch) is now a permanent project rule (PROJECT_STATE §3) — every defect diagnosis on derived artifacts must first pass this gate. This discipline is a strength of the validation story, not a limitation: it rendered the false diagnosis discoverable through direct measurement and ensures that all remaining claims rest on direct evidence from the authoritative source.

## 5. Usage Notes

### Reader Requirements

When loading event metadata and labels from the distributed CSV files, readers must invoke `pandas.read_csv()` with the parameters `keep_default_na=False, na_values=[""]`. This is critical: the per-point basin attribute uses the literal string `"NA"` to denote North Atlantic, which collides with pandas' default NA sentinel list. Pandas' default missing-value parser destroys this distinction, conflating the basin identifier with NaN. This bug class—confusing `"NA"` string with missing data—occurred 6 times during this project's development and is the primary motivation for this requirement. Failure to set these flags will silently corrupt basin assignments and any downstream storm-level grouping.

For RI classification tasks, users must exclude the 19 events with NULL labels (represented as empty cells in CSV, `null` in JSON sidecars). NULL is a label state meaning *undefined* — not "no RI" — because the event occurs at storm end or during a tracking gap where the exact 24 h partner does not exist. Coercing NULL to 0 introduces label noise and violates the temporal integrity of the dataset. The reference implementation (`analysis/feature_ablation_kfold.py::load_dev_events`) demonstrates the correct filtering.

The frozen test split (2,679 events; 112 RI positives, 6 NULL) is distributed in `splits.csv` and `frozen_splits.json` for reproducibility and verification. However, the property "unread during model development" applies only to this project's historical development (documented in Technical Validation and the repository's provenance record). Once published, third parties may read the test set; benchmark comparability with this project's historical metrics therefore cannot be re-established by subsequent authors and is not claimed. Replication studies may use the public frozen split for consistency, but new performance claims should employ cross-validation or independent test sets to avoid circular benchmarking.

### Known Limitations

1. Seasonal coverage only (June–November); no off-season tropical cyclones.
2. East Pacific domain truncated at 140°W; two basins only (East Pacific and North Atlantic).
3. Intensities are USA_WIND-centric (agency metadata differences and reconciliation strategies are unrepresented).
4. ERA5 0.25° horizontal resolution cannot resolve inner-core structure; the dataset's channels derive from bulk reanalysis fields, limiting spatial attribution fidelity.
5. Nineteen events carry NULL labels (undefined at storm end or during reporting gaps).
6. RI-classification skill is modest: the baseline gradient-boosted tabular model (pooled out-of-fold PR-AUC ≈0.25) on development folds, compared against a 4.8% base rate of RI in the full valid set, provides usability evidence of the data's capacity to support learning, not a performance ceiling; lower bounds are not ceilings.
7. v1→v2 label correction history (0.45% of rows; zero valid-set flips) is fully documented in `label_diff_v1_v2.csv` with per-row provenance tags.

Two-basin heterogeneity (East Pacific and North Atlantic have distinct RI climatologies, seasonal patterns, and intensity distributions) is not controlled in the current experiments. Basin is a declared limitation, not a covariate in the pipeline, and users performing inter-basin comparisons should account for this nesting explicitly.

### Reconstruction

The event list and RI-label chain are byte-reproducible from raw IBTrACS; the geospatial cubes are reconstructible by re-downloading ERA5 from Copernicus CDS using the configuration file `config-template.yaml` and the pipeline orchestrator `run.py`. The recommended command sequence is:

```
python run.py prepare           # Download IBTrACS; generate event list and RI labels
python run.py download-era5     # Fetch ERA5 monthly files from Copernicus CDS
python run.py preprocess        # Extract 40×40×5×14 cubes with 14 channels
python run.py normalize         # Compute train-split-only normalization statistics
```

No credentials are stored in the repository; the CDS API key lives in `~/.cdsapirc` (see the setup instructions in `README.md`). Raw ERA5 files are not distributed (discarded after verified extraction by design); provenance manifests in `outputs/provenance/` document the windowed processing. Full verification of the labeling chain against raw IBTrACS is available via the replication-gate script `analysis/dv24_impact_assessment_v5_raw_reference.py`, which aborts on any mismatch and certifies byte reproducibility.


## 6. Code Availability

The full pipeline and analysis code are publicly available at **https://github.com/estefano-ferreira/cyclone-net** (MIT license; see `LICENSE` file). This data descriptor is accompanied by a single archived Zenodo record (concept DOI 10.5281/zenodo.18571957, resolves to the latest version) containing the repository snapshot at the release tag with the complete configuration and pipeline code required for reconstruction, together with the cubes, sidecars, splits, labels, and metadata, distributed under CC BY 4.0 with mandatory third-party attributions in `NOTICE`. (The two-separate-records plan was retired 2026-07-17; single-record model since v3.0.1.)

An interactive platform explorer is available at **https://estefano-ferreira.github.io/cyclone-net/**—a static, client-side visualization of the IBTrACS best-track events, observed storm tracks, and intensity curves, with no model predictions. Labeling reproducibility is verified by the replication-gate script at `analysis/dv24_impact_assessment_v5_raw_reference.py`, which enforces abort-on-mismatch and certifies byte-reproducible label generation from raw IBTrACS.


## 7. Acknowledgements, Author Contributions, and Competing Interests

### Data Attribution

This dataset and pipeline incorporate data and services from the Copernicus Programme, operated by the European Union, and from NOAA. Verbatim attribution notices (required for CC BY 4.0 compliance) are as follows:

**ERA5 Reanalysis Data:** Contains modified Copernicus Climate Change Service (C3S) information (1980–2023). Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains (Hersbach et al., 2018, https://doi.org/10.24381/cds.adbb2d47 and https://doi.org/10.24381/cds.bd0915c6).

**IBTrACS (Storm Tracks and Intensities):** International Best Track Archive for Climate Stewardship (IBTrACS), Project Version v04r00, NOAA National Centers for Environmental Information (Knapp et al., 2010, Bull. Amer. Meteor. Soc., 91, 363–376). Dataset DOI 10.25921/82ty-9e16. IBTrACS follows the World Data Center for Meteorology policy, providing full and open access to the data; WMO Resolution 40 is used as the guide for commercial use. NOAA/NCEI provide the data without warranty (NCEI ISO metadata gov.noaa.ncdc:C01552).

**ADT (Absolute Dynamic Topography Validation Auxiliary):** Derived from Copernicus Marine Service products (subset of events, 2020–2023). Generated using E.U. Copernicus Marine Service information.

### Author Contributions

⟦author block — as in Zenodo v3 .tex⟧

⟦coauthor slot — pending contact⟧

### Competing Interests

No competing interests are declared. This work was conducted without external funding; computational resources were provided by the author's personal infrastructure.
