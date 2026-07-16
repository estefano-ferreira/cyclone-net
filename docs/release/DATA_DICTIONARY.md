# CycloneNet Dataset v2 — Data Dictionary

DRAFT FOR AUTHOR REVIEW (T5 phase 2, 2026-07-16). Slots marked `⟦…⟧` are
unresolved and must be filled before packaging; everything else is verified
against the artifacts or the pipeline code.

## 1. Identity

| Field | Value |
|---|---|
| Name | CycloneNet two-basin tropical-cyclone RI dataset |
| Version | v2 (strict-temporal labels; see §6 and ERRATA items 6/8) |
| DOI | ⟦minted at Zenodo publication; separate dataset concept-DOI⟧ |
| License | **CC BY 4.0** (see `NOTICE` for mandatory attributions) |
| Code | pipeline at github.com/estefano-ferreira/cyclone-net (MIT ⟦after layer-A pass⟧) |

**Temporal coverage: 1980–2023, hurricane season (June–November) ONLY.**
The year range does not imply continuous coverage — off-season tropical
cyclones are absent by construction (`download.months: [6..11]`).

**Spatial coverage:** bounding box [60°N, 140°W, 0°N, 20°W]. Two basins:
East Pacific (truncated west of 140°W — declared limitation) and North
Atlantic.

## 2. Population

- **16,780 valid events** (6-hourly track points with a complete 24 h ERA5
  history window) from **992 storms** — 578 EP / 414 NA by genesis point.
- **RI labels: 799 positive / 15,962 negative / 19 NULL** (v2 semantics, §6).
- `rejected_events.csv` lists events excluded by quality control
  (composition transparency; their cubes are not distributed).
  Rejection-reason counts: ⟦from normalization_report.json at packaging⟧.

## 3. Files

| Pattern | Content |
|---|---|
| `cubes/<year>/{event_id}.npy` | float32 cube, shape **(40, 40, 5, 14)** = (H, W, T, C) |
| `cubes/<year>/{event_id}.json` | per-event metadata sidecar (§5) |
| `cubes/<year>/{event_id}_lats.npy` / `_lons.npy` | 40-point latitude/longitude grids of the window |
| `cubes/<year>/{event_id}_adt.npy` | ADT window (761 events, 2020–2023 only) |
| `event_list_augmented.csv` | full labeled track-point list (32,989 rows; superset of valid events) |
| `valid_events.csv` | event_id, sid, storm_name, ri_label for the 16,780 released events |
| `splits.csv` + `frozen_splits.json` | split assignment + frozen override map (§7) |
| `normalization_stats.json` | train-split-only channel statistics |
| `label_diff_v1_v2.csv` | v1→v2 label provenance, all 32,989 rows (§6) |
| `rejected_events.csv` | QC-excluded events (no cubes) |
| `provenance/*.json` | processing-window, basin-repair and label-correction manifests (md5) |
| `DATA_DICTIONARY.md`, `TECHNICAL_VALIDATION.md`, `NOTICE`, `LICENSE`, `CHECKSUMS.sha256`, `package_manifest.json` | this documentation + integrity |

`event_id` format: `era5_YYYY_MM_DD_HHMM_<SID>` (UTC).

Time axis T: index 0 = t0 (event time), then −6 h, −12 h, −18 h, −24 h
⟦confirm index order against preprocess code before packaging⟧.

## 4. Cube channels (C = 14, in this order)

| # | name | unit | source | RI-model input? |
|---|------|------|--------|------------------|
| 1 | `sst_K` | K | ERA5 single-level | yes |
| 2 | `mslp_Pa` | Pa | ERA5 single-level | yes |
| 3 | `u10_mps` | m s⁻¹ | ERA5 single-level | yes |
| 4 | `v10_mps` | m s⁻¹ | ERA5 single-level | yes |
| 5 | `wind_mps` | m s⁻¹ | derived: √(u10²+v10²) | yes |
| 6 | `vort_1ps` | s⁻¹ | derived: ∂v/∂x − ∂u/∂y | yes |
| 7 | `div_1ps` | s⁻¹ | derived: ∂u/∂x + ∂v/∂y | yes |
| 8 | `grad_mslp_Pa_per_m` | Pa m⁻¹ | derived | yes |
| 9 | `sst_anom_K` | K | derived (local mean removal) | yes |
| 10 | `latent_heat_flux_Wpm2` | W m⁻² | derived (bulk formula) | **no** — stored only |
| 11 | `sensible_heat_flux_Wpm2` | W m⁻² | derived (bulk formula) | **no** — stored only |
| 12 | `total_heat_flux_Wpm2` | W m⁻² | derived | **no** — `exclude_total_heat_flux_from_input: true` (anti-leakage) |
| 13 | `shear_850_200_mps` | m s⁻¹ | ERA5 pressure levels [850, 200] hPa | tested (H6: NULL) |
| 14 | `rh_mid` | % | ERA5 pressure levels [700, 600, 500] hPa | tested (H6: NULL) |

**14 stored vs 10 consumed:** the reference experiments consumed channels
1–9 plus the optional ADT extra. The three heat-flux channels are stored
for loss-function research and are deliberately NOT inputs (leakage
prevention). Channels 13–14 were added by the pressure-level backfill and
their contribution was formally tested (hypothesis H6: no detectable skill
gain at this resolution/regime — a documented null, not an endorsement).

## 5. Sidecar fields (`{event_id}.json`)

| Field | Meaning |
|---|---|
| `event_id`, `sid`, `storm_name`, `basin` | identity; `basin` is the PER-POINT IBTrACS basin code (§8) |
| `timestamp` | event time t0, UTC, `YYYY-MM-DD HH:MM` |
| `center_lat`, `center_lon` | storm center at t0 (IBTrACS) |
| `wind_kt` | best-track max sustained wind at t0 (USA_WIND, knots) |
| `pressure_mb` | central pressure when available (nullable) |
| `dv12_kt`, `dv24_kt` | strict-temporal intensity deltas (§6; nullable) |
| `ri_label` | RI label ∈ {0, 1, null} (§6) |
| `channels`, `cube_shape`, `units` | cube schema (units as in §4) |
| `timestamps`, `era5_selected_times`, `era5_time_indices`, `era5_time_name` | the 5 ERA5 records backing the T axis |
| `qc_flags` | `nan_fraction_per_channel` (sst/msl/u10/v10), `sst_range_ok`, `msl_range_ok`, `wind_abs_ok` |
| `temporal_integrity_ok` | all 5 timesteps are distinct ERA5 records |
| `adt_saved`, `fuel_potential_saved` | presence flags for optional artifacts |
| `source_files` | ERA5 source file names (provenance) |

Note: in the DISTRIBUTED sidecars, `fuel_potential_saved` is `false` for
every event — the `*_fuel_potential.npy` arrays are not part of this
release (the underlying heuristic's semantics were refuted; ERRATA item 4).
The packaging process rewrites this flag so the distributed sidecar
describes the distributed package; the divergence from the local pipeline's
sidecars is recorded in `package_manifest.json`.

## 6. Label semantics (v2 — the primary product)

- `dv24_kt` = wind_kt(partner at **exactly t0+24 h**, same storm) −
  wind_kt(t0); `dv12_kt` analogous at t0+12 h. **No tolerance window.**
- `ri_label` = 1 if dv24_kt ≥ 30 kt; 0 if < 30 kt; **NULL if no exact
  temporal partner exists** (storm end, reporting gap). NULL is a label
  state — *undefined*, never "no RI". CSV: empty cell; JSON: `null`.
- **Readers MUST NOT coerce NULL to 0** and must exclude NULL events from
  RI classification (19 of 16,780). Reference reader:
  `analysis/feature_ablation_kfold.py::load_dev_events`.
- v1→v2 provenance: `label_diff_v1_v2.csv` (per-row v1/v2 values + reason ∈
  {unchanged, flip_misaligned, null_no_partner, dv_drift_only}). History:
  ERRATA item 6 (the corrected defect: 0.45% positional misalignment,
  zero valid-set flips) and item 8 (retraction record of a misdiagnosis).
- Threshold: 30 kt / 24 h (Kaplan & DeMaria 2003), `labels.ri_threshold_kt_24h`.

## 7. Splits

- **By storm (SID)** — an event's split is its storm's split; no storm
  crosses splits (leakage-safe; tested in `tests/test_splits_stability.py`).
- Assignment = frozen override map first (`frozen_splits.json` — the
  historical benchmark wins), else sha256-hash of the SID into 70/15/15
  (train/val/test). Hash-deterministic: composition changes never move an
  existing storm.
- No label stratification (deliberate — stratification reintroduces
  composition dependence); class proportions per split are approximate.
- The frozen test split (2,679 events; 112 v2-positives, 6 NULL) is
  distributed for reproducibility. The property "never read during model
  development" belongs to THIS PROJECT'S HISTORY (documented in
  TECHNICAL_VALIDATION and the repository's provenance), not to the
  published dataset — once public, the test set is read. Benchmark
  comparability against this project's historical numbers therefore
  cannot be re-established by third parties and is not claimed.

## 8. `basin` semantics (post-repair; ERRATA item 7)

- Per-POINT IBTrACS basin code (`EP`, `NA`); 6 storms cross basins, so
  storm-level attribution uses the GENESIS point (578 EP / 414 NA).
- `"NA"` is the literal string "North Atlantic". **Read every CSV with
  `keep_default_na=False, na_values=[""]`** — pandas' default NA parsing
  destroys it (this bug class occurred 6 times in this project's history;
  see TECHNICAL_VALIDATION).

## 9. Known limitations (declared)

1. Seasonal coverage only (Jun–Nov); no off-season TCs.
2. East Pacific truncated at 140°W; two basins only.
3. Intensities are USA_WIND-centric (agency differences unrepresented).
4. ERA5 0.25° resolution cannot resolve inner-core structure (the reason
   hypothesis H7 was deferred).
5. 19 events carry NULL labels (undefined at storm end / gaps).
6. RI-classification skill on this data is modest (TECHNICAL_VALIDATION
   §7: baseline PR-AUC ≈ 0.25 vs ~4.8% base rate); the dataset's value is
   the auditable, leakage-safe construction, not a strong model. Model
   comparisons live in the repository's `BENCHMARK.md`, outside this
   descriptor.
7. Label misalignment history (v1) is fully documented and corrected
   (ERRATA 6/8; diff-manifest included).

## 10. Reconstruction recipe

`config-template.yaml` (includes the `pressure_levels` block — channels
13–14 are irreproducible without it) + `run.py`:
`prepare → download-era5 → preprocess → normalize`. Credentials are never
stored in config (see template header). ERA5 raws are not distributed
(discarded by design after verified extraction; window provenance
manifests included). Verification: the labeling chain is byte-reproducible
from raw IBTrACS — `analysis/dv24_impact_assessment_v5_raw_reference.py`
(abort-on-mismatch replication gate).
