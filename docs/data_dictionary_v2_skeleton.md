# DATA_DICTIONARY v2 — skeleton (T5 FASE 1, field-by-field TODO)

Status: SKELETON for author review — every `⟦…⟧` is a slot to fill from the
verified source named in it. Not packaged, not committed to release yet.
Facts below marked ✔ are already verified this session.

## 1. Dataset identity

- Name, version (v2 labels), DOI ⟦after repository decision⟧, license
  **CC BY 4.0** ✔ (Copernicus attribution non-strippable), NOTICE
  (Copernicus text ⟦exact wording pending verification⟧ + IBTrACS citation
  + AOML TCHP if ADT ships).
- **Temporal coverage: 1980–2023, JUNE–NOVEMBER ONLY** ✔ — the download
  window is the hurricane season (`download.months: [6..11]`). "1980–2023"
  does NOT mean continuous coverage; off-season TCs are absent. Real
  product limitation, declared up front.
- Spatial coverage: bbox [60N, 140W, 0N, 20W] ✔ — two-basin sector
  (EP truncated west of 140°W ✔ — declared limitation).

## 2. Population

- 16,780 valid events / 992 storms (578 EP / 414 NA by genesis) ✔;
  **799 RI positives, 19 NULL labels (v2)** ✔.
- `rejected_events.csv` ships for composition transparency (counts:
  ⟦from normalization_report.json⟧; cubes not included).
- Events are 6-hourly track points with a 24 h ERA5 history window.

## 3. Per-event files

- `{event_id}.npy` — float32 cube **(H=40, W=40, T=5, C=14)** ✔; axis
  order ⟦verify H,W order vs lat/lon grids in code⟧; T = t0,−6h,…,−24h.
- `{event_id}.json` — metadata sidecar; field-by-field table (§5).
- `{event_id}_lats.npy` / `_lons.npy` — 40-point grids.
- `{event_id}_fuel_potential.npy` — heuristic prior; **H1 semantics
  REFUTED** ✔ — must carry the caveat verbatim ⟦from ERRATA item 4⟧.
  DECISION pending: ship or omit.
- `{event_id}_adt.npy` — 761 valid events ✔ (2020–2023 subset); optional
  ocean channel, never a released-model input claim.

## 4. Channels (14) — cube order ✔ verified against sidecars

| # | name | units | source | model input? |
|---|------|-------|--------|--------------|
| 1 | sst_K | K | ERA5 | yes |
| 2 | mslp_Pa | Pa | ERA5 | yes |
| 3 | u10_mps | m/s | ERA5 | yes |
| 4 | v10_mps | m/s | ERA5 | yes |
| 5 | wind_mps | m/s | derived | yes |
| 6 | vort_1ps | 1/s | derived | yes |
| 7 | div_1ps | 1/s | derived | yes |
| 8 | grad_mslp_Pa_per_m | Pa/m | derived | yes |
| 9 | sst_anom_K | K | derived | yes |
| 10 | latent_heat_flux_Wpm2 | W/m² | derived | **no** (stored only) |
| 11 | sensible_heat_flux_Wpm2 | W/m² | derived | **no** (stored only) |
| 12 | total_heat_flux_Wpm2 | W/m² | derived | **no** — `exclude_total_heat_flux_from_input: true` (anti-leakage) ✔ |
| 13 | shear_850_200_mps | m/s | ERA5 pressure levels [850,200] ✔ | H6-tested (NULL) |
| 14 | rh_mid | ⟦unit from sidecar `units`⟧ | ERA5 PL [700,600,500] ✔ | H6-tested (NULL) |

**14 stored vs 10 consumed** ✔ (9 named + ADT when enabled): fluxes exist
and are NOT inputs — anti-leakage design, declared explicitly.

## 5. Sidecar fields (from a real sidecar ✔; each needs a one-line def)

`event_id, sid, timestamp, storm_name, basin, center_lat, center_lon,
wind_kt, pressure_mb, dv12_kt, dv24_kt, ri_label, channels, cube_shape,
units, timestamps, era5_selected_times, era5_time_indices, era5_time_name,
qc_flags ⟦glossary from preprocess code⟧, temporal_integrity_ok,
adt_saved, fuel_potential_saved, source_files`

Key semantics (already settled):
- **`ri_label ∈ {0, 1, NULL}`** ✔ — NULL (JSON `null`, CSV empty) = no
  exact same-storm partner at t0+24 h; the RI task view EXCLUDES the 19
  NULL events; never coerce NULL to 0. Strict-temporal v2 (ERRATA 6).
- **`dv12_kt`/`dv24_kt`** ✔ — strict-temporal deltas, nullable.
- **`basin`** ✔ — PER-POINT IBTrACS basin, post-repair (ERRATA 7):
  6 storms cross basins; storm-level attribution uses the GENESIS point
  (578 EP / 414 NA); "NA" is a literal string — readers MUST use
  `keep_default_na=False`.
- **`wind_kt`** — USA_WIND (source declared; agencies differ).

## 6. Labels file + provenance

- `event_list_augmented.csv` (32,989 rows) — column table ⟦full list ✔
  minus the dropped `wind_kt_shift_*`⟧.
- `label_diff_v1_v2.csv` ✔ — v1→v2 per-row provenance (reason ∈
  {unchanged, flip_misaligned, null_no_partner, dv_drift_only}).
- Provenance manifests: 22 processing windows + basin repair + label
  correction ✔ (md5 pre/post).

## 7. Splits

- By SID, hash-deterministic + frozen override map
  (`frozen_splits.json` wins) ✔; 70/15/15; no storm crosses splits
  (tested). Frozen test set: 2,679 events / 112 v2-positives ⟦recount at
  packaging: 115 v1 → −3 NULL⟧ — declared but NOT to be consumed for
  model development.

## 8. Known limitations (declared, not hidden)

- Seasonal coverage only (Jun–Nov) ✔; EP truncated at 140°W ✔;
  two basins only; USA_WIND-centric intensities; ERA5 0.25° resolution
  limits inner-core structure (H7 deferred for this reason) ✔; label
  UNDEFINED at storm ends (19 valid NULLs) ✔; fuel-potential prior is a
  refuted-hypothesis artifact (if shipped).
- Silent-default sweep findings (T5 FASE 1 item 1) ⟦link/summary of the
  final classification table⟧.

## 9. Reconstruction recipe

- `config-template.yaml` ⟦AFTER the reported diff is applied: add
  pressure_levels block; credential-field handling decision⟧ + `run.py`
  command sequence (prepare → download → preprocess → normalize) +
  raw-replication gate script as verification
  (`analysis/dv24_impact_assessment_v5_raw_reference.py`) ✔.
- ERA5 raws are NOT included (discarded by design after verified
  extraction; documented in the windowed-processing provenance) ✔.
