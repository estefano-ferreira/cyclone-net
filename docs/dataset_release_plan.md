# T5 — Dataset release plan (packaging WITH cubes)

Status: PLAN FOR AUTHOR REVIEW. Prepared 2026-07-16, after the v2 label
correction (`3db2cb8`). Companion to `docs/dv24_label_correction_design.md`.

## 1. What ships (the package)

A CycloneNet dataset release, **v2 labels**, valid events only:

| Component | Source | ~Size |
|---|---|---|
| Per-event cubes `{event_id}.npy` (40×40×5×14 float32) | `data/interim/` (valid events) | ~7.2 GB |
| Per-event metadata sidecars `{event_id}.json` | `data/interim/` | small |
| ADT extras (`*_adt.npy`, where present) | `data/interim/` | ~0.5 GB |
| Fuel-potential priors `*_fuel_potential.npy` | `data/interim/` | DECISION 4 |
| `event_list_augmented.csv` (32,989 rows, v2 labels, NULL semantics) | `data/` | 4 MB |
| `valid_events.csv`, `splits.csv`, `frozen_splits.json`, `normalization_stats.json` | `data/normalized/` | small |
| `rejected_events.csv` (composition transparency; no cubes) | `data/normalized/` | small |
| **`label_diff_v1_v2.csv`** (v1→v2 provenance; approved: ships) | `data/normalized/` | 3 MB |
| `DATA_DICTIONARY.md` (rewritten from `docs/DATASET.md` — see §4) | new | — |
| `LICENSE` (CC BY 4.0) + `NOTICE` (attributions, §3) | new | — |
| `CHECKSUMS.sha256` + `package_manifest.json` (per-file sha256, counts, provenance refs) | generated | — |
| `TECHNICAL_VALIDATION.md` (§5) | new | — |

Canonical counts the manifest must assert: 16,780 valid events / 992 storms
(578 EP / 414 NA) / **799 RI positives, 19 NULL labels** (v2) / splits
70-15-15 by SID, frozen.

## 2. Repository + versioning (DECISION 1)

**Recommendation: Zenodo, versioned record (concept DOI + version DOI).**
- Immediate (PANGAEA curation takes weeks); 50 GB/record is 6× our size.
- ESSD — the leading venue candidate (€1,400 flat) — accepts Zenodo-hosted
  datasets; the ESSD living-data process expects exactly this
  concept-DOI/version-DOI structure (verified 16/07 with primary sources).
- Publish as a NEW versioned record under the existing CycloneNet concept
  DOI (10.5281/zenodo.18751255) or as a separate dataset-only concept DOI —
  **sub-decision 1b: separate dataset DOI recommended** (the Data Descriptor
  cites the dataset, not the software; Force11 P7 wants version-specific
  citation of the DATA).
- Coordinated with the Zenodo re-publication of the software/paper record
  (the public copy still shows 0.347 + refuted framing): the public face
  changes ONCE, same day: dataset record (new) + software record (updated)
  + README/About/CITATION.cff.

## 3. Licensing + attribution (settled 16/07, verified)

- Dataset license: **CC BY 4.0** (CC0 eliminated — Copernicus attribution
  cannot be stripped).
- `NOTICE` file text:
  - "Contains modified Copernicus Climate Change Service information
    (1980–2023). Neither the European Commission nor ECMWF is responsible
    for any use that may be made of the Copernicus information or data it
    contains."
  - IBTrACS v04r00 citation (Knapp et al.; NOAA NCEI, public domain).
  - TCHP (NOAA/AOML) if ADT/TCHP extras ship.
- Code license is a SEPARATE, still-open author decision (repo is
  CC-BY-NC-4.0; MIT/Apache is the data-paper norm for pipeline
  reproduction) — NOT a blocker for the dataset package, but should be
  resolved before the Data Descriptor submission.

## 4. DATA_DICTIONARY — rewrite, not copy (docs/DATASET.md is stale)

Verified stale facts that MUST be fixed in the release copy:
- Period: says "1989–2024" in two places → **1980–2023**.
- Channels table lists 9 → cubes have **14** (verified against sidecars):
  the 9 documented + `latent_heat_flux_Wpm2`, `sensible_heat_flux_Wpm2`,
  `total_heat_flux_Wpm2` (stored, NOT model inputs — leakage note stands)
  + `shear_850_200_mps`, `rh_mid` (pressure-level backfill).
- RI label: says binary 0/1 with positional 6-h steps → **v2
  strict-temporal semantics, `ri_label ∈ {0,1,NULL}`**; NULL = no exact
  same-storm partner at t0+24h; the RI task view excludes NULL (19 events).
  dv12/dv24 same semantics. Reference: ERRATA items 6/8 + diff-manifest.
- Splits: add the frozen-map + hash-assignment description (currently only
  "fixed and stored").
- Add: file naming scheme, cube axis order, dtype, per-channel units (from
  sidecar `units`), QC flags glossary, provenance-manifest pointers.

## 5. TECHNICAL_VALIDATION.md — content list (evidence only, no claims)

1. **Byte-exact replication of the event list from raw IBTrACS**
   (32,989/32,989; ERRATA technical-validation note) — labeling pipeline
   reproducible from source.
2. v2 label correction: measured defect (0.45%), zero valid-set flips,
   diff-manifest + provenance manifest (md5 pre/post per file).
3. PL census gate PASS (14,101 dev events, 100% coverage).
4. Provenance manifests for all 22 processing windows (raw discarded only
   after verified extraction).
5. Split integrity: SID-hash determinism + frozen map tests
   (`tests/test_splits_stability.py`); no storm crosses splits.
6. Basin metadata repair (ERRATA item 7) — audit-exact verification.
7. NaN policy: QC gates + NULL label semantics.
8. Usability evidence ONLY (no interpretation, per the no-reference-model
   decision): GBM_SF pooled OOF PR-AUC 0.249 on dev folds; CNN test-set
   0.251 [0.179–0.331] as historical record of a retired architecture,
   protocols explicitly separated (BENCHMARK.md framing).

## 6. Public-face layer A (same day, one change)

- `CITATION.cff`: title drops "Physics-Guided" (retired label); abstract
  numbers → v2 (799 positives; 0.251 unchanged as historical test-set
  record); dataset DOI added alongside software DOI.
- `README.md`: dataset section → v2 numbers + release pointer.
- Zenodo software record: updated description (current verdicts, honest
  framing); Zenodo dataset record: new.
- GitHub About: author manual action (wording already in
  `.claude/TODO_recomendacoes.md`).
- Rename `outputs/results/dv24_impact/label_diff_v1_v2_dryrun.csv` →
  drop the misleading "dryrun" (author note 16/07); update SUPERSEDED.md
  references accordingly.

## 7. Packaging mechanics

- Layout: `cyclonenet-dataset-v2/` with `cubes/` sharded **per year**
  (`cubes/1980/…`), metadata alongside; zip archives per shard bundle
  (Zenodo-friendly download) + top-level small-files zip. DECISION 3
  confirms the sharding unit.
- The packaging script builds from a MANIFEST-FIRST flow: enumerate →
  verify (counts vs canonical numbers; every valid event has cube+sidecar;
  sha256 each file) → copy to staging → re-verify staged hashes → zip →
  hash the zips. Any mismatch → abort. Nothing under `data/` is modified
  (read-only source).
- **Sidecar rewrite at staging (author decision 16/07):** in the STAGED
  copies only, `fuel_potential_saved` is set to `false` (the priors are
  not distributed — the distributed sidecar must describe the distributed
  package). The divergence from local sidecars is recorded in
  `package_manifest.json` with the reason. No other sidecar field is
  touched; local `data/interim/` is never modified.
- Deterministic: fixed file ordering, stored mtimes normalized, so
  re-packaging reproduces identical zips (hash-stable) where the tooling
  allows.
- Authorship: given today's executor record, the packaging script is
  provenance-critical → **Fable writes it** (DECISION 5 to confirm);
  mechanical doc conversions (data dictionary table fill-in from sidecar
  `units`) can still go to Haiku with named prohibitions.

## 8. Decisions — CLOSED BY THE AUTHOR (2026-07-16)

1. **Repository: Zenodo**, versioned record; **separate dataset
   concept-DOI** (not a version of the software DOI). Zenodo's 100-file
   cap per record makes sharding mandatory regardless.
2. **Rejected events: cubes OUT; `rejected_events.csv` ships** for
   composition transparency.
3. **Shard unit: per-year zips** (~44 shards + 1 metadata zip ≤ 100-file
   cap).
4. **Fuel-potential priors: OUT** (−0.50 GiB → package ≈ 7.2 GiB raw /
   ~6.2 GiB compressed). The refuted-hypothesis trail stays documented in
   ERRATA/registry, not shipped as arrays.
5. **Code license: MIT** (dual with dataset CC BY 4.0). APPLIED IN THE
   COORDINATED LAYER-A PASS (one public-face change: LICENSE,
   CITATION.cff `license:` field, README) — not piecemeal.
6. Packaging-script authorship: Fable direct.

## 9. Sequencing (after decisions)

1. DATA_DICTIONARY + TECHNICAL_VALIDATION + NOTICE/LICENSE drafts → author
   review.
2. Packaging script + dry-run manifest (counts/hashes, no copies) → author
   review.
3. Staging build + verification → spot-check.
4. CITATION.cff/README/rename pass (layer A) — single commit.
5. Author (manual): Zenodo upload(s), re-publication, GitHub About.
