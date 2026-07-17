# INDEX — map of every document (so nobody gets lost)

One line per artifact: what it is, and its status.
Tags: **[CURRENT]** live and authoritative · **[RECORD]** historical, kept
intact, do not edit · **[SUPERSEDED]** replaced, kept for traceability ·
**[PENDING]** waiting on an author action · **[LOCAL]** gitignored, not in
the public repo.

## Start here

| File | Status | What it is |
|---|---|---|
| [`docs/PROJECT_STATE.md`](PROJECT_STATE.md) | [CURRENT] | State-and-resume file. Read first in every session; updated at the end. |
| [`docs/INDEX.md`](INDEX.md) | [CURRENT] | This map. |

## The paper (V3) and citable identity

| File | Status | What it is |
|---|---|---|
| [`MANUSCRIPT_V3.md`](../MANUSCRIPT_V3.md) | [CURRENT] | **The paper.** Zenodo v3 PREPRINT, 10 sections; supersedes the whole record line; §9 = correction record. Awaits author sign-off → PDF → upload. |
| [`docs/cyclonenet_v3_preprint.tex`](cyclonenet_v3_preprint.tex) | [CURRENT] | Compilable LaTeX of the paper (pdflatex; needs `pipeline_figure.png` beside it — author adds). Carries the reference list and pipeline figure not yet in the .md. |
| [`docs/manuscript_v3_skeleton.md`](manuscript_v3_skeleton.md) | [CURRENT] | The paper's spec: canonical-numbers table, 8 global rules, superseded-claim inventory. Governs any future edit to the paper. |
| [`docs/data_descriptor_draft.md`](data_descriptor_draft.md) | [CURRENT] | Raw material for the FUTURE reduced Data Descriptor (journal submission, ESSD vs Scientific Data undecided). Not the paper. |
| [`MANUSCRIPT_honest.md`](../MANUSCRIPT_honest.md) | [RECORD] | The v2-era audit paper. Preserved intact (same policy as the v2.0.0 PDF on Zenodo). |
| [`CITATION.cff`](../CITATION.cff) / [`docs/CITATION.md`](CITATION.md) | [CURRENT] | Citable identity; carries the finalized title. |
| [`README.md`](../README.md) | [CURRENT] | Public face of the repo; v2 labels + retired-model framing (pass of 2026-07-16). |
| `cyclonenet_honest.tex` | [RECORD] | LaTeX of the v2-era paper. |

## Correction and verdict record (the honesty chain)

| File | Status | What it is |
|---|---|---|
| [`ERRATA.md`](../ERRATA.md) | [CURRENT] | Canonical correction record, items 1–9 (9 = v1.0.0/v1.0.1 claims, added 2026-07-16). |
| [`docs/hypothesis_registry.md`](hypothesis_registry.md) | [CURRENT] | Living research agenda; H1–H9 verdicts, ex-ante qualifications, scope guards. |
| [`BENCHMARK.md`](../BENCHMARK.md) | [CURRENT] | All numbers with their protocols explicitly separated (CNN test read vs GBM dev folds). |
| `outputs/results/dv24_impact/SUPERSEDED.md` | [RECORD] | Retraction trail of the false "Defect 0" diagnosis (reports v1–v4); v5 is authoritative. |

## Pre-registrations (do not edit; two are closed, one cancelled)

| File | Status | What it is |
|---|---|---|
| [`docs/ablation_preregistration.md`](ablation_preregistration.md) | [RECORD] | H6 (closed NULL). |
| [`docs/tabular_baseline_preregistration.md`](tabular_baseline_preregistration.md) | [RECORD] | H9 (closed: V1 negative / V2 null). |
| [`docs/fuelmap_ablation_preregistration.md`](fuelmap_ablation_preregistration.md) | [RECORD] | H8 (CANCELLED; harness kept, not to be run). |

## Dataset release (what ships)

| File | Status | What it is |
|---|---|---|
| [`docs/release/DATA_DICTIONARY.md`](release/DATA_DICTIONARY.md) | [CURRENT] | The dataset's dictionary (v2 labels). DOI slot pending. |
| [`docs/release/TECHNICAL_VALIDATION.md`](release/TECHNICAL_VALIDATION.md) | [CURRENT] | The 7 data-only validations (byte-reproducibility leads). |
| [`docs/release/NOTICE`](release/NOTICE) | [CURRENT] | Verbatim, primary-source-verified attributions (ERA5/Hersbach, IBTrACS/Knapp, Copernicus Marine). Data-source citations live HERE, not in the literature review — by design. |
| [`docs/release/zenodo_v3_metadata.md`](release/zenodo_v3_metadata.md) | [CURRENT] | Paste-ready Zenodo v3 record metadata (preprint reality: paper PDF + snapshot; v2 PDF not re-attached). |
| `dist/cyclonenet-dataset-v2-zips/` | [PENDING] | The built package: 46 files / 6.03 GiB, verified. Waiting for the author's Zenodo upload (gitignored). |
| [`LICENSE`](../LICENSE) / [`LICENSE-DATA`](../LICENSE-DATA) | [CURRENT] | Code MIT / dataset CC BY 4.0. Both already on `main` (PR #15). |
| [`docs/dataset_release_plan.md`](dataset_release_plan.md) | [RECORD] | T5 planning document. |
| [`docs/data_dictionary_v2_skeleton.md`](data_dictionary_v2_skeleton.md) | [SUPERSEDED] | Working skeleton of the dictionary; the release version above replaced it. |

## Science and analysis docs

| File | Status | What it is |
|---|---|---|
| [`docs/literature_review.md`](literature_review.md) | [CURRENT] | The public review; entries 1–15 (15 = Kapoor & Narayanan, the only FULL-TEXT). The paper cites nothing this file does not list. |
| [`docs/DATASET.md`](DATASET.md) | [CURRENT, stale ranges] | Technical dataset spec; still carries v1-era "1989–2024" ranges — hygiene queue. |
| [`docs/fuelmap_validation.md`](fuelmap_validation.md) | [RECORD] | The external TCHP validation (negative result), full protocol. |
| [`docs/ri_precursors.md`](ri_precursors.md) | [RECORD] | Matched-pairs precursor tests (registry H4/H5 mapping). |
| [`docs/dv24_label_correction_design.md`](dv24_label_correction_design.md) | [RECORD] | Design of the v1→v2 label correction. |
| [`docs/INTERPRETATION.md`](INTERPRETATION.md) · [`docs/CASE_STUDIES.md`](CASE_STUDIES.md) · [`docs/diagnostic.md`](diagnostic.md) · [`docs/scalar_branch_design.md`](scalar_branch_design.md) · [`docs/ablation_progress.md`](ablation_progress.md) | [RECORD] | Supporting analyses from earlier phases. |

## Key scripts (the load-bearing ones)

| File | What it is |
|---|---|
| `analysis/dv24_impact_assessment_v5_raw_reference.py` | The **replication gate**: byte-exact reconstruction from raw IBTrACS, abort-on-mismatch. Permanent project rule (PROJECT_STATE §3). |
| `analysis/package_dataset_release.py` | Built the T5 package (manifest-first, canonical-number asserts). |
| `analysis/repair_basin_metadata.py` | The basin repair (ERRATA item 7). |
| `src/utils/splits.py` + `data/normalized/frozen_splits.json` | Hash-deterministic SID splits + frozen override map (inviolable). |
| `platform/build/build_events.py` | Platform build; reads `ri_label` from the event list (no independent RI logic). |

## External links

| Link | Status | What it is |
|---|---|---|
| https://github.com/estefano-ferreira/cyclone-net | [CURRENT] | The repo. `main` = public face; `feature/tchp` = working branch. |
| https://estefano-ferreira.github.io/cyclone-net/ | [CURRENT] | Platform explorer (observed data only, tri-state RI markers). |
| https://zenodo.org/records/18571958 | [RECORD, superseded] | v1.0.0 preprint (claims corrected by ERRATA item 9 / paper §9). |
| https://zenodo.org/records/18577056 | [RECORD, superseded] | v1.0.1 preprint (same). |
| https://zenodo.org/records/18751255 | [RECORD, superseded] | v2.0.0 preprint — current `/latest` of the line until v3 goes up. |
| Zenodo v3 software record | [PENDING] | Author upload: compiled paper PDF + repo snapshot tag `v3.0.0`. DOI slot in the paper. |
| Zenodo dataset record | [PENDING] | Author upload: the 46 files in `dist/`; mints the dataset concept DOI. |

## Local-only (gitignored — never commit)

`CLAUDE.md` · `.claude/` (TODO_recomendacoes, WORKFLOW_DELEGACAO, research
sweeps incl. VEREDITO_FINAL) · `.mcp.json` — assistant-environment files,
not project artifacts.
