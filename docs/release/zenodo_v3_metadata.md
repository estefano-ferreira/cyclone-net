# Zenodo record 10.5281/zenodo.18751255 — v3.0.0 metadata (paste-ready)

DRAFT FOR AUTHOR REVIEW — 2026-07-16. Slots marked ⟦…⟧ wait for the
dataset DOI.

## Title — DECIDED (author, 2026-07-16, final revision):

> **CycloneNet: A Reproducible Pipeline and Leakage-Safe Two-Basin Dataset
> for Tropical-Cyclone Rapid-Intensification Analysis**

(Names what the tool IS — pipeline + dataset, both adjectives verified
properties. Matches CITATION.cff / docs/CITATION.md — one public name.
The earlier "…Honest Audit…" variant was retired before any public use;
the audit framing lives in the abstract/description, not the title.)

## Description (rewrite — replaces the v2 text in full)

> CycloneNet is an open, configuration-driven pipeline for retrospective
> (hindcast) analysis of tropical-cyclone rapid intensification (RI) from
> ERA5 reanalysis and IBTrACS best tracks, released together with a
> leakage-safe two-basin dataset (East Pacific + North Atlantic,
> 1980–2023 hurricane seasons: 16,780 events, 992 storms — 799 RI
> positives, 15,962 negatives, 19 NULL labels — under strict-temporal v2
> labeling with full v1→v2 provenance).
>
> This version (3.0.0) opens with a SUPERSESSION NOTE. The v2.0.0
> manuscript is preserved unmodified below it as historical record; its
> central claims were since tested and refuted or superseded:
> energy-source localization via the learned FuelMap was externally
> validated against TCHP and does not beat a storm-centre baseline
> (p = 0.30, n = 226); the headline metrics (ROC-AUC 0.83) were never
> reproducible and are replaced by reproducible ones (ROC-AUC 0.796
> [0.753–0.837], PR-AUC 0.251 [0.179–0.331] on the frozen 1980–2023 test
> split); a pre-registered campaign (2026-07-16) found no detectable
> contribution from added pressure-level channels (H6 null) and retired
> the CNN architecture (H9: a gradient-boosted tabular baseline beats it,
> Δ₁ = −0.078 [−0.116, −0.042]; at a fixed information diet the
> architecture adds nothing, Δ₂ = +0.0005 [−0.029, +0.032]). The
> "physics-guided" label is retired. The project designates no reference
> model; the validated contribution is the auditable dataset and pipeline
> and the documented negative results.
>
> Code: MIT. Dataset: CC BY 4.0 (Copernicus/ERA5 and IBTrACS attributions
> mandatory — see NOTICE). Dataset record: ⟦dataset DOI when minted⟧.
> Repository: https://github.com/estefano-ferreira/cyclone-net (ERRATA.md,
> BENCHMARK.md, docs/hypothesis_registry.md).

## Keywords

REMOVE: `physics-guided`, `Spatio-Temporal Attention` (experimental module
never integrated), `Atmospheric Singularity Mapping` (if present).

KEEP/SET: `tropical cyclone` · `rapid intensification` · `ERA5` ·
`IBTrACS` · `reanalysis` · `machine learning` · `reproducibility` ·
`pre-registration` · `negative results` · `open dataset` ·
`North Atlantic` · `East Pacific`

## License (record-level field)

Zenodo takes ONE license for the record. This record's citable content is
the supersession note + manuscript PDF + repository snapshot:

- **Record license field: CC BY 4.0** — governs the note, the PDF and the
  documentation (the record's citable substance).
- The repository snapshot inside the record carries its own `LICENSE`
  (MIT, code) and `LICENSE-DATA` (CC BY 4.0, dataset) — the description
  states which applies to which, so the file-level licenses are
  discoverable and unambiguous.

(Rationale: the record is primarily a publication artifact; MIT at
record level would misstate the PDF/note. This mirrors the repo's
layer-A dual-license pass.)

## Version / dates

- Version: **3.0.0**
- Publication date: date of the v3 upload.
- Related identifiers:
  - `isSupplementedBy` → ⟦dataset concept DOI, when minted⟧
  - (keep the existing GitHub relation)

## Files in the record (proposed)

1. `cyclonenet_v3_supersession_note.pdf` (compiled from
   `docs/release/zenodo_v3_supersession_note.tex`) — FIRST file, so the
   record preview opens on it.
2. The v2.0.0 manuscript PDF — unmodified, historical record.
3. Repository snapshot (tag `v3.0.0`) with LICENSE + LICENSE-DATA.
