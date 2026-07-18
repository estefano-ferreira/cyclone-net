# Zenodo concept line 10.5281/zenodo.18571957 — v3.0.2 metadata (paste-ready)

REWORKED 2026-07-18 for the single-record reality: v3.0.1
(10.5281/zenodo.21413397, published 2026-07-18) joined the existing
concept line as a SINGLE archived record (paper + dataset package).
v3.0.2 is a file-version pass on that record: corrected paper PDF +
code snapshot added; the 46 dataset files are unchanged.
No slots remain — all DOIs below are resolved (conceptdoi read from the
Zenodo API for record 21413397).

## Title — DECIDED (author, 2026-07-16, final revision):

> **CycloneNet: A Reproducible Pipeline and Leakage-Safe Two-Basin Dataset
> for Tropical-Cyclone Rapid-Intensification Analysis**

(Names what the tool IS — pipeline + dataset, both adjectives verified
properties. Matches CITATION.cff / docs/CITATION.md — one public name.
The earlier "…Honest Audit…" variant was retired before any public use;
the audit framing lives in the abstract/description, not the title.)

## Description (paste in full — replaces the published v3.0.1 text, which
## still carries the pre-correction v1 wording)

> CycloneNet is an open, configuration-driven pipeline for retrospective
> (hindcast) analysis of tropical-cyclone rapid intensification (RI) from
> ERA5 reanalysis and IBTrACS best tracks, released together with a
> leakage-safe two-basin dataset (East Pacific + North Atlantic,
> 1980–2023 hurricane seasons: 16,780 events, 992 storms — 799 RI
> positives, 15,962 negatives, 19 NULL labels — under strict-temporal v2
> labeling with full v1→v2 provenance).
>
> This record is a full preprint plus the complete dataset package, and
> SUPERSEDES the entire record line (v1.0.0, v1.0.1, v2.0.0 — each
> preserved permanently by its version-DOI). Section 9 of the paper is
> the correction record for the whole line: the v1 records' headline
> claims (ROC-AUC 0.97, "sub-pixel spatial accuracy", ~26 km mean spatial
> error) reproduce exactly from the records' own validation artifacts but
> were measured on a 58-event named-storm benchmark at 44.8% RI
> prevalence and are withdrawn as scope overclaim — not comparable to
> full-archive evaluation and not to be cited as measures of model skill;
> energy-source localization via the learned FuelMap was externally
> validated against TCHP and does not beat a storm-centre baseline
> (p = 0.30, n = 226); the v2.0.0 headline metrics (ROC-AUC 0.83) were
> never reproducible and are replaced by reproducible ones (ROC-AUC 0.796
> [0.753–0.837], PR-AUC 0.251 [0.179–0.331] on the frozen 1980–2023 test
> split); a pre-registered campaign (2026-07-16) found no detectable
> contribution from added pressure-level channels (H6 null) and retired
> the CNN architecture (H9: a gradient-boosted tabular baseline beats it,
> Δ₁ = −0.078 [−0.116, −0.042]; on the same underlying fields the
> architecture adds nothing over aggregated statistics, Δ₂ = +0.0005
> [−0.029, +0.032]). The "physics-guided" label is retired. The project
> designates no reference model; the validated contribution is the
> auditable dataset and pipeline and the documented negative results.
>
> Version notes (3.0.2): replaces the v3.0.1 paper PDF, which printed
> unresolved DOI placeholders and carried five discussion-consistency
> defects (all fixed in this compile), and adds the repository code
> snapshot promised in the availability statement. The 46 dataset files
> are unchanged from v3.0.1 (identical checksums).
>
> Code: MIT (LICENSE inside the snapshot). Dataset: CC BY 4.0
> (Copernicus/ERA5 and IBTrACS attributions mandatory — see NOTICE).
> Repository: https://github.com/estefano-ferreira/cyclone-net
> (ERRATA.md, BENCHMARK.md, docs/hypothesis_registry.md).

## Keywords

REMOVE: `physics-guided`, `Spatio-Temporal Attention` (experimental module
never integrated), `Atmospheric Singularity Mapping` (if present).

KEEP/SET: `tropical cyclone` · `rapid intensification` · `ERA5` ·
`IBTrACS` · `reanalysis` · `machine learning` · `reproducibility` ·
`pre-registration` · `negative results` · `open dataset` ·
`North Atlantic` · `East Pacific`

## License (record-level field)

Zenodo takes ONE license for the record. This record's citable content is
the preprint PDF + the dataset package + the repository snapshot:

- **Record license field: CC BY 4.0** — governs the paper, the dataset
  package, and the documentation (the record's citable substance).
- The repository snapshot inside the record carries its own `LICENSE`
  (MIT, code) and `LICENSE-DATA` (CC BY 4.0, dataset) — the description
  states which applies to which, so the file-level licenses are
  discoverable and unambiguous.

## Version / dates

- Version: **3.0.2**
- Publication date: date of the v3.0.2 publish.
- DOIs: concept 10.5281/zenodo.18571957 (resolves to latest; the paper
  cites ONLY this one); version DOI minted at publish — record it in
  CITATION.cff / LICENSE-DATA / docs/CITATION.md / DATA_DICTIONARY §1
  after publish (the repo docs are the updatable layer; the PDF is not).
- Related identifiers: keep the existing GitHub relation. No
  `isSupplementedBy` — the dataset lives in THIS record (single-record
  model; the separate-dataset-record plan was retired 2026-07-17).

## Files in the record (v3.0.2)

1. `cyclonenet_v3_preprint.pdf` (compiled on Overleaf from
   `docs/cyclonenet_v3_preprint.tex`; requires `pipeline_figure.png`
   beside the .tex) — FIRST file, so the record preview opens on it.
   Replaces the v3.0.1 PDF. Section 9 carries the correction record for
   the whole line.
2. The 46 dataset files retained unchanged from v3.0.1: 44
   `cyclonenet-dataset-v2-cubes-<year>.zip` (1980–2023) +
   `cyclonenet-dataset-v2-metadata.zip` + `ZIP_CHECKSUMS.sha256`.
3. `cyclonenet-code-v3.0.2.zip` — repository snapshot at tag `v3.0.2`
   with LICENSE + LICENSE-DATA (NEW in this version; regenerate from the
   tag AFTER the paper-fix commit is tagged, and verify no local-only
   files inside).

(The v2.0.0 manuscript PDF is NOT re-attached: its version-DOI
10.5281/zenodo.18751255 preserves that record permanently — that is what
versioning does. The standalone supersession note was retired; its
content is Section 9 of the paper. Known cosmetic issue carried by the
retained files: `ZIP_CHECKSUMS.sha256` and the `CHECKSUMS.sha256` inside
the metadata zip use CRLF line endings, so a naive `sha256sum -c` fails
on Linux until the `\r` is stripped — documented, fix deferred to a
future file version to avoid churning the retained files' checksums.
Same policy for `DATA_DICTIONARY.md`: the copy inside the metadata zip is
the v3.0.1 snapshot with two then-unresolved packaging slots; the resolved
version (rejection-reason counts, T-axis order confirmed 2026-07-18) ships
inside the code snapshot and in the repo, which the dictionary header
declares authoritative.)
