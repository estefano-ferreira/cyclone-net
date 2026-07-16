# CycloneNet Dataset v2 — Technical Validation

DRAFT FOR AUTHOR REVIEW (T5 phase 2, 2026-07-16). Evidence only — every
claim below is backed by a committed artifact, script, or manifest in the
repository. No interpretation beyond what the evidence supports.

## 1. Labeling pipeline is byte-reproducible from the raw source

Reconstruction of the event list from the raw IBTrACS file
(`ibtracs.ALL.list.v04r00.csv`), replicating the builder chain (filters,
synoptic-hour selection, per-storm labeling, target dropna), reproduces the
shipped file **byte-for-byte**: 32,989/32,989 rows, `dv24_kt` and
`ri_label` identical on every row. Script:
`analysis/dv24_impact_assessment_v5_raw_reference.py` (aborts unless
replication is exact — the project's permanent "replication gate").

## 2. Label correction v1→v2, fully audited

The v1 labels were positional (row-shift) rather than strictly temporal.
Measured against raw IBTrACS on the full population: 148/32,989 rows
(0.45%) misaligned; **zero label flips in the released valid set**; 19
events reclassified NULL (undefined); positives 802→799. Corrected at the
origin and in the stored artifacts with per-file verification; per-row
provenance in `label_diff_v1_v2.csv`; md5 pre/post in
`provenance/dv24_label_correction_20260716_175443.json`. An intermediate
misdiagnosis during the assessment was retracted and is documented with
its cause (wrong-reference analysis on a derived artifact) — ERRATA item 8.

## 3. Split integrity (leakage safety)

Storm-level (SID) assignment, sha256-hash-deterministic with a frozen
override map; no storm crosses splits. Property-tested
(`tests/test_splits_stability.py`): assignments are invariant to dataset
composition; the frozen map preserves the historical benchmark. Events
that cannot be assigned fail loudly (no silent exclusion).

## 4. Pressure-level completeness census

After the 1980–2019 pressure-level backfill (21,662 events processed, zero
failures, per-window provenance manifests), an independent census verified
**100% coverage of the development set: 14,101/14,101 events** carry both
pressure-level channels (`outputs/results/pl_gate_census.json`; gate
consumed by every subsequent experiment run).

## 5. Processing provenance

Raw ERA5 monthly files (hundreds of GB for 44 seasons at 0.25°, far
exceeding sustainable local storage) are not part of this release; they
were discarded only through a windowed process in which every extraction
is verified BEFORE deletion, with a provenance manifest per window (22
windows: counts, checksums). The raws remain re-downloadable from the CDS
with the included configuration. Basin metadata was audited and repaired at
the origin in 2026-07 (a pandas NA-parsing bug had blanked the literal
"NA" basin code), with audit-exact verification (per-point
8,888 EP / 7,892 NA; 992 storms) — ERRATA item 7. A subsequent class-wide
sweep closed the same bug pattern in 7 further readers (including one that
had been masked by a compensating workaround) and added regression tests
(`tests/test_na_handling_readers.py`).

## 6. Quality control

Per-event QC at extraction (physical ranges, NaN budgets, temporal and
geospatial integrity; flags stored in each sidecar). Rejected events are
listed in `rejected_events.csv` and their cubes excluded. Normalization
statistics are computed on the training split only
(`normalization_stats.json`).

## 7. Usability

A gradient-boosted baseline over tabular features derived from the cubes
attains PR-AUC ≈ 0.25 under grouped k-fold by storm, against a positive
base rate of ~4.8%. Reported solely as evidence that the dataset supports
supervised learning. Model comparisons, hypothesis tests and their
verdicts are outside the scope of this descriptor; see the repository's
`BENCHMARK.md`.
