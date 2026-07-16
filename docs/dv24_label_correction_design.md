# dv24/dv12 label correction — design (T-dv24.2)

Status: DESIGN FOR AUTHOR REVIEW — no implementation authorized yet.
Author decisions this design implements (re-confirmed 2026-07-16 on the v5
numbers; earlier targets dictated on the retracted v1–v4 reports are void):

1. Option (b): correct the labels and publish only the corrected dataset
   (v2). Grounds: clean strict-temporal semantics in a Data Descriptor, at
   0.11% cost on the valid set. (The earlier anti-(a) argument — "labels
   with cross-storm leakage" — was retracted; no leakage exists.)
2. `ri_label ∈ {0, 1, NULL}`: undefined means NULL, never 0; NULL events
   REMAIN in the dataset; the RI-classification task view excludes them.
3. Versioned diff-manifest v1→v2 + frozen v1 label column as provenance.
4. The CNN test-set metric PR-AUC 0.251 STAYS in BENCHMARK (retired
   architecture, separate protocol): v5 proved the v1 labels are
   byte-reproducible from raw IBTrACS — legitimate historical record.
5. H6/H9 need NO v1 anchoring: zero label flips and dev PL-gated positives
   unchanged (687→687) make the verdicts immune by construction.

Authoritative numbers: `outputs/results/dv24_impact/report_v5_20260716_152525.json`
(script `analysis/dv24_impact_assessment_v5_raw_reference.py`). Reports
v1–v4 in the same directory are RETRACTED (see `SUPERSEDED.md`).

---

## 1. The defect being corrected (true inventory, v5)

`label_ri`/`add_wind_deltas` (src/processors/ri_labeling.py:12,21-22) compute
dv12/dv24 as positional per-SID shifts (−2/−4 rows) assuming a perfect 6-h
grid. The builder (src/processors/ibtracs.py:224-233) applies them to the
filtered pre-dropna IBTrACS series and then drops rows lacking either
target. Where the surviving series is not 6-h-regular (odd-minute IBTrACS
entries that pass the hour filter, bbox exits, interior missing wind), the
positional partner is not at t0+24h/t0+12h:

- Event list (32,989 rows): 148 rows misaligned for dv24 (0.45%), 77 for
  dv12. Label flips: exactly 1 (1→0), zero (0→1). Strictly-undefined
  (no exact temporal partner): 55 rows (3 positives). dv24 value drift with
  a valid partner: 67 rows; dv12: 37; dv12→NULL: 31.
- Valid set (16,780): ZERO label flips; 19 events → NULL (11 train / 2 val /
  6 test); positives 802→799 (−3, all test); 1 storm (test) loses its
  positives; dv12: 3 value changes + 8 NULL (5 train / 1 val / 2 test).
- Dev PL-gated: positives 687→687 — UNCHANGED.
- Rows currently dropped by the builder that strict-temporal labeling could
  define: 2 (0 positives). They do NOT enter v2 — no cubes exist for them
  (ERA5 raws discarded) and the released row set is frozen (see §4
  invariants).

What this correction is NOT: there is no cross-storm leakage, no phantom
positives, no mass NaN→0 coercion in the shipped data (all retracted; the
builder's dropna removes undefined rows, so the `NaN >= 30 → 0` expression
in `ri_labeling.py` is dead code w.r.t. the shipped list).

## 2. Consumer map (verified in code before any change)

Producer chain (labels are INHERITED downstream, never recomputed):

- `src/processors/ibtracs.py::build_event_list` — applies
  `add_wind_deltas` + `label_ri` on the filtered pre-dropna series, then
  `dropna(subset=["dv12_kt","dv24_kt"])` (line 233) →
  `data/event_list_augmented.csv` (32,989 rows).
- `src/processors/preprocess_scientific.py:374-376,625-627` — copies
  `ri_label`/`dv12_kt`/`dv24_kt` from the event-list row into each interim
  JSON sidecar. **Coercion site:** line 374 `int(row.get("ri_label", 0))`
  turns a missing label into 0.
- `src/data/normalization.py:342,442,503` — copies the sidecar values into
  `data/normalized/valid_events.csv` (`ri_label` column) and the
  normalization report. None-tolerant (`meta.get(..., None)`).

Readers of the label:

- `src/data/dataset.py:245` — **coercion site:** `int(meta.get("ri_label",
  0))` at training time; dv12/dv24 via `_safe_float` (None-tolerant).
- `src/evaluation/sla_validation.py:104` — **coercion site:**
  `int(m.get("ri_label", 0))`.
- `analysis/feature_ablation_kfold.py`, `analysis/tabular_baseline_kfold.py`
  (+ the CNN ablation harnesses) — labels from `valid_events.csv` merged
  with `splits.csv`. Closed experiments; NOT to be re-run; readers only
  matter for future use.
- `analysis/ri_precursors*.py`, `platform/build/build_events.py` — read the
  event list directly.
- `analysis/audit_core_integrity.py:216-316` — **independently
  re-implements** the positional labeling and asserts consistency with the
  shipped event list. Will FAIL against v2 unless updated in the same
  change (must assert the v2 temporal semantics, or pin to the
  diff-manifest).
- Row-set-only consumers (label-agnostic; protected by the row-set
  invariant): `src/pipeline/windowed.py`, `src/pipeline/pl_backfill.py`,
  `src/downloaders/era5*.py`, `src/downloaders/tchp.py`.

Conclusion: interim JSONs inherited builder labels at preprocess time →
the stored artifacts require a surgical patch (§4); fixing
`ri_labeling.py` alone corrects only future builds.

## 3. The fix (code)

### 3.1 `src/processors/ri_labeling.py` — strict-temporal semantics

Rewrite `add_wind_deltas` and `label_ri`:

- Partner = row of the SAME sid at EXACTLY t0+12h (dv12) / t0+24h (dv24).
  Implementation: per-SID self-merge on shifted timestamps (the v5 script's
  partner join is the reference implementation). No tolerance window.
- `dv12_kt`/`dv24_kt`: NULL (pd.NA/None) when no exact partner or wind
  missing on either side.
- `ri_label`: 1 if dv24 ≥ threshold; 0 if dv24 < threshold; **NULL if dv24
  is NULL** — the `astype(int)`-over-NaN coercion must become impossible,
  not just unused (nullable `Int64`).
- Requires a `timestamp` column; raise if absent (positional fallback is
  removed, not kept as an option).

### 3.2 `src/processors/ibtracs.py` — builder

- Line 233 `dropna(subset=["dv12_kt","dv24_kt"])` becomes a parametrized
  policy. Recommended default for future builds: keep rows and ship NULL
  labels (`drop_undefined=False`), with the RI task view doing the
  exclusion. NOTE: the v2 RELEASE does not re-run the builder (§4); this
  change is for semantic correctness of future use and for the §5
  regression check.

### 3.3 NULL-handling at the coercion sites

- `preprocess_scientific.py:374`: propagate None; never default to 0.
- `dataset.py:245`: NULL label → event EXCLUDED from the RI task view;
  loud, explicit filter (log count), never silent coercion; loading a NULL
  event without the filter raises.
- `sla_validation.py:104`: exclude NULL-label events from the RI/non-RI
  contrast (they are neither).
- `valid_events.csv`: `ri_label` becomes nullable Int64; NULL serialized as
  empty cell; every reader loads with `keep_default_na=False,
  na_values=[""]` (established repo rule) and nullable dtype.
- `audit_core_integrity.py`: update its independent re-implementation to
  the temporal rule (mirror of §3.1), so the audit stays meaningful.

### 3.4 Tests

- Unit tests for `ri_labeling`: regular grid (unchanged labels), irregular
  grid (misaligned partner → NULL or corrected value), storm boundaries,
  odd-minute timestamps, threshold edge (exactly +30 kt → 1).
- Regression: new builder code applied in-memory to the pre-dropna raw
  series must reproduce the v2 labels of the patch (§4) EXACTLY on the
  32,989-row frozen set.
- Reader tests: NULL label excluded by the dataset filter; direct load of a
  NULL event without the filter raises.

## 4. The patch (stored artifacts) — template: `analysis/repair_basin_metadata.py`

New script `analysis/apply_dv24_label_correction.py`. ERA5 raws 1980–2019
are discarded by design → pipeline regeneration is impossible; derived
artifacts are patched surgically, with provenance, exactly like the basin
repair.

Order of operations:

1. **Replication gate (§6):** rebuild the pre-dropna series from
   `data/raw/ibtracs.ALL.list.v04r00.csv` and prove the shipped event list
   is reproduced byte-exactly (32,989/32,989; dv12/dv24/ri_label equal on
   every row). ABORT on any mismatch. Reuse
   `dv24_impact_assessment_v5_raw_reference.build_pre_dropna_series`.
2. **Compute v2 labels** for the frozen 32,989-row set from the pre-dropna
   series (strict-temporal, NULL semantics). The ROW SET IS INVARIANT: no
   row added (including the 2 newly-definable dropped rows), no row
   removed.
3. **Emit the diff-manifest BEFORE writing anything:**
   `data/normalized/label_diff_v1_v2.csv` — sid, timestamp, event_id (when
   an interim/valid event exists), dv12_v1, dv12_v2, dv24_v1, dv24_v2,
   ri_label_v1, ri_label_v2, reason ∈ {unchanged | flip_misaligned |
   null_no_partner | dv_drift_only}. The v1 values in this file ARE the
   frozen v1-label artifact. Expected size: 32,989 rows, ~206 with
   reason ≠ unchanged (148 mis24 + 77 mis12 minus overlap; exact count
   asserted at runtime and recorded in the manifest).
4. **Dry-run mode first** (default): compute + write diff-manifest and
   verification report only; NO artifact modified. The author inspects and
   approves before `--apply`.
5. **`--apply`:** patch, in this order, each write re-read and verified
   field-by-field:
   a. `data/event_list_augmented.csv` — label columns only
      (`dv12_kt`, `dv24_kt`, `ri_label`, and the two `wind_kt_shift_*`
      helper columns updated consistently or dropped — author call, see
      Open decisions); all other columns byte-identical.
   b. Affected interim JSON sidecars — ONLY events whose
      dv12/dv24/ri_label changed and whose JSON exists; only those three
      fields touched; every other field byte-identical. `.npy` cubes are
      NEVER touched. This includes test-split sidecars — metadata-level
      repair, same category as the basin repair (which patched all 12,708
      affected JSONs across splits); requires the author's dated
      authorization line in PROJECT_STATE before `--apply` (no feature or
      cube data is read into any analysis).
   c. `data/normalized/valid_events.csv` — 19 rows get NULL (empty cell,
      nullable Int64); all other rows/columns identical.
6. **Provenance manifest** `outputs/provenance/dv24_label_correction_<UTC>.json`:
   repo-relative paths only, per-file counts, pre/post hashes, the v5
   report reference, and the verification results (§5).

## 5. Verification plan (targets = v5; anything off-target → STOP)

Event list:
- rows: 32,989 (unchanged); non-label columns byte-identical.
- ri_label flips: exactly 1 (1→0); zero (0→1).
- ri_label → NULL: 55 (3 previously positive).
- dv24 value changes: 67; dv12: 37; dv12 → NULL: 31.

Valid set / splits:
- valid_events.csv: 16,780 rows; label flips: 0; ri_label → NULL: 19
  (train 11 / val 2 / test 6); positives 802→799 (−3, all test).
- Dev PL-gated positives: 687→687 (INTACT).
- Storms losing all positives: 1, test split.
- dv12 (sidecars): 3 value changes + 8 NULL (train 5 / val 1 / test 2).

Invariants (any violation → STOP, revert, investigate):
- `data/normalized/splits.csv` and `frozen_splits.json`: md5 identical
  (splits are by SID; labels do not enter split assignment).
- Interim `.npy` cubes: untouched (no writes; spot-check mtimes/hashes).
- Interim JSONs: only the three label fields differ, only for listed
  event_ids.
- Regression (§3.4): new builder code reproduces the patched labels
  in-memory.
- `analysis/audit_core_integrity.py` (updated to v2 semantics) passes.
- Test split handled as aggregate counts only, per the dated exception in
  PROJECT_STATE.

## 6. Replication gate — PERMANENT rule (not just this task)

Any diagnosis of a defect in a derived artifact MUST first replicate the
shipped artifact byte-exactly from the raw source
(`data/raw/ibtracs.ALL.list.v04r00.csv` for the event chain). Without the
gate, "partner never existed" and "partner was used and then dropped by the
builder" are indistinguishable — this exact confusion fabricated the
retracted "Defect 0 / 84 phantom positives" (reports v1–v4). Lesson
recorded: arithmetic consistency across scopes does NOT detect a
wrong-reference error — the numbers agree with each other while measuring
the wrong thing. `analysis/dv24_impact_assessment_v5_raw_reference.py`
(abort-on-mismatch) is the template. Registered in PROJECT_STATE §3.

## 7. Mechanism note (for the data dictionary) — v5-basis numbers

Misalignment concentrates in, but is not exclusive to, the North Atlantic:
125/148 misaligned rows (84%) are NA (dv24 value drift: 59/67 NA;
strictly-undefined: 40/55 NA), spread over 39 storms and all decades
(1980s: 5, 1990s: 30, 2000s: 45, 2010s: 44, 2020s: 24). Consistent with
IBTrACS NA-basin extra entries (landfall fixes, odd-minute timestamps such
as hh:15/hh:30 that pass the synoptic-hour filter because `dt.hour`
matches). Phrase as a hypothesis about IBTrACS reporting practice, not as
established fact. (The retracted reports' claims "all 46 drift rows NA" and
"21/25 irregular SIDs NA" were measured on the wrong reference — do not
reuse.)

## 8. Technical Validation asset

The byte-exact replication of the shipped event list from raw IBTrACS
(32,989/32,989 rows, all label columns equal) is direct evidence of
labeling-pipeline reproducibility and goes into the Data Descriptor's
Technical Validation, alongside the v2 correction narrative (defect found
by systematic assessment, corrected at origin + surgical artifact patch,
with diff-manifest and provenance).

## 9. ERRATA surgery (coordinated with the docs pass, not piecemeal)

- Item 6 REWRITTEN to the true defect: positional misalignment on 148/32,989
  rows (0.45%; dv12: 77); ZERO label flips in the valid set (1 flip in the
  full event list); 19 valid events → NULL under strict semantics;
  positives 802→799 (−3, test). The old "~1.3%" estimate is superseded.
- Add the retraction record to item 6's history: the 16/07 assessment
  rounds v1–v4 misdiagnosed a cross-storm-leakage defect ("Defect 0", "84
  phantom positives"); reconstruction from raw refuted it
  (`outputs/results/dv24_impact/SUPERSEDED.md` holds the trail). NO new
  ERRATA item for "Defect 0" — it does not exist.
- dv24 correction itself becomes the item-6 resolution entry once §4 is
  applied (dataset v2, diff-manifest reference).

## Open decisions for the author (small, non-blocking for review)

1. `wind_kt_shift_12/24` helper columns in the event list: update
   consistently with v2 (shift value at the temporal partner) or drop them
   from v2 (they are derivable; dropping simplifies the data dictionary).
   Recommendation: drop in v2.
2. Builder default for future builds (§3.2): recommended
   `drop_undefined=False` (keep rows, NULL labels).
3. Whether the diff-manifest ships inside the v2 release package or only in
   the repo. Recommendation: ship it (it IS the v1 provenance).

## Sequencing (after author approval of this doc)

1. Author: approve doc + add the dated authorization line for patching
   test-split sidecars (metadata-level, §4.5b).
2. Haiku: §3 code fix + §3.4 tests → Fable line-by-line review.
3. Haiku: §4 patch script (dry-run default) → Fable line-by-line review,
   replication gate active.
4. Dry-run → diff-manifest + verification vs §5 targets → author inspects.
5. `--apply` → §5 verification → provenance manifest.
6. ERRATA/docs pass (§9) + PROJECT_STATE update; commit as a single
   reviewed change (no AI attribution, secret_guard clean).
