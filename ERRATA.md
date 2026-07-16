# Erratum & Correction Note — CycloneNet V2 preprint

**Paper:** *CycloneNet V2: A Forensic Physics-Guided Deep Learning Framework for Atmospheric Singularity Mapping* (Feb 23, 2026).

This note corrects discrepancies between the preprint and the publicly released
code/data, found during an audit of the pipeline. It is published in the interest of
scientific integrity and reproducibility. None of the corrections below indicate
fabrication; they are release/reproducibility errors and an over-broad framing that
this note makes precise.

---

## 1. The "physics-guided" losses were inactive in the released code

**What the paper says:** the model is physics-guided via a KL FuelMap-alignment loss
and an equation-consistency loss, "added when the required tensors are present."

**What the released code actually did:** the training configuration did not define the
`training.physics.*` loss weights, so every physics-guided term defaulted to weight
`0.0`. The released model was, in practice, a plain 3D-CNN classifier with **no active
physics-guided constraint**. The configuration flags that appeared to enable them
(`use_prior_loss`, `use_consistency_loss`, `use_tv_l1`) were not read by the trainer.

**Correction:** the loss terms are now correctly wired and active by default
(`training.physics` block). Additionally, we note that the equation-consistency term is
**near-degenerate by construction** — it compares vorticity/divergence recomputed from
the wind field against stored channels derived from the *same* wind field — and it is
therefore documented as a weak representational regularizer, not a physical constraint.
It is disabled by default.

## 2. The operating threshold did not match the described methodology

**What the paper says:** a validation threshold targeting recall ≥ 90% (value 0.0666),
applied unchanged to the test set.

**What the released code did:** the threshold was selected by maximizing F1, not by
the recall target. The reported value (0.0666) is not reproducible from the released
threshold-selection routine.

**Correction:** threshold selection now follows a configurable policy
(`precision_at_recall`, honoring `training.eval_target_recall`), consistent with the
paper's stated forensic high-recall intent.

## 3. The reported metrics were NOT reproducible from the public repository — **RESOLVED**

**What the paper reports:** a held-out test set of **2,193 samples (211 RI positives)**,
with **ROC-AUC = 0.83, recall = 0.905**.

**What the public repository contained at audit time:** a dataset covering **2020–2023
only** (972 valid events; test split = **155 samples, 9 RI positives**). The headline
metrics **could not be reproduced** from the released code and data. On that public
dataset, with the corrected (physics-active) pipeline, test ROC-AUC was on the order of
**~0.6**, on a test set with too few positives (9) to support a stable estimate.

**Cause:** the headline numbers derived from a larger dataset that was not included in
the public release.

**Resolution (option (a) — full data release):** the repository now builds and releases
the **full 1980–2023** dataset (16,780 valid events / 802 RI positives / 992 storms) via
a windowed, checksummed, provenance-manifested pipeline, with hash-deterministic
storm-level splits. The model was retrained deterministically on this dataset (the
committed checkpoint was reproduced digit-for-digit from scratch), and the headline
metrics are **replaced** by numbers reproducible from the public artifacts:
**ROC-AUC 0.796 [95% CI 0.753–0.837], PR-AUC 0.251 [0.179–0.331], recall 0.852** on a
test split of 2,679 events (115 RI positives). Both confidence intervals sit entirely
above chance — the first release of this project for which that is true. The original
0.83/0.905 figures remain non-reproducible and are superseded; they should not be cited.

## 4. FuelMap localization does not beat a trivial baseline (validated negative, three angles)

The paper correctly **does not claim** externally validated localization. Validations
added during and after the audit make the negative result robust from three independent
angles:

1. **Static TCHP validation** (n = 226 eligible test events, TCHP publicly gridded
   2022+): the FuelMap peak beats a random-point null (p = 0.0003) but does **not**
   locate the TCHP peak better than a naive "storm-centre" baseline — median 539 km vs
   561 km, closer in only 46% of events (p = 0.30, sign-flip permutation).
2. **Dynamic displacement test:** the FuelMap's apparent collapse-toward-centre during
   RI, initially a candidate signal, was tested against a control.
3. **Physics-prior control:** re-running the displacement analysis on the pure
   enthalpy-flux prior (no learned weights) reproduces the dynamic behavior — it is
   arithmetic of the physics prior, **not** learned skill.

The localization claim is therefore **unsupported**, not merely "pending." A
counterfactual ablation (occluding the FuelMap region vs. an equal-size control) does
show the model's RI prediction depends on the identified region (paired test, p ≈ 1e-6) —
but this demonstrates *model-internal* dependence, **not** that the region is the true
physical energy source. Full protocols: `docs/fuelmap_validation.md`.

## 5. Novelty positioning

A literature check (operational and research) indicates that the components and the
forensic/diagnostic framing are **established**, not new:
- Deep learning for RI, including CNNs over the North Atlantic / East Pacific.
- Altimetry-derived ocean heat content (SSHA/ADT → TCHP) as an RI predictor
  (foundational AOML work; TCHP is already a SHIPS-RII predictor).
- Physics-informed / interpretable ML for TC intensity (spatial attention, PINNs,
  transformer interpretability).
- Identifying/quantifying the ocean energy source is operational via TCHP/OHC products
  and coupled ocean–atmosphere models (HWRF, HAFS) that compute the air–sea enthalpy
  flux directly.

**Correction to framing:** the contribution is **engineering and reproducibility**
(an auditable, configuration-driven, tested hybrid physics+ML pipeline), **not** a
scientific discovery, a novel architecture, nor a new capability to "identify the energy
source feeding hurricanes." The title term "Atmospheric Singularity Mapping" overstates
the scope and **has been removed** from the project's documentation and citation
metadata; the revised manuscript carries an honest title.

## 6. dv24/dv12 were positional shifts, not strictly temporal deltas — **CORRECTED IN DATASET v2 (2026-07-16)**

*(This item was REWRITTEN on 2026-07-16. The previous text, based on a
235-point sample, estimated "~1.3% of points" and anticipated positional
label flips; the full-population measurement against raw IBTrACS showed the
flips are effectively zero and the incidence lower. Keeping the old text
would have declared the wrong defect.)*

**The real defect (measured against raw IBTrACS on the full population;
authoritative report `outputs/results/dv24_impact/report_v5_20260716_152525.*`):**
dv24/dv12 — the intensity changes underlying the RI label — were computed as
**positional shifts** (−4/−2 rows per storm), assuming a perfect 6 h grid.
Where the filtered best-track series is not 6 h-regular, the positional
partner is not at t0+24 h / t0+12 h:

- **148 / 32,989 event-list rows misaligned for dv24 (0.45%); 77 for dv12.**
- **Label flips: ZERO in the valid set** (the full event list has exactly
  one, a row outside the valid set: dv24 75.0 → 20.0 kt, RI 1→0).
- **19 valid events have no exact temporal partner** → label is UNDEFINED
  under the canonical definition (11 train / 2 val / 6 test). The old code
  silently coerced undefined to 0 upstream and dropped such rows at build
  time.
- **Impact:** valid-set positives 802 → 799 (−3, all in the frozen test
  split); dev PL-gated positives 687 → 687 (**intact** — H6/H9 verdicts are
  unaffected by construction); one storm (test split) loses its positives.
- **Mechanism (hypothesis, untested):** 84% of misalignments (125/148) are
  in the North Atlantic basin — consistent with IBTrACS extra entries /
  odd-minute timestamps (e.g. HH:30) that pass the synoptic-hour filter
  (`dt.hour ∈ {0,6,12,18}` accepts any minute). Dispersed across all
  decades (1980s–2020s), 39 storms.

**Resolution — dataset v2 (2026-07-16, this commit):** labels recomputed
with **strict-temporal semantics** — partner = exact match at t0+12 h /
t0+24 h, same storm; no partner → **NULL, never 0**
(`ri_label ∈ {0, 1, NULL}`; NULL events remain in the dataset, the RI task
view excludes them). Code fixed at the origin
(`src/processors/ri_labeling.py`: temporal join, nullable Int64; builder +
NULL-safe readers). Stored artifacts patched surgically (event list, 25
interim sidecars, `valid_events.csv`) with per-file verification;
diff-manifest `data/normalized/label_diff_v1_v2.csv` (v1 values preserved =
provenance); provenance manifest
`outputs/provenance/dv24_label_correction_20260716_175443.json`. Verified by
raw-replication gate, target assertions, and independent recomputation from
the patched files; the patch run is idempotent. Splits unchanged (by SID;
md5-verified).

## 7. The dataset is two-basin (EP + NA), not "North Atlantic sector" (2026-07-15)

**What the documentation said:** README, BENCHMARK and the manuscript described the
dataset as a "1980–2023 North Atlantic sector archive"; `docs/INTERPRETATION.md`
claimed the model was "trained on Atlantic hurricanes only."

**What the data actually contains:** the extraction bounding box
(`spatial_subset: [60, -140, 0, -20]`) spans **two basins**. Of the 1,498 storms in
the event list, **805 are East Pacific and 699 North Atlantic** (by first IBTrACS
basin); of the 992 storms in the valid dataset, **578 are EP and 414 NA**. The East
Pacific is the *majority* basin. The bounding box cuts the EP basin west of 140°W.

**Why it went unnoticed — causal chain:** `src/processors/ibtracs.py` (line ~120)
reads IBTrACS with pandas' default `na_values`, which parse the literal basin code
`"NA"` (North Atlantic) as a missing value; `_clean_text_column(default="")` then
writes it as an empty string into the event list, and
`src/processors/preprocess_scientific.py` (line ~369) propagates the empty string
into every interim event JSON. As a result the `basin` metadata field contains only
`"EP"` and `""`, and the visible label ("EP present, everything else blank")
supported the wrong "Atlantic archive" reading. A `keep_default_na=False` fix exists
in `platform/build/build_events.py` (line ~122) and
`analysis/audit_core_integrity.py` (line ~224), but it **never reached the main
event-list → metadata path**. This is the third appearance of the same pandas
pitfall in this codebase.

**No data was lost:** the empty string is exactly ≡ `NA` (verified event-by-event
against the raw IBTrACS read correctly: 99.9% correspondence, off-diagonals being
basin-crossing storms). The true basin is deterministically recoverable per SID.

**Basin is a per-point attribute, not a per-storm one (verified 2026-07-15):**
IBTrACS assigns `BASIN` to each track point, and **6 storms in the event list
genuinely cross between EP and NA inside the sector** (Debby 1988, Joan/Miriam
1988, Gert 1993, Hermine 2010, Otto 2016, Bonnie 2022 — all documented
crossovers; their label flips at the Central America boundary were checked
against the tracks). This is why per-basin SID counts (805 EP + 699 NA)
exceed the 1,498 unique storms by exactly 6. The storm-level split quoted
above (578 EP / 414 NA of the 992 valid storms) attributes each storm to the
basin of the **first point of its raw IBTrACS record (genesis basin)**;
alternative criteria move at most 2 storms (first valid event: 579/413;
majority of valid events: 580/412). Per-point, the 16,780 valid events split
8,888 EP / 7,892 NA, and exactly **one** valid storm (Joan/Miriam 1988) has
valid events in both basins (11 EP / 9 NA).

**Scientific impact:** none on H6/H8/H9 verdicts — splits do not use basin, and the
H9 tabular baseline uses the same (mislabeled but internally consistent) table in
all three arms. The impact is on the **public framing**, corrected as of this note
in README.md, BENCHMARK.md, MANUSCRIPT_honest.md / cyclonenet_honest.tex,
docs/INTERPRETATION.md and docs/DATASET.md. Note also that SHIPS-RII is fitted per
basin, so any future comparison requires separating basins.

**REPAIRED (2026-07-16), fix at the origin + rebuild:** with H6/H9 closed and
H8 cancelled, the `src/` freeze lifted and the repair was executed the same day:

- **Parser fix:** `src/processors/ibtracs.py` now reads IBTrACS with
  `keep_default_na=False, na_values=[" "]` (IBTrACS encodes missing fields as a
  single space; verified empirically — numeric columns are unaffected because
  they pass through `pd.to_numeric(errors="coerce")`). The three downstream
  readers of the event list (`preprocess_scientific.load_event_list`,
  `windowed._window_events`, `pl_backfill._window_events_from_list`) read with
  `keep_default_na=False, na_values=[""]` so the recovered `"NA"` survives
  re-reading. Regression-tested in `tests/test_ibtracs_basin.py`.
- **Event list rebuilt** from the raw IBTrACS CSV with the fixed parser:
  32,989 rows / 1,498 storms / 1,875 RI positives — identical to the pre-fix
  list in **every column except `basin`** (verified column-by-column; basin
  transitions exactly `"" → "NA"` ×16,602 and `"EP" → "EP"` ×16,387).
- **Interim metadata repaired** surgically (`analysis/repair_basin_metadata.py`;
  the historical raw ERA5 was discarded by design after windowed processing, so
  the 26,954 interim JSONs cannot be regenerated by the pipeline — the script
  writes the exact value the fixed parser produces, touching **only** the
  `basin` field; every write is re-read and verified field-by-field): 12,708
  JSONs `"" → "NA"`, 14,246 unchanged (`EP`), 0 unmatched.
- **Post-repair verification matches the 2026-07-15 audit exactly:** per-point
  8,888 EP / 7,892 NA over the 16,780 valid events; genesis-basin storms
  578 EP / 414 NA over 992; the same 6 crossers (Debby 1988, Joan/Miriam 1988,
  Gert 1993, Hermine 2010, Otto 2016, Bonnie 2022). `valid_events.csv`,
  `splits.csv` and the frozen split map are byte-identical (no basin there).
  Provenance manifest: `outputs/provenance/basin_metadata_repair_20260716T130158Z.json`.

The `coverage` strings inside released artifacts
(`models/checkpoints/dataset_provenance.json`, `outputs/results/test_metrics.json`)
still carry the old "North Atlantic sector" label: they are the historical record
of the retired model release and are deliberately **not** edited in place; they
will be corrected at the next retrain/re-release.

## 8. Retraction: the "Defect 0 / cross-storm label leakage" diagnosis (2026-07-16)

**What was diagnosed (impact-assessment reports v1–v4, 2026-07-16):**
cross-SID leakage in the shipped labels — "84 phantom positives", "12 storms
losing all their positives", "3,062 undefined labels (18.2% of the valid
set)".

**What it was: the defect DOES NOT EXIST.** Reconstruction from raw IBTrACS
refuted the diagnosis in full.

**Why the error happened (the part that matters):** the diagnosis was run on
`data/event_list_augmented.csv` — a **derived artifact**, from which the
builder's `dropna` (`ibtracs.py:233`) had already removed the partner rows
used in the original label computation. On that file, "partner never
existed" and "partner was used and then dropped" are indistinguishable, so
legitimate trailing-row labels looked like cross-storm bleed. The supporting
"global shift" evidence (959/5,829 matches) was chance collision of
5-kt-quantized deltas — real leakage would have matched ~100%, not 16%.

**The lesson (verbatim):** "consistência aritmética entre escopos NÃO
detecta erro de referência — os números fecham entre si e medem a coisa
errada." (*Arithmetic consistency across scopes does NOT detect a
wrong-reference error — the numbers agree with each other while measuring
the wrong thing.*) Consequence: the **raw-replication gate** is now a
permanent project rule (PROJECT_STATE §3) — any defect diagnosis on a
derived artifact must first replicate the shipped artifact byte-exactly from
the raw source, and abort if it cannot.

**Trail preserved:** `outputs/results/dv24_impact/SUPERSEDED.md` retracts
reports v1–v4 (kept on disk as the audit trail);
`report_v5_20260716_152525.*` is the authoritative assessment.

---

## Technical validation note (2026-07-16) — positive result, not an erratum

Reconstruction of the event list from
`data/raw/ibtracs.ALL.list.v04r00.csv`, replicating the builder chain
(filters, synoptic-hour selection, per-storm labeling, target dropna),
reproduces the shipped file **byte-for-byte** — 32,989/32,989 rows, dv24 and
`ri_label` identical on every row. **The labeling pipeline is fully
reproducible from the raw source.** This is a Technical Validation result
for the Data Descriptor, established independently of (and prior to) the v2
correction.

---

## Summary

The preprint's *intent* was honest (it explicitly disclaimed discovery, novel
architecture, and validated localization). The errors were: (1) the released code did
not implement the physics-guided method it described; (2) the threshold method differed;
(3) the headline metrics were not reproducible from the released artifacts; and (4) the
framing/title overstated the scope relative to existing literature and operational tools.

**Status:** all four items are now addressed. (1) and (2) are corrected in the current
code. (3) is resolved via option (a) — the full 1980–2023 dataset is released and the
superseded headline numbers are replaced by reproducible ones (ROC-AUC 0.796, PR-AUC
0.251, both CIs above chance). (4) is addressed by revising the title and framing: the
validated contribution is the auditable pipeline and the RI classification skill;
spatial energy-source attribution is documented as a validated-negative hypothesis
(item 4 above, three independent angles).

A sixth item, found after the above were closed and **corrected on
2026-07-16 (dataset v2)**: the dv24/dv12 labels were positional (4-row/2-row)
rather than strictly temporal deltas. Measured on the full population against
raw IBTrACS: 0.45% of rows misaligned, ZERO label flips in the valid set,
19 valid events relabeled NULL under strict semantics, positives 802→799
(−3, all test), dev set intact. Corrected at the origin and in the stored
artifacts, with diff-manifest and provenance (item 6 above).

An eighth item (2026-07-16, item 8 above) records a RETRACTION: an
intermediate diagnosis of "cross-storm label leakage / 84 phantom positives"
(assessment reports v1–v4) was refuted by reconstruction from raw IBTrACS —
the diagnosis had been run on a derived artifact from which the builder had
already dropped the partner rows. The raw-replication gate is now a permanent
project rule. A standalone technical validation note (above) records the
positive counterpart: the shipped event list is byte-reproducible from the
raw source.

A seventh item (2026-07-15, item 7 above): the dataset is **two-basin
(East Pacific majority + North Atlantic)**, not the "North Atlantic sector"
the documentation claimed — a pandas `"NA"`-parsing bug blanked the North
Atlantic basin labels and masked the composition. Framing corrected across
the documentation; the metadata itself was **repaired on 2026-07-16** (parser
fixed at the origin, event list rebuilt, 12,708 interim JSONs restored to
`"NA"`, verification matching the audit exactly — see item 7). No
experimental verdict is affected.
