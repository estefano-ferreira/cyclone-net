# PROJECT_STATE — state and resume

**READ THIS FILE FIRST when starting a session — it says where we stopped
and the next step.**

Rules for maintaining this file:
- UPDATE at the end of each session/milestone: move the "next step"
  forward, update the experiment's state, move completed pending items to
  "milestones".
- An OUT-OF-DATE state file is misleading — if anything changed and was
  not reflected here, fix it before trusting it.
- At the START of each session: READ this file first to locate yourself.

_Last updated: 2026-07-16 ~13:15 (H6 NULL; H9 V1 NEG / V2 NULL; H8
CANCELLED; label RETIRED; freeze LIFTED; **basin REPAIRED** — parser fix
+ rebuild + 12,708 JSONs, audit-exact verification)._

## 1. IMMEDIATE RESUME (what to do NOW)

**H6 IS CLOSED — verdict NULL (read once 16/07):** mean cross-seed
ΔPR-AUC +0.0185, 95% CI **[−0.0070, +0.0431] includes zero**; per-seed
+0.033/+0.011/+0.011. Formulation fixed by the pre-registration:
shear/rh_mid add no detectable skill at this resolution/regime for this
architecture; NOT a weak positive. Report:
`outputs/results/feature_ablation_cnn/aggregate_20260716T120517Z.json`.
Registry entry updated (H6 TESTED/NULL).

**H9 IS ALSO CLOSED (read once 16/07, order corrected by the author:
--compare-cnn BEFORE H8, because V2 gates H8):**
- **V1 NEGATIVE:** Δ₁(CNN−GBM_SF) = −0.0781, CI [−0.1162, −0.0422] < 0 —
  the tabular baseline beats the CNN; **GBM_SF is now the project's
  reference model** (pre-registered consequence).
- **V2 NULL:** Δ₂(CNN−GBM_F) = +0.0005, CI [−0.0285, +0.0316] ∈ 0 —
  **architecture NOT justified in its current form**; CNN = documented
  negative; redesign only as NEW pre-registration.
- Ex-ante qualifications recorded in the registry (GAP, intensity-blind,
  basin one-hot, 15-epoch budget). Report:
  `outputs/results/tabular_baseline/compare_20260716T121803Z.json`.

**H8 CANCELLED (author decision, 2026-07-16):** "H8 (FuelMap
physics-loss ablation) is CANCELLED, not deferred. Its question — do the
FuelMap physics losses help RI classification? — became undecidable when
H9's V2 retired the architecture those losses shape (pre-registered
joint reading, 2026-07-16: Δ₂ ≤ 0 → architecture retired/redesigned
regardless of Δ₁). Ablating a component of a retired model decides
nothing. The H8 pre-registration and harness
(analysis/fuelmap_ablation_cnn.py,
docs/fuelmap_ablation_preregistration.md) remain in the repo as record —
they are not to be run."

**"Physics-guided" label RETIRED (2026-07-16):** "The 'physics-guided'
label is retired. H8 was the honesty test for the name; it is moot
because the architecture carrying the four FuelMap-centred losses was
itself retired by H9/V2. Independently of H8, the label was already only
weakly supported: the KL prior-alignment term targets a heuristic prior
whose semantics were REFUTED in H1; the only equation-consistency term
is disabled (lambda_consistency = 0.0) and documented as
near-degenerate; there are no conservation laws or imposed dynamics. Do
not use 'physics-guided' in V3 or in public descriptions." Repo-wide
occurrence inventory: §4 relabel item (fix is a coordinated pass, not
piecemeal).

**`src/` FREEZE LIFTED (2026-07-16):** the freeze existed solely so
H8's arm B could train with the identical trainer that produced the
reused arm A. With H8 cancelled that reason is gone. Verified before
lifting: no training/backfill running; no pre-registered experiment
awaits the frozen code (H6/H9 closed, H7 deferred, H10/T2 design-only).

**REFERENCE-MODEL DECISION (author, 2026-07-16 — closes the open item
from `c1d3831`):** "The project reports NO single reference model. The
CNN has a frozen test-set number (PR-AUC 0.251 [0.179-0.331]) but was
retired by H9/V2. The GBM_SF is the empirical reference ON THE DEV FOLDS
(0.249 pooled OOF) and has NEVER been evaluated on the frozen test set —
promoting it would require a second test-set read, which is not
justified: the released contribution is the dataset, not a model. Both
are reported with their protocols explicitly separated. The CNN's
test-set metrics stand as historical record of a retired architecture."
Consequence: do NOT evaluate the GBM on the test set.

**basin metadata REPAIRED (2026-07-16, same session):** parser fixed at
the origin (`ibtracs.py` + 3 event-list readers, `keep_default_na=False`;
tests in `tests/test_ibtracs_basin.py`), event list rebuilt (32,989 rows —
identical except basin: ""→"NA" ×16,602), 12,708 interim JSONs repaired
surgically (`analysis/repair_basin_metadata.py`; raw ERA5 discarded by
design, pipeline regeneration impossible). Verification matches the 15/07
audit EXACTLY (per-point 8,888/7,892/16,780; genesis 578/414/992; same 6
crossers); `valid_events.csv`/`splits.csv` byte-identical. Manifest:
`outputs/provenance/basin_metadata_repair_20260716T130158Z.json`.
ERRATA item 7 updated to REPAIRED. Released artifacts keep the old label
(historical record; next re-release).

**PAPER 1 PREMISE VERIFIED (2026-07-16) — the cubes CAN ship:**
the TODO's "no cubes (license/size)" assumption is WRONG on both counts.
(a) Size, measured on disk: 448 KB/event exactly (40x40x5x14 float32);
data/interim totals 12.4 GB (11.26 GB cubes incl. 1,257 ADT extras);
valid-events-only subset ~7.7 GB — far under Zenodo's 50 GB/record.
(b) License: CDS/ERA5 moved to CC-BY on 2025-07-02, and even the prior
Copernicus licence allowed redistributing adapted products with the
notice "Contains modified Copernicus Climate Change Service information
(Year)" + EC/ECMWF liability disclaimer. IBTrACS is NOAA/public domain.
CONSEQUENCE: dataset license must be CC BY 4.0 (NOT CC0 — the Copernicus
attribution requirement cannot be stripped); the Data Descriptor has a
real product (per-event spatial cubes), not a recipe.

**dv24 STATUS UPGRADE (author, 2026-07-16):** for a Data Descriptor the
ri_label is the primary product — ERRATA item 6 (positional 4-row vs
strictly temporal 24 h delta, ~1.3% of points) is a DEFECT in the
primary data, not a caveat. Correcting before release is the path;
declaring is the minimum. Note: correcting changes ~1.3% of labels →
new dataset version (v2) with the v1 verdicts kept as record.

Next steps:
1. **dv24 label correction** (release-gating for Paper 1; design the fix
   + impact assessment before touching anything).
2. T5 packaging WITH cubes (~7.7 GB valid subset) + CC BY 4.0 + Copernicus
   attribution notices; repository choice (PANGAEA vs Zenodo versioned);
   coordinate with the Zenodo re-publication so the public face changes
   once. APC decision: Scientific Data GBP 2,190 / USD 2,850 vs ESSD
   EUR 1,400 flat (both have waiver policies; no automatic waiver for
   Brazil at either).
3. "physics-guided" relabel pass (inventory in §4; coordinate with the
   same re-release).
4. V3 skeleton (also unlocks the coauthor-contact gate).

**H9 executed 15/07** (run `20260715T123221Z`, commits `d6cc930` fix +
`143756f` results; 3rd dated pre-reg amendment: median imputation for the
LogReg reference, crash happened before any result). Pooled OOF PR-AUC by
seed (42/123/456): S 0.191/0.196/0.218; F 0.169/0.159/0.183;
**SF 0.241/0.245/0.261**; LogReg SF ~0.20. Uncomfortable direction for
the CNN, BUT: verdicts only via the paired `--compare-cnn` post-H6, and
CNN ≈ GBM(F) carries a mandatory qualifier — the current CNN does global
average pooling (aggregates spatially), so the tie was expected OF THIS
ARCHITECTURE and licenses no claim about spatial signal in the data.

**Basin QA — AUDITED + PUBLIC RELABEL DONE 15/07 (ERRATA item 7):** the
dataset is TWO-BASIN — 992 valid storms = 578 EP / 414 NA; "NA" (North
Atlantic) was blanked by pandas default `na_values` in `ibtracs.py:120`
(3rd appearance of the `keep_default_na` pitfall; the fix in
`build_events.py:122` never reached the metadata path). Empty ≡ NA,
recoverable per SID; no data lost. "North Atlantic sector" framing was
INCORRECT — corrected in README/BENCHMARK/MANUSCRIPT/tex/INTERPRETATION/
DATASET + ERRATA item 7 + H9 registry note (GBM effectively used basin as
a predictor via the mislabeled one-hot). METADATA repair (ibtracs.py fix +
rebuild or SID→basin map) stays blocked until H6/H8 close. Released
artifacts' `coverage` strings keep the old label until next retrain.

Night-2 operational notes: cell wall time varied 114–157 min (machine
load-dependent; ~110 min/cell when dedicated). A session background
watcher was killed again 15/07 ~08:00-08:28 with NO RestartManager/System
events in the window — 3rd occurrence of the session-tree kill pattern,
2nd without updater correlation; detached training was untouched both
nights. Keep everything critical detached; session watchers are
best-effort only.

**Incident 14/07 19:30 (resolved):** the night-2 trigger died at launch
(exit 1, no log): the `.ps1` was UTF-8 without BOM → PS 5.1 read it as
ANSI → accented username in hardcoded paths mangled → first `Out-File`
threw under `ErrorActionPreference=Stop`. Fix (permanent rule): every
operational `.ps1` is ASCII-only with paths via `$env:USERPROFILE`, saved
with BOM. Re-fired 19:35 via `Start-ScheduledTask`, pre-checks passed.

## 2. STATE OF EXPERIMENTS

**H6 (feature ablation) CLOSED — NULL** (seeds `c608f19`/`c6a3b20`/
`bb7adaa`; aggregate read once 16/07, `aggregate_20260716T120517Z.json`).
**H9 (tabular baseline) CLOSED — V1 NEGATIVE / V2 NULL** (paired read once
16/07, `compare_20260716T121803Z.json`). No re-runs, no re-reads of either.

**H8 CANCELLED (2026-07-16, §1):** no launcher scheduled, no cells run,
none will be. Pre-registration and harness preserved in the repo as
record; not to be run.

## 3. PROJECT PERMANENT RULES (inviolable)

- No commit/PR bears AI attribution.
- `secret_guard` CLEAN before every commit; never commit `config.yaml`,
  `run_snapshot.json`, `.cdsapirc`, `.netrc`.
- Paths always relative (`rel_to_root`) — absolute paths with accented
  username BREAK netCDF reading on this machine (not just hygiene, it is
  functional).
- Do not edit code that an active process (training/backfill) is using.
- Verify before discarding/overwriting (completeness gate,
  manifest+data together).
- SID-based hash-deterministic splits + `frozen_splits.json`; frozen test
  set, never read in development.
- Epistemic honesty: FuelMap = hypothesis maps; no inflating results.

## 4. PENDING QUEUE (by priority)

1. **DONE 16/07 — H6 closed (verdict NULL, read once).** Registry updated;
   aggregate JSON committed. No further reads.
2. **Experiments:**
   - **H9 CLOSED 16/07** — V1 NEGATIVE (GBM_SF beats CNN), V2 NULL
     (architecture not justified). Registry updated with ex-ante
     qualifications. GBM runs `20260715T123221Z` (`143756f`); paired
     report `compare_20260716T121803Z.json`.
   - **H8 CANCELLED 16/07** (see §1; registry has the verbatim record).
   - Local TODO/context: `.claude/TODO_recomendacoes.md`.
   - **`basin` metadata — REPAIRED 16/07** (see §1; ERRATA item 7 updated;
     manifest `basin_metadata_repair_20260716T130158Z.json`). No longer
     gates T5.
   - **"physics-guided" RELABEL (new 16/07, label retired — see §1):**
     inventory of occurrences (16/07): README.md (§structure/§losses/
     §preprocessing), MANUSCRIPT_honest.md (title + 7 mentions),
     cyclonenet_honest.tex (title + 6), CITATION.cff + docs/CITATION.md
     (title), ERRATA.md (historical mentions — keep, they describe the
     old paper), pyproject.toml (description), run.py, mcp_server.py,
     src/__init__.py, src/training/trainer.py,
     src/models/cyclone_net_physics_guided.py (class
     CycloneNetPhysicsGuided + filename), src/physics/* (module name +
     physics_guided_losses.py), src/processors/preprocess_scientific.py,
     src/data/dataset.py + normalization.py (config keys
     `physics_guided.*`), config-template.yaml (block name),
     tests/test_physics_losses.py + test_adt_input.py,
     analysis/fuelmap_ablation_cnn.py + fuelmap_ablation_
     preregistration.md (H8 record — keep as-is), docs/diagnostic.md,
     docs/scalar_branch_design.md. Historical/record files (ERRATA, H8
     pre-reg, registry) KEEP the term; public-facing and structural ones
     change in a coordinated pass.
3. **PR flow (corrected 15/07):** PRs #9/#10/#11/#12 (`feature/tchp` →
   `main`) are all MERGED (#12 on 14/07 19:22 absorbed up to `6691fc9`).
   **The live PR is #13** (opened 15/07 ~09:56, head `443ffe5`, 8 commits,
   mergeable clean, no conflicts). `origin/main` serves the CURRENT README
   (PR-AUC 0.251, honest framing) — the outdated 0.347 copy is the Zenodo
   snapshot (item 7 below), not GitHub.
4. **TODO — author manual action:** update the GitHub repo About text
   (Settings) — suggested wording in `.claude/TODO_recomendacoes.md`; the
   current one still sells the refuted energy-source premise.
5. **BLOCKED (pending verdict) — Post-ablation:** apply the result to V3,
   Form A (model on platform).
6. **TODO — PR #9:** open; merge is the user's call.
7. **TODO — hygiene/docs (details to confirm with the author; inherited
   from an earlier session):** dv24 entry in the root ERRATA.md, README
   link, V3 paragraphs.
   - **Zenodo snapshot is outdated** (confirmed by the author 14/07): the
     published copy still shows old metrics (0.347) and the
     FuelMap-central framing without the refutation — it does NOT reflect
     the current state (PR-AUC 0.251 [0.179–0.331], FuelMap refuted,
     corrected dataset). The LOCAL README is already current
     (README.md:267-268) — nothing to fix in the repo. Pending action
     (NOT now): re-publish the Zenodo snapshot AFTER the ablation closes
     and V3 is complete — Zenodo is the archived public face and must
     tell the current story before formal publication. Priority:
     low/medium — no active harm, but must be closed before publishing.
8. **TODO (security, pending with the author):** rotate CDS key and
   Copernicus password (leaked in git history; redaction at source
   already implemented).
9. **CONTINUOUS — hypothesis registry:** `docs/hypothesis_registry.md`
   is the living research agenda (H1..H9 + future agenda). Record
   hypothesis+test BEFORE running; honest verdict after. H7 re-classified
   DEFERRED (needs inner-core data); next verdicts due: H6 → then H8/H9.

## 5. MILESTONES ALREADY COMPLETE (do not redo)

- PL backfill 20/20 (1980–2019; 21,662 events, zero failures), complete
  provenance (`884ac36`).
- Core-integrity audit 5/5 (post-backfill closes items 3 and 5, `ff223cd`).
- PL census: gate PASS (14,101 dev events, 100% coverage).
- Platform live with environmental panel + basin (`08ad031`).
- Relative path hygiene in manifests (`ee7dc7c`).
- Ablation pre-registration locked in before any results (`eaa8ae8`).
- Post-backfill re-test of RI precursors on frozen pairs: H1–H4
  significant under Bonferroni ×4, H2/H4 with 394/394 pairs (`970a419`).
- Ablation night 1 / seed 42 complete (`c608f19`); night 2 / seed 123
  (`c6a3b20`); night 3 / seed 456 (`bb7adaa`).
- **H6 CLOSED 16/07 — verdict NULL** (Δ +0.0185, CI [−0.0070, +0.0431],
  read once; `aggregate_20260716T120517Z.json`). Do not re-read or re-run.
- **H9 CLOSED 16/07 — V1 NEGATIVE / V2 NULL** (read once;
  `compare_20260716T121803Z.json`): GBM_SF is the reference model; the
  GAP-CNN is a documented negative. Do not re-read or re-run.
- Dataset 1980–2023: 16,780 valid events / 802 RI positives / 992
  storms; splits with no leakage; frozen benchmark intact.

## 6. LONG PROCESS EXECUTION (external-interruption protection — MANDATORY)

Context: long processes running as children of the terminal tree were
killed 2x on this machine (13/07 ~01:00 and ~20:19). Strongest suspect for
the 2nd: Google Updater self-update + RestartManager session (runs every
~3h); 1st inconclusive (window only had Windows Update/Defender). The
common pattern: **only terminal session tree processes die**. A detached
run traversed 11 h and multiple RestartManager sessions untouched.

Rules for ANY process > ~15 min (training, backfill, download):

1. **NEVER run as child of terminal/session** (not even as "background
   task" of the session — that's exactly what died 2x).
2. **Always DETACHED**, with stdout/stderr to file and `-u` (unbuffered):
   ```powershell
   Start-Process -FilePath ".\venv\Scripts\python.exe" `
     -ArgumentList "-u","<script>","<args...>" `
     -WorkingDirectory "<repo root>" `
     -RedirectStandardOutput "C:\Users\Estéfano\cyclone-net-ops\<name>.log" `
     -RedirectStandardError  "C:\Users\Estéfano\cyclone-net-ops\<name>.err.log" `
     -WindowStyle Hidden -PassThru
   ```
   Or via **Task Scheduler** (spawns outside any terminal tree): launcher
   `.ps1` with pre-checks in `C:\Users\Estéfano\cyclone-net-ops\`, trigger
   with a generous `-ExecutionTimeLimit` (e.g., 16 h). Note: missed "Once"
   trigger (machine off) does NOT re-fire.
3. **Pre-checks before firing** (night 2's launcher is the model): prior
   step complete? machine free (no other training/backfill)? If it fails,
   abort and log the reason — never fire on top of it.
4. **Logs and progress outside Windows Temp** (Temp can be cleared): use
   `C:\Users\Estéfano\cyclone-net-ops\` for operational logs; scientific
   artifacts stay in `outputs/` as always.
5. **Monitor via disk artifacts, not terminal:** checkpoint timestamps,
   `ablation_eval.json` per cell, OOF/summary at end. The process does not
   depend on anyone watching.
6. **Machine:** must not suspend (already configured); automatic updates
   (Google Updater/Windows Update) may run at night — detached is immune,
   but avoid installing/updating software during training.
7. **Resume:** every long flow must be resumable (skip-if-exists, manifest
   per window, OOF per cell) — if it dies, re-running continues, does not
   restart. Before re-running, check what completed on disk.

## 7. KEY REFERENCE NUMBERS

- Production model: PR-AUC 0.251 [CI 0.179–0.331], ROC-AUC 0.796.
- Dataset: 1980–2023, 16,780 events, 802 RI positives (dev PL-gated:
  14,101 events / 687 positives / 839 storms).
- Training cost on this machine (CPU): ~110 min/cell (15 epochs) →
  ~11 h per ablation seed (6 cells).
- Seed 42 (intermediate): pooled OOF A=0.162 / B=0.195 (PR-AUC);
  ROC A=0.786 / B=0.825.
