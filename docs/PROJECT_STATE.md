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

_Last updated: 2026-07-16 ~09:30 (seed 456 complete `bb7adaa`; H6
AGGREGATED and CLOSED: verdict NULL; next: H8 night 1)._

## 1. IMMEDIATE RESUME (what to do NOW)

**H6 IS CLOSED — verdict NULL (read once 16/07):** mean cross-seed
ΔPR-AUC +0.0185, 95% CI **[−0.0070, +0.0431] includes zero**; per-seed
+0.033/+0.011/+0.011. Formulation fixed by the pre-registration:
shear/rh_mid add no detectable skill at this resolution/regime for this
architecture; NOT a weak positive. Report:
`outputs/results/feature_ablation_cnn/aggregate_20260716T120517Z.json`.
Registry entry updated (H6 TESTED/NULL).

Next steps, in order:
1. **H8 night 1** — `analysis/fuelmap_ablation_cnn.py --reuse-arm-a
   <3 H6 run dirs> --execute` (~5.5 h/seed, phased nights, detached via
   Task Scheduler launcher like nights 1–3; §6 rules). Verdict decides the
   "physics-guided" name. Pre-reg: `docs/fuelmap_ablation_preregistration.md`.
2. H9 `--compare-cnn` (V1/V2 co-primary, consequences fixed) — GBM runs
   DONE (§4 item 2); the paired read is now UNBLOCKED by H6 close (cheap,
   minutes; author decides when).
3. After H8 closes: basin metadata REPAIR + unfreeze `src/` (T2/V4 work).

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

## 2. STATE OF ONGOING EXPERIMENT (H8 FuelMap physics-loss ablation)

**H6 (feature ablation) is CLOSED — verdict NULL, see §1 and the registry.**
Seeds 42/123/456 complete (`c608f19`/`c6a3b20`/`bb7adaa`); aggregate read
once 16/07 (`aggregate_20260716T120517Z.json`). No re-runs, no re-reads.

**Current experiment: H8** — pre-reg
`docs/fuelmap_ablation_preregistration.md`; arm A reused from H6
`A_current` cells (fold-identity validated), arm B (all physics lambdas 0)
trains ~5.5 h/seed, phased one seed per night, detached (§6). Same
discipline: verdict = mean cross-seed ΔPR-AUC (B−A), 95% SID-cluster CI,
read ONCE after 3 seeds through the 3 pre-registered branches with FIXED
consequences (hurt → remove; null → remove for parsimony; help → keep as
regularization, never as validated physics).

| Seed | H8 status |
|---|---|
| 42 | pending (night 1) |
| 123 | pending |
| 456 | pending |

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
2. **IN PROGRESS — post-H6 experiments:**
   - **H8** FuelMap physics-loss ablation (CURRENT, night 1 pending):
     `analysis/fuelmap_ablation_cnn.py --reuse-arm-a <H6 run dirs>
     --execute` (~5.5 h/seed, phased, detached).
     Pre-reg: `docs/fuelmap_ablation_preregistration.md`.
   - **H9** factorial tabular baseline: GBM runs **DONE 15/07** (run
     `20260715T123221Z`, `143756f`). V1/V2 verdicts via `--compare-cnn`
     now UNBLOCKED (H6 closed); paired read pending, author triggers.
     Pre-reg (3 amendments, CNN−F co-primary):
     `docs/tabular_baseline_preregistration.md`.
   - Harness: `analysis/tabular_baseline_kfold.py`. Local TODO/context:
     `.claude/TODO_recomendacoes.md`.
   - **`basin` metadata REPAIR** (audit done 15/07, see §1 — empty ≡ NA):
     fix `ibtracs.py:120` + rebuild or SID→basin repair map. Blocked until
     H8 closes; still gates T5/benchmark release. Public relabel already
     done (ERRATA item 7).
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
