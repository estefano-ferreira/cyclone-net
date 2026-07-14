# Phased feature ablation (CNN) — state and resume

> **Project state and next step: see `docs/PROJECT_STATE.md`
> (read that first when starting a session). This file covers the
> operational details of the ablation.**

Pre-registered protocol in `docs/ablation_preregistration.md` (commit
`eaa8ae8`, locked in BEFORE any results): k=3 folds, 15 epochs,
seeds {42, 123, 456} run ONE PER NIGHT, CPU, arm A (9 current channels)
vs arm B (+shear_850_200_mps, +rh_mid). SINGLE verdict after aggregating
all 3 seeds — per-seed results are intermediate and must not be
interpreted in isolation.

## State (2026-07-14)

| Seed | Night | Status | Artifacts |
|---|---|---|---|
| 42 | 1 (13→14/07) | **COMPLETE** — committed and pushed (`c608f19`) | `outputs/results/feature_ablation_cnn/20260713T232126Z/` (seed42/oof_predictions.csv + summary.json on git) |
| 123 | 2 | **PENDING** — not yet run | — |
| 456 | 3 | **PENDING** | — |

Intermediate record for seed 42 (NO verdict): pooled-OOF ΔPR-AUC
(B−A) = +0.033; per fold +0.051 / +0.043 / +0.019. Observed actual cost:
~110 min/cell → ~11 h wall time per seed (original calibration predicted
8.5 h).

## To resume (with the machine on)

1. **Seed 123** (night 2):
   ```
   ./venv/Scripts/python.exe analysis/feature_ablation_cnn.py --folds 3 --epochs 15 --seeds 123 --execute
   ```
   - Run DETACHED (outside the terminal tree — e.g., `Start-Process` with
     redirected logs, or Task Scheduler). Child processes of the terminal
     session were killed 2x on this machine (~01:00 and ~20:19 on 13/07;
     suspect: RestartManager/Google Updater). Detached runs survived
     11 h without issue.
   - ~11 h wall time; machine must not suspend (already configured).
   - An automated task already exists: **`CycloneNet-Ablation-Night2-Seed123`**
     (Task Scheduler, fires 14/07 at 19:30 IF the machine is on and
     logged in; missed trigger does NOT re-fire). Launcher with pre-checks:
     `C:\Users\Estéfano\cyclone-net-ops\ablation_night2_seed123.ps1` —
     aborts if seed 42 is incomplete or if training is active (so it is
     safe to coexist with manual firing). Logs in
     `C:\Users\Estéfano\cyclone-net-ops\`.
2. **Seed 456** (night 3): same command with `--seeds 456`, same method.
3. At the end of each seed: commit `oof_predictions.csv` + `summary.json`
   from the run dir (checkpoints .pt are gitignored).

## Final verdict — with all 3 seeds only

```
./venv/Scripts/python.exe analysis/feature_ablation_cnn.py --aggregate outputs/results/feature_ablation_cnn
```

Computes mean ΔPR-AUC across seeds with 95% bootstrap CI per SID cluster
and writes `aggregate_*.json` with the verdict through the 3 pre-registered
decision branches. **Do NOT run the aggregator with fewer than 3 seeds. Read the CI
once; no mining, no re-run.**
