# analysis/fuelmap_ablation_cnn.py
"""
Paired CNN ablation of FuelMap physics-guided losses (PREPARED, never auto-runs).

Question: do the FuelMap physics-guided losses (prior alignment, forward
constraint, total-variation, L1, and equation consistency) help or hurt RI
classification when compared head-to-head in a 3D-CNN architecture, evaluated
with the same SID-grouped cross-validation protocol as ``analysis/feature_ablation_cnn.py``,
but using the real trainer (``src/training/trainer.py::train``) with physics
losses toggled?

Arms
----
  A_physics_on   = model trained WITH FuelMap physics losses as configured in
                   config.yaml (production behavior; the 5 lambda parameters must
                   be present and at least one must be > 0).
  B_physics_off  = identical model/channels trained with ALL physics lambdas = 0.0
                   (plain 3D-CNN classifier; the trainer docstring documents this
                   configuration); regression targets are still provided but not used.

Design
------
* SID-grouped folds over DEV events only (train+val; test is never loaded).
  Fold construction reuses the exact StratifiedGroupKFold recipe from
  feature_ablation_kfold.py, run on all development events.
* Training goes through the SAME trainer entry point the project releases,
  ``src.training.trainer.train(cfg)`` -- imported and called, never
  reimplemented. Evaluation of the trained checkpoint reuses the trainer's
  own model-building / batch-moving / logit-extraction helpers
  (``trainer._build_model``, ``trainer._to_device``,
  ``trainer._extract_ri_logit``) for pure inference; no gradient step is
  duplicated.
* Per-fold, per-arm normalization statistics are recomputed from scratch on
  THAT fold's training portion only, via
  ``src.data.normalization.compute_norm_stats_from_splits``. They are
  written under the run directory and NEVER touch the project's global
  ``data/normalized/normalization_stats.json``.
* Per-fold, per-arm checkpoints/history/results are written under
  ``outputs/results/fuelmap_ablation_cnn/{run_id}/seed{s}/fold{k}/{arm}/``
  -- never under the project's global ``models/checkpoints`` or
  ``outputs/results``.
* Paired verdict machinery identical in spirit to feature_ablation_kfold.py:
  per-fold deltas (B-A, where B=physics_off, A=physics_on) for PR-AUC/ROC-AUC,
  plus a cluster (by-SID) bootstrap over pooled out-of-fold predictions.
* Gated like feature_ablation_kfold.py: refuses to run unless
  outputs/results/pl_gate_census.json reports gate_pass=true (unless
  --no-require-gate).
* --dry-run is the DEFAULT: it prints the run plan (event/positive counts
  per fold, #trainings, a wall-time estimate) and exits without touching
  the trainer, a checkpoint, or the interim data cubes at all beyond the
  cheap metadata scan needed to build the plan. Real training only happens
  with --execute.
* REUSE MODE (--reuse-arm-a PATH [PATH...]): load prob_A (physics-on
  predictions) from prior feature_ablation_cnn run directories, validate
  that the fold splits are identical, then train ONLY arm B (physics-off)
  for each (seed, fold). This allows running arm B overnight without
  retraining arm A (which is already in the prior run).

Usage:
    # Safe by default -- prints the plan only.
    python analysis/fuelmap_ablation_cnn.py --folds 3 --epochs 15 --seeds 42

    # Actually trains both arms (expensive; folds x seeds x 2 full trainings).
    python analysis/fuelmap_ablation_cnn.py --folds 3 --epochs 15 --seeds 42 --execute

    # Reuse arm-A predictions from a prior run, train only arm B.
    python analysis/fuelmap_ablation_cnn.py --folds 3 --epochs 15 --seeds 42 \\
        --reuse-arm-a outputs/results/feature_ablation_cnn/20260713T232126Z \\
        --execute

Pre-registration
----------------
See docs/fuelmap_ablation_preregistration.md for the pre-registered hypothesis
and decision procedure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from analysis.feature_ablation_cnn import (  # noqa: E402
    aggregate_across_seeds as _aggregate_base,
    build_folds,
    build_run_plan,
    cluster_bootstrap_ci,
    enforce_gate,
    load_pl_gated_dev_events,
    _build_arm_cfg,
    _evaluate_checkpoint_on_val,
    _write_fold_splits_csv,
)
from src.utils.config import cfg_get, load_config  # noqa: E402
from src.utils.paths import rel_to_root  # noqa: E402

ARM_A_NAME = "A_physics_on"
ARM_B_NAME = "B_physics_off"
PHYSICS_LAMBDA_KEYS = [
    "lambda_prior_align",
    "lambda_forward",
    "lambda_tv",
    "lambda_l1",
    "lambda_consistency",
]
RESULTS_SUBDIR = "fuelmap_ablation_cnn"


# ---------------------------------------------------------------------------
# Channel resolution.
# ---------------------------------------------------------------------------

def resolve_channels(cfg: Dict[str, Any]) -> List[str]:
    """Return the production input channel names from config.

    Both arms (physics on/off) use the same channel set.
    """
    channels = list(cfg_get(cfg, "model.input_channels_names", []))
    if not channels:
        raise ValueError("config.yaml must define model.input_channels_names")
    return channels


# ---------------------------------------------------------------------------
# Physics-aware arm configuration builder.
# ---------------------------------------------------------------------------

def _build_physics_arm_cfg(cfg: Dict[str, Any], channels: List[str], splits_csv: Path,
                          norm_stats_path: Path, arm_dir: Path, epochs: int, seed: int,
                          physics_on: bool) -> Dict[str, Any]:
    """Build a physics-toggled cfg for one (fold, seed, arm) run.

    Reuses _build_arm_cfg to construct the base scoped config, then:
      - if physics_on: validates that at least one of PHYSICS_LAMBDA_KEYS has weight > 0
        (else raises RuntimeError).
      - if not physics_on: sets all PHYSICS_LAMBDA_KEYS to 0.0 in the training.physics dict.

    Returns the cfg_copy with effective physics settings recorded in
    cfg_copy["training"]["physics"] for the audit trail.
    """
    cfg_copy = _build_arm_cfg(cfg, channels, splits_csv, norm_stats_path,
                             arm_dir, epochs, seed)

    physics_dict = cfg_copy.setdefault("training", {}).setdefault("physics", {})

    if physics_on:
        # Validate that arm A actually has physics on.
        has_nonzero_lambda = any(
            physics_dict.get(key, 0.0) > 0.0 for key in PHYSICS_LAMBDA_KEYS
        )
        if not has_nonzero_lambda:
            raise RuntimeError(
                f"Arm A (physics_on) must have at least one physics lambda > 0, but all are zero. "
                f"Check config.yaml training.physics settings. Current physics: {physics_dict}"
            )
    else:
        # Set all lambdas to 0 for arm B.
        for key in PHYSICS_LAMBDA_KEYS:
            physics_dict[key] = 0.0

    return cfg_copy


# ---------------------------------------------------------------------------
# Run-plan formatting (reuses and refines the original).
# ---------------------------------------------------------------------------

def format_physics_run_plan_table(plan: Dict[str, Any], channels: List[str],
                                  mode: str, reuse_sources: Optional[Dict[int, Path]] = None) -> str:
    """Format the run plan with FuelMap-specific information."""
    lines = []
    lines.append("=" * 72)
    lines.append("FUELMAP ABLATION CNN -- RUN PLAN (dry-run; pass --execute to actually train)")
    lines.append("=" * 72)
    lines.append(f"dev events: n={plan['n_dev_events_pl_gated']} "
                 f"positives={plan['n_positives']} storms={plan['n_storms']}")
    lines.append(f"channels (both arms): {channels}")
    lines.append(f"mode: {mode}")
    if mode == "reuse_arm_a" and reuse_sources:
        for seed, src in reuse_sources.items():
            lines.append(f"  seed {seed}: reuse from {rel_to_root(src)}")
    lines.append(f"arm {ARM_A_NAME}: {"(REUSED from prior run)" if mode == "reuse_arm_a" else "train from scratch"}")
    lines.append(f"arm {ARM_B_NAME}: train (physics lambdas = 0.0)")
    if mode == "reuse_arm_a":
        lines.append(f"#trainings = folds x seeds x 1 (arm B only) = {plan['n_trainings'] // 2}")
    else:
        lines.append(f"folds={plan['folds']}  seeds={plan['seeds']}  arms=2")
        lines.append(f"#trainings = folds x seeds x 2 = {plan['n_trainings']}")
    lines.append(f"epochs/training={plan['epochs_per_training']}  "
                 f"minutes/epoch(assumed)={plan['minutes_per_epoch_assumed']}")
    if mode == "reuse_arm_a":
        est_min = (plan['n_trainings'] // 2) * plan['epochs_per_training'] * plan['minutes_per_epoch_assumed']
        lines.append(f"estimated wall time (arm B only) = {est_min:.0f} min (~{est_min / 60.0:.1f} h)")
    else:
        lines.append(f"estimated wall time = #trainings x epochs x minutes/epoch = "
                     f"{plan['estimated_wall_minutes']:.0f} min (~{plan['estimated_wall_hours']:.1f} h)")
    lines.append("-" * 72)
    lines.append(f"{'seed':>6} {'fold':>6} {'n_train':>9} {'n_val':>7} {'n_pos_val':>10}")
    for row in plan["per_fold"]:
        lines.append(f"{row['seed']:>6} {row['fold']:>6} {row['n_train']:>9} "
                     f"{row['n_val']:>7} {row['n_pos_val']:>10}")
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reuse mode: load and validate prior arm-A predictions.
# ---------------------------------------------------------------------------

def _resolve_reuse_csv(reuse_sources: List[Path], seed: int) -> Tuple[Path, List[Path]]:
    """Resolve a single oof_predictions.csv for the given seed from reuse_sources.

    Returns (oof_csv_path, fold_splits_csv_list).

    Raises ValueError if 0 or >1 matches found for the seed.
    """
    matches = []
    for src_dir in reuse_sources:
        p = src_dir / f"seed{seed}" / "oof_predictions.csv"
        if p.exists():
            matches.append(p)

    if len(matches) == 0:
        raise ValueError(
            f"No oof_predictions.csv found for seed {seed} in reuse sources: {reuse_sources}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous: multiple oof_predictions.csv found for seed {seed}: {matches}. "
            "Disambiguate by passing one reuse source per seed."
        )

    oof_csv = matches[0]
    seed_dir = oof_csv.parent
    folds_list = []
    fold_k = 0
    while True:
        splits_csv = seed_dir / f"fold{fold_k}" / "splits.csv"
        if not splits_csv.exists():
            break
        folds_list.append(splits_csv)
        fold_k += 1

    return oof_csv, folds_list


def _validate_reuse_folds(events_df: pd.DataFrame, folds: List[Tuple[np.ndarray, np.ndarray]],
                         fold_splits_csvs: List[Path], oof_csv: Path, seed: int) -> None:
    """Validate that rebuilt folds match the saved fold splits from the prior
    run, and that the reused OOF labels agree with the current dev labels.

    Raises ValueError on any mismatch.
    """
    oof_df = pd.read_csv(oof_csv)
    merged = oof_df.merge(events_df[["event_id", "ri_label"]], on="event_id", how="inner")
    if len(merged) != len(oof_df):
        raise ValueError(
            f"Seed {seed}: {len(oof_df) - len(merged)} events in the reused OOF csv "
            "are not in the current dev set — the reuse assumption is broken."
        )
    if not np.array_equal(merged["y"].to_numpy(dtype=np.float64),
                          merged["ri_label"].to_numpy(dtype=np.float64)):
        raise ValueError(
            f"Seed {seed}: labels in the reused OOF csv disagree with the current "
            "dev ri_label — the reuse assumption is broken."
        )

    if len(fold_splits_csvs) != len(folds):
        raise ValueError(
            f"Fold count mismatch: current build has {len(folds)} folds, "
            f"but reuse source has {len(fold_splits_csvs)}. "
            f"Ensure --folds matches the prior run."
        )

    for fold_k, (tr, te) in enumerate(folds):
        # Build what we would write.
        our_train_ids = set(events_df.iloc[tr]["event_id"].tolist())
        our_val_ids = set(events_df.iloc[te]["event_id"].tolist())

        # Load what was saved.
        saved_df = pd.read_csv(fold_splits_csvs[fold_k])
        saved_train_ids = set(saved_df[saved_df["split"] == "train"]["event_id"].tolist())
        saved_val_ids = set(saved_df[saved_df["split"] == "val"]["event_id"].tolist())

        if our_train_ids != saved_train_ids or our_val_ids != saved_val_ids:
            raise ValueError(
                f"Fold {fold_k} (seed {seed}): train/val split mismatch with reuse source. "
                f"This breaks the reuse assumption (fold construction is deterministic). "
                f"Ensure build_folds(same events_df, same folds, same seed) is used."
            )


# ---------------------------------------------------------------------------
# Execution (only reached with --execute).
# ---------------------------------------------------------------------------

def run_execute(cfg: Dict[str, Any], config_path: Path, events_df: pd.DataFrame,
               folds_by_seed: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
               channels: List[str], args: argparse.Namespace,
               reuse_data: Optional[Dict[int, Tuple[Path, List[Path]]]] = None) -> Dict[str, Any]:
    from src.data.normalization import compute_norm_stats_from_splits
    from src.training import trainer

    interim_dir = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    results_root = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = results_root / RESULTS_SUBDIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    sid_by_event = events_df.set_index("event_id")["sid"].astype(str).to_dict()

    per_seed_results: Dict[str, Any] = {}
    reuse_sources_recorded: Dict[int, str] = {}

    for seed in args.seeds:
        folds = folds_by_seed[seed]
        fold_rows = []
        oof_y: Dict[str, float] = {}
        oof_a: Dict[str, float] = {}
        oof_b: Dict[str, float] = {}

        # Load reused arm-A predictions if in reuse mode.
        reuse_oof_csv_path: Optional[Path] = None
        if reuse_data and seed in reuse_data:
            reuse_oof_csv_path, _ = reuse_data[seed]

        for fold_k, (tr, te) in enumerate(folds):
            fold_dir = run_dir / f"seed{seed}" / f"fold{fold_k}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            splits_csv = fold_dir / "splits.csv"
            _write_fold_splits_csv(events_df, tr, te, splits_csv)

            fold_row: Dict[str, Any] = {
                "fold": fold_k, "n_train": int(len(tr)), "n_val": int(len(te)),
                "n_pos_val": int(events_df.iloc[te]["ri_label"].sum()),
                "arms": {},
            }

            fold_val_ids = set(events_df.iloc[te]["event_id"].tolist())

            # Arm A: physics on. In reuse mode, load from prior run; otherwise train.
            if reuse_oof_csv_path:
                # Extract arm-A probs for this fold's val events; per-fold metrics
                # are recomputed from the reused predictions so the fold report
                # stays complete (diff_pr_auc below needs arm A's pr_auc).
                reuse_df = pd.read_csv(reuse_oof_csv_path)
                fold_reuse = reuse_df[reuse_df["event_id"].isin(fold_val_ids)]
                y_true = fold_reuse["y"].to_numpy(dtype=np.float64)
                y_prob = fold_reuse["prob_A"].to_numpy(dtype=np.float64)
                has_both_classes = len(np.unique(y_true)) > 1
                pr_auc = float(average_precision_score(y_true, y_prob)) if has_both_classes else None
                roc_auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else None
                for _, row in fold_reuse.iterrows():
                    eid = row["event_id"]
                    oof_y[eid] = row["y"]
                    oof_a[eid] = row["prob_A"]
                arm_result_a: Dict[str, Any] = {
                    "arm": ARM_A_NAME, "channels": channels,
                    "source": "reused",
                    "reused_from": rel_to_root(reuse_oof_csv_path),
                    "pr_auc": pr_auc, "roc_auc": roc_auc,
                }
            else:
                # Train arm A from scratch.
                arm_dir = fold_dir / ARM_A_NAME
                arm_dir.mkdir(parents=True, exist_ok=True)
                norm_stats_path = arm_dir / "normalization_stats.json"
                compute_norm_stats_from_splits(
                    interim_dir=interim_dir, splits_csv=splits_csv,
                    out_path=norm_stats_path, input_names=channels,
                )
                cfg_arm = _build_physics_arm_cfg(cfg, channels, splits_csv, norm_stats_path,
                                               arm_dir, args.epochs, seed, physics_on=True)

                t0 = time.time()
                train_summary = trainer.train(cfg_arm)
                elapsed_sec = time.time() - t0

                device = trainer._resolve_device(cfg_arm)
                y_true, y_prob, ids = _evaluate_checkpoint_on_val(cfg_arm, device)
                has_both_classes = len(np.unique(y_true)) > 1
                pr_auc = float(average_precision_score(y_true, y_prob)) if has_both_classes else None
                roc_auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else None

                for eid, yt, yp in zip(ids, y_true.tolist(), y_prob.tolist()):
                    oof_y[eid] = yt
                    oof_a[eid] = yp

                # Extract and record effective physics lambdas for the audit trail.
                physics_dict = cfg_arm.get("training", {}).get("physics", {})
                physics_effective = {k: physics_dict.get(k, 0.0) for k in PHYSICS_LAMBDA_KEYS}

                arm_result_a = {
                    "arm": ARM_A_NAME, "channels": channels,
                    "elapsed_sec": elapsed_sec, "pr_auc": pr_auc, "roc_auc": roc_auc,
                    "physics_lambdas_effective": physics_effective,
                    "train_summary": train_summary,
                    "normalization_stats_path": rel_to_root(norm_stats_path),
                    "checkpoints_dir": rel_to_root(arm_dir / "checkpoints"),
                }
                (arm_dir / "ablation_eval.json").write_text(json.dumps(arm_result_a, indent=2), encoding="utf-8")

            fold_row["arms"][ARM_A_NAME] = {k: v for k, v in arm_result_a.items() if k != "train_summary"}

            # Arm B: physics off (always train, even in reuse mode).
            arm_dir = fold_dir / ARM_B_NAME
            arm_dir.mkdir(parents=True, exist_ok=True)
            norm_stats_path = arm_dir / "normalization_stats.json"
            compute_norm_stats_from_splits(
                interim_dir=interim_dir, splits_csv=splits_csv,
                out_path=norm_stats_path, input_names=channels,
            )
            cfg_arm = _build_physics_arm_cfg(cfg, channels, splits_csv, norm_stats_path,
                                           arm_dir, args.epochs, seed, physics_on=False)

            t0 = time.time()
            train_summary = trainer.train(cfg_arm)
            elapsed_sec = time.time() - t0

            device = trainer._resolve_device(cfg_arm)
            y_true, y_prob, ids = _evaluate_checkpoint_on_val(cfg_arm, device)
            has_both_classes = len(np.unique(y_true)) > 1
            pr_auc = float(average_precision_score(y_true, y_prob)) if has_both_classes else None
            roc_auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else None

            for eid, yt, yp in zip(ids, y_true.tolist(), y_prob.tolist()):
                if eid not in oof_y:
                    oof_y[eid] = yt
                oof_b[eid] = yp

            # Extract and record effective physics lambdas for the audit trail (all should be 0).
            physics_dict = cfg_arm.get("training", {}).get("physics", {})
            physics_effective = {k: physics_dict.get(k, 0.0) for k in PHYSICS_LAMBDA_KEYS}

            arm_result_b = {
                "arm": ARM_B_NAME, "channels": channels,
                "elapsed_sec": elapsed_sec, "pr_auc": pr_auc, "roc_auc": roc_auc,
                "physics_lambdas_effective": physics_effective,
                "train_summary": train_summary,
                "normalization_stats_path": rel_to_root(norm_stats_path),
                "checkpoints_dir": rel_to_root(arm_dir / "checkpoints"),
            }
            (arm_dir / "ablation_eval.json").write_text(json.dumps(arm_result_b, indent=2), encoding="utf-8")
            fold_row["arms"][ARM_B_NAME] = {k: v for k, v in arm_result_b.items() if k != "train_summary"}

            fold_row["diff_pr_auc_B_minus_A"] = (
                fold_row["arms"][ARM_B_NAME]["pr_auc"] - fold_row["arms"][ARM_A_NAME]["pr_auc"]
                if fold_row["arms"][ARM_A_NAME]["pr_auc"] is not None
                and fold_row["arms"][ARM_B_NAME]["pr_auc"] is not None else None
            )
            fold_rows.append(fold_row)

        if reuse_oof_csv_path:
            reuse_sources_recorded[seed] = rel_to_root(reuse_oof_csv_path)

        common_ids = [eid for eid in oof_y if eid in oof_a and eid in oof_b]
        y_pool = np.array([oof_y[e] for e in common_ids])
        a_pool = np.array([oof_a[e] for e in common_ids])
        b_pool = np.array([oof_b[e] for e in common_ids])
        groups_pool = np.array([sid_by_event[e] for e in common_ids])

        # Persist the raw OOF predictions (same format as feature_ablation_cnn for compatibility).
        oof_csv = run_dir / f"seed{seed}" / "oof_predictions.csv"
        pd.DataFrame({
            "event_id": common_ids,
            "sid": groups_pool,
            "y": y_pool,
            "prob_A": a_pool,
            "prob_B": b_pool,
        }).to_csv(oof_csv, index=False)

        bootstrap = cluster_bootstrap_ci(y_pool, a_pool, b_pool, groups_pool,
                                         seed=seed, n_boot=args.n_boot)
        pooled = {
            "n_pooled_events": int(len(common_ids)),
            "A_pr_auc": float(average_precision_score(y_pool, a_pool)) if len(np.unique(y_pool)) > 1 else None,
            "A_roc_auc": float(roc_auc_score(y_pool, a_pool)) if len(np.unique(y_pool)) > 1 else None,
            "B_pr_auc": float(average_precision_score(y_pool, b_pool)) if len(np.unique(y_pool)) > 1 else None,
            "B_roc_auc": float(roc_auc_score(y_pool, b_pool)) if len(np.unique(y_pool)) > 1 else None,
        }

        per_seed_results[str(seed)] = {
            "per_fold": fold_rows,
            "pooled_oof": pooled,
            "cluster_bootstrap_by_sid": bootstrap,
            "oof_predictions_csv": rel_to_root(oof_csv),
        }

    summary = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": rel_to_root(config_path),
        "mode": "reuse_arm_a" if reuse_data else "fresh_both_arms",
        "channels": channels,
        "folds": args.folds,
        "epochs": args.epochs,
        "seeds": args.seeds,
        "n_boot": args.n_boot,
        "n_events": int(len(events_df)),
        "n_positives": int(events_df["ri_label"].sum()),
        "n_storms": int(events_df["sid"].nunique()),
        "per_seed": per_seed_results,
        "run_dir": rel_to_root(run_dir),
    }

    if reuse_sources_recorded:
        summary["reuse_sources"] = reuse_sources_recorded

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Cross-seed aggregation (the pre-registered final verdict).
# ---------------------------------------------------------------------------

def aggregate_across_seeds(oof_csvs: Sequence[Path], n_boot: int = 10_000,
                           boot_seed: int = 42, ci: float = 0.95) -> Dict[str, Any]:
    """Pre-registered final metric over phased per-seed runs.

    Delegates the whole point-estimate / SID-cluster-bootstrap machinery to
    feature_ablation_cnn.aggregate_across_seeds (single implementation, no
    drift between the two experiments) and replaces only the verdict wording
    with this experiment's pre-registered branches. Here A=physics_on,
    B=physics_off, so delta (B-A) > 0 means the physics losses HURT.
    """
    result = _aggregate_base(oof_csvs, n_boot=n_boot, boot_seed=boot_seed, ci=ci)
    result["experiment"] = "fuelmap_physics_loss_ablation"
    result["arms"] = {"A": "physics_on (production lambdas)", "B": "physics_off (all lambdas 0.0)"}

    # Per-arm absolute PR-AUCs alongside the delta — a verdict report that
    # shows only the delta hides the arms themselves.
    per_seed_absolute: Dict[str, Dict[str, float]] = {}
    for p in oof_csvs:
        f = pd.read_csv(Path(p))
        y = f["y"].to_numpy(dtype=np.float64)
        seed_key = Path(p).parent.name  # e.g. "seed42"
        per_seed_absolute[seed_key] = {
            "A_pr_auc": float(average_precision_score(y, f["prob_A"].to_numpy(dtype=np.float64))),
            "B_pr_auc": float(average_precision_score(y, f["prob_B"].to_numpy(dtype=np.float64))),
        }
    result["per_seed_absolute"] = per_seed_absolute
    result["A_pr_auc_mean"] = float(np.mean([v["A_pr_auc"] for v in per_seed_absolute.values()]))
    result["B_pr_auc_mean"] = float(np.mean([v["B_pr_auc"] for v in per_seed_absolute.values()]))

    lo, hi = result["delta_pr_auc_ci_low"], result["delta_pr_auc_ci_high"]
    if lo is None:
        result["verdict"] = "INCONCLUSIVE: insufficient valid bootstrap draws to compute a CI."
    elif lo > 0:
        result["verdict"] = (
            f"CI [{lo:.4f}, {hi:.4f}] excludes zero (positive): the FuelMap physics losses "
            "HURT RI classification — pre-registered action: remove them from the production model."
        )
    elif hi < 0:
        result["verdict"] = (
            f"CI [{lo:.4f}, {hi:.4f}] excludes zero (negative): the FuelMap physics losses "
            "HELP RI classification — pre-registered action: keep them, reframed as regularization "
            "(NOT as validated physics; H1 refutation stands)."
        )
    else:
        result["verdict"] = (
            f"CI [{lo:.4f}, {hi:.4f}] includes zero: NULL — no detectable effect; "
            "pre-registered action: remove for parsimony (refuted semantics + no measurable gain)."
        )

    return result


# ---------------------------------------------------------------------------
# Main CLI.
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml (relative paths resolve against the project root).")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated seeds, e.g. 42,43,44")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--minutes-per-epoch", type=float, default=3.0,
                        help="Assumed minutes/epoch used only for the dry-run wall-time estimate.")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True,
                        help="Print the run plan and exit (default: on).")
    parser.add_argument("--execute", action="store_true", default=False,
                        help="Actually run the trainings. Overrides --dry-run.")
    parser.add_argument("--require-gate", action=argparse.BooleanOptionalAction, default=True,
                        help="Refuse to run unless outputs/results/pl_gate_census.json reports "
                             "gate_pass=true (default: on). Use --no-require-gate to bypass.")
    parser.add_argument("--reuse-arm-a", nargs="+", default=None, metavar="PATH",
                        help="Prior feature_ablation_cnn run directories. For each seed, "
                             "locate the oof_predictions.csv and corresponding fold splits, "
                             "validate fold identity, then train only arm B (physics_off). "
                             "Incompatible with --aggregate.")
    parser.add_argument("--aggregate", nargs="+", default=None, metavar="PATH",
                        help="Per-seed run dirs (searched recursively) or oof_predictions.csv "
                             "paths. Computes the pre-registered cross-seed verdict from saved "
                             "OOF predictions and exits -- no training, no gate needed.")
    args = parser.parse_args()
    args.seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    cfg = load_config(str(config_path))

    if args.aggregate:
        csvs: List[Path] = []
        for a in args.aggregate:
            p = Path(a)
            csvs.extend(sorted(p.rglob("oof_predictions.csv")) if p.is_dir() else [p])
        if not csvs:
            print("No oof_predictions.csv found under the given paths.")
            sys.exit(1)
        result = aggregate_across_seeds(csvs, n_boot=args.n_boot)
        out_dir = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve() / RESULTS_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"aggregate_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"\nFINAL VERDICT: {result['verdict']}")
        print(f"report: {out_path}")
        return

    enforce_gate(cfg, args.require_gate)

    events_df, dev_audit = load_pl_gated_dev_events(cfg)
    print("DEV AUDIT (PL-gated):", json.dumps(dev_audit, indent=2))

    if len(events_df) == 0 or events_df["ri_label"].sum() < args.folds:
        print("Not enough PL-gated dev events/positives for the requested fold count -- aborting.")
        sys.exit(1)

    channels = resolve_channels(cfg)
    folds_by_seed = {s: build_folds(events_df, args.folds, s) for s in args.seeds}

    # Handle reuse mode.
    reuse_data: Optional[Dict[int, Tuple[Path, List[Path]]]] = None
    if args.reuse_arm_a:
        reuse_sources = [Path(p).resolve() for p in args.reuse_arm_a]
        reuse_data = {}
        print("REUSE MODE: validating prior run fold structures...")
        for seed in args.seeds:
            try:
                oof_csv, fold_splits_csvs = _resolve_reuse_csv(reuse_sources, seed)
                _validate_reuse_folds(events_df, folds_by_seed[seed], fold_splits_csvs, oof_csv, seed)
                reuse_data[seed] = (oof_csv, fold_splits_csvs)
                print(f"  seed {seed}: validated ({len(fold_splits_csvs)} folds, {oof_csv})")
            except ValueError as e:
                print(f"  seed {seed}: VALIDATION FAILED — {e}")
                sys.exit(1)

    plan = build_run_plan(events_df, folds_by_seed, args.seeds, args.folds,
                          args.epochs, args.minutes_per_epoch)

    mode = "reuse_arm_a" if reuse_data else "fresh_both_arms"
    reuse_sources_for_display = None
    if reuse_data:
        reuse_sources_for_display = {
            seed: reuse_data[seed][0] for seed in reuse_data
        }
    print(format_physics_run_plan_table(plan, channels, mode, reuse_sources_for_display))

    # Execution requires the EXPLICIT --execute flag.
    if not args.execute:
        print("\nDRY RUN: no training was performed. Pass --execute to actually train.")
        return

    print("\nEXECUTING: this will run real trainings via src.training.trainer.train(). "
          "This may take a long time -- see the wall-time estimate above.")
    summary = run_execute(cfg, config_path, events_df, folds_by_seed, channels, args, reuse_data)
    print(f"\nrun_dir: {summary['run_dir']}")


if __name__ == "__main__":
    main()
