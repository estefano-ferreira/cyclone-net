# analysis/feature_ablation_cnn.py
"""
Paired CNN feature-ablation runner (PREPARED, never auto-runs).

Question: does adding the two pressure-level (PL) channels
(``shear_850_200_mps``, ``rh_mid``) to the RELEASED CycloneNet CNN's input
tensor improve RI discrimination, evaluated with the same SID-grouped
cross-validation protocol as ``analysis/feature_ablation_kfold.py``, but
using the real trainer (``src/training/trainer.py::train``) instead of the
tabular logistic-regression surrogate?

Arms
----
  A_current       = model.input_channels_names, as configured in config.yaml
                     (the released 9-channel surface set).
  B_with_pressure  = A_current + [shear_850_200_mps, rh_mid]

Design
------
* SID-grouped folds over DEV events only (train+val; test is never loaded).
  Fold construction reuses the exact StratifiedGroupKFold recipe from
  feature_ablation_kfold.py, restricted to events whose metadata has BOTH
  PL channels (so A and B are compared on identical events) -- identical
  folds are used for arm A and arm B within a given (seed, --folds N).
* Training goes through the SAME trainer entry point the project releases,
  ``src.training.trainer.train(cfg)`` -- imported and called, never
  reimplemented. Evaluation of the trained checkpoint reuses the trainer's
  own model-building / batch-moving / logit-extraction helpers
  (``trainer._build_model``, ``trainer._to_device``,
  ``trainer._extract_ri_logit``) for pure inference; no gradient step is
  duplicated.
* Per-fold, per-arm normalization statistics are recomputed from scratch on
  THAT fold's training portion only, for the arm's channel set, via
  ``src.data.normalization.compute_norm_stats_from_splits``. They are
  written under the run directory and NEVER touch the project's global
  ``data/normalized/normalization_stats.json``.
* Per-fold, per-arm checkpoints/history/results are written under
  ``outputs/results/feature_ablation_cnn/{run_id}/seed{s}/fold{k}/{arm}/``
  -- never under the project's global ``models/checkpoints`` or
  ``outputs/results``.
* Paired verdict machinery identical in spirit to feature_ablation_kfold.py:
  per-fold deltas (B-A) for PR-AUC/ROC-AUC, plus a cluster (by-SID)
  bootstrap over pooled out-of-fold predictions.
* Gated like feature_ablation_kfold.py: refuses to run unless
  outputs/results/pl_gate_census.json reports gate_pass=true (unless
  --no-require-gate).
* --dry-run is the DEFAULT: it prints the run plan (event/positive counts
  per fold, #trainings, a wall-time estimate) and exits without touching
  the trainer, a checkpoint, or the interim data cubes at all beyond the
  cheap metadata scan needed to build the plan. Real training only happens
  with --execute.

Usage:
    # Safe by default -- prints the plan only.
    python analysis/feature_ablation_cnn.py --folds 3 --epochs 15 --seeds 42

    # Actually trains (expensive; folds x seeds x 2 full trainings).
    python analysis/feature_ablation_cnn.py --folds 3 --epochs 15 --seeds 42 --execute
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from analysis.feature_ablation_kfold import (  # noqa: E402
    PL_CHANNELS,
    cluster_bootstrap_ci,
    enforce_gate,
    load_dev_events,
)
from src.utils.config import cfg_get, load_config  # noqa: E402
from src.utils.paths import rel_to_root  # noqa: E402

ARM_A_NAME = "A_current"
ARM_B_NAME = "B_with_pressure"


# ---------------------------------------------------------------------------
# Dev-event selection (metadata only -- no cubes read here).
# ---------------------------------------------------------------------------

def load_pl_gated_dev_events(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Dev events (train+val) restricted to those whose metadata has BOTH PL
    channels, so arm A and arm B are compared on exactly the same events.

    Reads only the per-event ``.json`` metadata sidecar (cheap) -- never a
    ``.npy`` cube. The test split is never touched (inherited from
    ``load_dev_events``).
    """
    interim = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    dev = load_dev_events(cfg)

    keep_ids: List[str] = []
    n_missing = 0
    n_no_pl = 0
    for r in dev.itertuples():
        meta_path = interim / f"{r.event_id}.json"
        if not meta_path.exists():
            n_missing += 1
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        channels = list(meta.get("channels", []))
        if not all(ch in channels for ch in PL_CHANNELS):
            n_no_pl += 1
            continue
        keep_ids.append(r.event_id)

    out = dev[dev["event_id"].isin(keep_ids)].reset_index(drop=True)
    audit = {
        "n_dev_events": int(len(dev)),
        "n_used": int(len(out)),
        "n_missing_artifact": n_missing,
        "n_without_pressure_channels": n_no_pl,
        "n_positives_used": int(out["ri_label"].sum()) if len(out) else 0,
        "n_storms_used": int(out["sid"].nunique()) if len(out) else 0,
    }
    return out, audit


def resolve_arms(cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    """Arm A = configured input channels; Arm B = A + PL channels."""
    base_channels = list(cfg_get(cfg, "model.input_channels_names", []))
    if not base_channels:
        raise ValueError("config.yaml must define model.input_channels_names")
    arm_a = list(base_channels)
    arm_b = list(base_channels) + [c for c in PL_CHANNELS if c not in base_channels]
    return {ARM_A_NAME: arm_a, ARM_B_NAME: arm_b}


def build_folds(events_df: pd.DataFrame, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Identical SID-grouped fold construction recipe as feature_ablation_kfold.py.

    X is a dummy zero column: StratifiedGroupKFold only uses its length, not
    its values -- the real event features are irrelevant to fold assignment.
    """
    y = events_df["ri_label"].to_numpy()
    groups = events_df["sid"].astype(str).to_numpy()
    x_dummy = np.zeros((len(y), 1))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(x_dummy, y, groups=groups))


# ---------------------------------------------------------------------------
# Run-plan construction (used by both --dry-run and --execute).
# ---------------------------------------------------------------------------

def build_run_plan(
    events_df: pd.DataFrame,
    folds_by_seed: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
    seeds: Sequence[int],
    n_folds: int,
    epochs: int,
    minutes_per_epoch: float,
) -> Dict[str, Any]:
    y = events_df["ri_label"].to_numpy()
    rows = []
    for seed in seeds:
        for fold_k, (tr, te) in enumerate(folds_by_seed[seed]):
            rows.append({
                "seed": seed, "fold": fold_k,
                "n_train": int(len(tr)), "n_val": int(len(te)),
                "n_pos_val": int(y[te].sum()),
            })
    n_trainings = len(seeds) * n_folds * 2
    est_minutes = n_trainings * epochs * minutes_per_epoch
    return {
        "n_dev_events_pl_gated": int(len(events_df)),
        "n_positives": int(events_df["ri_label"].sum()) if len(events_df) else 0,
        "n_storms": int(events_df["sid"].nunique()) if len(events_df) else 0,
        "folds": n_folds,
        "seeds": list(seeds),
        "epochs_per_training": epochs,
        "minutes_per_epoch_assumed": minutes_per_epoch,
        "n_trainings": n_trainings,
        "estimated_wall_minutes": est_minutes,
        "estimated_wall_hours": est_minutes / 60.0,
        "per_fold": rows,
    }


def format_run_plan_table(plan: Dict[str, Any], arms: Dict[str, List[str]]) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("FEATURE-ABLATION CNN -- RUN PLAN (dry-run; pass --execute to actually train)")
    lines.append("=" * 72)
    lines.append(f"dev events (PL-gated): n={plan['n_dev_events_pl_gated']} "
                 f"positives={plan['n_positives']} storms={plan['n_storms']}")
    lines.append(f"arm {ARM_A_NAME}: {arms[ARM_A_NAME]}")
    lines.append(f"arm {ARM_B_NAME}: {arms[ARM_B_NAME]}")
    lines.append(f"folds={plan['folds']}  seeds={plan['seeds']}  arms=2")
    lines.append(f"epochs/training={plan['epochs_per_training']}  "
                 f"minutes/epoch(assumed)={plan['minutes_per_epoch_assumed']}")
    lines.append(f"#trainings = folds x seeds x 2 = {plan['n_trainings']}")
    lines.append(f"estimated wall time = #trainings x epochs x minutes/epoch = "
                 f"{plan['estimated_wall_minutes']:.0f} min "
                 f"(~{plan['estimated_wall_hours']:.1f} h)")
    lines.append("-" * 72)
    lines.append(f"{'seed':>6} {'fold':>6} {'n_train':>9} {'n_val':>7} {'n_pos_val':>10}")
    for row in plan["per_fold"]:
        lines.append(f"{row['seed']:>6} {row['fold']:>6} {row['n_train']:>9} "
                     f"{row['n_val']:>7} {row['n_pos_val']:>10}")
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Execution (only reached with --execute).
# ---------------------------------------------------------------------------

def _write_fold_splits_csv(events_df: pd.DataFrame, train_idx: np.ndarray,
                           test_idx: np.ndarray, out_path: Path) -> None:
    """Scoped splits.csv for one fold: fold-train -> 'train', held-out -> 'val'.

    Written under the run directory only -- never overwrites
    data/normalized/splits.csv.
    """
    rows = []
    for i in train_idx:
        rows.append({"event_id": events_df.iloc[i]["event_id"], "split": "train"})
    for i in test_idx:
        rows.append({"event_id": events_df.iloc[i]["event_id"], "split": "val"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _build_arm_cfg(cfg: Dict[str, Any], arm_channels: List[str], splits_csv: Path,
                   norm_stats_path: Path, arm_dir: Path, epochs: int, seed: int) -> Dict[str, Any]:
    """In-memory cfg copy scoped to one (fold, seed, arm) run.

    Never mutates the caller's cfg; every path override stays under arm_dir
    except paths.interim_data (read-only cube/metadata source, shared and
    untouched) and paths.normalized_dir-derived event lists that are not
    used by the trainer path (splits_csv / normalization_stats are
    overridden explicitly below).
    """
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.setdefault("model", {})["input_channels_names"] = list(arm_channels)
    cfg_copy["model"]["input_channels"] = len(arm_channels)
    cfg_copy.setdefault("paths", {})
    cfg_copy["paths"]["splits_csv"] = str(splits_csv)
    cfg_copy["paths"]["normalization_stats"] = str(norm_stats_path)
    cfg_copy["paths"]["checkpoints_dir"] = str(arm_dir / "checkpoints")
    cfg_copy["paths"]["results_dir"] = str(arm_dir)
    cfg_copy.setdefault("training", {})["epochs"] = int(epochs)
    cfg_copy["training"]["seed"] = int(seed)
    cfg_copy.setdefault("repro", {})["seed"] = int(seed)
    return cfg_copy


def _evaluate_checkpoint_on_val(cfg_copy: Dict[str, Any], device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Pure inference of the best-AUC checkpoint on the fold's held-out ('val') part.

    Reuses the trainer's own model-building / device-moving / logit
    extraction helpers so this is evaluation of an already-trained
    checkpoint, not a reimplementation of training.
    """
    import torch
    from torch.utils.data import DataLoader

    from src.data.dataset import PhysicsDataset
    from src.training import trainer

    val_ds = PhysicsDataset(cfg=cfg_copy, split="val")
    loader = DataLoader(val_ds, batch_size=int(cfg_get(cfg_copy, "training.batch_size", 16)),
                        shuffle=False, num_workers=0)

    model = trainer._build_model(cfg_copy).to(device)
    ckpt_dir = Path(cfg_get(cfg_copy, "paths.checkpoints_dir"))
    ckpt_path = ckpt_dir / "best_auc_model.pt"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "best_model.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    y_true_all, y_prob_all, ids_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids_all.extend(list(batch["event_id"]))
            batch_dev = trainer._to_device(batch, device)
            prior = batch_dev.get("prior_map_t0", None)
            outputs = (model(batch_dev["x"], prior_map_t0=prior)
                      if isinstance(prior, torch.Tensor) else model(batch_dev["x"]))
            logit = trainer._extract_ri_logit(outputs)
            y_true_all.append(batch_dev["y"].float().view(-1).cpu().numpy())
            y_prob_all.append(torch.sigmoid(logit).cpu().numpy())

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.float32)
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.array([], dtype=np.float32)
    return y_true, y_prob, ids_all


def run_execute(cfg: Dict[str, Any], config_path: Path, events_df: pd.DataFrame,
               folds_by_seed: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
               arms: Dict[str, List[str]], args: argparse.Namespace) -> Dict[str, Any]:
    from src.data.normalization import compute_norm_stats_from_splits
    from src.training import trainer

    interim_dir = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    results_root = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = results_root / "feature_ablation_cnn" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    sid_by_event = events_df.set_index("event_id")["sid"].astype(str).to_dict()

    per_seed_results: Dict[str, Any] = {}
    for seed in args.seeds:
        folds = folds_by_seed[seed]
        fold_rows = []
        oof_y: Dict[str, float] = {}
        oof_a: Dict[str, float] = {}
        oof_b: Dict[str, float] = {}

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

            for arm_name, arm_channels in arms.items():
                arm_dir = fold_dir / arm_name
                arm_dir.mkdir(parents=True, exist_ok=True)
                norm_stats_path = arm_dir / "normalization_stats.json"
                compute_norm_stats_from_splits(
                    interim_dir=interim_dir, splits_csv=splits_csv,
                    out_path=norm_stats_path, input_names=arm_channels,
                )
                cfg_arm = _build_arm_cfg(cfg, arm_channels, splits_csv, norm_stats_path,
                                        arm_dir, args.epochs, seed)

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
                    if arm_name == ARM_A_NAME:
                        oof_a[eid] = yp
                    else:
                        oof_b[eid] = yp

                arm_result = {
                    "arm": arm_name, "channels": arm_channels,
                    "elapsed_sec": elapsed_sec, "pr_auc": pr_auc, "roc_auc": roc_auc,
                    "train_summary": train_summary,
                    "normalization_stats_path": rel_to_root(norm_stats_path),
                    "checkpoints_dir": rel_to_root(arm_dir / "checkpoints"),
                }
                (arm_dir / "ablation_eval.json").write_text(json.dumps(arm_result, indent=2), encoding="utf-8")
                fold_row["arms"][arm_name] = {k: v for k, v in arm_result.items() if k != "train_summary"}

            fold_row["diff_pr_auc_B_minus_A"] = (
                fold_row["arms"][ARM_B_NAME]["pr_auc"] - fold_row["arms"][ARM_A_NAME]["pr_auc"]
                if fold_row["arms"][ARM_A_NAME]["pr_auc"] is not None
                and fold_row["arms"][ARM_B_NAME]["pr_auc"] is not None else None
            )
            fold_rows.append(fold_row)

        common_ids = [eid for eid in oof_y if eid in oof_a and eid in oof_b]
        y_pool = np.array([oof_y[e] for e in common_ids])
        a_pool = np.array([oof_a[e] for e in common_ids])
        b_pool = np.array([oof_b[e] for e in common_ids])
        groups_pool = np.array([sid_by_event[e] for e in common_ids])

        # Persist the raw OOF predictions: this is what makes phased
        # (one-seed-per-night) execution aggregatable later -- the final
        # pre-registered verdict needs raw scores, not per-seed aggregates.
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
        "arms": arms,
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
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# Cross-seed aggregation (the pre-registered final verdict for phased runs).
# ---------------------------------------------------------------------------

def aggregate_across_seeds(oof_csvs: Sequence[Path], n_boot: int = 10_000,
                           boot_seed: int = 42, ci: float = 0.95) -> Dict[str, Any]:
    """Pre-registered final metric over phased per-seed runs.

    Point estimate: mean over seeds of (PR-AUC_B - PR-AUC_A), each computed
    on that seed's pooled OOF predictions. CI: paired cluster bootstrap by
    SID -- ONE shared storm resampling per replicate, the per-seed deltas
    recomputed on the resampled events and averaged across seeds, so
    within-storm correlation AND cross-seed pairing stay intact. Single-class
    draws are skipped and counted, same convention as cluster_bootstrap_ci.
    """
    frames = [pd.read_csv(Path(p)) for p in oof_csvs]
    if not frames:
        raise ValueError("no oof_predictions.csv given")

    base = frames[0].sort_values("event_id").reset_index(drop=True)
    y = base["y"].to_numpy(dtype=np.float64)
    groups = base["sid"].astype(str).to_numpy()
    a_mat: List[np.ndarray] = []
    b_mat: List[np.ndarray] = []
    for i, f in enumerate(frames):
        f = f.sort_values("event_id").reset_index(drop=True)
        if not (f["event_id"].to_numpy() == base["event_id"].to_numpy()).all():
            raise ValueError(f"oof csv #{i} covers a different event set -- "
                             "all seeds must share the same dev events")
        if not np.array_equal(f["y"].to_numpy(dtype=np.float64), y):
            raise ValueError(f"oof csv #{i} labels disagree with the first csv")
        a_mat.append(f["prob_A"].to_numpy(dtype=np.float64))
        b_mat.append(f["prob_B"].to_numpy(dtype=np.float64))

    deltas_point = [float(average_precision_score(y, b) - average_precision_score(y, a))
                    for a, b in zip(a_mat, b_mat)]

    rng = np.random.default_rng(boot_seed)
    unique_groups = np.unique(groups)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}
    draws: List[float] = []
    n_skipped = 0
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            n_skipped += 1
            continue
        rep = [average_precision_score(yb, b[idx]) - average_precision_score(yb, a[idx])
               for a, b in zip(a_mat, b_mat)]
        draws.append(float(np.mean(rep)))

    arr = np.asarray(draws, dtype=np.float64)
    alpha = (1.0 - ci) / 2.0
    result: Dict[str, Any] = {
        "method": "cross_seed_mean_delta_pr_auc__cluster_bootstrap_by_sid",
        "n_seeds": len(frames),
        "oof_csvs": [rel_to_root(Path(p)) for p in oof_csvs],
        "n_events": int(len(y)),
        "n_storms": int(len(unique_groups)),
        "per_seed_delta_pr_auc": deltas_point,
        "delta_pr_auc_mean_across_seeds": float(np.mean(deltas_point)),
        "ci_level": ci,
        "n_boot_requested": n_boot,
        "n_boot_used": int(arr.size),
        "n_boot_skipped_single_class": n_skipped,
        "delta_pr_auc_ci_low": float(np.quantile(arr, alpha)) if arr.size else None,
        "delta_pr_auc_ci_high": float(np.quantile(arr, 1.0 - alpha)) if arr.size else None,
    }
    lo, hi = result["delta_pr_auc_ci_low"], result["delta_pr_auc_ci_high"]
    if lo is None:
        result["verdict"] = "INCONCLUSIVE: insufficient valid bootstrap draws to compute a CI."
    elif lo > 0:
        result["verdict"] = (f"CI [{lo:.4f}, {hi:.4f}] excludes zero (positive): shear/rh add "
                             "quantified skill (report the delta and CI; do not oversell).")
    elif hi < 0:
        result["verdict"] = (f"CI [{lo:.4f}, {hi:.4f}] excludes zero (NEGATIVE): investigate "
                             "before reporting (pre-registered branch).")
    else:
        result["verdict"] = (f"CI [{lo:.4f}, {hi:.4f}] includes zero: NULL -- the added "
                             "channels do not add detectable skill at this resolution/regime.")
    return result


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
        out_dir = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve() / "feature_ablation_cnn"
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

    arms = resolve_arms(cfg)
    folds_by_seed = {s: build_folds(events_df, args.folds, s) for s in args.seeds}
    plan = build_run_plan(events_df, folds_by_seed, args.seeds, args.folds,
                          args.epochs, args.minutes_per_epoch)

    print(format_run_plan_table(plan, arms))

    # Execution requires the EXPLICIT --execute flag; --no-dry-run alone is
    # not enough (a mistyped flag must never launch hours of training).
    if not args.execute:
        print("\nDRY RUN: no training was performed. Pass --execute to actually train.")
        return

    print("\nEXECUTING: this will run real trainings via src.training.trainer.train(). "
          "This may take a long time -- see the wall-time estimate above.")
    summary = run_execute(cfg, config_path, events_df, folds_by_seed, arms, args)
    print(f"\nrun_dir: {summary['run_dir']}")


if __name__ == "__main__":
    main()
