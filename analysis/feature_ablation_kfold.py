# analysis/feature_ablation_kfold.py
"""
Paired feature-set comparison: surface-only vs surface + pressure-level
predictors, evaluated with SID-grouped stratified k-fold cross-validation.

Question: do the two SHIPS-style environmental channels added by
``src/processors/pressure_channels.py`` (deep-layer shear ``shear_850_200_mps``
and mid-level humidity ``rh_mid``) improve RI discrimination beyond the
surface-only cube?

Design (fairness and leakage controls):
  * Development set only: events in the train+val splits. The held-out test
    split is NEVER loaded — it stays untouched for final evaluation.
  * StratifiedGroupKFold grouped by SID: a storm's events are always in the
    same fold (no storm-level leakage), stratified on the RI label.
  * Identical folds for both feature sets (paired comparison).
  * Identical estimator for both: StandardScaler + LogisticRegression
    (same hyperparameters as src/baselines/tabular_lr.py), with scaling
    fitted inside each training fold only.
  * Identical feature construction (extract_features_from_cube) for both
    sets; the ONLY difference is which channels feed it:
      A) the 12 surface channels
      B) the same 12 + [shear_850_200_mps, rh_mid]
  * Only events whose cube contains BOTH pressure channels are compared,
    so A and B see exactly the same events.

Primary metric: PR-AUC (average precision) per fold; the paired per-fold
difference (B - A) with mean, std, and sign counts is the decision quantity.

Usage:
    python analysis/feature_ablation_kfold.py [--n-splits 5] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.tabular_lr import extract_features_from_cube  # noqa: E402
from src.processors.pressure_channels import RH_CHANNEL, SHEAR_CHANNEL  # noqa: E402
from src.utils.config import cfg_get, load_config  # noqa: E402

PL_CHANNELS = [SHEAR_CHANNEL, RH_CHANNEL]


def load_dev_events(cfg) -> pd.DataFrame:
    """Development events = train + val splits with their RI labels.

    The test split is intentionally excluded and never read.
    """
    normalized = Path(cfg_get(cfg, "paths.normalized_dir", "./data/normalized")).resolve()
    splits = pd.read_csv(normalized / "splits.csv")
    events = pd.read_csv(normalized / "valid_events.csv")
    df = splits.merge(events[["event_id", "sid", "ri_label"]], on="event_id", how="inner")
    dev = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    return dev


def build_feature_matrices(cfg, dev: pd.DataFrame):
    """Build paired feature matrices (A: surface-only, B: +pressure channels).

    Returns (X_a, X_b, y, groups, audit) using only events whose cube has both
    pressure channels, so both sets are computed on the same sample.
    """
    interim = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()

    rows_a, rows_b, y, groups, used_ids = [], [], [], [], []
    n_no_artifact = 0
    n_surface_only = 0
    names_a = names_b = None

    for r in dev.itertuples():
        meta_path = interim / f"{r.event_id}.json"
        cube_path = interim / f"{r.event_id}.npy"
        if not (meta_path.exists() and cube_path.exists()):
            n_no_artifact += 1
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        channels = list(meta.get("channels", []))
        if not all(ch in channels for ch in PL_CHANNELS):
            n_surface_only += 1
            continue

        cube = np.load(cube_path)
        surface = [ch for ch in channels if ch not in PL_CHANNELS]
        idx_surface = [channels.index(ch) for ch in surface]
        idx_all = idx_surface + [channels.index(ch) for ch in PL_CHANNELS]

        feats_a = extract_features_from_cube(cube[..., idx_surface], surface)
        feats_b = extract_features_from_cube(cube[..., idx_all], surface + PL_CHANNELS)
        if names_a is None:
            names_a = sorted(feats_a.keys())
            names_b = sorted(feats_b.keys())
        rows_a.append([feats_a[k] for k in names_a])
        rows_b.append([feats_b[k] for k in names_b])
        y.append(int(r.ri_label))
        groups.append(str(r.sid))
        used_ids.append(r.event_id)

    audit = {
        "n_dev_events": int(len(dev)),
        "n_used": int(len(y)),
        "n_missing_artifact": int(n_no_artifact),
        "n_without_pressure_channels": int(n_surface_only),
        "n_positives_used": int(np.sum(y)),
        "n_storms_used": int(len(set(groups))),
        "n_features": {"surface_only": len(names_a or []), "with_pressure": len(names_b or [])},
    }
    return (np.asarray(rows_a, dtype=np.float64), np.asarray(rows_b, dtype=np.float64),
            np.asarray(y, dtype=int), np.asarray(groups), audit)


def make_estimator():
    """Same classifier as the released tabular baseline, with fold-internal scaling."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    dev = load_dev_events(cfg)
    X_a, X_b, y, groups, audit = build_feature_matrices(cfg, dev)

    print("AVAILABILITY AUDIT:", json.dumps(audit, indent=2))
    if audit["n_used"] == 0 or audit["n_positives_used"] < args.n_splits:
        print("Not enough usable events/positives for the requested k-fold — aborting.")
        sys.exit(1)

    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    folds = list(cv.split(X_a, y, groups=groups))  # identical folds for A and B

    per_fold = []
    for k, (tr, te) in enumerate(folds):
        if y[te].sum() == 0:
            per_fold.append({"fold": k, "n_test": int(len(te)), "n_pos_test": 0,
                             "note": "no positives in fold — metrics undefined"})
            continue
        row = {"fold": k, "n_test": int(len(te)), "n_pos_test": int(y[te].sum())}
        for label, X in (("A_surface", X_a), ("B_with_pressure", X_b)):
            est = make_estimator()
            est.fit(X[tr], y[tr])
            prob = est.predict_proba(X[te])[:, 1]
            row[f"{label}_pr_auc"] = float(average_precision_score(y[te], prob))
            row[f"{label}_roc_auc"] = float(roc_auc_score(y[te], prob))
        row["diff_pr_auc_B_minus_A"] = row["B_with_pressure_pr_auc"] - row["A_surface_pr_auc"]
        per_fold.append(row)

    valid = [r for r in per_fold if "diff_pr_auc_B_minus_A" in r]
    diffs = np.array([r["diff_pr_auc_B_minus_A"] for r in valid])
    a_scores = np.array([r["A_surface_pr_auc"] for r in valid])
    b_scores = np.array([r["B_with_pressure_pr_auc"] for r in valid])

    summary = {
        "protocol": "StratifiedGroupKFold by SID, paired A/B, dev set only (test untouched)",
        "seed": args.seed,
        "n_splits": args.n_splits,
        "availability": audit,
        "pr_auc": {
            "A_surface_mean": float(a_scores.mean()), "A_surface_std": float(a_scores.std(ddof=1)),
            "B_with_pressure_mean": float(b_scores.mean()), "B_with_pressure_std": float(b_scores.std(ddof=1)),
            "diff_mean_B_minus_A": float(diffs.mean()), "diff_std": float(diffs.std(ddof=1)),
            "folds_B_better": int((diffs > 0).sum()), "n_valid_folds": int(len(diffs)),
        },
        "per_fold": per_fold,
        "chance_level_pr_auc": float(y.mean()),
        "interpretation_guard": (
            "With ~33 positives in the development set, per-fold PR-AUC rests on "
            "6-8 positives; treat differences smaller than the fold-to-fold std "
            "as indistinguishable from noise."
        ),
    }

    out_dir = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve() / "feature_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "kfold_comparison.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["pr_auc"], indent=2))
    print(f"chance level (prevalence): {summary['chance_level_pr_auc']:.3f}")
    print(f"report: {out_path}")


if __name__ == "__main__":
    main()
