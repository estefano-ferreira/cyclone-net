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
  * Gate: refuses to run (unless --no-require-gate) until
    outputs/results/pl_gate_census.json reports gate_pass=true, i.e. the
    dev set has complete PL-channel coverage (see analysis/pl_gate_census.py).

Uncertainty quantification (paired):
  * Per-fold paired deltas (B - A) for PR-AUC and ROC-AUC (as before).
  * Pooled out-of-fold (OOF) predictions per arm (every dev event gets
    exactly one held-out prediction per arm, from whichever fold contained
    it in its test partition) -> event-level paired comparison with a
    CLUSTER bootstrap by SID: storms (not events) are resampled with
    replacement, so within-storm correlation does not leak into the CI.
    10,000 draws by default, seeded from --seed. Reports delta PR-AUC and
    delta ROC-AUC with 95% CI, plus both arms' absolute pooled metrics with
    the same CIs. A CI crossing zero is reported mechanically as an honest
    null — no cheerleading language.

No threshold selection anywhere: only ranking metrics (PR-AUC, ROC-AUC).

Usage:
    python analysis/feature_ablation_kfold.py [--n-splits 5] [--seed 42]
                                               [--config config.yaml]
                                               [--n-boot 10000]
                                               [--require-gate | --no-require-gate]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from src.utils.paths import rel_to_root  # noqa: E402

PL_CHANNELS = [SHEAR_CHANNEL, RH_CHANNEL]
GATE_FILENAME = "pl_gate_census.json"


# ---------------------------------------------------------------------------
# Gate: shared by feature_ablation_kfold, feature_ablation_cnn, ri_precursors.
# ---------------------------------------------------------------------------

def load_gate_census(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Load outputs/results/pl_gate_census.json, or None if absent."""
    path = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve() / GATE_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def enforce_gate(cfg: Dict[str, Any], require_gate: bool) -> None:
    """Refuse to proceed (sys.exit(1)) unless the PL-coverage gate passed.

    Bypassed only by explicit --no-require-gate, which prints a loud warning
    so a bypassed run is never mistaken for a gated one.
    """
    if not require_gate:
        print("WARNING: --no-require-gate: PL-channel coverage gate check was SKIPPED. "
              "Results are NOT protected against partial-coverage bias.")
        return

    census = load_gate_census(cfg)
    if census is None:
        print(f"GATE FAIL: {GATE_FILENAME} not found under paths.results_dir. "
              "Run: python analysis/pl_gate_census.py first.")
        sys.exit(1)
    if not bool(census.get("gate_pass", False)):
        print(f"GATE FAIL: {GATE_FILENAME} reports gate_pass=false "
              f"({census.get('gate_verdict', 'no verdict recorded')}).")
        sys.exit(1)
    print(f"GATE PASS: {GATE_FILENAME} confirms complete dev-set PL coverage "
          f"(generated_at={census.get('generated_at')}).")


# ---------------------------------------------------------------------------
# Data loading / feature construction (unchanged fairness controls).
# ---------------------------------------------------------------------------

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
            np.asarray(y, dtype=int), np.asarray(groups), audit, used_ids)


def make_estimator():
    """Same classifier as the released tabular baseline, with fold-internal scaling."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    )


# ---------------------------------------------------------------------------
# Paired cluster (by-SID) bootstrap.
# ---------------------------------------------------------------------------

def cluster_bootstrap_ci(
    y: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    groups: np.ndarray,
    seed: int = 42,
    n_boot: int = 10_000,
    ci: float = 0.95,
) -> Dict[str, Any]:
    """Paired cluster bootstrap over storms (SID), on pooled predictions.

    Resamples unique groups (storms) WITH replacement (n_boot draws); each
    draw's event set is the union of all events belonging to the sampled
    storms (a storm sampled k times contributes its events k times). This
    keeps within-storm correlation intact instead of pretending events are
    i.i.d.

    Draws where the resampled label vector has only one class are skipped
    (PR-AUC/ROC-AUC undefined) and counted in ``n_boot_skipped_single_class``
    — an expected occurrence with ~35 positive storms, not a bug.

    Returns mean / 95% CI (or the requested ``ci``) for both arms' absolute
    PR-AUC/ROC-AUC and their paired deltas (B - A).
    """
    y = np.asarray(y)
    prob_a = np.asarray(prob_a)
    prob_b = np.asarray(prob_b)
    groups = np.asarray(groups)

    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}

    keys = ["a_pr_auc", "b_pr_auc", "a_roc_auc", "b_roc_auc", "delta_pr_auc", "delta_roc_auc"]
    draws: Dict[str, List[float]] = {k: [] for k in keys}
    n_skipped = 0

    for _ in range(n_boot):
        sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled_groups])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            n_skipped += 1
            continue
        pa = prob_a[idx]
        pb = prob_b[idx]
        a_pr = average_precision_score(yb, pa)
        b_pr = average_precision_score(yb, pb)
        a_roc = roc_auc_score(yb, pa)
        b_roc = roc_auc_score(yb, pb)
        draws["a_pr_auc"].append(a_pr)
        draws["b_pr_auc"].append(b_pr)
        draws["a_roc_auc"].append(a_roc)
        draws["b_roc_auc"].append(b_roc)
        draws["delta_pr_auc"].append(b_pr - a_pr)
        draws["delta_roc_auc"].append(b_roc - a_roc)

    alpha = (1.0 - ci) / 2.0
    out: Dict[str, Any] = {
        "method": "cluster_bootstrap_by_sid",
        "seed": seed,
        "ci_level": ci,
        "n_boot_requested": n_boot,
        "n_boot_used": n_boot - n_skipped,
        "n_boot_skipped_single_class": n_skipped,
        "n_groups": int(n_groups),
    }
    for k in keys:
        arr = np.asarray(draws[k], dtype=np.float64)
        if arr.size == 0:
            out[k] = {"mean": None, "ci_low": None, "ci_high": None}
            continue
        out[k] = {
            "mean": float(arr.mean()),
            "ci_low": float(np.quantile(arr, alpha)),
            "ci_high": float(np.quantile(arr, 1.0 - alpha)),
        }
    return out


def _mechanical_verdict(name: str, ci_stats: Dict[str, Any]) -> str:
    """State whether a delta's CI crosses zero -- mechanically, no cheerleading."""
    lo, hi = ci_stats.get("ci_low"), ci_stats.get("ci_high")
    if lo is None or hi is None:
        return f"{name}: insufficient valid bootstrap draws to compute a CI."
    if lo <= 0.0 <= hi:
        return (f"{name} 95% CI [{lo:.4f}, {hi:.4f}] includes zero: the null "
                "hypothesis of no difference cannot be rejected.")
    return f"{name} 95% CI [{lo:.4f}, {hi:.4f}] excludes zero."


def _config_digest(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml (relative paths resolve against the project root).")
    parser.add_argument("--require-gate", action=argparse.BooleanOptionalAction, default=True,
                        help="Refuse to run unless outputs/results/pl_gate_census.json reports "
                             "gate_pass=true (default: on). Use --no-require-gate to bypass.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    cfg = load_config(str(config_path))

    enforce_gate(cfg, args.require_gate)

    dev = load_dev_events(cfg)
    X_a, X_b, y, groups, audit, used_ids = build_feature_matrices(cfg, dev)

    print("AVAILABILITY AUDIT:", json.dumps(audit, indent=2))
    if audit["n_used"] == 0 or audit["n_positives_used"] < args.n_splits:
        print("Not enough usable events/positives for the requested k-fold — aborting.")
        sys.exit(1)

    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    folds = list(cv.split(X_a, y, groups=groups))  # identical folds for A and B

    n_used = len(y)
    oof_a = np.full(n_used, np.nan, dtype=np.float64)
    oof_b = np.full(n_used, np.nan, dtype=np.float64)

    per_fold = []
    for k, (tr, te) in enumerate(folds):
        est_a = make_estimator()
        est_a.fit(X_a[tr], y[tr])
        prob_a_te = est_a.predict_proba(X_a[te])[:, 1]
        oof_a[te] = prob_a_te

        est_b = make_estimator()
        est_b.fit(X_b[tr], y[tr])
        prob_b_te = est_b.predict_proba(X_b[te])[:, 1]
        oof_b[te] = prob_b_te

        if y[te].sum() == 0:
            per_fold.append({"fold": k, "n_test": int(len(te)), "n_pos_test": 0,
                             "note": "no positives in fold — per-fold ranking metrics undefined "
                                     "(predictions still contribute to the pooled OOF bootstrap)"})
            continue

        row = {
            "fold": k, "n_test": int(len(te)), "n_pos_test": int(y[te].sum()),
            "A_surface_pr_auc": float(average_precision_score(y[te], prob_a_te)),
            "A_surface_roc_auc": float(roc_auc_score(y[te], prob_a_te)),
            "B_with_pressure_pr_auc": float(average_precision_score(y[te], prob_b_te)),
            "B_with_pressure_roc_auc": float(roc_auc_score(y[te], prob_b_te)),
        }
        row["diff_pr_auc_B_minus_A"] = row["B_with_pressure_pr_auc"] - row["A_surface_pr_auc"]
        row["diff_roc_auc_B_minus_A"] = row["B_with_pressure_roc_auc"] - row["A_surface_roc_auc"]
        per_fold.append(row)

    assert not np.isnan(oof_a).any() and not np.isnan(oof_b).any(), (
        "OOF prediction arrays must be fully populated: folds must partition all dev events exactly once."
    )

    valid = [r for r in per_fold if "diff_pr_auc_B_minus_A" in r]
    diffs_pr = np.array([r["diff_pr_auc_B_minus_A"] for r in valid])
    diffs_roc = np.array([r["diff_roc_auc_B_minus_A"] for r in valid])
    a_scores = np.array([r["A_surface_pr_auc"] for r in valid])
    b_scores = np.array([r["B_with_pressure_pr_auc"] for r in valid])

    per_fold_summary = {
        "A_surface_mean": float(a_scores.mean()), "A_surface_std": float(a_scores.std(ddof=1)),
        "B_with_pressure_mean": float(b_scores.mean()), "B_with_pressure_std": float(b_scores.std(ddof=1)),
        "diff_pr_auc_mean_B_minus_A": float(diffs_pr.mean()), "diff_pr_auc_std": float(diffs_pr.std(ddof=1)),
        "diff_roc_auc_mean_B_minus_A": float(diffs_roc.mean()), "diff_roc_auc_std": float(diffs_roc.std(ddof=1)),
        "folds_B_better_pr_auc": int((diffs_pr > 0).sum()), "n_valid_folds": int(len(diffs_pr)),
    }

    # Pooled OOF absolute metrics (one prediction per dev event per arm).
    pooled = {
        "A_surface_pr_auc": float(average_precision_score(y, oof_a)),
        "A_surface_roc_auc": float(roc_auc_score(y, oof_a)),
        "B_with_pressure_pr_auc": float(average_precision_score(y, oof_b)),
        "B_with_pressure_roc_auc": float(roc_auc_score(y, oof_b)),
    }
    pooled["diff_pr_auc_B_minus_A"] = pooled["B_with_pressure_pr_auc"] - pooled["A_surface_pr_auc"]
    pooled["diff_roc_auc_B_minus_A"] = pooled["B_with_pressure_roc_auc"] - pooled["A_surface_roc_auc"]

    bootstrap = cluster_bootstrap_ci(y, oof_a, oof_b, groups, seed=args.seed, n_boot=args.n_boot)

    verdict_pr_auc = _mechanical_verdict("delta_pr_auc (B_with_pressure - A_surface)", bootstrap["delta_pr_auc"])
    verdict_roc_auc = _mechanical_verdict("delta_roc_auc (B_with_pressure - A_surface)", bootstrap["delta_roc_auc"])

    summary = {
        "protocol": "StratifiedGroupKFold by SID, paired A/B, dev set only (test untouched)",
        "seed": args.seed,
        "n_splits": args.n_splits,
        "n_boot": args.n_boot,
        "config_path": rel_to_root(config_path),
        "config_digest_sha256": _config_digest(config_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "availability": audit,
        "n_events": audit["n_used"],
        "n_positives": audit["n_positives_used"],
        "n_storms": audit["n_storms_used"],
        "pr_auc": per_fold_summary,
        "per_fold": per_fold,
        "pooled_oof": pooled,
        "cluster_bootstrap_by_sid": bootstrap,
        "chance_level_pr_auc": float(y.mean()),
        "verdict": {
            "pr_auc": verdict_pr_auc,
            "roc_auc": verdict_roc_auc,
        },
        "interpretation_guard": (
            f"With {audit['n_positives_used']} positives in the development set, per-fold "
            "PR-AUC rests on a handful of positives per fold; treat differences smaller than "
            "the fold-to-fold std, or bootstrap CIs crossing zero, as indistinguishable from "
            "noise."
        ),
    }

    out_dir = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # New canonical output path (deliverable).
    out_path = out_dir / "feature_ablation_kfold.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Legacy path kept for backward compatibility with existing references.
    legacy_dir = out_dir / "feature_ablation"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    (legacy_dir / "kfold_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(per_fold_summary, indent=2))
    print(json.dumps(pooled, indent=2))
    print(f"chance level (prevalence): {summary['chance_level_pr_auc']:.3f}")
    print(verdict_pr_auc)
    print(verdict_roc_auc)
    print(f"report: {out_path}")


if __name__ == "__main__":
    main()
