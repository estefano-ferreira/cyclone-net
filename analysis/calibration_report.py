# analysis/calibration_report.py
"""
Post-hoc calibration and PR/ROC-curve report of the RELEASED model's saved test predictions.

Purpose
-------
Re-analyzes ``outputs/results/test_predictions.csv`` — the frozen test-set
predictions saved during model evaluation. This script computes calibration
diagnostics (reliability diagrams, ECE, Brier score, Murphy decomposition),
precision-recall and ROC curves, and operating-point tables. It does NOT
load the model, read the data cube, or touch the test split loader — only
the saved predictions CSV is analyzed.

Output
------
outputs/results/calibration/
  ├── calibration_summary.json     # All scalar metrics, file inventory, timestamp
  ├── reliability_table_quantile.csv    # 10 quantile bins
  ├── reliability_table_fixed_width.csv # 10 fixed-width bins
  ├── pr_curve_points.csv          # Precision-recall curve coordinates
  ├── roc_curve_points.csv         # ROC curve coordinates
  ├── operating_points.csv         # Thresholds for recall targets
  ├── score_distribution.csv       # Per-class score quantiles
  ├── reliability_diagram_quantile.png  (if matplotlib available)
  ├── reliability_diagram_fixed_width.png
  ├── pr_curve.png
  └── score_histograms.png

Usage:
    python analysis/calibration_report.py [--predictions <path>] [--out-dir <dir>] [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import cfg_get, load_config  # noqa: E402
from src.utils.paths import rel_to_root  # noqa: E402

# Try to import matplotlib; plotting is optional.
try:
    import matplotlib
    matplotlib.use("Agg")  # Use non-interactive backend.
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except (ImportError, RuntimeError):
    HAS_MATPLOTLIB = False
    plt = None


def load_predictions(predictions_csv: Path) -> pd.DataFrame:
    """Load the test predictions CSV."""
    if not predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")
    df = pd.read_csv(predictions_csv)
    return df


def compute_reliability_table_quantile(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Compute reliability table using quantile-based binning (equal count).

    Each bin contains ~n_samples/n_bins predictions. This gives more stable
    estimates in the tails.
    """
    # Use quantiles to assign bins.
    quantile_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, np.quantile(y_prob, quantile_edges[1:-1])) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    rows = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        n = mask.sum()
        mean_pred = y_prob[mask].mean()
        frac_pos = y_true[mask].mean()
        gap = abs(mean_pred - frac_pos)
        rows.append({
            "bin": i,
            "n": int(n),
            "mean_pred": float(mean_pred),
            "frac_pos": float(frac_pos),
            "gap": float(gap),
        })
    return pd.DataFrame(rows)


def compute_reliability_table_fixed_width(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Compute reliability table using fixed-width binning.

    Bins are [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0].
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    rows = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() == 0:
            continue
        n = mask.sum()
        mean_pred = y_prob[mask].mean()
        frac_pos = y_true[mask].mean()
        gap = abs(mean_pred - frac_pos)
        rows.append({
            "bin": i,
            "lower": float(bin_edges[i]),
            "upper": float(bin_edges[i + 1]),
            "n": int(n),
            "mean_pred": float(mean_pred),
            "frac_pos": float(frac_pos),
            "gap": float(gap),
        })
    return pd.DataFrame(rows)


def compute_ece(rel_table: pd.DataFrame) -> float:
    """Compute Expected Calibration Error (weighted by bin count).

    ECE = sum( n_i / N * |p_i - a_i| )
    where n_i is the count in bin i, N is total, p_i is mean prediction,
    and a_i is the fraction of positives in that bin.
    """
    total = rel_table["n"].sum()
    if total == 0:
        return 0.0
    ece = (rel_table["n"] * rel_table["gap"]).sum() / total
    return float(ece)


def compute_brier_and_murphy(y_true: np.ndarray, y_prob: np.ndarray, rel_table: pd.DataFrame,
                             n_bins: int = 10) -> Dict[str, float]:
    """Brier score and the GENERALIZED Murphy decomposition over quantile bins.

    The classic 3-term identity (Brier = REL - RES + UNC) is exact only for
    grouped/discrete forecasts. Binning continuous scores introduces two
    within-bin terms (Stephenson, Coelho & Jolliffe 2008):

      Brier = REL - RES + UNC + WBV - WBC          (exact)
      WBV = (1/N) sum_i sum_{k in bin i} (p_k - pbar_i)^2
      WBC = (2/N) sum_i sum_{k in bin i} (p_k - pbar_i)(y_k - abar_i)

    All five components are reported, plus the 3-term residual (WBV - WBC),
    so the published arithmetic closes exactly for any reader who recomputes
    it. The binning here replicates compute_reliability_table_quantile.
    """
    brier = float(brier_score_loss(y_true, y_prob))

    quantile_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, np.quantile(y_prob, quantile_edges[1:-1])) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    n = len(y_true)
    bar_a = float(y_true.mean())
    uncertainty = bar_a * (1.0 - bar_a)

    reliability = resolution = wbv = wbc = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        n_i = int(mask.sum())
        if n_i == 0:
            continue
        pbar_i = float(y_prob[mask].mean())
        abar_i = float(y_true[mask].mean())
        reliability += n_i * (pbar_i - abar_i) ** 2
        resolution += n_i * (abar_i - bar_a) ** 2
        wbv += float(((y_prob[mask] - pbar_i) ** 2).sum())
        wbc += 2.0 * float(((y_prob[mask] - pbar_i) * (y_true[mask] - abar_i)).sum())
    reliability /= n
    resolution /= n
    wbv /= n
    wbc /= n

    return {
        "brier_score": float(brier),
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "within_bin_variance": float(wbv),
        "within_bin_covariance": float(wbc),
        "three_term_sum_rel_minus_res_plus_unc": float(reliability - resolution + uncertainty),
        "three_term_residual_wbv_minus_wbc": float(wbv - wbc),
        "five_term_identity_check_abs_error": float(
            abs(brier - (reliability - resolution + uncertainty + wbv - wbc))),
    }


def compute_pr_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[pd.DataFrame, float]:
    """Compute PR curve points and PR-AUC.

    Returns DataFrame with (threshold, precision, recall) and the scalar PR-AUC.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = float(average_precision_score(y_true, y_prob))

    # precision_recall_curve returns one more precision/recall than thresholds
    # (the last point has precision=1, recall=0 with no threshold).
    # We'll align them by using thresholds[:-1] for the last n-1 points.
    rows = []
    for i, (p, r) in enumerate(zip(precision[:-1], recall[:-1])):
        rows.append({
            "threshold": float(thresholds[i]),
            "precision": float(p),
            "recall": float(r),
        })
    # Add the final point (threshold undefined, precision=1, recall=0).
    rows.append({
        "threshold": np.nan,
        "precision": float(precision[-1]),
        "recall": float(recall[-1]),
    })

    return pd.DataFrame(rows), pr_auc


def compute_roc_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[pd.DataFrame, float]:
    """Compute ROC curve points and ROC-AUC.

    Returns DataFrame with (threshold, fpr, tpr) and the scalar ROC-AUC.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = float(roc_auc_score(y_true, y_prob))

    rows = []
    for f, t, th in zip(fpr, tpr, thresholds):
        rows.append({
            "threshold": float(th),
            "fpr": float(f),
            "tpr": float(t),
        })

    return pd.DataFrame(rows), roc_auc


def compute_score_distribution(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Compute score statistics split by class (positive/negative).

    For each class: min, max, quantiles (1, 5, 25, 50, 75, 95, 99), and count.
    """
    quantiles_list = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

    rows = []
    for label in [0, 1]:
        mask = y_true == label
        scores = y_prob[mask]
        n = mask.sum()

        if n == 0:
            continue

        row = {
            "class": int(label),
            "count": int(n),
            "min": float(scores.min()),
            "max": float(scores.max()),
        }
        for q in quantiles_list:
            row[f"q{int(q*100)}"] = float(np.quantile(scores, q))
        rows.append(row)

    return pd.DataFrame(rows)


def compute_operating_points(y_true: np.ndarray, y_prob: np.ndarray, recall_targets: List[float]) -> pd.DataFrame:
    """Compute operating points for given recall targets.

    For each recall target, find the threshold that achieves (approximately)
    that recall, and report threshold, precision, FPR, and alerts per 100 events.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    fpr_vals, tpr_vals, roc_thresholds = roc_curve(y_true, y_prob)

    # Flip if necessary (sklearn may return in descending order).
    if len(thresholds) > 0 and thresholds[-1] < thresholds[0]:
        precision = precision[::-1]
        recall = recall[::-1]
        thresholds = thresholds[::-1]

    # For each recall target, find closest threshold.
    rows = []
    for target_recall in recall_targets:
        # Find recall value closest to target.
        idx = np.argmin(np.abs(recall[:-1] - target_recall))
        threshold = thresholds[idx]
        prec = precision[idx]
        recall_actual = recall[idx]

        # Find FPR at this threshold using the ROC curve.
        # Interp if needed, or just take closest.
        roc_idx = np.argmin(np.abs(roc_thresholds - threshold))
        fpr = fpr_vals[roc_idx]

        # Alerts per 100 events: num_positive_predictions / n_total * 100
        alerts_per_100 = (y_prob >= threshold).sum() / len(y_true) * 100.0

        rows.append({
            "recall_target": float(target_recall),
            "threshold": float(threshold),
            "precision": float(prec),
            "recall_actual": float(recall_actual),
            "fpr": float(fpr),
            "alerts_per_100_events": float(alerts_per_100),
        })

    return pd.DataFrame(rows)


def plot_reliability_diagrams(rel_table_quantile: pd.DataFrame, rel_table_fixed: pd.DataFrame, out_dir: Path) -> None:
    """Plot reliability diagrams (calibration curves) for both binning strategies."""
    if not HAS_MATPLOTLIB:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Quantile bins.
    ax1.scatter(rel_table_quantile["mean_pred"], rel_table_quantile["frac_pos"], s=rel_table_quantile["n"], alpha=0.6)
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title("Reliability diagram (quantile bins)")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Fixed-width bins.
    ax2.scatter(rel_table_fixed["mean_pred"], rel_table_fixed["frac_pos"], s=rel_table_fixed["n"], alpha=0.6)
    ax2.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("Fraction of positives")
    ax2.set_title("Reliability diagram (fixed-width bins)")
    ax2.legend()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "reliability_diagrams.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pr_curve(pr_df: pd.DataFrame, pr_auc: float, prevalence: float, out_dir: Path) -> None:
    """Plot PR curve with prevalence baseline."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(pr_df["recall"], pr_df["precision"], label=f"PR curve (AUC={pr_auc:.3f})")
    ax.axhline(prevalence, color="red", linestyle="--", label=f"Prevalence={prevalence:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "pr_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_histograms(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path) -> None:
    """Plot score distributions by class (log y-axis)."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(y_prob[y_true == 0], bins=50, alpha=0.6, label="Negative (y=0)", log=True)
    ax.hist(y_prob[y_true == 1], bins=50, alpha=0.6, label="Positive (y=1)", log=True)
    ax.set_xlabel("Predicted probability (ri_score)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Score Distribution by Class")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "score_histograms.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=str,
        default="outputs/results/test_predictions.csv",
        help="Path to test predictions CSV (relative to project root or absolute).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/results/calibration",
        help="Output directory for calibration reports (relative to project root or absolute).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (relative paths resolve against the project root).",
    )
    args = parser.parse_args()

    # Resolve paths.
    predictions_path = Path(args.predictions)
    if not predictions_path.is_absolute():
        predictions_path = (PROJECT_ROOT / predictions_path).resolve()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions.
    print(f"Loading predictions from {rel_to_root(predictions_path)}...")
    df = load_predictions(predictions_path)

    # Extract y_true and y_prob.
    # Expected columns: event_id, y_true, ri_score, ...
    if "y_true" not in df.columns or "ri_score" not in df.columns:
        raise ValueError(f"Expected 'y_true' and 'ri_score' in predictions CSV. Got: {df.columns.tolist()}")

    y_true = df["y_true"].values
    y_prob = df["ri_score"].values
    n_rows = len(df)
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    prevalence = n_pos / n_rows if n_rows > 0 else 0.0

    print(f"  n_rows={n_rows}, n_pos={n_pos}, n_neg={n_neg}, prevalence={prevalence:.4f}")
    print(f"\nRe-analyzing released predictions (does NOT reload test set or model)...")

    # Compute all diagnostics.
    rel_table_quantile = compute_reliability_table_quantile(y_true, y_prob)
    rel_table_fixed = compute_reliability_table_fixed_width(y_true, y_prob)
    ece_quantile = compute_ece(rel_table_quantile)
    ece_fixed = compute_ece(rel_table_fixed)

    murphy = compute_brier_and_murphy(y_true, y_prob, rel_table_quantile)

    pr_df, pr_auc = compute_pr_curve_points(y_true, y_prob)
    roc_df, roc_auc = compute_roc_curve_points(y_true, y_prob)

    score_dist = compute_score_distribution(y_true, y_prob)

    recall_targets = [0.99, 0.95, 0.90, 0.80, 0.50]
    op_points = compute_operating_points(y_true, y_prob, recall_targets)

    # Save CSVs.
    rel_table_quantile.to_csv(out_dir / "reliability_table_quantile.csv", index=False)
    rel_table_fixed.to_csv(out_dir / "reliability_table_fixed_width.csv", index=False)
    pr_df.to_csv(out_dir / "pr_curve_points.csv", index=False)
    roc_df.to_csv(out_dir / "roc_curve_points.csv", index=False)
    op_points.to_csv(out_dir / "operating_points.csv", index=False)
    score_dist.to_csv(out_dir / "score_distribution.csv", index=False)

    # Optionally create plots.
    plot_reliability_diagrams(rel_table_quantile, rel_table_fixed, out_dir)
    plot_pr_curve(pr_df, pr_auc, float(prevalence), out_dir)
    plot_score_histograms(y_true, y_prob, out_dir)

    # Create summary JSON.
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_predictions_csv": rel_to_root(predictions_path),
        "n_rows": int(n_rows),
        "n_positives": int(n_pos),
        "n_negatives": int(n_neg),
        "prevalence": float(prevalence),
        "metrics": {
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "ece_quantile_bins": float(ece_quantile),
            "ece_fixed_width_bins": float(ece_fixed),
            **murphy,
        },
        "released_scores_for_comparison": {
            "pr_auc": 0.251,
            "roc_auc": 0.796,
        },
        "files_created": {
            "reliability_table_quantile": rel_to_root(out_dir / "reliability_table_quantile.csv"),
            "reliability_table_fixed_width": rel_to_root(out_dir / "reliability_table_fixed_width.csv"),
            "pr_curve_points": rel_to_root(out_dir / "pr_curve_points.csv"),
            "roc_curve_points": rel_to_root(out_dir / "roc_curve_points.csv"),
            "operating_points": rel_to_root(out_dir / "operating_points.csv"),
            "score_distribution": rel_to_root(out_dir / "score_distribution.csv"),
        },
    }

    if HAS_MATPLOTLIB:
        summary["files_created"]["reliability_diagrams_plot"] = rel_to_root(out_dir / "reliability_diagrams.png")
        summary["files_created"]["pr_curve_plot"] = rel_to_root(out_dir / "pr_curve.png")
        summary["files_created"]["score_histograms_plot"] = rel_to_root(out_dir / "score_histograms.png")
    else:
        summary["matplotlib_available"] = False

    summary_path = out_dir / "calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print summary.
    print(f"\n{'='*70}")
    print("CALIBRATION REPORT SUMMARY")
    print(f"{'='*70}")
    print(f"Predictions: {n_rows} samples ({n_pos} pos, {n_neg} neg, prevalence={prevalence:.4f})")
    print(f"\nMetrics:")
    print(f"  PR-AUC:           {pr_auc:.4f}  (released: 0.251)")
    print(f"  ROC-AUC:          {roc_auc:.4f}  (released: 0.796)")
    print(f"  ECE (quantile):   {ece_quantile:.4f}")
    print(f"  ECE (fixed-width):{ece_fixed:.4f}")
    print(f"  Brier score:      {murphy['brier_score']:.4f}")
    print(f"    Reliability:    {murphy['reliability']:.4f}")
    print(f"    Resolution:     {murphy['resolution']:.4f}")
    print(f"    Uncertainty:    {murphy['uncertainty']:.4f}")
    print(f"    Within-bin var: {murphy['within_bin_variance']:.4f}  "
          f"cov: {murphy['within_bin_covariance']:.4f}")
    print(f"    5-term identity |error|: {murphy['five_term_identity_check_abs_error']:.2e}")
    print(f"\nOutput directory: {rel_to_root(out_dir)}")
    print(f"Summary: {rel_to_root(summary_path)}")
    print(f"Plots: {'available' if HAS_MATPLOTLIB else 'NOT available (matplotlib not installed)'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
