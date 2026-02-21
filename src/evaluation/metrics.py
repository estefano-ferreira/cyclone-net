"""
CycloneNet V2.1 — Evaluation metrics utilities (MissionEvaluator).

This provides the missing MissionEvaluator class referenced by evaluate.py.

Design:
  - Computes standard binary classification metrics.
  - Optionally computes spatial error (km) when pred/true coordinates exist.
  - Writes a per-sample CSV and a summary JSON (audit trail).
  - Also generates reliability diagram if requested.
"""

from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)
from geopy.distance import geodesic


@dataclass
class MissionSummary:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    brier: float
    spatial_error_km_mean: Optional[float] = None
    spatial_error_km_median: Optional[float] = None


class MissionEvaluator:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rows: List[Dict[str, Any]] = []

    def add(self, row: Dict[str, Any]) -> None:
        self.rows.append(dict(row))

    def _safe_float(self, x) -> Optional[float]:
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    def finalize(self, prefix: str = "evaluation", plot_reliability: bool = True, threshold: float = 0.5) -> MissionSummary:
        """
        Finalize the evaluation, compute metrics and save artifacts.

        Args:
            prefix: Prefix for output files.
            plot_reliability: Whether to generate reliability diagram.
            threshold: Decision threshold for binary classification metrics (precision, recall, f1).

        Returns:
            MissionSummary object.
        """
        if not self.rows:
            raise RuntimeError("No rows added to evaluator.")

        y_true = np.asarray([int(r.get("y_true", 0))
                            for r in self.rows], dtype=np.int32)
        y_score = np.asarray([float(r.get("y_score", 0.0))
                             for r in self.rows], dtype=np.float32)
        y_pred = (y_score >= threshold).astype(np.int32)

        roc_auc = float(roc_auc_score(y_true, y_score)) if len(
            np.unique(y_true)) >= 2 else 0.0
        pr_auc = float(average_precision_score(y_true, y_score)
                       ) if len(np.unique(y_true)) >= 2 else 0.0
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        brier = float(brier_score_loss(y_true, y_score))

        # Spatial errors if coordinates exist
        errs = []
        for r in self.rows:
            pl = self._safe_float(r.get("pred_lat"))
            plo = self._safe_float(r.get("pred_lon"))
            tl = self._safe_float(r.get("true_lat"))
            tlo = self._safe_float(r.get("true_lon"))
            if None in (pl, plo, tl, tlo):
                continue
            errs.append(geodesic((tl, tlo), (pl, plo)).km)

        summary = MissionSummary(
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            precision=precision,
            recall=recall,
            f1=f1,
            brier=brier,
            spatial_error_km_mean=float(np.mean(errs)) if errs else None,
            spatial_error_km_median=float(np.median(errs)) if errs else None,
        )

        # Write artifacts
        csv_path = self.output_dir / f"{prefix}_samples.csv"
        keys = sorted({k for r in self.rows for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)

        json_path = self.output_dir / f"{prefix}_summary.json"
        json_path.write_text(json.dumps(
            summary.__dict__, indent=2), encoding="utf-8")

        if plot_reliability:
            self._plot_reliability(
                y_true, y_score, self.output_dir / f"{prefix}_reliability.png"
            )

        return summary

    def _plot_reliability(self, y_true, y_prob, save_path):
        """Generate and save reliability diagram."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        frac_pos = []
        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
            if np.sum(mask) > 0:
                frac_pos.append(np.mean(y_true[mask]))
            else:
                frac_pos.append(np.nan)

        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.plot(bin_centers, frac_pos, marker='o', label='Model')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Reliability Diagram')
        plt.legend()
        plt.savefig(save_path, dpi=150)
        plt.close()
