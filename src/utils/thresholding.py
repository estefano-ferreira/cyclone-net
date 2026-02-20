"""Threshold selection policies (validation-only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class ThresholdConfig:
    method: str = "precision_at_recall"
    min_recall: float = 0.90
    fallback_threshold: float = 0.5


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def select_threshold(y_true: np.ndarray, y_score: np.ndarray, cfg: ThresholdConfig) -> Tuple[float, Dict[str, float]]:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_score = np.asarray(y_score).astype(float).reshape(-1)
    if y_true.size == 0:
        return cfg.fallback_threshold, {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    thresholds = np.unique(y_score)
    best_thr = cfg.fallback_threshold
    best = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        if cfg.method == "max_f1":
            if f1 > best["f1"]:
                best_thr = float(thr)
                best = {"precision": precision, "recall": recall, "f1": f1}
        elif cfg.method == "precision_at_recall":
            if recall >= cfg.min_recall and precision > best["precision"]:
                best_thr = float(thr)
                best = {"precision": precision, "recall": recall, "f1": f1}
        else:
            raise ValueError(f"Unknown threshold method: {cfg.method}")

    return best_thr, best
