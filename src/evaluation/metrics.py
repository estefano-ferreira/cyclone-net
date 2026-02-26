from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    # Mann-Whitney U statistic approach
    scores = scores.astype(float)
    labels = labels.astype(int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # ranks
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    sum_ranks_pos = ranks[labels == 1].sum()
    n_pos = pos.size
    n_neg = neg.size
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)

def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(average_precision_score(labels, scores))


def brier(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = scores.astype(float)
    labels = labels.astype(float)
    return float(np.mean((scores - labels) ** 2))

def f1_precision_recall(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float, float]:
    pred = (scores >= threshold).astype(int)
    labels = labels.astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return float(f1), float(precision), float(recall)

def select_threshold_for_recall(scores: np.ndarray, labels: np.ndarray, target_recall: float) -> float:
    # choose highest threshold achieving recall >= target_recall (max precision under constraint)
    uniq = np.unique(scores)
    best_t = float(uniq.min())
    best_p = -1.0
    for t in uniq:
        _, p, r = f1_precision_recall(scores, labels, float(t))
        if r >= target_recall and p > best_p:
            best_p = p
            best_t = float(t)
    return best_t
