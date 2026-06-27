"""Tests for validation-only threshold selection policies."""
import numpy as np

from src.utils.thresholding import ThresholdConfig, select_threshold


def _data():
    rng = np.random.default_rng(7)
    y = np.r_[np.ones(60), np.zeros(540)].astype(int)
    s = np.r_[rng.beta(6, 2, 60), rng.beta(2, 6, 540)]
    return y, s


def test_precision_at_recall_meets_recall_target():
    y, s = _data()
    thr, m = select_threshold(y, s, ThresholdConfig(method="precision_at_recall", min_recall=0.9))
    assert m["recall"] >= 0.9


def test_max_f1_is_at_least_as_good_as_any_threshold():
    y, s = _data()
    thr, m = select_threshold(y, s, ThresholdConfig(method="max_f1"))
    # Brute-force the best achievable F1 and confirm the policy matches it.
    best = 0.0
    for t in np.unique(s):
        pred = (s >= t).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        best = max(best, f1)
    assert m["f1"] >= best - 1e-9


def test_empty_input_returns_fallback():
    thr, m = select_threshold(np.array([]), np.array([]), ThresholdConfig(fallback_threshold=0.42))
    assert thr == 0.42
