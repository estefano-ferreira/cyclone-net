# src/evaluation/calibration_metrics.py
"""
Calibration metrics for probabilistic predictions.
Computes reliability diagram data, Expected Calibration Error (ECE),
and Maximum Calibration Error (MCE).

These metrics assess how well the predicted probabilities match the observed
frequencies, which is crucial for scientific interpretability.
"""

import numpy as np
from typing import Dict, Any, Tuple


def compute_reliability(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Compute reliability diagram data.

    Args:
        y_true: Binary labels (0 or 1).
        y_prob: Predicted probabilities in [0, 1].
        n_bins: Number of bins for the diagram.

    Returns:
        Dictionary containing:
            - bin_centers: Centers of each bin.
            - bin_accuracies: Observed fraction of positives in each bin.
            - bin_confidences: Average predicted probability in each bin.
            - bin_counts: Number of samples in each bin.
            - bin_edges: Edges of the bins.
            - n_bins: Number of bins.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            bin_counts[i] = mask.sum()
            bin_accuracies[i] = y_true[mask].mean()
            bin_confidences[i] = y_prob[mask].mean()
        else:
            bin_accuracies[i] = np.nan
            bin_confidences[i] = np.nan

    return {
        "bin_centers": bin_centers.tolist(),
        "bin_accuracies": bin_accuracies.tolist(),
        "bin_confidences": bin_confidences.tolist(),
        "bin_counts": bin_counts.tolist(),
        "bin_edges": bins.tolist(),
        "n_bins": n_bins,
    }


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error.

    ECE = Σ_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins.

    Returns:
        ECE value.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            weight = mask.sum() / n
            ece += weight * abs(acc - conf)

    return float(ece)


def compute_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Maximum Calibration Error.

    MCE = max_b |acc(B_b) - conf(B_b)|

    Args:
        y_true: Binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins.

    Returns:
        MCE value.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    max_err = 0.0

    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            err = abs(acc - conf)
            if err > max_err:
                max_err = err

    return float(max_err)