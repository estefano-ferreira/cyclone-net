"""Probability calibration utilities (validation-only)."""

from __future__ import annotations

from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
import numpy as np


@dataclass(frozen=True)
class PlattScaler:
    a: float
    b: float

    def predict_from_logits(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        z = self.a * logits + self.b
        return 1.0 / (1.0 + np.exp(-z))


def fit_platt_scaler(logits: np.ndarray, y_true: np.ndarray) -> PlattScaler:
    logits = logits.reshape(-1, 1)
    y_true = y_true.astype(int)
    lr = LogisticRegression(C=1e10, solver='lbfgs')
    lr.fit(logits, y_true)
    a = lr.coef_[0, 0]
    b = lr.intercept_[0]
    return PlattScaler(a, b)
