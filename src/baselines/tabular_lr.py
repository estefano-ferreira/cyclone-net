# src/baselines/tabular_lr.py
"""
Tabular baseline using logistic regression on aggregated features.
This provides a simple benchmark for the CycloneNet model.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.data.dataset import PhysicsDataset
from src.utils.config import cfg_get

logger = logging.getLogger(__name__)


def extract_features_from_cube(cube: np.ndarray, channel_names: List[str]) -> Dict[str, float]:
    """
    Extract aggregated features from a single event cube.

    Args:
        cube: (H, W, T, C) numpy array
        channel_names: list of channel names corresponding to last dimension

    Returns:
        Dictionary of feature_name -> value.
    """
    features = {}
    for c, ch in enumerate(channel_names):
        data = cube[..., c].flatten()
        data = data[np.isfinite(data)]
        if len(data) == 0:
            data = np.array([0.0], dtype=np.float32)
        features[f"{ch}_mean"] = float(np.mean(data))
        features[f"{ch}_std"] = float(np.std(data))
        features[f"{ch}_max"] = float(np.max(data))
        features[f"{ch}_min"] = float(np.min(data))
    return features


class TabularBaseline:
    """
    A simple baseline that uses aggregated features and a logistic regression classifier.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        self.feature_names: Optional[List[str]] = None

    def _load_split_data(self, split: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load all events from a given split and extract features.

        Returns:
            X: (n_samples, n_features) numpy array
            y: (n_samples,) numpy array of labels
            event_ids: list of event IDs
        """
        dataset = PhysicsDataset(self.cfg, split=split)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=0
        )
        X_list = []
        y_list = []
        event_ids = []
        input_channels = self.cfg["model"]["input_channels_names"]

        for batch in loader:
            eid = batch["event_id"][0]  # batch size 1
            # x shape: (C, T, H, W) -> convert to (H, W, T, C)
            cube_np = batch["x"].squeeze(0).permute(1, 2, 3, 0).numpy()
            y = batch["y"].item()
            features = extract_features_from_cube(cube_np, input_channels)
            # Store feature names on first run
            if self.feature_names is None:
                self.feature_names = sorted(features.keys())
            # Build feature vector in consistent order
            feat_vec = [features[name] for name in self.feature_names]
            X_list.append(feat_vec)
            y_list.append(y)
            event_ids.append(eid)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=int)
        return X, y, event_ids

    def train(self) -> None:
        """Train the baseline model on the training split."""
        logger.info("Loading training data for baseline...")
        X_train, y_train, _ = self._load_split_data("train")
        logger.info(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        self.clf.fit(X_train, y_train)
        logger.info("Baseline training complete.")

    def evaluate(self, split: str = "test") -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Evaluate on the given split and return metrics and predictions.

        Returns:
            metrics: dictionary with ROC-AUC, PR-AUC, Brier score, accuracy.
            predictions_df: DataFrame with columns event_id, y_true, y_prob, y_pred.
        """
        X, y, event_ids = self._load_split_data(split)
        y_prob = self.clf.predict_proba(X)[:, 1]
        y_pred = self.clf.predict(X)

        metrics = {
            "roc_auc": float(roc_auc_score(y, y_prob)),
            "pr_auc": float(average_precision_score(y, y_prob)),
            "brier": float(brier_score_loss(y, y_prob)),
            "accuracy": float(np.mean(y_pred == y)),
            "n_samples": len(y),
            "n_pos": int(np.sum(y)),
            "n_neg": int(len(y) - np.sum(y)),
        }

        pred_df = pd.DataFrame({
            "event_id": event_ids,
            "y_true": y,
            "y_prob": y_prob,
            "y_pred": y_pred,
        })
        return metrics, pred_df

    def save_results(self, out_dir: Path, metrics: Dict[str, Any], pred_df: pd.DataFrame) -> None:
        """Save metrics and predictions to the output directory."""
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "baseline_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        pred_df.to_csv(out_dir / "baseline_predictions.csv", index=False)
        if self.feature_names:
            with open(out_dir / "feature_names.json", "w") as f:
                json.dump(self.feature_names, f, indent=2)
        logger.info(f"Baseline results saved to {out_dir}")