# src/utils/sweep.py
"""
Run training with multiple seeds and aggregate results.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.training.trainer import train
from src.utils.config import load_config, cfg_get

logger = logging.getLogger(__name__)


def run_sweep(base_cfg: Dict[str, Any], seeds: List[int], results_dir: Path) -> Dict[str, Any]:
    """
    Run training for each seed and aggregate metrics.
    """
    summaries = []
    for seed in seeds:
        logger.info(f"Running training with seed {seed}")
        # Create a copy of the config and update seeds
        cfg = base_cfg.copy()
        cfg["splits"]["seed"] = seed
        cfg["training"]["seed"] = seed  # if needed

        # Run training
        train_results = train(cfg)

        # Load the test metrics from the evaluation (assuming evaluate was run)
        # For simplicity, we'll rely on the metrics saved during training (validation metrics)
        # But ideally we want test metrics. We'll need to run evaluate after each train.
        # Let's do a full pipeline: train -> evaluate (test)
        from src.evaluation.evaluate import run_evaluate
        run_evaluate(cfg, split="test", calibrate=False)

        # Collect test metrics
        test_metrics_path = Path(cfg["paths"]["results_dir"]) / "test_metrics.json"
        if test_metrics_path.exists():
            with open(test_metrics_path, "r") as f:
                test_metrics = json.load(f)
            summaries.append({
                "seed": seed,
                **test_metrics
            })
        else:
            logger.warning(f"Test metrics not found for seed {seed}")

    # Aggregate
    if not summaries:
        return {"error": "No results"}

    # Extract metrics of interest
    keys = ["roc_auc", "pr_auc", "brier", "f1", "precision", "recall", "threshold"]
    aggregated = {}
    for key in keys:
        values = [s.get(key, float("nan")) for s in summaries if key in s]
        if values:
            aggregated[key] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "values": values,
            }

    # Save individual summaries
    sweep_dir = results_dir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with open(sweep_dir / "individual.json", "w") as f:
        json.dump(summaries, f, indent=2)
    with open(sweep_dir / "aggregated.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    logger.info(f"Sweep results saved to {sweep_dir}")
    return aggregated