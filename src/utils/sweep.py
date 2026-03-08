
from __future__ import annotations

"""CycloneNet — reproducible multi-seed sweep utility."""

import copy
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def run_sweep(base_cfg: Dict[str, Any], seeds: List[int], results_dir: Path) -> Dict[str, Any]:
    from src.training.trainer import train
    from src.evaluation.evaluate import run_evaluate

    results_dir = Path(results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []

    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("splits", {})["seed"] = int(seed)
        cfg.setdefault("training", {})["seed"] = int(seed)

        seed_dir = results_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        cfg.setdefault("paths", {})["results_dir"] = str(seed_dir)

        logger.info("Running sweep seed %d", seed)
        train(cfg)
        run_evaluate(cfg, split="test", calibrate=False)

        metrics_path = seed_dir / "test_metrics.json"
        if not metrics_path.exists():
            logger.warning("Test metrics not found for seed %d", seed)
            continue

        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        summaries.append({"seed": seed, **metrics})

        cfg_dump = seed_dir / "effective_config.json"
        with cfg_dump.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    if not summaries:
        return {"error": "No successful sweep runs."}

    aggregated: Dict[str, Any] = {"n_runs": len(summaries), "seeds": seeds}
    numeric_keys = sorted(
        {
            key
            for row in summaries
            for key, value in row.items()
            if isinstance(value, (int, float)) and key != "seed"
        }
    )

    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in summaries if key in row], dtype=np.float64)
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values.tolist(),
        }

    with (results_dir / "individual_runs.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    with (results_dir / "aggregated_summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    logger.info("Sweep completed. Results saved to %s", results_dir)
    return aggregated