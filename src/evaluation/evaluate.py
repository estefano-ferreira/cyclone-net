"""
CycloneNet V2.1 — Scientific evaluation (paper-safe).

This script loads the best model checkpoint and artifacts (threshold + calibration if available),
evaluates classification metrics on the TEST split, computes spatial "hotspot" distance to the
declared target (proxy), and produces a full audit trail: per-sample CSV and summary JSON.

Scientific notes:
- If true_energy_lat/lon is defined as cyclone center, spatial error is a sanity check only.
- If true_energy_lat/lon is computed via a physical proxy (e.g., max energy field),
  you must document that proxy in the paper.

Author: Estefano Senhor Ferreira
License: CC BY-NC 4.0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from geopy.distance import geodesic

from src.utils.config import CONFIG, cfg_get
from src.data.dataset import PhysicsDataset
from src.models.cyclone_net_ri_only import CycloneNetRIOnly
from src.evaluation.metrics import MissionEvaluator
from src.evaluation.interpretability import integrated_gradients
from src.utils.geometry_utils import soft_argmax, normalized_to_geographic

logger = logging.getLogger(__name__)


def _resolve_path(p: Any) -> Path:
    """Convert to absolute Path, expanding user and resolving."""
    return Path(str(p)).expanduser().resolve()


def _load_json(p: Path) -> Dict[str, Any]:
    """Load JSON file and return dictionary."""
    return json.loads(p.read_text(encoding="utf-8"))


def _load_artifacts(checkpoints_dir: Path) -> Dict[str, Any]:
    """Load best_model_ri_artifacts.json if it exists; otherwise return empty dict."""
    art = checkpoints_dir / "best_model_ri_artifacts.json"
    if not art.exists():
        logger.warning(
            "Artifacts not found: %s (will use config fallbacks).", art)
        return {}
    try:
        return _load_json(art)
    except Exception as e:
        logger.warning("Failed to read artifacts (%s): %s", art, e)
        return {}


def _predict_scores_and_coords(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    interim_dir: Path,
) -> list[dict]:
    """
    Iterate over loader, collect predictions and coordinates.

    For each sample:
        - Inference is run without gradients to obtain the probability score.
        - A separate forward/backward pass with gradients is used to compute
          Integrated Gradients and generate the spatial heatmap.
    """
    model.eval()
    results = []

    for batch in loader:
        # Move input to device (no gradients for the initial inference)
        x = batch["input"].to(device, non_blocking=True)

        # ----- Inference for probability (no gradients) -----
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

        # ----- Process each sample in the batch -----
        for i in range(x.size(0)):
            event_id = batch["event_id"][i]
            y_true = batch["ri_label"][i].item()
            true_lat = batch["true_lat"][i].item()
            true_lon = batch["true_lon"][i].item()
            storm_name = batch["storm_name"][i] if "storm_name" in batch else ""
            timestamp = batch["timestamp"][i] if "timestamp" in batch else ""

            # Create a copy of the input that requires gradients for IG
            sample_x = x[i:i+1].clone().detach().requires_grad_(True)

            # ----- Integrated Gradients (requires gradients) -----
            ig = integrated_gradients(
                model,
                sample_x,
                steps=int(cfg_get(CONFIG, "evaluation.ig_steps", 100))
            )
            # Aggregate along channels and time to produce a 2D spatial map.
            # Use sum over channels (abs) to combine contributions from all physical variables,
            # then max over time to capture the moment of strongest influence (configurable).
            heat = ig.abs().sum(dim=1).squeeze(0)  # (T, H, W)
            agg = str(
                cfg_get(CONFIG, "evaluation.saliency_aggregation", "max")).lower()
            if agg == "mean":
                heat2d = heat.mean(dim=0).detach().cpu().numpy()
            else:
                heat2d = heat.max(dim=0)[0].detach().cpu().numpy()

            # Normalize heatmap to [0,1] for soft-argmax
            heat2d = heat2d - heat2d.min()
            if heat2d.max() > 0:
                heat2d = heat2d / (heat2d.max() + 1e-12)

            # Convert to tensor (without gradients) and apply soft-argmax
            heat_t = torch.tensor(heat2d[None, None, ...], dtype=torch.float32)
            coords = soft_argmax(
                heat_t,
                temperature=float(
                    cfg_get(CONFIG, "model.localizer.softargmax_temperature", 1.0))
            )
            y_norm = coords[0, 0].item()
            x_norm = coords[0, 1].item()

            # Load geographic grids for this event
            lats_2d = np.load(interim_dir / f"{event_id}_lats.npy")
            lons_2d = np.load(interim_dir / f"{event_id}_lons.npy")
            pred_lat, pred_lon = normalized_to_geographic(
                y_norm, x_norm, lats_2d, lons_2d)

            # Compute spatial error (km)
            error_km = geodesic((true_lat, true_lon), (pred_lat, pred_lon)).km

            results.append({
                "event_id": event_id,
                "y_true": int(y_true),
                "y_score": float(probs[i]),
                "pred_lat": pred_lat,
                "pred_lon": pred_lon,
                "true_lat": true_lat,
                "true_lon": true_lon,
                "error_km": error_km,
                "storm_name": storm_name,
                "timestamp": timestamp,
            })

    return results


def main(checkpoint: Optional[str] = None) -> None:
    """
    Main evaluation routine.
    If checkpoint is not provided, the best model from training is used.
    """
    # Resolve directories
    checkpoints_dir = _resolve_path(
        cfg_get(CONFIG, "paths.checkpoints", "./models/checkpoints"))
    interim_dir = _resolve_path(
        cfg_get(CONFIG, "paths.interim_data", "./data/interim"))
    results_dir = _resolve_path(
        cfg_get(CONFIG, "paths.results", "./outputs/results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = CycloneNetRIOnly(CONFIG).to(device)
    if checkpoint is None:
        checkpoint = checkpoints_dir / "best_model_ri.pt"
    else:
        checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    logger.info("Loaded model from %s", checkpoint)

    # Load artifacts (threshold, calibration, etc.)
    artifacts = _load_artifacts(checkpoints_dir)

    # Threshold (artifacts override config)
    fallback_threshold = float(
        cfg_get(CONFIG, "thresholding.fallback_threshold", 0.5))
    threshold = float(artifacts.get("selected_threshold", fallback_threshold))

    # Build test loader
    batch_size = int(cfg_get(CONFIG, "training.batch_size", 64))
    test_ds = PhysicsDataset(split="test", balance_ri=False, augment=False)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info("Test samples: %d", len(test_ds))

    # Run prediction and coordinate extraction
    results = _predict_scores_and_coords(
        model, test_loader, device, interim_dir)

    # Initialize evaluator and add results
    evaluator = MissionEvaluator(results_dir)
    for r in results:
        evaluator.add(r)

    # Finalize and save summaries
    summary = evaluator.finalize(prefix="test_set")

    logger.info("Evaluation complete.")
    logger.info("ROC-AUC: %.4f, PR-AUC: %.4f", summary.roc_auc, summary.pr_auc)
    logger.info("Spatial error (km) - mean: %.2f, median: %.2f",
                summary.spatial_error_km_mean, summary.spatial_error_km_median)


if __name__ == "__main__":
    logging.basicConfig(level=cfg_get(CONFIG, "logging.level", "INFO"))
    main()
