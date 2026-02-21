"""
CycloneNet V2.1 — Scientific evaluation (paper-safe).

This script loads the best model checkpoint and artifacts (threshold + calibration if available),
evaluates classification metrics on the TEST split, computes spatial "hotspot" distance to the
declared target (proxy), and produces a full audit trail: per-sample CSV and summary JSON.

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
from tqdm import tqdm

from src.utils.config import CONFIG, cfg_get
from src.data.dataset import PhysicsDataset
from src.models.cyclone_net_ri_only import CycloneNetRIOnly
from src.evaluation.metrics import MissionEvaluator
from src.evaluation.interpretability import integrated_gradients
from src.utils.geometry_utils import soft_argmax, normalized_to_geographic

logger = logging.getLogger(__name__)


def _resolve_path(p: Any) -> Path:
    return Path(str(p)).expanduser().resolve()


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _load_artifacts(checkpoints_dir: Path) -> Dict[str, Any]:
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
    calibration_params: Optional[Dict[str, float]] = None,
) -> list[dict]:
    """
    Iterate over loader, collect predictions and coordinates.

    Args:
        calibration_params: dict with 'a' and 'b' for Platt scaling (optional)
    """
    model.eval()
    results = []
    total_batches = len(loader)

    # Barra de progresso sobre os batches
    pbar = tqdm(loader, desc="Evaluating", unit="batch", total=total_batches)
    for batch in pbar:
        # Inference without gradients to get logits
        with torch.no_grad():
            x = batch["input"].to(device, non_blocking=True)
            logits = model(x)  # shape (B, 1)
            logits_np = logits.detach().cpu().numpy().reshape(-1)

        # Apply calibration if available
        if calibration_params is not None:
            a = calibration_params["a"]
            b = calibration_params["b"]
            calibrated_logits = a * logits_np + b
            probs = 1.0 / (1.0 + np.exp(-calibrated_logits))
        else:
            probs = 1.0 / (1.0 + np.exp(-logits_np))

        # Process each sample in batch with gradients for IG
        for i in range(x.size(0)):
            event_id = batch["event_id"][i]
            y_true = batch["ri_label"][i].item()
            true_lat = batch["true_lat"][i].item()
            true_lon = batch["true_lon"][i].item()
            storm_name = batch.get("storm_name", [""])[i]
            timestamp = batch.get("timestamp", [""])[i]

            # Create tensor with requires_grad for this sample
            sample_x = x[i:i+1].clone().detach().requires_grad_(True)

            # Integrated gradients
            ig = integrated_gradients(
                model,
                sample_x,
                steps=int(cfg_get(CONFIG, "evaluation.ig_steps", 100))
            )
            heat = ig.abs().sum(dim=1).squeeze(0)  # (T, H, W)

            agg = str(
                cfg_get(CONFIG, "evaluation.saliency_aggregation", "max")).lower()
            if agg == "mean":
                heat2d = heat.mean(dim=0).detach().cpu().numpy()
            else:
                heat2d = heat.max(dim=0)[0].detach().cpu().numpy()

            # Normalize to [0,1]
            heat2d = heat2d - heat2d.min()
            if heat2d.max() > 0:
                heat2d = heat2d / (heat2d.max() + 1e-12)

            # Soft-argmax
            heat_t = torch.tensor(heat2d[None, None, ...], dtype=torch.float32)
            y_norm, x_norm = soft_argmax(
                heat_t,
                temperature=float(
                    cfg_get(CONFIG, "model.localizer.softargmax_temperature", 1.0))
            )
            y_norm = y_norm.item()
            x_norm = x_norm.item()

            # Convert to geographic coordinates
            lats_2d = np.load(interim_dir / f"{event_id}_lats.npy")
            lons_2d = np.load(interim_dir / f"{event_id}_lons.npy")
            pred_lat, pred_lon = normalized_to_geographic(
                y_norm, x_norm, lats_2d, lons_2d)

            # Spatial error
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

        # Atualiza a barra com info adicional (opcional)
        pbar.set_postfix({"batch_size": x.size(0)})

    pbar.close()
    return results


def main(checkpoint: Optional[str] = None) -> None:
    checkpoints_dir = _resolve_path(
        cfg_get(CONFIG, "paths.checkpoints", "./models/checkpoints"))
    interim_dir = _resolve_path(
        cfg_get(CONFIG, "paths.interim_data", "./data/interim"))
    results_dir = _resolve_path(
        cfg_get(CONFIG, "paths.results", "./outputs/results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CycloneNetRIOnly(CONFIG).to(device)
    if checkpoint is None:
        checkpoint = checkpoints_dir / "best_model_ri.pt"
    else:
        checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    # Load artifacts
    artifacts = _load_artifacts(checkpoints_dir)
    logger.info(f"Artifacts loaded. Keys: {list(artifacts.keys())}")

    # Extract threshold from val_artifacts.thresholding.selected_threshold
    try:
        threshold = artifacts["val_artifacts"]["thresholding"]["selected_threshold"]
        logger.info(f"Using threshold from artifacts: {threshold:.4f}")
    except (KeyError, TypeError) as e:
        threshold = float(
            cfg_get(CONFIG, "thresholding.fallback_threshold", 0.5))
        logger.warning(
            f"Could not read threshold from artifacts: {e}. Using fallback: {threshold}")

    # Extract calibration parameters from val_artifacts.calibration
    calibration_params = None
    try:
        cal = artifacts["val_artifacts"]["calibration"]
        if cal.get("enabled"):
            calibration_params = {"a": cal["a"], "b": cal["b"]}
            logger.info(
                f"Calibration enabled: a={calibration_params['a']:.4f}, b={calibration_params['b']:.4f}")
        else:
            logger.info("Calibration is disabled in artifacts.")
    except (KeyError, TypeError):
        logger.info("No calibration parameters found in artifacts.")

    # Build test loader
    batch_size = int(cfg_get(CONFIG, "training.batch_size", 64))
    test_ds = PhysicsDataset(split="test", balance_ri=False, augment=False)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    logger.info(f"Test samples: {len(test_ds)}")

    results = _predict_scores_and_coords(
        model, test_loader, device, interim_dir,
        calibration_params=calibration_params
    )

    # Optional: check score distribution
    scores = [r["y_score"] for r in results]
    logger.info(
        f"Score stats - min: {min(scores):.4f}, max: {max(scores):.4f}, mean: {np.mean(scores):.4f}")

    evaluator = MissionEvaluator(results_dir)
    for r in results:
        evaluator.add(r)

    # Pass the threshold to finalize for binarization metrics
    summary = evaluator.finalize(prefix="test_set", threshold=threshold)

    logger.info("Evaluation complete.")
    logger.info(
        f"ROC-AUC: {summary.roc_auc:.4f}, PR-AUC: {summary.pr_auc:.4f}")
    logger.info(
        f"Spatial error (km) - mean: {summary.spatial_error_km_mean:.2f}, median: {summary.spatial_error_km_median:.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=cfg_get(CONFIG, "logging.level", "INFO"))
    main()
