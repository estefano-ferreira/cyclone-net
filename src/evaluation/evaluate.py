from __future__ import annotations

"""
CycloneNet — Evaluation Module (Internationally Standard, Scientifically Sound).

This module evaluates a trained model on a given dataset split (test or validation).
It computes standard classification metrics, extracts FuelMap coordinates,
and optionally validates against external TCHP data.

Key features:
- Loads the best model checkpoint based on validation AUC (if available) or loss.
- Uses the same robust model builder as training.
- Computes ROC‑AUC, PR‑AUC, Brier score, and precision/recall at target recall.
- Extracts predicted energy source locations via soft‑argmax on FuelMap.
- If TCHP maxima are present in metadata, calculates spatial error.
- Optionally applies Platt calibration to improve probability estimates.

All external products (e.g., TCHP) are used for validation only and never as inputs.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import PhysicsDataset
from src.evaluation.metrics import (
    roc_auc,
    pr_auc,
    brier,
    f1_precision_recall,
    select_threshold_for_recall,
)
from src.training.trainer import _build_model  # reuse the robust model builder
from src.utils.calibration import fit_platt_scaler, PlattScaler
from src.utils.config import cfg_get, load_config

logger = logging.getLogger(__name__)


def soft_argmax_2d(m: np.ndarray, temperature: float = 10.0) -> Tuple[float, float]:
    """
    Compute expected coordinates (in pixel space) from a 2D heatmap.

    Args:
        m: 2D numpy array (H, W)
        temperature: softmax temperature (higher = more diffuse)

    Returns:
        (y, x) floating-point pixel coordinates.
    """
    H, W = m.shape
    a = m.astype(np.float64) * float(temperature)
    a = a - np.max(a)
    w = np.exp(a)
    w = w / (np.sum(w) + 1e-12)
    ys = np.arange(H, dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)
    y = float((w.sum(axis=1) * ys).sum())
    x = float((w.sum(axis=0) * xs).sum())
    return y, x


def pixel_to_geo(y: float, x: float, lats: np.ndarray, lons: np.ndarray) -> Tuple[float, float]:
    """
    Convert pixel coordinates to geographic coordinates using precomputed lat/lon grids.

    Args:
        y, x: pixel coordinates (floating point)
        lats: 2D array of latitudes (H, W)
        lons: 2D array of longitudes (H, W)

    Returns:
        (lat, lon) in degrees.
    """
    yi = int(np.clip(round(y), 0, lats.shape[0] - 1))
    xi = int(np.clip(round(x), 0, lats.shape[1] - 1))
    return float(lats[yi, xi]), float(lons[yi, xi])


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points on Earth in kilometers."""
    R = 6371.0
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dphi = p2 - p1
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def _get_fuelmap(out: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """Extract fuelmap tensor from model output, supporting both 'fuelmap' and 'fuelmap_logits' keys."""
    if "fuelmap" in out:
        return out["fuelmap"]
    if "fuelmap_logits" in out:
        return out["fuelmap_logits"]
    return None


def evaluate(
    cfg: Dict[str, Any],
    loader: DataLoader,
    model: torch.nn.Module,
    interim_dir: Path,
    out_csv: Path,
    out_json: Path,
    calibrate: bool = False,
    cal_loader: Optional[DataLoader] = None,
) -> None:
    """
    Core evaluation routine.

    Args:
        cfg: configuration dictionary.
        loader: DataLoader for the split to evaluate.
        model: trained model.
        interim_dir: directory containing per-event metadata and grids.
        out_csv: path to save detailed predictions CSV.
        out_json: path to save summary metrics JSON.
        calibrate: if True, perform Platt calibration using cal_loader.
        cal_loader: DataLoader for calibration (usually validation set).
    """
    device = next(model.parameters()).device
    temperature = float(cfg.get("evaluation", {}).get(
        "fuelmap_temperature", 10.0))
    target_recall = float(cfg_get(cfg, "training.eval_target_recall", 0.90))

    model.eval()
    rows = []
    all_logits = []   # raw logits (before sigmoid) for potential calibration
    all_scores = []   # probabilities after sigmoid
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            eids = batch["event_id"]
            x = batch["x"].to(device)
            y = batch["y"].cpu().numpy().astype(int)

            # Pass optional prior_map_t0 if available (model may use it)
            prior = batch.get("prior_map_t0", None)
            if prior is not None:
                prior = prior.to(device)
            out = model(x, prior_map_t0=prior)

            logits = out["ri_logit"].cpu().numpy()          # raw logits
            scores = torch.sigmoid(out["ri_logit"]).cpu().numpy()
            all_logits.extend(logits)
            all_scores.extend(scores)
            all_labels.extend(y)

            fm_t = _get_fuelmap(out)

            for i, eid in enumerate(eids):
                row = {
                    "event_id": str(eid),
                    "y_true": int(y[i]),
                    "y_score": float(scores[i]),
                }

                if fm_t is not None:
                    lats_path = interim_dir / f"{eid}_lats.npy"
                    lons_path = interim_dir / f"{eid}_lons.npy"
                    if lats_path.exists() and lons_path.exists():
                        lats = np.load(lats_path)
                        lons = np.load(lons_path)
                        m = fm_t[i, 0].cpu().numpy()        # (H, W)
                        py, px = soft_argmax_2d(m, temperature=temperature)
                        plat, plon = pixel_to_geo(py, px, lats, lons)
                        row["pred_lat"] = plat
                        row["pred_lon"] = plon

                        # Optional TCHP ground truth from metadata
                        meta_path = interim_dir / f"{eid}.json"
                        if meta_path.exists():
                            meta = json.loads(
                                meta_path.read_text(encoding="utf-8"))
                            if "tchp_max_lat" in meta and "tchp_max_lon" in meta:
                                tl = float(meta["tchp_max_lat"])
                                tn = float(meta["tchp_max_lon"])
                                row["tchp_max_lat"] = tl
                                row["tchp_max_lon"] = tn
                                row["tchp_dist_km"] = haversine_km(
                                    plat, plon, tl, tn)

                rows.append(row)

    # Convert to numpy arrays
    all_labels = np.array(all_labels, dtype=int)
    all_scores = np.array(all_scores, dtype=float)
    all_logits = np.array(all_logits, dtype=float)

    # Optional Platt calibration
    if calibrate and cal_loader is not None:
        logger.info("Fitting Platt scaler on calibration set...")
        # Collect logits and labels from calibration loader
        cal_logits, cal_labels = [], []
        model.eval()
        with torch.no_grad():
            for batch in cal_loader:
                x = batch["x"].to(device)
                y = batch["y"].cpu().numpy()
                out = model(x)
                cal_logits.extend(out["ri_logit"].cpu().numpy())
                cal_labels.extend(y)
        cal_logits = np.array(cal_logits)
        cal_labels = np.array(cal_labels, dtype=int)
        scaler = fit_platt_scaler(cal_logits, cal_labels)
        calibrated_scores = scaler.predict_from_logits(all_logits)
        logger.info("Calibration applied.")
        scores_for_metrics = calibrated_scores
    else:
        scores_for_metrics = all_scores

    # Compute metrics with threshold optimized for target recall
    thr = select_threshold_for_recall(
        scores_for_metrics, all_labels, target_recall=target_recall)
    f1, p, r = f1_precision_recall(
        scores_for_metrics, all_labels, threshold=thr)

    summary = {
        "roc_auc": roc_auc(scores_for_metrics, all_labels),
        "pr_auc": pr_auc(scores_for_metrics, all_labels),
        "brier": brier(scores_for_metrics, all_labels),
        "threshold": float(thr),
        "f1": float(f1),
        "precision": float(p),
        "recall": float(r),
        "n": int(all_labels.size),
        "positives": int((all_labels == 1).sum()),
        "negatives": int((all_labels == 0).sum()),
        "note": "External products (e.g., TCHP) are validation-only and never used as inputs.",
    }

    # Add TCHP statistics if any rows contain distance
    tchp_dists = [row["tchp_dist_km"] for row in rows if "tchp_dist_km" in row]
    if tchp_dists:
        tchp_dists = np.array(tchp_dists, dtype=float)
        summary["tchp_mean_dist_km"] = float(np.mean(tchp_dists))
        summary["tchp_median_dist_km"] = float(np.median(tchp_dists))
        summary["tchp_std_dist_km"] = float(np.std(tchp_dists))
        summary["tchp_n"] = int(tchp_dists.size)

    # Save outputs
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(f"Saved predictions to: {out_csv}")
    logger.info(f"Saved metrics to: {out_json}")


def run_evaluate(cfg: Dict[str, Any], split: str = "test", calibrate: bool = False) -> None:
    """
    Config-driven evaluation entrypoint.

    Args:
        cfg: configuration dictionary.
        split: which dataset split to evaluate ("test" or "val").
        calibrate: whether to apply Platt calibration using validation set.
    """
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg_get(cfg, "training.device", "auto") in ("auto", "cuda")
        else "cpu"
    )

    batch_size = int(cfg_get(cfg, "training.batch_size", 16))
    num_workers = int(cfg_get(cfg, "repro.num_workers", 4))

    # Build dataset and loader for the requested split
    ds = PhysicsDataset(cfg, split=split)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # Build model (same architecture as training)
    model = _build_model(cfg).to(device)

    # Determine checkpoint path: prefer AUC-based if exists, else best loss
    ckpt_dir = Path(cfg_get(cfg, "paths.checkpoints_dir",
                    "./models/checkpoints")).resolve()
    best_auc_path = ckpt_dir / "best_auc_model.pt"
    best_loss_path = ckpt_dir / "best_model.pt"

    if best_auc_path.exists():
        checkpoint = torch.load(best_auc_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        logger.info(f"Loaded model from {best_auc_path} (AUC-based)")
    elif best_loss_path.exists():
        checkpoint = torch.load(best_loss_path, map_location=device)
        # Checkpoint may contain full dict or just state_dict
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        logger.info(f"Loaded model from {best_loss_path} (loss-based)")
    else:
        logger.warning(
            "No checkpoint found; using randomly initialized model.")

    model.eval()

    interim_dir = Path(cfg_get(cfg, "paths.interim_data",
                       "./data/interim")).resolve()
    results_dir = Path(cfg_get(cfg, "paths.results_dir",
                       "./outputs/results")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    out_csv = results_dir / f"{split}_predictions.csv"
    out_json = results_dir / f"{split}_metrics.json"

    # If calibration requested, also load validation loader
    cal_loader = None
    if calibrate:
        ds_val = PhysicsDataset(cfg, split="val")
        cal_loader = DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    evaluate(
        cfg,
        loader,
        model,
        interim_dir,
        out_csv,
        out_json,
        calibrate=calibrate,
        cal_loader=cal_loader,
    )


def main(cfg: Optional[Dict[str, Any]] = None) -> None:
    """
    CLI-compatible entrypoint.

    Usage:
        python -m src.evaluation.evaluate [--split test] [--calibrate]
    """
    import argparse

    parser = argparse.ArgumentParser(description="CycloneNet evaluation")
    parser.add_argument("--split", default="test",
                        choices=["test", "val"], help="Split to evaluate")
    parser.add_argument("--calibrate", action="store_true",
                        help="Apply Platt calibration using validation set")
    args = parser.parse_args()

    if cfg is None:
        cfg = load_config("config.yaml")

    run_evaluate(cfg, split=args.split, calibrate=args.calibrate)


if __name__ == "__main__":
    main()
