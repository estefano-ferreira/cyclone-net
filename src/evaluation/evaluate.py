from __future__ import annotations

"""
CycloneNet — Evaluation Module (Internationally Standard, Scientifically Sound).

This module evaluates a trained model on a given dataset split (test or validation).
It computes standard classification metrics, extracts FuelMap coordinates,
and optionally validates against external TCHP data.

Key features:
- Loads the best model checkpoint based on validation AUC (if available) or loss.
- Uses the same robust model builder as training.
- Loads the threshold saved during training (from best_threshold.json) and uses it.
- Computes ROC‑AUC, PR‑AUC, Brier score, and precision/recall at that threshold.
- Extracts predicted energy source locations via soft‑argmax on FuelMap.
- If TCHP maxima are present in metadata, calculates spatial error.
- Optionally applies Platt calibration to improve probability estimates.
- Generates calibration report (reliability diagram data, ECE, MCE).
- Optionally computes advanced spatial metrics (overlap, rank correlation) when full TCHP maps are available.

All external products (e.g., TCHP) are used for validation only and never as inputs.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader
from scipy.interpolate import griddata

from src.data.dataset import PhysicsDataset
from src.evaluation.metrics import (
    roc_auc,
    pr_auc,
    brier,
    f1_precision_recall,
)
from src.evaluation.spatial_metrics import compute_spatial_metrics
from src.training.trainer import _build_model
from src.utils.calibration import fit_platt_scaler, PlattScaler
from src.utils.config import cfg_get, load_config
from src.utils.tchp_utils import get_tchp_file_path

from src.evaluation.calibration_metrics import (
    compute_reliability,
    compute_ece,
    compute_mce,
)

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


def load_tchp_map_for_event(
    meta: Dict[str, Any],
    tchp_dir: Path,
    window_deg: float = 5.0
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load the full TCHP map for a given event from the corresponding NetCDF file.

    Args:
        meta: Event metadata (must contain 'timestamp', 'center_lat', 'center_lon').
        tchp_dir: Directory containing TCHP files.
        window_deg: Half-width of the spatial window in degrees.

    Returns:
        (tchp_values, lats, lons) or None if file not found or region empty.
    """
    timestamp = pd.to_datetime(meta.get("timestamp"))
    lat = meta.get("center_lat")
    lon = meta.get("center_lon")
    if timestamp is None or lat is None or lon is None:
        return None

    year = timestamp.year
    if year >= 2022:
        src = "noaa"
    elif year >= 1993:
        src = "aoml"
    else:
        return None

    tchp_file = get_tchp_file_path(tchp_dir, year, src)
    if not tchp_file.exists():
        return None

    try:
        ds = xr.open_dataset(tchp_file)
        ds_t = ds.sel(time=timestamp, method="nearest")
        lon_min, lon_max = lon - window_deg, lon + window_deg
        lat_min, lat_max = lat - window_deg, lat + window_deg

        if "lon" in ds_t.coords and ds_t.lon.min() >= 0 and lon_min < 0:
            lon_min += 360
            lon_max += 360

        ds_region = ds_t.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        if ds_region.sizes["lat"] == 0 or ds_region.sizes["lon"] == 0:
            ds.close()
            return None

        var_names = ["tchp", "Tropical_Cyclone_Heat_Potential", "TCHP"]
        tchp_var = None
        for v in var_names:
            if v in ds_region:
                tchp_var = v
                break
        if tchp_var is None:
            ds.close()
            return None

        tchp = ds_region[tchp_var].values
        lats = ds_region["lat"].values
        lons = ds_region["lon"].values
        ds.close()
        return tchp, lats, lons
    except Exception as e:
        logger.warning(f"Error loading TCHP map for event {meta.get('event_id', 'unknown')}: {e}")
        return None


def evaluate(
    cfg: Dict[str, Any],
    loader: DataLoader,
    model: torch.nn.Module,
    interim_dir: Path,
    out_csv: Path,
    out_json: Path,
    threshold: float = 0.5,
    calibrate: bool = False,
    cal_loader: Optional[DataLoader] = None,
    calibration_report: bool = True,
    full_spatial_metrics: bool = False,
    tchp_dir: Optional[Path] = None,
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
        threshold: fixed threshold for binary classification (from training).
        calibrate: if True, perform Platt calibration using cal_loader.
        cal_loader: DataLoader for calibration (usually validation set).
        calibration_report: if True, generate reliability diagram data and ECE/MCE.
        full_spatial_metrics: if True, compute advanced spatial metrics (overlap, rank correlation)
                              by loading the full TCHP map for each event.
        tchp_dir: directory containing TCHP files (required if full_spatial_metrics=True).
    """
    device = next(model.parameters()).device
    temperature = float(cfg.get("evaluation", {}).get("fuelmap_temperature", 10.0))

    model.eval()
    rows = []
    all_logits = []
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            eids = batch["event_id"]
            x = batch["x"].to(device)
            y = batch["y"].cpu().numpy().astype(int)

            prior = batch.get("prior_map_t0", None)
            if prior is not None:
                prior = prior.to(device)
            out = model(x, prior_map_t0=prior)

            logits = out["ri_logit"].cpu().numpy()
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
                            meta = json.loads(meta_path.read_text(encoding="utf-8"))
                            if "tchp_max_lat" in meta and "tchp_max_lon" in meta:
                                tl = float(meta["tchp_max_lat"])
                                tn = float(meta["tchp_max_lon"])
                                row["tchp_max_lat"] = tl
                                row["tchp_max_lon"] = tn
                                row["tchp_dist_km"] = haversine_km(plat, plon, tl, tn)

                                # Advanced spatial metrics if requested and TCHP map available
                                if full_spatial_metrics and tchp_dir is not None:
                                    tchp_data = load_tchp_map_for_event(meta, tchp_dir, window_deg=5.0)
                                    if tchp_data is not None:
                                        tchp_map, lats_tchp, lons_tchp = tchp_data
                                        # Interpolate TCHP onto FuelMap grid
                                        lon_grid_tchp, lat_grid_tchp = np.meshgrid(lons_tchp, lats_tchp)
                                        points_tchp = np.column_stack((lon_grid_tchp.ravel(), lat_grid_tchp.ravel()))
                                        values_tchp = tchp_map.ravel()
                                        # FuelMap grid (lats, lons are 2D)
                                        tchp_on_fm = griddata(points_tchp, values_tchp, (lons, lats), method='linear')
                                        if np.any(np.isfinite(tchp_on_fm)):
                                            spatial_metrics = compute_spatial_metrics(
                                                plat, plon, tl, tn,
                                                m, tchp_on_fm
                                            )
                                            for key, val in spatial_metrics.items():
                                                row[f"tchp_{key}"] = val

                rows.append(row)

    # Convert to numpy arrays
    all_labels = np.array(all_labels, dtype=int)
    all_scores = np.array(all_scores, dtype=float)
    all_logits = np.array(all_logits, dtype=float)

    # Optional Platt calibration
    if calibrate and cal_loader is not None:
        logger.info("Fitting Platt scaler on calibration set...")
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

    # Compute metrics using the fixed threshold
    f1, p, r = f1_precision_recall(scores_for_metrics, all_labels, threshold=threshold)

    summary = {
        "roc_auc": roc_auc(scores_for_metrics, all_labels),
        "pr_auc": pr_auc(scores_for_metrics, all_labels),
        "brier": brier(scores_for_metrics, all_labels),
        "threshold": float(threshold),
        "f1": float(f1),
        "precision": float(p),
        "recall": float(r),
        "n": int(all_labels.size),
        "positives": int((all_labels == 1).sum()),
        "negatives": int((all_labels == 0).sum()),
        "note": "External products (e.g., TCHP) are validation-only and never used as inputs.",
    }

    # Add TCHP distance statistics
    tchp_dists = [row["tchp_dist_km"] for row in rows if "tchp_dist_km" in row]
    if tchp_dists:
        tchp_dists = np.array(tchp_dists, dtype=float)
        summary["tchp_mean_dist_km"] = float(np.mean(tchp_dists))
        summary["tchp_median_dist_km"] = float(np.median(tchp_dists))
        summary["tchp_std_dist_km"] = float(np.std(tchp_dists))
        summary["tchp_min_dist_km"] = float(np.min(tchp_dists))
        summary["tchp_max_dist_km"] = float(np.max(tchp_dists))
        summary["tchp_n"] = int(tchp_dists.size)

    # Add advanced spatial metrics if available
    for metric in ["peak_distance_km", "top10_overlap", "rank_correlation"]:
        values = [row[f"tchp_{metric}"] for row in rows if f"tchp_{metric}" in row]
        if values:
            values = np.array(values, dtype=float)
            summary[f"tchp_{metric}_mean"] = float(np.mean(values))
            summary[f"tchp_{metric}_median"] = float(np.median(values))
            summary[f"tchp_{metric}_std"] = float(np.std(values))

    # Calibration report
    if calibration_report:
        cal_data = compute_reliability(all_labels, scores_for_metrics, n_bins=10)
        cal_data["ece"] = compute_ece(all_labels, scores_for_metrics, n_bins=10)
        cal_data["mce"] = compute_mce(all_labels, scores_for_metrics, n_bins=10)
        cal_path = out_json.parent / "calibration_data.json"
        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump(cal_data, f, indent=2)
        logger.info(f"Calibration data saved to {cal_path}")
        summary["ece"] = cal_data["ece"]
        summary["mce"] = cal_data["mce"]

    # Save outputs
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(f"Saved predictions to: {out_csv}")
    logger.info(f"Saved metrics to: {out_json}")


def run_evaluate(
    cfg: Dict[str, Any],
    split: str = "test",
    calibrate: bool = False,
    calibration_report: bool = True,
    full_spatial_metrics: bool = False,
) -> None:
    """
    Config-driven evaluation entrypoint.

    Args:
        cfg: configuration dictionary.
        split: which dataset split to evaluate ("test" or "val").
        calibrate: whether to apply Platt calibration using validation set.
        calibration_report: whether to generate calibration report.
        full_spatial_metrics: whether to compute advanced spatial metrics (requires TCHP maps).
    """
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg_get(cfg, "training.device", "auto") in ("auto", "cuda")
        else "cpu"
    )

    batch_size = int(cfg_get(cfg, "training.batch_size", 16))
    num_workers = int(cfg_get(cfg, "repro.num_workers", 4))

    ds = PhysicsDataset(cfg, split=split)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = _build_model(cfg).to(device)

    ckpt_dir = Path(cfg_get(cfg, "paths.checkpoints_dir", "./models/checkpoints")).resolve()
    best_auc_path = ckpt_dir / "best_auc_model.pt"
    best_loss_path = ckpt_dir / "best_model.pt"

    if best_auc_path.exists():
        checkpoint = torch.load(best_auc_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        logger.info(f"Loaded model from {best_auc_path} (AUC-based)")
    elif best_loss_path.exists():
        checkpoint = torch.load(best_loss_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        logger.info(f"Loaded model from {best_loss_path} (loss-based)")
    else:
        logger.warning("No checkpoint found; using randomly initialized model.")

    model.eval()

    threshold_path = ckpt_dir / "best_threshold.json"
    threshold = 0.5
    if threshold_path.exists():
        with open(threshold_path, "r") as f:
            thr_data = json.load(f)
        threshold = thr_data["threshold"]
        logger.info(f"Loaded threshold {threshold:.4f} from {threshold_path}")
    else:
        logger.warning("No threshold file found; using default 0.5")

    interim_dir = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    results_dir = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    out_csv = results_dir / f"{split}_predictions.csv"
    out_json = results_dir / f"{split}_metrics.json"

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

    tchp_dir = Path(cfg_get(cfg, "paths.tchp_dir", "./data/external/tchp")).resolve() if full_spatial_metrics else None

    evaluate(
        cfg,
        loader,
        model,
        interim_dir,
        out_csv,
        out_json,
        threshold=threshold,
        calibrate=calibrate,
        cal_loader=cal_loader,
        calibration_report=calibration_report,
        full_spatial_metrics=full_spatial_metrics,
        tchp_dir=tchp_dir,
    )


def main(cfg: Optional[Dict[str, Any]] = None) -> None:
    """
    CLI-compatible entrypoint.

    Usage:
        python -m src.evaluation.evaluate [--split test] [--calibrate] [--no-calibration-report] [--full-spatial]
    """
    import argparse

    parser = argparse.ArgumentParser(description="CycloneNet evaluation")
    parser.add_argument("--split", default="test", choices=["test", "val"], help="Split to evaluate")
    parser.add_argument("--calibrate", action="store_true", help="Apply Platt calibration using validation set")
    parser.add_argument("--no-calibration-report", action="store_true", help="Disable generation of calibration report")
    parser.add_argument("--full-spatial", action="store_true", help="Compute advanced spatial metrics (requires TCHP maps)")
    args = parser.parse_args()

    if cfg is None:
        cfg = load_config("config.yaml")

    run_evaluate(
        cfg,
        split=args.split,
        calibrate=args.calibrate,
        calibration_report=not args.no_calibration_report,
        full_spatial_metrics=args.full_spatial,
    )


if __name__ == "__main__":
    main()