from __future__ import annotations

"""
CycloneNet — release-aligned evaluation.

Scientific intent
-----------------
This module evaluates the released CycloneNet model in a way that remains
consistent with the paper's claims:

- Classification and probabilistic metrics are primary.
- The operating threshold is selected on validation and then reused unchanged.
- Calibration is optional and must be fit on validation only.
- FuelMap-derived coordinates may be exported for case-study inspection.
- External spatial validation is optional and explicitly treated as
  experimental, not as a core released metric.

Important note
--------------
This evaluator is intentionally conservative. It does not overstate spatial
claims beyond what the current release supports.
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
from src.evaluation.calibration_metrics import (
    compute_ece,
    compute_mce,
    compute_reliability,
)
from src.evaluation.metrics import (
    brier,
    f1_precision_recall,
    pr_auc,
    roc_auc,
)
from src.training.trainer import _build_model
from src.utils.calibration import PlattScaler, fit_platt_scaler
from src.utils.config import cfg_get, load_config

logger = logging.getLogger(__name__)


def soft_argmax_2d(m: np.ndarray, temperature: float = 10.0) -> Tuple[float, float]:
    """
    Compute a soft-argmax over a 2D map.
    """
    if m.ndim != 2:
        raise ValueError(f"Expected a 2D map, got shape {m.shape}")

    h, w = m.shape
    z = m.astype(np.float64) * float(temperature)
    z = z - np.nanmax(z)
    p = np.exp(z)
    denom = float(np.sum(p))

    if not np.isfinite(denom) or denom <= 0.0:
        y, x = np.unravel_index(int(np.nanargmax(m)), m.shape)
        return float(y), float(x)

    p = p / denom
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    y = float(np.sum(yy * p))
    x = float(np.sum(xx * p))
    return y, x


def pixel_to_geo(y: float, x: float, lats: np.ndarray, lons: np.ndarray) -> Tuple[float, float]:
    """
    Convert a floating-point pixel coordinate to geographic coordinates.
    """
    if lats.shape != lons.shape:
        raise ValueError(f"Latitude/longitude shape mismatch: {lats.shape} vs {lons.shape}")

    h, w = lats.shape
    iy = int(np.clip(round(y), 0, h - 1))
    ix = int(np.clip(round(x), 0, w - 1))
    return float(lats[iy, ix]), float(lons[iy, ix])


def _select_checkpoint_path(cfg: Dict[str, Any]) -> Path:
    """
    Select the released checkpoint path from the configured checkpoint directory.
    """
    ckpt_dir = Path(cfg_get(cfg, "paths.checkpoints_dir", "./models/checkpoints")).resolve()
    best_auc = ckpt_dir / "best_auc_model.pt"
    best_loss = ckpt_dir / "best_model.pt"

    if best_auc.exists():
        return best_auc
    if best_loss.exists():
        return best_loss

    raise FileNotFoundError(
        f"No released checkpoint found in {ckpt_dir}. "
        f"Expected best_auc_model.pt or best_model.pt."
    )


def _load_threshold(cfg: Dict[str, Any]) -> float:
    """
    Load the operating threshold selected during training.
    """
    ckpt_dir = Path(cfg_get(cfg, "paths.checkpoints_dir", "./models/checkpoints")).resolve()
    threshold_path = ckpt_dir / "best_threshold.json"

    if not threshold_path.exists():
        logger.warning("best_threshold.json not found. Falling back to threshold=0.5.")
        return 0.5

    with threshold_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return float(payload.get("threshold", 0.5))


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    """
    Load a checkpoint into the model, tolerating shape changes (e.g. a different
    input-channel count after enabling/disabling the ADT input). Shape-mismatched
    tensors are skipped with a loud warning instead of crashing, so a stale
    checkpoint is reported clearly rather than producing a cryptic size-mismatch.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint["model_state"] if (isinstance(checkpoint, dict) and "model_state" in checkpoint) else checkpoint

    model_state = model.state_dict()
    compatible = {k: v for k, v in state.items()
                  if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)}
    skipped = [k for k in state if k not in compatible and k in model_state]

    if skipped:
        logger.warning(
            "Checkpoint has %d tensor(s) with mismatched shapes (skipped): %s. "
            "This usually means the input-channel count changed (e.g. ADT was enabled). "
            "RE-TRAIN the model — evaluating on this checkpoint is invalid.",
            len(skipped), skipped[:4],
        )
    model.load_state_dict(compatible, strict=False)


def _build_loader(cfg: Dict[str, Any], split: str) -> DataLoader:
    """
    Build a deterministic evaluation loader.
    """
    batch_size = int(cfg_get(cfg, "training.batch_size", 16))
    num_workers = int(cfg_get(cfg, "repro.num_workers", 0))

    ds = PhysicsDataset(cfg=cfg, split=split)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _extract_fuelmap(outputs: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Extract FuelMap logits from model outputs if available.
    """
    if "fuelmap_logits" in outputs:
        return outputs["fuelmap_logits"]
    if "fuelmap" in outputs:
        return outputs["fuelmap"]
    return None


def _fit_optional_calibrator(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
) -> Optional[PlattScaler]:
    """
    Fit Platt scaling on validation only.
    """
    val_loader = _build_loader(cfg, split="val")
    logits_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].to(device)
            prior = batch.get("prior_map_t0", None)
            if isinstance(prior, torch.Tensor):
                prior = prior.to(device)

            outputs = model(x, prior_map_t0=prior) if prior is not None else model(x)
            logits = outputs["ri_logit"].detach().cpu().numpy().reshape(-1)
            labels = batch["y"].detach().cpu().numpy().astype(int).reshape(-1)

            logits_list.append(logits)
            labels_list.append(labels)

    if not logits_list:
        logger.warning("Calibration skipped because validation predictions are empty.")
        return None

    logits_np = np.concatenate(logits_list)
    labels_np = np.concatenate(labels_list)

    if len(np.unique(labels_np)) < 2:
        logger.warning(
            "Calibration skipped because validation labels contain fewer than two classes."
        )
        return None

    return fit_platt_scaler(logits_np, labels_np)


def _run_core_inference(
    cfg: Dict[str, Any],
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    calibrator: Optional[PlattScaler],
    save_predicted_coordinates: bool,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Run core evaluation inference over one split.
    """
    interim_dir = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    fuelmap_temperature = float(cfg_get(cfg, "evaluation.fuelmap_temperature", 10.0))

    rows: list[Dict[str, Any]] = []
    all_probs: list[float] = []
    all_labels: list[int] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            event_ids = batch["event_id"]
            x = batch["x"].to(device)
            y = batch["y"].detach().cpu().numpy().astype(int).reshape(-1)

            prior = batch.get("prior_map_t0", None)
            if isinstance(prior, torch.Tensor):
                prior = prior.to(device)

            outputs = model(x, prior_map_t0=prior) if prior is not None else model(x)

            ri_logits = outputs["ri_logit"].detach().cpu().numpy().reshape(-1)
            ri_probs = 1.0 / (1.0 + np.exp(-ri_logits))

            if calibrator is not None:
                ri_probs = calibrator.predict_from_logits(ri_logits)

            fuelmap = _extract_fuelmap(outputs)

            dv12_pred = (
                outputs["dv12"].detach().cpu().numpy().reshape(-1)
                if "dv12" in outputs
                else None
            )
            dv24_pred = (
                outputs["dv24"].detach().cpu().numpy().reshape(-1)
                if "dv24" in outputs
                else None
            )

            for i, event_id in enumerate(event_ids):
                row: Dict[str, Any] = {
                    "event_id": str(event_id),
                    "y_true": int(y[i]),
                    "ri_score": float(ri_probs[i]),
                }

                if dv12_pred is not None:
                    row["dv12_pred"] = float(dv12_pred[i])
                if dv24_pred is not None:
                    row["dv24_pred"] = float(dv24_pred[i])

                if save_predicted_coordinates and fuelmap is not None:
                    lats_path = interim_dir / f"{event_id}_lats.npy"
                    lons_path = interim_dir / f"{event_id}_lons.npy"

                    if lats_path.exists() and lons_path.exists():
                        lats = np.load(lats_path)
                        lons = np.load(lons_path)
                        fmap = fuelmap[i, 0].detach().cpu().numpy()

                        py, px = soft_argmax_2d(fmap, temperature=fuelmap_temperature)
                        plat, plon = pixel_to_geo(py, px, lats, lons)

                        row["pred_y"] = float(py)
                        row["pred_x"] = float(px)
                        row["pred_lat"] = float(plat)
                        row["pred_lon"] = float(plon)

                rows.append(row)

            all_probs.extend(ri_probs.tolist())
            all_labels.extend(y.tolist())

    pred_df = pd.DataFrame(rows)
    y_true = np.asarray(all_labels, dtype=np.int64)
    y_prob = np.asarray(all_probs, dtype=np.float64)
    return pred_df, y_true, y_prob


def _compute_core_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    include_calibration_report: bool,
) -> Dict[str, Any]:
    """
    Compute the released core evaluation metrics.
    """
    f1, precision, recall = f1_precision_recall(y_prob, y_true, threshold=threshold)

    metrics: Dict[str, Any] = {
        "roc_auc": roc_auc(y_prob, y_true),
        "pr_auc": pr_auc(y_prob, y_true),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier": brier(y_prob, y_true),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "release_note": (
            "Core released metrics are classification/probabilistic metrics. "
            "External spatial validation is optional and experimental."
        ),
    }

    if include_calibration_report:
        calibration_data = compute_reliability(y_true, y_prob, n_bins=10)
        metrics["ece"] = compute_ece(y_true, y_prob, n_bins=10)
        metrics["mce"] = compute_mce(y_true, y_prob, n_bins=10)
        metrics["calibration_data"] = calibration_data

    return metrics


def _try_external_spatial_validation(
    cfg: Dict[str, Any],
    pred_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Validate the predicted FuelMap hotspot against the TCHP peak.

    The TCHP peaks are produced once and audited by `run.py preprocess-tchp`, then
    stored in each event's interim metadata. This validation reuses that audited
    output (single source of truth) rather than re-opening TCHP NetCDFs here.
    """
    out: Dict[str, Any] = {
        "note": (
            "Spatial validation compares the predicted FuelMap peak to the TCHP peak "
            "computed and audited during 'run.py preprocess-tchp'."
        ),
    }

    if pred_df.empty or "pred_lat" not in pred_df.columns or "pred_lon" not in pred_df.columns:
        out["status"] = "skipped"
        out["reason"] = "predicted_coordinates_unavailable"
        return out

    try:
        from src.evaluation.spatial_metrics import compute_spatial_metrics_from_predictions
    except Exception as exc:
        out["status"] = "skipped"
        out["reason"] = f"spatial_metrics_import_failed: {exc}"
        return out

    try:
        interim_dir = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
        if not interim_dir.exists():
            out["status"] = "skipped"
            out["reason"] = "interim_metadata_directory_missing"
            return out

        results = compute_spatial_metrics_from_predictions(
            pred_df=pred_df,
            interim_dir=interim_dir,
        )
        out["status"] = results.get("status", "ok")
        out["metrics"] = results
        return out

    except Exception as exc:
        out["status"] = "failed"
        out["reason"] = str(exc)
        return out


def run_evaluate(
    cfg: Dict[str, Any],
    split: str = "test",
    calibrate: bool = False,
    calibration_report: bool = True,
    save_predicted_coordinates: bool = True,
    experimental_external_spatial: bool = False,
) -> None:
    """
    Run evaluation for one split.
    """
    device_cfg = str(cfg_get(cfg, "training.device", "auto")).lower()
    if device_cfg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device='cuda' but CUDA is not available.")
    device = torch.device("cuda" if (device_cfg == "cuda" or (device_cfg == "auto" and torch.cuda.is_available())) else "cpu")

    results_dir = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = _select_checkpoint_path(cfg)
    threshold = _load_threshold(cfg)

    model = _build_model(cfg).to(device)
    _load_checkpoint(model, ckpt_path, device)
    logger.info("Loaded checkpoint from %s", ckpt_path)

    calibrator = _fit_optional_calibrator(cfg, model, device) if calibrate else None
    loader = _build_loader(cfg, split=split)

    pred_df, y_true, y_prob = _run_core_inference(
        cfg=cfg,
        model=model,
        loader=loader,
        device=device,
        calibrator=calibrator,
        save_predicted_coordinates=save_predicted_coordinates,
    )

    metrics = _compute_core_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        include_calibration_report=calibration_report,
    )

    pred_df["ri_pred"] = (pred_df["ri_score"].to_numpy() >= threshold).astype(int)

    if experimental_external_spatial:
        metrics["external_spatial_validation"] = _try_external_spatial_validation(
            cfg=cfg,
            pred_df=pred_df,
        )

    pred_path = results_dir / f"{split}_predictions.csv"
    metrics_path = results_dir / f"{split}_metrics.json"

    pred_df.to_csv(pred_path, index=False)

    calibration_payload = metrics.pop("calibration_data", None)
    if calibration_payload is not None:
        cal_path = results_dir / f"{split}_calibration_data.json"
        with cal_path.open("w", encoding="utf-8") as f:
            json.dump(calibration_payload, f, indent=2)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved predictions to %s", pred_path)
    logger.info("Saved metrics to %s", metrics_path)


def main(cfg: Optional[Dict[str, Any]] = None) -> None:
    """
    CLI entrypoint.
    """
    import argparse

    parser = argparse.ArgumentParser(description="CycloneNet evaluation")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--no-calibration-report", action="store_true")
    parser.add_argument("--no-predicted-coordinates", action="store_true")
    parser.add_argument("--experimental-external-spatial", action="store_true")
    args = parser.parse_args()

    if cfg is None:
        cfg = load_config("config.yaml")

    run_evaluate(
        cfg=cfg,
        split=args.split,
        calibrate=args.calibrate,
        calibration_report=not args.no_calibration_report,
        save_predicted_coordinates=not args.no_predicted_coordinates,
        experimental_external_spatial=args.experimental_external_spatial,
    )


if __name__ == "__main__":
    main()