from __future__ import annotations

"""CycloneNet — evaluation (leakage-safe) with optional external validation hooks.

- Computes standard classification metrics.
- Extracts predicted fuel-source location from FuelMap if the model provides it.
- If metadata provides external energy maximum (e.g., tchp_max_lat/lon), reports distance.

External products MUST NOT be part of model inputs. They are validation-only.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import roc_auc, pr_auc, brier, f1_precision_recall, select_threshold_for_recall

logger = logging.getLogger(__name__)


def soft_argmax_2d(m: np.ndarray, temperature: float = 10.0) -> Tuple[float, float]:
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
    yi = int(np.clip(round(y), 0, lats.shape[0] - 1))
    xi = int(np.clip(round(x), 0, lats.shape[1] - 1))
    return float(lats[yi, xi]), float(lons[yi, xi])


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1 = np.deg2rad(lat1)
    p2 = np.deg2rad(lat2)
    dphi = p2 - p1
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def _get_fuelmap(out: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    if "fuelmap" in out:
        return out["fuelmap"]
    if "fuelmap_logits" in out:
        return out["fuelmap_logits"]
    return None


def evaluate(cfg: Dict[str, Any], loader: DataLoader, model: torch.nn.Module, interim_dir: Path, out_csv: Path, out_json: Path) -> None:
    device = next(model.parameters()).device
    temperature = float(cfg.get("evaluation", {}).get("fuelmap_temperature", 10.0))
    target_recall = float(cfg.get("training", {}).get("eval_target_recall", 0.90))

    rows = []
    scores_all = []
    labels_all = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            eids = batch["event_id"]
            x = batch["x"].to(device)
            y = batch["y"].cpu().numpy().astype(int)

            out = model(x, prior_map_t0=batch.get("prior_map_t0", None).to(device) if "prior_map_t0" in batch else None)
            scores = torch.sigmoid(out["ri_logit"]).cpu().numpy()
            fm_t = _get_fuelmap(out)

            for i, eid in enumerate(eids):
                row = {"event_id": str(eid), "y_true": int(y[i]), "y_score": float(scores[i])}

                if fm_t is not None:
                    lats = np.load(Path(interim_dir) / f"{eid}_lats.npy")
                    lons = np.load(Path(interim_dir) / f"{eid}_lons.npy")
                    m = fm_t[i, 0].cpu().numpy()
                    py, px = soft_argmax_2d(m, temperature=temperature)
                    plat, plon = pixel_to_geo(py, px, lats, lons)
                    row["pred_lat"] = plat
                    row["pred_lon"] = plon

                    meta_path = Path(interim_dir) / f"{eid}.json"
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        if "tchp_max_lat" in meta and "tchp_max_lon" in meta:
                            tl = float(meta["tchp_max_lat"])
                            tn = float(meta["tchp_max_lon"])
                            row["tchp_max_lat"] = tl
                            row["tchp_max_lon"] = tn
                            row["tchp_dist_km"] = haversine_km(plat, plon, tl, tn)

                rows.append(row)

            scores_all.append(scores)
            labels_all.append(y)

    scores_all = np.concatenate(scores_all, axis=0).astype(float)
    labels_all = np.concatenate(labels_all, axis=0).astype(int)

    thr = select_threshold_for_recall(scores_all, labels_all, target_recall=target_recall)
    f1, p, r = f1_precision_recall(scores_all, labels_all, threshold=thr)

    summary = {
        "roc_auc": roc_auc(scores_all, labels_all),
        "pr_auc": pr_auc(scores_all, labels_all),
        "brier": brier(scores_all, labels_all),
        "threshold": float(thr),
        "f1": float(f1),
        "precision": float(p),
        "recall": float(r),
        "n": int(labels_all.size),
        "positives": int((labels_all == 1).sum()),
        "negatives": int((labels_all == 0).sum()),
        "note": "External products (e.g., TCHP) are validation-only and never used as inputs.",
    }

    tchp_dists = [row.get("tchp_dist_km") for row in rows if "tchp_dist_km" in row]
    if tchp_dists:
        tchp_dists = np.array(tchp_dists, dtype=float)
        summary["tchp_mean_dist_km"] = float(np.mean(tchp_dists))
        summary["tchp_median_dist_km"] = float(np.median(tchp_dists))
        summary["tchp_n"] = int(tchp_dists.size)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(f"Saved: {out_csv}")
    logger.info(f"Saved: {out_json}")

# -----------------------------------------------------------------------------
# Public entrypoints expected by run.py (config-driven)
# -----------------------------------------------------------------------------
def run_evaluate(cfg: Dict[str, Any]) -> None:
    """
    Config-driven evaluation entrypoint.

    Builds the test DataLoader + model, loads the best checkpoint (if present),
    then calls the internal:
        evaluate(cfg, loader, model, interim_dir, out_csv, out_json)

    Scientific note:
    - External products (e.g., TCHP/OHC) are validation-only and MUST NOT be used as inputs.
    """
    from src.utils.config import cfg_get
    from src.data.dataset import PhysicsDataset
    from src.training.trainer import _build_model  # robust, signature-safe model builder

    dev = str(cfg_get(cfg, "training.device", "auto")).lower()
    device = torch.device("cuda" if torch.cuda.is_available() and dev in ("auto", "cuda") else "cpu")

    batch_size = int(cfg_get(cfg, "training.batch_size", 16))
    num_workers = int(cfg_get(cfg, "repro.num_workers", 4))

    ds_test = PhysicsDataset(cfg, split="test")
    loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = _build_model(cfg).to(device)
    model.eval()

    ckpt_dir = Path(cfg_get(cfg, "paths.checkpoints_dir", "./models/checkpoints")).resolve()
    best_path = ckpt_dir / "best_model.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state, strict=False)

    interim_dir = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    results_dir = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    out_csv = results_dir / "test_predictions.csv"
    out_json = results_dir / "test_metrics.json"

    evaluate(cfg, loader, model, interim_dir, out_csv, out_json)


def main(cfg: Optional[Dict[str, Any]] = None) -> None:
    """
    CLI-compatible entrypoint.
    Supports both:
      - python -m src.evaluation.evaluate
      - run.py calling main(cfg)
    """
    from src.utils.config import load_config
    if cfg is None:
        cfg = load_config("config.yaml")
    run_evaluate(cfg)


if __name__ == "__main__":
    main()
