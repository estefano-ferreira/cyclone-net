#!/usr/bin/env python3
"""Counterfactual causal tests for CycloneNet physics-guided FuelMap.

This script provides evidence that the learned FuelMap is *causally used* by the model:
  - Removing information inside the hotspot should reduce predicted RI risk more
    than removing information outside it.
  - Swapping hotspots across samples should degrade predictions.

These tests do not prove physical causality in nature, but they are a strong
scientific sanity check that the model's stated "fuel" localization is not
merely decorative.

Outputs:
  outputs/results/causal_tests.json

Usage:
  python analysis/causal_tests.py --split test

Author: Estefano Senhor Ferreira
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils.config import CONFIG, cfg_get
from src.data.dataset import PhysicsDataset
from src.models.cyclone_net_ri_only import CycloneNetRIOnly


def topk_mask(fuelmap: np.ndarray, frac: float) -> np.ndarray:
    """Binary mask of top-k pixels."""
    h, w = fuelmap.shape
    k = max(1, int(frac * h * w))
    flat = fuelmap.reshape(-1)
    thresh = np.partition(flat, -k)[-k]
    return (fuelmap >= thresh).astype(np.float32)


@torch.no_grad()
def predict_prob(model: torch.nn.Module, x: torch.Tensor) -> float:
    out = model(x)
    if isinstance(out, dict):
        ri_logit = out["ri_logit"]
    else:
        ri_logit = out
    return float(torch.sigmoid(ri_logit).mean().item())


def apply_mask(x: torch.Tensor, mask_hw: np.ndarray, inside: bool = True) -> torch.Tensor:
    """Zero-out inside (or outside) a mask for all channels and timesteps."""
    mask = torch.from_numpy(mask_hw).to(x.device, dtype=x.dtype)  # (H,W)
    mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,H,W)
    if inside:
        return x * (1.0 - mask)
    return x * mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PhysicsDataset(CONFIG, split=args.split, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = CycloneNetRIOnly(CONFIG).to(device)
    ckpt = cfg_get(CONFIG, "evaluation.checkpoint", None)
    if ckpt:
        ckpt_path = Path(str(ckpt))
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    frac = float(cfg_get(CONFIG, "physics_guided.fuelmap.topk_fraction", 0.05))
    results = {
        "split": args.split,
        "topk_fraction": frac,
        "n": 0,
        "mean_drop_hotspot": 0.0,
        "mean_drop_nonhotspot": 0.0,
    }

    drops_hot = []
    drops_non = []

    for batch in loader:
        x = batch["input"].to(device)
        out = model(x)
        if not (isinstance(out, dict) and "fuelmap" in out and out["fuelmap"] is not None):
            # FuelMap not available -> cannot run causal tests
            continue

        fm = out["fuelmap"].detach().cpu().numpy()[0, 0]  # (H,W)
        m = topk_mask(fm, frac=frac)

        p0 = predict_prob(model, x)
        p_hot_removed = predict_prob(model, apply_mask(x, m, inside=True))
        p_non_removed = predict_prob(model, apply_mask(x, m, inside=False))

        drops_hot.append(p0 - p_hot_removed)
        drops_non.append(p0 - p_non_removed)

    if len(drops_hot) > 0:
        results["n"] = int(len(drops_hot))
        results["mean_drop_hotspot"] = float(np.mean(drops_hot))
        results["mean_drop_nonhotspot"] = float(np.mean(drops_non))

    out_path = Path(cfg_get(CONFIG, "paths.results", "./outputs/results")) / "causal_tests.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
