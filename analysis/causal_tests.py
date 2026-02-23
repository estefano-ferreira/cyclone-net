from __future__ import annotations

"""CycloneNet — contrafactual (ablation) tests for physics-guided claims.

Goal:
- Provide *causal-style* evidence that the model's predicted intensification depends on the
  localized "fuel" region identified by FuelMap / physical prior.

Method (simple and reproducible):
1) Run inference to get dv24_hat (or pRI) and FuelMap logits.
2) Build a mask from the top-k% pixels of the FuelMap.
3) Apply ablation on selected physical channels within that mask (e.g., reduce SST anomaly, reduce wind).
4) Re-run inference and measure delta.

This is not a full causal inference framework, but it is a strong, reviewer-friendly diagnostic.

Usage:
  python analysis/causal_tests.py --interim data/interim --splits data/splits.csv --norm data/processed/norm_profile9.json --profile 9 --k 0.05

Requires:
- A trained model checkpoint and a small loader wrapper in your project.
This script is provided as a template (project-specific checkpoint loading may vary).
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch

def topk_mask(fuelmap: np.ndarray, k: float) -> np.ndarray:
    # fuelmap: (H,W) logits or scores
    flat = fuelmap.reshape(-1)
    thr = np.quantile(flat, 1.0 - k)
    return (fuelmap >= thr).astype(np.float32)

def ablate(x: torch.Tensor, mask: torch.Tensor, channel_indices: list[int], factor: float) -> torch.Tensor:
    # x: (B,C,T,H,W), mask: (B,1,H,W)
    x2 = x.clone()
    for ci in channel_indices:
        # apply to all timesteps for that channel
        x2[:, ci, :, :, :] = x2[:, ci, :, :, :] * (1.0 - factor * mask)
    return x2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=0.05, help="Top-k fraction for ablation mask")
    ap.add_argument("--factor", type=float, default=0.5, help="Ablation strength (0..1)")
    ap.add_argument("--channels", nargs="+", default=["sst_K", "u10_mps", "v10_mps"], help="Channel names to ablate")
    ap.add_argument("--note", default="Template script; integrate with your checkpoint loading.")
    args = ap.parse_args()

    print("This is a template. Integrate with your checkpoint/model/dataloader to run.")
    print("Suggested approach: use FuelMap logits -> mask -> ablate physical channels -> delta dv24/pRI.")
    print("k=", args.k, "factor=", args.factor, "channels=", args.channels)

if __name__ == "__main__":
    main()
