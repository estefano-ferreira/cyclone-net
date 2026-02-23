from __future__ import annotations

"""CycloneNet — supervised physical fuel potential maps P(x,y,t).

This module constructs deterministic, physically motivated proxy maps that can be used to
*supervise* the model's FuelMap output. This is a core requirement for a defensible
"physics-guided" claim.

Default proxy (no extra ERA5 variables required):
  P ∝ relu(SST_anomaly) * wind_speed * (1 + w_conv * relu(-divergence))

Notes:
- This is a transparent proxy, not a perfect physical truth.
- Upgrade path: replace P with bulk surface heat flux estimates when t2m/d2m (or humidity) is available.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FuelPotentialConfig:
    eps: float = 1e-8
    w_conv: float = 1.0
    w_gradp: float = 0.0


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0).astype(np.float32)


def normalize_per_timestep(P: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize each timestep to [0,1] for stable supervision."""
    H, W, T = P.shape
    out = np.zeros_like(P, dtype=np.float32)
    for t in range(T):
        x = P[:, :, t].astype(np.float32)
        xmin = float(np.nanmin(x))
        xmax = float(np.nanmax(x))
        if not np.isfinite(xmin) or not np.isfinite(xmax) or (xmax - xmin) < eps:
            out[:, :, t] = 0.0
        else:
            out[:, :, t] = (x - xmin) / (xmax - xmin + eps)
    return out


def build_fuel_potential(
    sst_anom_K: np.ndarray,
    wind_mps: np.ndarray,
    divergence_1ps: np.ndarray,
    grad_mslp_Pa_per_m: np.ndarray | None = None,
    cfg: FuelPotentialConfig | None = None,
) -> np.ndarray:
    cfg = cfg or FuelPotentialConfig()
    if sst_anom_K.shape != wind_mps.shape or sst_anom_K.shape != divergence_1ps.shape:
        raise ValueError("Inputs must have identical shape (H,W,T).")

    P = relu(sst_anom_K) * wind_mps.astype(np.float32) * (1.0 + cfg.w_conv * relu(-divergence_1ps))

    if cfg.w_gradp > 0.0 and grad_mslp_Pa_per_m is not None:
        if grad_mslp_Pa_per_m.shape != P.shape:
            raise ValueError("grad_mslp must have shape (H,W,T).")
        gp = normalize_per_timestep(grad_mslp_Pa_per_m.astype(np.float32), eps=cfg.eps)
        P = P / (1.0 + cfg.w_gradp * gp)

    return normalize_per_timestep(P, eps=cfg.eps)
