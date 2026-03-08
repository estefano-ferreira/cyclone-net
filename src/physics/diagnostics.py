from __future__ import annotations

"""
CycloneNet — diagnostic variable computations from base ERA5 fields.

This module computes physically motivated diagnostic channels from the base
reanalysis variables used by the CycloneNet preprocessing pipeline.

Scientific intent
-----------------
The core diagnostic channels are:
- wind speed
- vertical vorticity
- horizontal divergence
- mean-sea-level-pressure gradient magnitude
- sea-surface-temperature anomaly

Optional surface heat-flux channels are also supported:
- latent heat flux
- sensible heat flux
- total heat flux

Critical invariant
------------------
The output order of `compute_diagnostics()` MUST exactly match the order of
`enabled_channels`. This is essential for scientific auditability because the
preprocessing pipeline stacks arrays and records channel names separately.
If the order is not identical, tensor channels and metadata become inconsistent.
"""

import math
from typing import Dict, List

import numpy as np

from src.physics.heat_flux import compute_heat_fluxes


SUPPORTED_CHANNELS = {
    "wind_speed",
    "vorticity",
    "divergence",
    "grad_mslp",
    "sst_anom",
    "latent_heat_flux",
    "sensible_heat_flux",
    "total_heat_flux",
}


def _as_float32_2d(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected '{name}' to be 2D, got shape {arr.shape}.")
    return arr


def _finite_diff_x(a: np.ndarray, dx: float) -> np.ndarray:
    if dx <= 0.0 or not np.isfinite(dx):
        raise ValueError(f"dx must be positive and finite, got {dx}.")
    a = _as_float32_2d(a, "a")
    out = np.empty_like(a, dtype=np.float32)
    out[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / (2.0 * dx)
    out[:, 0] = (a[:, 1] - a[:, 0]) / dx
    out[:, -1] = (a[:, -1] - a[:, -2]) / dx
    return out


def _finite_diff_y(a: np.ndarray, dy: float) -> np.ndarray:
    if dy <= 0.0 or not np.isfinite(dy):
        raise ValueError(f"dy must be positive and finite, got {dy}.")
    a = _as_float32_2d(a, "a")
    out = np.empty_like(a, dtype=np.float32)
    out[1:-1, :] = (a[2:, :] - a[:-2, :]) / (2.0 * dy)
    out[0, :] = (a[1, :] - a[0, :]) / dy
    out[-1, :] = (a[-1, :] - a[-2, :]) / dy
    return out


def estimate_dx_dy_meters(lats: np.ndarray, lons: np.ndarray) -> tuple[float, float]:
    lats = _as_float32_2d(lats, "lats")
    lons = _as_float32_2d(lons, "lons")
    if lats.shape != lons.shape:
        raise ValueError(f"Latitude/longitude shape mismatch: {lats.shape} vs {lons.shape}.")

    lat0 = float(np.nanmean(lats))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))

    dlat = float(np.nanmedian(np.abs(lats[1:, :] - lats[:-1, :])))
    dlon = float(np.nanmedian(np.abs(lons[:, 1:] - lons[:, :-1])))

    dy = max(1e-6, dlat * m_per_deg_lat)
    dx = max(1e-6, dlon * m_per_deg_lon)
    return float(dx), float(dy)


def wind_speed(u10: np.ndarray, v10: np.ndarray) -> np.ndarray:
    u10 = _as_float32_2d(u10, "u10")
    v10 = _as_float32_2d(v10, "v10")
    return np.sqrt(u10**2 + v10**2).astype(np.float32)


def vorticity(u10: np.ndarray, v10: np.ndarray, dx: float, dy: float) -> np.ndarray:
    u10 = _as_float32_2d(u10, "u10")
    v10 = _as_float32_2d(v10, "v10")
    return (_finite_diff_x(v10, dx) - _finite_diff_y(u10, dy)).astype(np.float32)


def divergence(u10: np.ndarray, v10: np.ndarray, dx: float, dy: float) -> np.ndarray:
    u10 = _as_float32_2d(u10, "u10")
    v10 = _as_float32_2d(v10, "v10")
    return (_finite_diff_x(u10, dx) + _finite_diff_y(v10, dy)).astype(np.float32)


def grad_mslp_mag(mslp: np.ndarray, dx: float, dy: float) -> np.ndarray:
    mslp = _as_float32_2d(mslp, "mslp")
    dp_dx = _finite_diff_x(mslp, dx)
    dp_dy = _finite_diff_y(mslp, dy)
    return np.sqrt(dp_dx**2 + dp_dy**2).astype(np.float32)


def sst_anomaly(sst_k: np.ndarray) -> np.ndarray:
    sst_k = _as_float32_2d(sst_k, "sst_k")
    return (sst_k - float(np.nanmean(sst_k))).astype(np.float32)


def _validate_enabled_channels(enabled_channels: List[str]) -> None:
    unknown = [ch for ch in enabled_channels if ch not in SUPPORTED_CHANNELS]
    if unknown:
        raise ValueError(
            f"Unknown diagnostic channels: {unknown}. Supported channels: {sorted(SUPPORTED_CHANNELS)}"
        )


def _validate_base_fields(base: Dict[str, np.ndarray]) -> None:
    required = ["sst", "msl", "u10", "v10"]
    missing = [k for k in required if k not in base]
    if missing:
        raise ValueError(f"Missing base fields: {missing}")

    sst = _as_float32_2d(base["sst"], "base['sst']")
    shape = sst.shape
    for key in ["msl", "u10", "v10"]:
        arr = _as_float32_2d(base[key], f"base['{key}']")
        if arr.shape != shape:
            raise ValueError(f"Base field shape mismatch: sst={shape}, {key}={arr.shape}")


def compute_diagnostics(
    base: Dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    enabled_channels: List[str],
    t2m: np.ndarray | None = None,
    d2m: np.ndarray | None = None,
    heat_flux_params: Dict[str, float] | None = None,
) -> List[np.ndarray]:
    """Compute diagnostic channels in exactly the requested order."""
    _validate_enabled_channels(enabled_channels)
    _validate_base_fields(base)

    sst = _as_float32_2d(base["sst"], "base['sst']")
    msl = _as_float32_2d(base["msl"], "base['msl']")
    u10 = _as_float32_2d(base["u10"], "base['u10']")
    v10 = _as_float32_2d(base["v10"], "base['v10']")

    if t2m is not None:
        t2m = _as_float32_2d(t2m, "t2m")
    if d2m is not None:
        d2m = _as_float32_2d(d2m, "d2m")

    dx, dy = estimate_dx_dy_meters(lats, lons)
    heat_flux_params = heat_flux_params or {}

    needs_heat_flux = any(
        name in enabled_channels
        for name in ["latent_heat_flux", "sensible_heat_flux", "total_heat_flux"]
    )

    heat_fluxes: Dict[str, np.ndarray] | None = None
    if needs_heat_flux:
        heat_fluxes = compute_heat_fluxes(
            sst=sst,
            u10=u10,
            v10=v10,
            msl=msl,
            t2m=t2m,
            d2m=d2m,
            Ce=float(heat_flux_params.get("Ce", 1.2e-3)),
            Ch=float(heat_flux_params.get("Ch", 1.2e-3)),
        )

    cache: Dict[str, np.ndarray] = {}

    def get_channel(name: str) -> np.ndarray:
        if name in cache:
            return cache[name]
        if name == "wind_speed":
            cache[name] = wind_speed(u10, v10)
        elif name == "vorticity":
            cache[name] = vorticity(u10, v10, dx, dy)
        elif name == "divergence":
            cache[name] = divergence(u10, v10, dx, dy)
        elif name == "grad_mslp":
            cache[name] = grad_mslp_mag(msl, dx, dy)
        elif name == "sst_anom":
            cache[name] = sst_anomaly(sst)
        elif name in {"latent_heat_flux", "sensible_heat_flux", "total_heat_flux"}:
            if heat_fluxes is None:
                raise RuntimeError(f"Heat-flux channel '{name}' requested but not available.")
            cache[name] = np.asarray(heat_fluxes[name], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported channel '{name}'.")
        return cache[name]

    out: List[np.ndarray] = []
    for channel_name in enabled_channels:
        arr = _as_float32_2d(get_channel(channel_name), channel_name)
        if arr.shape != sst.shape:
            raise ValueError(
                f"Diagnostic channel '{channel_name}' shape mismatch: expected {sst.shape}, got {arr.shape}."
            )
        out.append(arr)
    return out