"""
CycloneNet: Diagnostic variable computations from base ERA5 fields.
Now includes heat flux calculations.
All comments in English.
"""

from __future__ import annotations
import numpy as np

# Import the new heat flux module
from src.physics.heat_flux import compute_heat_fluxes


def _finite_diff_x(a: np.ndarray, dx: float) -> np.ndarray:
    # Central difference with edge replication.
    out = np.empty_like(a, dtype=np.float32)
    out[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / (2.0 * dx)
    out[:, 0] = (a[:, 1] - a[:, 0]) / dx
    out[:, -1] = (a[:, -1] - a[:, -2]) / dx
    return out


def _finite_diff_y(a: np.ndarray, dy: float) -> np.ndarray:
    out = np.empty_like(a, dtype=np.float32)
    out[1:-1, :] = (a[2:, :] - a[:-2, :]) / (2.0 * dy)
    out[0, :] = (a[1, :] - a[0, :]) / dy
    out[-1, :] = (a[-1, :] - a[-2, :]) / dy
    return out


def estimate_dx_dy_meters(lats: np.ndarray, lons: np.ndarray) -> tuple[float, float]:
    """Estimate grid spacing (dx, dy) in meters from lat/lon arrays."""
    lat0 = float(np.nanmean(lats))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
    dlat = np.nanmedian(np.abs(lats[1:, :] - lats[:-1, :]))
    dlon = np.nanmedian(np.abs(lons[:, 1:] - lons[:, :-1]))
    dy = max(1e-6, float(dlat) * m_per_deg_lat)
    dx = max(1e-6, float(dlon) * m_per_deg_lon)
    return dx, dy


def wind_speed(u10: np.ndarray, v10: np.ndarray) -> np.ndarray:
    return np.sqrt(u10.astype(np.float32)**2 + v10.astype(np.float32)**2).astype(np.float32)


def vorticity(u10: np.ndarray, v10: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dv_dx = _finite_diff_x(v10.astype(np.float32), dx)
    du_dy = _finite_diff_y(u10.astype(np.float32), dy)
    return (dv_dx - du_dy).astype(np.float32)


def divergence(u10: np.ndarray, v10: np.ndarray, dx: float, dy: float) -> np.ndarray:
    du_dx = _finite_diff_x(u10.astype(np.float32), dx)
    dv_dy = _finite_diff_y(v10.astype(np.float32), dy)
    return (du_dx + dv_dy).astype(np.float32)


def grad_mslp_mag(mslp: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dp_dx = _finite_diff_x(mslp.astype(np.float32), dx)
    dp_dy = _finite_diff_y(mslp.astype(np.float32), dy)
    return np.sqrt(dp_dx**2 + dp_dy**2).astype(np.float32)


def sst_anomaly(sst_k: np.ndarray) -> np.ndarray:
    s = sst_k.astype(np.float32)
    mu = float(np.nanmean(s))
    return (s - mu).astype(np.float32)


def compute_diagnostics(
    base: dict[str, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    enabled_channels: list[str],
    # New optional arguments for heat fluxes
    t2m: np.ndarray = None,
    d2m: np.ndarray = None,
    heat_flux_params: dict = None
) -> list[np.ndarray]:
    """
    Compute selected diagnostic channels in a stable, reproducible order.
    Now includes heat flux channels if requested.
    """
    dx, dy = estimate_dx_dy_meters(lats, lons)
    out: list[np.ndarray] = []

    if heat_flux_params is None:
        heat_flux_params = {}

    # Heat fluxes (latent, sensible, total)
    if any(ch in enabled_channels for ch in ['latent_heat_flux', 'sensible_heat_flux', 'total_heat_flux']):
        heat_fluxes = compute_heat_fluxes(
            sst=base['sst'],
            u10=base['u10'],
            v10=base['v10'],
            msl=base['msl'],
            t2m=t2m,
            d2m=d2m,
            Ce=heat_flux_params.get('Ce', 1.2e-3),
            Ch=heat_flux_params.get('Ch', 1.2e-3)
        )
        if 'latent_heat_flux' in enabled_channels:
            out.append(heat_fluxes['latent_heat_flux'])
        if 'sensible_heat_flux' in enabled_channels:
            out.append(heat_fluxes['sensible_heat_flux'])
        if 'total_heat_flux' in enabled_channels:
            out.append(heat_fluxes['total_heat_flux'])

    # Original diagnostics
    if "wind_speed" in enabled_channels:
        out.append(wind_speed(base["u10"], base["v10"]))
    if "vorticity" in enabled_channels:
        out.append(vorticity(base["u10"], base["v10"], dx, dy))
    if "divergence" in enabled_channels:
        out.append(divergence(base["u10"], base["v10"], dx, dy))
    if "grad_mslp" in enabled_channels:
        out.append(grad_mslp_mag(base["msl"], dx, dy))
    if "sst_anom" in enabled_channels:
        out.append(sst_anomaly(base["sst"]))

    return out
