# src/processors/pressure_channels.py
"""
Pressure-level derived channels: vertical wind shear and mid-level humidity.

Adds the two SHIPS-RII-style environmental predictors that the surface-only
cube lacks:

  * ``shear_850_200_mps`` — magnitude of the vector wind difference between
    850 hPa and 200 hPa: sqrt((u850-u200)^2 + (v850-v200)^2). Analogous to
    the SHIPS deep-layer shear predictor (SHRD).
  * ``rh_mid`` — relative humidity averaged over the 700/600/500 hPa layer,
    in percent. Analogous to the SHIPS mid-level moisture predictor (RHMD).

Source files are monthly ERA5 *pressure-levels* NetCDFs downloaded by
``src.downloaders.era5_pressure`` (dataset ``reanalysis-era5-pressure-levels``):

  * ``era5pl_wind_YYYY_MM.nc`` — u/v_component_of_wind at [850, 200] hPa
  * ``era5pl_rh_YYYY_MM.nc``   — relative_humidity at [700, 600, 500] hPa

Extraction is ALL-OR-NOTHING per event: if any required timestep or level is
unavailable, no pressure channels are added and the event keeps its original
surface-only cube — existing artifacts and downstream consumers are never
broken. New channels are appended at the END of the channel list.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from src.utils.config import cfg_get

logger = logging.getLogger(__name__)

SHEAR_CHANNEL = "shear_850_200_mps"
RH_CHANNEL = "rh_mid"

PL_UNITS = {SHEAR_CHANNEL: "m s-1", RH_CHANNEL: "%"}

_LEVEL_DIM_CANDIDATES = ("pressure_level", "level", "isobaricInhPa", "plev")

def _open_cached(path: Path) -> xr.Dataset:
    """Open a monthly file through the shared fully-loaded cache (never close)."""
    from src.processors.preprocess_scientific import open_month_cached

    return open_month_cached(path)


def month_file_wind(raw_dir: Path, dt: datetime) -> Path:
    return raw_dir / f"era5pl_wind_{dt.strftime('%Y_%m')}.nc"


def month_file_rh(raw_dir: Path, dt: datetime) -> Path:
    return raw_dir / f"era5pl_rh_{dt.strftime('%Y_%m')}.nc"


def _find_level_dim(ds: xr.Dataset) -> Optional[str]:
    for name in _LEVEL_DIM_CANDIDATES:
        if name in ds.dims or name in ds.coords:
            return name
    return None


def _resolve(ds: xr.Dataset, candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None


def _window_2d(field: np.ndarray, ds: xr.Dataset, lat0: float, lon0: float,
               window_size_px: int) -> Optional[np.ndarray]:
    """Slice the event-centered window using the dataset's own coordinates.

    Reuses the exact index/convention logic of the surface preprocess so the
    pressure-level window is co-registered with the surface window (both are
    0.25-degree grids over the same download area).
    """
    from src.processors.preprocess_scientific import (
        ensure_2d, extract_window_by_index, nearest_index,
    )

    lat_name = "latitude" if "latitude" in ds.coords else "lat" if "lat" in ds.coords else None
    lon_name = "longitude" if "longitude" in ds.coords else "lon" if "lon" in ds.coords else None
    if lat_name is None or lon_name is None:
        return None
    lats_1d = ds[lat_name].values
    lons_1d = ds[lon_name].values
    i = nearest_index(lats_1d, lat0)
    lon_value = (lon0 % 360.0
                 if float(np.nanmin(lons_1d)) >= 0.0 and float(np.nanmax(lons_1d)) > 180.0
                 else ((lon0 + 180.0) % 360.0) - 180.0)
    j = nearest_index(lons_1d, lon_value)
    return extract_window_by_index(ensure_2d(field), i, j, window_size_px).astype(np.float32)


def _select_level(da: xr.DataArray, level_dim: str, level_hpa: float) -> Optional[np.ndarray]:
    try:
        sel = da.sel({level_dim: level_hpa}, method="nearest")
        used = float(sel[level_dim].values)
        if abs(used - level_hpa) > 1.0:  # nearest level must actually be the requested one
            return None
        return np.asarray(sel.values)
    except Exception:
        return None


def _extract_timestep(raw_dir: Path, dt: datetime, lat0: float, lon0: float,
                      window_size_px: int, wind_levels: Tuple[float, float],
                      rh_levels: Sequence[float]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (shear_2d, rh_mid_2d) for one timestep, or None if unavailable."""
    from src.processors.preprocess_scientific import select_time_slice

    wind_path = month_file_wind(raw_dir, dt)
    rh_path = month_file_rh(raw_dir, dt)
    if not (wind_path.exists() and rh_path.exists()):
        return None

    lo_hpa, hi_hpa = wind_levels  # e.g. (850, 200)

    # Datasets come from the monthly cache and must NOT be closed here.
    ds_w = _open_cached(wind_path)
    ds_wt, _, _, selected = select_time_slice(ds_w, dt)
    if not selected:
        return None
    level_dim = _find_level_dim(ds_wt)
    u_name = _resolve(ds_wt, ["u", "u_component_of_wind"])
    v_name = _resolve(ds_wt, ["v", "v_component_of_wind"])
    if level_dim is None or u_name is None or v_name is None:
        return None
    u_lo = _select_level(ds_wt[u_name], level_dim, lo_hpa)
    u_hi = _select_level(ds_wt[u_name], level_dim, hi_hpa)
    v_lo = _select_level(ds_wt[v_name], level_dim, lo_hpa)
    v_hi = _select_level(ds_wt[v_name], level_dim, hi_hpa)
    if any(x is None for x in (u_lo, u_hi, v_lo, v_hi)):
        return None
    shear_full = np.hypot(u_lo - u_hi, v_lo - v_hi)
    shear = _window_2d(shear_full, ds_wt, lat0, lon0, window_size_px)

    ds_r = _open_cached(rh_path)
    ds_rt, _, _, selected = select_time_slice(ds_r, dt)
    if not selected:
        return None
    level_dim = _find_level_dim(ds_rt)
    r_name = _resolve(ds_rt, ["r", "relative_humidity"])
    if level_dim is None or r_name is None:
        return None
    slabs = []
    for lvl in rh_levels:
        slab = _select_level(ds_rt[r_name], level_dim, float(lvl))
        if slab is None:
            return None
        slabs.append(np.asarray(slab, dtype=np.float64))
    rh_full = np.mean(np.stack(slabs, axis=0), axis=0)
    rh_mid = _window_2d(rh_full, ds_rt, lat0, lon0, window_size_px)

    if shear is None or rh_mid is None:
        return None
    return shear, rh_mid


def extract_pressure_volume(
    raw_dir: Path,
    dt0: datetime,
    offsets_hours: Sequence[int],
    lat0: float,
    lon0: float,
    window_size_px: int,
    cfg: Dict[str, Any],
) -> Optional[Tuple[np.ndarray, List[str], Dict[str, str]]]:
    """Extract the (H, W, T, 2) pressure-channel volume for one event.

    Returns (volume, channel_names, units) with channels ordered
    [shear_850_200_mps, rh_mid], or None if ANY timestep is unavailable
    (all-or-nothing: partial pressure data never produces a partial cube).
    """
    wind_levels = tuple(cfg_get(cfg, "download.pressure_levels.wind_levels", [850, 200]))
    rh_levels = list(cfg_get(cfg, "download.pressure_levels.rh_levels", [700, 600, 500]))

    shear_t: List[np.ndarray] = []
    rh_t: List[np.ndarray] = []
    for offset_h in offsets_hours:
        dt = dt0 + timedelta(hours=int(offset_h))
        result = _extract_timestep(raw_dir, dt, lat0, lon0, window_size_px,
                                   wind_levels, rh_levels)
        if result is None:
            return None
        shear_t.append(result[0])
        rh_t.append(result[1])

    volume = np.stack(
        [np.stack(shear_t, axis=2), np.stack(rh_t, axis=2)], axis=-1
    ).astype(np.float32)  # (H, W, T, 2)
    return volume, [SHEAR_CHANNEL, RH_CHANNEL], dict(PL_UNITS)
