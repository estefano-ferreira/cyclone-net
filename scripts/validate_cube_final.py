#!/usr/bin/env python3
"""
CycloneNet V2.1 – Cube Validation Tool (final version with robust NetCDF opening)

Validates that a preprocessed cube (.npy) exactly matches the source ERA5 monthly
NetCDF files, after applying the same physical normalizations used in preprocessing.
Uses the lists "timestamps", "center_lats", and "center_lons" stored in the JSON.

Features:
- Converts paths to short 8.3 format on Windows to avoid Unicode issues.
- Tries netcdf4 engine first, then falls back to h5netcdf.
- Checks minimum file size to detect corruption.
- If lists are missing, aborts with error (requires reprocessing).

Usage:
    python validate_cube_final.py <event_id> [--config CONFIG_PATH] [--log-level LEVEL]
"""

from src.utils.io_utils import get_short_path_windows
from src.utils.config import cfg_get, load_config
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# ---- Automatic project root detection (works from root or scripts/) ----
script_path = Path(__file__).resolve()
if script_path.parent.name == "scripts":
    project_root = script_path.parent.parent
else:
    project_root = script_path.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# -------------------------------------------------------------------------


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Robust NetCDF opening (identical strategy to preprocess_scientific.py)
# ----------------------------------------------------------------------

def open_netcdf_safe(path: Path) -> xr.Dataset:
    """
    Open NetCDF with CF decoding enabled, trying netcdf4 first then h5netcdf.
    On Windows, uses short path to avoid Unicode issues.
    Raises exception if file cannot be opened or is too small.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.stat().st_size < 10_000_000:  # 10 MB minimum for a monthly file
        raise ValueError(
            f"File too small (likely corrupt): {path} (size={path.stat().st_size} bytes)")

    # Get short path on Windows
    path_to_open = get_short_path_windows(path)

    # Try netcdf4 engine first
    try:
        ds = xr.open_dataset(path_to_open, engine='netcdf4', decode_cf=True)
        logger.debug(f"Opened {path.name} with netcdf4")
        return ds
    except Exception as e:
        logger.debug(f"netcdf4 failed for {path.name}: {e}, trying h5netcdf")
        try:
            ds = xr.open_dataset(
                path_to_open, engine='h5netcdf', decode_cf=True)
            logger.debug(f"Opened {path.name} with h5netcdf")
            return ds
        except Exception as e2:
            raise RuntimeError(
                f"Failed to open {path.name} with both engines: {e2}") from e2


# ----------------------------------------------------------------------
# Helper functions (identical to preprocess_scientific.py)
# ----------------------------------------------------------------------

def _find_monthly_file(raw_dirs: List[Path], year: int, month: int) -> Optional[Path]:
    """Find a monthly file for given year/month in any of the provided directories."""
    pattern = f"era5_{year}_{month:02d}*.nc"
    for d in raw_dirs:
        matches = sorted(d.glob(pattern))
        if matches:
            return matches[0]
    return None


def _get_time_coordinate(ds: xr.Dataset) -> Optional[str]:
    """Return the name of the time coordinate in the dataset."""
    for candidate in ['time', 'valid_time']:
        if candidate in ds.coords:
            return candidate
    return None


def ensure_2d(a: np.ndarray) -> np.ndarray:
    """Ensure array is 2D (lat, lon); squeeze singleton dims if needed."""
    arr = np.asarray(a)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D field, got shape={arr.shape}")
    return arr


def normalize_sst_to_kelvin(sst_raw: np.ndarray) -> np.ndarray:
    """
    Normalize SST to Kelvin (same heuristic as preprocessing).
    - If mean in [-5,45] -> Celsius -> convert to Kelvin.
    - If mean in [240,330] -> already Kelvin.
    """
    sst = ensure_2d(sst_raw).astype(np.float32)
    m = float(np.nanmean(sst))
    if -5.0 <= m <= 45.0:
        return sst + np.float32(273.15)
    if 240.0 <= m <= 330.0:
        return sst
    raise ValueError(f"SST out of plausible range after read (mean={m:.2f})")


def normalize_msl_to_pa(msl_raw: np.ndarray) -> np.ndarray:
    """
    Normalize MSLP to Pascal.
    - If mean in [800,1100] -> hPa -> convert to Pa.
    - If mean in [80_000,110_000] -> already Pa.
    """
    msl = ensure_2d(msl_raw).astype(np.float32)
    m = float(np.nanmean(msl))
    if 800.0 <= m <= 1100.0:
        return msl * np.float32(100.0)
    if 80_000.0 <= m <= 110_000.0:
        return msl
    raise ValueError(f"MSLP out of plausible range after read (mean={m:.2f})")


def extract_window_by_index(
    ds: xr.Dataset,
    lat_center: float,
    lon_center: float,
    window_size_px: int = 40
) -> xr.Dataset:
    """Extract fixed-size spatial window using index slicing (identical to preprocessing)."""
    lat_idx = int(np.abs(ds.latitude - lat_center).argmin())
    lon_idx = int(np.abs(ds.longitude - lon_center).argmin())
    offset = window_size_px // 2

    lat_start = max(0, lat_idx - offset)
    lat_end = min(ds.sizes['latitude'], lat_idx + offset)
    lon_start = max(0, lon_idx - offset)
    lon_end = min(ds.sizes['longitude'], lon_idx + offset)

    subset = ds.isel(latitude=slice(lat_start, lat_end),
                     longitude=slice(lon_start, lon_end))

    h, w = subset.sizes['latitude'], subset.sizes['longitude']
    if h < window_size_px or w < window_size_px:
        pad_h_before = (window_size_px - h) // 2
        pad_h_after = window_size_px - h - pad_h_before
        pad_w_before = (window_size_px - w) // 2
        pad_w_after = window_size_px - w - pad_w_before
        subset = subset.pad(
            latitude=(pad_h_before, pad_h_after),
            longitude=(pad_w_before, pad_w_after),
            mode='reflect'
        )
    return subset


def _get_variable(ds: xr.Dataset, candidates: List[str]) -> xr.DataArray:
    """Return first matching variable."""
    for name in candidates:
        if name in ds.data_vars:
            return ds[name]
        if name in ds.variables:
            return ds[name]
    raise KeyError(f"None of {candidates} found.")


# ----------------------------------------------------------------------
# Core validation
# ----------------------------------------------------------------------

def validate_event(event_id: str, config: Dict[str, Any]) -> bool:
    raw_dir = Path(cfg_get(config, "paths.raw_data", "./data/raw")).resolve()
    interim_dir = Path(cfg_get(config, "paths.interim_data",
                       "./data/interim")).resolve()

    alt_dirs = cfg_get(config, "preprocess.additional_raw_dirs", [])
    raw_dirs = [raw_dir] + [Path(d).resolve() for d in alt_dirs if d]

    cube_path = interim_dir / f"{event_id}.npy"
    json_path = interim_dir / f"{event_id}.json"
    lats_path = interim_dir / f"{event_id}_lats.npy"
    lons_path = interim_dir / f"{event_id}_lons.npy"

    if not cube_path.exists():
        logger.error(f"Cube not found: {cube_path}")
        return False
    if not json_path.exists():
        logger.error(f"Metadata not found: {json_path}")
        return False
    if not lats_path.exists() or not lons_path.exists():
        logger.error(f"Lat/lon grids not found for {event_id}")
        return False

    cube = np.load(cube_path)                     # (H, W, T, C)
    with open(json_path, 'r') as f:
        meta = json.load(f)

    T = cube.shape[2]
    window_size = cube.shape[0]  # should be 40

    # --- Obter listas do JSON (obrigatórias) ---
    timestamps = meta.get("timestamps")
    center_lats = meta.get("center_lats")
    center_lons = meta.get("center_lons")

    if timestamps is None or center_lats is None or center_lons is None:
        logger.error("JSON does not contain 'timestamps', 'center_lats', and 'center_lons'. "
                     "Please reprocess the event with the updated preprocessor.")
        return False

    ts_list = [pd.to_datetime(ts) for ts in timestamps]
    if len(ts_list) != T:
        logger.error(
            f"JSON has {len(ts_list)} timestamps, but cube has {T} frames.")
        return False
    if len(center_lats) != T or len(center_lons) != T:
        logger.error("Coordinate lists in JSON have incorrect length.")
        return False

    # --- Validation frame by frame ---
    all_match = True
    rtol = 1e-5
    atol = 1e-5

    for t_idx, (ts, lat_c, lon_c) in enumerate(zip(ts_list, center_lats, center_lons)):
        logger.debug(
            f"Validating timestep {t_idx}: {ts} (lat={lat_c}, lon={lon_c})")

        monthly_file = _find_monthly_file(raw_dirs, ts.year, ts.month)
        if monthly_file is None:
            logger.error(
                f"Monthly file not found for {ts.year}-{ts.month:02d}")
            all_match = False
            continue

        try:
            ds = open_netcdf_safe(monthly_file)
        except Exception as e:
            logger.error(f"Failed to open {monthly_file.name}: {e}")
            all_match = False
            continue

        time_coord = _get_time_coordinate(ds)
        if time_coord is None:
            logger.error(f"No time coordinate in {monthly_file.name}")
            ds.close()
            all_match = False
            continue

        try:
            ds_t = ds.sel({time_coord: ts}, method=None)  # exact match
        except KeyError:
            logger.error(f"Timestamp {ts} not found in {monthly_file.name}")
            ds.close()
            all_match = False
            continue

        subset = extract_window_by_index(
            ds_t, lat_c, lon_c, window_size).load()

        # Extract raw variables
        sst_raw = _get_variable(
            subset, ["sst", "sea_surface_temperature"]).values
        msl_raw = _get_variable(
            subset, ["msl", "mean_sea_level_pressure"]).values
        u10_raw = _get_variable(
            subset, ["u10", "10m_u_component_of_wind"]).values
        v10_raw = _get_variable(
            subset, ["v10", "10m_v_component_of_wind"]).values

        # Apply the same normalizations as in preprocessing
        sst_norm = normalize_sst_to_kelvin(sst_raw)
        msl_norm = normalize_msl_to_pa(msl_raw)
        u10 = ensure_2d(u10_raw).astype(np.float32)
        v10 = ensure_2d(v10_raw).astype(np.float32)

        # Stack into (H, W, 4)
        extracted = np.stack([sst_norm, msl_norm, u10, v10],
                             axis=-1).astype(np.float32)

        cube_slice = cube[:, :, t_idx, :]

        if not np.allclose(cube_slice, extracted, rtol=rtol, atol=atol):
            diff = np.abs(cube_slice - extracted).max()
            logger.error(
                f"Timestep {t_idx} ({ts}) mismatch! Max diff = {diff:.6f}")
            all_match = False
        else:
            logger.info(f"Timestep {t_idx} ({ts}) OK (max diff <= {atol}).")

        ds.close()

    # --- Validate lat/lon grids against the last timestep ---
    last_ts = ts_list[-1]
    last_lat = center_lats[-1]
    last_lon = center_lons[-1]
    monthly_last = _find_monthly_file(raw_dirs, last_ts.year, last_ts.month)
    if monthly_last:
        try:
            ds_last = open_netcdf_safe(monthly_last)
        except Exception as e:
            logger.error(
                f"Failed to open {monthly_last.name} for grid validation: {e}")
            all_match = False
        else:
            time_coord = _get_time_coordinate(ds_last)
            ds_t_last = ds_last.sel({time_coord: last_ts}, method=None)
            subset_last = extract_window_by_index(
                ds_t_last, last_lat, last_lon, window_size).load()
            lons_2d, lats_2d = np.meshgrid(
                subset_last.longitude.values, subset_last.latitude.values)
            ds_last.close()

            lats_cube = np.load(lats_path)
            lons_cube = np.load(lons_path)

            if not np.allclose(lats_cube, lats_2d, rtol=1e-7) or not np.allclose(lons_cube, lons_2d, rtol=1e-7):
                logger.error(
                    "Lat/lon grids do not match the last timestep's NetCDF window.")
                all_match = False
            else:
                logger.info("Lat/lon grids match the last timestep.")
    else:
        logger.warning(
            "Could not verify lat/lon grids (monthly file missing).")

    return all_match


def main():
    parser = argparse.ArgumentParser(
        description="Validate a preprocessed CycloneNet cube.")
    parser.add_argument("event_id", type=str,
                        help="Event ID (e.g., era5_1980_10_11_0000)")
    parser.add_argument("--config", type=str,
                        default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--log-level", default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(message)s")

    config = load_config(args.config)
    success = validate_event(args.event_id, config)

    if success:
        logger.info(
            "✅ Validation passed: all data traceable to original NetCDF files.")
        sys.exit(0)
    else:
        logger.error("❌ Validation failed: some mismatches detected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
