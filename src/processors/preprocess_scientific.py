#!/usr/bin/env python3
"""
CycloneNet V2.1 – Scientific preprocessing with robust file location and quality checks.

This module reads ERA5 monthly NetCDF files and extracts spatio-temporal windows
for each event, saving them as .npy cubes and JSON metadata.
No intermediate daily files are created. Original monthly files remain untouched.

Features:
- Flexible file location: searches in raw_data and alternative directories.
- Strict rejection of events with missing or corrupt data (no fallback).
- Physical unit normalization (SST to Kelvin, MSLP to Pascal) and range checks.
- Parallel processing with proper error handling.
- JSON metadata now includes full lists of timestamps and center coordinates for
  each timestep in the window, enabling validation independent of event_list.csv.

Author: Estefano Senhor Ferreira
License: CC BY-NC 4.0
"""

from __future__ import annotations
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from src.utils.config import CONFIG, cfg_get

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def _resolve_path(p: Union[str, Path]) -> Path:
    """Convert to absolute Path, expanding user and resolving."""
    return Path(str(p)).expanduser().resolve()


def _clean_nan(obj: Any) -> Any:
    """
    Recursively convert float NaN to None for JSON serialization.
    This ensures that JSON files are strictly compliant (no NaN values).
    """
    if isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_clean_nan(v) for v in obj]
    else:
        return obj


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write JSON with indentation, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Clean NaN before serialization
    obj_clean = _clean_nan(obj)
    path.write_text(json.dumps(obj_clean, indent=2), encoding="utf-8")


def _find_monthly_file(raw_dirs: List[Path], year: int, month: int) -> Optional[Path]:
    """
    Find a monthly file for given year/month in any of the provided directories.
    The file must match the pattern 'era5_YYYY_MM*.nc' (any suffix allowed).
    """
    pattern = f"era5_{year}_{month:02d}*.nc"
    for d in raw_dirs:
        matches = sorted(d.glob(pattern))
        if matches:
            logger.debug(f"Found monthly file: {matches[0]} in {d}")
            return matches[0]
    return None


def open_netcdf_safe(path: Path) -> xr.Dataset:
    """
    Open NetCDF with CF decoding enabled.
    Raises exception if file cannot be opened or is too small.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.stat().st_size < 10_000:  # 10 KB minimum
        raise ValueError(
            f"File too small (likely corrupt): {path} (size={path.stat().st_size} bytes)")

    # Try netcdf4 engine first, fall back to h5netcdf
    try:
        ds = xr.open_dataset(path, engine='netcdf4', decode_cf=True)
    except Exception as e:
        logger.debug(f"netcdf4 failed for {path}: {e}, trying h5netcdf")
        ds = xr.open_dataset(path, engine='h5netcdf', decode_cf=True)
    return ds


def get_var(ds: xr.Dataset, candidates: List[str]) -> xr.DataArray:
    """Return the first matching variable among candidates."""
    for name in candidates:
        if name in ds.data_vars:
            return ds[name]
        if name in ds.variables:
            return ds[name]
    raise KeyError(
        f"None of {candidates} found in dataset. Available: {list(ds.data_vars)}")


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
    Normalize SST to Kelvin.

    Heuristic:
    - Mean in [-5, 45]   -> likely Celsius -> convert to Kelvin
    - Mean in [240, 330] -> likely Kelvin  -> keep
    Otherwise -> raise (data likely scaled/decoded incorrectly)
    """
    sst = ensure_2d(sst_raw).astype(np.float32)
    m = float(np.nanmean(sst))
    if -5.0 <= m <= 45.0:
        return sst + np.float32(273.15)
    if 240.0 <= m <= 330.0:
        return sst
    raise ValueError(
        f"SST out of plausible range after read (mean={m:.2f}). Check scaling/units.")


def normalize_msl_to_pa(msl_raw: np.ndarray) -> np.ndarray:
    """
    Normalize mean sea level pressure to Pascal (Pa).

    Heuristic:
    - Mean in [800, 1100]      -> likely hPa -> convert to Pa
    - Mean in [80_000, 110_000] -> likely Pa  -> keep
    Otherwise -> raise
    """
    msl = ensure_2d(msl_raw).astype(np.float32)
    m = float(np.nanmean(msl))
    if 800.0 <= m <= 1100.0:
        return msl * np.float32(100.0)
    if 80_000.0 <= m <= 110_000.0:
        return msl
    raise ValueError(
        f"MSLP out of plausible range after read (mean={m:.2f}). Check scaling/units.")


def qc_physical_ranges(sst_k: np.ndarray, msl_pa: np.ndarray, u10: np.ndarray, v10: np.ndarray) -> None:
    """
    Hard QC checks: fail fast to prevent corrupt samples from being saved.
    """
    sst_m = float(np.nanmean(sst_k))
    msl_m = float(np.nanmean(msl_pa))
    u_abs = float(np.nanmax(np.abs(u10)))
    v_abs = float(np.nanmax(np.abs(v10)))

    # Conservative global plausibility bounds
    if not (240.0 <= sst_m <= 330.0):
        raise ValueError(f"QC fail: SST mean {sst_m:.2f} K outside [240, 330]")
    if not (80_000.0 <= msl_m <= 110_000.0):
        raise ValueError(
            f"QC fail: MSLP mean {msl_m:.2f} Pa outside [80000, 110000]")
    if u_abs > 120.0 or v_abs > 120.0:
        raise ValueError(
            f"QC fail: wind magnitude too high (u_max={u_abs:.2f}, v_max={v_abs:.2f})")


def compute_energy_proxy(subset: xr.Dataset) -> Optional[Tuple[float, float, np.ndarray]]:
    """
    Energy proxy: (SST_C - mean(SST_C)) * (101325 / MSLP_Pa)
    Uses normalization functions to ensure physical units.
    """
    try:
        sst_raw = get_var(subset, ["sst", "sea_surface_temperature"]).values
        msl_raw = get_var(subset, ["msl", "mean_sea_level_pressure"]).values
        sst_k = normalize_sst_to_kelvin(sst_raw)
        msl_pa = normalize_msl_to_pa(msl_raw)

        sst_c = sst_k - np.float32(273.15)
        sst_anom = sst_c - np.nanmean(sst_c)
        energy = sst_anom * (np.float32(101325.0) / msl_pa)

        # Find max energy location
        lat_vals = subset.latitude.values
        lon_vals = subset.longitude.values
        max_idx = np.nanargmax(energy)
        r, c = np.unravel_index(max_idx, energy.shape)
        return float(lat_vals[r]), float(lon_vals[c]), energy.astype(np.float32)
    except Exception as e:
        logger.debug(f"Energy proxy computation failed: {e}")
        return None


def extract_window_by_index(
    ds: xr.Dataset,
    lat_center: float,
    lon_center: float,
    window_size_px: int = 40
) -> xr.Dataset:
    """
    Extract a fixed-size spatial window using index slicing.
    Ensures exactly window_size_px x window_size_px grid points.
    If the window goes beyond dataset bounds, pad with reflection.
    """
    # Find nearest indices
    lat_idx = int(np.abs(ds.latitude - lat_center).argmin())
    lon_idx = int(np.abs(ds.longitude - lon_center).argmin())
    offset = window_size_px // 2  # e.g., 20 for 40x40

    # Compute slice boundaries (clamp to dataset bounds)
    lat_start = max(0, lat_idx - offset)
    lat_end = min(ds.sizes['latitude'], lat_idx + offset)
    lon_start = max(0, lon_idx - offset)
    lon_end = min(ds.sizes['longitude'], lon_idx + offset)

    subset = ds.isel(latitude=slice(lat_start, lat_end),
                     longitude=slice(lon_start, lon_end))

    # If the sliced size is smaller than target, pad symmetrically
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


def _get_time_coordinate(ds: xr.Dataset) -> Optional[str]:
    """Return the name of the time coordinate in the dataset."""
    for candidate in ['time', 'valid_time']:
        if candidate in ds.coords:
            return candidate
    return None


# ----------------------------------------------------------------------
# Core event processing
# ----------------------------------------------------------------------

def process_event(
    rows: pd.DataFrame,
    interim_dir: Path,
    raw_dirs: List[Path],
    basin: str,
    window_size_px: int = 40,
    temporal_depth: int = 5,
    energy_enabled: bool = False,
) -> bool:
    """
    Process one event window (T timesteps) by reading directly from monthly files.
    Returns True if successful, False otherwise (event rejected).
    """
    start_time = time.time()
    last = rows.iloc[-1]
    event_id = str(last["event_id"]) if "event_id" in rows.columns else str(
        last["nc_filename"]).replace(".nc", "")
    logger.debug(f"Processing event {event_id}...")

    try:
        # Group timesteps by year-month to open monthly files efficiently
        timesteps = []
        for _, r in rows.iterrows():
            dt = pd.to_datetime(r["timestamp"])
            ym = (dt.year, dt.month)
            timesteps.append({
                "dt": dt,
                "ym": ym,
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
            })

        # We'll open monthly files as needed and cache them per event
        cubes_t: List[np.ndarray] = []
        grids_lats, grids_lons = None, None
        energy_field = None
        true_lat, true_lon = None, None

        # Cache for opened datasets (to avoid reopening the same file multiple times)
        monthly_datasets: Dict[Tuple[int, int], xr.Dataset] = {}

        try:
            for ts in timesteps:
                ym = ts["ym"]
                if ym not in monthly_datasets:
                    # Find the monthly file for this year-month
                    mfile = _find_monthly_file(raw_dirs, ym[0], ym[1])
                    if mfile is None:
                        raise FileNotFoundError(f"No monthly file for {ym}")
                    monthly_datasets[ym] = open_netcdf_safe(mfile)

                ds = monthly_datasets[ym]

                # Select the nearest time (should be exact if timestamps match)
                time_coord = _get_time_coordinate(ds)
                if time_coord is None:
                    raise ValueError(f"No time coordinate in dataset for {ym}")
                ds_t = ds.sel({time_coord: ts["dt"]}, method='nearest')

                # Extract spatial window
                subset = extract_window_by_index(
                    ds_t, ts["lat"], ts["lon"], window_size_px).load()

                # For last timestep, save grids and compute energy if enabled
                if ts is timesteps[-1]:
                    lons_2d, lats_2d = np.meshgrid(
                        subset.longitude.values, subset.latitude.values)
                    grids_lats = lats_2d.astype(np.float32)
                    grids_lons = lons_2d.astype(np.float32)

                    if energy_enabled:
                        res = compute_energy_proxy(subset)
                        if res is not None:
                            true_lat, true_lon, energy_field = res

                # Extract raw variables
                sst_raw = get_var(
                    subset, ["sst", "sea_surface_temperature"]).values
                msl_raw = get_var(
                    subset, ["msl", "mean_sea_level_pressure"]).values
                u10_raw = get_var(
                    subset, ["u10", "10m_u_component_of_wind"]).values
                v10_raw = get_var(
                    subset, ["v10", "10m_v_component_of_wind"]).values

                # Reject event if SST or MSLP entirely NaN
                if np.isnan(sst_raw).all() or np.isnan(msl_raw).all():
                    raise ValueError("SST or MSLP entirely NaN.")

                # Normalize units safely
                sst = normalize_sst_to_kelvin(sst_raw)
                msl = normalize_msl_to_pa(msl_raw)
                u10 = ensure_2d(u10_raw).astype(np.float32)
                v10 = ensure_2d(v10_raw).astype(np.float32)

                # Hard QC (prevents corrupt .npy)
                qc_physical_ranges(sst, msl, u10, v10)

                cube_t = np.stack([sst, msl, u10, v10],
                                  axis=-1).astype(np.float32)
                if cube_t.shape != (window_size_px, window_size_px, 4):
                    raise ValueError(f"Unexpected cube_t shape {cube_t.shape}")

                cubes_t.append(cube_t)

        finally:
            for ds in monthly_datasets.values():
                ds.close()

        # Combine timesteps -> (H, W, T, C)
        if len(cubes_t) != temporal_depth:
            raise ValueError(
                f"Expected {temporal_depth} timesteps, got {len(cubes_t)}")

        cube = np.stack(cubes_t, axis=2)  # axis=2 is time
        if cube.shape != (window_size_px, window_size_px, temporal_depth, 4):
            raise ValueError(f"Final cube shape mismatch: {cube.shape}")

        # Target definition
        center_lat = float(rows.iloc[-1]["lat"])
        center_lon = float(rows.iloc[-1]["lon"])
        if not energy_enabled or true_lat is None:
            true_lat, true_lon = center_lat, center_lon
            target_def = "cyclone_center_proxy"
        else:
            target_def = "energy_proxy_max_sst_anomaly_pressure"

        # Build metadata dictionary
        meta = {
            "event_id": event_id,
            "sid": str(last.get("sid", "")),
            "timestamp": str(pd.to_datetime(last["timestamp"])),
            "ri_label": int(last["ri_label"]),
            "storm_name": str(last.get("storm_name", "")),
            "basin": str(last.get("basin", basin)),
            "wind_knots": float(last.get("wind_knots", np.nan)),
            "pressure_mb": float(last.get("pressure_mb", np.nan)),
            "center_lat": center_lat,
            "center_lon": center_lon,
            "true_energy_lat": float(true_lat),
            "true_energy_lon": float(true_lon),
            "target_definition": target_def,
            "cube_shape": list(cube.shape),
            # New fields for full traceability
            "timestamps": [str(ts["dt"]) for ts in timesteps],
            "center_lats": [ts["lat"] for ts in timesteps],
            "center_lons": [ts["lon"] for ts in timesteps],
        }

        interim_dir.mkdir(parents=True, exist_ok=True)
        np.save(interim_dir / f"{event_id}.npy", cube)
        _save_json(interim_dir / f"{event_id}.json", meta)

        if grids_lats is not None and grids_lons is not None:
            np.save(interim_dir / f"{event_id}_lats.npy", grids_lats)
            np.save(interim_dir / f"{event_id}_lons.npy", grids_lons)
        if energy_field is not None:
            np.save(interim_dir / f"{event_id}_energy.npy", energy_field)

        elapsed = time.time() - start_time
        logger.debug(
            f"Event {event_id} processed successfully in {elapsed:.2f}s")
        return True

    except Exception as e:
        logger.error(f"Event {event_id} failed: {e}", exc_info=True)
        return False


# ----------------------------------------------------------------------
# Main orchestration
# ----------------------------------------------------------------------

def process_all_events(event_list_path: Union[str, Path]) -> None:
    """
    Main preprocessing routine: load event list, create tasks, run parallel processing.
    """
    raw_dir = _resolve_path(cfg_get(CONFIG, "paths.raw_data", "./data/raw"))
    # Additional directories to search for monthly files (configurable)
    alt_dirs = cfg_get(CONFIG, "preprocess.additional_raw_dirs", [])
    raw_dirs = [raw_dir] + [_resolve_path(d) for d in alt_dirs if d]

    interim_dir = _resolve_path(
        cfg_get(CONFIG, "paths.interim_data", "./data/interim"))
    event_list_path = _resolve_path(event_list_path)

    logger.info(f"Raw data directories: {raw_dirs}")
    logger.info(f"Interim directory: {interim_dir}")

    if not event_list_path.exists():
        raise FileNotFoundError(f"event_list.csv not found: {event_list_path}")

    df = pd.read_csv(event_list_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["sid", "timestamp"]).reset_index(drop=True)

    T = int(cfg_get(CONFIG, "model.temporal_depth", 5))
    max_workers = int(cfg_get(CONFIG, "preprocess.max_workers", 4))
    window_size_px = int(
        cfg_get(CONFIG, "preprocess.target_shape", [40, 40])[0])
    energy_enabled = bool(
        cfg_get(CONFIG, "physics.energy_source.enabled", False))

    # Build sliding windows of length T for each storm
    all_windows = []
    for sid, g in df.groupby("sid"):
        g = g.reset_index(drop=True)
        if len(g) < T:
            continue
        for i in range(T - 1, len(g)):
            window = g.iloc[i - T + 1: i + 1].copy()
            all_windows.append(window)

    logger.info(f"Total windows (before file check): {len(all_windows)}")

    # Pre-filter windows that have all required monthly files available
    tasks = []
    for w in all_windows:
        required_years_months = {(pd.to_datetime(r["timestamp"]).year,
                                  pd.to_datetime(r["timestamp"]).month) for _, r in w.iterrows()}
        all_files_exist = all(
            _find_monthly_file(raw_dirs, ym[0], ym[1]) is not None
            for ym in required_years_months
        )
        if all_files_exist:
            tasks.append(w)
        else:
            logger.debug(
                f"Skipping window due to missing monthly file(s): last timestamp = {w.iloc[-1]['timestamp']}")

    total_tasks = len(tasks)
    logger.info(
        f"Windows with all files present: {total_tasks} / {len(all_windows)}")

    basin_mapping = cfg_get(CONFIG, "data_selection.basin_mapping", {}) or {}
    ok = 0

    # Parallel processing with interrupt handling
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for w in tasks:
                basin_raw = str(w.iloc[-1].get("basin", "NA")).strip()
                basin = basin_mapping.get(basin_raw, basin_raw.lower())
                futures.append(
                    executor.submit(
                        process_event, w, interim_dir, raw_dirs, basin,
                        window_size_px, T, energy_enabled
                    )
                )

            with tqdm(total=total_tasks, desc="Processing events", unit="event") as pbar:
                for future in as_completed(futures):
                    try:
                        # 2 minutes per task timeout
                        result = future.result(timeout=120)
                        if result:
                            ok += 1
                    except Exception as e:
                        logger.error(f"Task exception: {e}")
                    finally:
                        pbar.update(1)
    except KeyboardInterrupt:
        logger.warning(
            "KeyboardInterrupt received. Cancelling pending tasks...")
        for fut in futures:
            fut.cancel()
        raise

    logger.info(f"Preprocessing done. Successful events: {ok} / {total_tasks}")

    # Consolidate metadata into a single CSV
    metadata_list = []
    for w in tqdm(tasks, desc="Collecting metadata", unit="event"):
        last_row = w.iloc[-1]
        event_id = (
            str(last_row["event_id"]) if "event_id" in last_row
            else str(last_row["nc_filename"]).replace(".nc", "")
        )
        json_path = interim_dir / f"{event_id}.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                row = {
                    "event_id": data["event_id"],
                    "sid": data["sid"],
                    "timestamp": data["timestamp"],
                    "ri_label": data["ri_label"],
                    "storm_name": data["storm_name"],
                    "basin": data["basin"],
                    "wind_knots": data["wind_knots"],
                    "pressure_mb": data["pressure_mb"],
                    "center_lat": data["center_lat"],
                    "center_lon": data["center_lon"],
                    "true_energy_lat": data["true_energy_lat"],
                    "true_energy_lon": data["true_energy_lon"],
                    "target_definition": data["target_definition"],
                }
                metadata_list.append(row)
            except Exception as e:
                logger.warning(f"Could not read {json_path}: {e}")

    if metadata_list:
        metadata_df = pd.DataFrame(metadata_list)
        csv_path = interim_dir / "samples_metadata.csv"
        metadata_df.to_csv(csv_path, index=False)
        logger.info(f"Generated {csv_path} with {len(metadata_df)} entries.")
    else:
        logger.error(
            "No metadata files found – preprocessing may have failed for all events.")


def main() -> None:
    """Entry point for the preprocessing script."""
    event_list = cfg_get(CONFIG, "paths.event_list", "./data/event_list.csv")
    process_all_events(event_list)


if __name__ == "__main__":
    logging.basicConfig(level=cfg_get(CONFIG, "logging.level", "INFO"))
    main()
