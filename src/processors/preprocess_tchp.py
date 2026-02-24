# src/processors/preprocess_tchp.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import maximum_filter, gaussian_filter
from tqdm import tqdm

from src.utils.config import cfg_get
from src.utils.tchp_utils import get_tchp_file_path

logger = logging.getLogger(__name__)


def load_tchp_file(
    tchp_path: Path,
    timestamp: pd.Timestamp,
    lat: float,
    lon: float,
    window_deg: float = 5.0
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load TCHP data for a given timestamp and region from a NetCDF file.

    Args:
        tchp_path: Path to the NetCDF file.
        timestamp: Target time.
        lat, lon: Center coordinates.
        window_deg: Half-width of the spatial window in degrees.

    Returns:
        (tchp_values, lats, lons) or None if file not found or region empty.
    """
    if not tchp_path.exists():
        return None
    try:
        ds = xr.open_dataset(tchp_path)
        # Select nearest time
        ds_t = ds.sel(time=timestamp, method="nearest")
        # Define spatial window
        lon_min, lon_max = lon - window_deg, lon + window_deg
        lat_min, lat_max = lat - window_deg, lat + window_deg

        # Handle longitude wrap if dataset uses 0-360
        if "lon" in ds_t.coords and ds_t.lon.min() >= 0 and lon_min < 0:
            lon_min += 360
            lon_max += 360

        # Subset
        ds_region = ds_t.sel(
            lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)
        )
        if ds_region.sizes["lat"] == 0 or ds_region.sizes["lon"] == 0:
            return None

        # Variable name may vary; try common names
        var_names = ["tchp", "Tropical_Cyclone_Heat_Potential", "TCHP"]
        tchp_var = None
        for v in var_names:
            if v in ds_region:
                tchp_var = v
                break
        if tchp_var is None:
            raise KeyError(f"No TCHP variable found in {tchp_path}")

        tchp = ds_region[tchp_var].values
        lats = ds_region["lat"].values
        lons = ds_region["lon"].values
        ds.close()
        return tchp, lats, lons
    except Exception as e:
        logger.warning(f"Error loading TCHP from {tchp_path}: {e}")
        return None


def find_tchp_max(
    tchp: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    window_px: int = 3
) -> Tuple[float, float]:
    """
    Find coordinates of the local maximum of TCHP (smoothed).

    Args:
        tchp: 2D array of TCHP values.
        lats: 2D array of latitudes.
        lons: 2D array of longitudes.
        window_px: Size of the local maximum filter.

    Returns:
        (lat, lon) of the peak.
    """
    tchp_smooth = gaussian_filter(tchp, sigma=1)
    local_max = maximum_filter(tchp_smooth, size=window_px) == tchp_smooth
    peaks = np.argwhere(local_max)
    if len(peaks) == 0:
        i, j = np.unravel_index(np.argmax(tchp_smooth), tchp_smooth.shape)
    else:
        vals = tchp_smooth[peaks[:, 0], peaks[:, 1]]
        best = peaks[np.argmax(vals)]
        i, j = best
    return float(lats[i]), float(lons[j])


def add_tchp_to_metadata(cfg: Dict[str, Any]) -> None:
    """
    For each event in data/interim, load corresponding TCHP file,
    find the maximum in the vicinity, and add tchp_max_lat/lon to the JSON metadata.
    """
    interim_dir = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    tchp_dir = Path(cfg_get(cfg, "paths.tchp_dir", "./data/external/tchp")).resolve()
    if not tchp_dir.exists():
        logger.error(f"TCHP directory not found: {tchp_dir}")
        return

    json_files = sorted(interim_dir.glob("era5_*.json"))
    if not json_files:
        logger.warning("No event JSON files found in interim directory.")
        return

    updated = 0
    skipped = 0
    missing_tchp = 0

    for json_path in tqdm(json_files, desc="Adding TCHP to metadata"):
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if "tchp_max_lat" in meta and meta["tchp_max_lat"] is not None:
            skipped += 1
            continue

        timestamp = pd.to_datetime(meta.get("timestamp"))
        lat = meta.get("center_lat")
        lon = meta.get("center_lon")
        if timestamp is None or lat is None or lon is None:
            logger.debug(f"Skipping {json_path.name}: missing timestamp/center")
            continue

        year = timestamp.year
        # Determine source based on year (consistent with downloader logic)
        if year >= 2022:
            src = "noaa"
        elif year >= 1993:
            src = "aoml"
        else:
            missing_tchp += 1
            continue  # No TCHP data for years < 1993

        tchp_file = get_tchp_file_path(tchp_dir, year, src)
        if not tchp_file.exists():
            missing_tchp += 1
            continue

        tchp_data = load_tchp_file(tchp_file, timestamp, lat, lon, window_deg=5)
        if tchp_data is None:
            missing_tchp += 1
            continue
        tchp, lats_tchp, lons_tchp = tchp_data
        tchp_max_lat, tchp_max_lon = find_tchp_max(tchp, lats_tchp, lons_tchp, window_px=3)

        meta["tchp_max_lat"] = tchp_max_lat
        meta["tchp_max_lon"] = tchp_max_lon
        meta["tchp_max_value"] = float(np.max(tchp))  # Optional: store the value

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        updated += 1

    logger.info(
        f"TCHP metadata added to {updated} events ({skipped} already had, {missing_tchp} missing TCHP files)."
    )


def run_preprocess_tchp(cfg: Dict[str, Any]) -> None:
    """Entrypoint for preprocess-tchp command."""
    add_tchp_to_metadata(cfg)