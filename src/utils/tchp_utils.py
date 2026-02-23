# src/utils/tchp_utils.py
from __future__ import annotations

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional, Tuple
from scipy.ndimage import maximum_filter, gaussian_filter
import logging

logger = logging.getLogger(__name__)


def load_tchp_file(tchp_path: Path, timestamp, lat: float, lon: float, window_deg: float = 5.0) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load TCHP data for a given timestamp and region.

    Returns:
        (tchp_values, lats, lons) or None if file not found or region empty.
    """
    if not tchp_path.exists():
        return None
    try:
        ds = xr.open_dataset(tchp_path)
        # Select nearest time
        ds_t = ds.sel(time=timestamp, method='nearest')
        # Define spatial window
        lon_min, lon_max = lon - window_deg, lon + window_deg
        lat_min, lat_max = lat - window_deg, lat + window_deg
        # Handle longitude wrap (if dataset uses 0-360)
        if 'lon' in ds_t.coords and ds_t.lon.min() >= 0 and lon_min < 0:
            lon_min += 360
            lon_max += 360
        # Subset
        ds_region = ds_t.sel(lat=slice(lat_min, lat_max),
                             lon=slice(lon_min, lon_max))
        if ds_region.sizes['lat'] == 0 or ds_region.sizes['lon'] == 0:
            return None
        tchp = ds_region['tchp'].values if 'tchp' in ds_region else ds_region['Tropical_Cyclone_Heat_Potential'].values
        lats = ds_region['lat'].values
        lons = ds_region['lon'].values
        ds.close()
        return tchp, lats, lons
    except Exception as e:
        logger.warning(f"Error loading TCHP from {tchp_path}: {e}")
        return None


def find_tchp_max(tchp: np.ndarray, lats: np.ndarray, lons: np.ndarray, window_px: int = 3) -> Tuple[float, float]:
    """
    Find coordinates of the local maximum of TCHP (smoothed).
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


def get_tchp_file_path(tchp_dir: Path, year: int, source: str = 'auto') -> Path:
    """
    Determine the expected filename for a given year and source.
    """
    if source == 'noaa' or (source == 'auto' and year >= 2022):
        return tchp_dir / f"tchp_noaa_{year}.nc"
    elif source == 'aoml' or (source == 'auto' and 1993 <= year <= 2021):
        return tchp_dir / f"tchp_aoml_{year}.nc"
    elif source == 'copernicus' or (source == 'auto' and year < 1993):
        return tchp_dir / f"tchp_copernicus_{year}.nc"
    else:
        return tchp_dir / f"tchp_{year}.nc"   # fallback genérico
