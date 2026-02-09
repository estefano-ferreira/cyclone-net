"""
CycloneNet: Physics-Guided Framework for Targeted RI Detection.
---------------------------------------------------------------
Software Engineer: Estefano Senhor Ferreira
License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)

This work is licensed under CC BY-NC 4.0. 
Commercial use is strictly prohibited without prior authorization.
Copyright (c) 2026 Estefano Senhor Ferreira
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def parse_hurdat2(filepath: Path) -> pd.DataFrame:
    """
    Parses a HURDAT2 file and returns a structured DataFrame.
    Args:
        filepath: Path to the HURDAT2 file
    Returns:
        DataFrame with columns: datetime, storm_id, name, lat, lon, wind, pressure
    """
    logger.info(f"Parsing HURDAT2 file: {filepath}")

    records = []
    current_storm = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Hurricane header line
            if line.startswith('AL') or line.startswith('EP'):
                if current_storm is not None and current_storm['entries']:
                    records.extend(current_storm['entries'])

                parts = line.split(',')
                current_storm = {
                    'id': parts[0].strip(),
                    'name': parts[1].strip().upper(),
                    'entries': []
                }
                continue

            # Hurricane data line
            if current_storm is not None and line:
                parts = [p.strip() for p in line.split(',')]

                if len(parts) >= 7:
                    try:
                        # Format date/time
                        date_str = parts[0]
                        time_str = parts[1].zfill(4)
                        dt_str = f"{date_str} {time_str}"

                        # Parseia coordinates (ex: "27.0N" -> 27.0)
                        lat_str = parts[4]
                        lat = float(lat_str[:-1])
                        if lat_str[-1] == 'S':
                            lat = -lat

                        lon_str = parts[5]
                        lon = float(lon_str[:-1])
                        if lon_str[-1] == 'W':
                            lon = -lon

                        entry = {
                            'datetime': dt_str,
                            'storm_id': current_storm['id'],
                            'name': current_storm['name'],
                            'lat': lat,
                            'lon': lon,
                            'wind': int(parts[6]) if parts[6] else 0,
                            'pressure': int(parts[7]) if len(parts) > 7 and parts[7] else 0,
                            'record_type': parts[2] if len(parts) > 2 else '',
                            'status': parts[3] if len(parts) > 3 else ''
                        }

                        current_storm['entries'].append(entry)

                    except (ValueError, IndexError) as e:
                        logger.warning(
                            f"Error parsing line: {line}. Error: {e}")
                        continue

    # Adds latest records
    if current_storm is not None and current_storm['entries']:
        records.extend(current_storm['entries'])

    # Create DataFrame
    df = pd.DataFrame(records)

    if not df.empty:

        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M')

        df = df.sort_values(['storm_id', 'datetime'])

    logger.info(
        f"Parseados {len(df)} records of {df['storm_id'].nunique()} hurricanes")
    return df


def find_ri_events(df: pd.DataFrame, storm_name: str, year: str,
                   threshold_knots: int = 30, window_hours: int = 24) -> list:
    """
    Finds rapid intensification (RI) events in a hurricane.
    Args:
        df: DataFrame with HURDAT2 data
        storm_name: Hurricane name (e.g., “DORIAN”)
        year: Hurricane year (e.g., “2019”)
        threshold_knots: Wind increase threshold in knots
        window_hours: Time window for calculation (hours)
    Returns:
        List of dictionaries with RI events
    """
    logger.info(f"Looking for IR events for {storm_name} {year}")

    # Filter specific hurricane
    storm_mask = (df['name'] == storm_name.upper()) & (
        df['datetime'].dt.year == int(year))
    storm_df = df[storm_mask].copy()

    if storm_df.empty:
        logger.warning(f"Hurricane {storm_name} {year} not found")

        available_storms = df[df['datetime'].dt.year ==
                              int(year)]['name'].unique()
        logger.info(
            f"Hurricanes available in {year}: {list(available_storms)}")
        return []

    # Sort by time
    storm_df = storm_df.sort_values('datetime')
    logger.info(
        f"Found {len(storm_df)} records for {storm_name} {year}")

    # Calculates wind change using shift (more robust than rolling with lambda)
    # HURDAT2 has data every 6 hours, so 24 hours = 4 periods
    steps_24h = window_hours // 6

    if len(storm_df) <= steps_24h:
        logger.warning(
            f"Insufficient data to calculate RI: only {len(storm_df)} records")
        return []

    # Create a column using the wind from 24 hours ago
    storm_df['wind_24h_ago'] = storm_df['wind'].shift(steps_24h)

    # Calculate the change
    storm_df['wind_change'] = storm_df['wind'] - storm_df['wind_24h_ago']

    # Identifies IR events
    ri_events = []
    for idx, row in storm_df.iterrows():
        if pd.notna(row['wind_change']) and row['wind_change'] >= threshold_knots:
            current_time = row['datetime']

            # Checks if it's a new event (avoids nearby duplicates)
            is_new_event = True
            if ri_events:
                # Convert the string from the last event back to a timestamp for comparison.
                last_event_str = ri_events[-1]['datetime']
                last_event_time = pd.to_datetime(
                    last_event_str, format='%Y%m%d %H%M')
                time_diff = (current_time - last_event_time).total_seconds()

                if time_diff <= window_hours * 3600:
                    is_new_event = False
                    # If we find a stronger event in the same period, replace it.
                    if row['wind_change'] > ri_events[-1]['wind_change']:
                        ri_events[-1] = {
                            'datetime': current_time.strftime('%Y%m%d %H%M'),
                            'lat': row['lat'],
                            'lon': row['lon'],
                            'wind': row['wind'],
                            'wind_change': row['wind_change'],
                            'pressure': row['pressure'],
                            'storm_id': row['storm_id'],
                            'name': row['name']
                        }

            if is_new_event:
                event = {
                    'datetime': current_time.strftime('%Y%m%d %H%M'),
                    'lat': row['lat'],
                    'lon': row['lon'],
                    'wind': row['wind'],
                    'wind_change': row['wind_change'],
                    'pressure': row['pressure'],
                    'storm_id': row['storm_id'],
                    'name': row['name']
                }
                ri_events.append(event)

    logger.info(f"Found {len(ri_events)} IR event(s)")

    for i, event in enumerate(ri_events):
        logger.info(f"Event {i+1}: {event['datetime']}, "
                    f"wind: {event['wind']} node, "
                    f"change: {event['wind_change']:.1f} node")

    return ri_events


def extract_storm_cube(
    era5_file: Path,
    storm_center_lat: float,
    storm_center_lon: float,
    window_size_deg: float = 10
) -> np.ndarray:
    """
    Extract an ERA5 cube centered on the cyclone..
    Returns an array (H, W, C) with:
        C0 = SST
        C1 = MSLP
        C2 = U10
        C3 = V10
    """

    if isinstance(era5_file, str):
        era5_file = Path(era5_file)

    logger.info(f"Extracting ERA5 from: {era5_file}")

    if not era5_file.exists():
        logger.error("ERA5 file not found → fallback")
        return create_fallback_cube(storm_center_lat, storm_center_lon, window_size_deg)

    try:
        file_size = era5_file.stat().st_size
        logger.info(f"Size: {file_size/1024:.1f} KB")

        if file_size < 1000:
            logger.warning("Very small file → fallback")
            return create_fallback_cube(storm_center_lat, storm_center_lon, window_size_deg)
    except Exception as e:
        logger.warning(f"Error checking size: {e}")

    # =============================
    # Opening of NETCDF
    # =============================
    try:
        try:
            ds = xr.open_dataset(era5_file, engine="h5netcdf")
        except Exception:
            try:
                ds = xr.open_dataset(era5_file)
            except Exception:
                ds = xr.open_dataset(era5_file, engine="scipy")

        logger.info(f"Dimensions: {dict(ds.sizes)}")
        logger.info(f"Variables: {list(ds.data_vars.keys())}")

    except Exception as e:
        logger.error(f"Failed to open ERA5: {e}")
        return create_fallback_cube(storm_center_lat, storm_center_lon, window_size_deg)

    # =============================
    # GET VARIABLES
    # =============================
    def get_var(names):
        for n in names:
            if n in ds:
                return ds[n]
        return None

    sst = get_var(["sst", "sea_surface_temperature"])
    mslp = get_var(["msl", "mean_sea_level_pressure"])
    u10 = get_var(["u10", "10m_u_component_of_wind"])
    v10 = get_var(["v10", "10m_v_component_of_wind"])

    if any(v is None for v in [sst, mslp, u10, v10]):
        logger.warning("Missing variables → fallback")
        return create_fallback_cube(storm_center_lat, storm_center_lon, window_size_deg)

    # =============================
    # SPATIAL CUTOUT (PRECISION ENGINE)
    # =============================
    # 1. Find the index closest to the actual center
    # ERA5 generally stores Lat from North to South (90 -> -90)
    lat_idx = np.abs(ds.latitude - storm_center_lat).argmin().item()
    lon_idx = np.abs(ds.longitude - storm_center_lon).argmin().item()

    # 2. Set the offset to ensure 40 pixels (20 on each side).
    offset = 20

    # 3. Slice by INDEX INTERVAL (isel) to ensure shape (40, 40)
    # This eliminates the variation of 41 or 40 pixels
    lat_start, lat_end = lat_idx - offset, lat_idx + offset
    lon_start, lon_end = lon_idx - offset, lon_idx + offset

    def crop_by_index(var):
        try:
            # isel uses integer indexes, ensuring total consistency
            return var.isel(latitude=slice(lat_start, lat_end),
                            longitude=slice(lon_start, lon_end))
        except Exception as e:
            logger.error(f"Erro no i-crop: {e}")
            return var

    sst = crop_by_index(sst)
    mslp = crop_by_index(mslp)
    u10 = crop_by_index(u10)
    v10 = crop_by_index(v10)

    # =============================
    # Convert to Numpy 2D
    # =============================
    def to_2d(var):
        arr = var.values
        if arr.ndim == 3:
            arr = arr[0]
        return arr.astype(np.float32)

    sst = to_2d(sst)
    mslp = to_2d(mslp)
    u10 = to_2d(u10)
    v10 = to_2d(v10)

    # =============================
    # STACK → (H, W, C)
    # =============================
    try:
        cube = np.stack([sst, mslp, u10, v10], axis=-1)
        logger.info(f"✅ ERA5 cube created: {cube.shape}")

        # ===== CRITICAL NUMERICAL SANITIZATION =====
        if np.isnan(cube).any():
            logger.warning("Cube contains NaN. Applying numerical cleaning...")

            # replace NaN with the channel's own average.
            for c in range(cube.shape[-1]):
                channel = cube[:, :, c]
                mask = np.isnan(channel)

                if mask.all():
                    # If the entire channel is NaN → zeros
                    cube[:, :, c] = 0.0
                else:
                    mean_val = np.nanmean(channel)
                    channel[mask] = mean_val
                    cube[:, :, c] = channel

        # ensures correct type
        cube = cube.astype(np.float32)

        return cube

    except Exception as e:
        logger.error(f"Error assembling cube: {e}")
        return create_fallback_cube(storm_center_lat, storm_center_lon, window_size_deg)


def create_fallback_cube(lat: float, lon: float, window_size: float) -> np.ndarray:
    """
    Creates a simulated cube as a fallback when extraction fails.
        FIX: Returns (H, W, C) instead of (C, H, W)
    Args:
        lat: Latitude of the center
        lon: Longitude of the center
        window_size: Window size in degrees
    Returns:
        Simulated numpy cube (40, 40, 4)
    """
    logger.warning("Creating a simulated cube as a fallback")

    # Creates realistic patterns based on location.
    H, W = 40, 40

    # 1. SST (Sea Surface Temperature)
    # Warmer in the center, cooler at the edges
    y, x = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W))
    r = np.sqrt(x**2 + y**2)
    sst = 27 + 2 * np.exp(-r**2) + 0.5 * \
        np.random.randn(H, W)  # ~27°C in the center

    # 2. MSLP (Sea Level Pressure)
    # Lowest in the center (hurricane)
    mslp = 1010 - 20 * np.exp(-r**2) + 3 * \
        np.random.randn(H, W)  # ~990 hPa at the center

    # 3. U10 (Zonal wind) - Cyclonic pattern
    u10 = -10 * y * np.exp(-r**2) + 2 * np.random.randn(H, W)

    # 4. V10 (Southern wind)
    v10 = 10 * x * np.exp(-r**2) + 2 * np.random.randn(H, W)

    # CORRECTION: Stacks in (H, W, C) format
    cube = np.stack([sst, mslp, u10, v10], axis=-1)

    logger.info(f"✅ Simulated cube created: shape {cube.shape}")
    return cube


def create_simulated_variable(var_name: str, shape: tuple) -> np.ndarray:
    """
    Creates simulated data for a specific variable.

    Args:
        var_name: Variable name
        shape: Desired shape (H, W)

    Returns:
        Numpy array with simulated data
    """
    H, W = shape

    if var_name == 'sst':
        return 27 + 2 * np.random.randn(H, W)  # ~27°C ± 2
    elif var_name == 'mslp':
        return 1010 + 5 * np.random.randn(H, W)  # ~1010 hPa ± 5
    elif var_name == 'u10':
        return 10 * np.random.randn(H, W)  # Wind U ~0 ± 10 m/s
    elif var_name == 'v10':
        return 10 * np.random.randn(H, W)  # Wind V ~0 ± 10 m/s
    else:
        return np.zeros((H, W), dtype=np.float32)


def locate_era5_files(era5_files_list, era5_dir):
    """
    Locates ERA5 files in multiple possible locations.

    Args:
        era5_files_list: List of file paths
        era5_dir: Directory expected by the pipeline

    Returns:
        List of actual paths found
    """
    actual_files = []

    for file_path in era5_files_list:
        # If it already exists on the expected path
        if file_path.exists():
            actual_files.append(file_path)
            continue

        # If it doesn't exist, search in data/era5/
        alt_path = Path("data/era5") / file_path.name
        if alt_path.exists():
            logger.info(f"Found {file_path.name} in data/era5/")
            actual_files.append(alt_path)
            continue

        # If you still haven't found it, look in the project's root directory.
        alt_path_2 = Path.cwd() / file_path.name
        if alt_path_2.exists():
            logger.info(f"Found {file_path.name} in the root directory")
            actual_files.append(alt_path_2)
            continue

        logger.error(f"File not found: {file_path.name}")

    return actual_files
