#!/usr/bin/env python3
"""
CycloneNet V2.1 – IBTrACS event list generator with RI labeling.

This module reads the official IBTrACS CSV file (v04r00) and produces:
    - event_list.csv : master list of all storm track points that meet the
      configured region, intensity, and temporal regularity criteria.
    - required_timestamps.csv : list of unique (year, month, day, hour) tuples
      needed for ERA5 download.

Scientific guarantees:
    - No modification of original NetCDF files – this script only reads the
      IBTrACS CSV and writes two derived CSV files.
    - All filtering decisions are config‑driven (via config.yaml) and fully
      traceable.
    - RI labels are added using the `ri_labeling` module with a configurable
      mode (default: event_window, which labels all states within an RI interval).
    - Storms with irregular 6‑hourly spacing are discarded to ensure valid
      RI calculations.
    - Pressure values may be missing (NaN) – this is accepted; they are not
      used for filtering but are retained for metadata.

Author: Estefano Senhor Ferreira
License: CC BY-NC 4.0
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.processors.ri_labeling import label_ri
from src.utils.config import CONFIG, cfg_get

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Configuration helpers
# ----------------------------------------------------------------------


def _get_storm_filter_config() -> dict:
    """Retrieve storm filtering parameters from config."""
    return cfg_get(CONFIG, "storm_filter", {})


def _get_ri_config() -> dict:
    """Retrieve RI labeling parameters from config."""
    return cfg_get(CONFIG, "ri_labeling", {})


def _get_download_years() -> list[int]:
    """Get the year range for filtering."""
    return cfg_get(CONFIG, "download.years", [1980, 2024])

# ----------------------------------------------------------------------
# Core filtering logic (adapted from V1 but made config‑driven)
# ----------------------------------------------------------------------


def filter_storms_by_region_and_intensity(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Keep only storms that satisfy the configured minimum wind and region criteria.

    A storm is retained if:
        - It has at least one record with wind speed >= min_wind_knots.
        - If require_point_in_region is True, that record must lie inside the
          specified bounding box [north, west, south, east].

    Args:
        df: DataFrame with columns 'sid', 'wind_knots', 'lat', 'lon'.
        cfg: Dictionary containing 'min_wind_knots', 'region', and
             'require_point_in_region'.

    Returns:
        Filtered DataFrame containing only the rows of accepted storms.
    """
    min_wind = float(cfg.get("min_wind_knots", 64))
    region = cfg.get("region", [60, -140, 0, -20])
    north, west, south, east = region
    require_in_region = bool(cfg.get("require_point_in_region", True))

    def in_region(row: pd.Series) -> bool:
        return (south <= row["lat"] <= north) and (west <= row["lon"] <= east)

    storm_groups = df.groupby("sid")
    keep_sids = []

    for sid, group in storm_groups:
        strong = group["wind_knots"] >= min_wind
        if not strong.any():
            continue  # storm never reaches minimum intensity

        if require_in_region:
            # Check if at least one strong point is inside the region
            region_mask = group.apply(in_region, axis=1)
            if not (strong & region_mask).any():
                continue

        keep_sids.append(sid)

    logger.info(
        f"Region/intensity filter: kept {len(keep_sids)} out of {len(storm_groups)} storms.")
    return df[df["sid"].isin(keep_sids)].copy()


def has_regular_6h_interval(group: pd.DataFrame) -> bool:
    """
    Check that all consecutive timestamps in a storm group are exactly 6 hours apart.
    """
    if len(group) < 2:
        return True
    diffs = group["timestamp"].diff().dropna()
    # Use .all() on a boolean Series; pd.Timedelta(hours=6) is the expected difference.
    return (diffs == pd.Timedelta(hours=6)).all()


# ----------------------------------------------------------------------
# Main event list generation
# ----------------------------------------------------------------------

def generate_event_list(csv_path: Path) -> pd.DataFrame:
    """
    Generate event_list.csv and required_timestamps.csv from IBTrACS data.

    Steps:
        1. Load IBTrACS CSV, skipping the second row (units row).
        2. Rename columns to standard names (sid, storm_name, timestamp, ...).
        3. Convert columns to appropriate types and drop rows missing essential
           coordinates or wind speed.
        4. Filter by configured year range and synoptic hours (0,6,12,18).
        5. Discard storms that do not have regular 6‑hourly spacing.
        6. Apply region and intensity filters (configurable).
        7. Add RI labels using `ri_labeling.label_ri`.
        8. Construct ERA5 filename pattern for each row.
        9. Save event_list.csv and required_timestamps.csv.

    Args:
        csv_path: Path to the downloaded IBTrACS CSV file.

    Returns:
        DataFrame containing the final event list.
    """
    logger.info(f"Reading IBTrACS from: {csv_path}")

    # ------------------------------------------------------------------
    # 1. Load CSV, skipping the units row (second line)
    # ------------------------------------------------------------------
    try:
        df = pd.read_csv(
            csv_path,
            skiprows=[1],               # skip the units row
            low_memory=False,
            encoding="utf-8",
            dtype={"BASIN": str},
            na_filter=False,             # we will handle missing values explicitly
        )
    except Exception as e:
        logger.error(f"Failed to read IBTrACS CSV: {e}")
        raise

    df.columns = df.columns.str.strip()

    # ------------------------------------------------------------------
    # 2. Rename columns to our internal naming scheme
    # ------------------------------------------------------------------
    column_map = {
        "SID": "sid",
        "NAME": "storm_name",
        "ISO_TIME": "timestamp",
        "LAT": "lat",
        "LON": "lon",
        "USA_WIND": "wind_knots",
        "USA_PRES": "pressure_mb",
        "BASIN": "basin",
    }
    # Keep only columns that exist in the file
    existing_cols = {k: v for k, v in column_map.items() if k in df.columns}
    df = df[list(existing_cols.keys())].rename(columns=existing_cols)

    # Check that mandatory columns are present
    mandatory = ["sid", "timestamp", "lat", "lon", "wind_knots", "basin"]
    missing = [col for col in mandatory if col not in df.columns]
    if missing:
        raise KeyError(f"IBTrACS CSV is missing required columns: {missing}")

    # ------------------------------------------------------------------
    # 3. Convert numeric columns and drop rows with missing essentials
    # ------------------------------------------------------------------
    for col in ["wind_knots", "lat", "lon", "pressure_mb"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where lat/lon or wind is NaN (these are essential)
    essential = ["wind_knots", "lat", "lon"]
    df = df.dropna(subset=essential).copy()
    logger.info(f"After dropping missing essentials: {len(df)} rows remain.")

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # ------------------------------------------------------------------
    # 4. Year filter
    # ------------------------------------------------------------------
    years = _get_download_years()
    df = df[df["timestamp"].dt.year.between(years[0], years[1])].copy()
    logger.info(f"After year filter {years}: {len(df)} rows.")

    # ------------------------------------------------------------------
    # 5. Keep only synoptic hours (0, 6, 12, 18 UTC)
    # ------------------------------------------------------------------
    df = df[df["timestamp"].dt.hour.isin([0, 6, 12, 18])].copy()
    logger.info(f"After synoptic hour filter: {len(df)} rows.")

    # ------------------------------------------------------------------
    # 6. Ensure regular 6‑hourly spacing per storm
    # ------------------------------------------------------------------
    valid_groups = []
    irregular_storms = 0
    for sid, group in df.groupby("sid", sort=False):
        group = group.sort_values("timestamp")
        if has_regular_6h_interval(group):
            valid_groups.append(group)
        else:
            irregular_storms += 1
            logger.debug(f"Storm {sid} has irregular intervals, discarding.")
    if not valid_groups:
        raise RuntimeError("No storms with regular 6h intervals remain.")
    df = pd.concat(valid_groups, ignore_index=True)
    logger.info(
        f"After regularity check: {len(df)} rows. Discarded {irregular_storms} storms.")

    # ------------------------------------------------------------------
    # 7. Apply region and intensity filters (config‑driven)
    # ------------------------------------------------------------------
    storm_filter_cfg = _get_storm_filter_config()
    df = filter_storms_by_region_and_intensity(df, storm_filter_cfg)

    # ------------------------------------------------------------------
    # 8. Sort for RI labeling
    # ------------------------------------------------------------------
    df = df.sort_values(["sid", "timestamp"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 9. Add RI labels
    # ------------------------------------------------------------------
    ri_cfg = _get_ri_config()
    df["ri_label"] = label_ri(
        df,
        mode=ri_cfg.get("mode", "event_window"),
        delta_v_kt=float(ri_cfg.get("delta_v_kt", 30.0)),
        window_hours=int(ri_cfg.get("window_hours", 24)),
        time_step_hours=int(ri_cfg.get("time_step_hours", 6)),
        sid_col=ri_cfg.get("sid_col", "sid"),
        wind_col=ri_cfg.get("wind_col", "wind_knots"),
    )

    # ------------------------------------------------------------------
    # 10. Add ERA5 filename pattern (used by preprocessor)
    # ------------------------------------------------------------------
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour
    df["nc_filename"] = (
        "era5_"
        + df["year"].astype(str) + "_"
        + df["month"].astype(str).str.zfill(2) + "_"
        + df["day"].astype(str).str.zfill(2) + "_"
        + df["hour"].astype(str).str.zfill(2) + "00.nc"
    )

    # ------------------------------------------------------------------
    # 11. Select final columns for event list
    # ------------------------------------------------------------------
    final_cols = [
        "sid",
        "timestamp",
        "lat",
        "lon",
        "wind_knots",
        "pressure_mb",
        "storm_name",
        "basin",
        "ri_label",
        "nc_filename",
    ]
    # Ensure all columns exist; fill missing with NaN
    for col in final_cols:
        if col not in df.columns:
            df[col] = np.nan

    df_final = df[final_cols].copy()

    # ------------------------------------------------------------------
    # 12. Save event_list.csv
    # ------------------------------------------------------------------
    out_path = Path(
        cfg_get(CONFIG, "paths.event_list", "./data/event_list.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(out_path, index=False)
    logger.info(f"Event list saved: {out_path} ({len(df_final)} samples)")

    # ------------------------------------------------------------------
    # 13. Generate required_timestamps.csv for ERA5 download
    # ------------------------------------------------------------------
    req_ts = df_final[["timestamp"]].copy()
    req_ts["year"] = req_ts["timestamp"].dt.year
    req_ts["month"] = req_ts["timestamp"].dt.month
    req_ts["day"] = req_ts["timestamp"].dt.day
    req_ts["hour"] = req_ts["timestamp"].dt.hour
    req_ts = req_ts[["year", "month", "day", "hour"]
                    ].drop_duplicates().reset_index(drop=True)

    req_path = Path(cfg_get(CONFIG, "paths.raw_data",
                    "./data/raw")) / "required_timestamps.csv"
    req_ts.to_csv(req_path, index=False)
    logger.info(
        f"Required timestamps saved: {req_path} ({len(req_ts)} unique slots)")

    return df_final


# ----------------------------------------------------------------------
# Optional main for testing
# ----------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # For testing, you can call generate_event_list with a known path
    # test_path = Path("data/raw/ibtracs.ALL.list.v04r00.csv")
    # generate_event_list(test_path)
