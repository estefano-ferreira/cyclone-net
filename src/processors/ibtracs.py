from __future__ import annotations

"""
CycloneNet V2.1 — IBTrACS event-list processor.

This module builds the scientific event list used by the preprocessing stage.
It reads the original IBTrACS CSV in read-only mode, standardizes key columns,
computes rapid-intensification targets, and writes a clean event table.

Scientific intent
-----------------
The event list provides storm-center position and best-track intensity metadata
at 6-hourly cadence. It is intentionally limited to information required for
hindcast sample extraction and target generation.

Output schema
-------------
Minimum required output columns:
- sid
- storm_name
- name                      (legacy alias)
- basin
- timestamp                 (ISO-like pandas datetime column serialized to CSV)
- datetime                  (legacy string format: YYYYmmdd HHMM)
- lat
- lon
- wind_kt
- pressure_mb
- dv12_kt
- dv24_kt
- ri_label

Scientific notes
----------------
- Rapid Intensification (RI) follows the classic threshold:
  ΔV24 >= 30 kt
- The expected cadence is 6 hours: 00, 06, 12, 18 UTC
- Future-target rows lacking dv12/dv24 are removed
- This step does NOT modify ERA5 or any spatial fields
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.downloaders.ibtracs import download_ibtracs
from src.processors.ri_labeling import add_wind_deltas, label_ri
from src.utils.config import cfg_get

logger = logging.getLogger(__name__)


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first matching column name from a list of candidates."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _standardize_longitude(lon: pd.Series) -> pd.Series:
    """Normalize longitude to the [-180, 180) convention.

    This keeps the event list consistent with downstream geospatial handling.
    """
    lon = pd.to_numeric(lon, errors="coerce")
    return ((lon + 180.0) % 360.0) - 180.0


def _clean_text_column(series: pd.Series, default: str = "") -> pd.Series:
    """Convert a text-like column to a safe string series.

    NA is handled BEFORE stringification: `astype(str)` turns NaN into the
    literal string "nan", which would poison keys like `sid`. The trailing
    replace stays as a second line of defense against stringified NA forms
    that arrive already encoded in the source.
    """
    out = series.fillna(default).astype(str)
    out = out.replace({"nan": default, "None": default, "<NA>": default})
    return out


def build_event_list(
    ibtracs_csv: Path,
    out_csv: Path,
    basin_filter: Optional[str] = None,
    min_wind_kt: Optional[float] = None,
    year_range: Optional[Tuple[int, int]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    ri_threshold_kt_24h: float = 30.0,
    drop_undefined: bool = False,
) -> None:
    """Build the CycloneNet event list from IBTrACS.

    Parameters
    ----------
    ibtracs_csv
        Path to the downloaded IBTrACS CSV file.
    out_csv
        Destination path for the standardized event list.
    basin_filter
        Optional substring filter for basin names/codes. NOTE: fixed 2026-07-16
        -- the file is now read with keep_default_na=False so the literal
        basin code 'NA' (North Atlantic) survives parsing.
    min_wind_kt
        Optional minimum best-track wind threshold in knots.
    year_range
        Optional inclusive year range (start_year, end_year).
    bbox
        Optional (N, W, S, E) bounding box in degrees; events whose center
        falls outside are dropped. Should match ``download.spatial_subset``
        so the event list only contains extractable events.
    ri_threshold_kt_24h
        RI threshold in knots over 24 hours.
    drop_undefined
        If False (default), keep rows with NULL labels (strict-temporal partners
        do not exist). If True, drop rows without both dv12_kt and dv24_kt (old
        positional-semantics behavior for legacy builds).

    Raises
    ------
    ValueError
        If required IBTrACS columns cannot be resolved.
    """
    ibtracs_csv = Path(ibtracs_csv)
    out_csv = Path(out_csv)

    logger.info("Reading IBTrACS file: %s", ibtracs_csv)
    # IBTrACS uses 'NA' as the North Atlantic basin code; pandas' default
    # na_values converts it to NaN. Any future read of IBTrACS MUST pass
    # keep_default_na=False. SUBBASIN has the same collision (96,909 rows)
    # and is not currently consumed -- same rule applies if it ever is.
    # IBTrACS encodes missing fields as a single space (verified 2026-07-16);
    # numeric columns below go through pd.to_numeric(errors="coerce"), which
    # is unaffected by this change.
    df = pd.read_csv(ibtracs_csv, low_memory=False, keep_default_na=False, na_values=[" "])

    # ------------------------------------------------------------------
    # Resolve source columns (IBTrACS naming can vary across releases)
    # ------------------------------------------------------------------
    col_sid = _resolve_column(df, ["SID", "sid"])
    col_name = _resolve_column(df, ["NAME", "name"])
    col_basin = _resolve_column(df, ["BASIN", "basin"])
    col_time = _resolve_column(df, ["ISO_TIME", "time"])
    col_lat = _resolve_column(df, ["LAT", "lat"])
    col_lon = _resolve_column(df, ["LON", "lon"])
    col_wind = _resolve_column(df, ["USA_WIND", "wind", "WIND"])
    col_pres = _resolve_column(df, ["USA_PRES", "pressure", "PRES"])

    required_map = {
        "sid": col_sid,
        "time": col_time,
        "lat": col_lat,
        "lon": col_lon,
        "wind": col_wind,
    }
    missing = [key for key, value in required_map.items() if value is None]
    if missing:
        raise ValueError(
            f"IBTrACS column mapping failed. Missing required fields: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # ------------------------------------------------------------------
    # Build canonical output table
    # ------------------------------------------------------------------
    out = pd.DataFrame()

    out["sid"] = _clean_text_column(df[col_sid], default="")
    out["storm_name"] = (
        _clean_text_column(df[col_name], default="") if col_name is not None else ""
    )
    # Preserve the legacy alias used by older parts of the pipeline.
    out["name"] = out["storm_name"]

    out["basin"] = (
        _clean_text_column(df[col_basin], default="") if col_basin is not None else ""
    )

    out["timestamp"] = pd.to_datetime(df[col_time], errors="coerce")
    out["lat"] = pd.to_numeric(df[col_lat], errors="coerce")
    out["lon"] = _standardize_longitude(df[col_lon])

    # Best-track wind must be in knots for RI labeling consistency.
    out["wind_kt"] = pd.to_numeric(df[col_wind], errors="coerce")

    out["pressure_mb"] = (
        pd.to_numeric(df[col_pres], errors="coerce") if col_pres is not None else pd.NA
    )

    # ------------------------------------------------------------------
    # Drop unusable rows before scientific target generation
    # ------------------------------------------------------------------
    out = out.dropna(subset=["timestamp", "lat", "lon", "wind_kt"]).copy()

    # Keep only the standard 6-hourly analysis times expected by the pipeline.
    out["hour"] = out["timestamp"].dt.hour
    out = out[out["hour"].isin([0, 6, 12, 18])].copy()
    out = out.drop(columns=["hour"])

    # Optional basin filter.
    if basin_filter:
        out = out[out["basin"].str.contains(basin_filter, na=False)].copy()

    # Optional minimum wind filter.
    if min_wind_kt is not None:
        out = out[out["wind_kt"] >= float(min_wind_kt)].copy()

    # Optional spatial bounding-box filter (N, W, S, E) — keeps the event list
    # consistent with the ERA5 download area so every listed event is extractable.
    if bbox is not None:
        north, west, south, east = (float(v) for v in bbox)
        out = out[
            (out["lat"] <= north) & (out["lat"] >= south)
            & (out["lon"] >= west) & (out["lon"] <= east)
        ].copy()

    # Optional inclusive year filter.
    if year_range is not None:
        start_year, end_year = year_range
        out = out[
            (out["timestamp"].dt.year >= int(start_year))
            & (out["timestamp"].dt.year <= int(end_year))
        ].copy()

    # Sort before computing forward deltas.
    out = out.sort_values(["sid", "timestamp"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Generate scientific targets
    # ------------------------------------------------------------------
    # dv12_kt and dv24_kt are continuous targets.
    out = add_wind_deltas(out)

    # RI label uses the standard 24-hour threshold.
    out = label_ri(out, ri_threshold_kt_24h=float(ri_threshold_kt_24h))

    # Legacy datetime string expected by parts of the older codebase.
    out["datetime"] = out["timestamp"].dt.strftime("%Y%m%d %H%M")

    # Optionally remove rows without future intensity targets.
    # Default (drop_undefined=False) keeps NULL-labeled rows for later filtering.
    # For legacy workflows, drop_undefined=True reproduces old behavior.
    if drop_undefined:
        out = out.dropna(subset=["dv12_kt", "dv24_kt"]).copy()

    # Reorder columns for readability and reproducibility.
    # NOTE: wind_kt_shift_* columns are NOT included (deprecated positional semantics).
    preferred_order = [
        "sid",
        "storm_name",
        "name",
        "basin",
        "timestamp",
        "datetime",
        "lat",
        "lon",
        "wind_kt",
        "pressure_mb",
        "dv12_kt",
        "dv24_kt",
        "ri_label",
    ]
    remaining = [c for c in out.columns if c not in preferred_order]
    out = out[preferred_order + remaining]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    logger.info("Wrote event list: %s | rows=%d", out_csv, len(out))
    logger.info(
        "Event-list summary | storms=%d | RI positives=%d | years=%s-%s",
        out["sid"].nunique(),
        int(out["ri_label"].sum()) if "ri_label" in out.columns else 0,
        int(out["timestamp"].dt.year.min()) if len(out) else -1,
        int(out["timestamp"].dt.year.max()) if len(out) else -1,
    )


def run_prepare(cfg: dict, force: bool = False) -> None:
    """Entrypoint for the prepare command.

    Workflow
    --------
    1. Ensure IBTrACS is downloaded
    2. Build the standardized event list
    """
    ibtracs_path = download_ibtracs(cfg, force_download=force)

    out_csv = Path(cfg_get(cfg, "paths.event_list", "./data/event_list_augmented.csv"))
    basin_filter = cfg_get(cfg, "data.basin", None)
    min_wind_kt = cfg_get(cfg, "data.min_wind_kt", None)
    ri_threshold = float(cfg_get(cfg, "labels.ri_threshold_kt_24h", 30.0))

    year_range_cfg = cfg_get(cfg, "download.years", None)
    year_range: Optional[Tuple[int, int]] = None
    if year_range_cfg is not None and len(year_range_cfg) == 2:
        year_range = (int(year_range_cfg[0]), int(year_range_cfg[1]))

    bbox_cfg = cfg_get(cfg, "download.spatial_subset", None)
    bbox: Optional[Tuple[float, float, float, float]] = None
    if bbox_cfg is not None and len(bbox_cfg) == 4:
        bbox = tuple(float(v) for v in bbox_cfg)  # (N, W, S, E)

    build_event_list(
        ibtracs_csv=Path(ibtracs_path),
        out_csv=out_csv,
        basin_filter=basin_filter,
        min_wind_kt=min_wind_kt,
        year_range=year_range,
        bbox=bbox,
        ri_threshold_kt_24h=ri_threshold,
    )

    logger.info("Event list saved to %s", out_csv)