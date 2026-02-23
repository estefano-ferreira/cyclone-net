from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd

from src.processors.ri_labeling import label_ri, add_wind_deltas

logger = logging.getLogger(__name__)

def build_event_list(ibtracs_csv: Path, out_csv: Path, basin_filter: str | None = None, min_wind_kt: float | None = None, ri_threshold_kt_24h: float = 30.0) -> None:
    """Build an event list from IBTrACS best-track file.

    Output schema (minimum):
      sid, name, basin, datetime (YYYYmmdd HHMM), lat, lon, wind_kt, pressure_mb, ri_label, dv12_kt, dv24_kt

    Notes:
      - Designed for 6-hour cadence (00/06/12/18Z).
      - This step DOES NOT touch any ERA5 NetCDF files.
    """
    df = pd.read_csv(ibtracs_csv, low_memory=False)

    # Heuristic column mapping (IBTrACS varies)
    col_sid = "SID" if "SID" in df.columns else "sid"
    col_name = "NAME" if "NAME" in df.columns else ("name" if "name" in df.columns else None)
    col_basin = "BASIN" if "BASIN" in df.columns else ("basin" if "basin" in df.columns else None)
    col_time = "ISO_TIME" if "ISO_TIME" in df.columns else ("time" if "time" in df.columns else None)
    col_lat = "LAT" if "LAT" in df.columns else "lat"
    col_lon = "LON" if "LON" in df.columns else "lon"
    col_wind = "USA_WIND" if "USA_WIND" in df.columns else ("wind" if "wind" in df.columns else None)
    col_pres = "USA_PRES" if "USA_PRES" in df.columns else ("pressure" if "pressure" in df.columns else None)

    if col_time is None or col_wind is None:
        raise ValueError("IBTrACS columns not found for time or wind. Please map columns explicitly.")

    out = pd.DataFrame()
    out["sid"] = df[col_sid].astype(str)
    out["name"] = df[col_name].astype(str) if col_name else ""
    out["basin"] = df[col_basin].astype(str) if col_basin else ""
    out["timestamp"] = pd.to_datetime(df[col_time], errors="coerce")
    out["lat"] = pd.to_numeric(df[col_lat], errors="coerce")
    out["lon"] = pd.to_numeric(df[col_lon], errors="coerce")
    out["wind_kt"] = pd.to_numeric(df[col_wind], errors="coerce")  # knots
    out["pressure_mb"] = pd.to_numeric(df[col_pres], errors="coerce") if col_pres else pd.NA

    out = out.dropna(subset=["timestamp", "lat", "lon", "wind_kt"]).copy()

    # Filter to 6-hourly records (most IBTrACS is 6h anyway)
    out["hour"] = out["timestamp"].dt.hour
    out = out[out["hour"].isin([0, 6, 12, 18])].copy()
    out = out.drop(columns=["hour"])

    if basin_filter:
        out = out[out["basin"].str.contains(basin_filter, na=False)].copy()
    if min_wind_kt is not None:
        out = out[out["wind_kt"] >= float(min_wind_kt)].copy()

    out = out.sort_values(["sid", "timestamp"]).reset_index(drop=True)

    out = add_wind_deltas(out)
    out = label_ri(out, ri_threshold_kt_24h=ri_threshold_kt_24h)

    # Create legacy 'datetime' used by your earlier pipeline: YYYYmmdd HHMM
    out["datetime"] = out["timestamp"].dt.strftime("%Y%m%d %H%M")

    # Remove rows lacking future targets (dv12/dv24 require future steps)
    out = out.dropna(subset=["dv12_kt", "dv24_kt"]).copy()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    logger.info(f"Wrote event list: {out_csv} rows={len(out)}")
