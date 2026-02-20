from pathlib import Path
from typing import Optional, Tuple
import re
import pandas as pd
from src.utils.config import CONFIG, cfg_get


def _resolve_nc_path(raw_dir: Path, fname: str, basin: str) -> Tuple[Optional[Path], Optional[pd.Timestamp]]:
    """
    Resolve a NetCDF file for a given expected daily filename.

    This is config-driven and robust:
    - Tries both CONFIG paths: raw_data and daily_data.
    - Tries common filename variants if naming is inconsistent.
    - Falls back to monthly batch file if daily is not found (legacy mode).

    Returns:
        (path, inferred_timestamp_for_monthly) where timestamp is only needed
        if the returned path is a monthly file requiring time selection.
    """
    # Primary directories from config (single source of truth)
    raw_base = Path(cfg_get(CONFIG, "paths.raw_data", raw_dir)).resolve()
    daily_base = Path(cfg_get(CONFIG, "paths.daily_data",
                      raw_base / "daily")).resolve()

    # Ensure filename only (no accidental subpaths)
    fname = Path(fname).name

    # Build candidate filenames (handle naming mismatches)
    # Canonical expected: era5_YYYY_MM_DD_HHMM.nc (your event list uses HH00) :contentReference[oaicite:2]{index=2}
    candidates = [fname]

    # Try variants: HH00 -> HH0000, HH_00, etc (if your files were saved differently)
    # Example: era5_2002_07_24_1200.nc -> try era5_2002_07_24_12_00.nc
    m = re.match(r"^era5_(\d{4})_(\d{2})_(\d{2})_(\d{2})(\d{2})\.nc$", fname)
    if m:
        Y, M, D, hh, mm = m.group(1), m.group(
            2), m.group(3), m.group(4), m.group(5)
        candidates += [
            f"era5_{Y}_{M}_{D}_{hh}_{mm}.nc",
            f"era5_{Y}_{M}_{D}_{hh}{mm}00.nc",
            f"era5_{Y}{M}{D}_{hh}{mm}.nc",
            f"{Y}{M}{D}_{hh}{mm}.nc",
        ]

    # 1) Try daily files in both directories
    for base in (raw_base, daily_base):
        for cn in candidates:
            p = base / cn
            if p.exists():
                return p, None  # daily file: no time selection required

    # 2) Monthly fallback: if fname encodes datetime, try era5_YYYY_MM_{basin}.nc
    parts = fname.replace(".nc", "").split("_")
    if len(parts) >= 5:
        year, month, day, hm = parts[1], parts[2], parts[3], parts[4]
        monthly = raw_base / f"era5_{year}_{month}_{basin}.nc"
        if monthly.exists():
            ts = pd.to_datetime(f"{year}-{month}-{day} {hm[:2]}:{hm[2:]}")
            return monthly, ts

    return None, None
