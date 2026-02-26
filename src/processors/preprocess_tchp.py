# src/processors/preprocess_tchp.py
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter, maximum_filter
from tqdm import tqdm

from src.utils.config import cfg_get
from src.utils.tchp_utils import get_tchp_file_path

logger = logging.getLogger(__name__)


# ----------------------------
# Small utilities (auditable)
# ----------------------------

def _to_timestamp_naive_utc(value: Any) -> pd.Timestamp:
    """
    Convert arbitrary timestamp input into tz-naive UTC pandas Timestamp.

    Why:
        - JSON timestamps may include 'Z' or offsets.
        - xarray indices are frequently tz-naive; tz-aware indexing can error.
    """
    ts = pd.to_datetime(value, utc=True)
    # tz_convert(None) => tz-naive in UTC
    return ts.tz_convert(None)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two points (degrees) in kilometers.
    """
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def _json_dump_atomic(path: Path, obj: Dict[str, Any]) -> None:
    """
    Write JSON deterministically and safely.
    """
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    tmp.replace(path)


def _resolve_lat_lon_coord_names(ds: xr.Dataset) -> Tuple[str, str]:
    """
    Resolve latitude/longitude coordinate names across common conventions.
    """
    lat_candidates = ["lat", "latitude", "Latitude", "nav_lat"]
    lon_candidates = ["lon", "longitude", "Longitude", "nav_lon"]

    lat_name = next((n for n in lat_candidates if n in ds.coords), None)
    lon_name = next((n for n in lon_candidates if n in ds.coords), None)

    if lat_name is None or lon_name is None:
        raise KeyError(f"Latitude/longitude coordinates not found. coords={list(ds.coords)}")

    return lat_name, lon_name


def _coerce_lon_to_dataset_convention(lon: float, lon_min: float, ds_lon_min: float) -> Tuple[float, float, float]:
    """
    Coerce lon / lon bounds to match dataset convention if dataset uses 0..360.

    Returns:
        (lon_adj, lon_min_adj, lon_max_adj)
    """
    # If dataset longitudes are 0..360 but requested lon bounds are negative, wrap them.
    if ds_lon_min >= 0.0 and lon_min < 0.0:
        lon_adj = lon + 360.0
        lon_min_adj = lon_min + 360.0
        # lon_max will be handled by caller (lon_max = lon + window)
        return lon_adj, lon_min_adj, None  # lon_max computed outside
    return lon, lon_min, None


def _nan_safe_gaussian_filter(x: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian smoothing that ignores NaNs (by normalizing with a mask).

    This is important for scientific robustness: we do not want NaNs to
    artificially create peaks or bias local maxima.
    """
    if sigma <= 0:
        return x

    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x).astype(float)

    # Replace NaNs with 0 for convolution, then normalize by convolved mask.
    x0 = np.where(np.isfinite(x), x, 0.0)

    num = gaussian_filter(x0, sigma=sigma)
    den = gaussian_filter(mask, sigma=sigma)

    out = np.full_like(num, np.nan, dtype=float)
    valid = den > 1e-12
    out[valid] = num[valid] / den[valid]
    return out


@dataclass(frozen=True)
class TCHPWindowResult:
    tchp_2d: np.ndarray
    lats_1d: np.ndarray
    lons_1d: np.ndarray
    used_time: pd.Timestamp
    var_name: str
    lat_name: str
    lon_name: str


# ----------------------------
# Core scientific extraction
# ----------------------------

def load_tchp_window(
    tchp_path: Path,
    target_time: pd.Timestamp,
    center_lat: float,
    center_lon: float,
    window_deg: float,
    var_candidates: Optional[List[str]] = None,
) -> Optional[TCHPWindowResult]:
    """
    Load a local spatial window (lat/lon box) from a yearly TCHP NetCDF.

    Scientific/robust behavior:
      - Select nearest time.
      - Resolve coordinate names dynamically.
      - Handle longitude convention (0..360 vs -180..180).
      - Force output to 2D (lat, lon) and validate shapes.

    Returns:
        TCHPWindowResult or None if file missing or window empty.
    """
    if not tchp_path.exists():
        return None

    if var_candidates is None:
        var_candidates = ["tchp", "Tropical_Cyclone_Heat_Potential", "TCHP"]

    try:
        ds = xr.open_dataset(tchp_path)

        # Time selection (nearest)
        # Always use tz-naive UTC to prevent timezone-indexing issues.
        ts = _to_timestamp_naive_utc(target_time)
        ds_t = ds.sel(time=ts, method="nearest")

        # Resolve coords
        lat_name, lon_name = _resolve_lat_lon_coord_names(ds_t)

        # Compute bounds
        lat_min = center_lat - window_deg
        lat_max = center_lat + window_deg
        lon_min = center_lon - window_deg
        lon_max = center_lon + window_deg

        ds_lon_min = float(ds_t[lon_name].min().values)
        if ds_lon_min >= 0.0 and lon_min < 0.0:
            lon_min += 360.0
            lon_max += 360.0

        # Subset region
        ds_region = ds_t.sel(
            **{
                lat_name: slice(lat_min, lat_max),
                lon_name: slice(lon_min, lon_max),
            }
        )

        if ds_region.sizes.get(lat_name, 0) == 0 or ds_region.sizes.get(lon_name, 0) == 0:
            ds.close()
            return None

        # Resolve variable
        var_name = next((v for v in var_candidates if v in ds_region.data_vars), None)
        if var_name is None:
            raise KeyError(f"No TCHP variable found. vars={list(ds_region.data_vars)} file={tchp_path}")

        tchp = np.asarray(ds_region[var_name].values)
        tchp = np.squeeze(tchp)  # remove size-1 dims (e.g., time)

        if tchp.ndim != 2:
            raise ValueError(f"Expected 2D (lat, lon) after selection, got shape={tchp.shape} file={tchp_path}")

        lats = np.asarray(ds_region[lat_name].values)
        lons = np.asarray(ds_region[lon_name].values)

        # Used time in the dataset (auditable)
        used_time = pd.to_datetime(ds_region["time"].values).to_pydatetime()
        used_time = pd.Timestamp(used_time)

        ds.close()
        return TCHPWindowResult(
            tchp_2d=tchp,
            lats_1d=lats,
            lons_1d=lons,
            used_time=used_time,
            var_name=var_name,
            lat_name=lat_name,
            lon_name=lon_name,
        )

    except Exception as e:
        logger.warning(f"Error loading TCHP window from {tchp_path}: {e}")
        return None


def find_peak_location(
    tchp_2d: np.ndarray,
    lats_1d: np.ndarray,
    lons_1d: np.ndarray,
    smooth_sigma: float = 1.0,
    local_window_px: int = 3,
) -> Optional[Tuple[float, float, float]]:
    """
    Find a robust local peak in TCHP near the storm.

    Method:
      - NaN-safe Gaussian smoothing
      - Local maxima detection (maximum_filter)
      - Pick the highest local maximum
      - Fall back to global nanmax if no local maxima found

    Returns:
        (peak_lat, peak_lon, peak_value) or None if all values are NaN.
    """
    x = np.asarray(tchp_2d, dtype=float)

    if not np.isfinite(x).any():
        return None

    x_smooth = _nan_safe_gaussian_filter(x, sigma=smooth_sigma)

    # If smoothing produced all-NaN (can happen if window is all NaN)
    if not np.isfinite(x_smooth).any():
        return None

    # Local maxima mask
    local_max = maximum_filter(x_smooth, size=int(local_window_px)) == x_smooth
    peaks = np.argwhere(local_max & np.isfinite(x_smooth))

    if peaks.size == 0:
        i, j = np.unravel_index(int(np.nanargmax(x_smooth)), x_smooth.shape)
    else:
        vals = x_smooth[peaks[:, 0], peaks[:, 1]]
        best_idx = int(np.nanargmax(vals))
        i, j = int(peaks[best_idx, 0]), int(peaks[best_idx, 1])

    peak_lat = float(lats_1d[i])
    peak_lon = float(lons_1d[j])
    peak_val = float(x[i, j]) if np.isfinite(x[i, j]) else float(x_smooth[i, j])

    return peak_lat, peak_lon, peak_val


def sample_at_center(
    tchp_2d: np.ndarray,
    lats_1d: np.ndarray,
    lons_1d: np.ndarray,
    center_lat: float,
    center_lon: float,
) -> Optional[float]:
    """
    Sample TCHP at the nearest grid point to the storm center.
    """
    x = np.asarray(tchp_2d, dtype=float)
    if not np.isfinite(x).any():
        return None

    i = int(np.argmin(np.abs(lats_1d - center_lat)))
    j = int(np.argmin(np.abs(lons_1d - center_lon)))
    val = float(x[i, j]) if np.isfinite(x[i, j]) else None
    return val


def compute_window_stats(tchp_2d: np.ndarray) -> Dict[str, Any]:
    """
    Compute auditable window statistics (NaN-aware).
    """
    x = np.asarray(tchp_2d, dtype=float)
    total = x.size
    nan_count = int(np.isnan(x).sum())
    finite = x[np.isfinite(x)]

    if finite.size == 0:
        return {
            "n_total": total,
            "nan_fraction": 1.0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    return {
        "n_total": total,
        "nan_fraction": float(nan_count / total),
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x)),
    }


def qc_physical_plausibility(
    stats: Dict[str, Any],
    qc_min: float,
    qc_max: float,
    nan_fraction_max: float,
) -> Dict[str, Any]:
    """
    Minimal physical plausibility checks.

    Notes:
      - We do NOT hardcode strict scientific bounds here; bounds are configurable.
      - Default bounds are conservative to flag obvious corrupt data.
    """
    flags: Dict[str, Any] = {
        "nan_fraction_ok": stats["nan_fraction"] is not None and stats["nan_fraction"] <= nan_fraction_max,
        "range_ok": True,
    }

    if stats["max"] is None or stats["min"] is None:
        flags["range_ok"] = False
        return flags

    # Conservative check: TCHP should not be negative in standard definitions.
    # Upper bound is configurable to prevent silently accepting corrupted outputs.
    flags["range_ok"] = (stats["min"] >= qc_min) and (stats["max"] <= qc_max)
    return flags


# ----------------------------
# Pipeline step (metadata + audit)
# ----------------------------

def _select_tchp_file_for_year(tchp_dir: Path, year: int) -> Optional[Tuple[str, Path]]:
    """
    Select the best available TCHP file for a given year.

    Rationale:
      - Your downloader can produce different sources (noaa/aoml/copernicus).
      - The preprocessing must be robust and auditable: try in a fixed order.
    """
    # Prefer ERDDAP/NOAA output if present, then legacy FTP (AOML), then Copernicus.
    for src in ["noaa", "aoml", "copernicus"]:
        p = get_tchp_file_path(tchp_dir, year, src)
        if p.exists():
            return src, p
    return None


def add_tchp_to_metadata(cfg: Dict[str, Any]) -> None:
    """
    For each event JSON in interim directory:
      - load corresponding TCHP year file
      - extract a local window around storm center
      - find peak location + value
      - save results and audit fields to JSON metadata

    This step is for external validation only (no leakage into model inputs).
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

    # Configurable parameters (conservative defaults)
    window_deg = float(cfg_get(cfg, "download.tchp.window_deg", 5.0))
    smooth_sigma = float(cfg_get(cfg, "download.tchp.peak_smooth_sigma", 1.0))
    local_window_px = int(cfg_get(cfg, "download.tchp.peak_local_window_px", 3))

    qc_min = float(cfg_get(cfg, "download.tchp.qc.value_min", 0.0))
    qc_max = float(cfg_get(cfg, "download.tchp.qc.value_max", 300.0))
    nan_fraction_max = float(cfg_get(cfg, "download.tchp.qc.nan_fraction_max", 0.75))

    # Optional audit log (JSON lines)
    audit_path = Path(cfg_get(cfg, "paths.logs_dir", "./outputs/logs")).resolve() / "preprocess_tchp_audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    updated = 0
    skipped = 0
    missing_tchp = 0
    invalid = 0

    for json_path in tqdm(json_files, desc="Adding TCHP to metadata"):
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # If already computed, skip (idempotent behavior)
        if meta.get("tchp_peak_lat") is not None and meta.get("tchp_peak_lon") is not None:
            skipped += 1
            continue

        # Required fields
        if meta.get("timestamp") is None or meta.get("center_lat") is None or meta.get("center_lon") is None:
            logger.debug(f"Skipping {json_path.name}: missing timestamp/center")
            continue

        ts = _to_timestamp_naive_utc(meta["timestamp"])
        center_lat = float(meta["center_lat"])
        center_lon = float(meta["center_lon"])
        year = int(ts.year)

        if year < 1993:
            missing_tchp += 1
            continue

        sel = _select_tchp_file_for_year(tchp_dir, year)
        if sel is None:
            missing_tchp += 1
            continue

        src, tchp_file = sel

        win = load_tchp_window(
            tchp_path=tchp_file,
            target_time=ts,
            center_lat=center_lat,
            center_lon=center_lon,
            window_deg=window_deg,
        )
        if win is None:
            missing_tchp += 1
            continue

        stats = compute_window_stats(win.tchp_2d)
        qc = qc_physical_plausibility(stats, qc_min=qc_min, qc_max=qc_max, nan_fraction_max=nan_fraction_max)

        peak = find_peak_location(
            tchp_2d=win.tchp_2d,
            lats_1d=win.lats_1d,
            lons_1d=win.lons_1d,
            smooth_sigma=smooth_sigma,
            local_window_px=local_window_px,
        )

        center_val = sample_at_center(
            tchp_2d=win.tchp_2d,
            lats_1d=win.lats_1d,
            lons_1d=win.lons_1d,
            center_lat=center_lat,
            center_lon=center_lon,
        )

        if peak is None:
            # If everything is NaN, treat as invalid (auditable)
            invalid += 1
            meta["tchp_audit"] = {
                "status": "invalid_all_nan",
                "source": src,
                "file": str(tchp_file),
                "requested_time": str(ts),
                "used_time": str(win.used_time),
                "window_deg": window_deg,
                "coords": {"lat_name": win.lat_name, "lon_name": win.lon_name},
                "var_name": win.var_name,
                "stats": stats,
                "qc": qc,
            }
            _json_dump_atomic(json_path, meta)
            _append_audit_line(audit_path, meta, json_path.name)
            continue

        peak_lat, peak_lon, peak_val = peak
        peak_dist_km = _haversine_km(center_lat, center_lon, peak_lat, peak_lon)

        # Update metadata (auditable, explicit fields)
        meta["tchp_peak_lat"] = float(peak_lat)
        meta["tchp_peak_lon"] = float(peak_lon)
        meta["tchp_peak_value"] = float(peak_val)
        meta["tchp_center_value"] = float(center_val) if center_val is not None else None
        meta["tchp_peak_distance_km"] = float(peak_dist_km)

        # Provide a structured audit block
        meta["tchp_audit"] = {
            "status": "ok" if (qc["nan_fraction_ok"] and qc["range_ok"]) else "qc_flagged",
            "source": src,
            "file": str(tchp_file),
            "requested_time": str(ts),
            "used_time": str(win.used_time),
            "window_deg": float(window_deg),
            "peak_method": {
                "smooth_sigma": float(smooth_sigma),
                "local_window_px": int(local_window_px),
                "nan_safe_smoothing": True,
            },
            "coords": {"lat_name": win.lat_name, "lon_name": win.lon_name},
            "var_name": win.var_name,
            "stats": stats,
            "qc": {
                **qc,
                "thresholds": {
                    "value_min": float(qc_min),
                    "value_max": float(qc_max),
                    "nan_fraction_max": float(nan_fraction_max),
                },
            },
        }

        _json_dump_atomic(json_path, meta)
        _append_audit_line(audit_path, meta, json_path.name)
        updated += 1

    logger.info(
        "TCHP enrichment done: "
        f"updated={updated}, skipped={skipped}, missing_tchp={missing_tchp}, invalid={invalid}. "
        f"audit_log={audit_path}"
    )


def _append_audit_line(audit_path: Path, meta: Dict[str, Any], json_filename: str) -> None:
    """
    Append an auditable per-event summary line to a JSONL file.
    This is useful to review QC flags and extraction behavior at scale.
    """
    line = {
        "event_json": json_filename,
        "event_id": meta.get("event_id"),
        "timestamp": meta.get("timestamp"),
        "center_lat": meta.get("center_lat"),
        "center_lon": meta.get("center_lon"),
        "tchp_peak_lat": meta.get("tchp_peak_lat"),
        "tchp_peak_lon": meta.get("tchp_peak_lon"),
        "tchp_peak_value": meta.get("tchp_peak_value"),
        "tchp_center_value": meta.get("tchp_center_value"),
        "tchp_peak_distance_km": meta.get("tchp_peak_distance_km"),
        "tchp_audit": meta.get("tchp_audit"),
    }
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def run_preprocess_tchp(cfg: Dict[str, Any]) -> None:
    """
    Entrypoint for preprocess-tchp command.
    """
    add_tchp_to_metadata(cfg)