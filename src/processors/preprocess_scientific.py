from __future__ import annotations

"""CycloneNet — scientific preprocessing (portable, temporally safe, physics-guided ready).

This preprocessor reads monthly ERA5 NetCDF (READ-ONLY) and writes per-event artifacts:
- Cube:  (H, W, T, C) -> data/interim/{event_id}.npy
- Meta:  JSON audit -> data/interim/{event_id}.json
- Grids: {event_id}_lats.npy and {event_id}_lons.npy

Physics-guided additions (required):
- Saves a supervised physical prior map:
    data/interim/{event_id}_fuel_potential.npy   (H,W,T)
  This enables FuelMap supervision (KL alignment) during training.

Scientific guards:
- Temporal integrity: skip events if selected ERA5 times collapse (unique_selected != T)
- Geospatial integrity: skip events if cyclone center does not fall inside the extracted patch
- Fixed shapes: pad/crop ensures stacking is always valid

Anti-leakage:
- This preprocessor does NOT embed external validation products (e.g., TCHP/OHC) as input channels.
  If you compute external energy ground truth, store it separately or as metadata fields (e.g., tchp_max_lat/lon).
"""

import ctypes
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from src.utils.config import cfg_get
from src.physics.fuel_potential import build_fuel_potential, FuelPotentialConfig
from src.physics.diagnostics import compute_diagnostics

logger = logging.getLogger(__name__)


# -----------------------------
# Helpers
# -----------------------------
def ensure_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[0] == 1:
        return a[0]
    if a.ndim == 3 and a.shape[-1] == 1:
        return a[..., 0]
    raise ValueError(f"Expected 2D array, got shape {a.shape}")


def pad_or_crop_2d(a: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    a = ensure_2d(a).astype(np.float32)
    th, tw = target_hw
    h, w = a.shape

    pad_h_before = max(0, (th - h) // 2)
    pad_h_after = max(0, th - h - pad_h_before)
    pad_w_before = max(0, (tw - w) // 2)
    pad_w_after = max(0, tw - w - pad_w_before)
    if pad_h_before or pad_h_after or pad_w_before or pad_w_after:
        a = np.pad(a, ((pad_h_before, pad_h_after),
                   (pad_w_before, pad_w_after)), mode="reflect")

    h, w = a.shape
    if h > th:
        s = (h - th) // 2
        a = a[s:s + th, :]
    if w > tw:
        s = (w - tw) // 2
        a = a[:, s:s + tw]
    if a.shape != (th, tw):
        raise ValueError(
            f"pad_or_crop failed: got {a.shape}, expected {(th, tw)}")
    return a


def normalize_sst_to_kelvin(sst: np.ndarray) -> np.ndarray:
    sst = sst.astype(np.float32)
    if float(np.nanmean(sst)) < 150.0:
        sst = sst + 273.15
    return sst


def normalize_mslp_to_pa(msl: np.ndarray) -> np.ndarray:
    msl = msl.astype(np.float32)
    if float(np.nanmean(msl)) < 2000.0:
        msl = msl * 100.0
    return msl


def _nan_fraction(x: np.ndarray) -> float:
    return float(np.mean(~np.isfinite(x)))


def qc_physical_ranges(
    sst_k: np.ndarray,
    msl_pa: np.ndarray,
    u10: np.ndarray,
    v10: np.ndarray,
    sst_range: Tuple[float, float],
    msl_range: Tuple[float, float],
    wind_abs_max: float,
    max_nan_fraction: float,
) -> Dict[str, Any]:
    flags: Dict[str, Any] = {}
    nf_sst = _nan_fraction(sst_k)
    nf_msl = _nan_fraction(msl_pa)
    nf_u = _nan_fraction(u10)
    nf_v = _nan_fraction(v10)
    flags["nan_fraction_per_channel"] = {
        "sst": nf_sst, "msl": nf_msl, "u10": nf_u, "v10": nf_v}

    if max(nf_sst, nf_msl, nf_u, nf_v) > float(max_nan_fraction):
        raise ValueError(
            f"Too many NaNs (max={max(nf_sst,nf_msl,nf_u,nf_v):.2f})")

    sst_min, sst_max = float(sst_range[0]), float(sst_range[1])
    msl_min, msl_max = float(msl_range[0]), float(msl_range[1])

    flags["sst_range_ok"] = bool(
        np.nanmin(sst_k) >= sst_min and np.nanmax(sst_k) <= sst_max)
    flags["msl_range_ok"] = bool(
        np.nanmin(msl_pa) >= msl_min and np.nanmax(msl_pa) <= msl_max)
    flags["wind_abs_ok"] = bool(np.nanmax(
        np.abs(u10)) <= wind_abs_max and np.nanmax(np.abs(v10)) <= wind_abs_max)

    if not flags["sst_range_ok"]:
        raise ValueError("SST out of physical range")
    if not flags["msl_range_ok"]:
        raise ValueError("MSLP out of physical range")
    if not flags["wind_abs_ok"]:
        raise ValueError("Wind out of physical range")
    return flags


def _windows_short_path(p: Path) -> str:
    try:
        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW.argtypes = [
            ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        GetShortPathNameW.restype = ctypes.c_uint
        out_buf = ctypes.create_unicode_buffer(4096)
        res = GetShortPathNameW(str(p), out_buf, 4096)
        if res > 0 and out_buf.value:
            return out_buf.value
    except Exception:
        pass
    return str(p)


def open_netcdf_safe(path: Path) -> xr.Dataset:
    p = Path(path)
    open_path = _windows_short_path(p) if os.name == "nt" else str(p)
    try:
        return xr.open_dataset(open_path, engine="netcdf4", cache=False)
    except Exception:
        pass
    try:
        return xr.open_dataset(open_path, engine="h5netcdf", cache=False)
    except Exception:
        pass
    return xr.open_dataset(open_path, engine="scipy", cache=False)


def _find_time_name(ds: xr.Dataset) -> str | None:
    candidates = ["time", "valid_time", "Time", "TIME"]
    for name in candidates:
        if name in ds.coords or name in ds.dims or name in ds.variables:
            return name
    for name in list(ds.coords):
        if "time" in name.lower():
            return name
    for name in list(ds.dims):
        if "time" in name.lower():
            return name
    return None


def _select_time_slice(ds: xr.Dataset, target_dt: datetime) -> tuple[xr.Dataset, str | None, int | None, str | None]:
    try:
        ds = xr.decode_cf(ds)
    except Exception:
        pass

    tname = _find_time_name(ds)
    if tname is None:
        return ds, None, None, None

    if tname not in ds.coords and tname in ds.variables:
        try:
            ds = ds.set_coords(tname)
        except Exception:
            pass

    if tname in ds.coords:
        try:
            ds_t = ds.sel({tname: np.datetime64(target_dt)}, method="nearest")
            sel_val = np.asarray(ds_t[tname].values).reshape(-1)[0]
            sel_str = pd.to_datetime(sel_val).strftime("%Y-%m-%d %H:%M")
            idx = None
            try:
                times = pd.to_datetime(ds[tname].values)
                idx = int(np.argmin(np.abs(times - np.datetime64(target_dt))))
            except Exception:
                idx = None
            return ds_t, tname, idx, sel_str
        except Exception:
            pass

    # fallback: argmin on values
    try:
        times = pd.to_datetime(ds[tname].values)
        idx = int(np.argmin(np.abs(times - np.datetime64(target_dt))))
        sel_str = pd.to_datetime(
            ds[tname].values[idx]).strftime("%Y-%m-%d %H:%M")
        ds_t = ds.isel({tname: idx}) if tname in ds.dims else ds
        return ds_t, tname, idx, sel_str
    except Exception:
        pass

    if tname in ds.dims and ds.dims[tname] > 1:
        logger.warning(
            f"Time dimension '{tname}' has no usable coordinate values; using index 0.")
        return ds.isel({tname: 0}), tname, 0, None

    return ds, tname, None, None


def _resolve_var(ds_t: xr.Dataset, preferred: str, fallbacks: List[str]) -> str:
    if preferred in ds_t.data_vars:
        return preferred
    for fb in fallbacks:
        if fb in ds_t.data_vars:
            return fb
    raise KeyError(
        f"No variable named '{preferred}'. Tried { [preferred] + fallbacks }. Vars={list(ds_t.data_vars)}")


def nearest_index(arr_1d: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(arr_1d - value)))


def extract_window_by_index(field2d: np.ndarray, i: int, j: int, size: int) -> np.ndarray:
    h, w = field2d.shape
    half = size // 2
    i0 = max(0, i - half)
    i1 = min(h, i + half + (0 if size % 2 == 0 else 1))
    j0 = max(0, j - half)
    j1 = min(w, j + half + (0 if size % 2 == 0 else 1))
    return pad_or_crop_2d(field2d[i0:i1, j0:j1], (size, size))


def _to_event_id(dt: datetime) -> str:
    return f"era5_{dt.strftime('%Y_%m_%d_%H%M')}"


def _month_file(raw_dir: Path, dt: datetime) -> Path:
    return raw_dir / f"era5_{dt.strftime('%Y_%m')}.nc"


def load_event_list(event_list_path: Path) -> pd.DataFrame:
    df = pd.read_csv(event_list_path)
    if "timestamp" in df.columns:
        df["dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "datetime" in df.columns:
        df["dt"] = pd.to_datetime(
            df["datetime"], format="%Y%m%d %H%M", errors="coerce")
    else:
        raise ValueError(
            "event_list must have a 'timestamp' or 'datetime' column")
    for col in ["lat", "lon"]:
        if col not in df.columns:
            raise ValueError(f"event_list missing required column: {col}")
    return df


@dataclass
class EventMeta:
    event_id: str
    sid: str
    timestamp: str
    storm_name: str
    basin: str
    ri_label: int
    dv12_kt: float | None
    dv24_kt: float | None
    wind_kt: float | None
    pressure_mb: float | None
    center_lat: float
    center_lon: float
    timestamps: List[str]
    cube_shape: List[int]
    channels: List[str]
    units: Dict[str, str]
    qc_flags: Dict[str, Any]
    source_files: List[str]
    era5_time_name: str | None
    era5_selected_times: List[str]
    era5_time_indices: List[int]
    temporal_integrity_ok: bool
    fuel_potential_saved: bool


def process_event(row: pd.Series, cfg: Dict[str, Any], raw_dir: Path, out_dir: Path) -> bool:
    """
    Build a physics-guided-ready sample from monthly ERA5 NetCDF (read-only).

    Critical scientific invariant enforced:
      len(meta["channels"]) MUST equal cube.shape[-1]
    Otherwise, the dataset becomes non-auditable and profile selection breaks.

    This implementation:
    - keeps your temporal + geospatial guards
    - builds cube (H,W,T,C)
    - builds channels list in the exact same order as data stacking
    - saves supervised physical prior map P (fuel potential) when possible
    """
    window_size_px = int(cfg_get(cfg, "data.window_size_px", 40))
    offsets_hours = list(
        cfg_get(cfg, "data.offsets_hours", [0, -6, -12, -18, -24]))
    qc_cfg = cfg_get(cfg, "data.qc", {})

    dt0: datetime = row["dt"].to_pydatetime()
    event_id = _to_event_id(dt0)

    sid = str(row.get("sid", "")) if pd.notna(row.get("sid", "")) else ""
    storm_name = str(row.get("storm_name", "")) if pd.notna(
        row.get("storm_name", "")) else ""
    basin = str(row.get("basin", "")) if pd.notna(row.get("basin", "")) else ""
    lat0 = float(row["lat"])
    lon0 = float(row["lon"])

    ri_label = int(row.get("ri_label", 0))
    dv12 = float(row["dv12_kt"]) if "dv12_kt" in row and pd.notna(
        row["dv12_kt"]) else None
    dv24 = float(row["dv24_kt"]) if "dv24_kt" in row and pd.notna(
        row["dv24_kt"]) else None
    wind_kt = float(row["wind_knots"]) if "wind_knots" in row and pd.notna(
        row["wind_knots"]) else None
    pressure_mb = float(row["pressure_mb"]) if "pressure_mb" in row and pd.notna(
        row["pressure_mb"]) else None

    # diagnostics requested (stable order)
    diag_channels = list(
        cfg_get(cfg, "physics.diagnostics.channels",
                ["wind_speed", "vorticity", "divergence", "grad_mslp", "sst_anom"])
    )

    # Map diagnostic keys -> output channel names (MUST match compute_diagnostics order)
    diag_to_channel = {
        "wind_speed": "wind_mps",
        "vorticity": "vort_1ps",
        "divergence": "div_1ps",
        "grad_mslp": "grad_mslp_Pa_per_m",
        "sst_anom": "sst_anom_K",
        # Optional heat flux channels (only if you enabled them in diagnostics.py)
        "latent_heat_flux": "latent_heat_flux_Wpm2",
        "sensible_heat_flux": "sensible_heat_flux_Wpm2",
        "total_heat_flux": "total_heat_flux_Wpm2",
    }

    # Validate diag_channels are known BEFORE processing
    unknown = [d for d in diag_channels if d not in diag_to_channel]
    if unknown:
        raise ValueError(
            f"Unknown diagnostics in config physics.diagnostics.channels: {unknown}")

    cubes_t: List[np.ndarray] = []
    timestamps: List[str] = []
    source_files: List[str] = []
    era5_selected_times: List[str] = []
    era5_time_indices: List[int] = []
    era5_time_name: str | None = None

    lats_win: np.ndarray | None = None
    lons_win: np.ndarray | None = None
    qc_flags_last: Dict[str, Any] = {}

    # store per-time diagnostics volumes for fuel potential (needs sst_anom, wind, div, optional gradp)
    sst_anom_vol: List[np.ndarray] = []
    wind_vol: List[np.ndarray] = []
    div_vol: List[np.ndarray] = []
    gradp_vol: List[np.ndarray] = []

    for oh in offsets_hours:
        dt = dt0 + timedelta(hours=int(oh))
        nc_path = _month_file(raw_dir, dt)
        if not nc_path.exists():
            logger.warning(
                f"Missing monthly ERA5 file: {nc_path} -> skipping event {event_id}")
            return False

        source_files.append(nc_path.name)
        ds = open_netcdf_safe(nc_path)

        ds_t, tname, tidx, selected_time = _select_time_slice(ds, dt)
        if era5_time_name is None:
            era5_time_name = tname
        era5_selected_times.append(selected_time or "")
        era5_time_indices.append(int(tidx) if tidx is not None else -1)

        # Resolve variable names (support long and short ERA5 conventions)
        sst_name = _resolve_var(ds_t, "sst", ["sea_surface_temperature"])
        msl_name = _resolve_var(ds_t, "msl", ["mean_sea_level_pressure"])
        u10_name = _resolve_var(ds_t, "u10", ["10m_u_component_of_wind"])
        v10_name = _resolve_var(ds_t, "v10", ["10m_v_component_of_wind"])

        sst_raw = ensure_2d(ds_t[sst_name].values)
        msl_raw = ensure_2d(ds_t[msl_name].values)
        u10_raw = ensure_2d(ds_t[u10_name].values)
        v10_raw = ensure_2d(ds_t[v10_name].values)

        lat_name = "latitude" if "latitude" in ds_t.coords else (
            "lat" if "lat" in ds_t.coords else None)
        lon_name = "longitude" if "longitude" in ds_t.coords else (
            "lon" if "lon" in ds_t.coords else None)
        if lat_name is None or lon_name is None:
            ds.close()
            raise ValueError(
                "ERA5 dataset missing latitude/longitude coordinates")

        lats1 = ds_t[lat_name].values
        lons1 = ds_t[lon_name].values
        if lats1.ndim != 1 or lons1.ndim != 1:
            ds.close()
            raise ValueError("Expected 1D lat/lon coordinates in ERA5")

        i = nearest_index(lats1, lat0)

        lons_min = float(np.nanmin(lons1))
        lons_max = float(np.nanmax(lons1))
        if lons_min >= 0.0 and lons_max > 180.0:
            lon_val = lon0 % 360.0
        else:
            lon_val = ((lon0 + 180.0) % 360.0) - 180.0
        j = nearest_index(lons1, lon_val)

        if lats_win is None or lons_win is None:
            lon_grid, lat_grid = np.meshgrid(lons1, lats1)
            lats_win = extract_window_by_index(lat_grid, i, j, window_size_px)
            lons_win = extract_window_by_index(lon_grid, i, j, window_size_px)

            # Geospatial integrity guard
            lons180 = ((lons_win.astype(np.float64) + 180.0) % 360.0) - 180.0
            ok_lat = float(np.nanmin(lats_win)) <= lat0 <= float(
                np.nanmax(lats_win))
            ok_lon = float(np.nanmin(lons180)) <= lon0 <= float(
                np.nanmax(lons180))
            if not (ok_lat and ok_lon):
                ds.close()
                logger.warning(
                    f"Center outside patch for {event_id}. Skipping event.")
                return False

        sst = normalize_sst_to_kelvin(
            extract_window_by_index(sst_raw, i, j, window_size_px))
        msl = normalize_mslp_to_pa(
            extract_window_by_index(msl_raw, i, j, window_size_px))
        u10 = extract_window_by_index(
            u10_raw, i, j, window_size_px).astype(np.float32)
        v10 = extract_window_by_index(
            v10_raw, i, j, window_size_px).astype(np.float32)

        qc_flags_last = qc_physical_ranges(
            sst, msl, u10, v10,
            sst_range=tuple(qc_cfg.get("sst_range_K", [240.0, 330.0])),
            msl_range=tuple(qc_cfg.get("msl_range_Pa", [80000.0, 110000.0])),
            wind_abs_max=float(qc_cfg.get("wind_abs_max_mps", 80.0)),
            max_nan_fraction=float(qc_cfg.get(
                "max_nan_fraction_per_channel", 0.5)),
        )

        base = {"sst": sst, "msl": msl, "u10": u10, "v10": v10}

        # extra list is in EXACT SAME ORDER as diag_channels
        extra = compute_diagnostics(base, lats_win, lons_win, diag_channels)

        # Build cube timestep (H,W,C) in EXACT SAME ORDER as channels list will be built later
        cube_t = np.stack([sst, msl, u10, v10] + extra,
                          axis=-1).astype(np.float32)
        cubes_t.append(cube_t)

        # Capture the needed diagnostics for fuel potential (by channel name)
        # Build a name->array map aligned with diag_channels order
        extras_by_name: Dict[str, np.ndarray] = {}
        for k, d in enumerate(diag_channels):
            cname = diag_to_channel[d]
            extras_by_name[cname] = extra[k]

        if "sst_anom_K" in extras_by_name:
            sst_anom_vol.append(extras_by_name["sst_anom_K"])
        if "wind_mps" in extras_by_name:
            wind_vol.append(extras_by_name["wind_mps"])
        if "div_1ps" in extras_by_name:
            div_vol.append(extras_by_name["div_1ps"])
        if "grad_mslp_Pa_per_m" in extras_by_name:
            gradp_vol.append(extras_by_name["grad_mslp_Pa_per_m"])

        timestamps.append(dt.strftime("%Y-%m-%d %H:%M"))
        ds.close()

    # Temporal integrity guard
    T = len(offsets_hours)
    unique_sel = len(set(era5_selected_times))
    if unique_sel != T:
        logger.warning(
            f"Temporal collapse for {event_id}: unique_selected={unique_sel}/{T} "
            f"selected_times={era5_selected_times} indices={era5_time_indices}. Skipping event."
        )
        return False

    cube = np.stack(cubes_t, axis=2).astype(np.float32)  # (H,W,T,C)

    # Channel list (base + diagnostics) — MUST MATCH cube stacking
    base_channels = ["sst_K", "mslp_Pa", "u10_mps", "v10_mps"]
    derived_channels = [diag_to_channel[d] for d in diag_channels]
    all_channels = base_channels + derived_channels

    # Units: include only what can exist (safe superset)
    units = {
        "sst_K": "K",
        "mslp_Pa": "Pa",
        "u10_mps": "m s-1",
        "v10_mps": "m s-1",
        "wind_mps": "m s-1",
        "vort_1ps": "s-1",
        "div_1ps": "s-1",
        "grad_mslp_Pa_per_m": "Pa m-1",
        "sst_anom_K": "K",
        "latent_heat_flux_Wpm2": "W m-2",
        "sensible_heat_flux_Wpm2": "W m-2",
        "total_heat_flux_Wpm2": "W m-2",
    }

    # Hard scientific guard: metadata must match cube last dimension
    if cube.shape[-1] != len(all_channels):
        raise ValueError(
            f"Metadata mismatch for {event_id}: cube has C={cube.shape[-1]} but channels list has {len(all_channels)}. "
            f"diag_channels={diag_channels} all_channels={all_channels}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{event_id}.npy", cube)
    np.save(out_dir / f"{event_id}_lats.npy",
            lats_win.astype(np.float32))  # type: ignore[arg-type]
    np.save(out_dir / f"{event_id}_lons.npy",
            lons_win.astype(np.float32))  # type: ignore[arg-type]

    # Save supervised physical prior map (fuel potential), if inputs available
    fuel_saved = False
    if len(sst_anom_vol) == T and len(wind_vol) == T and len(div_vol) == T:
        sst_anom_arr = np.stack(sst_anom_vol, axis=2).astype(np.float32)
        wind_arr = np.stack(wind_vol, axis=2).astype(np.float32)
        div_arr = np.stack(div_vol, axis=2).astype(np.float32)
        grad_arr = np.stack(gradp_vol, axis=2).astype(
            np.float32) if len(gradp_vol) == T else None

        P = build_fuel_potential(
            sst_anom_K=sst_anom_arr,
            wind_mps=wind_arr,
            divergence_1ps=div_arr,
            grad_mslp_Pa_per_m=grad_arr,
            cfg=FuelPotentialConfig(
                w_conv=float(
                    cfg_get(cfg, "physics_guided.fuel_potential.w_conv", 1.0)),
                w_gradp=float(
                    cfg_get(cfg, "physics_guided.fuel_potential.w_gradp", 0.0)),
            ),
        )
        np.save(out_dir / f"{event_id}_fuel_potential.npy",
                P.astype(np.float32))
        fuel_saved = True

    meta = EventMeta(
        event_id=event_id,
        sid=sid,
        timestamp=dt0.strftime("%Y-%m-%d %H:%M"),
        storm_name=storm_name,
        basin=basin,
        ri_label=ri_label,
        dv12_kt=dv12,
        dv24_kt=dv24,
        wind_kt=wind_kt,
        pressure_mb=pressure_mb,
        center_lat=lat0,
        center_lon=lon0,
        timestamps=timestamps,
        cube_shape=list(cube.shape),
        channels=all_channels,
        units=units,
        qc_flags=qc_flags_last,
        source_files=sorted(list(set(source_files))),
        era5_time_name=era5_time_name,
        era5_selected_times=era5_selected_times,
        era5_time_indices=era5_time_indices,
        temporal_integrity_ok=True,
        fuel_potential_saved=fuel_saved,
    )

    with (out_dir / f"{event_id}.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    return True


def run_preprocess(cfg: Dict[str, Any]) -> None:
    raw_dir = Path(cfg["paths"]["raw_data"])
    out_dir = Path(cfg["paths"]["interim_data"])
    event_list_path = Path(cfg["paths"]["event_list"])

    df = load_event_list(event_list_path)
    logger.info(f"Loaded event list: {len(df)} rows")
    logger.info(f"Raw dir: {raw_dir.resolve()}")
    logger.info(f"Interim dir: {out_dir.resolve()}")

    ok, skipped, failed = 0, 0, 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing events"):
        try:
            written = process_event(row, cfg, raw_dir, out_dir)
            if written:
                ok += 1
            else:
                skipped += 1
        except Exception as e:
            failed += 1
            eid = _to_event_id(row["dt"].to_pydatetime()) if pd.notna(
                row.get("dt")) else "unknown"
            logger.error(f"Event {eid} failed: {e}", exc_info=True)

    logger.info(
        f"Preprocess complete. OK={ok} SKIPPED={skipped} FAILED={failed}")
