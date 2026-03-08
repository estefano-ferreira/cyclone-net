from __future__ import annotations

"""CycloneNet — scientific preprocessing (portable, temporally safe, physics-guided ready).

This preprocessor reads monthly ERA5 NetCDF files in read-only mode and writes
per-event artifacts required by the scientific training pipeline:

Outputs
-------
- Cube:  (H, W, T, C) -> data/interim/{event_id}.npy
- Metadata audit:      -> data/interim/{event_id}.json
- Geographic grids:    -> data/interim/{event_id}_lats.npy
                           data/interim/{event_id}_lons.npy
- Physical prior:      -> data/interim/{event_id}_fuel_potential.npy

Scientific guarantees
---------------------
- Temporal integrity:
  Events are rejected if the selected ERA5 timestamps collapse to fewer than T unique times.
- Geospatial integrity:
  Events are rejected if the cyclone center is not contained inside the extracted patch.
- Fixed tensor shape:
  Pad/crop logic guarantees consistent spatial shape for all saved samples.
- Metadata/data consistency:
  The number and order of metadata channels must exactly match the cube last dimension.

Anti-leakage policy
-------------------
This preprocessor does NOT inject external validation products such as TCHP/OHC
as training input channels. Such products may be saved separately for external
scientific validation.
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

from src.physics.diagnostics import compute_diagnostics
from src.physics.fuel_potential import FuelPotentialConfig, build_fuel_potential
from src.utils.config import cfg_get

logger = logging.getLogger(__name__)


def ensure_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[..., 0]
    raise ValueError(f"Expected a 2D field, got shape {array.shape}.")


def pad_or_crop_2d(array: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    array = ensure_2d(array).astype(np.float32)
    target_h, target_w = target_hw
    h, w = array.shape

    pad_h_before = max(0, (target_h - h) // 2)
    pad_h_after = max(0, target_h - h - pad_h_before)
    pad_w_before = max(0, (target_w - w) // 2)
    pad_w_after = max(0, target_w - w - pad_w_before)

    if pad_h_before or pad_h_after or pad_w_before or pad_w_after:
        array = np.pad(
            array,
            ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
            mode="reflect",
        )

    h, w = array.shape
    if h > target_h:
        start = (h - target_h) // 2
        array = array[start:start + target_h, :]
    if w > target_w:
        start = (w - target_w) // 2
        array = array[:, start:start + target_w]

    if array.shape != (target_h, target_w):
        raise ValueError(f"pad_or_crop_2d failed: got {array.shape}, expected {(target_h, target_w)}.")
    return array


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


def nan_fraction(array: np.ndarray) -> float:
    return float(np.mean(~np.isfinite(array)))


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

    flags["nan_fraction_per_channel"] = {
        "sst": nan_fraction(sst_k),
        "msl": nan_fraction(msl_pa),
        "u10": nan_fraction(u10),
        "v10": nan_fraction(v10),
    }

    max_nf = max(flags["nan_fraction_per_channel"].values())
    if max_nf > float(max_nan_fraction):
        raise ValueError(f"Too many NaNs in event patch. Max channel NaN fraction={max_nf:.3f}.")

    flags["sst_range_ok"] = bool(np.nanmin(sst_k) >= sst_range[0] and np.nanmax(sst_k) <= sst_range[1])
    flags["msl_range_ok"] = bool(np.nanmin(msl_pa) >= msl_range[0] and np.nanmax(msl_pa) <= msl_range[1])
    flags["wind_abs_ok"] = bool(
        np.nanmax(np.abs(u10)) <= wind_abs_max and np.nanmax(np.abs(v10)) <= wind_abs_max
    )

    if not flags["sst_range_ok"]:
        raise ValueError("SST is outside the configured physical range.")
    if not flags["msl_range_ok"]:
        raise ValueError("MSLP is outside the configured physical range.")
    if not flags["wind_abs_ok"]:
        raise ValueError("10 m wind is outside the configured physical range.")
    return flags


def windows_short_path(path: Path) -> str:
    try:
        get_short_path = ctypes.windll.kernel32.GetShortPathNameW
        get_short_path.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        get_short_path.restype = ctypes.c_uint
        out_buf = ctypes.create_unicode_buffer(4096)
        result = get_short_path(str(path), out_buf, 4096)
        if result > 0 and out_buf.value:
            return out_buf.value
    except Exception:
        pass
    return str(path)


def open_netcdf_safe(path: Path) -> xr.Dataset:
    path = Path(path)
    open_path = windows_short_path(path) if os.name == "nt" else str(path)
    for engine in ["netcdf4", "h5netcdf", "scipy"]:
        try:
            return xr.open_dataset(open_path, engine=engine, cache=False)
        except Exception:
            continue
    raise RuntimeError(f"Unable to open NetCDF file: {path}")


def find_time_name(ds: xr.Dataset) -> str | None:
    for name in ["time", "valid_time", "Time", "TIME"]:
        if name in ds.coords or name in ds.dims or name in ds.variables:
            return name
    for name in list(ds.coords) + list(ds.dims):
        if "time" in name.lower():
            return name
    return None


def select_time_slice(ds: xr.Dataset, target_dt: datetime) -> tuple[xr.Dataset, str | None, int | None, str | None]:
    try:
        ds = xr.decode_cf(ds)
    except Exception:
        pass

    time_name = find_time_name(ds)
    if time_name is None:
        return ds, None, None, None

    if time_name not in ds.coords and time_name in ds.variables:
        try:
            ds = ds.set_coords(time_name)
        except Exception:
            pass

    if time_name in ds.coords:
        try:
            ds_t = ds.sel({time_name: np.datetime64(target_dt)}, method="nearest")
            selected_value = np.asarray(ds_t[time_name].values).reshape(-1)[0]
            selected_str = pd.to_datetime(selected_value).strftime("%Y-%m-%d %H:%M")
            times = pd.to_datetime(ds[time_name].values)
            idx = int(np.argmin(np.abs(times - np.datetime64(target_dt))))
            return ds_t, time_name, idx, selected_str
        except Exception:
            pass

    try:
        times = pd.to_datetime(ds[time_name].values)
        idx = int(np.argmin(np.abs(times - np.datetime64(target_dt))))
        selected_str = pd.to_datetime(ds[time_name].values[idx]).strftime("%Y-%m-%d %H:%M")
        ds_t = ds.isel({time_name: idx}) if time_name in ds.dims else ds
        return ds_t, time_name, idx, selected_str
    except Exception:
        pass

    if time_name in ds.dims and ds.dims[time_name] > 1:
        return ds.isel({time_name: 0}), time_name, 0, None
    return ds, time_name, None, None


def resolve_var(ds_t: xr.Dataset, preferred: str, fallbacks: List[str]) -> str:
    if preferred in ds_t.data_vars:
        return preferred
    for fb in fallbacks:
        if fb in ds_t.data_vars:
            return fb
    raise KeyError(f"No variable named '{preferred}'. Available variables: {list(ds_t.data_vars)}")


def nearest_index(arr_1d: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(arr_1d - value)))


def extract_window_by_index(field_2d: np.ndarray, i: int, j: int, size: int) -> np.ndarray:
    h, w = field_2d.shape
    half = size // 2
    i0 = max(0, i - half)
    i1 = min(h, i + half + (0 if size % 2 == 0 else 1))
    j0 = max(0, j - half)
    j1 = min(w, j + half + (0 if size % 2 == 0 else 1))
    return pad_or_crop_2d(field_2d[i0:i1, j0:j1], (size, size))


def to_event_id(dt: datetime) -> str:
    return f"era5_{dt.strftime('%Y_%m_%d_%H%M')}"


def month_file(raw_dir: Path, dt: datetime) -> Path:
    return raw_dir / f"era5_{dt.strftime('%Y_%m')}.nc"


def load_event_list(event_list_path: Path) -> pd.DataFrame:
    df = pd.read_csv(event_list_path)
    if "timestamp" in df.columns:
        df["dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "datetime" in df.columns:
        df["dt"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M", errors="coerce")
    else:
        raise ValueError("event_list must contain either 'timestamp' or 'datetime'.")

    for col in ["lat", "lon"]:
        if col not in df.columns:
            raise ValueError(f"event_list is missing required column '{col}'.")

    if "storm_name" not in df.columns and "name" in df.columns:
        df["storm_name"] = df["name"]
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
    window_size_px = int(cfg_get(cfg, "data.window_size_px", 40))
    offsets_hours = list(cfg_get(cfg, "data.offsets_hours", [0, -6, -12, -18, -24]))
    qc_cfg = cfg_get(cfg, "data.qc", {})

    dt0: datetime = row["dt"].to_pydatetime()
    event_id = to_event_id(dt0)

    sid = str(row.get("sid", "")) if pd.notna(row.get("sid", "")) else ""
    storm_name = str(row.get("storm_name", row.get("name", ""))) if pd.notna(row.get("storm_name", row.get("name", ""))) else ""
    basin = str(row.get("basin", "")) if pd.notna(row.get("basin", "")) else ""
    lat0 = float(row["lat"])
    lon0 = float(row["lon"])
    ri_label = int(row.get("ri_label", 0))
    dv12 = float(row["dv12_kt"]) if "dv12_kt" in row and pd.notna(row["dv12_kt"]) else None
    dv24 = float(row["dv24_kt"]) if "dv24_kt" in row and pd.notna(row["dv24_kt"]) else None

    if "wind_kt" in row and pd.notna(row["wind_kt"]):
        wind_kt = float(row["wind_kt"])
    elif "wind_knots" in row and pd.notna(row["wind_knots"]):
        wind_kt = float(row["wind_knots"])
    else:
        wind_kt = None

    pressure_mb = float(row["pressure_mb"]) if "pressure_mb" in row and pd.notna(row["pressure_mb"]) else None

    diag_channels = list(cfg_get(cfg, "physics.diagnostics.channels", ["wind_speed", "vorticity", "divergence", "grad_mslp", "sst_anom"]))
    diag_to_channel = {
        "wind_speed": "wind_mps",
        "vorticity": "vort_1ps",
        "divergence": "div_1ps",
        "grad_mslp": "grad_mslp_Pa_per_m",
        "sst_anom": "sst_anom_K",
        "latent_heat_flux": "latent_heat_flux_Wpm2",
        "sensible_heat_flux": "sensible_heat_flux_Wpm2",
        "total_heat_flux": "total_heat_flux_Wpm2",
    }

    cubes_t: List[np.ndarray] = []
    timestamps: List[str] = []
    source_files: List[str] = []
    era5_selected_times: List[str] = []
    era5_time_indices: List[int] = []
    era5_time_name: str | None = None
    lats_win: np.ndarray | None = None
    lons_win: np.ndarray | None = None
    qc_flags_last: Dict[str, Any] = {}
    total_heat_vol: List[np.ndarray | None] = []
    sst_anom_vol: List[np.ndarray] = []
    wind_vol: List[np.ndarray] = []
    div_vol: List[np.ndarray] = []
    gradp_vol: List[np.ndarray] = []

    for offset_h in offsets_hours:
        dt = dt0 + timedelta(hours=int(offset_h))
        nc_path = month_file(raw_dir, dt)
        if not nc_path.exists():
            logger.warning("Missing monthly ERA5 file: %s. Skipping event %s.", nc_path, event_id)
            return False

        source_files.append(nc_path.name)
        ds = open_netcdf_safe(nc_path)
        try:
            ds_t, time_name, time_idx, selected_time = select_time_slice(ds, dt)
            if era5_time_name is None:
                era5_time_name = time_name
            era5_selected_times.append(selected_time or "")
            era5_time_indices.append(int(time_idx) if time_idx is not None else -1)

            sst_name = resolve_var(ds_t, "sst", ["sea_surface_temperature"])
            msl_name = resolve_var(ds_t, "msl", ["mean_sea_level_pressure"])
            u10_name = resolve_var(ds_t, "u10", ["10m_u_component_of_wind"])
            v10_name = resolve_var(ds_t, "v10", ["10m_v_component_of_wind"])

            sst_raw = ensure_2d(ds_t[sst_name].values)
            msl_raw = ensure_2d(ds_t[msl_name].values)
            u10_raw = ensure_2d(ds_t[u10_name].values)
            v10_raw = ensure_2d(ds_t[v10_name].values)

            try:
                t2m_raw = ensure_2d(ds_t["t2m"].values)
                d2m_raw = ensure_2d(ds_t["d2m"].values)
                have_t2m_d2m = True
            except Exception:
                t2m_raw = None
                d2m_raw = None
                have_t2m_d2m = False

            lat_name = "latitude" if "latitude" in ds_t.coords else "lat" if "lat" in ds_t.coords else None
            lon_name = "longitude" if "longitude" in ds_t.coords else "lon" if "lon" in ds_t.coords else None
            if lat_name is None or lon_name is None:
                raise ValueError("ERA5 dataset is missing latitude/longitude coordinates.")

            lats_1d = ds_t[lat_name].values
            lons_1d = ds_t[lon_name].values
            i = nearest_index(lats_1d, lat0)
            lon_value = lon0 % 360.0 if float(np.nanmin(lons_1d)) >= 0.0 and float(np.nanmax(lons_1d)) > 180.0 else ((lon0 + 180.0) % 360.0) - 180.0
            j = nearest_index(lons_1d, lon_value)

            if lats_win is None or lons_win is None:
                lon_grid, lat_grid = np.meshgrid(lons_1d, lats_1d)
                lats_win = extract_window_by_index(lat_grid, i, j, window_size_px)
                lons_win = extract_window_by_index(lon_grid, i, j, window_size_px)
                lons_win_180 = ((lons_win.astype(np.float64) + 180.0) % 360.0) - 180.0
                ok_lat = float(np.nanmin(lats_win)) <= lat0 <= float(np.nanmax(lats_win))
                ok_lon = float(np.nanmin(lons_win_180)) <= lon0 <= float(np.nanmax(lons_win_180))
                if not (ok_lat and ok_lon):
                    return False

            sst = normalize_sst_to_kelvin(extract_window_by_index(sst_raw, i, j, window_size_px))
            msl = normalize_mslp_to_pa(extract_window_by_index(msl_raw, i, j, window_size_px))
            u10 = extract_window_by_index(u10_raw, i, j, window_size_px).astype(np.float32)
            v10 = extract_window_by_index(v10_raw, i, j, window_size_px).astype(np.float32)

            qc_flags_last = qc_physical_ranges(
                sst_k=sst,
                msl_pa=msl,
                u10=u10,
                v10=v10,
                sst_range=tuple(qc_cfg.get("sst_range_K", [240.0, 330.0])),
                msl_range=tuple(qc_cfg.get("msl_range_Pa", [80000.0, 110000.0])),
                wind_abs_max=float(qc_cfg.get("wind_abs_max_mps", 80.0)),
                max_nan_fraction=float(qc_cfg.get("max_nan_fraction_per_channel", 0.5)),
            )

            base = {"sst": sst, "msl": msl, "u10": u10, "v10": v10}
            extra = compute_diagnostics(base, lats_win, lons_win, diag_channels)
            cube_t = np.stack([sst, msl, u10, v10] + extra, axis=-1).astype(np.float32)

            if nan_fraction(cube_t) > float(qc_cfg.get("max_nan_fraction_cube", 0.20)):
                raise ValueError(f"Cube NaN fraction too high for {event_id}.")

            cubes_t.append(cube_t)

            if have_t2m_d2m and t2m_raw is not None and d2m_raw is not None:
                from src.physics.heat_flux import compute_heat_fluxes

                t2m = extract_window_by_index(t2m_raw, i, j, window_size_px).astype(np.float32)
                d2m = extract_window_by_index(d2m_raw, i, j, window_size_px).astype(np.float32)
                heat_fluxes = compute_heat_fluxes(
                    sst=sst,
                    u10=u10,
                    v10=v10,
                    msl=msl,
                    t2m=t2m,
                    d2m=d2m,
                    Ce=float(cfg_get(cfg, "physics.heat_flux.Ce", 1.2e-3)),
                    Ch=float(cfg_get(cfg, "physics.heat_flux.Ch", 1.2e-3)),
                )
                total_heat_vol.append(heat_fluxes["total_heat_flux"])
            else:
                total_heat_vol.append(None)

            extras_by_name = {diag_to_channel[diag_channels[k]]: extra[k] for k in range(len(diag_channels))}
            if "sst_anom_K" in extras_by_name:
                sst_anom_vol.append(extras_by_name["sst_anom_K"])
            if "wind_mps" in extras_by_name:
                wind_vol.append(extras_by_name["wind_mps"])
            if "div_1ps" in extras_by_name:
                div_vol.append(extras_by_name["div_1ps"])
            if "grad_mslp_Pa_per_m" in extras_by_name:
                gradp_vol.append(extras_by_name["grad_mslp_Pa_per_m"])

            timestamps.append(dt.strftime("%Y-%m-%d %H:%M"))
        finally:
            ds.close()

    t_steps = len(offsets_hours)
    if len(set(era5_selected_times)) != t_steps:
        return False

    cube = np.stack(cubes_t, axis=2).astype(np.float32)
    base_channels = ["sst_K", "mslp_Pa", "u10_mps", "v10_mps"]
    derived_channels = [diag_to_channel[d] for d in diag_channels]
    all_channels = base_channels + derived_channels
    if cube.shape[-1] != len(all_channels):
        raise ValueError(f"Metadata/channel mismatch for {event_id}: cube has {cube.shape[-1]}, metadata has {len(all_channels)}.")

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

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{event_id}.npy", cube)
    np.save(out_dir / f"{event_id}_lats.npy", lats_win.astype(np.float32))
    np.save(out_dir / f"{event_id}_lons.npy", lons_win.astype(np.float32))

    fuel_saved = False
    if all(v is not None for v in total_heat_vol):
        heat_arr = np.stack(total_heat_vol, axis=2).astype(np.float32)
        heat_norm = np.zeros_like(heat_arr, dtype=np.float32)
        for t in range(t_steps):
            slab = heat_arr[:, :, t]
            slab_min = float(np.nanmin(slab))
            slab_max = float(np.nanmax(slab))
            if slab_max - slab_min > 1e-6:
                heat_norm[:, :, t] = (slab - slab_min) / (slab_max - slab_min)
        if nan_fraction(heat_norm) > float(qc_cfg.get("max_nan_fraction_fuel_prior", 0.20)):
            raise ValueError(f"Fuel prior NaN fraction too high for {event_id}.")
        np.save(out_dir / f"{event_id}_fuel_potential.npy", heat_norm)
        fuel_saved = True
    elif len(sst_anom_vol) == t_steps and len(wind_vol) == t_steps and len(div_vol) == t_steps:
        prior = build_fuel_potential(
            sst_anom_K=np.stack(sst_anom_vol, axis=2).astype(np.float32),
            wind_mps=np.stack(wind_vol, axis=2).astype(np.float32),
            divergence_1ps=np.stack(div_vol, axis=2).astype(np.float32),
            grad_mslp_Pa_per_m=np.stack(gradp_vol, axis=2).astype(np.float32) if len(gradp_vol) == t_steps else None,
            cfg=FuelPotentialConfig(
                w_conv=float(cfg_get(cfg, "physics_guided.fuel_potential.w_conv", 1.0)),
                w_gradp=float(cfg_get(cfg, "physics_guided.fuel_potential.w_gradp", 0.0)),
            ),
        )
        if nan_fraction(prior) > float(qc_cfg.get("max_nan_fraction_fuel_prior", 0.20)):
            raise ValueError(f"Fuel prior NaN fraction too high for {event_id}.")
        np.save(out_dir / f"{event_id}_fuel_potential.npy", prior.astype(np.float32))
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
        source_files=sorted(set(source_files)),
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

    ok = 0
    skipped = 0
    failed = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing events"):
        try:
            written = process_event(row=row, cfg=cfg, raw_dir=raw_dir, out_dir=out_dir)
            if written:
                ok += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            event_id = to_event_id(row["dt"].to_pydatetime()) if pd.notna(row.get("dt")) else "unknown"
            logger.error("Event %s failed: %s", event_id, exc, exc_info=True)
    logger.info("Preprocessing complete. OK=%d | SKIPPED=%d | FAILED=%d", ok, skipped, failed)