from __future__ import annotations

"""
CycloneNet — sample ADT (absolute dynamic topography) onto each event's grid.

ADT is the surface signature of the SUBSURFACE ocean heat reservoir that sustains
rapid intensification. The SLA<->TCHP validation showed ADT robustly tracks TCHP at
the storm centre (Spearman rho ~0.30, replicated across 2022 and 2023), so it is a
physically-justified NEW MODEL INPUT — the first channel that gives the network
information about the ocean reservoir it cannot otherwise see.

For every event whose timestamp/region is covered by a downloaded SLA/ADT file, we
sample ADT onto the event's (40x40) grid and store it as a side-car `{event_id}_adt.npy`.
Events outside ADT coverage get no file; the dataset then supplies a neutral (zero,
masked) channel so the pipeline still runs on the full archive.

This is an INPUT (unlike TCHP, which is validation-only). ADT is an environmental
ocean field, not a future/label quantity, so it introduces no temporal leakage.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.utils.config import cfg_get

logger = logging.getLogger(__name__)


def _open_sla(path: Path):
    import xarray as xr
    for eng in ("h5netcdf", "scipy", "netcdf4"):
        try:
            return xr.open_dataset(path, engine=eng).sortby("latitude").sortby("longitude")
        except Exception:
            continue
    raise IOError(f"Could not open {path}")


def _resolve_ocean_var(ds) -> str:
    for c in ("adt", "sla"):
        if c in ds.data_vars:
            return c
    raise KeyError(f"No adt/sla variable in {list(ds.data_vars)}")


def _sample_to_grid(ds_time, var: str, lat1d: np.ndarray, lon1d: np.ndarray) -> np.ndarray:
    """Bilinear-interpolate the ocean field onto the event's 1D lat/lon axes -> (H,W)."""
    import xarray as xr
    field = ds_time[var].interp(
        latitude=xr.DataArray(lat1d, dims="lat"),
        longitude=xr.DataArray(lon1d, dims="lon"),
    ).values
    return np.asarray(field, dtype=np.float32)


def add_adt_to_events(cfg: Dict[str, Any]) -> Dict[str, int]:
    """
    Sample ADT into each event grid where SLA/ADT coverage exists.
    Writes `{event_id}_adt.npy` (H,W) and sets meta['adt_saved']=True/False.
    """
    interim = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    ocean_dir = Path(cfg_get(cfg, "paths.ocean_dir", "./data/external/ocean")).resolve()
    min_finite = float(cfg_get(cfg, "data.adt.min_finite_fraction", 0.5))

    sla_cache: Dict[int, Optional[Any]] = {}

    def sla_for_year(year: int):
        if year not in sla_cache:
            p = ocean_dir / f"ssh_sla_{year}.nc"
            sla_cache[year] = _open_sla(p) if p.exists() else None
        return sla_cache[year]

    n_saved = n_no_coverage = n_low_finite = 0
    for json_path in sorted(interim.glob("era5_*.json")):
        eid = json_path.stem
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        ts = meta.get("timestamp")
        lats_p = interim / f"{eid}_lats.npy"
        lons_p = interim / f"{eid}_lons.npy"
        if ts is None or not lats_p.exists() or not lons_p.exists():
            continue

        t = pd.to_datetime(ts)
        ds = sla_for_year(int(t.year))
        if ds is None:
            n_no_coverage += 1
            meta["adt_saved"] = False
            json_path.write_text(json.dumps(meta), encoding="utf-8")
            continue

        var = _resolve_ocean_var(ds)
        lats = np.load(lats_p).astype(np.float32)
        lons = np.load(lons_p).astype(np.float32)
        lat1d, lon1d = lats[:, 0], lons[0, :]

        try:
            ds_t = ds.sel(time=np.datetime64(t.tz_localize(None) if t.tzinfo else t), method="nearest")
            field = _sample_to_grid(ds_t, var, lat1d, lon1d)
        except Exception as exc:
            logger.debug("ADT sample failed for %s: %s", eid, exc)
            n_no_coverage += 1
            meta["adt_saved"] = False
            json_path.write_text(json.dumps(meta), encoding="utf-8")
            continue

        finite_frac = float(np.isfinite(field).mean())
        if finite_frac < min_finite:
            # Mostly outside coverage / over land -> treat as uncovered.
            n_low_finite += 1
            meta["adt_saved"] = False
            json_path.write_text(json.dumps(meta), encoding="utf-8")
            continue

        # Fill residual NaNs (small land patches) with the window mean.
        fill = float(np.nanmean(field)) if np.isfinite(field).any() else 0.0
        field = np.where(np.isfinite(field), field, fill).astype(np.float32)

        np.save(interim / f"{eid}_adt.npy", field)
        meta["adt_saved"] = True
        meta["adt_variable"] = var
        json_path.write_text(json.dumps(meta), encoding="utf-8")
        n_saved += 1

    logger.info("ADT sampling: saved=%d, no_coverage=%d, low_finite=%d",
                n_saved, n_no_coverage, n_low_finite)
    return {"saved": n_saved, "no_coverage": n_no_coverage, "low_finite": n_low_finite}


def compute_adt_train_stats(cfg: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Compute train-only ADT mean/std and inject them into normalization_stats.json
    (keys 'adt_mean','adt_std'). Train-only, to prevent leakage — mirrors the
    channel normalization policy.
    """
    interim = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    splits_csv = Path(cfg_get(cfg, "paths.splits_csv", "./data/normalized/splits.csv")).resolve()
    stats_path = Path(cfg_get(cfg, "paths.normalization_stats",
                              "./data/normalized/normalization_stats.json")).resolve()
    if not splits_csv.exists() or not stats_path.exists():
        logger.warning("Splits or stats missing; run 'normalize' before computing ADT stats.")
        return None

    df = pd.read_csv(splits_csv)
    train_ids = df.loc[df["split"] == "train", "event_id"].astype(str).tolist()

    vals = []
    for eid in train_ids:
        p = interim / f"{eid}_adt.npy"
        if p.exists():
            a = np.load(p).astype(np.float64).ravel()
            vals.append(a[np.isfinite(a)])
    if not vals:
        logger.warning("No train ADT files found; ADT stats not computed.")
        return None

    allv = np.concatenate(vals)
    mean = float(np.mean(allv))
    std = float(max(np.std(allv), 1e-6))

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["adt_mean"] = mean
    stats["adt_std"] = std
    stats["adt_n_train_events"] = int(len(vals))
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("ADT train stats: mean=%.4f std=%.4f from %d events -> %s",
                mean, std, len(vals), stats_path)
    return {"adt_mean": mean, "adt_std": std, "n": len(vals)}


def run_preprocess_adt(cfg: Dict[str, Any]) -> None:
    """Entrypoint: sample ADT into events, then compute train-only ADT stats."""
    add_adt_to_events(cfg)
    compute_adt_train_stats(cfg)
