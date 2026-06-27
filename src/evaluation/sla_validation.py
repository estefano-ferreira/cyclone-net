from __future__ import annotations

"""
CycloneNet — SLA co-location validation (SLA <-> TCHP <-> RI).

Scientific question
-------------------
Does Sea Level Anomaly (SLA) — a cheap, surface-observable field — actually carry
the SUBSURFACE ocean-heat signal that feeds rapid intensification? If yes, SLA is a
defensible new model input and an independent ground-truth for the energy source.

Three falsifiable tests, per event in the covered season/box:
  1. PROXY     — is SLA at the storm centre correlated with TCHP at the storm centre?
                 (If not, SLA is not a usable proxy for the subsurface reservoir.)
  2. CO-LOCATION — does the SLA maximum point at the TCHP peak better than the naive
                 storm-centre baseline?
  3. RI SIGNAL — do RI events sit over significantly higher SLA than non-RI events?
                 (The reservoir should be warmer/higher under intensifying storms.)

This uses observational data only (no model); it tests the physical premise behind
adding SLA, before any training.
"""

import glob
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.evaluation.spatial_metrics import _nan_safe_gaussian, haversine_distance
from src.processors.preprocess_tchp import find_peak_location, sample_at_center
from src.utils.config import cfg_get

logger = logging.getLogger(__name__)


def _local_peak(field2d: np.ndarray, lats1d: np.ndarray, lons1d: np.ndarray,
                detrend_deg: float = 3.0, smooth_sigma: float = 1.0):
    """
    Locate the LOCAL feature peak by removing the basin-scale background first.

    The raw field max in a window migrates to the warmest box edge (large-scale
    meridional gradient), which is an artifact, not the storm's fuel feature. We
    high-pass the field (subtract a heavily smoothed background ~detrend_deg wide)
    so a mesoscale eddy / warm ring dominates, then find that peak.
    """
    dlat = float(np.median(np.abs(np.diff(lats1d)))) or 0.25
    sigma_px = max(1.0, detrend_deg / dlat)
    background = _nan_safe_gaussian(field2d, sigma_px)
    local = field2d - background
    return find_peak_location(local, lats1d, lons1d, smooth_sigma=smooth_sigma)


def _open(path: Path):
    import xarray as xr
    for eng in ("h5netcdf", "scipy", "netcdf4"):
        try:
            return xr.open_dataset(path, engine=eng).sortby("latitude").sortby("longitude")
        except Exception:
            continue
    raise IOError(f"Could not open {path} with any engine.")


def _resolve_var(ds, candidates: List[str]) -> str:
    for c in candidates:
        if c in ds.data_vars:
            return c
    raise KeyError(f"None of {candidates} in {list(ds.data_vars)}")


def _window(ds, var: str, ts: pd.Timestamp, clat: float, clon: float, wdeg: float):
    """Nearest-time lat/lon box around (clat,clon). Returns (field2d, lats1d, lons1d) or None."""
    ds_t = ds.sel(time=np.datetime64(ts), method="nearest") if "time" in ds.coords else ds
    lon_min, lon_max = clon - wdeg, clon + wdeg
    if float(ds_t["longitude"].min()) >= 0.0 and lon_min < 0.0:
        lon_min += 360.0
        lon_max += 360.0
    reg = ds_t.sel(latitude=slice(clat - wdeg, clat + wdeg), longitude=slice(lon_min, lon_max))
    if reg.sizes.get("latitude", 0) == 0 or reg.sizes.get("longitude", 0) == 0:
        return None
    field = np.squeeze(np.asarray(reg[var].values))
    if field.ndim != 2:
        return None
    return field, np.asarray(reg["latitude"].values), np.asarray(reg["longitude"].values)


def _covered_events(cfg: Dict[str, Any], year: int, months: List[int],
                    box: Tuple[float, float, float, float]) -> List[Dict[str, Any]]:
    interim = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    lat_min, lat_max, lon_min, lon_max = box
    out = []
    for f in sorted(glob.glob(str(interim / "era5_*.json"))):
        m = json.loads(Path(f).read_text(encoding="utf-8"))
        ts, clat, clon = m.get("timestamp"), m.get("center_lat"), m.get("center_lon")
        if ts is None or clat is None or clon is None:
            continue
        t = pd.to_datetime(ts).tz_localize(None) if pd.to_datetime(ts).tzinfo else pd.to_datetime(ts)
        if t.year == year and t.month in months and lat_min <= clat <= lat_max and lon_min <= clon <= lon_max:
            out.append({"event_id": m.get("event_id", Path(f).stem),
                        "timestamp": t, "center_lat": float(clat), "center_lon": float(clon),
                        "ri_label": int(m.get("ri_label", 0))})
    return out


def _agg(x: List[float]) -> Dict[str, Any]:
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"n": 0, "mean": None, "median": None}
    return {"n": int(a.size), "mean": float(np.mean(a)), "median": float(np.median(a))}


def validate_sla_colocation(cfg: Dict[str, Any], year: int = 2023,
                            window_deg: float = 5.0, max_events: Optional[int] = None) -> Dict[str, Any]:
    ocean_dir = Path(cfg_get(cfg, "paths.ocean_dir", "./data/external/ocean")).resolve()
    tchp_dir = Path(cfg_get(cfg, "paths.tchp_dir", "./data/external/tchp")).resolve()
    months = list(cfg_get(cfg, "download.ssh.season_months", [8, 9, 10]))
    box = (float(cfg_get(cfg, "download.ssh.box.lat_min", 5.0)),
           float(cfg_get(cfg, "download.ssh.box.lat_max", 40.0)),
           float(cfg_get(cfg, "download.ssh.box.lon_min", -100.0)),
           float(cfg_get(cfg, "download.ssh.box.lon_max", -15.0)))

    sla_path = ocean_dir / f"ssh_sla_{year}.nc"
    tchp_path = tchp_dir / f"tchp_noaa_{year}.nc"
    if not sla_path.exists():
        return {"status": "skipped", "reason": f"SLA file missing: {sla_path}"}
    if not tchp_path.exists():
        return {"status": "skipped", "reason": f"TCHP file missing: {tchp_path}"}

    sla_ds = _open(sla_path)
    tchp_ds = _open(tchp_path)
    # Prefer ADT (absolute dynamic topography ~ total reservoir, comparable to absolute
    # TCHP) over SLA (anomaly, which removes the climatological warm pool).
    sla_var = _resolve_var(sla_ds, ["adt", "sla"])
    tchp_var = _resolve_var(tchp_ds, ["Tropical_Cyclone_Heat_Potential", "tchp", "TCHP"])
    logger.info("SLA validation using ocean variable '%s' vs TCHP '%s'", sla_var, tchp_var)

    events = _covered_events(cfg, year, months, box)
    if max_events:
        events = events[:max_events]

    dist_sla_tchp: List[float] = []
    dist_center_tchp: List[float] = []
    sla_center_all: List[float] = []
    tchp_center_all: List[float] = []
    sla_center_ri: List[float] = []
    sla_center_nonri: List[float] = []
    n_used = 0
    n_skipped = 0

    for ev in events:
        sw = _window(sla_ds, sla_var, ev["timestamp"], ev["center_lat"], ev["center_lon"], window_deg)
        tw = _window(tchp_ds, tchp_var, ev["timestamp"], ev["center_lat"], ev["center_lon"], window_deg)
        if sw is None or tw is None:
            n_skipped += 1
            continue
        sla2d, sla_lats, sla_lons = sw
        tchp2d, tchp_lats, tchp_lons = tw

        # Localize the LOCAL feature (high-passed), not the basin-scale warm edge.
        sla_peak = _local_peak(sla2d, sla_lats, sla_lons)
        tchp_peak = _local_peak(tchp2d, tchp_lats, tchp_lons)
        if sla_peak is None or tchp_peak is None:
            n_skipped += 1
            continue

        d_sla = haversine_distance(sla_peak[0], sla_peak[1], tchp_peak[0], tchp_peak[1])
        d_center = haversine_distance(ev["center_lat"], ev["center_lon"], tchp_peak[0], tchp_peak[1])
        dist_sla_tchp.append(d_sla)
        dist_center_tchp.append(d_center)

        sla_c = sample_at_center(sla2d, sla_lats, sla_lons, ev["center_lat"], ev["center_lon"])
        tchp_c = sample_at_center(tchp2d, tchp_lats, tchp_lons, ev["center_lat"], ev["center_lon"])
        if sla_c is not None:
            sla_center_all.append(sla_c)
            tchp_center_all.append(tchp_c if tchp_c is not None else np.nan)
            (sla_center_ri if ev["ri_label"] == 1 else sla_center_nonri).append(sla_c)
        n_used += 1

    report: Dict[str, Any] = {
        "status": "ok" if n_used > 0 else "no_events",
        "meta": {"year": year, "window_deg": window_deg, "season_months": months,
                 "ocean_variable": sla_var, "tchp_variable": tchp_var,
                 "localizer": "local-anomaly (basin background removed)",
                 "n_events_covered": len(events), "n_used": n_used, "n_skipped": n_skipped},
    }
    if n_used == 0:
        return report

    # 1. PROXY: SLA@centre vs TCHP@centre across events.
    proxy = {"status": "insufficient_data"}
    pairs = [(s, t) for s, t in zip(sla_center_all, tchp_center_all)
             if np.isfinite(s) and np.isfinite(t)]
    if len(pairs) >= 5:
        from scipy.stats import spearmanr
        s = np.array([p[0] for p in pairs]); t = np.array([p[1] for p in pairs])
        rho, pval = spearmanr(s, t)
        proxy = {"status": "ok", "n": len(pairs), "spearman_rho": float(rho),
                 "p_value": float(pval),
                 "interpretation": ("SLA tracks TCHP at the storm centre (positive, significant) "
                                    "-> SLA is a usable surface proxy for the subsurface reservoir."
                                    if (rho > 0 and pval < 0.05) else
                                    "SLA does NOT significantly track TCHP here.")}

    # 2. CO-LOCATION: SLA peak vs storm-centre baseline, both against TCHP peak.
    m = np.asarray(dist_sla_tchp); c = np.asarray(dist_center_tchp)
    coloc = {"status": "ok", "n": int(m.size),
             "sla_peak_to_tchp_mean_km": float(np.mean(m)),
             "center_to_tchp_mean_km": float(np.mean(c)),
             "sla_minus_center_km": float(np.mean(m - c)),
             "fraction_sla_closer": float(np.mean(m < c)),
             "interpretation": ("The SLA maximum locates the TCHP peak better than the storm centre."
                                if float(np.mean(m - c)) < 0 else
                                "The SLA maximum does NOT beat the storm-centre baseline.")}

    # 3. RI SIGNAL: SLA@centre for RI vs non-RI events.
    ri = {"status": "insufficient_data"}
    if len(sla_center_ri) >= 3 and len(sla_center_nonri) >= 3:
        from scipy.stats import mannwhitneyu
        u, pval = mannwhitneyu(sla_center_ri, sla_center_nonri, alternative="greater")
        ri = {"status": "ok",
              "n_ri": len(sla_center_ri), "n_nonri": len(sla_center_nonri),
              "sla_center_median_ri_m": float(np.median(sla_center_ri)),
              "sla_center_median_nonri_m": float(np.median(sla_center_nonri)),
              "mannwhitney_u": float(u), "p_value_ri_greater": float(pval),
              "interpretation": ("RI events sit over significantly higher SLA (warmer reservoir) "
                                 "than non-RI events."
                                 if pval < 0.05 else
                                 "No significant SLA difference between RI and non-RI events "
                                 "(small RI sample or weak signal).")}

    report["proxy_sla_vs_tchp"] = proxy
    report["colocation_vs_center_baseline"] = coloc
    report["ri_signal"] = ri
    return report


def run_and_save(cfg: Dict[str, Any], year: int, window_deg: float,
                 out_path: Path, max_events: Optional[int] = None) -> Dict[str, Any]:
    report = validate_sla_colocation(cfg, year=year, window_deg=window_deg, max_events=max_events)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("SLA validation report saved to %s", out_path)
    return report
