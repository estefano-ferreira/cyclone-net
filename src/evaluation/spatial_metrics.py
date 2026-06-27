# src/evaluation/spatial_metrics.py
"""
Spatial metrics for validating FuelMap against external proxies (e.g., TCHP).
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def peak_distance(
    pred_lat: float,
    pred_lon: float,
    true_lat: float,
    true_lon: float
) -> float:
    """Distance between predicted and true peak locations."""
    return haversine_distance(pred_lat, pred_lon, true_lat, true_lon)


def top_k_overlap(
    fuelmap: np.ndarray,
    proxy_map: np.ndarray,
    k: int = 10
) -> float:
    """
    Fraction of top-k fuelmap pixels that are also in top-k proxy pixels.
    Both fuelmap and proxy_map are 2D arrays.
    """
    fuel_flat = fuelmap.flatten()
    proxy_flat = proxy_map.flatten()

    fuel_top = np.argsort(fuel_flat)[-k:]
    proxy_top = np.argsort(proxy_flat)[-k:]

    overlap = len(set(fuel_top).intersection(set(proxy_top)))
    return overlap / k


def rank_correlation(
    fuelmap: np.ndarray,
    proxy_map: np.ndarray
) -> float:
    """
    Spearman rank correlation between fuelmap and proxy map values.
    """
    fuel_flat = fuelmap.flatten()
    proxy_flat = proxy_map.flatten()
    # Remove NaN or invalid values
    valid = np.isfinite(fuel_flat) & np.isfinite(proxy_flat)
    if valid.sum() < 2:
        return float('nan')
    corr, _ = spearmanr(fuel_flat[valid], proxy_flat[valid])
    return float(corr)


def compute_spatial_metrics(
    pred_lat: float,
    pred_lon: float,
    true_lat: float,
    true_lon: float,
    fuelmap: np.ndarray,
    proxy_map: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all spatial metrics for a single event.
    """
    metrics = {
        "peak_distance_km": peak_distance(pred_lat, pred_lon, true_lat, true_lon),
        "top10_overlap": top_k_overlap(fuelmap, proxy_map, k=10),
        "rank_correlation": rank_correlation(fuelmap, proxy_map),
    }
    return metrics


def _nan_safe_gaussian(field: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaNs (normalize by a smoothed validity mask)."""
    from scipy.ndimage import gaussian_filter

    if sigma <= 0:
        return field
    mask = np.isfinite(field).astype(float)
    filled = np.where(np.isfinite(field), field, 0.0)
    num = gaussian_filter(filled, sigma=sigma)
    den = gaussian_filter(mask, sigma=sigma)
    out = np.full_like(num, np.nan, dtype=float)
    valid = den > 1e-12
    out[valid] = num[valid] / den[valid]
    return out


def robust_peak_2d(
    field2d: np.ndarray,
    lats2d: np.ndarray,
    lons2d: np.ndarray,
    smooth_sigma: float = 1.0,
) -> Optional[Tuple[float, float, float]]:
    """
    Robustly locate the peak of a 2D physical field on a 2D lat/lon grid.

    NaN-safe smoothing then global argmax of the smoothed field. Returns
    (peak_lat, peak_lon, peak_value) or None if the field is entirely NaN.
    Used to extract the energy-source location DIRECTLY from physics (e.g. the
    air-sea enthalpy-flux field) with no neural network involved.
    """
    field = np.asarray(field2d, dtype=float)
    if not np.isfinite(field).any():
        return None
    smooth = _nan_safe_gaussian(field, smooth_sigma)
    if not np.isfinite(smooth).any():
        return None
    i, j = np.unravel_index(int(np.nanargmax(smooth)), smooth.shape)
    val = float(field[i, j]) if np.isfinite(field[i, j]) else float(smooth[i, j])
    return float(lats2d[i, j]), float(lons2d[i, j]), val


def physics_flux_peak(
    event_id: str,
    interim_dir: Path,
    flux_channel: str = "total_heat_flux_Wpm2",
    t0: int = 0,
) -> Optional[Tuple[float, float, float]]:
    """
    Compute the physically-derived energy-source location for one event: the peak
    of the air-sea enthalpy-flux field (latent+sensible) stored in the cube. This
    is the energy INPUT to the storm computed straight from bulk physics, used as a
    no-neural-network reference in the spatial validation.
    """
    interim_dir = Path(interim_dir)
    cube_path = interim_dir / f"{event_id}.npy"
    meta_path = interim_dir / f"{event_id}.json"
    lats_path = interim_dir / f"{event_id}_lats.npy"
    lons_path = interim_dir / f"{event_id}_lons.npy"
    if not (cube_path.exists() and meta_path.exists() and lats_path.exists() and lons_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        channels = list(meta.get("channels", []))
        if flux_channel not in channels:
            return None
        cube = np.load(cube_path)  # (H,W,T,C)
        ci = channels.index(flux_channel)
        field = cube[:, :, t0, ci].astype(float)
        lats = np.load(lats_path).astype(float)
        lons = np.load(lons_path).astype(float)
        return robust_peak_2d(field, lats, lons)
    except Exception as exc:
        logger.warning("physics_flux_peak failed for %s: %s", event_id, exc)
        return None


def _aggregate_distances(distances: List[float]) -> Dict[str, Any]:
    """Summarize a list of great-circle distances (km), NaN-safe."""
    arr = np.asarray([d for d in distances if np.isfinite(d)], dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean_km": None, "median_km": None, "std_km": None,
                "min_km": None, "max_km": None, "p90_km": None}
    return {
        "n": int(arr.size),
        "mean_km": float(np.mean(arr)),
        "median_km": float(np.median(arr)),
        "std_km": float(np.std(arr)),
        "min_km": float(np.min(arr)),
        "max_km": float(np.max(arr)),
        "p90_km": float(np.percentile(arr, 90)),
    }


def compute_spatial_metrics_from_predictions(
    pred_df: "pd.DataFrame",
    interim_dir: Path,
    accept_statuses: Tuple[str, ...] = ("ok", "qc_flagged"),
    physics_flux_channel: Optional[str] = "total_heat_flux_Wpm2",
) -> Dict[str, Any]:
    """
    Validate the model's predicted thermodynamic hotspot against the TCHP peak.

    Single source of truth: the TCHP peak (`tchp_peak_lat`/`tchp_peak_lon`) and its
    audit block are produced ONCE by `run.py preprocess-tchp` (see
    src/processors/preprocess_tchp.py) and stored in each event's metadata JSON.
    This function does NOT re-open TCHP NetCDFs; it reuses that audited output so
    the validation is reproducible and consistent with the preprocessing audit trail.

    For each prediction row with `pred_lat`/`pred_lon` it computes:
      - model_to_tchp_km: distance from the predicted FuelMap peak to the TCHP peak
        (the quantity we want to be small).
      - center_to_tchp_km: distance from the storm centre to the TCHP peak. This is a
        NAIVE baseline ("just predict the storm centre"). The model demonstrates
        spatial skill only if model_to_tchp_km < center_to_tchp_km on average.

    Returns an auditable dict with overall stats, the skill comparison, RI/non-RI
    breakdown, and coverage (how many events had a usable TCHP peak).

    Args:
        pred_df: predictions frame from evaluate.py (needs event_id, pred_lat, pred_lon;
                 optional y_true for the RI breakdown).
        interim_dir: directory holding the per-event metadata JSON files.
    """
    interim_dir = Path(interim_dir)
    required = {"event_id", "pred_lat", "pred_lon"}
    if not required.issubset(pred_df.columns):
        return {"status": "skipped",
                "reason": f"pred_df missing columns: {sorted(required - set(pred_df.columns))}"}

    model_dists: List[float] = []
    center_dists: List[float] = []
    physics_dists: List[float] = []
    model_dists_ri: List[float] = []
    model_dists_nonri: List[float] = []
    rows_paired: List[Dict[str, Any]] = []

    n_total = len(pred_df)
    n_meta_missing = 0
    n_no_tchp = 0
    n_rejected_status = 0
    n_used = 0

    for _, row in pred_df.iterrows():
        eid = str(row["event_id"])
        meta_path = interim_dir / f"{eid}.json"
        if not meta_path.exists():
            n_meta_missing += 1
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read metadata %s: %s", meta_path, exc)
            n_meta_missing += 1
            continue

        tchp_lat = meta.get("tchp_peak_lat")
        tchp_lon = meta.get("tchp_peak_lon")
        if tchp_lat is None or tchp_lon is None:
            n_no_tchp += 1
            continue

        status = (meta.get("tchp_audit") or {}).get("status")
        if status is not None and status not in accept_statuses:
            n_rejected_status += 1
            continue

        pred_lat = row.get("pred_lat")
        pred_lon = row.get("pred_lon")
        if pred_lat is None or pred_lon is None or not (np.isfinite(pred_lat) and np.isfinite(pred_lon)):
            continue

        d_model = haversine_distance(float(pred_lat), float(pred_lon),
                                     float(tchp_lat), float(tchp_lon))
        model_dists.append(d_model)

        d_center: Optional[float] = None
        c_lat, c_lon = meta.get("center_lat"), meta.get("center_lon")
        if c_lat is not None and c_lon is not None:
            d_center = haversine_distance(float(c_lat), float(c_lon),
                                          float(tchp_lat), float(tchp_lon))
            center_dists.append(d_center)

        # Physics-only reference: peak of the air-sea enthalpy-flux field (no NN).
        d_physics: Optional[float] = None
        if physics_flux_channel:
            fp = physics_flux_peak(eid, interim_dir, flux_channel=physics_flux_channel)
            if fp is not None:
                d_physics = haversine_distance(fp[0], fp[1], float(tchp_lat), float(tchp_lon))
                physics_dists.append(d_physics)

        y_true = row.get("y_true")
        if y_true is not None and np.isfinite(y_true):
            (model_dists_ri if int(y_true) == 1 else model_dists_nonri).append(d_model)

        rows_paired.append({
            "event_id": eid,
            "pred_lat": float(pred_lat), "pred_lon": float(pred_lon),
            "tchp_peak_lat": float(tchp_lat), "tchp_peak_lon": float(tchp_lon),
            "model_to_tchp_km": d_model,
            "center_to_tchp_km": d_center,
            "physics_flux_to_tchp_km": d_physics,
        })
        n_used += 1

    overall = _aggregate_distances(model_dists)

    # Three-way skill comparison against the TCHP peak, on events that have all of
    # model / storm-centre / physics-flux distances available:
    #   - storm-centre  : naive baseline ("just predict where the storm is")
    #   - physics-flux  : energy-source from bulk physics, NO neural network
    #   - model FuelMap : the learned localization
    # The learned model only earns its keep if it beats BOTH references.
    skill: Dict[str, Any] = {"status": "insufficient_data"}
    triple = [
        (r["model_to_tchp_km"], r["center_to_tchp_km"], r["physics_flux_to_tchp_km"])
        for r in rows_paired
        if r["center_to_tchp_km"] is not None and r["physics_flux_to_tchp_km"] is not None
    ]
    # Fall back to the model-vs-centre pair when physics flux is unavailable.
    pair_mc = [(r["model_to_tchp_km"], r["center_to_tchp_km"])
               for r in rows_paired if r["center_to_tchp_km"] is not None]
    if triple:
        m = np.array([t[0] for t in triple], dtype=float)
        c = np.array([t[1] for t in triple], dtype=float)
        p = np.array([t[2] for t in triple], dtype=float)
        skill = {
            "status": "ok",
            "n_paired": int(len(triple)),
            "model_mean_km": float(np.mean(m)),
            "center_baseline_mean_km": float(np.mean(c)),
            "physics_flux_mean_km": float(np.mean(p)),
            "model_minus_center_km": float(np.mean(m - c)),
            "model_minus_physics_km": float(np.mean(m - p)),
            "fraction_model_beats_center": float(np.mean(m < c)),
            "fraction_model_beats_physics": float(np.mean(m < p)),
            "note": ("Negative model_minus_* means the model is closer to the TCHP peak "
                     "than that reference. The learned FuelMap demonstrates value only if "
                     "it beats BOTH the storm-centre baseline and the physics-only flux peak."),
        }
    elif pair_mc:
        m = np.array([q[0] for q in pair_mc], dtype=float)
        c = np.array([q[1] for q in pair_mc], dtype=float)
        skill = {
            "status": "ok_no_physics_flux",
            "n_paired": int(len(pair_mc)),
            "model_mean_km": float(np.mean(m)),
            "center_baseline_mean_km": float(np.mean(c)),
            "model_minus_center_km": float(np.mean(m - c)),
            "fraction_model_beats_center": float(np.mean(m < c)),
        }

    return {
        "status": "ok" if n_used > 0 else "no_paired_events",
        "distance_model_to_tchp": overall,
        "distance_physics_flux_to_tchp": _aggregate_distances(physics_dists),
        "distance_center_to_tchp": _aggregate_distances(center_dists),
        "skill_comparison": skill,
        "by_class": {
            "ri": _aggregate_distances(model_dists_ri),
            "non_ri": _aggregate_distances(model_dists_nonri),
        },
        "coverage": {
            "n_predictions": int(n_total),
            "n_used": int(n_used),
            "n_metadata_missing": int(n_meta_missing),
            "n_without_tchp_peak": int(n_no_tchp),
            "n_rejected_by_audit_status": int(n_rejected_status),
            "n_with_physics_flux": int(len(physics_dists)),
            "accept_statuses": list(accept_statuses),
        },
        "source": "tchp_peak from interim metadata (run.py preprocess-tchp); physics flux from cube channel "
                  + str(physics_flux_channel),
    }