# analysis/fuelmap_displacement_dynamics.py
"""
Dynamic FuelMap displacement test.

The static co-location test (analysis/fuelmap_tchp_validation.py) asked WHERE
the FuelMap peak sits on average. This test asks whether the peak's
DISPLACEMENT from the storm center carries dynamical information:

  Q1. Magnitude: does the FuelMap displace FARTHER from the center at
      moments of intensification (|dv24| large, RI events) than otherwise?
  Q2. Direction: does the displacement bearing point systematically toward
      environmental structure — the SST maximum, the mid-level-humidity
      maximum, or the deep-layer-shear minimum in the same window?

Null models:
  * Magnitude: label permutation (RI flags shuffled across events).
  * Direction: uniform-bearing null via permutation of the pairing between
    FuelMap bearings and environmental-target bearings (concentration of the
    angular differences compared with shuffled pairings).

Honesty rules: out-of-sample only (test split predictions), N audited before
any conclusion, and a near-zero median displacement is itself a finding — it
confirms dynamically that the FuelMap is "position only".

Sources (--source):
  model — the learned FuelMap peak (pred_lat/pred_lon from test_predictions.csv)
  prior — CONTROL: the pure physics prior's peak (t0 slice of
          {event_id}_fuel_potential.npy, located with robust_peak_2d), with no
          neural network involved. If the prior shows the same dynamics as the
          model, the behavior is arithmetic of the flux formula, not learning.

Usage:
    python analysis/fuelmap_displacement_dynamics.py [--source model|prior]
        [--seed 42] [--n-perm 10000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.spatial_metrics import haversine_distance, robust_peak_2d  # noqa: E402
from src.utils.config import cfg_get, load_config  # noqa: E402


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees in [0, 360)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlon)
    return float((np.degrees(np.arctan2(x, y)) + 360.0) % 360.0)


def mean_resultant_length(angles_deg: np.ndarray) -> float:
    """Circular concentration of angles: 0 = uniform, 1 = perfectly aligned."""
    a = np.radians(angles_deg)
    return float(np.hypot(np.cos(a).mean(), np.sin(a).mean()))


def channel_peak_bearing(cube, lats, lons, channels, channel, center, minimum=False):
    """Bearing from the storm center to the t0 extreme of one cube channel."""
    if channel not in channels:
        return None
    field = np.asarray(cube[:, :, 0, channels.index(channel)], dtype=float)
    if minimum:
        field = -field
    peak = robust_peak_2d(field, lats, lons)
    if peak is None:
        return None
    plat, plon = peak[0], peak[1]
    if abs(plat - center[0]) < 1e-6 and abs(plon - center[1]) < 1e-6:
        return None  # peak exactly at center: bearing undefined
    return bearing_deg(center[0], center[1], plat, plon)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["model", "prior"], default="model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-perm", type=int, default=10_000)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    interim = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    results_dir = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve()

    preds = pd.read_csv(results_dir / "test_predictions.csv")

    rows = []
    n_meta_missing = 0
    n_prior_missing = 0
    for r in preds.itertuples():
        meta_path = interim / f"{r.event_id}.json"
        if not (np.isfinite(r.pred_lat) and np.isfinite(r.pred_lon)) or not meta_path.exists():
            n_meta_missing += 1
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        c_lat, c_lon = meta.get("center_lat"), meta.get("center_lon")
        if c_lat is None or c_lon is None:
            continue

        lats_path = interim / f"{r.event_id}_lats.npy"
        lons_path = interim / f"{r.event_id}_lons.npy"
        if not (lats_path.exists() and lons_path.exists()):
            continue
        lats = np.load(lats_path)
        lons = np.load(lons_path)

        # Coordinate under test: learned FuelMap peak, or the pure physics
        # prior's peak (control — no neural network involved).
        if args.source == "model":
            peak_lat, peak_lon = float(r.pred_lat), float(r.pred_lon)
        else:
            prior_path = interim / f"{r.event_id}_fuel_potential.npy"
            if not prior_path.exists():
                n_prior_missing += 1
                continue
            prior = np.load(prior_path, mmap_mode="r")
            peak = robust_peak_2d(np.asarray(prior[:, :, 0], dtype=float), lats, lons)
            if peak is None:
                n_prior_missing += 1
                continue
            peak_lat, peak_lon = peak[0], peak[1]

        entry = {
            "event_id": r.event_id,
            "y_true": int(r.y_true),
            "dv24_kt": meta.get("dv24_kt"),
            "peak_lat": peak_lat,
            "peak_lon": peak_lon,
            "disp_km": haversine_distance(float(c_lat), float(c_lon), peak_lat, peak_lon),
            "bearing_fuel": (bearing_deg(float(c_lat), float(c_lon), peak_lat, peak_lon)
                             if (abs(peak_lat - c_lat) > 1e-6 or abs(peak_lon - c_lon) > 1e-6)
                             else None),
        }
        # Environmental target bearings from the SAME event window (t0 slice).
        cube_path = interim / f"{r.event_id}.npy"
        if cube_path.exists():
            cube = np.load(cube_path, mmap_mode="r")
            channels = list(meta.get("channels", []))
            center = (float(c_lat), float(c_lon))
            entry["bearing_sst_max"] = channel_peak_bearing(cube, lats, lons, channels, "sst_K", center)
            entry["bearing_rh_max"] = channel_peak_bearing(cube, lats, lons, channels, "rh_mid", center)
            entry["bearing_shear_min"] = channel_peak_bearing(cube, lats, lons, channels,
                                                              "shear_850_200_mps", center, minimum=True)
        rows.append(entry)

    df = pd.DataFrame(rows)
    df_dv = df.dropna(subset=["dv24_kt"])
    n_ri = int(df["y_true"].sum())

    audit = {
        "source": args.source,
        "n_test_predictions": int(len(preds)),
        "n_used": int(len(df)),
        "n_missing_meta_or_pred": int(n_meta_missing),
        "n_missing_prior": int(n_prior_missing),
        "n_with_observed_dv24": int(len(df_dv)),
        "n_ri": n_ri,
        "n_with_shear_channel": int(df["bearing_shear_min"].notna().sum()) if "bearing_shear_min" in df else 0,
        "underpowered_ri": bool(n_ri < 10),
    }
    print("AVAILABILITY AUDIT:", json.dumps(audit, indent=2))

    disp = df["disp_km"].to_numpy()
    report = {
        "protocol": "FuelMap displacement dynamics (out-of-sample test split)",
        "source": args.source,
        "seed": args.seed,
        "availability": audit,
        "displacement_km": {
            "median": float(np.median(disp)),
            "mean": float(np.mean(disp)),
            "p10": float(np.percentile(disp, 10)),
            "p90": float(np.percentile(disp, 90)),
            "fraction_below_50km": float(np.mean(disp < 50.0)),
            "note": "the 40x40px window spans roughly +/-5 degrees (~550 km half-width)",
        },
    }

    # Q1a: magnitude vs |dv24| (continuous)
    rho, p_rho = spearmanr(df_dv["disp_km"], df_dv["dv24_kt"].abs())
    report["magnitude_vs_abs_dv24"] = {
        "spearman_rho": float(rho), "p_value": float(p_rho), "n": int(len(df_dv)),
    }

    # Q1b: RI vs non-RI displacement magnitude (one-sided permutation, RI larger)
    ri_disp = df[df["y_true"] == 1]["disp_km"].to_numpy()
    non_disp = df[df["y_true"] == 0]["disp_km"].to_numpy()
    obs_diff = float(np.median(ri_disp) - np.median(non_disp)) if len(ri_disp) else None
    p_perm = None
    if len(ri_disp) >= 3:
        pool = np.concatenate([ri_disp, non_disp])
        k = len(ri_disp)
        null = np.empty(args.n_perm)
        for b in range(args.n_perm):
            rng.shuffle(pool)
            null[b] = np.median(pool[:k]) - np.median(pool[k:])
        p_perm = float((1 + np.sum(null >= obs_diff)) / (args.n_perm + 1))
    report["ri_vs_nonri_displacement"] = {
        "median_ri_km": float(np.median(ri_disp)) if len(ri_disp) else None,
        "median_nonri_km": float(np.median(non_disp)),
        "median_difference_km": obs_diff,
        "p_value_one_sided_ri_larger": p_perm,
        "n_ri": int(len(ri_disp)), "n_nonri": int(len(non_disp)),
    }

    # Q2: directional alignment with environmental targets (permutation null:
    # shuffle the pairing, compare concentration of angular differences).
    directions = {}
    for target in ["bearing_sst_max", "bearing_rh_max", "bearing_shear_min"]:
        sub = df.dropna(subset=["bearing_fuel", target])
        if len(sub) < 20:
            directions[target] = {"status": "insufficient_n", "n": int(len(sub))}
            continue
        delta = (sub["bearing_fuel"].to_numpy() - sub[target].to_numpy()) % 360.0
        obs_r = mean_resultant_length(delta)
        bf = sub["bearing_fuel"].to_numpy().copy()
        tg = sub[target].to_numpy()
        null_r = np.empty(args.n_perm)
        for b in range(args.n_perm):
            rng.shuffle(bf)
            null_r[b] = mean_resultant_length((bf - tg) % 360.0)
        p_dir = float((1 + np.sum(null_r >= obs_r)) / (args.n_perm + 1))
        directions[target] = {
            "n": int(len(sub)),
            "mean_resultant_length": obs_r,   # 0 = no alignment, 1 = perfect
            "mean_angular_difference_deg": float(np.degrees(np.arctan2(
                np.sin(np.radians(delta)).mean(), np.cos(np.radians(delta)).mean())) % 360.0),
            "p_value_vs_shuffled_pairing": p_dir,
        }
    report["directional_alignment"] = directions

    # Verdict logic (stated, not implied)
    mag_signal = (report["magnitude_vs_abs_dv24"]["p_value"] < 0.05
                  or (p_perm is not None and p_perm < 0.05))
    dir_signal = any(isinstance(d, dict) and d.get("p_value_vs_shuffled_pairing", 1) < 0.05
                     for d in directions.values())
    if mag_signal or dir_signal:
        verdict = ("DYNAMIC SIGNAL: the FuelMap displacement correlates with "
                   "intensification and/or points toward environmental structure.")
    else:
        verdict = ("POSITION ONLY (dynamically confirmed): the displacement is neither "
                   "correlated with dv24 nor systematically directed — consistent with "
                   "the static finding that the FuelMap does not beat the storm center.")
    report["verdict"] = verdict

    out = results_dir / "spatial" / f"fuelmap_displacement_dynamics_{args.source}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    df.to_csv(results_dir / "spatial" / f"fuelmap_displacement_points_{args.source}.csv", index=False)

    print(json.dumps({k: report[k] for k in
                      ("displacement_km", "magnitude_vs_abs_dv24",
                       "ri_vs_nonri_displacement", "directional_alignment", "verdict")},
                     indent=2))
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
