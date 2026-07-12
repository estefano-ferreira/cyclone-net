# analysis/fuelmap_tchp_validation.py
"""
FuelMap × TCHP co-location validation with explicit null models.

Tests whether the coordinate predicted by the model's FuelMap head
(`pred_lat`/`pred_lon` in test_predictions.csv, the "target lock") co-locates
with an INDEPENDENT physical reference: the audited TCHP peak produced by
`run.py preprocess-tchp` and stored in each event's interim metadata.

This validates a *plausible-correlate hypothesis* (the learned hotspot tends
to sit near the subsurface ocean-heat maximum). It does NOT establish
causality, and it inherits every limitation of the audited TCHP source
(public gridded TCHP exists from 2022 onward only).

Null models (both mandatory before interpreting any distance):
  1. RANDOM-POINT NULL — for each event, a point drawn uniformly inside that
     event's own lat/lon window. Monte Carlo distribution of the null median
     distance across events; one-sided p-value for "FuelMap is closer than a
     random point".
  2. STORM-CENTER BASELINE — distance from the storm center to the TCHP peak,
     compared pairwise with the FuelMap distance via a sign-flip permutation
     test on the paired differences (one-sided: FuelMap closer than center).

Reuses `src.evaluation.spatial_metrics` (haversine, audited-peak reader,
three-way skill comparison) instead of reimplementing them.

Usage:
    python analysis/fuelmap_tchp_validation.py [--n-null 10000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.spatial_metrics import (  # noqa: E402
    compute_spatial_metrics_from_predictions,
    haversine_distance,
)
from src.utils.config import cfg_get, load_config  # noqa: E402

ACCEPT_STATUSES = ("ok", "qc_flagged")


def load_eligible_events(pred_csv: Path, interim_dir: Path) -> pd.DataFrame:
    """Join test predictions with audited TCHP peaks and window bounds.

    An event is eligible when it has (a) a finite FuelMap coordinate,
    (b) an audited TCHP peak with acceptable status, and (c) grid files
    defining its spatial window (needed for the random-point null).
    """
    preds = pd.read_csv(pred_csv)
    rows = []
    for _, r in preds.iterrows():
        eid = str(r["event_id"])
        meta_path = interim_dir / f"{eid}.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tchp_lat, tchp_lon = meta.get("tchp_peak_lat"), meta.get("tchp_peak_lon")
        if tchp_lat is None or tchp_lon is None:
            continue
        status = (meta.get("tchp_audit") or {}).get("status")
        if status is not None and status not in ACCEPT_STATUSES:
            continue
        if not (np.isfinite(r["pred_lat"]) and np.isfinite(r["pred_lon"])):
            continue
        lats_path = interim_dir / f"{eid}_lats.npy"
        lons_path = interim_dir / f"{eid}_lons.npy"
        if not (lats_path.exists() and lons_path.exists()):
            continue
        lats = np.load(lats_path).astype(float)
        lons = np.load(lons_path).astype(float)
        rows.append({
            "event_id": eid,
            "y_true": int(r["y_true"]) if np.isfinite(r.get("y_true", np.nan)) else None,
            "pred_lat": float(r["pred_lat"]),
            "pred_lon": float(r["pred_lon"]),
            "tchp_lat": float(tchp_lat),
            "tchp_lon": float(tchp_lon),
            "center_lat": meta.get("center_lat"),
            "center_lon": meta.get("center_lon"),
            "lat_min": float(np.nanmin(lats)), "lat_max": float(np.nanmax(lats)),
            "lon_min": float(np.nanmin(lons)), "lon_max": float(np.nanmax(lons)),
        })
    return pd.DataFrame(rows)


def random_point_null(df: pd.DataFrame, n_null: int, rng: np.random.Generator) -> dict:
    """Monte Carlo null: median distance from a uniform random point in each
    event's window to that event's TCHP peak.

    Sampling is uniform in the lat/lon box. Over a ±5° window the cos(lat)
    area distortion is small relative to the effect sizes of interest; this
    simplification is stated rather than hidden.
    """
    observed = df["fuel_to_tchp_km"].to_numpy()
    obs_median = float(np.median(observed))

    n = len(df)
    lat_min = df["lat_min"].to_numpy()
    lat_span = (df["lat_max"] - df["lat_min"]).to_numpy()
    lon_min = df["lon_min"].to_numpy()
    lon_span = (df["lon_max"] - df["lon_min"]).to_numpy()
    tchp_lat = df["tchp_lat"].to_numpy()
    tchp_lon = df["tchp_lon"].to_numpy()

    null_medians = np.empty(n_null)
    for b in range(n_null):
        rl = lat_min + rng.random(n) * lat_span
        rg = lon_min + rng.random(n) * lon_span
        d = np.array([
            haversine_distance(rl[i], rg[i], tchp_lat[i], tchp_lon[i]) for i in range(n)
        ])
        null_medians[b] = np.median(d)

    # One-sided Monte Carlo p-value with the standard +1 correction:
    # probability that a random-point median is <= the observed median.
    p = float((1 + np.sum(null_medians <= obs_median)) / (n_null + 1))
    return {
        "observed_median_km": obs_median,
        "null_median_km_mean": float(np.mean(null_medians)),
        "null_median_km_p2.5": float(np.percentile(null_medians, 2.5)),
        "null_median_km_p97.5": float(np.percentile(null_medians, 97.5)),
        "p_value_one_sided": p,
        "n_null_replicates": int(n_null),
    }


def signflip_vs_center(df: pd.DataFrame, n_perm: int, rng: np.random.Generator) -> dict:
    """Sign-flip permutation test on paired differences
    (fuel_to_tchp - center_to_tchp); one-sided for FuelMap closer than center."""
    paired = df.dropna(subset=["center_to_tchp_km"])
    diffs = (paired["fuel_to_tchp_km"] - paired["center_to_tchp_km"]).to_numpy()
    n = len(diffs)
    if n < 3:
        return {"status": "insufficient_pairs", "n_pairs": int(n)}
    obs_mean = float(np.mean(diffs))
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null_means = (signs * np.abs(diffs)).mean(axis=1)
    p = float((1 + np.sum(null_means <= obs_mean)) / (n_perm + 1))
    return {
        "status": "ok",
        "n_pairs": int(n),
        "mean_paired_difference_km": obs_mean,
        "median_fuel_km": float(np.median(paired["fuel_to_tchp_km"])),
        "median_center_km": float(np.median(paired["center_to_tchp_km"])),
        "fraction_fuel_closer": float(np.mean(diffs < 0)),
        "p_value_one_sided": p,
        "n_permutations": int(n_perm),
    }


def summarize(df: pd.DataFrame, label: str, n_null: int, rng: np.random.Generator) -> dict:
    """Full summary (distances + both nulls) for one event subset."""
    if len(df) == 0:
        return {"subset": label, "n": 0, "status": "no_eligible_events"}
    d = df["fuel_to_tchp_km"].to_numpy()
    out = {
        "subset": label,
        "n": int(len(df)),
        "underpowered": bool(len(df) < 10),
        "median_fuel_to_tchp_km": float(np.median(d)),
        "mean_fuel_to_tchp_km": float(np.mean(d)),
        "fraction_within_100km": float(np.mean(d <= 100.0)),
        "fraction_within_2deg_222km": float(np.mean(d <= 222.4)),
        "random_point_null": random_point_null(df, n_null, rng),
        "vs_storm_center": signflip_vs_center(df, n_null, rng),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-null", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    interim_dir = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    results_dir = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve()
    pred_csv = results_dir / "test_predictions.csv"

    rng = np.random.default_rng(args.seed)

    df = load_eligible_events(pred_csv, interim_dir)
    n_total_preds = len(pd.read_csv(pred_csv))
    print(f"AVAILABILITY: {len(df)}/{n_total_preds} test events eligible "
          f"(FuelMap coord + audited TCHP peak + grid window)")
    if len(df) == 0:
        print("No eligible events — run `python run.py preprocess-tchp` first.")
        sys.exit(1)

    df["fuel_to_tchp_km"] = [
        haversine_distance(r.pred_lat, r.pred_lon, r.tchp_lat, r.tchp_lon)
        for r in df.itertuples()
    ]
    df["center_to_tchp_km"] = [
        haversine_distance(float(r.center_lat), float(r.center_lon), r.tchp_lat, r.tchp_lon)
        if r.center_lat is not None and r.center_lon is not None else np.nan
        for r in df.itertuples()
    ]

    report = {
        "protocol": "FuelMap x TCHP co-location with random-point and storm-center nulls",
        "seed": args.seed,
        "availability": {
            "n_test_predictions": int(n_total_preds),
            "n_eligible": int(len(df)),
            "n_eligible_ri": int((df["y_true"] == 1).sum()),
            "note": "Public gridded TCHP exists from 2022 onward; earlier events cannot be validated.",
        },
        "all_eligible": summarize(df, "all_eligible", args.n_null, rng),
        "ri_only": summarize(df[df["y_true"] == 1], "ri_only", args.n_null, rng),
        # Standard audited three-way comparison (model / center / physics flux),
        # reused from the released evaluation pipeline for consistency.
        "three_way_skill": compute_spatial_metrics_from_predictions(
            pd.read_csv(pred_csv), interim_dir
        ),
        "interpretation_guard": (
            "This analysis tests a plausible-correlate hypothesis. A small p-value "
            "means the FuelMap sits closer to the TCHP peak than the null; it does "
            "not establish that the model uses subsurface ocean heat causally."
        ),
    }

    out_dir = results_dir / "spatial"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "fuelmap_tchp_validation.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    df.to_csv(out_dir / "fuelmap_tchp_pairs.csv", index=False)
    print(json.dumps({k: report[k] for k in ("availability", "all_eligible", "ri_only")}, indent=2))
    print(f"\nreport: {out_json}")


if __name__ == "__main__":
    main()
