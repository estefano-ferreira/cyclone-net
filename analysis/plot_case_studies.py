#!/usr/bin/env python3
# scripts/plot_case_studies.py
"""
Generate maps for case studies: FuelMap, TCHP, and storm track overlay.
Saves figures to docs/figures/.
"""

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from src.utils.config import load_config, cfg_get
from src.utils.tchp_utils import get_tchp_file_path


def load_fuelmap(event_id: str, interim_dir: Path):
    """Load FuelMap (probability map) and coordinates for an event."""
    fuelmap_path = interim_dir / f"{event_id}_fuel_potential.npy"
    if not fuelmap_path.exists():
        raise FileNotFoundError(f"FuelMap not found: {fuelmap_path}")
    fuelmap = np.load(fuelmap_path)  # (H,W,T) or (H,W)
    if fuelmap.ndim == 3:
        fuelmap = fuelmap[:, :, 0]  # use t0
    lats = np.load(interim_dir / f"{event_id}_lats.npy")
    lons = np.load(interim_dir / f"{event_id}_lons.npy")
    return fuelmap, lats, lons


def load_tchp_for_event(meta: dict, tchp_dir: Path, window_deg: float = 5.0):
    """Load TCHP map for the event."""
    timestamp = np.datetime64(meta["timestamp"])
    lat, lon = meta["center_lat"], meta["center_lon"]
    year = pd.to_datetime(timestamp).year
    if year >= 2022:
        src = "noaa"
    elif year >= 1993:
        src = "aoml"
    else:
        return None
    tchp_file = get_tchp_file_path(tchp_dir, year, src)
    if not tchp_file.exists():
        return None
    ds = xr.open_dataset(tchp_file)
    ds_t = ds.sel(time=timestamp, method="nearest")
    lon_min, lon_max = lon - window_deg, lon + window_deg
    lat_min, lat_max = lat - window_deg, lat + window_deg
    if "lon" in ds_t.coords and ds_t.lon.min() >= 0 and lon_min < 0:
        lon_min += 360
        lon_max += 360
    ds_region = ds_t.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    if ds_region.sizes["lat"] == 0 or ds_region.sizes["lon"] == 0:
        ds.close()
        return None
    var_names = ["tchp", "Tropical_Cyclone_Heat_Potential", "TCHP"]
    for v in var_names:
        if v in ds_region:
            tchp = ds_region[v].values
            lats = ds_region["lat"].values
            lons = ds_region["lon"].values
            ds.close()
            return tchp, lats, lons
    ds.close()
    return None


def plot_case(
    event_id: str,
    interim_dir: Path,
    tchp_dir: Path,
    output_dir: Path,
    track_df=None,
):
    """Generate and save a map for the given event."""
    # Load metadata
    meta_path = interim_dir / f"{event_id}.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Load FuelMap
    fuelmap, lats_fm, lons_fm = load_fuelmap(event_id, interim_dir)

    # Load TCHP
    tchp_data = load_tchp_for_event(meta, tchp_dir)
    if tchp_data is None:
        print(f"TCHP not available for {event_id}, skipping.")
        return

    tchp, lats_tchp, lons_tchp = tchp_data

    # Create figure
    fig = plt.figure(figsize=(12, 5))

    # FuelMap subplot
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax1.set_extent([lons_fm.min(), lons_fm.max(), lats_fm.min(), lats_fm.max()], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.BORDERS, linestyle=":")
    ax1.add_feature(cfeature.LAND, color="lightgray")
    im1 = ax1.pcolormesh(lons_fm, lats_fm, fuelmap, cmap="hot", shading="auto", alpha=0.8)
    plt.colorbar(im1, ax=ax1, label="FuelMap")
    ax1.plot(meta["center_lon"], meta["center_lat"], "ro", markersize=8, label="Storm center")
    ax1.set_title(f"FuelMap - {event_id}")

    # TCHP subplot
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax2.set_extent([lons_tchp.min(), lons_tchp.max(), lats_tchp.min(), lats_tchp.max()], crs=ccrs.PlateCarree())
    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.BORDERS, linestyle=":")
    ax2.add_feature(cfeature.LAND, color="lightgray")
    im2 = ax2.pcolormesh(lons_tchp, lats_tchp, tchp, cmap="viridis", shading="auto", alpha=0.8)
    plt.colorbar(im2, ax=ax2, label="TCHP (kJ/cm²)")
    ax2.plot(meta["center_lon"], meta["center_lat"], "ro", markersize=8, label="Storm center")
    ax2.set_title(f"TCHP - {event_id}")

    # Add track if provided
    # (optional: plot past/future positions)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{event_id}_map.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate case study maps.")
    parser.add_argument("event_ids", nargs="+", help="Event IDs to plot")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--output", default="docs/figures", help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    interim_dir = Path(cfg["paths"]["interim_data"])
    tchp_dir = Path(cfg["paths"]["tchp_dir"])

    output_dir = Path(args.output)

    for eid in args.event_ids:
        try:
            plot_case(eid, interim_dir, tchp_dir, output_dir)
        except Exception as e:
            print(f"Error processing {eid}: {e}")


if __name__ == "__main__":
    main()