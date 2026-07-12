#!/usr/bin/env python
"""
Build static forensic web platform data from pipeline outputs.

Generates:
  - events_index.json: summary of all events
  - events/<sid>.geojson: one GeoJSON file per storm with track and timesteps
  - definitions.json: citations and definitions for UI
  - manifest.json: build metadata and SHA256 checksums
"""

import sys
import os
import json
import math
import hashlib
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_git_hash() -> str:
    """Get short git commit hash; fallback to 'nogit' if not in repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "nogit"


def get_ibtracs_mtime() -> str:
    """Get IBTrACS raw file modification time as ISO 8601 UTC."""
    ibtracs_path = PROJECT_ROOT / "data" / "raw" / "ibtracs.ALL.list.v04r00.csv"
    if ibtracs_path.exists():
        mtime = ibtracs_path.stat().st_mtime
        dt = datetime.utcfromtimestamp(mtime)
        return dt.isoformat() + "Z"
    return "unknown"


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def round_floats(value: Any, decimals: int) -> Any:
    """Round floats to specified decimals; pass through other types."""
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return round(value, decimals)
    return value


def sanitize(obj: Any) -> Any:
    """Recursively replace NaN/Inf/NaT with None so output is strict JSON.

    Python's json module happily emits literal `NaN`, which is NOT valid JSON
    and makes browsers' JSON.parse throw. Pandas also silently re-coerces None
    back to NaN inside float columns, so sanitization must happen HERE, on the
    final Python objects, immediately before serialization.
    """
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    # Catch remaining pandas/numpy missing scalars (NaT, pd.NA, np.nan-as-object)
    try:
        if obj is not None and not isinstance(obj, (str, int, bool)) and pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def build_events():
    """Main build process."""
    # Paths
    augmented_csv = PROJECT_ROOT / "data" / "event_list_augmented.csv"
    config_file = PROJECT_ROOT / "config.yaml"
    tmp_dir = PROJECT_ROOT / "platform" / "site" / "data_build_tmp"
    final_dir = PROJECT_ROOT / "platform" / "site" / "data"
    events_dir = tmp_dir / "events"

    # Clean tmp dir if it exists
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)

    # Read config for RI threshold
    with open(config_file) as f:
        config = yaml.safe_load(f)
    ri_threshold = config["labels"]["ri_threshold_kt_24h"]

    # Read augmented event list
    df = pd.read_csv(augmented_csv)
    print(f"Loaded {len(df)} records from {augmented_csv}")

    # Sort by sid and timestamp to ensure proper ordering
    df = df.sort_values(["sid", "timestamp"]).reset_index(drop=True)

    # Compute dv6_kt: forward 6-hour wind delta
    # Convention: dv6_kt = wind(t+6h) - wind(t)
    # Within each sid, shift(-1) gets the next row (which is 6h forward)
    df["wind_kt"] = pd.to_numeric(df["wind_kt"], errors="coerce")
    df["wind_kt_shift_6"] = df.groupby("sid")["wind_kt"].shift(-1)
    df["dv6_kt"] = df["wind_kt_shift_6"] - df["wind_kt"]
    # Round to 1 decimal
    df["dv6_kt"] = df["dv6_kt"].apply(lambda x: round_floats(x, 1))

    # Convert pressure to numeric and round
    df["pressure_mb"] = pd.to_numeric(df["pressure_mb"], errors="coerce")
    df["pressure_mb_rounded"] = df["pressure_mb"].apply(lambda x: round_floats(x, 1))

    # Round other numeric columns
    df["wind_kt_rounded"] = df["wind_kt"].apply(lambda x: round_floats(x, 1))
    df["lat_rounded"] = df["lat"].apply(lambda x: round_floats(x, 4))
    df["lon_rounded"] = df["lon"].apply(lambda x: round_floats(x, 4))
    df["dv12_kt_rounded"] = df["dv12_kt"].apply(lambda x: round_floats(x, 1))
    df["dv24_kt_rounded"] = df["dv24_kt"].apply(lambda x: round_floats(x, 1))

    # Convert timestamp to datetime for ISO 8601 formatting
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])

    # Build events index
    events_index = []
    event_dict = {}  # sid -> event info

    for sid in df["sid"].unique():
        sid_data = df[df["sid"] == sid].copy()

        # Get summary stats
        start_ts = sid_data["timestamp_dt"].min().isoformat() + "Z"
        end_ts = sid_data["timestamp_dt"].max().isoformat() + "Z"
        max_wind = sid_data["wind_kt"].max()
        min_pressure = sid_data["pressure_mb"].min()
        has_ri = int((sid_data["ri_label"] == 1).any())
        n_points = len(sid_data)

        # Get storm name and basin from first row
        name = sid_data["storm_name"].iloc[0]
        basin = sid_data["basin"].iloc[0]

        event_info = {
            "sid": sid,
            "name": name,
            "basin": basin,
            "start": start_ts,
            "end": end_ts,
            "max_wind_kt": round_floats(max_wind, 1),
            "min_pressure_mb": round_floats(min_pressure, 1),
            "n_points": n_points,
            "has_ri": bool(has_ri)
        }
        events_index.append(event_info)
        event_dict[sid] = event_info

    # Sort by start descending
    events_index.sort(key=lambda x: x["start"], reverse=True)

    # Write events_index.json (compact, strict JSON: NaN -> null)
    index_path = tmp_dir / "events_index.json"
    with open(index_path, "w") as f:
        json.dump(sanitize(events_index), f, separators=(",", ":"), allow_nan=False)

    # Generate GeoJSON files for each event
    geojson_files = []
    for sid in df["sid"].unique():
        sid_data = df[df["sid"] == sid].copy()

        # Build track (LineString)
        coords = []
        points = []
        for _, row in sid_data.iterrows():
            lon = row["lon_rounded"]
            lat = row["lat_rounded"]
            coords.append([lon, lat])

            # Determine trend
            dv6 = row["dv6_kt"]
            if pd.isna(dv6):
                trend = "steady"
            elif dv6 > 0:
                trend = "strengthening"
            elif dv6 < 0:
                trend = "weakening"
            else:
                trend = "steady"

            # Build point feature
            point_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "t": row["timestamp_dt"].isoformat() + "Z",
                    "wind_kt": row["wind_kt_rounded"],
                    "pressure_mb": row["pressure_mb_rounded"],
                    "dv6_kt": row["dv6_kt"],
                    "dv12_kt": row["dv12_kt_rounded"],
                    "dv24_kt": row["dv24_kt_rounded"],
                    "ri_candidate": bool(row["ri_label"] == 1),
                    "trend": trend
                }
            }
            points.append(point_feature)

        # Build track feature (LineString)
        track_feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            },
            "properties": {
                "sid": sid,
                "name": sid_data["storm_name"].iloc[0]
            }
        }

        # Build FeatureCollection: track first, then all points
        geojson = {
            "type": "FeatureCollection",
            "features": [track_feature] + points
        }

        # Write GeoJSON (compact, strict JSON: NaN -> null)
        geojson_path = events_dir / f"{sid}.geojson"
        with open(geojson_path, "w") as f:
            json.dump(sanitize(geojson), f, separators=(",", ":"), allow_nan=False)
        geojson_files.append(geojson_path)

    # Build definitions.json
    definitions = {
        "ri": {
            "criterion": f"dv24_kt >= {ri_threshold}",
            "threshold_kt_per_24h": ri_threshold,
            "reference": "Kaplan & DeMaria (2003), Wea. Forecasting"
        },
        "dv_convention": "forward deltas: dv_h = wind(t+h) - wind(t), knots",
        "temporal_resolution": "6-hourly best-track fixes; no interpolation applied",
        "source": {
            "dataset": "IBTrACS v04r00 (ALL)",
            "url": "https://www.ncei.noaa.gov/products/international-best-track-archive",
            "doi": "10.25921/82ty-9e16",
            "accessed": get_ibtracs_mtime()
        },
        "note": "All values are historical observations from the best-track record, not predictions."
    }
    definitions_path = tmp_dir / "definitions.json"
    with open(definitions_path, "w") as f:
        json.dump(sanitize(definitions), f, indent=2, allow_nan=False)

    # Build manifest.json
    git_hash = get_git_hash()
    build_version = f"{git_hash}+{datetime.utcnow().isoformat()}Z"

    # Compute checksums
    artifacts = {}

    # Checksum events_index.json
    sha = compute_sha256(index_path)
    size = index_path.stat().st_size
    artifacts["events_index.json"] = {
        "sha256": sha,
        "bytes": size
    }

    # Checksum definitions.json
    sha = compute_sha256(definitions_path)
    size = definitions_path.stat().st_size
    artifacts["definitions.json"] = {
        "sha256": sha,
        "bytes": size
    }

    # Checksum each GeoJSON file
    for geojson_path in geojson_files:
        rel_path = f"events/{geojson_path.name}"
        sha = compute_sha256(geojson_path)
        size = geojson_path.stat().st_size
        artifacts[rel_path] = {
            "sha256": sha,
            "bytes": size
        }

    manifest = {
        "schema_version": 1,
        "build_version": build_version,
        "generated_by": "platform/build/build_events.py",
        "source_provenance": definitions["source"],
        "events": sorted(list(event_dict.keys())),
        "artifacts": artifacts
    }
    manifest_path = tmp_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(sanitize(manifest), f, indent=2, allow_nan=False)

    # Write README in data dir
    readme_path = tmp_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write("Generated by platform/build/build_events.py — never edit by hand; re-run the build.\n")

    # Atomic swap
    old_dir = final_dir.parent / "data_old"
    try:
        if final_dir.exists():
            if old_dir.exists():
                shutil.rmtree(old_dir)
            final_dir.rename(old_dir)
        tmp_dir.rename(final_dir)
        if old_dir.exists():
            shutil.rmtree(old_dir)
        print(f"\nSuccessfully swapped {tmp_dir} -> {final_dir}")
    except Exception as e:
        print(f"\nERROR during atomic swap: {e}")
        print(f"Build artifacts left in {tmp_dir} for inspection")
        raise

    # Summary
    total_bytes = sum(a["bytes"] for a in artifacts.values())
    n_events = len(event_dict)
    n_geojson = len([a for a in artifacts.keys() if a.startswith("events/")])

    print("\n" + "=" * 70)
    print("BUILD SUMMARY")
    print("=" * 70)
    print(f"Events (storms):        {n_events}")
    print(f"GeoJSON files:          {n_geojson}")
    print(f"Total data size:        {total_bytes:,} bytes ({total_bytes / 1e6:.2f} MB)")
    print(f"Build version:          {build_version}")
    print(f"Output directory:       {final_dir}")
    print("=" * 70)

    return final_dir, manifest, artifacts


if __name__ == "__main__":
    try:
        build_events()
        print("\nBuild completed successfully!")
    except Exception as e:
        print(f"\nBuild failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
