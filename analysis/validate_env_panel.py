#!/usr/bin/env python
"""Validate env panel values embedded in built event GeoJSON.

PREPARED, NOT RUN as part of this change. Run this manually AFTER the
data/interim backfill (shear_850_200_mps / rh_mid channels) completes and a
`python platform/build/build_events.py --with-env` build has been produced.
Read-only: touches nothing under data/ or platform/site/data/.

For N sampled storms (default 5, or an explicit --sids list), for every
track point in the built geojson that carries env_* properties, this
reopens the SOURCE interim cube (mmap, read-only) and recomputes the t0
patch mean per channel independently of platform/build/build_events.py's
own computation, then compares against the stored geojson value with a
tolerance of 0.01 (accounts for the build's own 2-decimal rounding).

Also re-hashes every geojson that carries env values against
manifest.json's recorded sha256, to catch any manifest/artifact drift.

Exit code: 0 if every sampled value matches and every hash checks out;
1 on ANY mismatch, missing cube, or hash drift.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
TOLERANCE = 0.01

# cube channel name -> (geojson property name, is SST-in-Kelvin needing -273.15)
CHANNEL_SPECS = {
    "sst_K": ("env_sst_c", True),
    "shear_850_200_mps": ("env_shear_mps", False),
    "rh_mid": ("env_rh_pct", False),
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def recompute_patch_mean(npy_path: Path, channel_idx: int, is_sst_kelvin: bool) -> Optional[float]:
    """Independent re-derivation of one point's env value straight from the
    cube: mmap open -> t0 slice -> mean over the full window -> unit
    conversion -> round. Mirrors the DEFINITION in build_events.py, not its
    code, so a bug shared by both wouldn't be masked by importing one from
    the other.
    """
    cube = None
    try:
        cube = np.load(npy_path, mmap_mode="r")
        if cube.ndim != 4 or cube.shape[2] < 1:
            return None
        patch = cube[:, :, 0, channel_idx]
        mean_val = float(np.mean(patch))
        if not math.isfinite(mean_val):
            return None
        if is_sst_kelvin:
            mean_val -= 273.15
        return round(mean_val, 2)
    finally:
        if cube is not None:
            mmap_obj = getattr(cube, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
            del cube


def event_id_for_point(sid: str, iso_ts: str) -> str:
    """Rebuild the candidate interim event_id from a built geojson point's
    't' property (ISO 8601, e.g. '1980-06-09T00:00:00Z')."""
    ts = iso_ts[:-1] if iso_ts.endswith("Z") else iso_ts
    dt = datetime.fromisoformat(ts)
    return f"era5_{dt.strftime('%Y_%m_%d_%H%M')}_{sid}"


def validate_storm(sid: str, interim_dir: Path, geojson_path: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "sid": sid,
        "points_checked": 0,
        "matches": {spec[0]: 0 for spec in CHANNEL_SPECS.values()},
        "mismatches": [],
    }

    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    point_features = [f for f in geojson["features"] if f["geometry"]["type"] == "Point"]
    env_props = [spec[0] for spec in CHANNEL_SPECS.values()]

    for feat in point_features:
        props = feat["properties"]
        if not any(k in props for k in env_props):
            continue  # this build has no env values at all -- nothing to check
        report["points_checked"] += 1

        event_id = event_id_for_point(sid, props["t"])
        json_path = interim_dir / f"{event_id}.json"
        npy_path = interim_dir / f"{event_id}.npy"

        if not json_path.exists() or not npy_path.exists():
            if any(props.get(k) is not None for k in env_props):
                report["mismatches"].append({
                    "event_id": event_id,
                    "reason": "cube/metadata missing on disk but geojson has a non-null env value",
                })
            continue

        meta = json.loads(json_path.read_text(encoding="utf-8"))
        channels = meta.get("channels", [])

        for channel_name, (prop_name, is_sst_kelvin) in CHANNEL_SPECS.items():
            stored = props.get(prop_name)

            if channel_name not in channels:
                if stored is not None:
                    report["mismatches"].append({
                        "event_id": event_id, "field": prop_name,
                        "reason": "channel absent from cube but geojson has a non-null value",
                        "stored": stored,
                    })
                continue

            idx = channels.index(channel_name)
            recomputed = recompute_patch_mean(npy_path, idx, is_sst_kelvin)

            if recomputed is None and stored is None:
                report["matches"][prop_name] += 1
            elif recomputed is None or stored is None or abs(recomputed - stored) > TOLERANCE:
                report["mismatches"].append({
                    "event_id": event_id, "field": prop_name,
                    "stored": stored, "recomputed": recomputed,
                })
            else:
                report["matches"][prop_name] += 1

    return report


def verify_manifest_hashes(site_data_dir: Path) -> List[str]:
    manifest_path = site_data_dir / "manifest.json"
    if not manifest_path.exists():
        return [f"manifest.json not found at {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    problems = []
    for rel_path, info in manifest.get("artifacts", {}).items():
        if not rel_path.startswith("events/"):
            continue
        full_path = site_data_dir / rel_path
        if not full_path.exists():
            problems.append(f"{rel_path}: listed in manifest but missing on disk")
            continue
        text = full_path.read_text(encoding="utf-8")
        if '"env_' not in text:
            continue  # only re-hash geojsons that actually carry env values
        actual = compute_sha256(full_path)
        if actual != info["sha256"]:
            problems.append(f"{rel_path}: manifest sha256 {info['sha256']} != actual {actual}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5, help="Number of storms to sample (ignored if --sids is given).")
    ap.add_argument("--sids", nargs="*", default=None, help="Explicit SIDs to validate instead of a random sample.")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    ap.add_argument("--site-data", default=str(PROJECT_ROOT / "platform" / "site" / "data"))
    ap.add_argument("--seed", type=int, default=1337, help="Sampling seed (for reproducible spot checks).")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    interim_dir = (PROJECT_ROOT / config["paths"]["interim_data"]).resolve()

    site_data_dir = Path(args.site_data)
    events_dir = site_data_dir / "events"
    if not events_dir.exists():
        print(f"No built events found at {events_dir} -- run the --with-env build first.", file=sys.stderr)
        return 1

    if args.sids:
        sids = args.sids
    else:
        all_sids = sorted(p.stem for p in events_dir.glob("*.geojson"))
        rng = random.Random(args.seed)
        sids = rng.sample(all_sids, min(args.n, len(all_sids)))

    overall_problems = 0
    per_variable_matches = {spec[0]: 0 for spec in CHANNEL_SPECS.values()}

    for sid in sids:
        geojson_path = events_dir / f"{sid}.geojson"
        if not geojson_path.exists():
            print(f"[{sid}] SKIP -- no geojson at {geojson_path}")
            overall_problems += 1
            continue

        report = validate_storm(sid, interim_dir, geojson_path)
        for k, v in report["matches"].items():
            per_variable_matches[k] += v
        n_mismatch = len(report["mismatches"])
        overall_problems += n_mismatch

        status = "OK" if n_mismatch == 0 else "MISMATCH"
        print(f"[{sid}] {status} -- points_checked={report['points_checked']} "
              f"matches={report['matches']} mismatches={n_mismatch}")
        for m in report["mismatches"]:
            print(f"    {m}")

    print(f"\nPer-variable match counts across sampled storms: {per_variable_matches}")

    hash_problems = verify_manifest_hashes(site_data_dir)
    if hash_problems:
        print("\nManifest hash problems:")
        for p in hash_problems:
            print(f"  {p}")
        overall_problems += len(hash_problems)

    if overall_problems:
        print(f"\nFAILED: {overall_problems} mismatch(es)/problem(s).")
        return 1

    print("\nOK: all sampled env values match independently recomputed patch means; manifest hashes clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
