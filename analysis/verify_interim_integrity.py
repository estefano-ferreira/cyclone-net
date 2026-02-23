"""Verify interim artifacts for physics-guided training readiness.

Checks (no NetCDF access required):
1) Temporal integrity: unique era5_selected_times == T
2) Center inside patch: center_lat/lon must fall within lats/lons patch bounds (lon wrap handled)

Writes a JSON report with counts and a few examples.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def lon_to_180(x: np.ndarray) -> np.ndarray:
    return ((x + 180.0) % 360.0) - 180.0

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--max_files", type=int, default=5000)
    ap.add_argument("--out", default="outputs/results/interim_integrity_report.json")
    args = ap.parse_args()

    interim = Path(args.interim)
    files = sorted(interim.glob("era5_*.json"))[: args.max_files]

    total = 0
    temporal_bad = 0
    center_bad = 0
    examples = {"temporal_bad": [], "center_bad": []}

    for jf in files:
        total += 1
        j = json.loads(jf.read_text(encoding="utf-8"))
        eid = j.get("event_id", jf.stem)

        sel = j.get("era5_selected_times", [])
        T = len(j.get("timestamps", sel))
        if len(set(sel)) != T:
            temporal_bad += 1
            if len(examples["temporal_bad"]) < 5:
                examples["temporal_bad"].append({"event_id": eid, "selected": sel, "timestamps": j.get("timestamps", [])})

        lats_path = jf.with_name(f"{eid}_lats.npy")
        lons_path = jf.with_name(f"{eid}_lons.npy")
        if lats_path.exists() and lons_path.exists():
            lats = np.load(lats_path)
            lons = np.load(lons_path)
            clat = float(j["center_lat"])
            clon = float(j["center_lon"])
            ok_lat = float(np.min(lats)) <= clat <= float(np.max(lats))
            lons180 = lon_to_180(lons.astype(float))
            ok_lon = float(np.min(lons180)) <= clon <= float(np.max(lons180))
            if not (ok_lat and ok_lon):
                center_bad += 1
                if len(examples["center_bad"]) < 5:
                    examples["center_bad"].append({
                        "event_id": eid,
                        "center": [clat, clon],
                        "lat_minmax": [float(np.min(lats)), float(np.max(lats))],
                        "lon_minmax_180": [float(np.min(lons180)), float(np.max(lons180))]
                    })

    report = {
        "checked": total,
        "temporal_bad": temporal_bad,
        "center_bad": center_bad,
        "temporal_bad_pct": (temporal_bad / max(1, total)) * 100.0,
        "center_bad_pct": (center_bad / max(1, total)) * 100.0,
        "examples": examples
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
