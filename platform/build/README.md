# Platform Build

This directory contains the build script that transforms pipeline outputs into static web platform data.

## Running the Build

From the project root:

```bash
venv\Scripts\python.exe platform\build\build_events.py
```

The script reads:
- `data/event_list_augmented.csv` — pipeline's IBTrACS-derived event list with RI labels (6-hourly points)
- `config.yaml` — only reads `labels.ri_threshold_kt_24h` for definitions

## Outputs

Generated in `platform/site/data/` (created atomically):

- **events_index.json** — Summary of all storms: sid, name, basin, start/end times, max wind, min pressure, point count, RI occurrence
- **events/<sid>.geojson** — Per-storm GeoJSON with full track (LineString) and timestep points (Point features with wind deltas and RI candidate flag)
- **definitions.json** — Machine-readable citations: RI criterion, delta convention, temporal resolution, IBTrACS provenance
- **manifest.json** — Build metadata (git hash, timestamp) and SHA256 checksums for all artifacts
- **README.md** — Generation notice

All JSON files use compact encoding (no indent) except manifest.json and definitions.json (indent=2 for human readability and auditability).

## Data Conventions

- **dv6_kt, dv12_kt, dv24_kt**: Forward deltas (wind at t+h minus wind at t). Positive = intensifying, negative = weakening.
- **Rounding**: 1 decimal for wind/pressure (knots, mb), 4 decimals for coordinates (lat/lon).
- **Timestamps**: ISO 8601 UTC (e.g., "2020-08-28T12:00:00Z").
- **Null values**: `null` for missing pressure measurements.
- **RI candidate**: true if ri_label == 1 (≥30 kt intensification over preceding 24h).
- **Trend**: "strengthening" (dv6_kt > 0), "weakening" (dv6_kt < 0), "steady" (0 or null).

## Verification

After running, the build script computes SHA256 hashes for all artifacts and stores them in manifest.json. To verify integrity:

```python
import json
import hashlib
from pathlib import Path

data_dir = Path("platform/site/data")
with open(data_dir / "manifest.json") as f:
    manifest = json.load(f)

for artifact_path, expected_hash in manifest["artifacts"].items():
    file_path = data_dir / artifact_path
    actual_sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
    assert actual_sha == expected_hash["sha256"], f"Hash mismatch: {artifact_path}"
    print(f"✓ {artifact_path}")
```
