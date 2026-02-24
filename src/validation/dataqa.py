# src/validation/dataqa.py
"""
Data Quality Assurance (QA) module.
Performs various checks on the processed dataset to ensure scientific integrity.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.config import cfg_get

logger = logging.getLogger(__name__)


def run_dataqa(cfg: Dict[str, Any], split: str = "test") -> Dict[str, Any]:
    """
    Run data quality checks on a given split (train/val/test).
    Produces a report with missing files, value ranges, NaNs, and temporal consistency.
    """
    interim_dir = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    splits_csv = Path(cfg_get(cfg, "paths.splits_csv", "./data/normalized/splits.csv")).resolve()

    df = pd.read_csv(splits_csv)
    events = df[df["split"] == split]["event_id"].tolist()

    report = {
        "split": split,
        "n_events_total": len(events),
        "checks": {},
        "outliers": [],
        "nan_fraction": [],
    }

    # 1. Check file existence
    missing = []
    for eid in events:
        npy_path = interim_dir / f"{eid}.npy"
        json_path = interim_dir / f"{eid}.json"
        if not npy_path.exists() or not json_path.exists():
            missing.append(eid)
    report["checks"]["missing_files"] = missing[:50]  # first 50
    report["checks"]["n_missing"] = len(missing)

    # 2. Value ranges (sample up to 200 events for performance)
    qc_params = cfg_get(cfg, "data.qc", {})
    sst_range = tuple(qc_params.get("sst_range_K", [240.0, 330.0]))
    msl_range = tuple(qc_params.get("msl_range_Pa", [80000.0, 110000.0]))
    wind_max = float(qc_params.get("wind_abs_max_mps", 80.0))

    out_of_range = []
    sample_size = min(200, len(events))
    sampled_events = events[:sample_size]

    for eid in sampled_events:
        npy_path = interim_dir / f"{eid}.npy"
        json_path = interim_dir / f"{eid}.json"
        if not npy_path.exists() or not json_path.exists():
            continue

        cube = np.load(npy_path).astype(np.float32)  # (H,W,T,C)
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        channels = meta.get("channels", [])

        # Helper to get channel index
        def idx(name):
            try:
                return channels.index(name)
            except ValueError:
                return None

        # SST check
        i_sst = idx("sst_K")
        if i_sst is not None:
            sst = cube[..., i_sst]
            if np.any((sst < sst_range[0]) | (sst > sst_range[1])):
                out_of_range.append((eid, "sst"))

        # MSLP check
        i_msl = idx("mslp_Pa")
        if i_msl is not None:
            msl = cube[..., i_msl]
            if np.any((msl < msl_range[0]) | (msl > msl_range[1])):
                out_of_range.append((eid, "mslp"))

        # Wind speed check (if available)
        i_wind = idx("wind_mps")
        if i_wind is not None:
            wind = cube[..., i_wind]
            if np.any(np.abs(wind) > wind_max):
                out_of_range.append((eid, "wind"))

        # NaN fraction overall
        nan_frac = float(np.mean(~np.isfinite(cube)))
        report["nan_fraction"].append({"event_id": eid, "nan_fraction": nan_frac})

    report["checks"]["out_of_range_events"] = out_of_range[:50]

    # 3. Temporal consistency (check that timestamps are strictly increasing)
    temporal_issues = []
    for eid in sampled_events:
        json_path = interim_dir / f"{eid}.json"
        if not json_path.exists():
            continue
        with open(json_path, "r") as f:
            meta = json.load(f)
        timestamps = meta.get("timestamps", [])
        if len(timestamps) > 1:
            # Convert to datetime objects
            from datetime import datetime
            try:
                dt_list = [datetime.fromisoformat(ts) for ts in timestamps]
                diffs = [(dt_list[i+1] - dt_list[i]).total_seconds() for i in range(len(dt_list)-1)]
                if any(d <= 0 for d in diffs):
                    temporal_issues.append(eid)
            except Exception:
                temporal_issues.append(eid)
    report["checks"]["temporal_issues"] = temporal_issues[:20]

    # 4. Shape consistency
    shape_issues = []
    for eid in sampled_events:
        npy_path = interim_dir / f"{eid}.npy"
        json_path = interim_dir / f"{eid}.json"
        if not npy_path.exists() or not json_path.exists():
            continue
        cube = np.load(npy_path)
        with open(json_path, "r") as f:
            meta = json.load(f)
        expected_shape = tuple(meta.get("cube_shape", []))
        if cube.shape != expected_shape:
            shape_issues.append((eid, cube.shape, expected_shape))
    report["checks"]["shape_issues"] = shape_issues[:20]

    # Summary
    report["summary"] = {
        "n_checked": sample_size,
        "n_out_of_range": len(out_of_range),
        "n_temporal_issues": len(temporal_issues),
        "n_shape_issues": len(shape_issues),
    }

    return report