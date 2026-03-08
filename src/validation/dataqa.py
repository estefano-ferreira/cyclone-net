from __future__ import annotations

"""CycloneNet — data quality assurance for processed event artifacts."""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.config import cfg_get

logger = logging.getLogger(__name__)


def _nan_fraction(arr: np.ndarray) -> float:
    return float(np.mean(~np.isfinite(arr)))


def _channel_index(channels: List[str], name: str) -> int | None:
    try:
        return channels.index(name)
    except ValueError:
        return None


def run_dataqa(cfg: Dict[str, Any], split: str = "test") -> Dict[str, Any]:
    interim_dir = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    splits_csv = Path(cfg_get(cfg, "paths.splits_csv", "./data/normalized/splits.csv")).resolve()
    seed = int(cfg_get(cfg, "training.seed", 42))

    df = pd.read_csv(splits_csv)
    events = df[df["split"] == split]["event_id"].tolist()

    qc_params = cfg_get(cfg, "data.qc", {})
    sst_range = tuple(qc_params.get("sst_range_K", [240.0, 330.0]))
    msl_range = tuple(qc_params.get("msl_range_Pa", [80000.0, 110000.0]))
    wind_max = float(qc_params.get("wind_abs_max_mps", 80.0))
    max_cube_nan = float(qc_params.get("max_nan_fraction_cube", 0.20))
    max_fuel_nan = float(qc_params.get("max_nan_fraction_fuel_prior", 0.20))

    report: Dict[str, Any] = {
        "split": split,
        "n_events_total": len(events),
        "checks": {},
        "missing_files": [],
        "out_of_range": [],
        "nan_issues": [],
        "temporal_issues": [],
        "channel_summary": {},
    }

    required_suffixes = [".npy", ".json", "_lats.npy", "_lons.npy"]
    missing = []
    valid_events = []
    for eid in events:
        missing_here = [suffix for suffix in required_suffixes if not (interim_dir / f"{eid}{suffix}").exists()]
        if missing_here:
            missing.append({"event_id": eid, "missing": missing_here})
        else:
            valid_events.append(eid)

    report["missing_files"] = missing[:100]
    report["checks"]["n_missing_events"] = len(missing)

    rng = random.Random(seed)
    sample_size = min(int(cfg_get(cfg, "validation.dataqa_sample_size", 200)), len(valid_events))
    sampled_events = rng.sample(valid_events, sample_size) if sample_size < len(valid_events) else valid_events

    per_channel_nan: Dict[str, List[float]] = {}

    for eid in sampled_events:
        cube = np.load(interim_dir / f"{eid}.npy").astype(np.float32)
        with (interim_dir / f"{eid}.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        channels = list(meta.get("channels", []))

        if cube.ndim != 4:
            report["out_of_range"].append({"event_id": eid, "issue": f"Expected 4D cube, got {cube.shape}"})
            continue

        if len(channels) != cube.shape[-1]:
            report["out_of_range"].append({
                "event_id": eid,
                "issue": f"Channel metadata mismatch: metadata={len(channels)}, cube={cube.shape[-1]}",
            })

        for c_idx, c_name in enumerate(channels):
            frac = _nan_fraction(cube[..., c_idx])
            per_channel_nan.setdefault(c_name, []).append(frac)

        i_sst = _channel_index(channels, "sst_K")
        if i_sst is not None:
            sst = cube[..., i_sst]
            if np.any(np.isfinite(sst) & ((sst < sst_range[0]) | (sst > sst_range[1]))):
                report["out_of_range"].append({"event_id": eid, "channel": "sst_K"})

        i_msl = _channel_index(channels, "mslp_Pa")
        if i_msl is not None:
            msl = cube[..., i_msl]
            if np.any(np.isfinite(msl) & ((msl < msl_range[0]) | (msl > msl_range[1]))):
                report["out_of_range"].append({"event_id": eid, "channel": "mslp_Pa"})

        i_wind = _channel_index(channels, "wind_mps")
        if i_wind is not None:
            wind = cube[..., i_wind]
            if np.any(np.isfinite(wind) & (wind > wind_max)):
                report["out_of_range"].append({"event_id": eid, "channel": "wind_mps"})

        cube_nan = _nan_fraction(cube)
        if cube_nan > max_cube_nan:
            report["nan_issues"].append({"event_id": eid, "cube_nan_fraction": cube_nan})

        fuel_path = interim_dir / f"{eid}_fuel_potential.npy"
        if fuel_path.exists():
            fuel = np.load(fuel_path).astype(np.float32)
            fuel_nan = _nan_fraction(fuel)
            if fuel_nan > max_fuel_nan:
                report["nan_issues"].append({"event_id": eid, "fuel_nan_fraction": fuel_nan})

        timestamps = meta.get("timestamps", [])
        if len(timestamps) != int(cfg_get(cfg, "data.sequence_length", 5)):
            report["temporal_issues"].append({"event_id": eid, "issue": "Unexpected timestamp count"})
        else:
            try:
                dt = pd.to_datetime(timestamps)
                deltas = (dt[1:] - dt[:-1]).total_seconds() / 3600.0
                if not np.allclose(np.abs(deltas), 6.0):
                    report["temporal_issues"].append({"event_id": eid, "issue": "Timestamps are not 6-hourly"})
            except Exception:
                report["temporal_issues"].append({"event_id": eid, "issue": "Failed to parse timestamps"})

    channel_summary = {}
    for name, values in per_channel_nan.items():
        arr = np.asarray(values, dtype=np.float64)
        channel_summary[name] = {
            "nan_fraction_mean": float(np.mean(arr)),
            "nan_fraction_max": float(np.max(arr)),
        }

    report["channel_summary"] = channel_summary
    report["checks"]["sampled_events"] = len(sampled_events)
    report["checks"]["passed"] = (
        len(report["missing_files"]) == 0
        and len(report["out_of_range"]) == 0
        and len(report["nan_issues"]) == 0
        and len(report["temporal_issues"]) == 0
    )
    return report