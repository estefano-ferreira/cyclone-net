# src/data/normalization.py
from __future__ import annotations

"""
CycloneNet — train-only normalization for explicit input channel names (leakage-safe).

This module computes mean/std ONLY for model.input_channels_names using TRAIN split only.
It is designed for scientific correctness when cubes contain extra channels
(diagnostics, heat flux, priors) that must NOT be normalized as model inputs.

Key guarantees:
- Uses ONLY events in split=train from paths.splits_csv.
- Selects channels strictly by name using each event's meta["channels"].
- Optional anti-leakage guard: removes total_heat_flux from inputs when configured as loss-only.
- Uses a fast, deterministic accumulation (sum, sumsq, count) over finite values.

Outputs:
- paths.normalization_stats (JSON):
    channels: ordered input channel names actually normalized
    mean: per-channel mean
    std: per-channel std
    count: per-channel count of finite samples
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.config import load_config, cfg_get

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # type: ignore

logger = logging.getLogger(__name__)


def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve_input_names(cfg: Dict[str, Any]) -> List[str]:
    """
    Resolve model input channel names and enforce anti-leakage if configured.
    """
    input_names = cfg_get(cfg, "model.input_channels_names", None)
    if not input_names:
        raise ValueError("config.yaml must define model.input_channels_names.")
    input_names = list(input_names)

    exclude_thf = bool(
        cfg_get(cfg, "physics_guided.losses.exclude_total_heat_flux_from_input", True))
    thf_name = str(cfg_get(
        cfg, "physics_guided.losses.total_heat_flux_channel_name", "total_heat_flux_Wpm2"))

    # If total heat flux is loss-only, it must not be normalized as an input channel.
    if exclude_thf and thf_name in input_names:
        input_names = [c for c in input_names if c != thf_name]
        logger.info(
            f"Removed '{thf_name}' from normalization inputs (physics-loss-only).")

    if len(input_names) == 0:
        raise ValueError(
            "Resolved model.input_channels_names is empty after leakage guards.")
    return input_names


def compute_norm_stats_from_splits(
    interim_dir: Path,
    splits_csv: Path,
    out_path: Path,
    input_names: List[str],
) -> Dict[str, Any]:
    """
    Train-only normalization using a splits CSV with columns:
      - event_id
      - split (train/val/test)

    Computes mean/std per channel using finite values only.
    """
    interim_dir = Path(interim_dir).resolve()
    splits_csv = Path(splits_csv).resolve()
    out_path = Path(out_path).resolve()

    df = pd.read_csv(splits_csv)
    if "event_id" not in df.columns or "split" not in df.columns:
        raise ValueError("splits_csv must contain columns: event_id, split")

    train_ids = df[df["split"] == "train"]["event_id"].astype(str).tolist()
    if not train_ids:
        raise RuntimeError(
            f"No train events found in splits CSV: {splits_csv}")

    C = len(input_names)
    sum_c = np.zeros(C, dtype=np.float64)
    sumsq_c = np.zeros(C, dtype=np.float64)
    count_c = np.zeros(C, dtype=np.int64)

    iterable = train_ids
    if tqdm is not None:
        iterable = tqdm(train_ids, desc="Normalize (train-only)", unit="event")

    missing_files = 0
    missing_channels = 0
    used_events = 0

    for eid in iterable:
        meta_path = interim_dir / f"{eid}.json"
        npy_path = interim_dir / f"{eid}.npy"
        if not meta_path.exists() or not npy_path.exists():
            missing_files += 1
            continue

        meta = _load_json(meta_path)
        all_channels = list(meta.get("channels", []))
        if not all_channels:
            missing_channels += 1
            continue

        if not all(c in all_channels for c in input_names):
            missing_channels += 1
            continue

        idx = [all_channels.index(c) for c in input_names]

        cube = np.load(npy_path).astype(np.float64)  # (H,W,T,C_total)
        x = cube[:, :, :, idx].reshape(-1, C)        # (N,C)

        finite = np.isfinite(x)
        # Accumulate per channel
        for c in range(C):
            vals = x[:, c]
            m = finite[:, c]
            if not np.any(m):
                continue
            vv = vals[m]
            sum_c[c] += float(np.sum(vv))
            sumsq_c[c] += float(np.sum(vv * vv))
            count_c[c] += int(vv.size)

        used_events += 1

    if used_events == 0:
        raise RuntimeError(
            "No events contributed to normalization. "
            "Check that interim_dir has cubes and that model.input_channels_names exist in meta['channels']."
        )

    # mean/std with numerical safety
    denom = np.maximum(1, count_c).astype(np.float64)
    mean = (sum_c / denom).astype(np.float32)
    var = (sumsq_c / denom) - (mean.astype(np.float64) ** 2)
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var).astype(np.float32)

    out: Dict[str, Any] = {
        "channels": list(input_names),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": count_c.tolist(),
        "notes": (
            "Train-only stats for model.input_channels_names (after anti-leakage guards). "
            "Other cube channels are excluded by design."
        ),
        "debug": {
            "used_events": used_events,
            "missing_files": missing_files,
            "missing_channels": missing_channels,
            "splits_csv": str(splits_csv),
            "interim_dir": str(interim_dir),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def compute_norm_stats() -> Dict[str, Any]:
    """
    Entrypoint expected by run.py.

    Loads config.yaml, resolves leakage-safe input channels, computes train-only stats,
    writes to paths.normalization_stats, and returns the stats dict.
    """
    cfg = load_config("config.yaml")

    interim_dir = Path(cfg_get(cfg, "paths.interim_data",
                       "./data/interim")).resolve()
    splits_csv = Path(cfg_get(cfg, "paths.splits_csv",
                      "./data/normalized/splits.csv")).resolve()
    out_path = Path(cfg_get(cfg, "paths.normalization_stats",
                    "./data/normalized/normalization_stats.json")).resolve()

    input_names = _resolve_input_names(cfg)

    stats = compute_norm_stats_from_splits(
        interim_dir=interim_dir,
        splits_csv=splits_csv,
        out_path=out_path,
        input_names=input_names,
    )
    logger.info(f"Saved normalization stats to: {out_path}")
    logger.info(f"Channels: {stats['channels']}")
    return stats


def main() -> None:
    compute_norm_stats()


if __name__ == "__main__":
    main()
