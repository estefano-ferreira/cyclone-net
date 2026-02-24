# src/validation/check_splits.py
"""
Check that train/val/test splits are properly separated by storm ID (SID)
and that there is no leakage between splits.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Set

import pandas as pd


def check_splits(splits_csv: Path, events_dir: Path) -> Dict[str, Any]:
    """
    Verify that no storm ID (SID) appears in more than one split.
    Returns a dictionary with counts, overlap information, and a passed flag.
    """
    df = pd.read_csv(splits_csv)
    if "event_id" not in df.columns or "split" not in df.columns:
        raise ValueError("splits_csv must contain columns: event_id, split")

    # Map event_id to SID by reading metadata JSONs
    sid_by_event = {}
    missing_meta = []
    for eid in df["event_id"]:
        meta_path = events_dir / f"{eid}.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            sid = meta.get("sid", None)
            if sid is not None:
                sid_by_event[eid] = sid
            else:
                sid_by_event[eid] = f"unknown_{eid}"
        else:
            missing_meta.append(eid)
            sid_by_event[eid] = f"missing_{eid}"

    # Group SIDs by split
    split_sids: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    for split in ["train", "val", "test"]:
        events_in_split = df[df["split"] == split]["event_id"].tolist()
        sids = {sid_by_event[e] for e in events_in_split if e in sid_by_event}
        split_sids[split] = sids

    # Find overlaps
    overlaps = {}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        inter = split_sids[a].intersection(split_sids[b])
        if inter:
            overlaps[f"{a}_{b}"] = list(inter)

    result = {
        "n_events": len(df),
        "split_counts": df["split"].value_counts().to_dict(),
        "missing_metadata": missing_meta[:20],  # limit
        "overlaps": overlaps,
        "passed": len(overlaps) == 0,
    }
    return result