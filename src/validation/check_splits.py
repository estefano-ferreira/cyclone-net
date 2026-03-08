# src/validation/check_splits.py
"""
Check that train/val/test splits are properly separated by storm ID (SID)
and that there is no leakage between splits.
"""


import json
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd


VALID_SPLITS = {"train", "val", "test"}


def check_splits(splits_csv: Path, events_dir: Path) -> Dict[str, Any]:
    df = pd.read_csv(splits_csv)
    if "event_id" not in df.columns or "split" not in df.columns:
        raise ValueError("splits_csv must contain columns: event_id, split")

    unknown_splits = sorted(set(df["split"].unique()) - VALID_SPLITS)
    if unknown_splits:
        raise ValueError(f"Unexpected split names: {unknown_splits}")

    sid_by_event: Dict[str, str] = {}
    label_by_event: Dict[str, int | None] = {}
    missing_meta: List[str] = []
    missing_sid: List[str] = []

    for eid in df["event_id"].tolist():
        meta_path = Path(events_dir) / f"{eid}.json"
        if not meta_path.exists():
            missing_meta.append(eid)
            continue
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        sid = str(meta.get("sid", "")).strip()
        if not sid:
            missing_sid.append(eid)
            continue
        sid_by_event[eid] = sid
        label = meta.get("ri_label", None)
        label_by_event[eid] = int(label) if label is not None else None

    split_sids: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}
    split_stats: Dict[str, Dict[str, Any]] = {}

    for split in ["train", "val", "test"]:
        events_in_split = df[df["split"] == split]["event_id"].tolist()
        sids = {sid_by_event[e] for e in events_in_split if e in sid_by_event}
        split_sids[split] = sids

        labels = [label_by_event[e] for e in events_in_split if e in label_by_event and label_by_event[e] is not None]
        positives = int(sum(labels)) if labels else 0
        split_stats[split] = {
            "n_events": len(events_in_split),
            "n_unique_sids": len(sids),
            "n_labels": len(labels),
            "n_positive": positives,
            "positive_rate": float(positives / len(labels)) if labels else None,
        }

    overlaps = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        inter = sorted(split_sids[a].intersection(split_sids[b]))
        if inter:
            overlaps[f"{a}_{b}"] = inter

    passed = len(overlaps) == 0 and len(missing_meta) == 0 and len(missing_sid) == 0
    return {
        "n_events": int(len(df)),
        "split_counts": df["split"].value_counts().to_dict(),
        "split_stats": split_stats,
        "missing_metadata": missing_meta[:50],
        "missing_sid": missing_sid[:50],
        "overlaps": overlaps,
        "passed": passed,
    }