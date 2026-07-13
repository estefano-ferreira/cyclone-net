# analysis/pl_gate_census.py
"""
PL-channel coverage census + gate for the development set.

Purpose
-------
The feature-ablation and RI-precursor re-test experiments (analysis/
feature_ablation_kfold.py, analysis/feature_ablation_cnn.py,
analysis/ri_precursors.py) compare surface-only vs surface+pressure-level
(PL) predictors. That comparison is only fair if EVERY development-set event
(train+val) actually has the two PL channels (``shear_850_200_mps``,
``rh_mid``) after the 1980-2019 backfill completes. This script is the
single, cheap, auditable gate for that precondition.

Design
------
* Metadata only. This script NEVER loads a ``.npy`` cube — only the small
  per-event ``.json`` metadata sidecar (its ``channels`` list) is read. This
  keeps the census safe to run frequently, including while a backfill is
  in progress elsewhere.
* The frozen TEST split is never opened for content: this script reads
  ``splits.csv`` (which only records the event_id -> split ASSIGNMENT, not
  any channel/label content) to get per-split counts, but it never opens
  a ``data/interim/{event_id}.json`` file for an event whose split is
  "test". Only train+val (the "dev" set) metadata is inspected.
* Join mechanism mirrors ``load_dev_events`` in
  ``analysis/feature_ablation_kfold.py``: splits.csv merged with
  valid_events.csv on event_id.

Gate verdict
------------
PASS only if every dev-set (train+val) event has BOTH PL channels present
in its metadata ``channels`` list. A missing artifact (no metadata file at
all) also counts as a gate failure for that event — it cannot be used
either way.

Output
------
outputs/results/pl_gate_census.json (machine-readable): counts, per-year /
per-split breakdown for the dev set, gate_pass bool, timestamp, and (on
failure) up to 20 offending event_ids plus per-year missing counts.

Usage:
    python analysis/pl_gate_census.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.processors.pressure_channels import RH_CHANNEL, SHEAR_CHANNEL  # noqa: E402
from src.utils.config import cfg_get, load_config  # noqa: E402
from src.utils.paths import rel_to_root  # noqa: E402

PL_CHANNELS = [SHEAR_CHANNEL, RH_CHANNEL]
DEV_SPLITS = ("train", "val")

_EVENT_ID_YEAR_RE = re.compile(r"^era5_(\d{4})_\d{2}_\d{2}_\d{4}_")


def _year_from_event_id(event_id: str) -> Optional[int]:
    """Best-effort year extraction from the ``era5_YYYY_MM_DD_HHMM_SID`` convention.

    Returns None (bucketed as "unknown") rather than raising -- this is a
    reporting aid, not a scientific computation.
    """
    m = _EVENT_ID_YEAR_RE.match(str(event_id))
    return int(m.group(1)) if m else None


def load_split_join(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Every event_id with its split assignment, sid and ri_label.

    Same merge as ``load_dev_events`` in feature_ablation_kfold.py, except
    it keeps ALL splits (train/val/test) so per-split totals can be
    reported; the test split's per-event metadata is never subsequently
    opened by this module.
    """
    normalized = Path(cfg_get(cfg, "paths.normalized_dir", "./data/normalized")).resolve()
    splits = pd.read_csv(normalized / "splits.csv")
    events = pd.read_csv(normalized / "valid_events.csv")
    df = splits.merge(events[["event_id", "sid", "ri_label"]], on="event_id", how="inner")
    df["year"] = df["event_id"].map(_year_from_event_id)
    return df


def audit_dev_pl_coverage(cfg: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Per-event PL-channel presence for the DEV set only (train+val).

    Reads only ``{event_id}.json`` (metadata) -- never the ``.npy`` cube,
    never a test-split event.
    """
    interim = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    dev = df[df["split"].isin(DEV_SPLITS)].copy()

    statuses: List[str] = []
    has_pl: List[bool] = []
    for event_id in dev["event_id"]:
        meta_path = interim / f"{event_id}.json"
        if not meta_path.exists():
            statuses.append("missing_artifact")
            has_pl.append(False)
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive, corrupt metadata
            statuses.append(f"unreadable_metadata:{exc}")
            has_pl.append(False)
            continue
        channels = list(meta.get("channels", []))
        present = all(ch in channels for ch in PL_CHANNELS)
        statuses.append("ok" if present else "missing_pl_channels")
        has_pl.append(present)

    dev = dev.reset_index(drop=True)
    dev["status"] = statuses
    dev["has_pl"] = has_pl
    return dev


def build_census(cfg: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    df = load_split_join(cfg)
    per_split_counts = {str(k): int(v) for k, v in df["split"].value_counts().items()}

    dev = audit_dev_pl_coverage(cfg, df)
    n_dev = int(len(dev))
    n_with_pl = int(dev["has_pl"].sum())
    n_without_pl = n_dev - n_with_pl
    n_missing_artifact = int((dev["status"] == "missing_artifact").sum())

    # Per-year x per-split breakdown, dev set only.
    per_year_per_split: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"with_pl": 0, "without_pl": 0})
    )
    for row in dev.itertuples():
        year_key = str(row.year) if row.year is not None else "unknown"
        bucket = "with_pl" if row.has_pl else "without_pl"
        per_year_per_split[year_key][row.split][bucket] += 1
    # Convert defaultdicts to plain dicts for JSON serialization, sorted by year.
    per_year_per_split_out = {
        year: {split: dict(counts) for split, counts in sorted(splits.items())}
        for year, splits in sorted(per_year_per_split.items())
    }

    offending = dev[~dev["has_pl"]].copy()
    offending_sorted = offending.sort_values(["year", "event_id"])
    offending_event_ids_sample = offending_sorted["event_id"].head(20).tolist()

    offending_per_year: Dict[str, int] = defaultdict(int)
    for row in offending.itertuples():
        year_key = str(row.year) if row.year is not None else "unknown"
        offending_per_year[year_key] += 1

    gate_pass = n_without_pl == 0
    if gate_pass:
        gate_verdict = (
            f"PASS: all {n_dev} dev-set (train+val) events have both PL channels "
            f"({', '.join(PL_CHANNELS)})."
        )
    else:
        gate_verdict = (
            f"FAIL: {n_without_pl} of {n_dev} dev-set (train+val) events lack "
            f"at least one PL channel ({', '.join(PL_CHANNELS)}) or their artifact "
            "is missing. Feature-ablation and CNN-ablation tooling must refuse to "
            "run with --require-gate until this is zero."
        )

    census = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": rel_to_root(config_path),
        "interim_dir": rel_to_root(Path(cfg_get(cfg, "paths.interim_data", "./data/interim"))),
        "pl_channels_checked": PL_CHANNELS,
        "total_events": int(len(df)),
        "counts_per_split": per_split_counts,
        "dev_splits": list(DEV_SPLITS),
        "dev_total_events": n_dev,
        "dev_events_with_pl": n_with_pl,
        "dev_events_without_pl": n_without_pl,
        "dev_missing_artifact": n_missing_artifact,
        "per_year_per_split_dev_only": per_year_per_split_out,
        "gate_pass": bool(gate_pass),
        "gate_verdict": gate_verdict,
        "offending_event_ids_sample": offending_event_ids_sample,
        "offending_event_ids_sample_truncated_at_20": bool(len(offending) > 20),
        "offending_per_year_missing_counts": dict(sorted(offending_per_year.items())),
        "test_split_policy": (
            "The frozen test split is never opened for content by this script: "
            "only its event COUNT (from splits.csv, an assignment table with no "
            "channel/label content) is reported in counts_per_split; no "
            "data/interim/{event_id}.json belonging to a test-split event is ever "
            "read."
        ),
    }
    return census


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml (relative paths resolve against the project root).")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    cfg = load_config(str(config_path))

    census = build_census(cfg, config_path)

    out_dir = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pl_gate_census.json"
    out_path.write_text(json.dumps(census, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in census.items() if k != "per_year_per_split_dev_only"}, indent=2))
    print(f"\nper_year_per_split_dev_only: {json.dumps(census['per_year_per_split_dev_only'], indent=2)}")
    print(f"\nGATE: {'PASS' if census['gate_pass'] else 'FAIL'}")
    print(f"report: {out_path}")

    if not census["gate_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
