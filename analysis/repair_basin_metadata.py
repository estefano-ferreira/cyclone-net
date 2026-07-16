"""
Repair basin metadata in interim JSON files from corrected event-list parser.

The IBTrACS basin code "NA" (North Atlantic) was lost during event-list
generation due to pandas' default na_values including "NA". The parser
(src.processors.ibtracs) now reads with keep_default_na=False, so the
literal "NA" code survives. This script regenerates basin fields in
existing interim JSON files from the corrected event list, mimicking
exactly what process_event would have written. Data cubes (.npy) and
raw ERA5 files are unchanged. This is ERRATA item 7 repair.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.processors.preprocess_scientific import to_event_id  # noqa: E402

logger = logging.getLogger(__name__)


def load_event_list_map(
    event_list_path: Path,
) -> Dict[str, str]:
    """Load event list and build event_id -> basin map."""
    df = pd.read_csv(event_list_path, keep_default_na=False, na_values=[""])

    if "timestamp" not in df.columns:
        raise ValueError("event_list must contain 'timestamp' column")
    if "basin" not in df.columns:
        raise ValueError("event_list must contain 'basin' column")
    if "sid" not in df.columns:
        raise ValueError("event_list must contain 'sid' column")

    df["dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["dt"].isna().any():
        raise ValueError("Some timestamp values could not be parsed")

    event_map: Dict[str, str] = {}
    for _, row in df.iterrows():
        eid = to_event_id(row["dt"].to_pydatetime(), str(row["sid"]))
        event_map[eid] = str(row["basin"])

    return event_map


def compare_event_lists(
    new_list_path: Path,
    old_list_path: Path,
) -> Tuple[bool, str]:
    """Compare new and old event lists. Return (ok, detail_message)."""
    new_df = pd.read_csv(new_list_path, keep_default_na=False, na_values=[""])
    old_df = pd.read_csv(old_list_path, keep_default_na=False, na_values=[""])

    if len(new_df) != len(old_df):
        return False, f"Row count mismatch: new {len(new_df)}, old {len(old_df)}"

    if set(new_df.columns) != set(old_df.columns):
        return (
            False,
            f"Column mismatch: new {set(new_df.columns)}, old {set(old_df.columns)}",
        )

    # Rows are positionally aligned: both lists come from the same
    # deterministic parser over the same raw file (sorted by sid, timestamp).
    # Series.equals treats NaN == NaN, which is what we want here.
    for col in new_df.columns:
        if col == "basin":
            continue
        new_col = new_df[col].reset_index(drop=True)
        old_col = old_df[col].reset_index(drop=True)
        if not new_col.equals(old_col):
            n_diff = int((~(new_col.fillna("__nan__") == old_col.fillna("__nan__"))).sum())
            return (
                False,
                f"Column '{col}' differs in {n_diff} rows (non-basin regression)",
            )

    basin_transitions: Dict[str, int] = defaultdict(int)
    for old_b, new_b in zip(
        old_df["basin"].fillna("").astype(str), new_df["basin"].fillna("").astype(str)
    ):
        basin_transitions[f"{old_b!r} -> {new_b!r}"] += 1

    detail_lines = [f"  {k}: {v}" for k, v in sorted(basin_transitions.items())]
    return True, "Basin transitions:\n" + "\n".join(detail_lines)


def repair_interim_jsons(
    event_map: Dict[str, str],
    interim_dir: Path,
    valid_events_path: Optional[Path] = None,
    execute: bool = False,
) -> Dict[str, Any]:
    """Scan and repair interim JSON files.

    Returns dict with repair statistics.
    """
    stats = {
        "unchanged": 0,
        "updated": 0,
        "unmatched": 0,
        "stem_mismatch": 0,
        "unmatched_examples": [],
        "repairs_by_transition": defaultdict(int),
        "basin_dist_before": defaultdict(int),
        "basin_dist_after": defaultdict(int),
        "valid_event_basin_before": defaultdict(int),
        "valid_event_basin_after": defaultdict(int),
        "genesis_basin_count": defaultdict(int),
        "multibasin_sids": [],
    }

    if valid_events_path and valid_events_path.exists():
        valid_df = pd.read_csv(valid_events_path, keep_default_na=False, na_values=[""])
        valid_event_ids = set(valid_df["event_id"].values) if "event_id" in valid_df.columns else set()
    else:
        valid_event_ids = set()

    json_files = sorted(interim_dir.glob("*.json"))

    for json_file in json_files:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read {json_file}: {e}")
            continue

        event_id = data.get("event_id")
        if not event_id:
            logger.warning(f"No event_id in {json_file}")
            continue

        expected_stem = f"{event_id}.json"
        if json_file.name != expected_stem:
            stats["stem_mismatch"] += 1
            logger.warning(
                f"Stem mismatch: {json_file.name} vs expected {expected_stem}"
            )
            continue

        stats["basin_dist_before"][str(data.get("basin", ""))] += 1
        if event_id in valid_event_ids:
            stats["valid_event_basin_before"][str(data.get("basin", ""))] += 1

        if event_id not in event_map:
            stats["unmatched"] += 1
            if len(stats["unmatched_examples"]) < 20:
                stats["unmatched_examples"].append(event_id)
            # Unmatched files keep their current basin; count them in the
            # after-distribution too so before/after totals stay comparable.
            stats["basin_dist_after"][str(data.get("basin", ""))] += 1
            if event_id in valid_event_ids:
                stats["valid_event_basin_after"][str(data.get("basin", ""))] += 1
            continue

        expected_basin = event_map[event_id]
        current_basin = str(data.get("basin", ""))

        if current_basin == expected_basin:
            stats["unchanged"] += 1
        else:
            transition = f"{current_basin!r} -> {expected_basin!r}"
            stats["repairs_by_transition"][transition] += 1
            stats["updated"] += 1

            if execute:
                data_copy = dict(data)
                data_copy["basin"] = expected_basin

                with json_file.open("w", encoding="utf-8") as f:
                    json.dump(data_copy, f, indent=2)

                data_verify = json.loads(json_file.read_text(encoding="utf-8"))
                for key in data:
                    if key == "basin":
                        continue
                    if data[key] != data_verify[key]:
                        raise RuntimeError(
                            f"Post-write verification failed for {json_file}: {key}"
                        )

        stats["basin_dist_after"][expected_basin] += 1
        if event_id in valid_event_ids:
            stats["valid_event_basin_after"][expected_basin] += 1

    return stats


def load_genesis_basin_per_sid(event_list_path: Path) -> Dict[str, str]:
    """Load event list and build sid -> first basin (genesis)."""
    df = pd.read_csv(event_list_path, keep_default_na=False, na_values=[""])
    df["dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("dt")

    genesis_map: Dict[str, str] = {}
    for sid, group in df.groupby("sid"):
        first_basin = group.iloc[0]["basin"]
        genesis_map[str(sid)] = str(first_basin)

    return genesis_map


def compute_multibasin_sids(event_list_path: Path) -> list:
    """Find SIDs with events in multiple basins."""
    df = pd.read_csv(event_list_path, keep_default_na=False, na_values=[""])

    result = []
    for sid, group in df.groupby("sid"):
        basins = set(group["basin"].values)
        if len(basins) > 1:
            name = group.iloc[0].get("storm_name", "")
            result.append({
                "sid": str(sid),
                "name": str(name),
                "basins": sorted(basins),
                "event_count": len(group),
            })

    return sorted(result, key=lambda x: x["event_count"], reverse=True)


def build_manifest(
    event_list_path: Path,
    old_event_list_path: Optional[Path],
    interim_dir: Path,
    valid_events_path: Optional[Path],
    provenance_dir: Path,
    stats: Dict[str, Any],
    genesis_basin_count: Dict[str, int],
    multibasin_sids: list,
    regression_ok: bool,
    regression_detail: str,
    execute: bool,
) -> Dict[str, Any]:
    """Build and save the repair manifest."""
    utc_now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _rel(p: Optional[Path]) -> Optional[str]:
        # Manifests are committed: paths must be repo-relative (absolute
        # paths leak the local environment and break on other machines).
        # Paths outside the repo are reduced to their basename.
        if p is None:
            return None
        try:
            return Path(p).resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return f"<outside-repo>/{Path(p).name}"

    manifest = {
        "timestamp_utc": utc_now,
        "dry_run": not execute,
        "regression_ok": regression_ok,
        "regression_detail": regression_detail,
        "args": {
            "event_list": _rel(event_list_path),
            "old_event_list": _rel(old_event_list_path),
            "interim_dir": _rel(interim_dir),
            "valid_events": _rel(valid_events_path),
            "provenance_dir": _rel(provenance_dir),
            "execute": execute,
        },
        "summary": {
            "jsons_scanned": (
                stats["unchanged"]
                + stats["updated"]
                + stats["unmatched"]
                + stats["stem_mismatch"]
            ),
            "jsons_unchanged": stats["unchanged"],
            "jsons_updated": stats["updated"],
            "jsons_unmatched": stats["unmatched"],
            "jsons_stem_mismatch": stats["stem_mismatch"],
            "unmatched_examples": stats["unmatched_examples"],
        },
        "repairs_by_transition": dict(stats["repairs_by_transition"]),
        "basin_distribution_before": dict(stats["basin_dist_before"]),
        "basin_distribution_after": dict(stats["basin_dist_after"]),
        "valid_event_basin_before": dict(stats["valid_event_basin_before"]),
        "valid_event_basin_after": dict(stats["valid_event_basin_after"]),
        "genesis_basin_count": dict(genesis_basin_count),
        "multibasin_storm_count": len(multibasin_sids),
        "multibasin_storms": multibasin_sids[:20],
    }

    # Expected values from the 2026-07-15 basin audit (ERRATA item 7).
    # NOTE: the audit's genesis criterion was the first point of the raw
    # IBTrACS record; here genesis is the first event-list record per SID.
    # The two can differ for basin-crossing storms (audit: alternative
    # criteria move <= 2 storms), so a small genesis delta is explainable;
    # the per-event distribution has no such caveat.
    va = manifest["valid_event_basin_after"]
    gb = manifest["genesis_basin_count"]
    manifest["expected_checks"] = {
        "valid_events_per_point": {
            "expected": {"EP": 8888, "NA": 7892, "total": 16780},
            "obtained": {
                "EP": va.get("EP", 0),
                "NA": va.get("NA", 0),
                "total": sum(va.values()),
            },
            "match": va.get("EP", 0) == 8888
            and va.get("NA", 0) == 7892
            and sum(va.values()) == 16780,
        },
        "valid_storms_genesis": {
            "expected": {"EP": 578, "NA": 414, "total": 992},
            "obtained": {
                "EP": gb.get("EP", 0),
                "NA": gb.get("NA", 0),
                "total": sum(gb.values()),
            },
            "match": gb.get("EP", 0) == 578
            and gb.get("NA", 0) == 414
            and sum(gb.values()) == 992,
        },
    }

    manifest_path = (
        provenance_dir / f"basin_metadata_repair_{utc_now}.json"
    )
    provenance_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def print_summary(manifest: Dict[str, Any]) -> None:
    """Print human-readable summary to stdout (ASCII only)."""
    print("\n" + "=" * 70)
    print("Basin Metadata Repair Summary")
    print("=" * 70)
    print()

    if not manifest["dry_run"]:
        print("[EXECUTED] Changes written to JSON files.")
    else:
        print("[DRY RUN] No changes written (use --execute to apply).")
    print()

    print(f"JSONs scanned: {manifest['summary']['jsons_scanned']}")
    print(f"  Unchanged: {manifest['summary']['jsons_unchanged']}")
    print(f"  Updated: {manifest['summary']['jsons_updated']}")
    print(f"  Unmatched: {manifest['summary']['jsons_unmatched']}")
    print(f"  Stem mismatch: {manifest['summary']['jsons_stem_mismatch']}")
    print()

    if manifest["summary"]["unmatched_examples"]:
        print("Unmatched event_id examples (first 20):")
        for eid in manifest["summary"]["unmatched_examples"][:5]:
            print(f"  {eid}")
        if len(manifest["summary"]["unmatched_examples"]) > 5:
            print(f"  ... and {len(manifest['summary']['unmatched_examples']) - 5} more")
        print()

    if manifest["repairs_by_transition"]:
        print("Basin transitions:")
        for transition, count in sorted(manifest["repairs_by_transition"].items()):
            print(f"  {transition}: {count}")
        print()

    print("Basin distribution (after repair):")
    for basin, count in sorted(manifest["basin_distribution_after"].items()):
        print(f"  {basin}: {count}")
    print()

    if manifest["valid_event_basin_after"]:
        print("Valid-event basin distribution (after repair):")
        total_valid = sum(manifest["valid_event_basin_after"].values())
        for basin in sorted(manifest["valid_event_basin_after"].keys()):
            count = manifest["valid_event_basin_after"][basin]
            print(f"  {basin}: {count}")
        print(f"  Total: {total_valid}")
        print()

    if manifest["genesis_basin_count"]:
        print("Genesis basin (first event per SID) in valid events:")
        for basin in sorted(manifest["genesis_basin_count"].keys()):
            count = manifest["genesis_basin_count"][basin]
            print(f"  {basin}: {count}")
        print()

    if "expected_checks" in manifest:
        print("Audit expectation checks (2026-07-15 basin audit):")
        for name, chk in manifest["expected_checks"].items():
            status = "MATCH" if chk["match"] else "MISMATCH"
            print(f"  {name}: {status} expected={chk['expected']} obtained={chk['obtained']}")
        print()

    print(f"Manifest saved to: {manifest['args']['provenance_dir']}/basin_metadata_repair_*.json")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repair basin metadata in interim JSON files from corrected event list."
    )
    parser.add_argument(
        "--event-list",
        type=Path,
        default=Path("./data/event_list_augmented.csv"),
        help="Path to the corrected event list (default: ./data/event_list_augmented.csv)",
    )
    parser.add_argument(
        "--old-event-list",
        type=Path,
        default=None,
        help="Path to pre-fix event list for regression comparison (optional)",
    )
    parser.add_argument(
        "--interim-dir",
        type=Path,
        default=Path("./data/interim"),
        help="Directory containing interim JSON files (default: ./data/interim)",
    )
    parser.add_argument(
        "--valid-events",
        type=Path,
        default=Path("./data/normalized/valid_events.csv"),
        help="Path to valid events CSV for filtering statistics (default: ./data/normalized/valid_events.csv)",
    )
    parser.add_argument(
        "--provenance-dir",
        type=Path,
        default=Path("./outputs/provenance"),
        help="Directory for repair manifest (default: ./outputs/provenance)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually write repaired JSON files; otherwise dry-run",
    )

    args = parser.parse_args()

    event_list_path = args.event_list
    old_event_list_path = args.old_event_list
    interim_dir = args.interim_dir
    valid_events_path = args.valid_events
    provenance_dir = args.provenance_dir
    execute = args.execute

    if not event_list_path.exists():
        print(f"Error: event list not found: {event_list_path}", file=sys.stderr)
        return 1

    if not interim_dir.exists():
        print(f"Error: interim directory not found: {interim_dir}", file=sys.stderr)
        return 1

    try:
        event_map = load_event_list_map(event_list_path)
    except Exception as e:
        print(f"Error loading event list: {e}", file=sys.stderr)
        return 1

    regression_ok = True
    regression_detail = ""

    if old_event_list_path:
        if not old_event_list_path.exists():
            print(
                f"Error: old event list not found: {old_event_list_path}",
                file=sys.stderr,
            )
            return 1

        try:
            regression_ok, regression_detail = compare_event_lists(
                event_list_path, old_event_list_path
            )
            if not regression_ok and execute:
                print(f"Regression detected; aborting:\n{regression_detail}", file=sys.stderr)
                return 1
        except Exception as e:
            print(f"Error comparing event lists: {e}", file=sys.stderr)
            return 1

    try:
        stats = repair_interim_jsons(
            event_map, interim_dir, valid_events_path, execute
        )
    except Exception as e:
        print(f"Error repairing interim JSONs: {e}", file=sys.stderr)
        return 1

    try:
        genesis_map = load_genesis_basin_per_sid(event_list_path)
    except Exception as e:
        print(f"Error loading genesis map: {e}", file=sys.stderr)
        return 1

    if valid_events_path and valid_events_path.exists():
        try:
            valid_df = pd.read_csv(
                valid_events_path, keep_default_na=False, na_values=[""]
            )
            if "sid" in valid_df.columns:
                genesis_basin_count = defaultdict(int)
                for sid in valid_df["sid"].unique():
                    basin = genesis_map.get(str(sid), "")
                    genesis_basin_count[basin] += 1
                genesis_basin_count = dict(genesis_basin_count)
            else:
                genesis_basin_count = {}
        except Exception as e:
            logger.warning(f"Could not compute genesis basin count: {e}")
            genesis_basin_count = {}
    else:
        genesis_basin_count = {}

    try:
        multibasin_sids = compute_multibasin_sids(event_list_path)
    except Exception as e:
        logger.warning(f"Could not compute multibasin SIDs: {e}")
        multibasin_sids = []

    try:
        manifest = build_manifest(
            event_list_path,
            old_event_list_path,
            interim_dir,
            valid_events_path,
            provenance_dir,
            stats,
            genesis_basin_count,
            multibasin_sids,
            regression_ok,
            regression_detail,
            execute,
        )
    except Exception as e:
        print(f"Error building manifest: {e}", file=sys.stderr)
        return 1

    print_summary(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
