#!/usr/bin/env python3
"""
READ-ONLY impact assessment of positional vs temporal RI labeling in dv24.

Quantifies two defects:
1. Positional: row-shift assumes perfect 6h grid, but IBTrACS has irregular timestamps
2. Border: trailing rows get NaN, which coerce to ri_label=0 ("no RI" instead of "undefined")

Computes exact-match temporal labeling as alternative and reports transition matrices,
impact on valid dataset, and per-split/basin breakdowns.

Outputs: JSON + MD reports to outputs/results/dv24_impact/
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np


def convert_to_python_types(obj):
    """Recursively convert numpy/pandas types to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

# ==============================================================================
# CONFIGURATION
# ==============================================================================

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs" / "results" / "dv24_impact"
RI_THRESHOLD_KT = 30.0

# CSV read parameters to preserve "NA" (North Atlantic basin name)
CSV_PARAMS = {
    "keep_default_na": False,
    "na_values": [""],
}


# ==============================================================================
# STEP 0: DATA SANITY
# ==============================================================================

def check_data_sanity(df):
    """Parse timestamps, check sortedness and duplicates."""
    report = {
        "unsorted_sids": [],
        "unsorted_sids_count": 0,
        "duplicate_timestamps_count": 0,
        "duplicate_sid_timestamp_pairs": [],
    }

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    for sid in df["sid"].unique():
        sid_df = df[df["sid"] == sid].sort_values("timestamp", ignore_index=True)
        sid_df_original = df[df["sid"] == sid]

        # Check if original is sorted
        if not sid_df["timestamp"].equals(sid_df_original["timestamp"].reset_index(drop=True)):
            report["unsorted_sids"].append(sid)
            report["unsorted_sids_count"] += 1

        # Check for duplicate timestamps
        dups = sid_df["timestamp"].duplicated().sum()
        if dups > 0:
            report["duplicate_timestamps_count"] += dups
            dup_pairs = sid_df[sid_df["timestamp"].duplicated(keep=False)][["timestamp"]].drop_duplicates()
            for ts in dup_pairs["timestamp"].unique():
                report["duplicate_sid_timestamp_pairs"].append({"sid": sid, "timestamp": str(ts), "count": len(sid_df[sid_df["timestamp"] == ts])})

    return df, report


def verify_positional_labels(df):
    """Reproduce positional labeling and verify against shipped values."""
    df = df.copy()
    df["wind_kt"] = pd.to_numeric(df["wind_kt"], errors="coerce")

    # Reproduce positional labeling per src/processors/ri_labeling.py
    df["dv12_pos"] = df.groupby("sid")["wind_kt"].shift(-2) - df["wind_kt"]
    df["dv24_pos"] = df.groupby("sid")["wind_kt"].shift(-4) - df["wind_kt"]
    df["ri_label_pos"] = (df["dv24_pos"] >= RI_THRESHOLD_KT).astype(int)

    # Verify against shipped values (handle NaN carefully)
    def compare_with_na_handling(a, b):
        """NaN == NaN counts as match."""
        return ((a.isna() & b.isna()) | ((~a.isna()) & (~b.isna()) & (a == b))).sum()

    dv12_matches = compare_with_na_handling(df["dv12_pos"], df["dv12_kt"])
    dv24_matches = compare_with_na_handling(df["dv24_pos"], df["dv24_kt"])
    ri_label_matches = (df["ri_label_pos"] == df["ri_label"]).sum()

    report = {
        "dv12_matches": dv12_matches,
        "dv12_mismatches": len(df) - dv12_matches,
        "dv24_matches": dv24_matches,
        "dv24_mismatches": len(df) - dv24_matches,
        "ri_label_matches": ri_label_matches,
        "ri_label_mismatches": len(df) - ri_label_matches,
    }

    return df, report


# ==============================================================================
# STEP 1: TEMPORAL GRID CENSUS
# ==============================================================================

def analyze_temporal_grid(df):
    """Analyze time deltas and report grid regularity."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Sort by (sid, timestamp) for temporal analysis
    df = df.sort_values(["sid", "timestamp"], ignore_index=True)

    report = {
        "time_delta_distribution": {},
        "sids_with_non_6h_delta": 0,
        "rows_with_non_6h_next_delta": 0,
        "sub_6h_extra_points": 0,
    }

    for sid in df["sid"].unique():
        sid_df = df[df["sid"] == sid].sort_values("timestamp", ignore_index=True)

        # Compute deltas to next row
        if len(sid_df) > 1:
            deltas = sid_df["timestamp"].diff()[1:].dt.total_seconds() / 3600  # hours
            delta_counts = deltas.value_counts().to_dict()

            for delta_hours, count in delta_counts.items():
                delta_hours = int(delta_hours) if delta_hours == int(delta_hours) else delta_hours
                if delta_hours not in report["time_delta_distribution"]:
                    report["time_delta_distribution"][delta_hours] = 0
                report["time_delta_distribution"][delta_hours] += count

            # Count non-6h deltas
            non_6h = (deltas != 6).sum()
            if non_6h > 0:
                report["sids_with_non_6h_delta"] += 1
                report["rows_with_non_6h_next_delta"] += non_6h

            # Count sub-6h points (extra landfall fixes, etc.)
            sub_6h = (deltas < 6).sum()
            if sub_6h > 0:
                report["sub_6h_extra_points"] += sub_6h

    # Sort delta distribution by key for readability
    report["time_delta_distribution"] = dict(sorted(report["time_delta_distribution"].items()))

    return report


# ==============================================================================
# STEP 2: TEMPORAL LABELING (EXACT-MATCH)
# ==============================================================================

def compute_temporal_labels(df):
    """Compute exact-match temporal labels via self-merge."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["wind_kt"] = pd.to_numeric(df["wind_kt"], errors="coerce")

    # Sort by (sid, timestamp) for merge
    df = df.sort_values(["sid", "timestamp"], ignore_index=True)

    # Handle duplicate timestamps: keep first for right side
    df_right = df.drop_duplicates(subset=["sid", "timestamp"], keep="first").copy()

    # Merge for dv12 (timestamp + 12h)
    df["timestamp_plus_12h"] = df["timestamp"] + pd.Timedelta(hours=12)
    dv12_merge = df[["sid", "timestamp_plus_12h"]].merge(
        df_right[["sid", "timestamp", "wind_kt"]].rename(columns={"wind_kt": "wind_kt_12h"}),
        left_on=["sid", "timestamp_plus_12h"],
        right_on=["sid", "timestamp"],
        how="left",
    )
    df["dv12_tmp"] = dv12_merge["wind_kt_12h"] - df["wind_kt"]

    # Merge for dv24 (timestamp + 24h)
    df["timestamp_plus_24h"] = df["timestamp"] + pd.Timedelta(hours=24)
    dv24_merge = df[["sid", "timestamp_plus_24h"]].merge(
        df_right[["sid", "timestamp", "wind_kt"]].rename(columns={"wind_kt": "wind_kt_24h"}),
        left_on=["sid", "timestamp_plus_24h"],
        right_on=["sid", "timestamp"],
        how="left",
    )
    df["dv24_tmp"] = dv24_merge["wind_kt_24h"] - df["wind_kt"]

    # Assign temporal labels
    df["ri_label_tmp"] = pd.NA

    # Only assign 0 or 1 if both wind values are defined
    defined_mask = df["wind_kt"].notna() & (~df["dv24_tmp"].isna())
    df.loc[defined_mask & (df["dv24_tmp"] < RI_THRESHOLD_KT), "ri_label_tmp"] = 0
    df.loc[defined_mask & (df["dv24_tmp"] >= RI_THRESHOLD_KT), "ri_label_tmp"] = 1

    # Rest remain pd.NA (UNDEFINED)

    # Drop temporary columns
    df = df.drop(columns=["timestamp_plus_12h", "timestamp_plus_24h"])

    return df


# ==============================================================================
# STEP 3: IMPACT QUANTIFICATION (FULL EVENT LIST)
# ==============================================================================

def quantify_impact_full_list(df):
    """Quantify impacts on the full 32,989-row event list."""
    report = {
        "dv12": {},
        "dv24": {},
    }

    for dv in ["dv12", "dv24"]:
        pos_col = f"{dv}_pos"
        tmp_col = f"{dv}_tmp"
        pos_label_col = "ri_label_pos"
        tmp_label_col = "ri_label_tmp"

        # Rows where positional partner exists but is NOT temporal partner
        pos_defined = df[pos_col].notna()
        tmp_defined = df[tmp_col].notna()
        pos_but_not_tmp = (pos_defined & ~tmp_defined).sum()

        report[dv]["rows_pos_not_tmp"] = int(pos_but_not_tmp)
        report[dv]["rows_pos_not_tmp_pct"] = round(100.0 * pos_but_not_tmp / len(df), 2)

        # Continuous dv value changes
        both_defined = pos_defined & tmp_defined
        changes = df.loc[both_defined, pos_col] - df.loc[both_defined, tmp_col]
        changes = changes.abs()

        report[dv]["continuous_value_changes"] = {
            "count": int(both_defined.sum()),
            "min": float(changes.min()) if len(changes) > 0 else None,
            "median": float(changes.median()) if len(changes) > 0 else None,
            "p95": float(changes.quantile(0.95)) if len(changes) > 0 else None,
            "max": float(changes.max()) if len(changes) > 0 else None,
        }

    # dv24 transition matrix and border defect
    pos_0 = df["ri_label_pos"] == 0
    pos_1 = df["ri_label_pos"] == 1
    tmp_0 = df["ri_label_tmp"] == 0
    tmp_1 = df["ri_label_tmp"] == 1
    tmp_undef = df["ri_label_tmp"].isna()

    report["dv24"]["transition_matrix"] = {
        "pos_0_to_tmp_0": int((pos_0 & tmp_0).sum()),
        "pos_0_to_tmp_1": int((pos_0 & tmp_1).sum()),
        "pos_0_to_tmp_undef": int((pos_0 & tmp_undef).sum()),
        "pos_1_to_tmp_0": int((pos_1 & tmp_0).sum()),
        "pos_1_to_tmp_1": int((pos_1 & tmp_1).sum()),
        "pos_1_to_tmp_undef": int((pos_1 & tmp_undef).sum()),
    }

    # Border defect: positional 0 that are truly undefined (no dv24 partner)
    # This is the count of pos_0 that have no temporal partner
    report["dv24"]["border_defect_count"] = int((pos_0 & tmp_undef).sum())

    # Positives before vs after
    pos_positives = (df["ri_label_pos"] == 1).sum()
    tmp_positives = (df["ri_label_tmp"] == 1).sum()

    report["positives_shipped"] = int(pos_positives)
    report["positives_temporal"] = int(tmp_positives)
    report["positives_difference"] = int(tmp_positives - pos_positives)

    return report


# ==============================================================================
# STEP 4: IMPACT ON VALID SET AND SPLITS
# ==============================================================================

def parse_event_id(event_id):
    """Parse event_id to extract timestamp and sid.
    Format: era5_YYYY_MM_DD_HHMM_<SID>
    """
    parts = event_id.split("_")
    # era5_YYYY_MM_DD_HHMM_SID
    year = parts[1]
    month = parts[2]
    day = parts[3]
    hhmm = parts[4]
    sid = "_".join(parts[5:])  # SID may contain underscores

    ts_str = f"{year}-{month}-{day} {hhmm[:2]}:{hhmm[2:]}:00"
    timestamp = pd.to_datetime(ts_str)
    return timestamp, sid


def quantify_impact_valid_set(event_list, valid_events, splits):
    """Quantify impacts on valid dataset, broken by split and basin."""
    event_list = event_list.copy()
    valid_events = valid_events.copy()
    splits = splits.copy()

    # Parse event_id in valid_events to timestamp + sid
    valid_events[["timestamp_parsed", "sid_parsed"]] = valid_events["event_id"].apply(
        lambda x: pd.Series(parse_event_id(x))
    )

    # Drop original sid/timestamp from valid_events (we have parsed versions now)
    valid_events = valid_events.drop(columns=["sid", "timestamp"], errors="ignore")

    # Prepare event_list subset for merge
    event_list_subset = event_list[["sid", "timestamp", "ri_label_pos", "ri_label_tmp", "dv24_pos", "dv24_tmp", "basin"]].copy()

    # Join valid_events to event_list
    valid_with_list = valid_events.merge(
        event_list_subset,
        left_on=["sid_parsed", "timestamp_parsed"],
        right_on=["sid", "timestamp"],
        how="left",
    )

    join_coverage = valid_with_list["ri_label_pos"].notna().sum()
    report = {
        "join_coverage": int(join_coverage),
        "join_coverage_total": len(valid_events),
        "join_misses": len(valid_events) - join_coverage,
    }

    # If join is incomplete, stop trusting downstream numbers
    if join_coverage < len(valid_events):
        report["warning"] = f"Join incomplete: {len(valid_events) - join_coverage} misses"
        return report

    # Merge with splits
    valid_with_list = valid_with_list.merge(splits[["event_id", "split"]], left_on="event_id", right_on="event_id", how="left")

    # Transition matrix restricted to valid events, by split and basin
    report["valid_transitions_by_split"] = {}
    report["valid_transitions_by_basin"] = {}

    for split in ["train", "val", "test"]:
        subset = valid_with_list[valid_with_list["split"] == split]
        if len(subset) == 0:
            continue

        pos_0 = subset["ri_label_pos"] == 0
        pos_1 = subset["ri_label_pos"] == 1
        tmp_0 = subset["ri_label_tmp"] == 0
        tmp_1 = subset["ri_label_tmp"] == 1
        tmp_undef = subset["ri_label_tmp"].isna()

        report["valid_transitions_by_split"][split] = {
            "pos_0_to_tmp_0": int((pos_0 & tmp_0).sum()),
            "pos_0_to_tmp_1": int((pos_0 & tmp_1).sum()),
            "pos_0_to_tmp_undef": int((pos_0 & tmp_undef).sum()),
            "pos_1_to_tmp_0": int((pos_1 & tmp_0).sum()),
            "pos_1_to_tmp_1": int((pos_1 & tmp_1).sum()),
            "pos_1_to_tmp_undef": int((pos_1 & tmp_undef).sum()),
        }

    for basin in valid_with_list["basin"].unique():
        if pd.isna(basin):
            continue
        subset = valid_with_list[valid_with_list["basin"] == basin]

        pos_0 = subset["ri_label_pos"] == 0
        pos_1 = subset["ri_label_pos"] == 1
        tmp_0 = subset["ri_label_tmp"] == 0
        tmp_1 = subset["ri_label_tmp"] == 1
        tmp_undef = subset["ri_label_tmp"].isna()

        report["valid_transitions_by_basin"][basin] = {
            "pos_0_to_tmp_0": int((pos_0 & tmp_0).sum()),
            "pos_0_to_tmp_1": int((pos_0 & tmp_1).sum()),
            "pos_0_to_tmp_undef": int((pos_0 & tmp_undef).sum()),
            "pos_1_to_tmp_0": int((pos_1 & tmp_0).sum()),
            "pos_1_to_tmp_1": int((pos_1 & tmp_1).sum()),
            "pos_1_to_tmp_undef": int((pos_1 & tmp_undef).sum()),
        }

    # Storms whose positive count changes
    report["sids_gaining_first_positive"] = []
    report["sids_losing_last_positive"] = []

    for sid in valid_with_list["sid"].unique():
        sid_subset = valid_with_list[valid_with_list["sid"] == sid]
        pos_pos_count = (sid_subset["ri_label_pos"] == 1).sum()
        tmp_pos_count = (sid_subset["ri_label_tmp"] == 1).sum()

        # Gain first positive: had 0 positives in positional, now has >= 1 in temporal
        if pos_pos_count == 0 and tmp_pos_count > 0:
            # For train/val, list the sid; for test, count only
            split_set = sid_subset["split"].unique()
            if all(s in ["train", "val"] for s in split_set if not pd.isna(s)):
                report["sids_gaining_first_positive"].append(sid)

        # Lose last positive: had >= 1 in positional, now has 0 in temporal
        if pos_pos_count > 0 and tmp_pos_count == 0:
            split_set = sid_subset["split"].unique()
            if all(s in ["train", "val"] for s in split_set if not pd.isna(s)):
                report["sids_losing_last_positive"].append(sid)

    report["sids_gaining_first_positive_count"] = len(report["sids_gaining_first_positive"])
    report["sids_losing_last_positive_count"] = len(report["sids_losing_last_positive"])

    # Count sids gaining/losing in test split only
    test_subset = valid_with_list[valid_with_list["split"] == "test"]
    report["test_sids_gaining_first_positive"] = 0
    report["test_sids_losing_last_positive"] = 0

    for sid in test_subset["sid"].unique():
        sid_subset = test_subset[test_subset["sid"] == sid]
        pos_pos_count = (sid_subset["ri_label_pos"] == 1).sum()
        tmp_pos_count = (sid_subset["ri_label_tmp"] == 1).sum()

        if pos_pos_count == 0 and tmp_pos_count > 0:
            report["test_sids_gaining_first_positive"] += 1
        if pos_pos_count > 0 and tmp_pos_count == 0:
            report["test_sids_losing_last_positive"] += 1

    # Count valid events becoming UNDEFINED under temporal rule, by split
    report["valid_events_becoming_undefined"] = {}
    for split in ["train", "val", "test"]:
        subset = valid_with_list[valid_with_list["split"] == split]
        undef_count = subset["ri_label_tmp"].isna().sum()
        report["valid_events_becoming_undefined"][split] = int(undef_count)

    return report


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Run impact assessment."""
    print("[1/6] Loading data...", file=sys.stderr)
    event_list = pd.read_csv(DATA_DIR / "event_list_augmented.csv", **CSV_PARAMS)
    valid_events = pd.read_csv(DATA_DIR / "normalized" / "valid_events.csv", **CSV_PARAMS)
    splits = pd.read_csv(DATA_DIR / "normalized" / "splits.csv", **CSV_PARAMS)

    print(f"  - event_list: {len(event_list)} rows", file=sys.stderr)
    print(f"  - valid_events: {len(valid_events)} rows", file=sys.stderr)
    print(f"  - splits: {len(splits)} rows", file=sys.stderr)

    print("[2/6] Data sanity checks...", file=sys.stderr)
    event_list, sanity_report = check_data_sanity(event_list)
    event_list, verify_report = verify_positional_labels(event_list)

    print(f"  - Unsorted sids: {sanity_report['unsorted_sids_count']}", file=sys.stderr)
    print(f"  - Duplicate timestamps: {sanity_report['duplicate_timestamps_count']}", file=sys.stderr)
    print(f"  - dv24 label mismatches: {verify_report['dv24_mismatches']}", file=sys.stderr)

    print("[3/6] Temporal grid census...", file=sys.stderr)
    grid_report = analyze_temporal_grid(event_list)
    print(f"  - Sids with non-6h deltas: {grid_report['sids_with_non_6h_delta']}", file=sys.stderr)
    print(f"  - Rows with non-6h next-delta: {grid_report['rows_with_non_6h_next_delta']}", file=sys.stderr)

    print("[4/6] Computing temporal labels...", file=sys.stderr)
    event_list = compute_temporal_labels(event_list)

    print("[5/6] Quantifying impacts...", file=sys.stderr)
    impact_report = quantify_impact_full_list(event_list)
    valid_impact_report = quantify_impact_valid_set(event_list, valid_events, splits)

    print("[6/6] Writing outputs...", file=sys.stderr)

    # Create output directory
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate UTC timestamp for filename
    utc_now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Combine all reports
    full_report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "description": "Exact-match temporal RI labeling vs positional shift-based labeling",
            "temporal_rule": "Partner identified by exact timestamp match (t0+12h, t0+24h) within same SID",
            "nan_semantics": "UNDEFINED (pd.NA) when no temporal partner or wind is NaN; never coerce to 0",
            "read_only": True,
        },
        "inputs": {
            "event_list": "data/event_list_augmented.csv",
            "valid_events": "data/normalized/valid_events.csv",
            "splits": "data/normalized/splits.csv",
        },
        "sanity": sanity_report,
        "positional_verification": verify_report,
        "temporal_grid": grid_report,
        "full_list_impact": impact_report,
        "valid_set_impact": valid_impact_report,
    }

    # Write JSON report
    json_path = OUTPUTS_DIR / f"report_{utc_now}.json"
    with open(json_path, "w") as f:
        json.dump(convert_to_python_types(full_report), f, indent=2)

    # Write MD report
    md_report = generate_md_report(full_report)
    md_path = OUTPUTS_DIR / f"report_{utc_now}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    # Print MD to stdout (with UTF-8 encoding and error handling)
    try:
        print("\n" + md_report)
    except UnicodeEncodeError:
        # Fallback for terminals with limited encoding support
        print(md_report.encode('ascii', errors='replace').decode('ascii'))

    print(f"\nOutputs written to:", file=sys.stderr)
    print(f"  JSON: {json_path.relative_to(REPO_ROOT)}", file=sys.stderr)
    print(f"  MD:   {md_path.relative_to(REPO_ROOT)}", file=sys.stderr)


def generate_md_report(full_report):
    """Generate human-readable MD report."""
    md = []
    md.append("# DV24 Impact Assessment Report\n")
    md.append(f"**Generated:** {full_report['generated_utc']}\n")
    md.append("**Method:** Exact-match temporal RI labeling vs positional shift-based (read-only analysis)\n\n")

    # Data Sanity
    sanity = full_report["sanity"]
    md.append("## Data Sanity\n")
    md.append(f"- Unsorted SIDs: {sanity['unsorted_sids_count']}\n")
    md.append(f"- Duplicate timestamps: {sanity['duplicate_timestamps_count']}\n")
    if sanity["unsorted_sids_count"] > 0:
        md.append(f"  - **Note:** Temporal analysis used sorted copy\n")
    md.append("\n")

    # Positional Verification
    verify = full_report["positional_verification"]
    md.append("## Positional Label Verification\n")
    md.append(f"- dv24 label matches: {verify['dv24_matches']} / {verify['dv24_matches'] + verify['dv24_mismatches']}\n")
    if verify["dv24_mismatches"] > 0:
        md.append(f"  - **Warning:** {verify['dv24_mismatches']} mismatches found\n")
    md.append("\n")

    # Temporal Grid
    grid = full_report["temporal_grid"]
    md.append("## Temporal Grid Regularity\n")
    md.append(f"- SIDs with non-6h deltas: {grid['sids_with_non_6h_delta']}\n")
    md.append(f"- Rows with non-6h next-delta: {grid['rows_with_non_6h_next_delta']}\n")
    md.append(f"- Sub-6h extra points: {grid['sub_6h_extra_points']}\n")
    md.append(f"- Time delta distribution (hours): {grid['time_delta_distribution']}\n")
    md.append("\n")

    # Impact on Full List
    impact = full_report["full_list_impact"]
    md.append("## Impact on Full Event List (32,989 rows)\n\n")
    md.append("### dv24 Transition Matrix\n")
    tm = impact["dv24"]["transition_matrix"]
    md.append(f"| From | To=0 | To=1 | To=UNDEF |\n")
    md.append(f"|------|------|------|----------|\n")
    md.append(f"| pos=0 | {tm['pos_0_to_tmp_0']} | {tm['pos_0_to_tmp_1']} | {tm['pos_0_to_tmp_undef']} |\n")
    md.append(f"| pos=1 | {tm['pos_1_to_tmp_0']} | {tm['pos_1_to_tmp_1']} | {tm['pos_1_to_tmp_undef']} |\n")
    md.append("\n")

    md.append(f"### Border Defect (positional=0 that are actually UNDEFINED)\n")
    md.append(f"- Count: {impact['dv24']['border_defect_count']}\n")
    md.append("\n")

    md.append(f"### Positives Count\n")
    md.append(f"- Shipped (positional): {impact['positives_shipped']}\n")
    md.append(f"- Temporal (exact-match): {impact['positives_temporal']}\n")
    md.append(f"- Difference: {impact['positives_difference']:+d}\n")
    md.append("\n")

    md.append(f"### Rows with positional != temporal partner (dv24)\n")
    md.append(f"- Count: {impact['dv24']['rows_pos_not_tmp']}\n")
    md.append(f"- Percentage: {impact['dv24']['rows_pos_not_tmp_pct']}%\n")
    md.append("\n")

    # Valid Set Impact
    valid_impact = full_report["valid_set_impact"]
    md.append("## Impact on Valid Dataset (16,780 events)\n\n")

    if "warning" in valid_impact:
        md.append(f"**WARNING:** {valid_impact['warning']}\n")
        md.append("Downstream numbers may be unreliable.\n\n")
    else:
        md.append(f"- Join coverage: {valid_impact['join_coverage']} / {valid_impact['join_coverage_total']}\n")
        md.append("\n")

        md.append("### Transition Matrix by Split\n\n")
        for split in ["train", "val", "test"]:
            if split not in valid_impact["valid_transitions_by_split"]:
                continue
            tm = valid_impact["valid_transitions_by_split"][split]
            md.append(f"**{split.upper()}:**\n")
            md.append(f"| From | To=0 | To=1 | To=UNDEF |\n")
            md.append(f"|------|------|------|----------|\n")
            md.append(f"| pos=0 | {tm['pos_0_to_tmp_0']} | {tm['pos_0_to_tmp_1']} | {tm['pos_0_to_tmp_undef']} |\n")
            md.append(f"| pos=1 | {tm['pos_1_to_tmp_0']} | {tm['pos_1_to_tmp_1']} | {tm['pos_1_to_tmp_undef']} |\n")
            md.append("\n")

        md.append("### Transition Matrix by Basin\n\n")
        for basin in sorted(valid_impact["valid_transitions_by_basin"].keys()):
            tm = valid_impact["valid_transitions_by_basin"][basin]
            md.append(f"**{basin}:**\n")
            md.append(f"| From | To=0 | To=1 | To=UNDEF |\n")
            md.append(f"|------|------|------|----------|\n")
            md.append(f"| pos=0 | {tm['pos_0_to_tmp_0']} | {tm['pos_0_to_tmp_1']} | {tm['pos_0_to_tmp_undef']} |\n")
            md.append(f"| pos=1 | {tm['pos_1_to_tmp_0']} | {tm['pos_1_to_tmp_1']} | {tm['pos_1_to_tmp_undef']} |\n")
            md.append("\n")

        md.append("### Storms with Positive Count Changes\n")
        md.append(f"- SIDs gaining first positive (train+val): {valid_impact['sids_gaining_first_positive_count']}\n")
        if valid_impact['sids_gaining_first_positive']:
            md.append(f"  - SIDs: {', '.join(valid_impact['sids_gaining_first_positive'][:10])}")
            if len(valid_impact['sids_gaining_first_positive']) > 10:
                md.append(f" ... and {len(valid_impact['sids_gaining_first_positive']) - 10} more\n")
            else:
                md.append("\n")
        md.append(f"- SIDs losing last positive (train+val): {valid_impact['sids_losing_last_positive_count']}\n")
        if valid_impact['sids_losing_last_positive']:
            md.append(f"  - SIDs: {', '.join(valid_impact['sids_losing_last_positive'][:10])}")
            if len(valid_impact['sids_losing_last_positive']) > 10:
                md.append(f" ... and {len(valid_impact['sids_losing_last_positive']) - 10} more\n")
            else:
                md.append("\n")
        md.append(f"- SIDs gaining/losing in test split: {valid_impact['test_sids_gaining_first_positive']} gaining / {valid_impact['test_sids_losing_last_positive']} losing\n")
        md.append("\n")

        md.append("### Valid Events Becoming UNDEFINED\n")
        for split in ["train", "val", "test"]:
            if split in valid_impact["valid_events_becoming_undefined"]:
                count = valid_impact["valid_events_becoming_undefined"][split]
                md.append(f"- {split.upper()}: {count}\n")

    return "".join(md)


if __name__ == "__main__":
    main()
