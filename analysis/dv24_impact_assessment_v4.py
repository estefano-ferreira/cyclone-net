#!/usr/bin/env python3
"""
DV24 Impact Assessment v3: Complete three-effect decomposition with all scopes.

Fixes from v2:
1. Phantom positives: ship=1→UNDEF (not 1→0), from matrix
2. Effect 1 concentration: over |Δdv|>0 changed rows, not 0
3. Full dv12 decomposition: shipped vs correct-pos vs temporal
4. Per-split breakdown: train/val/test separate matrices per effect
5. Dev PL-gated scope: derived from interim JSONs, cross-checked
6. SIDs losing all positives: properly scoped per split
7. Evidence fix: global shift matches ONLY in trailing rows (expect ~959)
8. Irregular-grid census detail: SIDs/years/basins of 43 non-6h deltas
   + landfall proximity for effects

Scopes: full list, valid set (with per-split), dev PL-gated, test (aggregate).
Both dv12 and dv24.
READ-ONLY except outputs/results/dv24_impact/
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs" / "results" / "dv24_impact"
CSV_PARAMS = {"keep_default_na": False, "na_values": [""]}
RI_THRESHOLD_KT = 30.0


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
# LOAD & COMPUTE LABELS
# ==============================================================================

def load_and_compute_correct_positional(event_list_path):
    """Load event_list, compute CORRECT-POSITIONAL labels (per-SID groupby shift)."""
    df = pd.read_csv(event_list_path, **CSV_PARAMS)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["wind_kt"] = pd.to_numeric(df["wind_kt"], errors="coerce")

    # Shipped (as-is)
    df["dv12_shipped"] = df["dv12_kt"]
    df["dv24_shipped"] = df["dv24_kt"]
    df["ri_label_shipped"] = df["ri_label"]

    # CORRECT-POSITIONAL: per-SID groupby shift, NaN → undefined
    df["dv12_pos_correct"] = df.groupby("sid")["wind_kt"].shift(-2) - df["wind_kt"]
    df["dv24_pos_correct"] = df.groupby("sid")["wind_kt"].shift(-4) - df["wind_kt"]
    df["ri_label_pos_correct"] = pd.NA
    defined_24 = df["dv24_pos_correct"].notna()
    df.loc[defined_24 & (df["dv24_pos_correct"] < RI_THRESHOLD_KT), "ri_label_pos_correct"] = 0
    df.loc[defined_24 & (df["dv24_pos_correct"] >= RI_THRESHOLD_KT), "ri_label_pos_correct"] = 1

    return df


def compute_temporal_labels(df):
    """Compute exact-match temporal labels (t0+12h, t0+24h)."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["wind_kt"] = pd.to_numeric(df["wind_kt"], errors="coerce")

    df = df.sort_values(["sid", "timestamp"], ignore_index=True)
    df_right = df.drop_duplicates(subset=["sid", "timestamp"], keep="first").copy()

    # dv12
    df["timestamp_plus_12h"] = df["timestamp"] + pd.Timedelta(hours=12)
    dv12_merge = df[["sid", "timestamp_plus_12h"]].merge(
        df_right[["sid", "timestamp", "wind_kt"]].rename(columns={"wind_kt": "wind_kt_12h"}),
        left_on=["sid", "timestamp_plus_12h"],
        right_on=["sid", "timestamp"],
        how="left",
    )
    df["dv12_tmp"] = dv12_merge["wind_kt_12h"] - df["wind_kt"]

    # dv24
    df["timestamp_plus_24h"] = df["timestamp"] + pd.Timedelta(hours=24)
    dv24_merge = df[["sid", "timestamp_plus_24h"]].merge(
        df_right[["sid", "timestamp", "wind_kt"]].rename(columns={"wind_kt": "wind_kt_24h"}),
        left_on=["sid", "timestamp_plus_24h"],
        right_on=["sid", "timestamp"],
        how="left",
    )
    df["dv24_tmp"] = dv24_merge["wind_kt_24h"] - df["wind_kt"]

    # Temporal labels
    df["ri_label_tmp"] = pd.NA
    defined_tmp = df["wind_kt"].notna() & (~df["dv24_tmp"].isna())
    df.loc[defined_tmp & (df["dv24_tmp"] < RI_THRESHOLD_KT), "ri_label_tmp"] = 0
    df.loc[defined_tmp & (df["dv24_tmp"] >= RI_THRESHOLD_KT), "ri_label_tmp"] = 1

    df = df.drop(columns=["timestamp_plus_12h", "timestamp_plus_24h"])
    return df


# ==============================================================================
# DEFECT 0 DIAGNOSIS (FIXED FOR TRAILING ROWS ONLY)
# ==============================================================================

def diagnose_defect_0_trailing_only(df_original):
    """Diagnose shipped bleed: global shift matches in sorted order AND original order."""
    df = df_original.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["sid", "timestamp"], ignore_index=True)

    report = {
        "shipped_vs_correct_pos_full": {
            "dv12_mismatches": int((df["dv12_shipped"].notna() & df["dv12_pos_correct"].isna()).sum()),
            "dv24_mismatches": int((df["dv24_shipped"].notna() & df["dv24_pos_correct"].isna()).sum()),
        },
        "trailing_row_analysis": {
            "dv24_trailing_mismatches": 0,
            "dv24_interior_exact_matches": 0,
            "dv24_trailing_global_shift_matches_sorted_order": 0,
            "dv24_trailing_global_shift_matches_original_order": 0,
        },
        "sids_losing_all_positives": {
            "full_list_count": 0,
            "sids": [],
        },
    }

    # Trailing analysis per SID
    for sid in df["sid"].unique():
        sid_df = df[df["sid"] == sid]
        n = len(sid_df)

        if n > 0:
            trailing_indices = list(range(max(0, n - 4), n))
            interior_indices = list(range(0, max(0, n - 4)))

            if len(trailing_indices) > 0:
                trailing = sid_df.iloc[trailing_indices]
                trailing_mismatch = (
                    (trailing["dv24_shipped"].notna() & trailing["dv24_pos_correct"].isna()).sum()
                )
                report["trailing_row_analysis"]["dv24_trailing_mismatches"] += trailing_mismatch

            if len(interior_indices) > 0:
                interior = sid_df.iloc[interior_indices]
                interior_match = (
                    (interior["dv24_shipped"] == interior["dv24_pos_correct"]).sum()
                )
                report["trailing_row_analysis"]["dv24_interior_exact_matches"] += interior_match

        # SIDs losing all positives
        shipped_pos = (sid_df["ri_label_shipped"] == 1).sum()
        correct_pos = (sid_df["ri_label_pos_correct"] == 1).sum()

        if shipped_pos > 0 and correct_pos == 0:
            report["sids_losing_all_positives"]["full_list_count"] += 1
            report["sids_losing_all_positives"]["sids"].append(sid)

    # Global shift matches: ONLY in trailing rows (SORTED ORDER)
    df_global_sorted = df.sort_values("timestamp", ignore_index=True).copy()
    df_global_sorted["dv24_global"] = df_global_sorted["wind_kt"].shift(-4) - df_global_sorted["wind_kt"]

    for sid in df["sid"].unique():
        sid_df = df[df["sid"] == sid]
        n = len(sid_df)
        trailing_indices = list(range(max(0, n - 4), n))

        if len(trailing_indices) > 0:
            trailing = sid_df.iloc[trailing_indices]
            for orig_idx in trailing.index:
                if orig_idx < len(df_global_sorted) - 4:
                    if (df.loc[orig_idx, "dv24_shipped"] is not pd.NA and
                        df.loc[orig_idx, "dv24_shipped"] == df_global_sorted.loc[orig_idx, "dv24_global"]):
                        report["trailing_row_analysis"]["dv24_trailing_global_shift_matches_sorted_order"] += 1

    # Global shift matches in ORIGINAL file order
    df_orig_copy = df_original.copy()
    df_orig_copy["timestamp"] = pd.to_datetime(df_orig_copy["timestamp"])
    df_orig_copy["dv24_global"] = df_orig_copy["wind_kt"].shift(-4) - df_orig_copy["wind_kt"]

    # Find trailing rows in original order (by SID sorting within each SID)
    for sid in df_orig_copy["sid"].unique():
        sid_orig = df_orig_copy[df_orig_copy["sid"] == sid].copy()
        sid_orig = sid_orig.sort_values("timestamp", ignore_index=True)
        n = len(sid_orig)

        if n > 0:
            trailing_indices = list(range(max(0, n - 4), n))
            if len(trailing_indices) > 0:
                trailing = sid_orig.iloc[trailing_indices]
                for _, row in trailing.iterrows():
                    # Find this row in the original (unsorted) dataframe
                    orig_row = df_orig_copy[
                        (df_orig_copy["sid"] == row["sid"]) &
                        (df_orig_copy["timestamp"] == row["timestamp"])
                    ]
                    if len(orig_row) > 0:
                        orig_idx = orig_row.index[0]
                        if (df_orig_copy.loc[orig_idx, "dv24_shipped"] is not pd.NA and
                            df_orig_copy.loc[orig_idx, "dv24_shipped"] == df_orig_copy.loc[orig_idx, "dv24_global"]):
                            report["trailing_row_analysis"]["dv24_trailing_global_shift_matches_original_order"] += 1

    return report


# ==============================================================================
# IRREGULAR GRID CENSUS
# ==============================================================================

def analyze_irregular_grid(df, splits):
    """Analyze 43 non-6h deltas: SIDs attributed to train/val/test/unsplit (properly)."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["sid", "timestamp"], ignore_index=True)

    # Build SID→split mapping from splits (which has event_id → split)
    split_map = {}
    for _, row in splits.iterrows():
        event_id = row["event_id"]
        split = row["split"]
        # Extract SID from event_id (format: era5_YYYY_MM_DD_HHMM_<SID>)
        parts = event_id.split("_")
        sid = "_".join(parts[5:])
        if sid not in split_map:
            split_map[sid] = split

    report = {
        "total_non_6h_deltas": 0,
        "affected_sids_by_category": {
            "train": {"count": 0, "sids": []},
            "val": {"count": 0, "sids": []},
            "test": {"count": 0},
            "unsplit_rejected": {"count": 0, "sids": []},
        },
        "deltas_by_years": defaultdict(int),
        "deltas_by_basin": defaultdict(int),
        "deltas_by_category_years": {
            "train": defaultdict(int),
            "val": defaultdict(int),
            "unsplit_rejected": defaultdict(int),
        },
        "deltas_by_category_basin": {
            "train": defaultdict(int),
            "val": defaultdict(int),
            "unsplit_rejected": defaultdict(int),
        },
    }

    for sid in df["sid"].unique():
        sid_df = df[df["sid"] == sid].sort_values("timestamp")
        if len(sid_df) > 1:
            deltas = sid_df["timestamp"].diff()[1:].dt.total_seconds() / 3600
            non_6h = (deltas != 6).sum()

            if non_6h > 0:
                report["total_non_6h_deltas"] += non_6h

                # Determine split category
                year = int(sid[:4])
                basin = sid_df["basin"].iloc[0]
                report["deltas_by_years"][year] += 1
                report["deltas_by_basin"][basin] += 1

                if sid in split_map:
                    category = split_map[sid]
                    if category in ["train", "val"]:
                        report["affected_sids_by_category"][category]["sids"].append(sid)
                        report["affected_sids_by_category"][category]["count"] += 1
                        report["deltas_by_category_years"][category][year] += 1
                        report["deltas_by_category_basin"][category][basin] += 1
                    else:  # test
                        report["affected_sids_by_category"]["test"]["count"] += 1
                else:
                    # Not in valid set (unsplit/rejected)
                    report["affected_sids_by_category"]["unsplit_rejected"]["sids"].append(sid)
                    report["affected_sids_by_category"]["unsplit_rejected"]["count"] += 1
                    report["deltas_by_category_years"]["unsplit_rejected"][year] += 1
                    report["deltas_by_category_basin"]["unsplit_rejected"][basin] += 1

    return report


# ==============================================================================
# PL-GATE DERIVATION
# ==============================================================================

def derive_pl_gated_subset(valid_events, splits):
    """Derive PL-gated membership from interim JSON metadata (train+val only)."""
    interim_dir = DATA_DIR / "interim"

    # Load split assignments
    train_val_splits = splits[splits["split"].isin(["train", "val"])]

    pl_gated_count = 0
    pl_gated_event_ids = set()

    print(f"  [PL-gate] Scanning {len(train_val_splits)} train+val events for PL channels...", file=sys.stderr)

    for event_id in train_val_splits["event_id"]:
        interim_path = interim_dir / f"{event_id}.json"

        if interim_path.exists():
            try:
                with open(interim_path) as f:
                    metadata = json.load(f)

                # Check for required PL channels
                channels = metadata.get("channels", [])
                if "shear_850_200_mps" in channels and "rh_mid" in channels:
                    pl_gated_count += 1
                    pl_gated_event_ids.add(event_id)
            except (json.JSONDecodeError, IOError):
                pass

    return pl_gated_count, pl_gated_event_ids


# ==============================================================================
# THREE-EFFECT DECOMPOSITION (DV12 & DV24)
# ==============================================================================

def effect_0_shipped_vs_correct(df, dv_col):
    """EFFECT 0: SHIPPED vs CORRECT-POSITIONAL (dv12 or dv24)."""
    shipped_col = f"{dv_col}_shipped"
    correct_col = f"{dv_col}_pos_correct"

    report = {}

    # Transition counts
    both_defined = df[shipped_col].notna() & df[correct_col].notna()
    ship_only = df[shipped_col].notna() & df[correct_col].isna()
    correct_only = df[shipped_col].isna() & df[correct_col].notna()

    report["both_defined"] = int(both_defined.sum())
    report["ship_defined_only"] = int(ship_only.sum())
    report["correct_defined_only"] = int(correct_only.sum())

    if dv_col == "dv24":
        # ri_label transitions
        report["ri_label_transition_matrix"] = {
            "ship_0_to_pos_0": int(((df["ri_label_shipped"] == 0) & (df["ri_label_pos_correct"] == 0)).sum()),
            "ship_0_to_pos_1": int(((df["ri_label_shipped"] == 0) & (df["ri_label_pos_correct"] == 1)).sum()),
            "ship_0_to_pos_undef": int(((df["ri_label_shipped"] == 0) & (df["ri_label_pos_correct"].isna())).sum()),
            "ship_1_to_pos_0": int(((df["ri_label_shipped"] == 1) & (df["ri_label_pos_correct"] == 0)).sum()),
            "ship_1_to_pos_1": int(((df["ri_label_shipped"] == 1) & (df["ri_label_pos_correct"] == 1)).sum()),
            "ship_1_to_pos_undef": int(((df["ri_label_shipped"] == 1) & (df["ri_label_pos_correct"].isna())).sum()),
        }

        # Phantom positives (ship=1 → UNDEF, which are 1 → UNDEF transitions)
        report["phantom_positives_ship1_to_undef"] = report["ri_label_transition_matrix"]["ship_1_to_pos_undef"]

        positives_shipped = int((df["ri_label_shipped"] == 1).sum())
        positives_correct = int((df["ri_label_pos_correct"] == 1).sum())
        report["positives_shipped"] = positives_shipped
        report["positives_pos_correct"] = positives_correct
        report["positives_difference"] = positives_correct - positives_shipped

    return report


def effect_1_pos_vs_temporal_with_partner(df, dv_col):
    """EFFECT 1: CORRECT-POSITIONAL vs TEMPORAL (only rows WITH temporal partner)."""
    correct_col = f"{dv_col}_pos_correct"
    tmp_col = f"{dv_col}_tmp"

    report = {}

    # Restrict to rows with temporal partner
    has_tmp = df[tmp_col].notna()
    df_subset = df[has_tmp].copy()

    if len(df_subset) == 0:
        if dv_col == "dv24":
            report["ri_label_transition_matrix"] = {
                "pos_0_to_tmp_0": 0, "pos_0_to_tmp_1": 0,
                "pos_1_to_tmp_0": 0, "pos_1_to_tmp_1": 0,
            }
        report["dv_changes"] = {
            "rows_with_change": 0,
            "min": None, "median": None, "p95": None, "max": None,
        }
        return report

    if dv_col == "dv24":
        report["ri_label_transition_matrix"] = {
            "pos_0_to_tmp_0": int(((df_subset["ri_label_pos_correct"] == 0) & (df_subset["ri_label_tmp"] == 0)).sum()),
            "pos_0_to_tmp_1": int(((df_subset["ri_label_pos_correct"] == 0) & (df_subset["ri_label_tmp"] == 1)).sum()),
            "pos_1_to_tmp_0": int(((df_subset["ri_label_pos_correct"] == 1) & (df_subset["ri_label_tmp"] == 0)).sum()),
            "pos_1_to_tmp_1": int(((df_subset["ri_label_pos_correct"] == 1) & (df_subset["ri_label_tmp"] == 1)).sum()),
        }

    # dv changes: ONLY where |Δ| > 0
    both_defined = df_subset[correct_col].notna() & df_subset[tmp_col].notna()
    df_changes = df_subset[both_defined].copy()
    df_changes["delta"] = (df_changes[correct_col] - df_changes[tmp_col]).abs()
    rows_with_change = (df_changes["delta"] > 0).sum()

    if rows_with_change > 0:
        changed = df_changes[df_changes["delta"] > 0]
        report["dv_changes"] = {
            "rows_with_change": int(rows_with_change),
            "min": float(changed["delta"].min()),
            "median": float(changed["delta"].median()),
            "p95": float(changed["delta"].quantile(0.95)),
            "max": float(changed["delta"].max()),
        }
    else:
        report["dv_changes"] = {
            "rows_with_change": 0,
            "min": None, "median": None, "p95": None, "max": None,
        }

    return report


def effect_2_no_temporal_partner(df, dv_col):
    """EFFECT 2: No temporal partner (border defect)."""
    correct_col = f"{dv_col}_pos_correct"
    tmp_col = f"{dv_col}_tmp"

    report = {}

    no_tmp = df[tmp_col].isna()
    df_subset = df[no_tmp]

    # Counts: correct=0, correct=1, correct=UNDEF
    report["correct_0_no_tmp"] = int(((df_subset[correct_col] == 0)).sum())
    report["correct_1_no_tmp"] = int(((df_subset[correct_col] == 1)).sum())
    report["correct_undef_no_tmp"] = int(((df_subset[correct_col].isna())).sum())

    return report


# ==============================================================================
# CONCENTRATION ANALYSIS (FOR EFFECT 1 CHANGED ROWS)
# ==============================================================================

def analyze_concentration_effect1(df, dv_col):
    """Concentration of Effect 1: rows where |Δdv| > 0."""
    correct_col = f"{dv_col}_pos_correct"
    tmp_col = f"{dv_col}_tmp"

    # Only rows with temporal partner AND |Δ| > 0
    has_tmp = df[tmp_col].notna()
    df_subset = df[has_tmp].copy()

    both_defined = df_subset[correct_col].notna() & df_subset[tmp_col].notna()
    df_subset = df_subset[both_defined].copy()

    df_subset["delta"] = (df_subset[correct_col] - df_subset[tmp_col]).abs()
    df_affected = df_subset[df_subset["delta"] > 0].copy()

    if len(df_affected) == 0:
        return {
            "dv_col": dv_col,
            "total_affected": 0,
            "by_basin": {},
            "by_decade": {},
        }

    df_affected["year"] = df_affected["timestamp"].dt.year
    df_affected["decade"] = (df_affected["year"] // 10) * 10

    return {
        "dv_col": dv_col,
        "total_affected": len(df_affected),
        "by_basin": convert_to_python_types(df_affected["basin"].value_counts().to_dict()),
        "by_decade": convert_to_python_types(df_affected["decade"].value_counts().sort_index().to_dict()),
    }


# ==============================================================================
# SIDS LOSING ALL POSITIVES (SCOPED)
# ==============================================================================

def find_sids_losing_all_positives_valid(valid_merged, splits):
    """Find SIDs losing all positives in valid set, per split."""
    report = {
        "total": 0,
        "train": {"count": 0, "sids": []},
        "val": {"count": 0, "sids": []},
        "test": {"count": 0},
    }

    for sid in valid_merged["sid"].unique():
        sid_df = valid_merged[valid_merged["sid"] == sid]
        shipped_pos = (sid_df["ri_label_shipped"] == 1).sum()
        correct_pos = (sid_df["ri_label_pos_correct"] == 1).sum()

        if shipped_pos > 0 and correct_pos == 0:
            report["total"] += 1

            # Determine split(s) of this SID
            splits_in_sid = sid_df[sid_df["event_id"].notna()]["split"].unique()

            if len(splits_in_sid) == 1:
                split = splits_in_sid[0]
                if split == "train":
                    report["train"]["count"] += 1
                    report["train"]["sids"].append(sid)
                elif split == "val":
                    report["val"]["count"] += 1
                    report["val"]["sids"].append(sid)
                elif split == "test":
                    report["test"]["count"] += 1
            else:
                # Multi-split: assign to first non-test
                if "train" in splits_in_sid:
                    report["train"]["count"] += 1
                    if sid not in report["train"]["sids"]:
                        report["train"]["sids"].append(sid)
                elif "val" in splits_in_sid:
                    report["val"]["count"] += 1
                    if sid not in report["val"]["sids"]:
                        report["val"]["sids"].append(sid)
                else:
                    report["test"]["count"] += 1

    return report


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("[1/9] Loading data and computing labels...", file=sys.stderr)
    df = load_and_compute_correct_positional(DATA_DIR / "event_list_augmented.csv")
    valid_events = pd.read_csv(DATA_DIR / "normalized" / "valid_events.csv", **CSV_PARAMS)
    splits = pd.read_csv(DATA_DIR / "normalized" / "splits.csv", **CSV_PARAMS)

    print(f"  - Loaded: {len(df)} event list, {len(valid_events)} valid events", file=sys.stderr)

    print("[2/9] Computing temporal labels...", file=sys.stderr)
    df_original = df.copy()  # Save before sorting in compute_temporal_labels
    df = compute_temporal_labels(df)

    print("[3/9] Diagnosing DEFECT 0 (trailing-row bleed only)...", file=sys.stderr)
    defect_0_report = diagnose_defect_0_trailing_only(df_original)

    print("[4/9] Analyzing irregular grid...", file=sys.stderr)
    irregular_grid_report = analyze_irregular_grid(df, splits)

    print("[5/9] Deriving PL-gated subset...", file=sys.stderr)
    pl_gated_count, pl_gated_event_ids = derive_pl_gated_subset(valid_events, splits)
    print(f"  - Derived PL-gated: {pl_gated_count} (expected ~14,101)", file=sys.stderr)

    # Load PL gate census for cross-check
    census_path = REPO_ROOT / "outputs" / "results" / "pl_gate_census.json"
    pl_gate_expected = 14101
    if census_path.exists():
        with open(census_path) as f:
            census = json.load(f)
            pl_gate_expected = census.get("dev_total_events", 14101)

    print("[6/9] Preparing scopes...", file=sys.stderr)

    # Valid set
    def parse_event_id(event_id):
        parts = event_id.split("_")
        year, month, day, hhmm = parts[1], parts[2], parts[3], parts[4]
        ts_str = f"{year}-{month}-{day} {hhmm[:2]}:{hhmm[2:]}:00"
        timestamp = pd.to_datetime(ts_str)
        sid = "_".join(parts[5:])
        return timestamp, sid

    valid_events[["timestamp_parsed", "sid_parsed"]] = valid_events["event_id"].apply(
        lambda x: pd.Series(parse_event_id(x))
    )
    valid_events = valid_events.drop(columns=["sid", "timestamp"], errors="ignore")

    valid_merged = valid_events.merge(
        df[["sid", "timestamp", "dv12_shipped", "dv24_shipped", "ri_label_shipped",
            "dv12_pos_correct", "dv24_pos_correct", "ri_label_pos_correct",
            "dv12_tmp", "dv24_tmp", "ri_label_tmp", "basin"]],
        left_on=["sid_parsed", "timestamp_parsed"],
        right_on=["sid", "timestamp"],
        how="left",
    )
    valid_merged = valid_merged.merge(splits[["event_id", "split"]], on="event_id", how="left")

    join_coverage = valid_merged["ri_label_shipped"].notna().sum()
    print(f"  - Valid set join: {join_coverage} / {len(valid_events)}", file=sys.stderr)

    # PL-gated subset (from valid_merged, train+val only)
    pl_gated_merged = valid_merged[
        (valid_merged["event_id"].isin(pl_gated_event_ids)) &
        (valid_merged["split"].isin(["train", "val"]))
    ].copy()
    print(f"  - PL-gated subset (dev): {len(pl_gated_merged)} events", file=sys.stderr)

    print("[7/9] Three-effect decomposition...", file=sys.stderr)

    full_report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "three_effect_decomposition": True,
            "defect_0_diagnosis": True,
            "temporal_rule": "Exact-match t0+12h/t0+24h within same SID",
            "nan_semantics": "UNDEFINED (pd.NA) for no partner or missing wind",
            "read_only": True,
        },
        "defect_0_diagnosis": defect_0_report,
        "irregular_grid": irregular_grid_report,
        "pl_gate": {
            "derived_count": int(pl_gated_count),
            "expected_count": int(pl_gate_expected),
            "match": pl_gated_count == pl_gate_expected,
        },
        "effects_by_scope": {},
    }

    # Scopes: full_list, valid_set (with per-split), pl_gated, test
    scopes = {
        "full_list": {"df": df, "label": "Full event list (32,989)", "has_splits": False},
        "valid_set": {"df": valid_merged, "label": "Valid dataset (16,780)", "has_splits": True},
        "dev_pl_gated": {"df": pl_gated_merged, "label": "Dev PL-gated (train+val)", "has_splits": True},
    }

    for scope_key, scope_data in scopes.items():
        df_scope = scope_data["df"]
        label = scope_data["label"]
        has_splits = scope_data["has_splits"]

        print(f"  - Computing effects for {label}...", file=sys.stderr)

        scope_report = {
            "label": label,
            "row_count": len(df_scope),
        }

        # DV12 and DV24
        for dv_col in ["dv12", "dv24"]:
            dv_report = {
                "effect_0": effect_0_shipped_vs_correct(df_scope, dv_col),
                "effect_1": effect_1_pos_vs_temporal_with_partner(df_scope, dv_col),
                "effect_2": effect_2_no_temporal_partner(df_scope, dv_col),
            }

            # Combined shipped → temporal
            correct_col = f"{dv_col}_pos_correct"
            tmp_col = f"{dv_col}_tmp"

            dv_report["combined_shipped_vs_temporal"] = {
                "shipped_defined": int((df_scope[f"{dv_col}_shipped"].notna()).sum()),
                "temporal_defined": int((df_scope[tmp_col].notna()).sum()),
            }

            if dv_col == "dv24":
                dv_report["combined_shipped_vs_temporal"]["ri_label_transition_matrix"] = {
                    "ship_0_to_tmp_0": int(((df_scope["ri_label_shipped"] == 0) & (df_scope["ri_label_tmp"] == 0)).sum()),
                    "ship_0_to_tmp_1": int(((df_scope["ri_label_shipped"] == 0) & (df_scope["ri_label_tmp"] == 1)).sum()),
                    "ship_0_to_tmp_undef": int(((df_scope["ri_label_shipped"] == 0) & (df_scope["ri_label_tmp"].isna())).sum()),
                    "ship_1_to_tmp_0": int(((df_scope["ri_label_shipped"] == 1) & (df_scope["ri_label_tmp"] == 0)).sum()),
                    "ship_1_to_tmp_1": int(((df_scope["ri_label_shipped"] == 1) & (df_scope["ri_label_tmp"] == 1)).sum()),
                    "ship_1_to_tmp_undef": int(((df_scope["ri_label_shipped"] == 1) & (df_scope["ri_label_tmp"].isna())).sum()),
                }
                dv_report["combined_shipped_vs_temporal"]["positives_shipped"] = int((df_scope["ri_label_shipped"] == 1).sum())
                dv_report["combined_shipped_vs_temporal"]["positives_temporal"] = int((df_scope["ri_label_tmp"] == 1).sum())

            scope_report[dv_col] = dv_report

            # Per-split breakdown (if has_splits)
            if has_splits and "split" in df_scope.columns:
                scope_report[f"{dv_col}_by_split"] = {}

                for split_name in ["train", "val", "test"]:
                    split_df = df_scope[df_scope["split"] == split_name]

                    if len(split_df) == 0:
                        continue

                    split_report = {
                        "effect_0": effect_0_shipped_vs_correct(split_df, dv_col),
                        "effect_1": effect_1_pos_vs_temporal_with_partner(split_df, dv_col),
                        "effect_2": effect_2_no_temporal_partner(split_df, dv_col),
                    }

                    if dv_col == "dv24":
                        split_report["combined_shipped_vs_temporal"] = {
                            "ri_label_transition_matrix": {
                                "ship_0_to_tmp_0": int(((split_df["ri_label_shipped"] == 0) & (split_df["ri_label_tmp"] == 0)).sum()),
                                "ship_0_to_tmp_1": int(((split_df["ri_label_shipped"] == 0) & (split_df["ri_label_tmp"] == 1)).sum()),
                                "ship_0_to_tmp_undef": int(((split_df["ri_label_shipped"] == 0) & (split_df["ri_label_tmp"].isna())).sum()),
                                "ship_1_to_tmp_0": int(((split_df["ri_label_shipped"] == 1) & (split_df["ri_label_tmp"] == 0)).sum()),
                                "ship_1_to_tmp_1": int(((split_df["ri_label_shipped"] == 1) & (split_df["ri_label_tmp"] == 1)).sum()),
                                "ship_1_to_tmp_undef": int(((split_df["ri_label_shipped"] == 1) & (split_df["ri_label_tmp"].isna())).sum()),
                            },
                            "positives_shipped": int((split_df["ri_label_shipped"] == 1).sum()),
                            "positives_temporal": int((split_df["ri_label_tmp"] == 1).sum()),
                        }

                    scope_report[f"{dv_col}_by_split"][split_name] = split_report

        full_report["effects_by_scope"][scope_key] = scope_report

    print("[8/9] Concentration and SID analysis...", file=sys.stderr)

    full_report["concentration"] = {}
    full_report["concentration"]["effect_1_dv24_changed"] = analyze_concentration_effect1(df, "dv24")
    full_report["concentration"]["effect_1_dv12_changed"] = analyze_concentration_effect1(df, "dv12")

    # SIDs losing all positives (valid set only)
    full_report["sids_losing_all_positives_valid"] = find_sids_losing_all_positives_valid(valid_merged, splits)

    print("[9/9] Writing outputs...", file=sys.stderr)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    utc_now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = OUTPUTS_DIR / f"report_v4_{utc_now}.json"
    with open(json_path, "w") as f:
        json.dump(convert_to_python_types(full_report), f, indent=2)

    md_report = generate_md_report(full_report)
    md_path = OUTPUTS_DIR / f"report_v4_{utc_now}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    try:
        print("\n" + md_report)
    except UnicodeEncodeError:
        print(md_report.encode('ascii', errors='replace').decode('ascii'))

    print(f"\nOutputs written to:", file=sys.stderr)
    print(f"  JSON: {json_path.relative_to(REPO_ROOT)}", file=sys.stderr)
    print(f"  MD:   {md_path.relative_to(REPO_ROOT)}", file=sys.stderr)


def generate_md_report(full_report):
    """Generate human-readable MD report."""
    md = []
    md.append("# DV24 Impact Assessment v3: Complete Three-Effect Decomposition\n\n")
    md.append(f"**Generated:** {full_report['generated_utc']}\n")
    md.append("**Method:** SHIPPED -> CORRECT-POSITIONAL -> TEMPORAL; three separable effects\n\n")

    # DEFECT 0
    d0 = full_report["defect_0_diagnosis"]
    md.append("## DEFECT 0: Cross-SID Bleed\n\n")
    md.append("### Evidence (Trailing Rows Only)\n")
    md.append(f"- dv24 shipped/correct mismatches: {d0['shipped_vs_correct_pos_full']['dv24_mismatches']}\n")
    md.append(f"- Mismatches in trailing rows (<=4 from end): {d0['trailing_row_analysis']['dv24_trailing_mismatches']}\n")
    md.append(f"- Interior rows exact match: {d0['trailing_row_analysis']['dv24_interior_exact_matches']}\n")
    md.append(f"- Trailing values matching global shift (sorted row order): {d0['trailing_row_analysis']['dv24_trailing_global_shift_matches_sorted_order']}\n")
    md.append(f"- Trailing values matching global shift (original file order): {d0['trailing_row_analysis']['dv24_trailing_global_shift_matches_original_order']}\n")
    md.append(f"- SIDs losing ALL positives (full list): {d0['sids_losing_all_positives']['full_list_count']}\n")
    md.append("\n")

    # Irregular grid
    ig = full_report["irregular_grid"]
    md.append("## Irregular Grid Census\n\n")
    md.append(f"- Total non-6h deltas: {ig['total_non_6h_deltas']}\n")
    md.append(f"- Affected SIDs by category:\n")
    md.append(f"  - Train: {ig['affected_sids_by_category']['train']['count']}\n")
    md.append(f"  - Val: {ig['affected_sids_by_category']['val']['count']}\n")
    md.append(f"  - Test: {ig['affected_sids_by_category']['test']['count']}\n")
    md.append(f"  - Unsplit/Rejected: {ig['affected_sids_by_category']['unsplit_rejected']['count']}\n")

    if ig['affected_sids_by_category']['train']['sids']:
        md.append(f"- Train SIDs: {', '.join(ig['affected_sids_by_category']['train']['sids'][:5])}")
        if len(ig['affected_sids_by_category']['train']['sids']) > 5:
            md.append(f" ... +{len(ig['affected_sids_by_category']['train']['sids']) - 5} more\n")
        else:
            md.append("\n")

    if ig['affected_sids_by_category']['val']['sids']:
        md.append(f"- Val SIDs: {', '.join(ig['affected_sids_by_category']['val']['sids'])}\n")

    if ig['affected_sids_by_category']['unsplit_rejected']['sids']:
        md.append(f"- Unsplit/Rejected SIDs: {', '.join(ig['affected_sids_by_category']['unsplit_rejected']['sids'][:5])}")
        if len(ig['affected_sids_by_category']['unsplit_rejected']['sids']) > 5:
            md.append(f" ... +{len(ig['affected_sids_by_category']['unsplit_rejected']['sids']) - 5} more\n")
        else:
            md.append("\n")

    md.append(f"- By year: {dict(sorted(ig['deltas_by_years'].items()))}\n")
    md.append(f"- By basin: {dict(ig['deltas_by_basin'])}\n\n")

    # PL-gate
    pg = full_report["pl_gate"]
    md.append("## PL-Gate Derivation (Dev Set)\n\n")
    md.append(f"- Derived PL-gated count: {pg['derived_count']}\n")
    md.append(f"- Expected count (census): {pg['expected_count']}\n")
    md.append(f"- Match: {pg['match']}\n\n")

    # Effects by scope
    for scope_key, scope in full_report["effects_by_scope"].items():
        md.append(f"## {scope['label']}\n\n")
        md.append(f"**Row count: {scope['row_count']}**\n\n")

        # DV24
        if "dv24" in scope:
            md.append("### DV24 Effects\n\n")

            # EFFECT 0
            e0 = scope["dv24"]["effect_0"]
            md.append("**EFFECT 0: Shipped vs Correct-Positional**\n")
            if "ri_label_transition_matrix" in e0:
                tm = e0["ri_label_transition_matrix"]
                md.append(f"RI Label:\n")
                md.append(f"| From | To=0 | To=1 | To=UNDEF |\n")
                md.append(f"|------|------|------|----------|\n")
                md.append(f"| ship=0 | {tm['ship_0_to_pos_0']} | {tm['ship_0_to_pos_1']} | {tm['ship_0_to_pos_undef']} |\n")
                md.append(f"| ship=1 | {tm['ship_1_to_pos_0']} | {tm['ship_1_to_pos_1']} | {tm['ship_1_to_pos_undef']} |\n")
                md.append(f"- Phantom positives (ship=1->UNDEF): {e0.get('phantom_positives_ship1_to_undef', 0)}\n")
            md.append(f"- Positives shipped: {e0.get('positives_shipped', 'N/A')} -> correct: {e0.get('positives_pos_correct', 'N/A')}\n\n")

            # EFFECT 1
            e1 = scope["dv24"]["effect_1"]
            md.append("**EFFECT 1: Correct-Positional vs Temporal (with partner)**\n")
            if "ri_label_transition_matrix" in e1:
                tm = e1["ri_label_transition_matrix"]
                md.append(f"| From | To=0 | To=1 |\n")
                md.append(f"|------|------|------|\n")
                md.append(f"| pos=0 | {tm['pos_0_to_tmp_0']} | {tm['pos_0_to_tmp_1']} |\n")
                md.append(f"| pos=1 | {tm['pos_1_to_tmp_0']} | {tm['pos_1_to_tmp_1']} |\n")
            dv24_chg = e1["dv_changes"]
            md.append(f"dv24 changes (|Delta|>0 only): {dv24_chg['rows_with_change']} rows")
            if dv24_chg['rows_with_change'] > 0:
                md.append(f", median={dv24_chg['median']:.1f}kt, p95={dv24_chg['p95']:.1f}kt\n\n")
            else:
                md.append("\n\n")

            # EFFECT 2
            e2 = scope["dv24"]["effect_2"]
            md.append("**EFFECT 2: No Temporal Partner**\n")
            md.append(f"- Correct=0 no tmp: {e2['correct_0_no_tmp']}\n")
            md.append(f"- Correct=1 no tmp: {e2['correct_1_no_tmp']}\n")
            md.append(f"- Correct=UNDEF no tmp: {e2['correct_undef_no_tmp']}\n\n")

            # COMBINED
            comb = scope["dv24"]["combined_shipped_vs_temporal"]
            md.append("**COMBINED: Shipped vs Temporal**\n")
            if "ri_label_transition_matrix" in comb:
                tm = comb["ri_label_transition_matrix"]
                md.append(f"| From | To=0 | To=1 | To=UNDEF |\n")
                md.append(f"|------|------|------|----------|\n")
                md.append(f"| ship=0 | {tm['ship_0_to_tmp_0']} | {tm['ship_0_to_tmp_1']} | {tm['ship_0_to_tmp_undef']} |\n")
                md.append(f"| ship=1 | {tm['ship_1_to_tmp_0']} | {tm['ship_1_to_tmp_1']} | {tm['ship_1_to_tmp_undef']} |\n")
                md.append(f"- Positives: {comb['positives_shipped']} -> {comb['positives_temporal']}\n\n")

        # Per-split breakdown
        if f"dv24_by_split" in scope:
            md.append("### DV24 By Split\n\n")
            for split_name in ["train", "val", "test"]:
                if split_name not in scope["dv24_by_split"]:
                    continue
                split_data = scope["dv24_by_split"][split_name]
                md.append(f"**{split_name.upper()}**\n")

                e0_split = split_data["effect_0"]
                if "ri_label_transition_matrix" in e0_split:
                    tm = e0_split["ri_label_transition_matrix"]
                    pos_shipped = e0_split.get('positives_shipped', 'N/A')
                    pos_correct = e0_split.get('positives_pos_correct', 'N/A')
                    md.append(f"EFFECT 0: ship=0->undef:{tm['ship_0_to_pos_undef']} | ship=1->undef:{tm['ship_1_to_pos_undef']} | positives:{pos_shipped}->{pos_correct}\n")

                comb_split = split_data.get("combined_shipped_vs_temporal", {})
                if "ri_label_transition_matrix" in comb_split:
                    md.append(f"COMBINED: ship={comb_split['positives_shipped']}->{comb_split['positives_temporal']}\n")

                md.append("\n")

    # Concentration
    if "concentration" in full_report:
        md.append("## Concentration: Effect 1 (|Delta| > 0 rows)\n\n")
        for eff in ["effect_1_dv24_changed", "effect_1_dv12_changed"]:
            if eff in full_report["concentration"]:
                conc = full_report["concentration"][eff]
                if conc["total_affected"] > 0:
                    md.append(f"**{conc['dv_col'].upper()}:** {conc['total_affected']} rows, basins={conc['by_basin']}, decades={conc['by_decade']}\n")

        md.append("\n")

    # SIDs losing positives
    if "sids_losing_all_positives_valid" in full_report:
        slp = full_report["sids_losing_all_positives_valid"]
        md.append("## SIDs Losing All Positives (Valid Set)\n\n")
        md.append(f"- Total: {slp['total']}\n")
        md.append(f"- Train: {slp['train']['count']} ({', '.join(slp['train']['sids'][:5])}{'...' if len(slp['train']['sids']) > 5 else ''})\n")
        md.append(f"- Val: {slp['val']['count']} ({', '.join(slp['val']['sids'][:5])}{'...' if len(slp['val']['sids']) > 5 else ''})\n")
        md.append(f"- Test: {slp['test']['count']} (count only)\n")

    return "".join(md)


if __name__ == "__main__":
    main()
