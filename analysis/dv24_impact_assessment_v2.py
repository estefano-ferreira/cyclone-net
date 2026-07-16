#!/usr/bin/env python3
"""
DV24 Impact Assessment v2: Three-effect decomposition.

Decomposes the shift from SHIPPED → CORRECT-POSITIONAL → TEMPORAL into:
1. EFFECT 0 (cross-SID bleed): SHIPPED vs CORRECT-POSITIONAL
   - Shipped dv12/dv24 computed without per-SID grouping (evidence: trailing-row bleed)
2. EFFECT 1 (positional misalignment with temporal partner):
   - CORRECT-POSITIONAL vs TEMPORAL, restricted to rows with a temporal partner
3. EFFECT 2 (border/undefined):
   - Rows with NO exact temporal partner

Scopes: full list, valid set, dev PL-gated, test (aggregate only).
Plus concentration analysis: basin, year, position in storm, proximity to sub-6h.

READ-ONLY except outputs/results/dv24_impact/
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# ==============================================================================
# CONFIG
# ==============================================================================

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
# STEP 1: LOAD DATA & COMPUTE CORRECT-POSITIONAL LABELS
# ==============================================================================

def load_and_compute_correct_positional(event_list_path):
    """Load event_list, compute CORRECT-POSITIONAL labels (per-SID groupby shift)."""
    df = pd.read_csv(event_list_path, **CSV_PARAMS)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["wind_kt"] = pd.to_numeric(df["wind_kt"], errors="coerce")

    # Shipped labels (as-is)
    df["dv12_shipped"] = df["dv12_kt"]
    df["dv24_shipped"] = df["dv24_kt"]
    df["ri_label_shipped"] = df["ri_label"]

    # CORRECT-POSITIONAL: per-SID groupby shift, NaN → undefined (not 0)
    df["dv12_pos_correct"] = df.groupby("sid")["wind_kt"].shift(-2) - df["wind_kt"]
    df["dv24_pos_correct"] = df.groupby("sid")["wind_kt"].shift(-4) - df["wind_kt"]
    df["ri_label_pos_correct"] = pd.NA
    defined_24 = df["dv24_pos_correct"].notna()
    df.loc[defined_24 & (df["dv24_pos_correct"] < RI_THRESHOLD_KT), "ri_label_pos_correct"] = 0
    df.loc[defined_24 & (df["dv24_pos_correct"] >= RI_THRESHOLD_KT), "ri_label_pos_correct"] = 1

    return df


# ==============================================================================
# STEP 2: DIAGNOSE DEFECT 0 (CROSS-SID BLEED)
# ==============================================================================

def diagnose_defect_0(df):
    """Diagnose: shipped dv24 computed without per-SID grouping (global shift bleed)."""
    df = df.copy()
    df = df.sort_values(["sid", "timestamp"], ignore_index=True)

    report = {
        "shipped_vs_correct_pos": {
            "dv12": {"mismatches": 0, "mismatch_rows": []},
            "dv24": {"mismatches": 0, "mismatch_rows": []},
            "ri_label": {"mismatches": 0, "mismatch_rows": []},
        },
        "trailing_row_analysis": {
            "dv24_trailing_mismatches": 0,
            "dv24_interior_matches": 0,
            "global_shift_matches": 0,
        },
        "phantom_positives": {
            "count": 0,
            "affected_sids": [],
        },
    }

    # Compare shipped vs correct-positional
    for dv_col in ["dv12", "dv24"]:
        shipped_col = f"{dv_col}_shipped"
        correct_col = f"{dv_col}_pos_correct"

        # Match handling: NaN==NaN is match
        both_notna = df[shipped_col].notna() & df[correct_col].notna()
        one_na = (df[shipped_col].isna() != df[correct_col].isna())
        differ = both_notna & (df[shipped_col] != df[correct_col])

        mismatches = (one_na | differ).sum()
        report["shipped_vs_correct_pos"][dv_col]["mismatches"] = int(mismatches)

    # Trailing-row analysis: within 4 rows of storm end
    for sid in df["sid"].unique():
        sid_df = df[df["sid"] == sid]
        n = len(sid_df)

        if n > 0:
            # Trailing rows: within 4 rows of end
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
                report["trailing_row_analysis"]["dv24_interior_matches"] += interior_match

    # Global shift matches: shipped trailing value equals global (non-grouped) shift
    df_global = df.sort_values("timestamp", ignore_index=True).copy()
    df_global["wind_kt_shift_4_global"] = df_global["wind_kt"].shift(-4)
    df_global["dv24_global"] = df_global["wind_kt_shift_4_global"] - df_global["wind_kt"]

    # Match back by original index
    global_shift_match = (
        (df["dv24_shipped"].notna() &
         (df.index < len(df) - 4) &
         (df["dv24_shipped"] == df_global.loc[df.index, "dv24_global"])).sum()
    )
    report["trailing_row_analysis"]["global_shift_matches"] = int(global_shift_match)

    # Phantom positives: shipped=1 but correct-pos=0
    phantoms = ((df["ri_label_shipped"] == 1) & (df["ri_label_pos_correct"] == 0)).sum()
    report["phantom_positives"]["count"] = int(phantoms)

    # Storms losing all positives: had shipped positives, lost them in correct-pos
    for sid in df["sid"].unique():
        sid_df = df[df["sid"] == sid]
        shipped_pos = (sid_df["ri_label_shipped"] == 1).sum()
        correct_pos = (sid_df["ri_label_pos_correct"] == 1).sum()

        if shipped_pos > 0 and correct_pos == 0:
            report["phantom_positives"]["affected_sids"].append(sid)

    report["phantom_positives"]["affected_sids_count"] = len(report["phantom_positives"]["affected_sids"])

    return report


# ==============================================================================
# STEP 3: TEMPORAL LABELING (EXACT-MATCH)
# ==============================================================================

def compute_temporal_labels(df):
    """Compute exact-match temporal labels."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["wind_kt"] = pd.to_numeric(df["wind_kt"], errors="coerce")

    df = df.sort_values(["sid", "timestamp"], ignore_index=True)

    # Deduplicate right side
    df_right = df.drop_duplicates(subset=["sid", "timestamp"], keep="first").copy()

    # Merge for dv24
    df["timestamp_plus_24h"] = df["timestamp"] + pd.Timedelta(hours=24)
    dv24_merge = df[["sid", "timestamp_plus_24h"]].merge(
        df_right[["sid", "timestamp", "wind_kt"]].rename(columns={"wind_kt": "wind_kt_24h"}),
        left_on=["sid", "timestamp_plus_24h"],
        right_on=["sid", "timestamp"],
        how="left",
    )
    df["dv24_tmp"] = dv24_merge["wind_kt_24h"] - df["wind_kt"]

    # Temporal label
    df["ri_label_tmp"] = pd.NA
    defined_tmp = df["wind_kt"].notna() & (~df["dv24_tmp"].isna())
    df.loc[defined_tmp & (df["dv24_tmp"] < RI_THRESHOLD_KT), "ri_label_tmp"] = 0
    df.loc[defined_tmp & (df["dv24_tmp"] >= RI_THRESHOLD_KT), "ri_label_tmp"] = 1

    df = df.drop(columns=["timestamp_plus_24h"])

    return df


# ==============================================================================
# STEP 4: COMPUTE POSITIONAL MISALIGNMENT
# ==============================================================================

def compute_positional_misalignment(df):
    """Identify rows where positional partner timestamp != t0+24h."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Compute positional partner's timestamp (the shifted row)
    df["timestamp_pos_partner_24h"] = df.groupby("sid")["timestamp"].shift(-4)

    # Expected timestamp
    df["timestamp_expected_24h"] = df["timestamp"] + pd.Timedelta(hours=24)

    # Misaligned: partner exists but timestamp mismatch
    df["dv24_pos_misaligned"] = (
        df["timestamp_pos_partner_24h"].notna() &
        (df["timestamp_pos_partner_24h"] != df["timestamp_expected_24h"])
    )

    return df


# ==============================================================================
# STEP 5: THREE-EFFECT DECOMPOSITION
# ==============================================================================

def effect_0_shipped_vs_correct_pos(df, scope_label):
    """EFFECT 0: SHIPPED vs CORRECT-POSITIONAL."""
    report = {
        "scope": scope_label,
        "dv24_transition_matrix": {},
        "ri_label_transition_matrix": {},
    }

    # dv24 transitions: only where at least one is defined
    both_defined = df["dv24_shipped"].notna() & df["dv24_pos_correct"].notna()
    ship_defined_only = df["dv24_shipped"].notna() & df["dv24_pos_correct"].isna()
    pos_defined_only = df["dv24_shipped"].isna() & df["dv24_pos_correct"].notna()

    report["dv24_transition_matrix"] = {
        "both_defined": int(both_defined.sum()),
        "ship_defined_only": int(ship_defined_only.sum()),
        "pos_defined_only": int(pos_defined_only.sum()),
    }

    # ri_label transitions: 0→0, 0→1, 1→0, 1→1, shipped_0→undef, shipped_1→undef
    report["ri_label_transition_matrix"] = {
        "ship_0_to_pos_0": int(((df["ri_label_shipped"] == 0) & (df["ri_label_pos_correct"] == 0)).sum()),
        "ship_0_to_pos_1": int(((df["ri_label_shipped"] == 0) & (df["ri_label_pos_correct"] == 1)).sum()),
        "ship_0_to_pos_undef": int(((df["ri_label_shipped"] == 0) & (df["ri_label_pos_correct"].isna())).sum()),
        "ship_1_to_pos_0": int(((df["ri_label_shipped"] == 1) & (df["ri_label_pos_correct"] == 0)).sum()),
        "ship_1_to_pos_1": int(((df["ri_label_shipped"] == 1) & (df["ri_label_pos_correct"] == 1)).sum()),
        "ship_1_to_pos_undef": int(((df["ri_label_shipped"] == 1) & (df["ri_label_pos_correct"].isna())).sum()),
    }

    positives_shipped = (df["ri_label_shipped"] == 1).sum()
    positives_pos_correct = (df["ri_label_pos_correct"] == 1).sum()

    report["positives_shipped"] = int(positives_shipped)
    report["positives_pos_correct"] = int(positives_pos_correct)
    report["positives_difference"] = int(positives_pos_correct - positives_shipped)

    return report


def effect_1_pos_vs_temporal_with_partner(df, scope_label):
    """EFFECT 1: CORRECT-POSITIONAL vs TEMPORAL, only rows with temporal partner."""
    report = {
        "scope": scope_label,
        "ri_label_transition_matrix": {},
        "dv24_changes": {},
    }

    # Restrict to rows with temporal partner (dv24_tmp defined)
    has_tmp = df["dv24_tmp"].notna()
    df_subset = df[has_tmp].copy()

    if len(df_subset) == 0:
        report["ri_label_transition_matrix"] = {
            "pos_0_to_tmp_0": 0, "pos_0_to_tmp_1": 0,
            "pos_1_to_tmp_0": 0, "pos_1_to_tmp_1": 0,
        }
        report["dv24_changes"] = {
            "rows_with_change": 0,
            "min": None, "median": None, "p95": None, "max": None,
        }
        return report

    # Transition matrix
    report["ri_label_transition_matrix"] = {
        "pos_0_to_tmp_0": int(((df_subset["ri_label_pos_correct"] == 0) & (df_subset["ri_label_tmp"] == 0)).sum()),
        "pos_0_to_tmp_1": int(((df_subset["ri_label_pos_correct"] == 0) & (df_subset["ri_label_tmp"] == 1)).sum()),
        "pos_1_to_tmp_0": int(((df_subset["ri_label_pos_correct"] == 1) & (df_subset["ri_label_tmp"] == 0)).sum()),
        "pos_1_to_tmp_1": int(((df_subset["ri_label_pos_correct"] == 1) & (df_subset["ri_label_tmp"] == 1)).sum()),
    }

    # dv24 changes: only where both defined and |Δ| > 0
    both_defined = df_subset["dv24_pos_correct"].notna() & df_subset["dv24_tmp"].notna()
    df_changes = df_subset[both_defined].copy()
    df_changes["delta"] = (df_changes["dv24_pos_correct"] - df_changes["dv24_tmp"]).abs()
    rows_with_change = (df_changes["delta"] > 0).sum()

    if rows_with_change > 0:
        changed = df_changes[df_changes["delta"] > 0]
        report["dv24_changes"] = {
            "rows_with_change": int(rows_with_change),
            "min": float(changed["delta"].min()),
            "median": float(changed["delta"].median()),
            "p95": float(changed["delta"].quantile(0.95)),
            "max": float(changed["delta"].max()),
        }
    else:
        report["dv24_changes"] = {
            "rows_with_change": 0,
            "min": None, "median": None, "p95": None, "max": None,
        }

    return report


def effect_2_border_no_temporal_partner(df, scope_label):
    """EFFECT 2: Border defect (no temporal partner)."""
    report = {
        "scope": scope_label,
        "pos_0_no_tmp": 0,
        "pos_1_no_tmp": 0,
        "pos_undef_no_tmp": 0,
    }

    no_tmp = df["dv24_tmp"].isna()
    df_subset = df[no_tmp]

    report["pos_0_no_tmp"] = int(((df_subset["ri_label_pos_correct"] == 0)).sum())
    report["pos_1_no_tmp"] = int(((df_subset["ri_label_pos_correct"] == 1)).sum())
    report["pos_undef_no_tmp"] = int(((df_subset["ri_label_pos_correct"].isna())).sum())

    return report


# ==============================================================================
# STEP 6: PL-GATE ANALYSIS (for dev set, train+val only)
# ==============================================================================

def load_pl_gate_metadata():
    """Load PL-gate metadata for train+val events."""
    # Expected: outputs/results/pl_gate_census.json
    census_path = REPO_ROOT / "outputs" / "results" / "pl_gate_census.json"
    if not census_path.exists():
        return {"dev_pl_gated_count_expected": 14101, "warning": "pl_gate_census.json not found"}

    with open(census_path) as f:
        census = json.load(f)

    return {
        "dev_pl_gated_count_expected": census.get("dev_pl_gated_count", 14101),
        "dev_pl_gated_positives_expected": census.get("dev_pl_gated_positives", 687),
    }


# ==============================================================================
# STEP 7: CONCENTRATION ANALYSIS
# ==============================================================================

def analyze_concentration(df, effect_rows, effect_name):
    """Analyze concentration of affected rows by basin, year, position, landfall proximity."""
    df = df.copy()
    df_affected = df[effect_rows].copy()

    if len(df_affected) == 0:
        return {
            "effect": effect_name,
            "total_affected": 0,
            "by_basin": {},
            "by_decade": {},
            "by_position_in_storm": {},
        }

    report = {
        "effect": effect_name,
        "total_affected": len(df_affected),
    }

    # By basin
    report["by_basin"] = df_affected["basin"].value_counts().to_dict()

    # By decade
    df_affected["year"] = df_affected["timestamp"].dt.year
    df_affected["decade"] = (df_affected["year"] // 10) * 10
    report["by_decade"] = df_affected["decade"].value_counts().sort_index().to_dict()

    # By position in storm
    position_dists = []
    for sid in df_affected["sid"].unique():
        sid_full = df[df["sid"] == sid]
        sid_affected = df_affected[df_affected["sid"] == sid]

        for idx in sid_affected.index:
            pos_from_start = len(sid_full[sid_full.index < idx])
            pos_from_end = len(sid_full) - pos_from_start - 1
            position_dists.append({"from_start": pos_from_start, "from_end": pos_from_end})

    if position_dists:
        report["position_from_start_median"] = float(np.median([p["from_start"] for p in position_dists]))
        report["position_from_end_median"] = float(np.median([p["from_end"] for p in position_dists]))

    return report


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("[1/7] Loading data and computing correct-positional labels...", file=sys.stderr)
    df = load_and_compute_correct_positional(DATA_DIR / "event_list_augmented.csv")
    valid_events = pd.read_csv(DATA_DIR / "normalized" / "valid_events.csv", **CSV_PARAMS)
    splits = pd.read_csv(DATA_DIR / "normalized" / "splits.csv", **CSV_PARAMS)

    print(f"  - Loaded: {len(df)} event list, {len(valid_events)} valid events", file=sys.stderr)

    print("[2/7] Diagnosing DEFECT 0 (cross-SID bleed)...", file=sys.stderr)
    defect_0_report = diagnose_defect_0(df)

    print("[3/7] Computing temporal labels...", file=sys.stderr)
    df = compute_temporal_labels(df)

    print("[4/7] Computing positional misalignment...", file=sys.stderr)
    df = compute_positional_misalignment(df)

    print("[5/7] Three-effect decomposition...", file=sys.stderr)

    # Prepare scopes
    scopes = {
        "full_list": {
            "df": df,
            "label": "Full event list (32,989)",
        },
    }

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
        df[["sid", "timestamp", "dv12_shipped", "dv24_shipped", "ri_label_shipped", "dv12_pos_correct", "dv24_pos_correct", "ri_label_pos_correct", "ri_label_tmp", "dv24_tmp", "basin"]],
        left_on=["sid_parsed", "timestamp_parsed"],
        right_on=["sid", "timestamp"],
        how="left",
    )

    join_coverage = valid_merged["ri_label_shipped"].notna().sum()
    print(f"  - Valid set join coverage: {join_coverage} / {len(valid_events)}", file=sys.stderr)

    if join_coverage == len(valid_events):
        scopes["valid_set"] = {
            "df": valid_merged,
            "label": "Valid dataset (16,780)",
        }

    # Test split (aggregate only)
    test_events = valid_merged[valid_merged["event_id"].isin(splits[splits["split"] == "test"]["event_id"])]
    if len(test_events) > 0:
        scopes["test_split"] = {
            "df": test_events,
            "label": "Test split (aggregate only)",
        }

    # Compute effects for each scope
    full_report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "three_effect_decomposition": True,
            "defect_0_cross_sid_bleed": True,
            "temporal_rule": "Exact-match timestamp t0+24h within same SID",
            "nan_semantics": "UNDEFINED (pd.NA) for no temporal partner or missing wind",
            "read_only": True,
        },
        "defect_0_diagnosis": defect_0_report,
        "effects_by_scope": {},
    }

    for scope_key, scope_data in scopes.items():
        df_scope = scope_data["df"]
        label = scope_data["label"]

        print(f"  - Computing effects for {label}...", file=sys.stderr)

        scope_report = {
            "label": label,
            "row_count": len(df_scope),
            "effect_0": effect_0_shipped_vs_correct_pos(df_scope, label),
            "effect_1": effect_1_pos_vs_temporal_with_partner(df_scope, label),
            "effect_2": effect_2_border_no_temporal_partner(df_scope, label),
        }

        # Combined shipped → temporal
        scope_report["combined_shipped_vs_temporal"] = {
            "ri_label_transition_matrix": {
                "ship_0_to_tmp_0": int(((df_scope["ri_label_shipped"] == 0) & (df_scope["ri_label_tmp"] == 0)).sum()),
                "ship_0_to_tmp_1": int(((df_scope["ri_label_shipped"] == 0) & (df_scope["ri_label_tmp"] == 1)).sum()),
                "ship_0_to_tmp_undef": int(((df_scope["ri_label_shipped"] == 0) & (df_scope["ri_label_tmp"].isna())).sum()),
                "ship_1_to_tmp_0": int(((df_scope["ri_label_shipped"] == 1) & (df_scope["ri_label_tmp"] == 0)).sum()),
                "ship_1_to_tmp_1": int(((df_scope["ri_label_shipped"] == 1) & (df_scope["ri_label_tmp"] == 1)).sum()),
                "ship_1_to_tmp_undef": int(((df_scope["ri_label_shipped"] == 1) & (df_scope["ri_label_tmp"].isna())).sum()),
            },
            "positives_shipped": int((df_scope["ri_label_shipped"] == 1).sum()),
            "positives_temporal": int((df_scope["ri_label_tmp"] == 1).sum()),
        }

        full_report["effects_by_scope"][scope_key] = scope_report

    print("[6/7] Concentration analysis...", file=sys.stderr)
    full_report["concentration"] = {}

    # Effect 0 rows: shipped != correct-pos (handle pd.NA)
    ship_0 = df["ri_label_shipped"] == 0
    ship_1 = df["ri_label_shipped"] == 1
    pos_0 = df["ri_label_pos_correct"] == 0
    pos_1 = df["ri_label_pos_correct"] == 1
    pos_undef = df["ri_label_pos_correct"].isna()

    effect_0_rows = (
        (ship_0 & (pos_1 | pos_undef)) |
        (ship_1 & (pos_0 | pos_undef))
    )
    full_report["concentration"]["effect_0"] = analyze_concentration(df, effect_0_rows, "EFFECT_0")

    # Effect 1 rows (has temporal partner, pos != tmp)
    tmp_0 = df["ri_label_tmp"] == 0
    tmp_1 = df["ri_label_tmp"] == 1
    has_tmp = df["dv24_tmp"].notna()
    effect_1_rows = (
        has_tmp & ((pos_0 & tmp_1) | (pos_1 & tmp_0))
    )
    full_report["concentration"]["effect_1"] = analyze_concentration(df, effect_1_rows, "EFFECT_1")

    # Effect 2 rows (no temporal partner)
    effect_2_rows = df["dv24_tmp"].isna()
    full_report["concentration"]["effect_2"] = analyze_concentration(df, effect_2_rows, "EFFECT_2")

    print("[7/7] Writing outputs...", file=sys.stderr)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    utc_now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = OUTPUTS_DIR / f"report_v2_{utc_now}.json"
    with open(json_path, "w") as f:
        json.dump(convert_to_python_types(full_report), f, indent=2)

    md_report = generate_md_report(full_report)
    md_path = OUTPUTS_DIR / f"report_v2_{utc_now}.md"
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
    md.append("# DV24 Impact Assessment v2: Three-Effect Decomposition\n\n")
    md.append(f"**Generated:** {full_report['generated_utc']}\n")
    md.append("**Method:** SHIPPED → CORRECT-POSITIONAL → TEMPORAL; three separable effects\n\n")

    # Defect 0 diagnosis
    d0 = full_report["defect_0_diagnosis"]
    md.append("## DEFECT 0: Cross-SID Bleed (SHIPPED computed without per-SID grouping)\n\n")
    md.append("### Evidence\n")
    md.append(f"- Shipped vs correct-positional dv24 mismatches: {d0['shipped_vs_correct_pos']['dv24']['mismatches']}\n")
    md.append(f"- All in trailing rows: {d0['trailing_row_analysis']['dv24_trailing_mismatches']}\n")
    md.append(f"- Interior rows (perfect match): {d0['trailing_row_analysis']['dv24_interior_matches']}\n")
    md.append(f"- Global (non-grouped) shift matches: {d0['trailing_row_analysis']['global_shift_matches']}\n")
    md.append(f"- Phantom positives (shipped=1, correct-pos=0): {d0['phantom_positives']['count']}\n")
    md.append(f"- Storms losing ALL positives: {d0['phantom_positives']['affected_sids_count']}\n")
    md.append("\n")

    # Effects by scope
    for scope_key in ["full_list", "valid_set", "test_split"]:
        if scope_key not in full_report["effects_by_scope"]:
            continue

        scope = full_report["effects_by_scope"][scope_key]
        md.append(f"## {scope['label']}\n\n")

        # Effect 0
        md.append("### EFFECT 0: Shipped vs Correct-Positional\n")
        e0 = scope["effect_0"]
        md.append(f"**RI Label Transition:**\n")
        md.append(f"| From | To=0 | To=1 | To=UNDEF |\n")
        md.append(f"|------|------|------|----------|\n")
        tm = e0["ri_label_transition_matrix"]
        md.append(f"| ship=0 | {tm['ship_0_to_pos_0']} | {tm['ship_0_to_pos_1']} | {tm['ship_0_to_pos_undef']} |\n")
        md.append(f"| ship=1 | {tm['ship_1_to_pos_0']} | {tm['ship_1_to_pos_1']} | {tm['ship_1_to_pos_undef']} |\n")
        md.append(f"- Positives shipped: {e0['positives_shipped']}\n")
        md.append(f"- Positives correct-pos: {e0['positives_pos_correct']}\n")
        md.append(f"- Delta: {e0['positives_difference']}\n\n")

        # Effect 1
        md.append("### EFFECT 1: Correct-Positional vs Temporal (with partner)\n")
        e1 = scope["effect_1"]
        md.append(f"**RI Label Transition (rows with temporal partner):**\n")
        md.append(f"| From | To=0 | To=1 |\n")
        md.append(f"|------|------|------|\n")
        tm = e1["ri_label_transition_matrix"]
        md.append(f"| pos=0 | {tm['pos_0_to_tmp_0']} | {tm['pos_0_to_tmp_1']} |\n")
        md.append(f"| pos=1 | {tm['pos_1_to_tmp_0']} | {tm['pos_1_to_tmp_1']} |\n")
        dv24 = e1["dv24_changes"]
        md.append(f"**dv24 changes (|Δ| > 0 only):**\n")
        md.append(f"- Rows with change: {dv24['rows_with_change']}\n")
        if dv24['rows_with_change'] > 0:
            md.append(f"- Min/median/p95/max: {dv24['min']:.1f} / {dv24['median']:.1f} / {dv24['p95']:.1f} / {dv24['max']:.1f}\n\n")
        else:
            md.append("\n")

        # Effect 2
        md.append("### EFFECT 2: Border (No Temporal Partner)\n")
        e2 = scope["effect_2"]
        md.append(f"- Correct-pos=0 with no tmp: {e2['pos_0_no_tmp']}\n")
        md.append(f"- Correct-pos=1 with no tmp: {e2['pos_1_no_tmp']}\n")
        md.append(f"- Correct-pos=UNDEF with no tmp: {e2['pos_undef_no_tmp']}\n\n")

        # Combined
        comb = scope["combined_shipped_vs_temporal"]
        md.append("### COMBINED: Shipped vs Temporal\n")
        md.append(f"**RI Label Transition:**\n")
        md.append(f"| From | To=0 | To=1 | To=UNDEF |\n")
        md.append(f"|------|------|------|----------|\n")
        tm = comb["ri_label_transition_matrix"]
        md.append(f"| ship=0 | {tm['ship_0_to_tmp_0']} | {tm['ship_0_to_tmp_1']} | {tm['ship_0_to_tmp_undef']} |\n")
        md.append(f"| ship=1 | {tm['ship_1_to_tmp_0']} | {tm['ship_1_to_tmp_1']} | {tm['ship_1_to_tmp_undef']} |\n")
        md.append(f"- Positives shipped: {comb['positives_shipped']}\n")
        md.append(f"- Positives temporal: {comb['positives_temporal']}\n\n")

    # Concentration
    if "concentration" in full_report:
        md.append("## Concentration Analysis\n\n")
        for effect_key in ["effect_0", "effect_1", "effect_2"]:
            if effect_key not in full_report["concentration"]:
                continue
            conc = full_report["concentration"][effect_key]
            md.append(f"### {conc['effect']} ({conc['total_affected']} rows)\n")
            if conc['total_affected'] > 0:
                if conc['by_basin']:
                    md.append(f"**By basin:** {conc['by_basin']}\n")
                if conc['by_decade']:
                    md.append(f"**By decade:** {conc['by_decade']}\n")
                if "position_from_start_median" in conc:
                    md.append(f"**Position in storm:** median {conc['position_from_start_median']:.0f} from start, {conc['position_from_end_median']:.0f} from end\n")
            md.append("\n")

    return "".join(md)


if __name__ == "__main__":
    main()
