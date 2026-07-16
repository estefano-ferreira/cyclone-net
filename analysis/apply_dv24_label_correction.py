"""§4 — dv24/dv12 label correction patch script (T-dv24.2).

MODES:
- DEFAULT (dry-run): compute diff-manifest, write only preview + verification report.
  NO files under data/ modified.
- --apply: patch event_list, interim JSONs, and valid_events.csv (re-read and verify
  field-by-field). Requires §5 verification to pass.

REPLICATION GATE (MANDATORY):
Rebuild pre-dropna from raw IBTrACS; compute v1 labels (old positional semantics)
and require EXACT match with shipped event_list (32,989 rows, all columns byte-equal).
IDEMPOTENCE: if event_list already has v2 labels, print "already applied" and exit 0.

DIFF-MANIFEST: all 32,989 rows with v1/v2 labels, reason classification.
VERIFICATION: assert §5 targets (flips, NULLs, value changes) exact match.

INVARIANTS (pre + post):
- Row set unchanged (32,989 rows, same SID/timestamp pairs).
- data/normalized/splits.csv and frozen_splits.json md5 unchanged.
- Interim .npy never touched.
- Interim JSONs: only three label fields differ, only for affected events.
- test split: aggregate counts only, never open test event sidecars except in apply.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.dv24_impact_assessment_v5_raw_reference import build_pre_dropna_series  # noqa: E402
from src.processors.ibtracs import _clean_text_column, _standardize_longitude  # noqa: E402
from src.processors.ri_labeling import add_wind_deltas, label_ri  # noqa: E402
from src.utils.config import load_config, cfg_get  # noqa: E402

logger = logging.getLogger("apply_dv24_label_correction")
logging.basicConfig(level=logging.INFO, format="%(message)s")

ROOT = Path(__file__).resolve().parents[1]
CSV_KWARGS = dict(keep_default_na=False, na_values=[""])
RI_KT = 30.0


def md5_file(path: Path) -> str:
    """Compute MD5 hash of a file."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def compute_v1_labels(pre_dropna: pd.DataFrame) -> pd.DataFrame:
    """Compute v1 labels using OLD POSITIONAL semantics on pre-dropna series.

    This replicates the old builder logic exactly for replication gate.
    """
    out = pre_dropna.copy()

    # Positional shifts (old behavior)
    out["wind_kt_shift_24"] = out.groupby("sid")["wind_kt"].shift(-4)
    out["dv24_kt_v1"] = out["wind_kt_shift_24"] - out["wind_kt"]
    out["ri_label_v1"] = (out["dv24_kt_v1"] >= RI_KT).astype(int)

    # Same for dv12
    out["wind_kt_shift_12"] = out.groupby("sid")["wind_kt"].shift(-2)
    out["dv12_kt_v1"] = out["wind_kt_shift_12"] - out["wind_kt"]

    # Drop rows without future targets (old builder behavior)
    out = out.dropna(subset=["dv12_kt_v1", "dv24_kt_v1"]).copy()
    out = out.reset_index(drop=True)

    return out[["sid", "timestamp", "dv12_kt_v1", "dv24_kt_v1", "ri_label_v1"]]


def compute_v2_labels(pre_dropna: pd.DataFrame, shipped_sids: set) -> pd.DataFrame:
    """Compute v2 labels using NEW STRICT-TEMPORAL semantics.

    Apply strict-temporal to the pre-dropna series, then subset to the frozen
    32,989-row set (same SID/timestamp pairs as shipped event_list).
    """
    out = pre_dropna.copy()

    # Apply strict-temporal labeling (new behavior)
    out = add_wind_deltas(out)
    out = label_ri(out, ri_threshold_kt_24h=RI_KT)

    # Convert nullable Int64 to allow NaN in comparisons
    out["ri_label"] = out["ri_label"].astype("float64", errors="ignore")

    # Subset to frozen row set
    out["pair"] = out["sid"].astype(str) + "|" + out["timestamp"].astype(str)
    out = out[out["pair"].isin(shipped_sids)]
    out = out.reset_index(drop=True)

    return out[["sid", "timestamp", "dv12_kt", "dv24_kt", "ri_label"]]


def classify_reason(v1_row: pd.Series, v2_row: pd.Series) -> str:
    """Classify the change reason."""
    # v1 should never be NA (comes from shipped event list)
    if pd.isna(v1_row["ri_label_v1"]):
        return "error"

    v1_label = int(v1_row["ri_label_v1"])
    v2_label_raw = v2_row["ri_label"]

    # Check if v2 label is undefined (NULL, no exact temporal partner)
    if pd.isna(v2_label_raw):
        return "null_no_partner"

    v2_label = int(v2_label_raw)

    # Check for label flips
    if v1_label != v2_label:
        return "flip_misaligned"

    # Check for value drift (same label, different dv values)
    v1_dv24 = v1_row["dv24_kt_v1"]
    v2_dv24 = v2_row["dv24_kt"]
    v1_dv12 = v1_row["dv12_kt_v1"]
    v2_dv12 = v2_row["dv12_kt"]

    if (not pd.isna(v1_dv24) and not pd.isna(v2_dv24) and v1_dv24 != v2_dv24) or \
       (not pd.isna(v1_dv12) and not pd.isna(v2_dv12) and v1_dv12 != v2_dv12):
        return "dv_drift_only"

    return "unchanged"


def build_diff_manifest(
    shipped: pd.DataFrame,
    v1_labels: pd.DataFrame,
    v2_labels: pd.DataFrame,
    valid_events: pd.DataFrame,
) -> pd.DataFrame:
    """Build the diff-manifest for all 32,989 rows."""
    # Rename v2_labels columns BEFORE merge to avoid collision with shipped columns
    v2_renamed = v2_labels.rename(columns={
        "dv12_kt": "dv12_kt_v2",
        "dv24_kt": "dv24_kt_v2",
        "ri_label": "ri_label_v2",
    })

    # Merge all three datasets on (sid, timestamp)
    m = shipped.merge(v1_labels, on=["sid", "timestamp"], how="left", suffixes=("", "_v1"))
    m = m.merge(v2_renamed, on=["sid", "timestamp"], how="inner")

    # Extract event_id from valid_events (era5_YYYY_MM_DD_HHMM_<SID>)
    valid_events_tmp = valid_events.copy()
    valid_events_tmp["ts"] = pd.to_datetime(
        valid_events_tmp["event_id"].str.extract(r"^era5_(\d{4}_\d{2}_\d{2}_\d{4})_")[0],
        format="%Y_%m_%d_%H%M"
    )
    valid_events_tmp["ts_sid"] = valid_events_tmp["ts"].astype(str) + "|" + valid_events_tmp["sid"].astype(str)

    m["ts_sid"] = m["timestamp"].astype(str) + "|" + m["sid"].astype(str)
    m = m.merge(
        valid_events_tmp[["ts_sid", "event_id"]],
        on="ts_sid",
        how="left",
    )
    m["event_id"] = m["event_id"].fillna("")
    m = m.drop(columns=["ts_sid"])

    # Classify reasons using vectorized np.select (NA-safe, vectorized)
    # Precedence: flip_misaligned > null_no_partner > dv_drift_only > unchanged
    conditions = [
        # flip_misaligned: ri_v1 and ri_v2 both defined and different
        (m["ri_label_v1"].notna() & m["ri_label_v2"].notna() & (m["ri_label_v1"] != m["ri_label_v2"])),
        # null_no_partner: ri_v1 defined and ri_v2 NA
        (m["ri_label_v1"].notna() & m["ri_label_v2"].isna()),
        # dv_drift_only: ri unchanged, but dv24 or dv12 changed
        ((m["ri_label_v1"].notna() & m["ri_label_v2"].notna() & (m["ri_label_v1"] == m["ri_label_v2"])) &
         (((m["dv24_kt_v1"].notna() & m["dv24_kt_v2"].notna() & (m["dv24_kt_v1"] != m["dv24_kt_v2"])) |
           (m["dv12_kt_v1"].notna() & m["dv12_kt_v2"].notna() & (m["dv12_kt_v1"] != m["dv12_kt_v2"])) |
           (m["dv12_kt_v1"].notna() & m["dv12_kt_v2"].isna())))),
    ]
    choices = ["flip_misaligned", "null_no_partner", "dv_drift_only"]
    m["reason"] = np.select(conditions, choices, default="unchanged")

    # Keep only needed columns
    manifest = m[[
        "sid", "timestamp", "event_id",
        "dv12_kt_v1", "dv12_kt_v2",
        "dv24_kt_v1", "dv24_kt_v2",
        "ri_label_v1", "ri_label_v2",
        "reason"
    ]].copy()

    manifest.columns = [
        "sid", "timestamp", "event_id",
        "dv12_v1", "dv12_v2",
        "dv24_v1", "dv24_v2",
        "ri_label_v1", "ri_label_v2",
        "reason"
    ]

    return manifest


def verify_against_targets(shipped: pd.DataFrame, v1_labels: pd.DataFrame, v2_labels: pd.DataFrame,
                            manifest: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Verify against §5 targets."""
    results = {}

    # Merge for comparison (v2_labels has different column names)
    m = shipped.merge(v1_labels, on=["sid", "timestamp"], how="inner")
    v2_renamed = v2_labels.rename(columns={
        "dv12_kt": "dv12_kt_v2",
        "dv24_kt": "dv24_kt_v2",
        "ri_label": "ri_label_v2",
    })
    m = m.merge(v2_renamed, on=["sid", "timestamp"], how="inner")

    # EVENT LIST CHECKS
    results["event_list_rows"] = len(shipped)
    assert results["event_list_rows"] == 32989, f"Event list row count: expected 32989, got {results['event_list_rows']}"

    # Label flips in full event list
    flips_10 = ((m["ri_label_v1"] == 1) & (m["ri_label_v2"] == 0)).sum()
    flips_01 = ((m["ri_label_v1"] == 0) & (m["ri_label_v2"] == 1)).sum()
    results["label_flips_1_to_0"] = int(flips_10)
    results["label_flips_0_to_1"] = int(flips_01)
    assert results["label_flips_1_to_0"] == 1, f"Expected 1 flip (1->0), got {results['label_flips_1_to_0']}"
    assert results["label_flips_0_to_1"] == 0, f"Expected 0 flips (0->1), got {results['label_flips_0_to_1']}"

    # ri_label -> NULL
    to_null = (m["ri_label_v1"].notna() & m["ri_label_v2"].isna()).sum()
    to_null_pos = ((m["ri_label_v1"] == 1) & m["ri_label_v2"].isna()).sum()
    results["ri_label_to_null"] = int(to_null)
    results["ri_label_to_null_positives"] = int(to_null_pos)
    assert results["ri_label_to_null"] == 55, f"Expected 55 ri->NULL, got {results['ri_label_to_null']}"
    assert results["ri_label_to_null_positives"] == 3, f"Expected 3 NULL positives, got {results['ri_label_to_null_positives']}"

    # dv24 value changes (same label, different value)
    dv24_changes = ((m["dv24_kt_v1"] != m["dv24_kt_v2"]) & m["dv24_kt_v2"].notna()).sum()
    results["dv24_value_changes"] = int(dv24_changes)
    assert results["dv24_value_changes"] == 67, f"Expected 67 dv24 changes, got {results['dv24_value_changes']}"

    # dv12 changes
    dv12_changes = ((m["dv12_kt_v1"] != m["dv12_kt_v2"]) & m["dv12_kt_v2"].notna()).sum()
    dv12_to_null = (m["dv12_kt_v1"].notna() & m["dv12_kt_v2"].isna()).sum()
    results["dv12_value_changes"] = int(dv12_changes)
    results["dv12_to_null"] = int(dv12_to_null)
    assert results["dv12_value_changes"] == 37, f"Expected 37 dv12 changes, got {results['dv12_value_changes']}"
    assert results["dv12_to_null"] == 31, f"Expected 31 dv12->NULL, got {results['dv12_to_null']}"

    # VALID SET CHECKS
    valid_events = pd.read_csv(
        ROOT / "data" / "normalized" / "valid_events.csv", **CSV_KWARGS
    )
    splits = pd.read_csv(
        ROOT / "data" / "normalized" / "splits.csv", **CSV_KWARGS
    )

    ve = valid_events.merge(splits, on="event_id")
    ve["ts"] = pd.to_datetime(
        ve["event_id"].str.extract(r"^era5_(\d{4}_\d{2}_\d{2}_\d{4})_")[0],
        format="%Y_%m_%d_%H%M"
    )

    vm = ve.merge(m, left_on=["sid", "ts"], right_on=["sid", "timestamp"], how="inner")

    # Label flips in valid set
    v_flips = ((vm["ri_label_v1"] == 1) & (vm["ri_label_v2"] == 0)).sum() + \
              ((vm["ri_label_v1"] == 0) & (vm["ri_label_v2"] == 1)).sum()
    results["valid_label_flips"] = int(v_flips)
    assert results["valid_label_flips"] == 0, f"Expected 0 valid flips, got {results['valid_label_flips']}"

    # ri_label -> NULL by split
    v_null = vm[vm["ri_label_v1"].notna() & vm["ri_label_v2"].isna()]
    v_null_by_split = v_null["split"].value_counts().to_dict()
    results["valid_ri_label_to_null"] = int(len(v_null))
    results["valid_ri_label_to_null_by_split"] = v_null_by_split
    assert results["valid_ri_label_to_null"] == 19, f"Expected 19 valid ri->NULL, got {results['valid_ri_label_to_null']}"
    assert v_null_by_split.get("train", 0) == 11, f"Expected 11 train NULLs, got {v_null_by_split.get('train', 0)}"
    assert v_null_by_split.get("val", 0) == 2, f"Expected 2 val NULLs, got {v_null_by_split.get('val', 0)}"
    assert v_null_by_split.get("test", 0) == 6, f"Expected 6 test NULLs, got {v_null_by_split.get('test', 0)}"

    # Positives v1 vs v2
    pos_v1 = (vm["ri_label_v1"] == 1).sum()
    pos_v2 = (vm["ri_label_v2"] == 1).sum()
    results["valid_positives_v1"] = int(pos_v1)
    results["valid_positives_v2"] = int(pos_v2)
    assert results["valid_positives_v1"] == 802, f"Expected 802 v1 positives, got {results['valid_positives_v1']}"
    assert results["valid_positives_v2"] == 799, f"Expected 799 v2 positives, got {results['valid_positives_v2']}"

    # Dev PL-gated positives (never test)
    dev_events = vm[vm["split"].isin(["train", "val"])]
    dev_pos_v1 = (dev_events["ri_label_v1"] == 1).sum()
    dev_pos_v2 = (dev_events["ri_label_v2"] == 1).sum()
    results["dev_positives_v1"] = int(dev_pos_v1)
    results["dev_positives_v2"] = int(dev_pos_v2)
    assert results["dev_positives_v1"] == 687, f"Expected 687 dev v1 positives, got {results['dev_positives_v1']}"
    assert results["dev_positives_v2"] == 687, f"Expected 687 dev v2 positives, got {results['dev_positives_v2']}"

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="§4 dv24/dv12 label correction patch script (T-dv24.2)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply patches to data/ files. Default: dry-run only (write manifest preview + verification)."
    )
    args = parser.parse_args()

    cfg = load_config(ROOT / "config.yaml")

    logger.info("=" * 70)
    logger.info("dv24/dv12 LABEL CORRECTION PATCH — %s", "APPLY MODE" if args.apply else "DRY-RUN MODE")
    logger.info("=" * 70)

    # ===================================================================
    # STEP 1: REPLICATION GATE
    # ===================================================================
    logger.info("\n[1/6] REPLICATION GATE: rebuild pre-dropna and verify shipped event list...")

    event_list_path = ROOT / "data" / "event_list_augmented.csv"
    shipped = pd.read_csv(event_list_path, **CSV_KWARGS)
    shipped["timestamp"] = pd.to_datetime(shipped["timestamp"])

    # Build pre-dropna series
    pre_dropna = build_pre_dropna_series()
    logger.info("  Built pre-dropna series: %d rows", len(pre_dropna))

    # Compute v1 labels (old positional semantics)
    v1_labels = compute_v1_labels(pre_dropna)
    logger.info("  Computed v1 labels (positional): %d rows after dropna", len(v1_labels))

    # Check replication vs v1 (positional). Order matters: only when the v1
    # match FAILS do we test the v2/idempotence hypothesis — with the old
    # order, a post-apply re-run aborted before ever reaching the
    # idempotence branch (dead code).
    shipped_pairs = set(shipped["sid"].astype(str) + "|" + shipped["timestamp"].astype(str))
    v1_pairs = set(v1_labels["sid"].astype(str) + "|" + v1_labels["timestamp"].astype(str))

    def _na_eq(a: pd.Series, b: pd.Series) -> bool:
        a = pd.to_numeric(a, errors="coerce")
        b = pd.to_numeric(b, errors="coerce")
        return bool(((a.isna() & b.isna()) | (a.notna() & b.notna() & (a == b))).all())

    m_check = shipped.merge(v1_labels, on=["sid", "timestamp"], how="inner")
    is_v1 = (
        shipped_pairs == v1_pairs
        and len(m_check) == len(shipped)
        and _na_eq(m_check["dv24_kt"], m_check["dv24_kt_v1"])
        and _na_eq(m_check["dv12_kt"], m_check["dv12_kt_v1"])
        and _na_eq(m_check["ri_label"], m_check["ri_label_v1"])
    )

    if is_v1:
        logger.info("  Replication check PASSED: shipped event list is v1 labels (%d/%d)",
                    len(m_check), len(shipped))
    else:
        # IDEMPOTENCE: is the event list already v2? NA-safe on all three fields.
        v2_pre = label_ri(add_wind_deltas(pre_dropna.copy()), ri_threshold_kt_24h=RI_KT)
        m_idem = shipped.merge(
            v2_pre[["sid", "timestamp", "dv12_kt", "dv24_kt", "ri_label"]].rename(
                columns={"dv12_kt": "dv12_v2", "dv24_kt": "dv24_v2", "ri_label": "ri_v2"}),
            on=["sid", "timestamp"], how="inner")
        is_v2 = (
            len(m_idem) == len(shipped)
            and _na_eq(m_idem["dv24_kt"], m_idem["dv24_v2"])
            and _na_eq(m_idem["dv12_kt"], m_idem["dv12_v2"])
            and _na_eq(m_idem["ri_label"], m_idem["ri_v2"])
        )
        if is_v2:
            logger.info("  Event list already carries v2 labels -- nothing to do (idempotent exit).")
            return 0
        logger.error("ERROR: event list matches neither v1 recompute nor v2 labels -- unknown state, aborting.")
        return 1

    # ===================================================================
    # STEP 2: COMPUTE V2 LABELS
    # ===================================================================
    logger.info("\n[2/6] COMPUTING V2 LABELS (strict-temporal semantics)...")

    v2_labels = compute_v2_labels(pre_dropna, shipped_pairs)
    logger.info("  Computed v2 labels: %d rows", len(v2_labels))

    if len(v2_labels) != len(shipped):
        logger.error("ERROR: v2 row count does not match shipped! %d vs %d", len(v2_labels), len(shipped))
        return 1

    # ===================================================================
    # STEP 3: BUILD DIFF-MANIFEST
    # ===================================================================
    logger.info("\n[3/6] BUILDING DIFF-MANIFEST...")

    valid_events = pd.read_csv(
        ROOT / "data" / "normalized" / "valid_events.csv", **CSV_KWARGS
    )

    manifest = build_diff_manifest(shipped, v1_labels, v2_labels, valid_events)
    logger.info("  Diff-manifest: %d rows", len(manifest))

    # Count reason categories
    reason_counts = manifest["reason"].value_counts().to_dict()
    logger.info("  Reason counts: %s", reason_counts)

    # Consistency asserts between manifest and verification
    try:
        assert reason_counts.get("flip_misaligned", 0) == 1, \
            f"Expected 1 flip_misaligned, got {reason_counts.get('flip_misaligned', 0)}"
        assert reason_counts.get("null_no_partner", 0) == 55, \
            f"Expected 55 null_no_partner, got {reason_counts.get('null_no_partner', 0)}"
        assert reason_counts.get("unchanged", 0) < 32989, \
            f"All rows unchanged? Got {reason_counts.get('unchanged', 0)}"

        # Count non-unchanged rows
        non_unchanged = len(manifest[manifest["reason"] != "unchanged"])
        # Count rows where any of the three fields differs (NA-safe)
        has_change = (
            (manifest["dv12_v1"] != manifest["dv12_v2"]) |
            (manifest["dv24_v1"] != manifest["dv24_v2"]) |
            (manifest["ri_label_v1"].astype(str) != manifest["ri_label_v2"].astype(str))
        ).sum()
        # Note: comparison with NaN requires care; use string conversion for NA-safe check
        logger.info("  Manifest consistency: %d non-unchanged rows, %d rows with field differences",
                    non_unchanged, has_change)
    except AssertionError as e:
        logger.error("MANIFEST CONSISTENCY CHECK FAILED: %s", str(e))
        return 1

    # ===================================================================
    # STEP 4: VERIFICATION vs §5 TARGETS
    # ===================================================================
    logger.info("\n[4/6] VERIFICATION vs §5 TARGETS...")

    try:
        verify_results = verify_against_targets(shipped, v1_labels, v2_labels, manifest, cfg)
        logger.info("  All §5 targets VERIFIED:")
        for key, val in verify_results.items():
            if isinstance(val, dict):
                logger.info("    %s: %s", key, val)
            else:
                logger.info("    %s: %s", key, val)
    except AssertionError as e:
        logger.error("VERIFICATION FAILED: %s", str(e))
        return 1

    # ===================================================================
    # STEP 5a: DRY-RUN OUTPUT
    # ===================================================================
    logger.info("\n[5/6] WRITING DRY-RUN OUTPUTS...")

    out_dir = ROOT / "outputs" / "results" / "dv24_impact"
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Write diff-manifest preview (csv with reason counts)
    manifest_out = out_dir / "label_diff_v1_v2_dryrun.csv"
    manifest.to_csv(manifest_out, index=False)
    logger.info("  Wrote diff-manifest preview: %s", manifest_out.relative_to(ROOT))

    # Count interim JSONs that would be touched (needed for both dry-run and --apply)
    logger.info("\n[5b/6] COUNTING INTERIM JSONs TO TOUCH...")
    interim_dir = ROOT / "data" / "interim"
    affected_count = {"train": 0, "val": 0, "test": 0}
    splits = pd.read_csv(ROOT / "data" / "normalized" / "splits.csv", **CSV_KWARGS)

    # Count rows with any field change that have an interim JSON
    changed_rows = manifest[manifest["reason"] != "unchanged"].copy()
    for _, row in changed_rows.iterrows():
        if row["event_id"] == "":
            continue

        json_path = interim_dir / f"{row['event_id']}.json"
        if json_path.exists():
            split_matches = splits[splits["event_id"] == row["event_id"]]
            if len(split_matches) > 0:
                split = split_matches["split"].iloc[0]
                if split in affected_count:
                    affected_count[split] += 1

    logger.info("  Interim JSONs with changes: train=%d, val=%d, test=%d",
                affected_count["train"], affected_count["val"], affected_count["test"])

    # Write verification report (json) with interim JSON counts
    verify_out = out_dir / f"dryrun_verification_{stamp}.json"
    verify_out.write_text(json.dumps({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "dry-run",
        "verification": verify_results,
        "reason_counts": reason_counts,
        "manifest_rows": len(manifest),
        "interim_jsons_to_touch": affected_count,
    }, indent=2), encoding="utf-8")
    logger.info("  Wrote verification report: %s", verify_out.relative_to(ROOT))

    if not args.apply:
        logger.info("\n" + "=" * 70)
        logger.info("DRY-RUN COMPLETE. No data/ files modified.")
        logger.info("To apply the patch, run: python %s --apply", Path(__file__).name)
        logger.info("=" * 70)
        return 0

    # ===================================================================
    # STEP 5b: APPLY PATCHES
    # ===================================================================
    logger.info("\n[5b/6] APPLYING PATCHES (--apply mode)...")

    try:
        dryrun_verification_file = f"dryrun_verification_{stamp}.json"  # Pass actual dryrun file
        result = apply_patches(ROOT, shipped, v2_labels, manifest, valid_events, splits, cfg,
                               affected_count, verify_results, dryrun_verification_file)
        if result != 0:
            return result
    except Exception as e:
        logger.error("PATCH APPLICATION FAILED: %s", str(e))
        return 1

    logger.info("\n" + "=" * 70)
    logger.info("PATCH APPLICATION COMPLETE. All files updated and verified.")
    logger.info("=" * 70)
    return 0


def apply_patches(
    root: Path,
    shipped: pd.DataFrame,
    v2_labels: pd.DataFrame,
    manifest: pd.DataFrame,
    valid_events: pd.DataFrame,
    splits: pd.DataFrame,
    cfg: Dict[str, Any],
    affected_count: Dict[str, int],
    verify_results: Dict[str, Any],
    dryrun_verification_file: str,
) -> int:
    """Apply patches to data files (--apply mode). Each write re-read and verified."""
    interim_dir = root / "data" / "interim"
    normalized_dir = root / "data" / "normalized"

    # Pre-apply invariants: capture baseline state
    event_list_path = root / "data" / "event_list_augmented.csv"
    valid_events_path = normalized_dir / "valid_events.csv"
    splits_path = normalized_dir / "splits.csv"
    frozen_splits_path = normalized_dir / "frozen_splits.json"

    pre_event_list_md5 = md5_file(event_list_path)
    pre_splits_md5 = md5_file(splits_path)
    pre_frozen_md5 = md5_file(frozen_splits_path)
    pre_valid_events = pd.read_csv(valid_events_path, **CSV_KWARGS)

    # ===================================================================
    # 5b.a: Patch event_list_augmented.csv
    # ===================================================================
    logger.info("\n[5b.a] Patching data/event_list_augmented.csv...")

    # Prepare v2 values: merge shipped with v2_labels on (sid, timestamp)
    event_list_out = shipped.copy()
    v2_renamed = v2_labels.rename(columns={
        "dv12_kt": "dv12_kt_v2",
        "dv24_kt": "dv24_kt_v2",
        "ri_label": "ri_label_v2",
    })
    m = event_list_out.merge(v2_renamed, on=["sid", "timestamp"], how="left")

    # Update v2 columns
    event_list_out["dv12_kt"] = m["dv12_kt_v2"]
    event_list_out["dv24_kt"] = m["dv24_kt_v2"]
    # ri_label: NaN becomes empty string for CSV (nullable Int64)
    event_list_out["ri_label"] = m["ri_label_v2"].astype("object").where(m["ri_label_v2"].notna(), "")

    # Drop wind_kt_shift columns (approved decision)
    if "wind_kt_shift_12" in event_list_out.columns:
        event_list_out = event_list_out.drop(columns=["wind_kt_shift_12"])
    if "wind_kt_shift_24" in event_list_out.columns:
        event_list_out = event_list_out.drop(columns=["wind_kt_shift_24"])

    # Write
    event_list_out.to_csv(event_list_path, index=False)

    # Re-read and verify (comprehensive checks per spec item 2)
    event_list_reread = pd.read_csv(event_list_path, **CSV_KWARGS)
    event_list_reread["timestamp"] = pd.to_datetime(event_list_reread["timestamp"])

    # Check (a): row count and column presence/absence
    if len(event_list_reread) != len(shipped):
        logger.error("ERROR: event list row count changed! %d -> %d", len(shipped), len(event_list_reread))
        return 1
    if "wind_kt_shift_12" in event_list_reread.columns or "wind_kt_shift_24" in event_list_reread.columns:
        logger.error("ERROR: shift columns not removed from re-read file")
        return 1

    # Check (b): non-label columns identical to pre-image, column-by-column
    non_label_cols = [c for c in shipped.columns if c not in ["dv12_kt", "dv24_kt", "ri_label", "wind_kt_shift_12", "wind_kt_shift_24"]]
    for col in non_label_cols:
        if col not in event_list_reread.columns:
            logger.error("ERROR: column %s missing from re-read file", col)
            return 1
        # Compare values (handle timestamps specially)
        if col == "timestamp":
            if not (event_list_reread[col] == shipped[col]).all():
                logger.error("ERROR: %s column values differ", col)
                return 1
        else:
            if not (event_list_reread[col].astype(str) == shipped[col].astype(str)).all():
                logger.error("ERROR: %s column values differ", col)
                return 1

    # Check (c): dv12/dv24/ri_label NA-safe comparison with v2 frame
    v2_renamed_check = v2_labels.rename(columns={
        "dv12_kt": "dv12_kt_v2",
        "dv24_kt": "dv24_kt_v2",
        "ri_label": "ri_label_v2",
    })
    m_check = event_list_reread[["sid", "timestamp"]].merge(v2_renamed_check, on=["sid", "timestamp"], how="left")

    # Compare dv12_kt NA-safely
    dv12_v2_numeric = pd.to_numeric(m_check["dv12_kt_v2"], errors="coerce")
    dv12_reread_numeric = pd.to_numeric(event_list_reread["dv12_kt"], errors="coerce")
    mismatch_dv12 = ((dv12_v2_numeric.isna() != dv12_reread_numeric.isna()) |
                     (dv12_v2_numeric.notna() & (dv12_v2_numeric != dv12_reread_numeric)))
    if mismatch_dv12.any():
        logger.error("ERROR: dv12_kt values don't match v2 frame")
        return 1

    # Compare dv24_kt NA-safely
    dv24_v2_numeric = pd.to_numeric(m_check["dv24_kt_v2"], errors="coerce")
    dv24_reread_numeric = pd.to_numeric(event_list_reread["dv24_kt"], errors="coerce")
    mismatch_dv24 = ((dv24_v2_numeric.isna() != dv24_reread_numeric.isna()) |
                     (dv24_v2_numeric.notna() & (dv24_v2_numeric != dv24_reread_numeric)))
    if mismatch_dv24.any():
        logger.error("ERROR: dv24_kt values don't match v2 frame")
        return 1

    # Compare ri_label NA-safely (numeric on BOTH sides — a string comparison
    # here is asymmetric: reread NULLs stringify as "nan", not "")
    ri_v2_numeric = pd.to_numeric(m_check["ri_label_v2"], errors="coerce")
    ri_reread_numeric = pd.to_numeric(event_list_reread["ri_label"], errors="coerce")
    mismatch_ri = ((ri_v2_numeric.isna() != ri_reread_numeric.isna()) |
                   (ri_v2_numeric.notna() & (ri_v2_numeric != ri_reread_numeric)))
    if mismatch_ri.any():
        logger.error("ERROR: ri_label values don't match v2 frame")
        return 1

    logger.info("  Wrote event_list_augmented.csv: %d rows, labels updated, shift columns dropped, all columns verified", len(event_list_reread))

    # ===================================================================
    # 5b.b: Patch interim JSON sidecars
    # ===================================================================
    logger.info("\n[5b.b] Patching interim JSON sidecars (%d affected)...", affected_count["train"] + affected_count["val"] + affected_count["test"])

    # Build lookup for v2 values by event_id
    v2_by_sid_ts = v2_labels.set_index(["sid", "timestamp"]).to_dict("index")

    # Get affected event_ids from manifest
    changed_manifest = manifest[manifest["reason"] != "unchanged"].copy()
    affected_event_ids = set(changed_manifest[changed_manifest["event_id"] != ""]["event_id"].tolist())

    # Track per-JSON md5s and split assignment
    json_md5s = {}  # {event_id: {pre_md5, post_md5}}
    json_by_split = {"train": [], "val": [], "test": []}  # Track which events in each split
    # O(1) split lookup — no per-row dataframe filtering inside the loop.
    split_by_eid = dict(zip(splits["event_id"].astype(str), splits["split"]))

    patched_count = 0
    for event_id in affected_event_ids:
        json_path = interim_dir / f"{event_id}.json"
        if not json_path.exists():
            continue

        # Load pre-image (deep copy stays pristine for the post-write comparison)
        pre_image_orig = json.loads(json_path.read_text(encoding="utf-8"))
        pre_image = copy.deepcopy(pre_image_orig)

        # Extract sid/timestamp from event_id (era5_YYYY_MM_DD_HHMM_<SID>)
        parts = event_id.split("_")
        ts_str = "_".join(parts[1:5])
        sid_str = parts[5]
        ts = pd.to_datetime(ts_str, format="%Y_%m_%d_%H%M")

        # Get v2 values
        key = (sid_str, ts)
        v2_row = v2_by_sid_ts.get(key)
        if v2_row is None:
            logger.warning("  WARNING: event_id %s not found in v2 labels", event_id)
            continue

        # Capture pre-JSON md5
        pre_json_md5 = md5_file(json_path)

        # Patch: update only ri_label, dv12_kt, dv24_kt (None for NA, not NaN)
        pre_image["ri_label"] = int(v2_row["ri_label"]) if pd.notna(v2_row["ri_label"]) else None
        pre_image["dv12_kt"] = float(v2_row["dv12_kt"]) if pd.notna(v2_row["dv12_kt"]) else None
        pre_image["dv24_kt"] = float(v2_row["dv24_kt"]) if pd.notna(v2_row["dv24_kt"]) else None

        # Write with allow_nan=False (enforces no NaN)
        json_path.write_text(json.dumps(pre_image, allow_nan=False, indent=2), encoding="utf-8")

        # Capture post-JSON md5
        post_json_md5 = md5_file(json_path)

        # Re-read and verify (spec item 3: compare post to original, not mutated pre_image)
        post_image = json.loads(json_path.read_text(encoding="utf-8"))

        # Verify (a): three fields match v2 intent
        if post_image.get("ri_label") != pre_image.get("ri_label"):
            logger.error("ERROR: ri_label mismatch after write for %s", event_id)
            return 1
        if post_image.get("dv12_kt") != pre_image.get("dv12_kt"):
            logger.error("ERROR: dv12_kt mismatch after write for %s", event_id)
            return 1
        if post_image.get("dv24_kt") != pre_image.get("dv24_kt"):
            logger.error("ERROR: dv24_kt mismatch after write for %s", event_id)
            return 1

        # Verify (b): other keys byte-identical to original (full dict equality, not len)
        post_other = {k: v for k, v in post_image.items() if k not in ["ri_label", "dv12_kt", "dv24_kt"]}
        orig_other = {k: v for k, v in pre_image_orig.items() if k not in ["ri_label", "dv12_kt", "dv24_kt"]}
        if post_other != orig_other:
            logger.error("ERROR: other keys changed for %s", event_id)
            return 1

        # Track md5s and split assignment
        json_md5s[event_id] = {"pre_md5": pre_json_md5, "post_md5": post_json_md5}
        split = split_by_eid.get(event_id)
        if split in json_by_split:
            json_by_split[split].append(event_id)

        patched_count += 1

    logger.info("  Patched %d interim JSON sidecars", patched_count)

    # ===================================================================
    # 5b.c: Patch valid_events.csv
    # ===================================================================
    logger.info("\n[5b.c] Patching data/normalized/valid_events.csv...")

    # Capture pre-write md5
    pre_valid_events_md5 = md5_file(valid_events_path)

    # Find NULL ri_label rows
    valid_out = pre_valid_events.copy()
    null_ri_event_ids = set(changed_manifest[changed_manifest["ri_label_v2"].isna() & (changed_manifest["event_id"] != "")]["event_id"].tolist())

    # Use isin mask (vectorized, not loop over per-row filtering).
    # Cast to object first: assigning "" into an int64 column is deprecated.
    valid_out["ri_label"] = valid_out["ri_label"].astype("object")
    valid_out.loc[valid_out["event_id"].isin(null_ri_event_ids), "ri_label"] = ""

    # Write and re-read
    valid_out.to_csv(valid_events_path, index=False)
    valid_reread = pd.read_csv(valid_events_path, **CSV_KWARGS)

    if len(valid_reread) != len(pre_valid_events):
        logger.error("ERROR: valid_events row count changed!")
        return 1

    # Verify (spec item 4): all cells outside patched ri_label rows identical to pre-image
    for col in valid_reread.columns:
        if col == "ri_label":
            # ri_label: non-null rows unchanged (numeric NA-safe — reread is
            # float64 after the NaN round-trip, pre-image is int64), null rows empty
            non_null_mask = ~valid_reread["event_id"].isin(null_ri_event_ids)
            a = pd.to_numeric(valid_reread.loc[non_null_mask, col], errors="coerce")
            b = pd.to_numeric(pre_valid_events.loc[non_null_mask, col], errors="coerce")
            if not ((a.isna() & b.isna()) | (a.notna() & b.notna() & (a == b))).all():
                logger.error("ERROR: non-null ri_label rows changed in valid_events")
                return 1
            if not pd.to_numeric(valid_reread.loc[~non_null_mask, col], errors="coerce").isna().all():
                logger.error("ERROR: patched NULL ri_label rows are not empty in valid_events")
                return 1
        else:
            # All other columns: row count and value must be identical
            if not (valid_reread[col].astype(str) == pre_valid_events[col].astype(str)).all():
                logger.error("ERROR: column %s changed in valid_events", col)
                return 1

    post_valid_events_md5 = md5_file(valid_events_path)
    logger.info("  Patched valid_events.csv: %d rows, %d NULL ri_label set to empty, other rows/columns verified identical", len(valid_reread), len(null_ri_event_ids))

    # ===================================================================
    # 5b.d: Verify invariants
    # ===================================================================
    logger.info("\n[5b.d] Verifying invariants...")

    if md5_file(splits_path) != pre_splits_md5:
        logger.error("ERROR: splits.csv md5 changed!")
        return 1

    if md5_file(frozen_splits_path) != pre_frozen_md5:
        logger.error("ERROR: frozen_splits.json md5 changed!")
        return 1

    if len(event_list_reread) != 32989:
        logger.error("ERROR: event list row count != 32989!")
        return 1

    logger.info("  Invariants verified: splits md5 unchanged, frozen_splits md5 unchanged, event list 32989 rows")

    # ===================================================================
    # 5b.e: Write canonical outputs
    # ===================================================================
    logger.info("\n[5b.e] Writing canonical outputs...")

    # root-based (not ROOT): apply_patches must stay fully parameterized on
    # `root` so a sandboxed run never writes into the real tree.
    prov_dir = root / "outputs" / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Canonical diff-manifest lives with the released data (design §4.5e)
    manifest_out = normalized_dir / "label_diff_v1_v2.csv"
    manifest.to_csv(manifest_out, index=False)

    # Write provenance manifest (spec item 5: proper md5s, measured verify_results, actual dryrun file)
    prov_out = prov_dir / f"dv24_label_correction_{stamp}.json"

    # Build per-JSON md5 summary (test split as aggregates only, per spec)
    json_md5_summary = {}
    for split in ["train", "val"]:
        json_md5_summary[split] = [
            {"event_id": eid, "pre_md5": json_md5s[eid]["pre_md5"], "post_md5": json_md5s[eid]["post_md5"]}
            for eid in json_by_split[split]
        ]

    # Test split: aggregate md5s
    test_pre_hashes = [json_md5s[eid]["pre_md5"] for eid in json_by_split["test"]]
    test_post_hashes = [json_md5s[eid]["post_md5"] for eid in json_by_split["test"]]
    combined_pre = hashlib.md5("".join(test_pre_hashes).encode()).hexdigest()
    combined_post = hashlib.md5("".join(test_post_hashes).encode()).hexdigest()
    json_md5_summary["test"] = {
        "count": len(json_by_split["test"]),
        "combined_pre_md5": combined_pre,
        "combined_post_md5": combined_post,
    }

    prov_out.write_text(json.dumps({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "apply",
        "reference_report": "report_v5_20260716_152525",
        "reference_dryrun": dryrun_verification_file,
        "repo_relative_paths": True,
        "files_patched": {
            "data/event_list_augmented.csv": {
                "pre_md5": pre_event_list_md5,
                "post_md5": md5_file(event_list_path),
                "rows": 32989,
                "changes": "dv12_kt/dv24_kt/ri_label updated; wind_kt_shift_12/24 dropped",
            },
            "data/normalized/valid_events.csv": {
                "pre_md5": pre_valid_events_md5,
                "post_md5": post_valid_events_md5,
                "rows": len(valid_reread),
                "ri_label_null_count": len(null_ri_event_ids),
                "changes": "ri_label emptied for NULL events",
            },
            "interim_json_sidecars_by_split": json_md5_summary,
        },
        "verification_results": verify_results,  # Measured, not hardcoded
    }, indent=2), encoding="utf-8")

    logger.info("  Wrote diff-manifest: %s", manifest_out.relative_to(root))
    logger.info("  Wrote provenance manifest: %s", prov_out.relative_to(root))

    return 0


if __name__ == "__main__":
    sys.exit(main())
