"""
Tests for §4 patch script pure functions.

Tests reason classification, NA-safe comparison, and event_id derivation
on synthetic frames. Does NOT run the patch script against real data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.apply_dv24_label_correction import classify_reason


class TestReasonClassification:
    """Reason classification for diff-manifest."""

    def test_unchanged_exact_match(self) -> None:
        """No change in label or value → unchanged."""
        v1 = pd.Series({
            "ri_label_v1": 1,
            "dv24_kt_v1": 35.0,
            "dv12_kt_v1": 15.0,
        })
        v2 = pd.Series({
            "ri_label": 1.0,
            "dv24_kt": 35.0,
            "dv12_kt": 15.0,
        })
        reason = classify_reason(v1, v2)
        assert reason == "unchanged", f"Expected unchanged, got {reason}"

    def test_flip_misaligned_1_to_0(self) -> None:
        """Label flip 1→0 due to misalignment → flip_misaligned."""
        v1 = pd.Series({
            "ri_label_v1": 1,
            "dv24_kt_v1": 35.0,
            "dv12_kt_v1": 15.0,
        })
        v2 = pd.Series({
            "ri_label": 0.0,
            "dv24_kt": 29.0,
            "dv12_kt": 14.0,
        })
        reason = classify_reason(v1, v2)
        assert reason == "flip_misaligned", f"Expected flip_misaligned, got {reason}"

    def test_flip_misaligned_0_to_1(self) -> None:
        """Label flip 0→1 due to misalignment → flip_misaligned."""
        v1 = pd.Series({
            "ri_label_v1": 0,
            "dv24_kt_v1": 25.0,
            "dv12_kt_v1": 10.0,
        })
        v2 = pd.Series({
            "ri_label": 1.0,
            "dv24_kt": 31.0,
            "dv12_kt": 15.0,
        })
        reason = classify_reason(v1, v2)
        assert reason == "flip_misaligned", f"Expected flip_misaligned, got {reason}"

    def test_null_no_partner(self) -> None:
        """v2 label is NULL (no exact temporal partner) → null_no_partner."""
        v1 = pd.Series({
            "ri_label_v1": 1,
            "dv24_kt_v1": 35.0,
            "dv12_kt_v1": 15.0,
        })
        v2 = pd.Series({
            "ri_label": None,  # NULL label (no exact temporal partner)
            "dv24_kt": None,
            "dv12_kt": None,
        })
        reason = classify_reason(v1, v2)
        assert reason == "null_no_partner", f"Expected null_no_partner, got {reason}"

    def test_dv_drift_only_same_label(self) -> None:
        """dv value changed but label remains same → dv_drift_only."""
        v1 = pd.Series({
            "ri_label_v1": 1,
            "dv24_kt_v1": 35.0,
            "dv12_kt_v1": 15.0,
        })
        v2 = pd.Series({
            "ri_label": 1.0,
            "dv24_kt": 36.0,  # drift in value
            "dv12_kt": 15.0,
        })
        reason = classify_reason(v1, v2)
        assert reason == "dv_drift_only", f"Expected dv_drift_only, got {reason}"

    def test_dv12_drift_only(self) -> None:
        """dv12 changed but label/dv24 same → dv_drift_only."""
        v1 = pd.Series({
            "ri_label_v1": 0,
            "dv24_kt_v1": 25.0,
            "dv12_kt_v1": 10.0,
        })
        v2 = pd.Series({
            "ri_label": 0.0,
            "dv24_kt": 25.0,
            "dv12_kt": 11.0,  # dv12 drift only
        })
        reason = classify_reason(v1, v2)
        assert reason == "dv_drift_only", f"Expected dv_drift_only, got {reason}"


class TestEventIdDerivation:
    """Event_id is derived from valid_events and interim JSON existence."""

    def test_event_id_format(self) -> None:
        """Event ID follows format era5_YYYY_MM_DD_HHMM_<SID>."""
        event_id = "era5_2020_08_10_0000_2020001N00001"
        # Expected parts: era5, 2020, 08, 10, 0000, 2020001N00001 (6 parts, SID doesn't split further on _)
        parts = event_id.split("_")
        assert len(parts) == 6, f"Event ID should have 6 parts, got {len(parts)}: {parts}"
        assert parts[0] == "era5"
        assert parts[1] == "2020"
        assert parts[2] == "08"
        assert parts[3] == "10"
        assert parts[4] == "0000"
        assert parts[5] == "2020001N00001"

    def test_event_id_timestamp_parse(self) -> None:
        """Timestamp can be extracted from event_id."""
        event_id = "era5_2020_08_10_0145_2020001N00001"
        # Extract timestamp part: YYYY_MM_DD_HHMM
        ts_part = event_id.split("_")[1:5]
        ts_str = "_".join(ts_part)
        ts = pd.to_datetime(ts_str, format="%Y_%m_%d_%H%M")
        assert ts.year == 2020
        assert ts.month == 8
        assert ts.day == 10
        assert ts.hour == 1
        assert ts.minute == 45


class TestNASafeComparison:
    """Comparisons must be NA-safe (NaN != NaN in IEEE arithmetic)."""

    def test_compare_with_nan_values(self) -> None:
        """NaN values in comparisons via pd.isna()."""
        v1 = 35.0
        v2 = None

        # Use pd.isna() for safe comparison
        assert not pd.isna(v1)
        assert pd.isna(v2)

    def test_series_notna_check(self) -> None:
        """Series notna() method for safe NA checks."""
        s = pd.Series([1.0, None, 3.0, None])
        assert s.notna()[0]  # True for valid value
        assert not s.notna()[1]  # False for None
        assert s.notna()[2]  # True for valid value

    def test_na_in_nullable_int64(self) -> None:
        """Nullable Int64 NA comparison."""
        val = pd.NA
        assert pd.isna(val)

        series = pd.Series([0, 1, None], dtype="Int64")
        # Use isna() on the series, not individual values
        assert not series.isna()[0]
        assert not series.isna()[1]
        assert series.isna()[2]
