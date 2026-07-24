"""
Tests for --drop-basin flag in analysis/tabular_baseline_kfold.py.

Validates that resolve_feature_sets correctly filters basin_* columns
when drop_basin=True, preserves order and backwards compatibility, and
integrates correctly with the published feature cache.
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.tabular_baseline_kfold import resolve_feature_sets  # noqa: E402


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Create a minimal synthetic DataFrame with all feature groups."""
    return pd.DataFrame({
        "event_id": ["e1", "e2"],
        "sid": ["s1", "s2"],
        "ri_label": [0, 1],
        "wind_kt": [50.0, 60.0],
        "basin_EP": [1.0, 0.0],
        "basin_": [0.0, 1.0],  # Basin with empty suffix (edge case)
        "doy_sin": [0.5, 0.6],
        "cube_mean_x": [10.0, 11.0],
        "cube_x_missing": [0.0, 0.0],
    })


class TestResolveFeatureSetsDefault:
    """Test default behavior (drop_basin=False) — must match current published output."""

    def test_default_state_only(self, synthetic_df: pd.DataFrame) -> None:
        """state_only should contain all non-cube_ cols in DataFrame order."""
        result = resolve_feature_sets(synthetic_df)
        expected = ["wind_kt", "basin_EP", "basin_", "doy_sin"]
        assert result["state_only"] == expected, (
            f"state_only order mismatch: got {result['state_only']}, "
            f"expected {expected}"
        )

    def test_default_fields_only(self, synthetic_df: pd.DataFrame) -> None:
        """fields_only should contain only cube_* cols."""
        result = resolve_feature_sets(synthetic_df)
        expected = ["cube_mean_x", "cube_x_missing"]
        assert result["fields_only"] == expected

    def test_default_state_plus_fields(self, synthetic_df: pd.DataFrame) -> None:
        """state_plus_fields should be state_only + fields_only."""
        result = resolve_feature_sets(synthetic_df)
        expected = (["wind_kt", "basin_EP", "basin_", "doy_sin"] +
                    ["cube_mean_x", "cube_x_missing"])
        assert result["state_plus_fields"] == expected

    def test_default_is_dict_not_lists(self, synthetic_df: pd.DataFrame) -> None:
        """Result must be a dict with string keys and list values."""
        result = resolve_feature_sets(synthetic_df)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"state_only", "fields_only", "state_plus_fields"}
        for v in result.values():
            assert isinstance(v, list)


class TestResolveFeatureSetsDropBasin:
    """Test drop_basin=True — removes basin_* columns."""

    def test_drop_basin_state_only(self, synthetic_df: pd.DataFrame) -> None:
        """state_only should exclude basin_* when drop_basin=True."""
        result = resolve_feature_sets(synthetic_df, drop_basin=True)
        # Should have wind_kt and doy_sin, but NOT basin_EP or basin_
        assert "basin_EP" not in result["state_only"]
        assert "basin_" not in result["state_only"]
        assert "wind_kt" in result["state_only"]
        assert "doy_sin" in result["state_only"]
        expected = ["wind_kt", "doy_sin"]
        assert result["state_only"] == expected

    def test_drop_basin_fields_only_unchanged(self, synthetic_df: pd.DataFrame) -> None:
        """fields_only should be unchanged (no basin_* prefix there)."""
        result_default = resolve_feature_sets(synthetic_df, drop_basin=False)
        result_drop = resolve_feature_sets(synthetic_df, drop_basin=True)
        assert result_drop["fields_only"] == result_default["fields_only"]

    def test_drop_basin_state_plus_fields(self, synthetic_df: pd.DataFrame) -> None:
        """state_plus_fields should equal dropped state_only + fields_only."""
        result = resolve_feature_sets(synthetic_df, drop_basin=True)
        expected = ["wind_kt", "doy_sin"] + ["cube_mean_x", "cube_x_missing"]
        assert result["state_plus_fields"] == expected

    def test_drop_basin_count(self, synthetic_df: pd.DataFrame) -> None:
        """Verify exact column count reduction."""
        result_default = resolve_feature_sets(synthetic_df, drop_basin=False)
        result_drop = resolve_feature_sets(synthetic_df, drop_basin=True)
        # Default has 2 basin cols, drop should remove exactly 2
        assert len(result_default["state_only"]) == 4
        assert len(result_drop["state_only"]) == 2
        assert len(result_default["state_only"]) - len(result_drop["state_only"]) == 2


class TestBackwardsCompatibility:
    """Test that old code (positional args only) still works."""

    def test_positional_call_only_df(self, synthetic_df: pd.DataFrame) -> None:
        """Old call: resolve_feature_sets(df) should work (default drop_basin=False)."""
        result = resolve_feature_sets(synthetic_df)
        assert "state_only" in result
        assert "basin_EP" in result["state_only"]

    def test_default_is_false(self, synthetic_df: pd.DataFrame) -> None:
        """Calling with only df should be identical to drop_basin=False."""
        result_pos = resolve_feature_sets(synthetic_df)
        result_explicit = resolve_feature_sets(synthetic_df, drop_basin=False)
        assert result_pos == result_explicit


class TestErrorHandling:
    """Test that ValueError is preserved and expected."""

    def test_missing_field_cols_raises(self) -> None:
        """Should raise ValueError if no cube_* columns exist."""
        df_no_fields = pd.DataFrame({
            "event_id": ["e1"],
            "sid": ["s1"],
            "ri_label": [0],
            "wind_kt": [50.0],
        })
        with pytest.raises(ValueError, match="factorial groups"):
            resolve_feature_sets(df_no_fields)

    def test_missing_state_cols_raises(self) -> None:
        """Should raise ValueError if no state columns exist after drop_basin."""
        df_only_basin_cube = pd.DataFrame({
            "event_id": ["e1"],
            "sid": ["s1"],
            "ri_label": [0],
            "basin_EP": [1.0],
            "cube_mean_x": [10.0],
        })
        with pytest.raises(ValueError, match="factorial groups"):
            resolve_feature_sets(df_only_basin_cube, drop_basin=True)


class TestCacheIntegration:
    """Integration test: verify against published cache and summary."""

    @pytest.mark.skipif(
        not Path("outputs/results/tabular_baseline/feature_table.parquet").exists()
        and not Path("outputs/results/tabular_baseline/feature_table.csv").exists(),
        reason="Cache not found (outputs/results/tabular_baseline/feature_table.{parquet,csv})"
    )
    def test_default_matches_published_summary(self) -> None:
        """Default resolve_feature_sets(cache_df) must match published summary exactly."""
        cache_dir = Path("outputs/results/tabular_baseline")
        summary_path = cache_dir / "20260715T123221Z" / "summary.json"

        # Load cache (try parquet first, then csv).
        cache_df = None
        if (cache_dir / "feature_table.parquet").exists():
            cache_df = pd.read_parquet(cache_dir / "feature_table.parquet")
        elif (cache_dir / "feature_table.csv").exists():
            cache_df = pd.read_csv(cache_dir / "feature_table.csv")
        else:
            pytest.skip("No feature table cache found")

        result = resolve_feature_sets(cache_df)

        # Load published summary if it exists.
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            published_sets = summary.get("feature_sets", {})
            for set_name in ("state_only", "fields_only", "state_plus_fields"):
                assert result[set_name] == published_sets.get(set_name), (
                    f"{set_name} mismatch: "
                    f"computed {result[set_name]} vs "
                    f"published {published_sets.get(set_name)}"
                )

    @pytest.mark.skipif(
        not Path("outputs/results/tabular_baseline/feature_table.parquet").exists()
        and not Path("outputs/results/tabular_baseline/feature_table.csv").exists(),
        reason="Cache not found"
    )
    def test_drop_basin_removes_exactly_basin_cols(self) -> None:
        """Verify that drop_basin=True removes only basin_* columns from cache."""
        cache_dir = Path("outputs/results/tabular_baseline")
        cache_df = None
        if (cache_dir / "feature_table.parquet").exists():
            cache_df = pd.read_parquet(cache_dir / "feature_table.parquet")
        elif (cache_dir / "feature_table.csv").exists():
            cache_df = pd.read_csv(cache_dir / "feature_table.csv")
        else:
            pytest.skip("No feature table cache found")

        result_default = resolve_feature_sets(cache_df, drop_basin=False)
        result_drop = resolve_feature_sets(cache_df, drop_basin=True)

        removed_cols = set(result_default["state_only"]) - set(result_drop["state_only"])
        # All removed cols should start with "basin_"
        for col in removed_cols:
            assert col.startswith("basin_"), f"Removed col {col} does not start with basin_"

        # Exactly the basin_* columns should be removed.
        basin_cols = [c for c in result_default["state_only"] if c.startswith("basin_")]
        assert removed_cols == set(basin_cols)
