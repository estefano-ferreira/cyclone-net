"""
Tests for IBTrACS basin code preservation under pandas parsing.

The literal string "NA" (North Atlantic) must survive read_csv with
keep_default_na=False, na_values=[" "]. This module verifies the fix
for ERRATA item 7 (basin code loss in event-list generation).
"""

from pathlib import Path

import pandas as pd
import pytest

from src.processors.ibtracs import build_event_list


@pytest.fixture
def mini_ibtracs_csv(tmp_path: Path) -> Path:
    """Write a minimal IBTrACS CSV with one NA and one EP storm.

    The CSV mimics IBTrACS structure:
    - header row
    - line 2: units row (all spaces or unit strings)
    - data rows

    One NA storm (2020222N30300): 6 six-hourly records 2020-08-10 00:00 to
    2020-08-11 06:00. One EP storm (2020223N15250): 6 six-hourly records
    2020-08-12 00:00 to 2020-08-13 06:00. USA_PRES in NA storm has blanks
    in first two records (encoded as single space in CSV).
    """
    csv_path = tmp_path / "mini_ibtracs.csv"

    header = "SID,NAME,BASIN,ISO_TIME,LAT,LON,USA_WIND,USA_PRES"
    units = " , , , ,degrees_north,degrees_east,kts,mb"

    na_records = [
        "2020222N30300,TESTNA,NA,2020-08-10 00:00,25.0,-60.0,40, ",
        "2020222N30300,TESTNA,NA,2020-08-10 06:00,25.2,-59.8,45, ",
        "2020222N30300,TESTNA,NA,2020-08-10 12:00,25.4,-59.6,50,990",
        "2020222N30300,TESTNA,NA,2020-08-10 18:00,25.6,-59.4,60,985",
        "2020222N30300,TESTNA,NA,2020-08-11 00:00,25.8,-59.2,75,980",
        "2020222N30300,TESTNA,NA,2020-08-11 06:00,26.0,-59.0,85,975",
    ]

    ep_records = [
        "2020223N15250,TESTEP,EP,2020-08-12 00:00,15.0,-110.0,35,1000",
        "2020223N15250,TESTEP,EP,2020-08-12 06:00,15.2,-109.8,40,998",
        "2020223N15250,TESTEP,EP,2020-08-12 12:00,15.4,-109.6,45,996",
        "2020223N15250,TESTEP,EP,2020-08-12 18:00,15.6,-109.4,55,994",
        "2020223N15250,TESTEP,EP,2020-08-13 00:00,15.8,-109.2,70,992",
        "2020223N15250,TESTEP,EP,2020-08-13 06:00,16.0,-109.0,80,990",
    ]

    lines = [header, units] + na_records + ep_records
    csv_path.write_text("\n".join(lines))

    return csv_path


def test_na_basin_survives_parser(mini_ibtracs_csv: Path, tmp_path: Path) -> None:
    """Basin code 'NA' (North Atlantic) must survive parsing in the event list."""
    out_csv = tmp_path / "event_list.csv"
    build_event_list(
        ibtracs_csv=mini_ibtracs_csv,
        out_csv=out_csv,
        bbox=None,
        ri_threshold_kt_24h=30.0,
    )

    df = pd.read_csv(out_csv, keep_default_na=False, na_values=[""])
    basins = set(df["basin"].unique())

    assert basins == {"NA", "EP"}, f"Expected {{'NA', 'EP'}}, got {basins}"
    assert not df["basin"].isna().any(), "Basin column contains NaN values"
    assert not (df["basin"] == "").any(), "Basin column contains empty strings"


def test_numeric_blank_pressure_is_nan(mini_ibtracs_csv: Path, tmp_path: Path) -> None:
    """Blank pressure fields (single space in IBTrACS) must map to NaN.

    Uses drop_undefined=True (old positional-semantics behavior) to drop rows
    without dv12/dv24 partners, so expectations match the old test format.
    """
    out_csv = tmp_path / "event_list.csv"
    build_event_list(
        ibtracs_csv=mini_ibtracs_csv,
        out_csv=out_csv,
        bbox=None,
        ri_threshold_kt_24h=30.0,
        drop_undefined=True,  # old behavior: drop rows without dv12/dv24
    )

    df = pd.read_csv(out_csv, keep_default_na=False, na_values=[""])

    # Only rows with BOTH +12h and +24h future targets survive the dropna:
    # with 6 six-hourly records per storm, that is the first 2 rows.
    na_records = df[df["sid"] == "2020222N30300"].sort_values("timestamp")
    pressure_na = na_records["pressure_mb"].values
    assert len(pressure_na) == 2
    assert pd.isna(pressure_na[0]), "Blank (space) NA-storm pressure should be NaN"
    assert pd.isna(pressure_na[1]), "Blank (space) NA-storm pressure should be NaN"

    ep_records = df[df["sid"] == "2020223N15250"].sort_values("timestamp")
    pressure_ep = ep_records["pressure_mb"].values
    assert len(pressure_ep) == 2
    assert pressure_ep[0] == 1000.0, "Numeric EP pressure should parse as float"
    assert pressure_ep[1] == 998.0, "Numeric EP pressure should parse as float"


def test_units_row_dropped(mini_ibtracs_csv: Path, tmp_path: Path) -> None:
    """Units row (line 2) must not appear in the event list.

    Uses drop_undefined=True (old positional-semantics behavior) to drop rows
    without dv12/dv24 partners, so expectations match the old test format.
    """
    out_csv = tmp_path / "event_list.csv"
    build_event_list(
        ibtracs_csv=mini_ibtracs_csv,
        out_csv=out_csv,
        bbox=None,
        ri_threshold_kt_24h=30.0,
        drop_undefined=True,  # old behavior: drop rows without dv12/dv24
    )

    df = pd.read_csv(out_csv, keep_default_na=False, na_values=[""])

    assert (df["sid"] == "").sum() == 0, "No empty SID values in event list"
    assert (df["sid"] == " ").sum() == 0, "No space-only SID values in event list"

    # Each storm has 6 six-hourly records spanning 30 h; rows survive only
    # with both +12h and +24h future targets, i.e. the first 2 per storm.
    records_per_storm = 6
    surviving_per_storm = records_per_storm - 4  # 24 h = 4 six-hourly steps
    expected_count = 2 * surviving_per_storm

    valid_row_count = len(df)
    assert (
        valid_row_count == expected_count
    ), f"Expected {expected_count} rows (rows with dv12+dv24), got {valid_row_count}"


def test_event_list_readers_preserve_na(tmp_path: Path) -> None:
    """Event-list readers in all three modules must preserve basin 'NA'."""
    csv_path = tmp_path / "event_list_minimal.csv"

    header = "sid,storm_name,name,basin,timestamp,lat,lon,wind_kt,pressure_mb,dv12_kt,dv24_kt,ri_label"
    na_line = "2020222N30300,TESTNA,TESTNA,NA,2020-08-10 00:00,25.0,-60.0,40,,35,40,1"
    ep_line = "2020223N15250,TESTEP,TESTEP,EP,2020-08-12 00:00,15.0,-110.0,35,1000,40,45,0"

    csv_path.write_text("\n".join([header, na_line, ep_line]))

    from src.processors.preprocess_scientific import load_event_list
    from src.pipeline.pl_backfill import _window_events_from_list
    from src.pipeline.windowed import _window_events

    df_preprocess = load_event_list(csv_path)
    assert set(df_preprocess["basin"].unique()) == {
        "NA",
        "EP",
    }, "preprocess_scientific.load_event_list lost basin codes"
    assert not df_preprocess["basin"].isna().any()

    df_windowed = _window_events(csv_path, y0=2020, y1=2020)
    assert set(df_windowed["basin"].unique()) == {
        "NA",
        "EP",
    }, "windowed._window_events lost basin codes"
    assert not df_windowed["basin"].isna().any()

    df_backfill = _window_events_from_list(csv_path, y0=2020, y1=2020)
    assert set(df_backfill["basin"].unique()) == {
        "NA",
        "EP",
    }, "pl_backfill._window_events_from_list lost basin codes"
    assert not df_backfill["basin"].isna().any()

    assert pd.isna(df_preprocess.iloc[0]["pressure_mb"]), "Blank pressure should be NaN"
    assert pd.isna(df_windowed.iloc[0]["pressure_mb"]), "Blank pressure should be NaN"
    assert pd.isna(df_backfill.iloc[0]["pressure_mb"]), "Blank pressure should be NaN"
