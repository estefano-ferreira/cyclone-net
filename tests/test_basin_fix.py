"""Tests for the basin-code fix in platform/build/build_events.py.

Bug: data/event_list_augmented.csv has an EMPTY basin for every North
Atlantic row (~16,602 rows) because an earlier upstream read of IBTrACS used
pandas' default NA-parsing, silently turning the literal basin code "NA"
into a missing value. This is fixed at BUILD time by joining basin from the
raw IBTrACS file (read with keep_default_na=False).

Entirely synthetic tmp_path fixtures -- no real data/ file is read. Reading
a raw IBTrACS CSV would be read-only and safe even during the live
data/interim backfill (it isn't a data/interim file), but these tests still
never touch anything under the real project's data/ directory: everything
is written under tmp_path.
"""
import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "platform" / "build"))
import build_events as be  # noqa: E402  (path must be extended first)


FIELDNAMES = [
    "sid", "storm_name", "name", "basin", "timestamp", "lat", "lon",
    "wind_kt", "pressure_mb", "dv12_kt", "dv24_kt", "ri_label",
]


def _write_raw_ibtracs(path: Path, rows) -> Path:
    """rows: list of (sid, basin) tuples. Writes a header line, a dummy
    units line (skiprows=[1] in the real reader drops it), then data rows.
    Extra IBTrACS columns are irrelevant here since load_basin_lookup uses
    usecols=["SID", "BASIN"]."""
    lines = ["SID,SEASON,BASIN,NAME\n", "num,year,text,text\n"]
    for sid, basin in rows:
        lines.append(f"{sid},1980,{basin},TESTSTORM\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _write_event_list(root: Path, rows) -> Path:
    csv_path = root / "event_list_augmented.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def _write_config(root: Path, interim_dir: Path) -> Path:
    cfg = {
        "labels": {"ri_threshold_kt_24h": 30.0},
        "paths": {"interim_data": str(interim_dir)},
        "data": {"window_size_px": 4},
    }
    cfg_path = root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _default_row(sid, timestamp, basin=""):
    return {
        "sid": sid, "storm_name": "TESTSTORM", "name": "TESTSTORM", "basin": basin,
        "timestamp": timestamp, "lat": 16.0, "lon": -80.0,
        "wind_kt": 50.0, "pressure_mb": 995.0,
        "dv12_kt": 5.0, "dv24_kt": 10.0, "ri_label": 0,
    }


def _run_build(root: Path, rows, raw_ibtracs_csv: Path = None, with_env: bool = False):
    interim_dir = root / "interim"
    raw_ibtracs_csv = raw_ibtracs_csv or (root / "no_such_ibtracs.csv")
    augmented_csv = _write_event_list(root, rows)
    config_file = _write_config(root, interim_dir)
    tmp_dir = root / "data_build_tmp"
    final_dir = root / "data"
    return be.build_events(
        with_env=with_env,
        augmented_csv=augmented_csv,
        config_file=config_file,
        tmp_dir=tmp_dir,
        final_dir=final_dir,
        interim_dir=interim_dir,
        raw_ibtracs_csv=raw_ibtracs_csv,
    )


# ---------------------------------------------------------------------
# load_basin_lookup / resolve_basin unit tests
# ---------------------------------------------------------------------

def test_load_basin_lookup_preserves_na_code(tmp_path):
    """The whole point of keep_default_na=False: without it, pandas would
    silently turn the literal string "NA" into a missing value."""
    raw = _write_raw_ibtracs(tmp_path / "ibtracs.csv", [
        ("1980161N09249", "NA"),
        ("1980162N10250", "EP"),
    ])
    lookup = be.load_basin_lookup(raw)
    assert lookup["1980161N09249"] == "NA"
    assert lookup["1980162N10250"] == "EP"


def test_load_basin_lookup_drops_duplicate_sids_keeping_first(tmp_path):
    raw = _write_raw_ibtracs(tmp_path / "ibtracs.csv", [
        ("1980161N09249", "NA"),
        ("1980161N09249", "NA"),  # second best-track fix, same storm/basin
    ])
    lookup = be.load_basin_lookup(raw)
    assert lookup == {"1980161N09249": "NA"}


def test_load_basin_lookup_missing_file_returns_empty_dict(tmp_path):
    assert be.load_basin_lookup(tmp_path / "does_not_exist.csv") == {}


def test_resolve_basin_fills_empty_string_from_lookup():
    assert be.resolve_basin("", "SID1", {"SID1": "NA"}) == "NA"


def test_resolve_basin_fills_nan_from_lookup():
    import math
    assert be.resolve_basin(float("nan"), "SID1", {"SID1": "NA"}) == "NA"


def test_resolve_basin_keeps_existing_nonempty_value():
    # Augmented CSV already has a real value (e.g. "EP") -- never overwritten.
    assert be.resolve_basin("EP", "SID1", {"SID1": "WRONG"}) == "EP"


def test_resolve_basin_falls_back_unchanged_when_sid_not_in_lookup():
    assert be.resolve_basin("", "UNKNOWN_SID", {}) == ""


# ---------------------------------------------------------------------
# Full build_events() integration
# ---------------------------------------------------------------------

def test_build_events_fills_basin_from_raw_ibtracs(tmp_path):
    raw = _write_raw_ibtracs(tmp_path / "ibtracs.csv", [("TEST0001", "NA")])
    rows = [_default_row("TEST0001", "1985-06-06 00:00:00", basin="")]  # empty, like the real bug

    final_dir, _, _ = _run_build(tmp_path, rows, raw_ibtracs_csv=raw)

    index = json.loads((final_dir / "events_index.json").read_text(encoding="utf-8"))
    entry = next(e for e in index if e["sid"] == "TEST0001")
    assert entry["basin"] == "NA"


def test_build_events_keeps_existing_basin_when_present(tmp_path):
    raw = _write_raw_ibtracs(tmp_path / "ibtracs.csv", [("TEST0002", "WRONG")])
    rows = [_default_row("TEST0002", "1985-06-06 00:00:00", basin="EP")]

    final_dir, _, _ = _run_build(tmp_path, rows, raw_ibtracs_csv=raw)

    index = json.loads((final_dir / "events_index.json").read_text(encoding="utf-8"))
    entry = next(e for e in index if e["sid"] == "TEST0002")
    assert entry["basin"] == "EP"


def test_build_events_basin_stays_empty_when_raw_file_absent(tmp_path):
    """No raw IBTrACS file available -> fix disabled gracefully, existing
    (empty) augmented-CSV basin is preserved as-is, no crash."""
    rows = [_default_row("TEST0003", "1985-06-06 00:00:00", basin="")]
    final_dir, _, _ = _run_build(tmp_path, rows)  # default raw_ibtracs_csv -> nonexistent

    index = json.loads((final_dir / "events_index.json").read_text(encoding="utf-8"))
    entry = next(e for e in index if e["sid"] == "TEST0003")
    assert entry["basin"] is None or entry["basin"] == ""


# ---------------------------------------------------------------------
# definitions.json basin_names map (frontend lookup source)
# ---------------------------------------------------------------------

@pytest.mark.parametrize("with_env", [False, True])
def test_definitions_basin_names_present(tmp_path, with_env):
    rows = [_default_row("TEST0001", "1985-06-06 00:00:00")]
    final_dir, _, _ = _run_build(tmp_path, rows, with_env=with_env)

    defs = json.loads((final_dir / "definitions.json").read_text(encoding="utf-8"))
    assert "basin_names" in defs
    expected = {
        "NA": "North Atlantic",
        "EP": "Eastern North Pacific",
        "WP": "Western North Pacific",
        "NI": "North Indian",
        "SI": "South Indian",
        "SP": "South Pacific",
        "SA": "South Atlantic",
    }
    assert defs["basin_names"] == expected
