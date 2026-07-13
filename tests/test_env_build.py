"""Tests for the --with-env feature of platform/build/build_events.py.

Entirely synthetic tmp_path fixtures: fake event_list_augmented.csv, fake
config.yaml, fake data/interim cubes (tiny 4x4x2xC) + metadata json. No real
data/ directory is read (`platform/build/build_events.py::build_events` takes
overridable path parameters precisely so tests never need to touch it) --
see CLAUDE.md / the task brief: a live backfill is rewriting the real
data/interim/ via atomic os.replace and must never be raced by a test.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "platform" / "build"))
import build_events as be  # noqa: E402  (path must be extended first)


FIELDNAMES = [
    "sid", "storm_name", "name", "basin", "timestamp", "lat", "lon",
    "wind_kt", "pressure_mb", "dv12_kt", "dv24_kt", "ri_label",
]


def _default_row(sid, timestamp, ri_label=0, lat=16.0, lon=-80.0):
    return {
        "sid": sid, "storm_name": "TESTSTORM", "name": "TESTSTORM", "basin": "AL",
        "timestamp": timestamp, "lat": lat, "lon": lon,
        "wind_kt": 50.0, "pressure_mb": 995.0,
        "dv12_kt": 5.0, "dv24_kt": 10.0, "ri_label": ri_label,
    }


def _write_event_list(root: Path, rows) -> Path:
    csv_path = root / "event_list_augmented.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def _write_config(root: Path, interim_dir: Path, window_size_px: int = 4) -> Path:
    cfg = {
        "labels": {"ri_threshold_kt_24h": 30.0},
        "paths": {"interim_data": str(interim_dir)},
        "data": {"window_size_px": window_size_px},
    }
    cfg_path = root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _write_cube(interim_dir: Path, event_id: str, channels, values: dict,
                timestamp: str, sid: str) -> None:
    """values: {channel_name: scalar_fill_value or np.nan} for t0 (index 0);
    channels not present in `values` are left as zeros."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    H, W, T, C = 4, 4, 2, len(channels)
    cube = np.zeros((H, W, T, C), dtype=np.float32)
    for name, val in values.items():
        cube[:, :, 0, channels.index(name)] = val
    np.save(interim_dir / f"{event_id}.npy", cube)

    meta = {
        "event_id": event_id,
        "sid": sid,
        "timestamp": timestamp,
        "channels": list(channels),
        "cube_shape": list(cube.shape),
    }
    (interim_dir / f"{event_id}.json").write_text(json.dumps(meta), encoding="utf-8")


def _run_build(root: Path, rows, with_env: bool, interim_dir: Path = None, window_size_px: int = 4,
               raw_ibtracs_csv: Path = None):
    interim_dir = interim_dir or (root / "interim")
    # Point at a deliberately nonexistent file by default so these tests
    # never depend on (or are affected by) whatever happens to sit at the
    # real project's data/raw/ibtracs.ALL.list.v04r00.csv path.
    raw_ibtracs_csv = raw_ibtracs_csv or (root / "no_such_ibtracs.csv")
    augmented_csv = _write_event_list(root, rows)
    config_file = _write_config(root, interim_dir, window_size_px=window_size_px)
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
# env_event_id / compute_env_values unit tests
# ---------------------------------------------------------------------

def test_env_event_id_matches_preprocess_convention():
    ts = pd.Timestamp("1985-06-06 18:00:00")
    assert be.env_event_id("1985157N16259", ts) == "era5_1985_06_06_1800_1985157N16259"


def test_compute_env_values_missing_cube_returns_all_none(tmp_path):
    result = be.compute_env_values(tmp_path, "era5_1985_06_06_0000_NOPE")
    assert result == {"env_sst_c": None, "env_shear_mps": None, "env_rh_pct": None}


def test_compute_env_values_correct_patch_means_and_celsius(tmp_path):
    channels = ["sst_K", "mslp_Pa", "u10_mps", "v10_mps", "shear_850_200_mps", "rh_mid"]
    event_id = "era5_1985_06_06_0000_TEST0001"
    _write_cube(tmp_path, event_id, channels,
                values={"sst_K": 300.0, "shear_850_200_mps": 12.345, "rh_mid": 55.555},
                timestamp="1985-06-06 00:00", sid="TEST0001")

    result = be.compute_env_values(tmp_path, event_id)
    assert result["env_sst_c"] == pytest.approx(26.85, abs=1e-9)  # 300 - 273.15
    assert result["env_shear_mps"] == 12.35  # rounded to 2 decimals
    assert result["env_rh_pct"] == 55.56  # rounded to 2 decimals (55.555 -> 55.56 via round-half-even or up)


def test_compute_env_values_null_for_missing_channel(tmp_path):
    channels = ["sst_K", "mslp_Pa", "u10_mps", "v10_mps"]  # no shear/rh channels
    event_id = "era5_1985_06_06_0000_TEST0002"
    _write_cube(tmp_path, event_id, channels, values={"sst_K": 300.0},
                timestamp="1985-06-06 00:00", sid="TEST0002")

    result = be.compute_env_values(tmp_path, event_id)
    assert result["env_sst_c"] == pytest.approx(26.85, abs=1e-9)
    assert result["env_shear_mps"] is None
    assert result["env_rh_pct"] is None


def test_compute_env_values_null_for_nan_patch(tmp_path):
    channels = ["sst_K", "mslp_Pa", "u10_mps", "v10_mps", "shear_850_200_mps", "rh_mid"]
    event_id = "era5_1985_06_06_0000_TEST0003"
    _write_cube(tmp_path, event_id, channels,
                values={"sst_K": np.nan, "shear_850_200_mps": 5.0, "rh_mid": 60.0},
                timestamp="1985-06-06 00:00", sid="TEST0003")

    result = be.compute_env_values(tmp_path, event_id)
    assert result["env_sst_c"] is None  # whole-patch NaN sst -> mean is NaN -> null
    assert result["env_shear_mps"] == 5.0
    assert result["env_rh_pct"] == 60.0


def test_compute_env_values_null_when_json_missing_but_npy_present(tmp_path):
    event_id = "era5_1985_06_06_0000_TEST0004"
    np.save(tmp_path / f"{event_id}.npy", np.zeros((4, 4, 2, 4), dtype=np.float32))
    result = be.compute_env_values(tmp_path, event_id)
    assert result == {"env_sst_c": None, "env_shear_mps": None, "env_rh_pct": None}


# ---------------------------------------------------------------------
# Full build_events() integration: --with-env off vs on
# ---------------------------------------------------------------------

def test_with_env_off_produces_no_env_keys_and_no_definitions_block(tmp_path):
    rows = [_default_row("TEST0001", "1985-06-06 00:00:00")]
    root = tmp_path / "off"
    root.mkdir()
    final_dir, manifest, _artifacts = _run_build(root, rows, with_env=False)

    geo_bytes = (final_dir / "events" / "TEST0001.geojson").read_bytes()
    assert b"env_" not in geo_bytes

    defs = json.loads((final_dir / "definitions.json").read_text(encoding="utf-8"))
    assert "env" not in defs


def test_with_env_off_is_byte_identical_across_two_runs(tmp_path):
    """No --with-env => build output must not depend on whether interim/
    cubes happen to exist on disk at all -- prove two independent builds
    (one with a populated interim dir sitting right next to the CSV, one
    without) produce byte-identical geojson/definitions."""
    rows = [_default_row("TEST0001", "1985-06-06 00:00:00")]

    root_a = tmp_path / "a"
    root_a.mkdir()
    final_dir_a, _, _ = _run_build(root_a, rows, with_env=False)

    root_b = tmp_path / "b"
    root_b.mkdir()
    interim_b = root_b / "interim"
    channels = ["sst_K", "mslp_Pa", "u10_mps", "v10_mps", "shear_850_200_mps", "rh_mid"]
    _write_cube(interim_b, "era5_1985_06_06_0000_TEST0001", channels,
                values={"sst_K": 300.0, "shear_850_200_mps": 10.0, "rh_mid": 50.0},
                timestamp="1985-06-06 00:00", sid="TEST0001")
    final_dir_b, _, _ = _run_build(root_b, rows, with_env=False, interim_dir=interim_b)

    geo_a = (final_dir_a / "events" / "TEST0001.geojson").read_bytes()
    geo_b = (final_dir_b / "events" / "TEST0001.geojson").read_bytes()
    assert geo_a == geo_b

    defs_a = (final_dir_a / "definitions.json").read_bytes()
    defs_b = (final_dir_b / "definitions.json").read_bytes()
    assert defs_a == defs_b


def test_with_env_on_embeds_correct_values_in_geojson(tmp_path):
    root = tmp_path / "on"
    root.mkdir()
    interim_dir = root / "interim"
    rows = [_default_row("TEST0001", "1985-06-06 00:00:00")]

    channels = ["sst_K", "mslp_Pa", "u10_mps", "v10_mps", "shear_850_200_mps", "rh_mid"]
    _write_cube(interim_dir, "era5_1985_06_06_0000_TEST0001", channels,
                values={"sst_K": 301.2, "shear_850_200_mps": 8.4, "rh_mid": 62.0},
                timestamp="1985-06-06 00:00", sid="TEST0001")

    final_dir, manifest, artifacts = _run_build(root, rows, with_env=True, interim_dir=interim_dir)

    geojson = json.loads((final_dir / "events" / "TEST0001.geojson").read_text(encoding="utf-8"))
    points = [f for f in geojson["features"] if f["geometry"]["type"] == "Point"]
    assert len(points) == 1
    props = points[0]["properties"]
    assert props["env_sst_c"] == pytest.approx(28.05, abs=1e-9)  # 301.2 - 273.15
    assert props["env_shear_mps"] == 8.4
    assert props["env_rh_pct"] == 62.0
    # Existing properties untouched.
    assert props["pressure_mb"] == 995.0

    # Manifest still covers the geojson (no new artifact type introduced).
    assert "events/TEST0001.geojson" in artifacts


def test_with_env_on_null_when_cube_absent_for_one_point(tmp_path):
    root = tmp_path / "partial"
    root.mkdir()
    interim_dir = root / "interim"
    rows = [
        _default_row("TEST0001", "1985-06-06 00:00:00"),
        _default_row("TEST0001", "1985-06-06 06:00:00"),
    ]
    channels = ["sst_K", "mslp_Pa", "u10_mps", "v10_mps"]
    # Only the FIRST timestep's cube exists.
    _write_cube(interim_dir, "era5_1985_06_06_0000_TEST0001", channels,
                values={"sst_K": 300.0}, timestamp="1985-06-06 00:00", sid="TEST0001")

    final_dir, _, _ = _run_build(root, rows, with_env=True, interim_dir=interim_dir)
    geojson = json.loads((final_dir / "events" / "TEST0001.geojson").read_text(encoding="utf-8"))
    points = [f for f in geojson["features"] if f["geometry"]["type"] == "Point"]
    assert points[0]["properties"]["env_sst_c"] == pytest.approx(26.85, abs=1e-9)
    assert points[1]["properties"]["env_sst_c"] is None
    assert points[1]["properties"]["env_shear_mps"] is None
    assert points[1]["properties"]["env_rh_pct"] is None


def test_with_env_on_adds_definitions_env_block(tmp_path):
    root = tmp_path / "defs"
    root.mkdir()
    rows = [_default_row("TEST0001", "1985-06-06 00:00:00")]
    final_dir, _, _ = _run_build(root, rows, with_env=True)

    defs = json.loads((final_dir / "definitions.json").read_text(encoding="utf-8"))
    assert "env" in defs
    env = defs["env"]
    for key in ("source", "spatial_definition", "temporal_definition", "properties",
                "null_meaning", "epistemic_note"):
        assert key in env, f"definitions.env missing '{key}'"
    assert set(env["properties"].keys()) == {"env_sst_c", "env_shear_mps", "env_rh_pct"}
    # Epistemic honesty guardrail: no causal claim in the definitions text either.
    blob = json.dumps(env).lower()
    for forbidden in ("favorable", "driving", "fueling", "conducive", "explains"):
        assert forbidden not in blob


def test_with_env_rounds_to_two_decimals(tmp_path):
    root = tmp_path / "round"
    root.mkdir()
    interim_dir = root / "interim"
    rows = [_default_row("TEST0001", "1985-06-06 00:00:00")]
    channels = ["sst_K", "mslp_Pa", "u10_mps", "v10_mps", "shear_850_200_mps", "rh_mid"]
    _write_cube(interim_dir, "era5_1985_06_06_0000_TEST0001", channels,
                values={"sst_K": 300.123456, "shear_850_200_mps": 9.999, "rh_mid": 70.001},
                timestamp="1985-06-06 00:00", sid="TEST0001")

    final_dir, _, _ = _run_build(root, rows, with_env=True, interim_dir=interim_dir)
    geojson = json.loads((final_dir / "events" / "TEST0001.geojson").read_text(encoding="utf-8"))
    props = geojson["features"][-1]["properties"]
    for key in ("env_sst_c", "env_shear_mps", "env_rh_pct"):
        value = props[key]
        assert value == round(value, 2)
