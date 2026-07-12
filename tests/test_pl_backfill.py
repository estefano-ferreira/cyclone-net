"""Tests for the pressure-level (PL) channel backfill pipeline.

Synthetic tmp_path fixtures only -- no network, no real data/ directory is
touched. ``ensure_pl_raw_for_window`` (the only function that would need
real CDS credentials / network access) is always monkeypatched to a no-op;
the required ``era5pl_wind_*``/``era5pl_rh_*`` files are placed directly on
disk by each test instead.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline import pl_backfill
from src.processors import pressure_channels

BASE_CHANNELS = [
    "sst_K", "mslp_Pa", "u10_mps", "v10_mps",
    "wind_mps", "vort_1ps", "div_1ps", "grad_mslp_Pa_per_m", "sst_anom_K",
    "latent_heat_flux_Wpm2", "sensible_heat_flux_Wpm2", "total_heat_flux_Wpm2",
]
BASE_UNITS = {
    "sst_K": "K", "mslp_Pa": "Pa", "u10_mps": "m s-1", "v10_mps": "m s-1",
    "wind_mps": "m s-1", "vort_1ps": "s-1", "div_1ps": "s-1",
    "grad_mslp_Pa_per_m": "Pa m-1", "sst_anom_K": "K",
    "latent_heat_flux_Wpm2": "W m-2", "sensible_heat_flux_Wpm2": "W m-2",
    "total_heat_flux_Wpm2": "W m-2",
}
H, W, T = 6, 6, 5  # small synthetic spatial/time dims


def _cfg(tmp_path: Path) -> dict:
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    results = tmp_path / "outputs" / "results"
    raw.mkdir(parents=True, exist_ok=True)
    interim.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return {
        "paths": {
            "raw_data": str(raw),
            "interim_data": str(interim),
            "results_dir": str(results),
            "event_list": str(tmp_path / "event_list.csv"),
        },
        "data": {"window_size_px": H, "offsets_hours": [0, -6, -12, -18, -24]},
        "download": {
            "pressure_levels": {
                "enabled": True,
                "wind_levels": [850, 200],
                "rh_levels": [700, 600, 500],
            }
        },
    }


def _write_event(interim_dir: Path, event_id: str, channels, units, cube: np.ndarray,
                 timestamp: str = "1985-06-06 18:00", sid: str = "1985157N16259") -> None:
    npy_path = interim_dir / f"{event_id}.npy"
    json_path = interim_dir / f"{event_id}.json"
    np.save(npy_path, cube.astype(np.float32))
    meta = {
        "event_id": event_id,
        "sid": sid,
        "timestamp": timestamp,
        "storm_name": "TESTSTORM",
        "basin": "AL",
        "ri_label": 0,
        "dv12_kt": 5.0,
        "dv24_kt": -5.0,
        "wind_kt": 50.0,
        "pressure_mb": None,
        "center_lat": 16.0,
        "center_lon": -80.0,
        "timestamps": [timestamp],
        "cube_shape": list(cube.shape),
        "channels": list(channels),
        "units": dict(units),
        "qc_flags": {},
        "source_files": ["era5_1985_06.nc"],
        "era5_time_name": "valid_time",
        "era5_selected_times": [timestamp],
        "era5_time_indices": [0],
        "temporal_integrity_ok": True,
        "fuel_potential_saved": True,
    }
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _fake_extract_ones(*_args, **_kwargs):
    vol = np.ones((H, W, T, 2), dtype=np.float32)
    return vol, [pressure_channels.SHEAR_CHANNEL, pressure_channels.RH_CHANNEL], dict(pressure_channels.PL_UNITS)


@pytest.fixture
def no_network(monkeypatch):
    """Stub out the only function that would reach real CDS credentials/network.

    NOT autouse: ``test_ensure_pl_raw_for_window_does_not_mutate_cfg`` below
    exercises the real ``ensure_pl_raw_for_window`` (with a fake downloader
    class instead), so it must not request this fixture.
    """
    monkeypatch.setattr(pl_backfill, "ensure_pl_raw_for_window", lambda *a, **kw: None)


def test_idempotent_skip_leaves_event_untouched(tmp_path, no_network):
    cfg = _cfg(tmp_path)
    interim = Path(cfg["paths"]["interim_data"])
    channels = BASE_CHANNELS + [pressure_channels.SHEAR_CHANNEL, pressure_channels.RH_CHANNEL]
    units = dict(BASE_UNITS)
    units.update(pressure_channels.PL_UNITS)
    cube = np.random.rand(H, W, T, 14).astype(np.float32)
    event_id = "era5_1985_06_06_1800_1985157N16259"
    _write_event(interim, event_id, channels, units, cube)

    npy_path = interim / f"{event_id}.npy"
    json_path = interim / f"{event_id}.json"
    before_npy = npy_path.read_bytes()
    before_json = json_path.read_bytes()

    manifest = pl_backfill.backfill_window(cfg, 1985, 1985)

    assert npy_path.read_bytes() == before_npy
    assert json_path.read_bytes() == before_json
    assert manifest["status"] == "completed"
    assert manifest["outcome_counts"] == {"skipped_already_present": 1}

    df = pd.read_csv(manifest["per_event_csv"])
    assert df.loc[df["event_id"] == event_id, "outcome"].iloc[0] == "skipped_already_present"


def test_atomicity_on_extract_exception(tmp_path, monkeypatch, no_network):
    cfg = _cfg(tmp_path)
    interim = Path(cfg["paths"]["interim_data"])
    cube = np.random.rand(H, W, T, 12).astype(np.float32)
    event_id = "era5_1985_06_06_1800_1985157N16259"
    _write_event(interim, event_id, BASE_CHANNELS, BASE_UNITS, cube)

    npy_path = interim / f"{event_id}.npy"
    json_path = interim / f"{event_id}.json"
    before_npy = npy_path.read_bytes()
    before_json = json_path.read_bytes()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("forced failure mid-window")

    monkeypatch.setattr(pl_backfill, "extract_pressure_volume", _boom)

    manifest = pl_backfill.backfill_window(cfg, 1985, 1985)

    assert npy_path.read_bytes() == before_npy
    assert json_path.read_bytes() == before_json
    assert manifest["outcome_counts"] == {"failed": 1}
    # No appended events -> verification trivially passes -> window completes,
    # but there is nothing to delete and no deletion record for PL raw.
    assert manifest["status"] == "completed"
    assert manifest["deletion"]["deleted_files"] == []
    assert manifest["deletion"]["freed_bytes"] == 0

    df = pd.read_csv(manifest["per_event_csv"])
    row = df.loc[df["event_id"] == event_id].iloc[0]
    assert row["outcome"] == "failed"


def test_append_correctness(tmp_path, monkeypatch, no_network):
    cfg = _cfg(tmp_path)
    interim = Path(cfg["paths"]["interim_data"])
    cube = np.zeros((H, W, T, 12), dtype=np.float32)
    event_id = "era5_1985_06_06_1800_1985157N16259"
    _write_event(interim, event_id, BASE_CHANNELS, BASE_UNITS, cube)

    monkeypatch.setattr(pl_backfill, "extract_pressure_volume", _fake_extract_ones)

    manifest = pl_backfill.backfill_window(cfg, 1985, 1985)

    assert manifest["status"] == "completed"
    assert manifest["outcome_counts"] == {"appended": 1}

    new_cube = np.load(interim / f"{event_id}.npy")
    assert new_cube.shape == (H, W, T, 14)
    assert new_cube.dtype == np.float32
    assert np.all(new_cube[..., -2:] == 1.0)

    new_meta = json.loads((interim / f"{event_id}.json").read_text(encoding="utf-8"))
    assert new_meta["channels"][-2:] == [pressure_channels.SHEAR_CHANNEL, pressure_channels.RH_CHANNEL]
    assert new_meta["units"][pressure_channels.SHEAR_CHANNEL] == "m s-1"
    assert new_meta["units"][pressure_channels.RH_CHANNEL] == "%"
    assert new_meta["cube_shape"] == [H, W, T, 14]

    df = pd.read_csv(manifest["per_event_csv"])
    row = df.loc[df["event_id"] == event_id].iloc[0]
    assert row["outcome"] == "appended"
    assert row["n_channels_before"] == 12
    assert row["n_channels_after"] == 14
    assert row["npy_sha256_before"] != row["npy_sha256_after"]


def test_deletion_gating_on_verification_failure(tmp_path, monkeypatch, no_network):
    cfg = _cfg(tmp_path)
    interim = Path(cfg["paths"]["interim_data"])
    raw = Path(cfg["paths"]["raw_data"])
    cube = np.zeros((H, W, T, 12), dtype=np.float32)
    event_id = "era5_1985_06_06_1800_1985157N16259"
    _write_event(interim, event_id, BASE_CHANNELS, BASE_UNITS, cube)

    pl_wind_file = raw / "era5pl_wind_1985_06.nc"
    pl_rh_file = raw / "era5pl_rh_1985_06.nc"
    pl_wind_file.write_bytes(b"fake-pl-wind-netcdf")
    pl_rh_file.write_bytes(b"fake-pl-rh-netcdf")

    monkeypatch.setattr(pl_backfill, "extract_pressure_volume", _fake_extract_ones)
    monkeypatch.setattr(pl_backfill, "verify_appended_event",
                        lambda *a, **kw: (False, "forced verification failure"))

    manifest = pl_backfill.backfill_window(cfg, 1985, 1985)

    assert manifest["status"] == "verification_failed"
    assert manifest["verification"]["passed"] is False
    assert manifest["deletion"]["performed"] is False
    assert manifest["deletion"]["deleted_files"] == []
    assert pl_wind_file.exists()
    assert pl_rh_file.exists()


def test_deletion_scope_never_touches_other_files(tmp_path, monkeypatch, no_network):
    cfg = _cfg(tmp_path)
    interim = Path(cfg["paths"]["interim_data"])
    raw = Path(cfg["paths"]["raw_data"])
    cube = np.zeros((H, W, T, 12), dtype=np.float32)
    event_id = "era5_1985_06_06_1800_1985157N16259"
    _write_event(interim, event_id, BASE_CHANNELS, BASE_UNITS, cube)

    surface_file = raw / "era5_1985_06.nc"  # surface raw of the SAME year -- must survive
    other_year_pl_wind = raw / "era5pl_wind_2020_08.nc"  # PL raw of ANOTHER year -- must survive
    other_year_pl_rh = raw / "era5pl_rh_2020_08.nc"
    this_window_pl_wind = raw / "era5pl_wind_1985_06.nc"  # this window's own PL raw -- must be deleted
    this_window_pl_rh = raw / "era5pl_rh_1985_06.nc"

    for f in (surface_file, other_year_pl_wind, other_year_pl_rh, this_window_pl_wind, this_window_pl_rh):
        f.write_bytes(b"placeholder-bytes")

    monkeypatch.setattr(pl_backfill, "extract_pressure_volume", _fake_extract_ones)

    manifest = pl_backfill.backfill_window(cfg, 1985, 1985)

    assert manifest["status"] == "completed"
    assert surface_file.exists()
    assert other_year_pl_wind.exists()
    assert other_year_pl_rh.exists()
    assert not this_window_pl_wind.exists()
    assert not this_window_pl_rh.exists()
    assert set(manifest["deletion"]["deleted_files"]) == {"era5pl_wind_1985_06.nc", "era5pl_rh_1985_06.nc"}


def test_ensure_pl_raw_for_window_does_not_mutate_cfg(tmp_path, monkeypatch):
    """Year-scoping must happen on an in-memory cfg COPY -- never mutate the
    caller's cfg dict or write to config.yaml on disk."""
    import src.downloaders.era5_pressure as era5_pressure_module

    cfg = _cfg(tmp_path)
    event_list = Path(cfg["paths"]["event_list"])
    pd.DataFrame({
        "timestamp": ["1985-06-06 18:00"],
        "sid": ["1985157N16259"],
        "lat": [16.0],
        "lon": [-80.0],
    }).to_csv(event_list, index=False)

    original_cfg_paths_event_list = cfg["paths"]["event_list"]
    original_download = json.loads(json.dumps(cfg["download"]))  # deep copy for comparison

    captured = {}

    class _FakeDownloader:
        def __init__(self, cfg_win):
            captured["years"] = cfg_win["download"].get("years")
            captured["event_list"] = cfg_win["paths"].get("event_list")

        def download_required_batch(self):
            captured["called"] = True

    monkeypatch.setattr(era5_pressure_module, "ERA5PressureDownloader", _FakeDownloader)

    prov_dir = pl_backfill._provenance_dir(cfg)
    pl_backfill.ensure_pl_raw_for_window(cfg, 1985, 1986, prov_dir)

    assert captured["called"] is True
    assert captured["years"] == [1985, 1986]
    assert captured["event_list"] != original_cfg_paths_event_list  # scoped to a window-local CSV
    assert cfg["paths"]["event_list"] == original_cfg_paths_event_list  # caller's cfg untouched
    assert cfg["download"] == original_download  # caller's cfg untouched
