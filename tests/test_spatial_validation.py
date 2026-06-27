"""
Tests for TCHP spatial validation: the physics-flux peak extraction and the
three-way skill comparison (storm-centre vs physics-only flux vs learned model).
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.spatial_metrics import (
    compute_spatial_metrics_from_predictions,
    physics_flux_peak,
    robust_peak_2d,
)

CHANNELS = [
    "sst_K", "mslp_Pa", "u10_mps", "v10_mps", "wind_mps", "vort_1ps",
    "div_1ps", "grad_mslp_Pa_per_m", "sst_anom_K", "total_heat_flux_Wpm2",
]
H = W = 20
T = 5


def _write_event(interim, eid, center, tchp_peak, flux_peak_ij):
    """Create a synthetic cube+meta+grids whose flux channel peaks at flux_peak_ij."""
    lat1d = 10.0 + 0.25 * np.arange(H)
    lon1d = -60.0 + 0.25 * np.arange(W)
    lons, lats = np.meshgrid(lon1d, lat1d)
    np.save(interim / f"{eid}_lats.npy", lats.astype(np.float32))
    np.save(interim / f"{eid}_lons.npy", lons.astype(np.float32))

    cube = np.random.default_rng(0).normal(size=(H, W, T, len(CHANNELS))).astype(np.float32)
    flux = np.zeros((H, W), dtype=np.float32)
    pi, pj = flux_peak_ij
    flux[pi, pj] = 500.0  # strong, unambiguous peak
    cube[:, :, 0, CHANNELS.index("total_heat_flux_Wpm2")] = flux
    np.save(interim / f"{eid}.npy", cube)

    meta = {
        "event_id": eid, "channels": CHANNELS,
        "center_lat": center[0], "center_lon": center[1],
        "tchp_peak_lat": tchp_peak[0], "tchp_peak_lon": tchp_peak[1],
        "tchp_audit": {"status": "ok"}, "ri_label": 1,
    }
    (interim / f"{eid}.json").write_text(json.dumps(meta), encoding="utf-8")
    return lats, lons


def test_robust_peak_2d_finds_injected_peak():
    lat1d = 10.0 + 0.25 * np.arange(H)
    lon1d = -60.0 + 0.25 * np.arange(W)
    lons, lats = np.meshgrid(lon1d, lat1d)
    field = np.zeros((H, W))
    field[5, 8] = 100.0
    lat, lon, val = robust_peak_2d(field, lats, lons, smooth_sigma=0.5)
    assert lat == pytest.approx(lats[5, 8], abs=0.26)
    assert lon == pytest.approx(lons[5, 8], abs=0.26)


def test_physics_flux_peak_reads_flux_channel(tmp_path):
    lats, lons = _write_event(tmp_path, "era5_2020_09_01_0000",
                              center=(12.0, -57.0), tchp_peak=(13.0, -56.0),
                              flux_peak_ij=(4, 9))
    peak = physics_flux_peak("era5_2020_09_01_0000", tmp_path)
    assert peak is not None
    lat, lon, val = peak
    assert lat == pytest.approx(lats[4, 9], abs=0.26)
    assert lon == pytest.approx(lons[4, 9], abs=0.26)


def test_three_way_skill_comparison_present(tmp_path):
    _write_event(tmp_path, "era5_2020_09_01_0000",
                 center=(12.0, -57.0), tchp_peak=(13.0, -56.0), flux_peak_ij=(12, 4))
    _write_event(tmp_path, "era5_2020_09_02_0000",
                 center=(20.0, -50.0), tchp_peak=(20.5, -49.5), flux_peak_ij=(2, 18))
    pred_df = pd.DataFrame([
        {"event_id": "era5_2020_09_01_0000", "pred_lat": 13.1, "pred_lon": -56.1, "y_true": 1},
        {"event_id": "era5_2020_09_02_0000", "pred_lat": 20.6, "pred_lon": -49.4, "y_true": 1},
    ])
    res = compute_spatial_metrics_from_predictions(pred_df, interim_dir=tmp_path)
    assert res["status"] == "ok"
    assert res["coverage"]["n_with_physics_flux"] == 2
    skill = res["skill_comparison"]
    assert skill["status"] == "ok"
    for key in ("model_mean_km", "center_baseline_mean_km", "physics_flux_mean_km",
                "model_minus_center_km", "model_minus_physics_km"):
        assert key in skill
    # distances are non-negative
    assert res["distance_physics_flux_to_tchp"]["mean_km"] >= 0.0
