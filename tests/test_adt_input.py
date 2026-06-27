"""
Tests for ADT as an appended model input channel.

Guards: (1) enabling model.use_adt_input adds exactly one channel; (2) covered
events carry the ADT signal with adt_mask=1; (3) uncovered events get a neutral,
masked channel (adt_mask=0) so the full archive still trains.
"""
import json

import numpy as np
import torch

from src.data.dataset import PhysicsDataset

NAMES = ["sst_K", "mslp_Pa", "u10_mps", "v10_mps", "wind_mps",
         "vort_1ps", "div_1ps", "grad_mslp_Pa_per_m", "sst_anom_K"]
H = W = 8
T = 5


def _make_event(interim, eid, with_adt: bool):
    cube = np.random.default_rng(0).normal(size=(H, W, T, len(NAMES))).astype(np.float32)
    np.save(interim / f"{eid}.npy", cube)
    lat = (10.0 + 0.25 * np.arange(H)).reshape(-1, 1).repeat(W, 1)
    lon = (-60.0 + 0.25 * np.arange(W)).reshape(1, -1).repeat(H, 0)
    np.save(interim / f"{eid}_lats.npy", lat.astype(np.float32))
    np.save(interim / f"{eid}_lons.npy", lon.astype(np.float32))
    meta = {"event_id": eid, "channels": NAMES, "ri_label": 1,
            "dv12_kt": 10.0, "dv24_kt": 20.0}
    (interim / f"{eid}.json").write_text(json.dumps(meta), encoding="utf-8")
    if with_adt:
        np.save(interim / f"{eid}_adt.npy", np.full((H, W), 0.6, dtype=np.float32))


def _cfg(interim, splits, stats):
    return {
        "paths": {"interim_data": str(interim), "splits_csv": str(splits),
                  "normalization_stats": str(stats)},
        "model": {"input_channels_names": NAMES, "use_adt_input": True},
        "physics_guided": {"losses": {"exclude_total_heat_flux_from_input": True,
                                      "total_heat_flux_channel_name": "total_heat_flux_Wpm2"}},
        "splits": {"seed": 1337},
    }


def _setup(tmp_path):
    interim = tmp_path / "interim"
    interim.mkdir()
    _make_event(interim, "era5_2023_09_01_0000", with_adt=True)
    _make_event(interim, "era5_2020_09_01_0000", with_adt=False)
    splits = tmp_path / "splits.csv"
    splits.write_text("event_id,split\nera5_2023_09_01_0000,train\nera5_2020_09_01_0000,train\n",
                      encoding="utf-8")
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({
        "channels": NAMES, "mean": [0.0] * len(NAMES), "std": [1.0] * len(NAMES),
        "adt_mean": 0.5, "adt_std": 0.2,
    }), encoding="utf-8")
    return _cfg(interim, splits, stats)


def test_adt_adds_exactly_one_channel(tmp_path):
    cfg = _setup(tmp_path)
    ds = PhysicsDataset(cfg, split="train")
    x = ds[0]["x"]
    assert x.shape[0] == len(NAMES) + 1  # 9 named + 1 ADT


def test_covered_event_has_mask_one_and_signal(tmp_path):
    cfg = _setup(tmp_path)
    ds = PhysicsDataset(cfg, split="train")
    items = {it["event_id"]: it for it in (ds[0], ds[1])}
    covered = items["era5_2023_09_01_0000"]
    uncovered = items["era5_2020_09_01_0000"]

    assert float(covered["adt_mask"]) == 1.0
    assert float(uncovered["adt_mask"]) == 0.0

    # Covered ADT channel = (0.6 - 0.5)/0.2 = 0.5 everywhere; uncovered = 0 (neutral).
    assert torch.allclose(covered["x"][-1], torch.full_like(covered["x"][-1], 0.5), atol=1e-5)
    assert torch.allclose(uncovered["x"][-1], torch.zeros_like(uncovered["x"][-1]))


def test_disabling_adt_keeps_nine_channels(tmp_path):
    cfg = _setup(tmp_path)
    cfg["model"]["use_adt_input"] = False
    ds = PhysicsDataset(cfg, split="train")
    assert ds[0]["x"].shape[0] == len(NAMES)
