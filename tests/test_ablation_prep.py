"""Tests for the PL-channel gate census and the ablation-tooling upgrades
(feature_ablation_kfold gate + cluster bootstrap, ri_precursors --pairs-csv).

Synthetic tmp_path fixtures only -- no network, no real data/ directory is
touched. Tiny fake metadata JSONs, splits/valid_events CSVs, and census
JSONs are written directly under tmp_path by each test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import pl_gate_census
from analysis.feature_ablation_kfold import cluster_bootstrap_ci, enforce_gate, load_gate_census
from analysis.ri_precursors import REQUIRED_PAIRS_CSV_COLUMNS, load_pairs_from_csv

PL_CHANNELS = ["shear_850_200_mps", "rh_mid"]
SURFACE_CHANNELS = [
    "sst_K", "mslp_Pa", "u10_mps", "v10_mps", "wind_mps",
    "vort_1ps", "div_1ps", "grad_mslp_Pa_per_m", "sst_anom_K",
    "latent_heat_flux_Wpm2", "sensible_heat_flux_Wpm2", "total_heat_flux_Wpm2",
]


# ---------------------------------------------------------------------------
# Shared synthetic-project fixture builder.
# ---------------------------------------------------------------------------

def _make_cfg(tmp_path: Path) -> dict:
    interim = tmp_path / "interim"
    normalized = tmp_path / "normalized"
    results = tmp_path / "results"
    interim.mkdir(parents=True, exist_ok=True)
    normalized.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return {
        "paths": {
            "interim_data": str(interim),
            "normalized_dir": str(normalized),
            "results_dir": str(results),
        }
    }


def _write_meta(interim: Path, event_id: str, channels: list[str]) -> None:
    meta = {"event_id": event_id, "channels": list(channels)}
    (interim / f"{event_id}.json").write_text(json.dumps(meta), encoding="utf-8")


def _write_splits_and_valid_events(normalized: Path, rows: list[dict]) -> None:
    """rows: list of {event_id, sid, ri_label, split}."""
    df = pd.DataFrame(rows)
    df[["event_id", "split"]].to_csv(normalized / "splits.csv", index=False)
    df[["event_id", "sid", "ri_label"]].to_csv(normalized / "valid_events.csv", index=False)


# ---------------------------------------------------------------------------
# 1) Census counting logic: mixed 12/14-channel events across splits.
# ---------------------------------------------------------------------------

def test_census_counts_mixed_channel_events_across_splits(tmp_path):
    cfg = _make_cfg(tmp_path)
    interim = Path(cfg["paths"]["interim_data"])
    normalized = Path(cfg["paths"]["normalized_dir"])

    rows = []
    # 2 train events WITH pl channels (14 channels).
    for i in range(2):
        eid = f"era5_1985_06_0{i+1}_0000_SID000{i}"
        _write_meta(interim, eid, SURFACE_CHANNELS + PL_CHANNELS)
        rows.append({"event_id": eid, "sid": f"SID000{i}", "ri_label": 0, "split": "train"})
    # 1 train event WITHOUT pl channels (12 channels) -- gate-failing.
    eid_bad = "era5_1985_06_03_0000_SID0002"
    _write_meta(interim, eid_bad, SURFACE_CHANNELS)
    rows.append({"event_id": eid_bad, "sid": "SID0002", "ri_label": 1, "split": "train"})
    # 1 val event WITH pl channels.
    eid_val = "era5_1986_07_01_0000_SID0003"
    _write_meta(interim, eid_val, SURFACE_CHANNELS + PL_CHANNELS)
    rows.append({"event_id": eid_val, "sid": "SID0003", "ri_label": 0, "split": "val"})
    # 1 test event -- metadata deliberately NOT written; census must never open it.
    eid_test = "era5_1987_08_01_0000_SID0004"
    rows.append({"event_id": eid_test, "sid": "SID0004", "ri_label": 0, "split": "test"})

    _write_splits_and_valid_events(normalized, rows)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("placeholder: true", encoding="utf-8")
    census = pl_gate_census.build_census(cfg, config_path)

    assert census["total_events"] == 5
    assert census["counts_per_split"] == {"train": 3, "val": 1, "test": 1}
    assert census["dev_total_events"] == 4  # train + val only
    assert census["dev_events_with_pl"] == 3
    assert census["dev_events_without_pl"] == 1
    assert census["dev_missing_artifact"] == 0
    assert census["gate_pass"] is False
    assert eid_bad in census["offending_event_ids_sample"]
    # The test-split event must never have been opened (no artifact was even written).
    assert not (interim / f"{eid_test}.json").exists()

    per_year = census["per_year_per_split_dev_only"]
    assert per_year["1985"]["train"]["with_pl"] == 2
    assert per_year["1985"]["train"]["without_pl"] == 1
    assert per_year["1986"]["val"]["with_pl"] == 1


def test_census_gate_passes_when_all_dev_events_have_pl(tmp_path):
    cfg = _make_cfg(tmp_path)
    interim = Path(cfg["paths"]["interim_data"])
    normalized = Path(cfg["paths"]["normalized_dir"])

    rows = []
    for i, split in enumerate(["train", "train", "val"]):
        eid = f"era5_2021_08_0{i+1}_0000_SID100{i}"
        _write_meta(interim, eid, SURFACE_CHANNELS + PL_CHANNELS)
        rows.append({"event_id": eid, "sid": f"SID100{i}", "ri_label": 0, "split": split})
    _write_splits_and_valid_events(normalized, rows)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("placeholder: true", encoding="utf-8")
    census = pl_gate_census.build_census(cfg, config_path)

    assert census["gate_pass"] is True
    assert census["dev_events_without_pl"] == 0
    assert census["offending_event_ids_sample"] == []


def test_census_missing_artifact_counts_as_without_pl(tmp_path):
    cfg = _make_cfg(tmp_path)
    normalized = Path(cfg["paths"]["normalized_dir"])
    rows = [{"event_id": "era5_1990_09_01_0000_SIDMISS", "sid": "SIDMISS",
             "ri_label": 0, "split": "train"}]
    _write_splits_and_valid_events(normalized, rows)
    # No metadata JSON written at all for this event.

    config_path = tmp_path / "config.yaml"
    config_path.write_text("placeholder: true", encoding="utf-8")
    census = pl_gate_census.build_census(cfg, config_path)

    assert census["dev_missing_artifact"] == 1
    assert census["dev_events_without_pl"] == 1
    assert census["gate_pass"] is False


# ---------------------------------------------------------------------------
# 2) Gate refusal / acceptance in feature_ablation_kfold.enforce_gate.
# ---------------------------------------------------------------------------

def test_enforce_gate_refuses_when_census_missing(tmp_path, capsys):
    cfg = _make_cfg(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        enforce_gate(cfg, require_gate=True)
    assert exc_info.value.code == 1
    assert "not found" in capsys.readouterr().out


def test_enforce_gate_refuses_when_gate_pass_false(tmp_path, capsys):
    cfg = _make_cfg(tmp_path)
    results = Path(cfg["paths"]["results_dir"])
    (results / "pl_gate_census.json").write_text(
        json.dumps({"gate_pass": False, "gate_verdict": "FAIL: synthetic"}), encoding="utf-8")
    with pytest.raises(SystemExit) as exc_info:
        enforce_gate(cfg, require_gate=True)
    assert exc_info.value.code == 1
    assert "GATE FAIL" in capsys.readouterr().out


def test_enforce_gate_accepts_when_gate_pass_true(tmp_path, capsys):
    cfg = _make_cfg(tmp_path)
    results = Path(cfg["paths"]["results_dir"])
    (results / "pl_gate_census.json").write_text(
        json.dumps({"gate_pass": True, "generated_at": "2026-01-01T00:00:00Z"}), encoding="utf-8")
    enforce_gate(cfg, require_gate=True)  # must NOT raise / exit
    assert "GATE PASS" in capsys.readouterr().out


def test_enforce_gate_bypassed_with_require_gate_false(tmp_path, capsys):
    cfg = _make_cfg(tmp_path)  # no census file at all
    enforce_gate(cfg, require_gate=False)  # must NOT raise
    assert "SKIPPED" in capsys.readouterr().out


def test_load_gate_census_roundtrip(tmp_path):
    cfg = _make_cfg(tmp_path)
    assert load_gate_census(cfg) is None
    results = Path(cfg["paths"]["results_dir"])
    payload = {"gate_pass": True}
    (results / "pl_gate_census.json").write_text(json.dumps(payload), encoding="utf-8")
    assert load_gate_census(cfg) == payload


# ---------------------------------------------------------------------------
# 3) Cluster bootstrap: CI must contain the TRUE delta on synthetic cases
#    with known effect.
# ---------------------------------------------------------------------------

def test_cluster_bootstrap_ci_contains_zero_when_arms_are_identical():
    """Known true effect: delta == 0 exactly (B and A are the SAME scores)."""
    rng = np.random.default_rng(0)
    n_storms = 60
    groups = np.repeat(np.arange(n_storms), 3)  # 3 events per storm
    n = len(groups)
    # ~15% prevalence, storm-correlated (all of a storm's events share the label).
    storm_is_positive = rng.random(n_storms) < 0.15
    y = storm_is_positive[groups].astype(int)  # groups indexes storms 0..n_storms-1 directly
    prob_a = rng.random(n)
    prob_b = prob_a.copy()  # identical arm -> true delta is exactly 0

    result = cluster_bootstrap_ci(y, prob_a, prob_b, groups, seed=1, n_boot=500)

    assert result["n_boot_used"] > 0
    for key in ("delta_pr_auc", "delta_roc_auc"):
        ci = result[key]
        assert ci["ci_low"] <= 0.0 <= ci["ci_high"], f"{key} CI {ci} does not contain the true delta 0"
        assert abs(ci["mean"]) < 1e-9


def test_cluster_bootstrap_ci_detects_strong_known_effect():
    """Known true effect: B is a PERFECT predictor (prob_b == y), A is pure
    noise -- the true direction (B strictly better) must be detected, i.e.
    the delta_pr_auc / delta_roc_auc CI must NOT contain a non-positive
    value across its whole range (it should exclude zero from below)."""
    rng = np.random.default_rng(2)
    n_storms = 80
    groups = np.repeat(np.arange(n_storms), 4)
    n = len(groups)
    storm_is_positive = rng.random(n_storms) < 0.20
    y = storm_is_positive[groups].astype(int)
    prob_a = rng.random(n)  # noise, unrelated to y
    prob_b = y.astype(float)  # perfect separation

    result = cluster_bootstrap_ci(y, prob_a, prob_b, groups, seed=2, n_boot=500)

    assert result["delta_pr_auc"]["ci_low"] > 0.0
    assert result["delta_roc_auc"]["ci_low"] > 0.0
    assert result["b_pr_auc"]["mean"] > result["a_pr_auc"]["mean"]


# ---------------------------------------------------------------------------
# 4) --pairs-csv loading validates columns (ri_precursors.load_pairs_from_csv).
# ---------------------------------------------------------------------------

def test_load_pairs_from_csv_valid(tmp_path):
    path = tmp_path / "pairs.csv"
    pd.DataFrame({
        "onset_id": ["era5_2021_08_01_0000_SIDA", "era5_2021_08_02_0000_SIDB"],
        "control_id": ["era5_2021_07_01_0000_SIDC", "era5_2021_07_02_0000_SIDD"],
    }).to_csv(path, index=False)

    df = load_pairs_from_csv(path)
    assert list(df.columns) >= list(REQUIRED_PAIRS_CSV_COLUMNS)
    assert len(df) == 2


def test_load_pairs_from_csv_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pairs_from_csv(tmp_path / "does_not_exist.csv")


def test_load_pairs_from_csv_missing_columns_raises_loudly(tmp_path):
    path = tmp_path / "bad_pairs.csv"
    pd.DataFrame({"onset_id": ["a"], "wrong_column": ["b"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        load_pairs_from_csv(path)
