"""Synthetic tests for the cross-seed ablation aggregator.

The aggregator computes the pre-registered final verdict from saved per-seed
oof_predictions.csv files: mean-over-seeds delta PR-AUC (B - A) with a 95%
cluster bootstrap CI by SID. Two synthetic regimes: a strong real effect must
give a CI excluding zero (positive); no effect must give a CI including zero.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.feature_ablation_cnn import aggregate_across_seeds

N_STORMS = 60
EVENTS_PER_STORM = 4
POSITIVE_STORMS = 12  # storm-level labels keep the clustering realistic


def _make_oof_csvs(tmp_path: Path, n_seeds: int, effect: bool) -> list:
    """Same event set / labels across seeds; probabilities vary by seed."""
    event_ids, sids, ys = [], [], []
    for s in range(N_STORMS):
        for e in range(EVENTS_PER_STORM):
            event_ids.append(f"ev_{s:03d}_{e}")
            sids.append(f"SID{s:03d}")
            ys.append(1.0 if s < POSITIVE_STORMS else 0.0)
    y = np.array(ys)

    csvs = []
    for k in range(n_seeds):
        rng = np.random.default_rng(1000 + k)
        prob_a = rng.uniform(0.0, 1.0, size=len(y))  # arm A: no skill
        if effect:
            # arm B: strongly informative, plus seed-dependent noise
            prob_b = np.clip(0.75 * y + 0.25 * rng.uniform(0.0, 1.0, size=len(y)), 0.0, 1.0)
        else:
            prob_b = rng.uniform(0.0, 1.0, size=len(y))  # arm B: no skill either
        p = tmp_path / f"seed{k}" / "oof_predictions.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"event_id": event_ids, "sid": sids, "y": y,
                      "prob_A": prob_a, "prob_B": prob_b}).to_csv(p, index=False)
        csvs.append(p)
    return csvs


def test_strong_effect_ci_excludes_zero_positive(tmp_path):
    csvs = _make_oof_csvs(tmp_path, n_seeds=3, effect=True)
    res = aggregate_across_seeds(csvs, n_boot=400, boot_seed=7)

    assert res["n_seeds"] == 3
    assert res["n_events"] == N_STORMS * EVENTS_PER_STORM
    assert res["n_storms"] == N_STORMS
    assert res["delta_pr_auc_ci_low"] > 0
    assert "excludes zero (positive)" in res["verdict"]


def test_null_effect_ci_includes_zero(tmp_path):
    csvs = _make_oof_csvs(tmp_path, n_seeds=3, effect=False)
    res = aggregate_across_seeds(csvs, n_boot=400, boot_seed=7)

    assert res["delta_pr_auc_ci_low"] < 0 < res["delta_pr_auc_ci_high"]
    assert "includes zero: NULL" in res["verdict"]


def test_mismatched_event_sets_are_rejected(tmp_path):
    csvs = _make_oof_csvs(tmp_path, n_seeds=2, effect=False)
    df = pd.read_csv(csvs[1])
    df.loc[0, "event_id"] = "ev_alien_0"
    df.to_csv(csvs[1], index=False)

    with pytest.raises(ValueError, match="different event set"):
        aggregate_across_seeds(csvs, n_boot=50)


def test_mismatched_labels_are_rejected(tmp_path):
    csvs = _make_oof_csvs(tmp_path, n_seeds=2, effect=False)
    df = pd.read_csv(csvs[1])
    df.loc[0, "y"] = 1.0 - df.loc[0, "y"]
    df.to_csv(csvs[1], index=False)

    with pytest.raises(ValueError, match="labels disagree"):
        aggregate_across_seeds(csvs, n_boot=50)
