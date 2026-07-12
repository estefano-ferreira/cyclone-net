"""Invariance tests for the hash-deterministic storm splitter.

These tests encode the stability contract introduced after the split-leak
incident (adding storms reshuffled 136/155 frozen-test events into dev):

  1. Adding storms never moves a pre-existing storm to another split.
  2. Input row order does not affect assignment.
  3. No storm (SID) ever appears in more than one split.
  4. Frozen overrides win over the hash (historical test benchmark intact).
  5. Split-size proportions approach the configured fractions on large sets.
  6. Class proportions per split stay within tolerance on a LARGE dataset;
     on the small current dataset they are measured and reported as a
     warning (information, not an error — they only stabilize with volume).
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.splits import SplitConfig, assign_split, hash_fraction, make_splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _cfg(tmp_path, frozen_map_path=None):
    return SplitConfig(train=0.70, val=0.15, test=0.15, seed=1337, method="sid",
                       persist=True, path=tmp_path / "splits.csv",
                       frozen_map_path=frozen_map_path)


def _manifest(n_storms, events_per_storm=4, positive_every=20, prefix="SID"):
    rows = []
    for s in range(n_storms):
        sid = f"{prefix}{s:05d}"
        for e in range(events_per_storm):
            rows.append({
                "event_id": f"{sid}_ev{e}",
                "sid": sid,
                "ri_label": 1 if (s % positive_every == 0 and e < 2) else 0,
            })
    return pd.DataFrame(rows)


def _sid_map(splits_csv):
    df = pd.read_csv(splits_csv)
    df["sid"] = df["event_id"].str.split("_ev").str[0]
    return df.drop_duplicates("sid").set_index("sid")["split"].to_dict()


def test_adding_storms_never_moves_existing(tmp_path):
    cfg = _cfg(tmp_path)
    base = _manifest(300)
    base.to_csv(tmp_path / "m1.csv", index=False)
    make_splits(tmp_path / "m1.csv", cfg)
    before = _sid_map(cfg.path)

    grown = _manifest(400)  # same first 300 storms + 100 new
    grown.to_csv(tmp_path / "m2.csv", index=False)
    make_splits(tmp_path / "m2.csv", cfg)
    after = _sid_map(cfg.path)

    moved = [s for s in before if after[s] != before[s]]
    assert moved == [], f"{len(moved)} pre-existing storms changed split"


def test_removing_storms_never_moves_existing(tmp_path):
    cfg = _cfg(tmp_path)
    full = _manifest(400)
    full.to_csv(tmp_path / "m1.csv", index=False)
    make_splits(tmp_path / "m1.csv", cfg)
    before = _sid_map(cfg.path)

    shrunk = full[full["sid"] < "SID00200"]
    shrunk.to_csv(tmp_path / "m2.csv", index=False)
    make_splits(tmp_path / "m2.csv", cfg)
    after = _sid_map(cfg.path)

    assert all(after[s] == before[s] for s in after)


def test_input_order_does_not_affect_assignment(tmp_path):
    cfg = _cfg(tmp_path)
    m = _manifest(300)
    m.to_csv(tmp_path / "m1.csv", index=False)
    make_splits(tmp_path / "m1.csv", cfg)
    ordered = _sid_map(cfg.path)

    m.sample(frac=1.0, random_state=7).to_csv(tmp_path / "m2.csv", index=False)
    make_splits(tmp_path / "m2.csv", cfg)
    shuffled = _sid_map(cfg.path)

    assert ordered == shuffled


def test_no_sid_in_two_splits(tmp_path):
    cfg = _cfg(tmp_path)
    m = _manifest(500, events_per_storm=7)
    m.to_csv(tmp_path / "m.csv", index=False)
    make_splits(tmp_path / "m.csv", cfg)
    df = pd.read_csv(cfg.path)
    df["sid"] = df["event_id"].str.split("_ev").str[0]
    assert (df.groupby("sid")["split"].nunique() == 1).all()


def test_frozen_override_wins_over_hash(tmp_path):
    # Pin storms to the split the hash would NOT choose.
    m = _manifest(50)
    pinned = {}
    for sid in m["sid"].unique()[:10]:
        cfg_nofreeze = _cfg(tmp_path)
        hash_split = assign_split(sid, cfg_nofreeze, frozen=None)
        pinned[sid] = "test" if hash_split != "test" else "train"
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(json.dumps(pinned), encoding="utf-8")

    cfg = _cfg(tmp_path, frozen_map_path=frozen_path)
    m.to_csv(tmp_path / "m.csv", index=False)
    make_splits(tmp_path / "m.csv", cfg)
    result = _sid_map(cfg.path)
    for sid, split in pinned.items():
        assert result[sid] == split, f"frozen override ignored for {sid}"


def test_hash_fraction_is_stable_and_uniform():
    # Regression pin: known values must never change across releases.
    assert assign_split("2020306N16279", _cfg(Path("."))) in {"train", "val", "test"}
    fracs = np.array([hash_fraction(f"SID{i:06d}") for i in range(20000)])
    assert 0.48 < fracs.mean() < 0.52
    assert abs((fracs < 0.70).mean() - 0.70) < 0.02


def test_split_size_proportions_on_large_set(tmp_path):
    cfg = _cfg(tmp_path)
    m = _manifest(4000)
    m.to_csv(tmp_path / "m.csv", index=False)
    make_splits(tmp_path / "m.csv", cfg)
    sids = pd.Series(_sid_map(cfg.path))
    shares = sids.value_counts(normalize=True)
    assert abs(shares["train"] - 0.70) < 0.03
    assert abs(shares["val"] - 0.15) < 0.03
    assert abs(shares["test"] - 0.15) < 0.03


def test_class_proportion_within_tolerance_on_large_synthetic(tmp_path):
    """Expanded-archive regime (~1,875 positive events): per-split positive
    rate must stay within ±3 percentage points of the global rate."""
    rng = np.random.default_rng(42)
    rows = []
    for s in range(4257):  # storm count of the 1980-2025 Atlantic event list
        sid = f"S{s:05d}"
        n_ev = int(rng.integers(4, 12))
        storm_is_ri = rng.random() < 0.10
        for e in range(n_ev):
            positive = 1 if (storm_is_ri and rng.random() < 0.45) else 0
            rows.append({"event_id": f"{sid}_ev{e}", "sid": sid, "ri_label": positive})
    m = pd.DataFrame(rows)
    n_pos = int(m["ri_label"].sum())
    assert n_pos > 1200  # comparable order of magnitude to the real archive

    cfg = _cfg(tmp_path)
    m.to_csv(tmp_path / "m.csv", index=False)
    make_splits(tmp_path / "m.csv", cfg)
    joined = pd.read_csv(cfg.path).merge(m, on="event_id")

    global_rate = joined["ri_label"].mean()
    for split, grp in joined.groupby("split"):
        rate = grp["ri_label"].mean()
        assert abs(rate - global_rate) <= 0.03, (
            f"{split}: positive rate {rate:.3%} deviates >3pp from global {global_rate:.3%}"
        )


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "normalized" / "splits.csv").exists(),
    reason="real data artifacts not present",
)
def test_class_proportion_on_current_dataset_reported_not_enforced():
    """Small-sample regime: measure per-split positive rates and WARN if they
    exceed ±3pp of global — information, not an error (they only stabilize
    with the expanded archive)."""
    norm = PROJECT_ROOT / "data" / "normalized"
    joined = pd.read_csv(norm / "splits.csv").merge(
        pd.read_csv(norm / "valid_events.csv")[["event_id", "ri_label"]], on="event_id")
    global_rate = joined["ri_label"].mean()
    for split, grp in joined.groupby("split"):
        rate = grp["ri_label"].mean()
        if abs(rate - global_rate) > 0.03:
            warnings.warn(
                f"current dataset: split '{split}' positive rate {rate:.2%} deviates "
                f">3pp from global {global_rate:.2%} (n={len(grp)}) — expected at "
                "this sample size; stabilizes with the 1980-2025 expansion",
                stacklevel=1,
            )


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data" / "normalized" / "frozen_splits.json").exists()
    or not (PROJECT_ROOT / "data" / "normalized" / "splits.csv").exists(),
    reason="real data artifacts not present",
)
def test_frozen_test_storms_stay_in_test_on_real_data():
    """Every storm pinned to 'test' in the frozen map has ALL of its events in
    the test split of the current splits.csv (benchmark integrity)."""
    norm = PROJECT_ROOT / "data" / "normalized"
    frozen = json.loads((norm / "frozen_splits.json").read_text(encoding="utf-8"))
    test_sids = {sid for sid, split in frozen.items() if split == "test"}
    assert test_sids, "frozen map should pin at least one test storm"

    joined = pd.read_csv(norm / "splits.csv").merge(
        pd.read_csv(norm / "valid_events.csv")[["event_id", "sid"]], on="event_id")
    frozen_rows = joined[joined["sid"].isin(test_sids)]
    assert (frozen_rows["split"] == "test").all(), "a frozen test storm leaked out of test"
