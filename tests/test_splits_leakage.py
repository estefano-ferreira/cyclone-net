"""
Leakage guarantee: storm-level (SID) splits must never place the same storm in
more than one of train/val/test. This is the central anti-leakage claim of the
project; a violation silently inflates every reported metric.
"""
import pandas as pd

from src.utils.splits import SplitConfig, make_splits


def _build_metadata(tmp_path):
    rows = []
    # 20 storms, 4 events each; alternate RI label by storm.
    for s in range(20):
        sid = f"AL{s:02d}2020"
        for e in range(4):
            rows.append({
                "event_id": f"{sid}_{e:02d}",
                "sid": sid,
                "ri_label": s % 3 == 0,  # some positive storms
            })
    csv = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_no_sid_appears_in_two_splits(tmp_path):
    csv = _build_metadata(tmp_path)
    cfg = SplitConfig(train=0.6, val=0.2, test=0.2, seed=0,
                      method="sid", persist=True, path=tmp_path / "splits.csv")
    make_splits(csv, cfg)

    splits = pd.read_csv(tmp_path / "splits.csv")
    meta = pd.read_csv(csv)
    merged = splits.merge(meta[["event_id", "sid"]], on="event_id")

    splits_per_sid = merged.groupby("sid")["split"].nunique()
    leaking = splits_per_sid[splits_per_sid > 1]
    assert leaking.empty, f"SIDs leaking across splits: {list(leaking.index)}"


def test_split_is_deterministic_under_same_seed(tmp_path):
    csv = _build_metadata(tmp_path)
    out1, out2 = tmp_path / "s1.csv", tmp_path / "s2.csv"
    for path in (out1, out2):
        make_splits(csv, SplitConfig(train=0.6, val=0.2, test=0.2, seed=123,
                                     method="sid", persist=True, path=path))
    a = pd.read_csv(out1).sort_values("event_id").reset_index(drop=True)
    b = pd.read_csv(out2).sort_values("event_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_every_event_assigned_a_split(tmp_path):
    csv = _build_metadata(tmp_path)
    make_splits(csv, SplitConfig(train=0.6, val=0.2, test=0.2, seed=0,
                                 method="sid", persist=True, path=tmp_path / "splits.csv"))
    splits = pd.read_csv(tmp_path / "splits.csv")
    assert set(splits["split"].unique()) <= {"train", "val", "test"}
    assert len(splits) == 80  # 20 storms x 4 events
