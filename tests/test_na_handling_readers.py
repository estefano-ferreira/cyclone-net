"""NA-handling regression tests for CSV readers and text cleaning.

Each test states what would make it fail — none is decorative:
- ``_clean_text_column`` stringifying NaN into the literal "nan" (poisons
  keys like ``sid``);
- ``make_splits`` silently dropping an event whose ``sid`` is missing
  (inviolable path must fail loudly), or letting pandas' default NA parsing
  eat an NA-like group key;
- ``load_dev_events`` letting a NULL (empty-cell) ``ri_label`` reach
  stratification as NaN, or coercing it to 0, or excluding it silently;
- ``pl_gate_census.load_split_join`` mangling NULL labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.processors.ibtracs import _clean_text_column
from src.utils.splits import SplitConfig, make_splits


# ---------------------------------------------------------------------------
# _clean_text_column
# ---------------------------------------------------------------------------

def test_clean_text_column_never_emits_nan_literal():
    s = pd.Series(["EP", None, np.nan, "NA", "nan"])
    out = _clean_text_column(s, default="")
    # Fails if: NaN reaches astype(str) first (literal "nan"), or the literal
    # basin code "NA" is mistaken for missing data.
    assert list(out) == ["EP", "", "", "NA", ""]
    assert "nan" not in set(out)


def test_clean_text_column_custom_default():
    s = pd.Series([np.nan, "X"])
    out = _clean_text_column(s, default="UNK")
    assert list(out) == ["UNK", "X"]


# ---------------------------------------------------------------------------
# make_splits — the inviolable path must scream, never silently drop
# ---------------------------------------------------------------------------

def _split_cfg(tmp_path, persist=False):
    return SplitConfig(
        train=0.70, val=0.15, test=0.15, seed=1, method="sid",
        persist=persist, path=tmp_path / "splits.csv", frozen_map_path=None,
    )


def test_make_splits_raises_on_missing_sid(tmp_path):
    meta = tmp_path / "meta.csv"
    meta.write_text(
        "event_id,sid\n"
        "era5_1980_06_10_0000_AAA,1980161N09249\n"
        "era5_1980_06_10_0600_BBB,\n",  # missing sid — must fail loudly
        encoding="utf-8",
    )
    # Fails if: the missing-sid event is silently dropped (old behavior).
    with pytest.raises(ValueError, match="missing 'sid'"):
        make_splits(meta, _split_cfg(tmp_path))


def test_make_splits_assigns_na_like_sid(tmp_path):
    # A group key that pandas' default NA parsing would destroy. Under the
    # old default read this row silently vanished from splits.csv.
    meta = tmp_path / "meta.csv"
    meta.write_text(
        "event_id,sid\n"
        "era5_1980_06_10_0000_AAA,1980161N09249\n"
        "era5_1980_06_10_0600_NAX,NA\n",
        encoding="utf-8",
    )
    make_splits(meta, _split_cfg(tmp_path, persist=True))
    out = pd.read_csv(tmp_path / "splits.csv", keep_default_na=False, na_values=[""])
    # Fails if: the "NA" sid row was NA-parsed and dropped.
    assert len(out) == 2
    assert set(out["split"]).issubset({"train", "val", "test"})


# ---------------------------------------------------------------------------
# load_dev_events / load_split_join — nullable v2 labels
# ---------------------------------------------------------------------------

@pytest.fixture()
def normalized_dir(tmp_path):
    d = tmp_path / "normalized"
    d.mkdir()
    (d / "splits.csv").write_text(
        "event_id,split\n"
        "e1,train\n"
        "e2,train\n"
        "e3,val\n"
        "e4,test\n",
        encoding="utf-8",
    )
    (d / "valid_events.csv").write_text(
        "event_id,sid,storm_name,ri_label\n"
        "e1,S1,ALPHA,1\n"
        "e2,S1,ALPHA,\n"      # NULL label (undefined under v2)
        "e3,S2,NADINE,0\n"
        "e4,S3,GAMMA,1\n",
        encoding="utf-8",
    )
    return d


def test_load_dev_events_excludes_null_labels(normalized_dir, capsys):
    from analysis.feature_ablation_kfold import load_dev_events

    cfg = {"paths": {"normalized_dir": str(normalized_dir)}}
    dev = load_dev_events(cfg)

    # Fails if: the NULL event reaches the dev frame (as NaN or coerced 0),
    # the test split leaks in, or the exclusion happens silently.
    assert set(dev["event_id"]) == {"e1", "e3"}
    assert dev["ri_label"].dtype.kind in "iu"
    assert not dev["ri_label"].isna().any()
    assert "excluded 1 dev events with NULL ri_label" in capsys.readouterr().out


def test_pl_gate_census_join_keeps_null_as_nan(normalized_dir):
    from analysis.pl_gate_census import load_split_join

    cfg = {"paths": {"normalized_dir": str(normalized_dir)}}
    df = load_split_join(cfg)

    # Fails if: NULL becomes 0 (silent negative) or the row is dropped —
    # the census counts events by PL coverage; NULL is a label state.
    assert len(df) == 4
    row = df.loc[df["event_id"] == "e2", "ri_label"]
    assert row.isna().all()
    assert (df.loc[df["event_id"] == "e1", "ri_label"] == 1).all()
