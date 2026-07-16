"""T-dv24.1 FINAL — label-defect impact assessment against the RAW reference.

WHY v5 EXISTS (supersedes v1-v4 in outputs/results/dv24_impact/):
  v1-v4 recomputed "correct" labels from data/event_list_augmented.csv itself.
  That file is the OUTPUT of src/processors/ibtracs.py:build_event_list, which
  computes dv12/dv24/ri_label with per-SID grouped shifts on the FILTERED
  IBTrACS series and then DROPS rows without both targets (dropna on
  dv12_kt/dv24_kt). Re-running a grouped shift on the post-dropna file cannot
  distinguish "partner never existed" from "partner row was dropped by the
  builder", which fabricated two findings:
    - "DEFECT 0 / cross-SID bleed / 84 phantom positives": FALSE. Trailing-row
      values were computed from real same-storm rows later removed by dropna.
      Verified directly against raw IBTrACS (wind at exactly t0+24h).
    - "EFFECT 2 / 3,062 undefined via NaN>=30->0 coercion": FALSE for the
      shipped data. The coercion in ri_labeling.py is dead code w.r.t. the
      event list because dropna removes every undefined row.
  This script rebuilds the PRE-dropna series from data/raw/ibtracs.ALL.list
  (replicating build_event_list's filters from config: bbox [60,-140,0,-20],
  years [1980,2025], synoptic hours, USA_WIND), validates that the replication
  reproduces the shipped file EXACTLY (32,989 rows, dv24 and ri_label equal on
  every row), and only then measures the true positional-vs-strict-temporal
  differences.

READ-ONLY. Writes only outputs/results/dv24_impact/report_v5_*.{json,md}.
Test split: aggregate counts only (dated exception in docs/PROJECT_STATE.md).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.processors.ibtracs import _clean_text_column, _standardize_longitude  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "results" / "dv24_impact"
RI_KT = 30.0
CSV = dict(keep_default_na=False, na_values=[""])


def build_pre_dropna_series() -> pd.DataFrame:
    raw = pd.read_csv(
        ROOT / "data" / "raw" / "ibtracs.ALL.list.v04r00.csv",
        low_memory=False, keep_default_na=False, na_values=[" "],
    )
    out = pd.DataFrame({
        "sid": _clean_text_column(raw["SID"], default=""),
        "timestamp": pd.to_datetime(raw["ISO_TIME"], errors="coerce"),
        "lat": pd.to_numeric(raw["LAT"], errors="coerce"),
        "lon": _standardize_longitude(raw["LON"]),
        "wind_kt": pd.to_numeric(raw["USA_WIND"], errors="coerce"),
    })
    out = out.dropna(subset=["timestamp", "lat", "lon", "wind_kt"])
    out = out[out["timestamp"].dt.hour.isin([0, 6, 12, 18])]
    out = out[(out["lat"] <= 60) & (out["lat"] >= 0)
              & (out["lon"] >= -140) & (out["lon"] <= -20)]
    out = out[(out["timestamp"].dt.year >= 1980) & (out["timestamp"].dt.year <= 2025)]
    out = out.sort_values(["sid", "timestamp"]).reset_index(drop=True)

    out["dv12"] = out.groupby("sid")["wind_kt"].shift(-2) - out["wind_kt"]
    out["dv24"] = out.groupby("sid")["wind_kt"].shift(-4) - out["wind_kt"]
    out["ri"] = (out["dv24"] >= RI_KT).astype(int)
    out["t12"] = out.groupby("sid")["timestamp"].shift(-2)
    out["t24"] = out.groupby("sid")["timestamp"].shift(-4)
    out["mis12"] = (out["t12"] - out["timestamp"]) != pd.Timedelta(hours=12)
    out["mis24"] = (out["t24"] - out["timestamp"]) != pd.Timedelta(hours=24)

    partners = out[["sid", "timestamp", "wind_kt"]].drop_duplicates(["sid", "timestamp"])
    for hours, col in ((12, "w12"), (24, "w24")):
        p = partners.assign(t=partners["timestamp"] - pd.Timedelta(hours=hours))
        p = p.rename(columns={"wind_kt": col})[["sid", "t", col]]
        out = out.merge(p, left_on=["sid", "timestamp"], right_on=["sid", "t"],
                        how="left").drop(columns=["t"])
    out["dv12_tmp"] = out["w12"] - out["wind_kt"]
    out["dv24_tmp"] = out["w24"] - out["wind_kt"]
    out["ri_tmp"] = pd.NA
    out.loc[out["dv24_tmp"].notna() & (out["dv24_tmp"] < RI_KT), "ri_tmp"] = 0
    out.loc[out["dv24_tmp"].notna() & (out["dv24_tmp"] >= RI_KT), "ri_tmp"] = 1
    return out


def main() -> None:
    series = build_pre_dropna_series()
    kept = series[series["dv12"].notna() & series["dv24"].notna()].copy()
    dropped = series[series["dv12"].isna() | series["dv24"].isna()]

    shipped = pd.read_csv(ROOT / "data" / "event_list_augmented.csv", **CSV)
    shipped["timestamp"] = pd.to_datetime(shipped["timestamp"])
    m = kept.merge(shipped[["sid", "timestamp", "dv12_kt", "dv24_kt", "ri_label"]],
                   on=["sid", "timestamp"], how="outer", indicator=True)
    replication = {
        "rows_rebuilt": int(len(kept)),
        "rows_shipped": int(len(shipped)),
        "rows_matched": int((m["_merge"] == "both").sum()),
        "dv24_equal": int((m["dv24"] == m["dv24_kt"]).sum()),
        "dv12_equal": int((m["dv12"] == m["dv12_kt"]).sum()),
        "ri_equal": int((m["ri"] == m["ri_label"]).sum()),
    }
    assert replication["rows_matched"] == len(shipped) == len(kept), (
        "replication does not reproduce the shipped event list — numbers below "
        "would be untrustworthy")
    assert replication["dv24_equal"] == replication["ri_equal"] == len(shipped)

    gain = dropped[dropped["dv12_tmp"].notna() & dropped["dv24_tmp"].notna()]
    flips_10 = kept[(kept["ri"] == 1) & (kept["ri_tmp"] == 0)]
    flips_01 = kept[(kept["ri"] == 0) & (kept["ri_tmp"] == 1)]
    to_null = kept[kept["ri_tmp"].isna()]
    event_list = {
        "rows": int(len(kept)),
        "mis24_rows": int(kept["mis24"].sum()),
        "mis24_pct": round(100 * float(kept["mis24"].mean()), 3),
        "mis12_rows": int(kept["mis12"].sum()),
        "label_flips_1_to_0": int(len(flips_10)),
        "label_flips_0_to_1": int(len(flips_01)),
        "ri_to_null": int(len(to_null)),
        "ri_to_null_positives": int((to_null["ri"] == 1).sum()),
        "dropped_rows_temporally_definable": int(len(gain)),
        "dropped_rows_temporally_positive": int((gain["dv24_tmp"] >= RI_KT).sum()),
        "dv24_value_changes": int(((kept["dv24"] != kept["dv24_tmp"])
                                   & kept["dv24_tmp"].notna()).sum()),
        "dv12_value_changes": int(((kept["dv12"] != kept["dv12_tmp"])
                                   & kept["dv12_tmp"].notna()).sum()),
        "dv12_to_null": int(kept["dv12_tmp"].isna().sum()),
    }

    ve = pd.read_csv(ROOT / "data" / "normalized" / "valid_events.csv", **CSV)
    sp = pd.read_csv(ROOT / "data" / "normalized" / "splits.csv", **CSV)
    ve = ve.merge(sp, on="event_id")
    ve["ts"] = pd.to_datetime(
        ve["event_id"].str.extract(r"^era5_(\d{4}_\d{2}_\d{2}_\d{4})_")[0],
        format="%Y_%m_%d_%H%M")
    vm = ve.merge(kept, left_on=["sid", "ts"], right_on=["sid", "timestamp"],
                  how="inner")
    assert len(vm) == len(ve), "valid-set join must be complete"

    v_null = vm[vm["ri_tmp"].isna()]
    v_flip = vm[((vm["ri"] == 1) & (vm["ri_tmp"] == 0))
                | ((vm["ri"] == 0) & (vm["ri_tmp"] == 1))]
    pos_by_sid_v1 = vm[vm["ri"] == 1].groupby("sid").size()
    pos_by_sid_v2 = vm[vm["ri_tmp"] == 1].groupby("sid").size()
    lost_sids = [s for s in pos_by_sid_v1.index if s not in pos_by_sid_v2.index]
    dv12_null = vm[vm["dv12_tmp"].isna()]
    valid_set = {
        "events": int(len(vm)),
        "label_flips": int(len(v_flip)),
        "ri_to_null": int(len(v_null)),
        "ri_to_null_by_split": {k: int(x) for k, x in
                                v_null["split"].value_counts().items()},
        "ri_to_null_positives": int((v_null["ri"] == 1).sum()),
        "ri_to_null_positives_by_split": {k: int(x) for k, x in
                                          v_null[v_null["ri"] == 1]["split"]
                                          .value_counts().items()},
        "positives_v1": int((vm["ri"] == 1).sum()),
        "positives_v2": int((vm["ri_tmp"] == 1).sum()),
        "sids_losing_all_positives": int(len(lost_sids)),
        "sids_losing_all_positives_by_split": {
            k: int(x) for k, x in ve[ve["sid"].isin(lost_sids)]
            .drop_duplicates("sid")["split"].value_counts().items()},
        "dv12_value_changes": int(((vm["dv12"] != vm["dv12_tmp"])
                                   & vm["dv12_tmp"].notna()).sum()),
        "dv12_to_null": int(len(dv12_null)),
        "dv12_to_null_by_split": {k: int(x) for k, x in
                                  dv12_null["split"].value_counts().items()},
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "reference": "raw IBTrACS (data/raw/ibtracs.ALL.list.v04r00.csv), "
                         "pre-dropna series rebuilt with build_event_list's "
                         "exact filter chain",
            "temporal_rule": "exact match at t0+12h / t0+24h, same SID",
            "undefined_semantics": "no exact temporal partner -> NULL, never 0",
            "supersedes": ["report_20260716_144350", "report_v2_20260716_145238",
                            "report_v3_20260716_145921", "report_v4_20260716_151027"],
            "retracted_findings": [
                "DEFECT 0 / cross-SID bleed / 84 phantom positives (artifact of "
                "recomputing on the post-dropna file)",
                "EFFECT 2 / 3,062 undefined valid events (same artifact)",
            ],
            "read_only": True,
        },
        "replication_check": replication,
        "event_list_32989": event_list,
        "valid_set_16780": valid_set,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"report_v5_{stamp}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# T-dv24.1 FINAL (v5) — impact against the RAW reference\n",
        f"Generated: {report['generated_utc']}\n",
        "**Supersedes v1-v4** (wrong reference: post-dropna event list). "
        "Retracted: 'Defect 0 / phantom positives' and 'Effect 2 / 3,062 "
        "undefined' — both artifacts of the wrong reference.\n",
        "## Replication check (must be exact)",
        f"- Rebuilt {replication['rows_rebuilt']} rows; shipped "
        f"{replication['rows_shipped']}; dv24/ri_label equal on "
        f"{replication['ri_equal']} rows. Shipped labels are byte-reproducible "
        "from raw IBTrACS with the current builder.\n",
        "## True defect (event list, 32,989 rows)",
        f"- Positional partner not at t0+24h: {event_list['mis24_rows']} rows "
        f"({event_list['mis24_pct']}%); dv12 misaligned: "
        f"{event_list['mis12_rows']}.",
        f"- Label flips: {event_list['label_flips_1_to_0']} (1→0), "
        f"{event_list['label_flips_0_to_1']} (0→1).",
        f"- ri_label → NULL (no exact temporal partner): "
        f"{event_list['ri_to_null']} rows, of which "
        f"{event_list['ri_to_null_positives']} positives.",
        f"- Dropped rows that a strict-temporal v2 could define: "
        f"{event_list['dropped_rows_temporally_definable']} "
        f"(positives: {event_list['dropped_rows_temporally_positive']}).\n",
        "## Valid set (16,780 events; test = aggregate counts only)",
        f"- Label flips: {valid_set['label_flips']}.",
        f"- ri_label → NULL: {valid_set['ri_to_null']} "
        f"({valid_set['ri_to_null_by_split']}), positives among them: "
        f"{valid_set['ri_to_null_positives']} "
        f"({valid_set['ri_to_null_positives_by_split']}).",
        f"- Positives: {valid_set['positives_v1']} → {valid_set['positives_v2']}.",
        f"- SIDs losing all positives: {valid_set['sids_losing_all_positives']} "
        f"({valid_set['sids_losing_all_positives_by_split']}).",
        f"- dv12: {valid_set['dv12_value_changes']} value changes, "
        f"{valid_set['dv12_to_null']} → NULL "
        f"({valid_set['dv12_to_null_by_split']}).",
    ]
    (OUT_DIR / f"report_v5_{stamp}.md").write_text("\n".join(md), encoding="utf-8")
    sys.stdout.reconfigure(errors="replace")
    print("\n".join(md))
    print(f"\nWrote report_v5_{stamp}.json/.md")


if __name__ == "__main__":
    main()
