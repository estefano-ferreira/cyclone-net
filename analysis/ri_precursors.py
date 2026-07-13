# analysis/ri_precursors.py
"""
Exploratory analysis of RI PRECURSORS — hypothesis GENERATION, not discovery.

Tests exactly FOUR physically motivated hypotheses about environmental
conditions in the 24 h window preceding rapid-intensification onset,
against intensity/basin-matched non-RI controls. No other variables are
scanned (multiple-comparison discipline); Bonferroni correction for the
four primary tests is applied.

Design:
  * t = 0 is RI ONSET: the first 6-hourly point of an RI episode
    (ri_label = 1 whose previous point in the same storm is 0 or absent).
    The event's own cube provides the full pre-window: its 5 time slices
    are offsets [0, -6, -12, -18, -24] hours.
  * Controls: non-RI points (ri_label = 0), from a different storm-time,
    matched 1:1 to each onset on initial intensity (wind_kt within a
    tolerance band) and basin (when both known), excluding any point that
    falls within 24 h BEFORE an onset of its own storm (pre-onset
    contamination guard). Sampling is seeded and without replacement.
  * Statistics per hypothesis: paired differences (onset minus its matched
    control) tested with a sign-flip permutation on the mean difference
    (10,000 permutations) — the explicit null that group labels are
    exchangeable within pairs — plus Cliff's delta as the effect size.
  * The PRIMARY test per hypothesis (Bonferroni x4) is at the
    physically-preferred quantity; per-lag results are reported as
    descriptive secondary structure (lag multiplicity is NOT separately
    corrected — stated in the report).

Hypotheses (channel -> spatial aggregate per time slice):
  H1 pressure fall:  min(mslp_Pa) fall over the window (t0 minus t-24; more
                     negative = faster deepening).
  H2 shear:          mean(shear_850_200_mps) level. Coverage depends on the
                     PL-channel backfill state; measured at runtime and
                     reported in audit["note"] -- do NOT assume a fixed
                     year range here.
  H3 warm water:     mean(sst_anom_K) level.
  H4 mid humidity:   mean(rh_mid) level. Same coverage caveat as H2.

Output: docs/ri_precursors.md + outputs/results/ri_precursors.json.

This is observational, hypothesis-generating analysis: a detected signal is
NOT a confirmed precursor and does not imply causality; prospective
validation would be required.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.feature_ablation_kfold import enforce_gate  # noqa: E402
from src.utils.config import cfg_get, load_config  # noqa: E402

LAGS_H = [0, -6, -12, -18, -24]  # cube time-axis order
REQUIRED_PAIRS_CSV_COLUMNS = {"onset_id", "control_id"}


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta in [-1, 1]: P(a > b) - P(a < b), computed exactly."""
    a = np.asarray(a)[:, None]
    b = np.asarray(b)[None, :]
    return float((np.sum(a > b) - np.sum(a < b)) / (a.size * b.size))


def signflip_pvalue(diffs: np.ndarray, rng: np.random.Generator,
                    n_perm: int = 10_000) -> float:
    """Two-sided sign-flip permutation p for mean(paired difference) != 0."""
    diffs = np.asarray(diffs, dtype=float)
    obs = abs(diffs.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
    null = np.abs((signs * np.abs(diffs)).mean(axis=1))
    return float((1 + np.sum(null >= obs)) / (n_perm + 1))


def load_pairs_from_csv(pairs_path: Path) -> pd.DataFrame:
    """Load and validate the exact matched onset/control pairs from a prior run.

    Fails loudly (raises) if the file is missing or lacks the expected
    columns, rather than silently proceeding with a malformed pairing.
    """
    if not pairs_path.exists():
        raise FileNotFoundError(f"--pairs-csv not found: {pairs_path}")
    pairs_df = pd.read_csv(pairs_path)
    missing_cols = REQUIRED_PAIRS_CSV_COLUMNS - set(pairs_df.columns)
    if missing_cols:
        raise ValueError(
            f"--pairs-csv {pairs_path} is missing required column(s) {sorted(missing_cols)}; "
            f"found columns {list(pairs_df.columns)}. Expected the exact format persisted by "
            "a prior run to outputs/results/ri_precursor_pairs.csv."
        )
    return pairs_df


def extract_features(event_id: str, interim: Path) -> dict | None:
    """Per-lag aggregates for one event's cube; None if artifacts missing."""
    meta_path = interim / f"{event_id}.json"
    cube_path = interim / f"{event_id}.npy"
    if not (meta_path.exists() and cube_path.exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    channels = list(meta.get("channels", []))
    cube = np.load(cube_path, mmap_mode="r")
    if cube.shape[2] != len(LAGS_H):
        return None

    def agg(channel: str, fn) -> list | None:
        if channel not in channels:
            return None
        ci = channels.index(channel)
        return [float(fn(np.asarray(cube[:, :, t, ci], dtype=float))) for t in range(len(LAGS_H))]

    return {
        "mslp_min_hPa": [v / 100.0 for v in agg("mslp_Pa", np.nanmin)] if "mslp_Pa" in channels else None,
        "shear_mean": agg("shear_850_200_mps", np.nanmean),
        # H3 uses RAW SST (patch mean): the stored sst_anom_K channel is a
        # SPATIAL anomaly (sst minus patch mean), whose patch mean is ~0 by
        # construction — aggregating it would test nothing. The spatial-
        # anomaly MAX (warmest spot relative to surroundings) is kept as a
        # descriptive secondary.
        "sst_mean": agg("sst_K", np.nanmean),
        "sst_anom_max": agg("sst_anom_K", np.nanmax),
        "rh_mid_mean": agg("rh_mid", np.nanmean),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--match-band-kt", type=float, default=10.0)
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml (relative paths resolve against the project root).")
    parser.add_argument("--pairs-csv", type=str, default=None,
                        help="Reuse the exact matched onset/control pairs persisted by a prior run "
                             "(outputs/results/ri_precursor_pairs.csv) instead of re-matching. Use "
                             "this for the post-backfill re-test so newly-available shear/rh "
                             "features are evaluated on the SAME frozen pairs, not a fresh sample.")
    parser.add_argument("--require-gate", action=argparse.BooleanOptionalAction, default=True,
                        help="Refuse to run unless outputs/results/pl_gate_census.json reports "
                             "gate_pass=true (default: on). Use --no-require-gate to bypass.")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    cfg = load_config(str(config_path))

    enforce_gate(cfg, args.require_gate)

    interim = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()

    events = pd.read_csv(PROJECT_ROOT / "data" / "event_list_augmented.csv")
    events["ts"] = pd.to_datetime(events["timestamp"])
    events = events.sort_values(["sid", "ts"]).reset_index(drop=True)
    events["event_id"] = [
        f"era5_{ts.strftime('%Y_%m_%d_%H%M')}_{sid}" for ts, sid in zip(events["ts"], events["sid"])
    ]

    # RI onsets: first point of each episode.
    prev_label = events.groupby("sid")["ri_label"].shift(1).fillna(0)
    events["is_onset"] = (events["ri_label"] == 1) & (prev_label == 0)

    # Controls: non-RI points not within 24h BEFORE any onset of their storm.
    onset_times = events[events["is_onset"]].groupby("sid")["ts"].apply(list).to_dict()

    def contaminated(row) -> bool:
        for t_on in onset_times.get(row.sid, []):
            delta_h = (t_on - row.ts).total_seconds() / 3600.0
            if 0 < delta_h <= 24:
                return True
        return False

    candidates = events[(events["ri_label"] == 0)].copy()
    candidates = candidates[~candidates.apply(contaminated, axis=1)]

    onsets = events[events["is_onset"]].copy()

    # 1:1 matching on intensity band (+ basin when both known), seeded,
    # without replacement -- UNLESS --pairs-csv is given, in which case the
    # exact frozen pairs from a prior run are reused verbatim. The
    # post-backfill re-test (newly-available shear/rh features) must be
    # evaluated on the SAME matched pairs as the original run, not a
    # freshly-sampled set, or the "re-test" would silently become a new,
    # uncontrolled experiment.
    if args.pairs_csv:
        pairs_path = Path(args.pairs_csv)
        if not pairs_path.is_absolute():
            pairs_path = (PROJECT_ROOT / pairs_path).resolve()
        pairs_df = load_pairs_from_csv(pairs_path)
        pairs = list(zip(pairs_df["onset_id"].astype(str), pairs_df["control_id"].astype(str)))
        n_unmatched = None  # matching was not performed this run -- pairs reused verbatim
        pairs_provenance = f"loaded verbatim from --pairs-csv={pairs_path} (frozen; no re-matching performed)"
    else:
        used = set()
        pairs = []
        n_unmatched = 0
        cand_by_idx = candidates.reset_index(drop=True)
        for onset in onsets.itertuples():
            band = args.match_band_kt
            pool = cand_by_idx[
                (~cand_by_idx.index.isin(used))
                & (cand_by_idx["wind_kt"].sub(onset.wind_kt).abs() <= band)
            ]
            if pd.notna(onset.basin) and str(onset.basin).strip():
                same_basin = pool[pool["basin"].astype(str) == str(onset.basin)]
                if len(same_basin):
                    pool = same_basin
            if not len(pool):
                pool = cand_by_idx[
                    (~cand_by_idx.index.isin(used))
                    & (cand_by_idx["wind_kt"].sub(onset.wind_kt).abs() <= band * 1.5)
                ]
            if not len(pool):
                n_unmatched += 1
                continue
            pick = pool.index[int(rng.integers(0, len(pool)))]
            used.add(pick)
            pairs.append((onset.event_id, cand_by_idx.loc[pick, "event_id"]))
        pairs_provenance = "freshly matched this run (seeded, without replacement)"

    # Feature extraction (audited).
    rows = []
    n_missing = {"onset": 0, "control": 0}
    for onset_id, control_id in pairs:
        fo = extract_features(onset_id, interim)
        fc = extract_features(control_id, interim)
        if fo is None:
            n_missing["onset"] += 1
            continue
        if fc is None:
            n_missing["control"] += 1
            continue
        rows.append({"onset_id": onset_id, "control_id": control_id, "onset": fo, "control": fc})

    def n_with(feature: str) -> int:
        return sum(1 for r in rows if r["onset"][feature] and r["control"][feature])

    n_pairs_h2_shear = n_with("shear_mean")
    n_pairs_h4_rh_mid = n_with("rh_mid_mean")
    n_pairs_total = len(rows)
    # Coverage note computed AT RUNTIME from the actual extracted pairs --
    # NOT a hardcoded "2020-2023 only" claim. Before the 1980-2019 PL
    # backfill this will show partial coverage; after it completes (and the
    # gate in analysis/pl_gate_census.py passes) it should show full
    # coverage for whichever years this analysis's matched pairs happen to
    # span.
    if n_pairs_total > 0:
        coverage_note = (
            f"shear/rh_mid channel coverage measured at runtime over this run's "
            f"{n_pairs_total} matched pairs: shear_850_200_mps available for "
            f"{n_pairs_h2_shear}/{n_pairs_total} pairs ({n_pairs_h2_shear / n_pairs_total:.1%}); "
            f"rh_mid available for {n_pairs_h4_rh_mid}/{n_pairs_total} pairs "
            f"({n_pairs_h4_rh_mid / n_pairs_total:.1%})."
        )
    else:
        coverage_note = "no matched pairs with artifacts -- shear/rh_mid coverage undefined."

    audit = {
        "n_ri_positive_points": int(events["ri_label"].sum()),
        "n_ri_onsets": int(len(onsets)),
        "n_matched_pairs": int(len(pairs)),
        "n_unmatched_onsets": n_unmatched,
        "pairs_provenance": pairs_provenance,
        "n_pairs_with_artifacts": int(len(rows)),
        "n_missing_artifacts": n_missing,
        "n_pairs_h1_mslp": n_with("mslp_min_hPa"),
        "n_pairs_h2_shear": n_pairs_h2_shear,
        "n_pairs_h3_sst": n_with("sst_mean"),
        "n_pairs_h4_rh_mid": n_pairs_h4_rh_mid,
        "note": coverage_note,
    }
    print("AVAILABILITY AUDIT:", json.dumps(audit, indent=2))

    hypotheses = {
        "H1_pressure_fall": {
            "feature": "mslp_min_hPa",
            "primary": "fall_24h",   # value(t0) - value(t-24); negative = deepening
            "physics": "storms already deepening faster in the prior 24h",
        },
        "H2_shear": {
            "feature": "shear_mean",
            "primary": "level_per_lag",
            "physics": "high deep-layer shear suppresses RI",
        },
        "H3_sst": {
            "feature": "sst_mean",
            "primary": "level_per_lag",
            "physics": "warmer water sustains intensification",
        },
        "H4_rh_mid": {
            "feature": "rh_mid_mean",
            "primary": "level_per_lag",
            "physics": "dry mid-levels suppress convection",
        },
    }

    results = {}
    for name, spec in hypotheses.items():
        feat = spec["feature"]
        valid = [r for r in rows if r["onset"][feat] and r["control"][feat]]
        if len(valid) < 10:
            results[name] = {"status": "underpowered", "n_pairs": len(valid)}
            continue

        per_lag = {}
        for li, lag in enumerate(LAGS_H):
            d = np.array([r["onset"][feat][li] - r["control"][feat][li] for r in valid])
            d = d[np.isfinite(d)]
            per_lag[f"t{lag:+d}h"] = {
                "mean_paired_diff": float(d.mean()),
                "cliffs_delta": cliffs_delta(
                    [r["onset"][feat][li] for r in valid],
                    [r["control"][feat][li] for r in valid]),
                "p_signflip": signflip_pvalue(d, rng),
            }

        if spec["primary"] == "fall_24h":
            # index 0 = t0, index 4 = t-24
            d = np.array([(r["onset"][feat][0] - r["onset"][feat][4])
                          - (r["control"][feat][0] - r["control"][feat][4]) for r in valid])
            d = d[np.isfinite(d)]
            primary = {
                "quantity": "24h fall (t0 minus t-24), onset minus control",
                "mean_paired_diff": float(d.mean()),
                "p_signflip": signflip_pvalue(d, rng),
            }
        else:
            # Primary = the t-24h lag (earliest, i.e. a true PREcursor;
            # later lags increasingly overlap the onset itself).
            key = "t-24h"
            primary = {"quantity": f"level at {key}", **per_lag[key]}

        results[name] = {
            "n_pairs": len(valid),
            "physics": spec["physics"],
            "primary": primary,
            # Bonferroni x4 is fixed at the ORIGINAL pre-registered family size
            # (H1..H4), unconditionally -- including in the post-backfill
            # re-test with --pairs-csv. The family is "the four primary
            # hypotheses tested", not "the two newly-available channels";
            # re-testing H2/H4 on frozen pairs with better data does not
            # shrink or grow the pre-registered family, so the correction
            # factor does NOT change to x2. Do not "helpfully" tighten this.
            "p_bonferroni_x4": min(1.0, primary["p_signflip"] * 4),
            "per_lag_descriptive": per_lag,
        }

    report = {"protocol": "RI precursors, 4 pre-registered hypotheses, matched pairs, "
                          "sign-flip permutation null, Bonferroni x4 on primaries",
              "seed": args.seed, "audit": audit, "results": results,
              "interpretation_guard": (
                  "Hypothesis-GENERATING observational analysis. Detected signals are "
                  "not confirmed precursors and do not imply causality; per-lag "
                  "secondary results are descriptive and not multiplicity-corrected.")}

    out = PROJECT_ROOT / "outputs" / "results" / "ri_precursors.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Persist the matched pairs for provenance and downstream control tests
    # (e.g. the H1 wind-trend control) — same pairs, no re-matching.
    pd.DataFrame(
        [{"onset_id": r["onset_id"], "control_id": r["control_id"]} for r in rows]
    ).to_csv(PROJECT_ROOT / "outputs" / "results" / "ri_precursor_pairs.csv", index=False)
    print(json.dumps(results, indent=2)[:3000])
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
