# analysis/ri_precursors_h1_control.py
"""
Control test for the H1 precursor: independent signal or circularity?

H1 found that RI onsets deepen (min MSLP) ~1.6 hPa/24 h faster than
intensity/basin-matched controls (docs/ri_precursors.md). Pressure and wind
are anticorrelated, so the fall may simply BE the already-visible start of
intensification. This test asks whether pressure carries information about
RI **beyond** the wind trend over the same pre-onset window.

Uses the SAME matched pairs persisted by analysis/ri_precursors.py
(outputs/results/ri_precursor_pairs.csv) — no re-matching.

TEST 1 — conditional (paired) logistic regression on within-pair differences:
    logit(onset vs control) ~ Δwind_trend + Δpressure_trend
  where trends are measured STRICTLY pre-onset (t-24h → t-6h, from the
  best-track wind and the cube's min-MSLP slices). The null for the pressure
  coefficient is a within-pair label swap (sign-flip of the joint difference
  vector), which preserves the wind-pressure coupling — exactly the
  "does pressure add anything once wind is in the model?" question.
  A quartile stratification by wind trend is reported as robustness.

TEST 2 — temporal lead: per-lag effect profiles for the pressure LEVEL and
  the wind LEVEL (onset minus control), plus a conditional model at t-24h
  only (pressure level given wind level and wind trend). If pressure
  separates before wind does, the precursor leads; if they move together,
  circularity is reinforced.

Output: printed report + outputs/results/ri_precursors_h1_control.json.
Hypothesis-generating; no causal claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.ri_precursors import LAGS_H, cliffs_delta, extract_features  # noqa: E402
from src.utils.config import cfg_get, load_config  # noqa: E402

IDX = {lag: i for i, lag in enumerate(LAGS_H)}  # lag hours -> cube time index


def conditional_logit_beta(deltas: np.ndarray) -> np.ndarray:
    """Conditional logistic coefficients for matched pairs.

    Implemented as an intercept-free logistic fit on the symmetric
    augmentation {+Δ -> 1, -Δ -> 0}, which maximizes the conditional
    likelihood for 1:1 matched sets.
    """
    X = np.vstack([deltas, -deltas])
    y = np.concatenate([np.ones(len(deltas)), np.zeros(len(deltas))])
    model = LogisticRegression(fit_intercept=False, penalty=None, max_iter=2000)
    model.fit(X, y)
    return model.coef_[0]


def permutation_p_for_coef(deltas: np.ndarray, coef_index: int,
                           rng: np.random.Generator, n_perm: int) -> float:
    """Two-sided permutation p for one coefficient: within-pair label swaps
    (joint sign-flips of each pair's difference vector) preserve the
    covariate coupling while breaking the onset/control assignment."""
    obs = abs(conditional_logit_beta(deltas)[coef_index])
    null = np.empty(n_perm)
    for b in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=(len(deltas), 1))
        null[b] = abs(conditional_logit_beta(deltas * signs)[coef_index])
    return float((1 + np.sum(null >= obs)) / (n_perm + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-perm", type=int, default=2000)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    cfg = load_config(str(PROJECT_ROOT / "config.yaml"))
    interim = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()

    pairs = pd.read_csv(PROJECT_ROOT / "outputs" / "results" / "ri_precursor_pairs.csv")

    # Best-track wind history: event_id -> (sid, ts, wind_kt) lookup.
    events = pd.read_csv(PROJECT_ROOT / "data" / "event_list_augmented.csv",
                         keep_default_na=False, na_values=[""])
    events["ts"] = pd.to_datetime(events["timestamp"])
    events["event_id"] = [
        f"era5_{ts.strftime('%Y_%m_%d_%H%M')}_{sid}" for ts, sid in zip(events["ts"], events["sid"])
    ]
    by_id = events.set_index("event_id")[["sid", "ts", "wind_kt"]]
    wind_at = {(r.sid, r.ts): r.wind_kt for r in events.itertuples()}

    def wind_lag(event_id: str, lag_h: int) -> float | None:
        if event_id not in by_id.index:
            return None
        row = by_id.loc[event_id]
        key = (row["sid"], row["ts"] + pd.Timedelta(hours=lag_h))
        val = wind_at.get(key)
        return float(val) if val is not None and np.isfinite(val) else None

    rows = []
    n_drop = {"wind_history": 0, "cube": 0}
    for pair in pairs.itertuples():
        rec = {}
        ok = True
        for role, eid in (("onset", pair.onset_id), ("control", pair.control_id)):
            feats = extract_features(eid, interim)
            if feats is None or not feats["mslp_min_hPa"]:
                n_drop["cube"] += 1
                ok = False
                break
            p = feats["mslp_min_hPa"]
            w24, w6, w0 = wind_lag(eid, -24), wind_lag(eid, -6), wind_lag(eid, 0)
            if w24 is None or w6 is None or w0 is None:
                n_drop["wind_history"] += 1
                ok = False
                break
            rec[role] = {
                # STRICTLY pre-onset trends: t-24h -> t-6h.
                "wind_trend": w6 - w24,
                "pressure_trend": p[IDX[-6]] - p[IDX[-24]],
                "wind_level_t24": w24,
                "pressure_level_t24": p[IDX[-24]],
                "wind_levels": {lag: wind_lag(eid, lag) for lag in LAGS_H},
                "pressure_levels": {lag: p[IDX[lag]] for lag in LAGS_H},
            }
        if ok:
            rows.append(rec)

    n = len(rows)
    audit = {"n_pairs_input": int(len(pairs)), "n_pairs_used": n, "dropped": n_drop,
             "trend_window": "t-24h to t-6h (strictly pre-onset)"}
    print("AVAILABILITY AUDIT:", json.dumps(audit, indent=2))

    def d(field: str) -> np.ndarray:
        return np.array([r["onset"][field] - r["control"][field] for r in rows])

    # Scale differences so coefficients are comparable across covariates.
    # SCALE ONLY — never center: in conditional (paired) logistic regression
    # the mean of the within-pair differences IS the signal; mean-centering
    # would remove the effect by construction (betas identically zero).
    def z(x: np.ndarray) -> np.ndarray:
        return x / x.std(ddof=1)

    # ---- TEST 1: pressure trend beyond wind trend ----
    dw, dp = d("wind_trend"), d("pressure_trend")
    deltas_wp = np.column_stack([z(dw), z(dp)])
    betas = conditional_logit_beta(deltas_wp)
    p_wind = permutation_p_for_coef(deltas_wp, 0, rng, args.n_perm)
    p_press = permutation_p_for_coef(deltas_wp, 1, rng, args.n_perm)

    # Robustness: within-quartile stratification by wind-trend difference.
    strata = []
    qs = np.quantile(dw, [0.25, 0.5, 0.75])
    bins = np.digitize(dw, qs)
    for b in range(4):
        mask = bins == b
        if mask.sum() >= 20:
            strata.append({
                "stratum": f"Q{b+1}", "n": int(mask.sum()),
                "mean_dp": float(dp[mask].mean()),
                "cliffs_delta_dp_vs_zero": float(np.mean(dp[mask] < 0) - np.mean(dp[mask] > 0)),
            })

    test1 = {
        "model": "conditional logistic: onset ~ z(d_wind_trend) + z(d_pressure_trend)",
        "beta_wind_trend": float(betas[0]), "p_wind_trend": p_wind,
        "beta_pressure_trend": float(betas[1]), "p_pressure_trend": p_press,
        "wind_trend_strata": strata,
        "raw_mean_diff_wind_trend_kt": float(dw.mean()),
        "raw_mean_diff_pressure_trend_hPa": float(dp.mean()),
    }

    # ---- TEST 2: temporal lead (per-lag level profiles + t-24h model) ----
    profile = {}
    for lag in LAGS_H:
        wo = np.array([r["onset"]["wind_levels"][lag] for r in rows])
        wc = np.array([r["control"]["wind_levels"][lag] for r in rows])
        po = np.array([r["onset"]["pressure_levels"][lag] for r in rows])
        pc = np.array([r["control"]["pressure_levels"][lag] for r in rows])
        profile[f"t{lag:+d}h"] = {
            "wind_mean_diff_kt": float((wo - wc).mean()),
            "wind_cliffs_delta": cliffs_delta(wo, wc),
            "pressure_mean_diff_hPa": float((po - pc).mean()),
            "pressure_cliffs_delta": cliffs_delta(po, pc),
        }

    deltas_t24 = np.column_stack([
        z(d("wind_level_t24")), z(d("wind_trend")), z(d("pressure_level_t24")),
    ])
    betas24 = conditional_logit_beta(deltas_t24)
    p_press24 = permutation_p_for_coef(deltas_t24, 2, rng, args.n_perm)

    test2 = {
        "per_lag_level_profile": profile,
        "t24_model": "onset ~ z(d_wind_level_t24) + z(d_wind_trend) + z(d_pressure_level_t24)",
        "beta_pressure_level_t24": float(betas24[2]),
        "p_pressure_level_t24": p_press24,
    }

    report = {"protocol": "H1 circularity control (same matched pairs)",
              "seed": args.seed, "n_permutations": args.n_perm,
              "audit": audit, "test1_wind_trend_control": test1,
              "test2_temporal_lead": test2}
    out = PROJECT_ROOT / "outputs" / "results" / "ri_precursors_h1_control.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"test1": test1, "test2_t24_model": {
        "beta": test2["beta_pressure_level_t24"], "p": test2["p_pressure_level_t24"]}}, indent=2))
    print("per-lag profile:")
    for lag, v in profile.items():
        print(f"  {lag}: wind diff={v['wind_mean_diff_kt']:+.2f}kt (d={v['wind_cliffs_delta']:+.2f}) | "
              f"pressure diff={v['pressure_mean_diff_hPa']:+.2f}hPa (d={v['pressure_cliffs_delta']:+.2f})")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
