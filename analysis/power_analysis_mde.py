"""
Retrospective MDE (design precision) for the pre-registered deltas H6, H9/V1, H9/V2.
Characterizes estimator precision only; does NOT re-read or revise any verdict.

Script computes Minimum Detectable Effect (MDE) at 80% and 90% power from bootstrap
draws of published results, reproducing confidence intervals exactly as a fidelity gate.
Includes a proxy for planned basin ablation (S vs SF on tabular arm only).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import average_precision_score  # noqa: E402
from analysis.feature_ablation_cnn import aggregate_across_seeds  # noqa: E402
from analysis.tabular_baseline_kfold import paired_cross_seed_bootstrap  # noqa: E402
from src.utils.paths import rel_to_root  # noqa: E402


def read_oof_csv(path: Path) -> pd.DataFrame:
    """Read OOF predictions CSV, preserving NA (North Atlantic) as string."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    # Convert numeric columns explicitly
    numeric_cols = [c for c in df.columns if c not in ("event_id", "sid")]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10_000,
                        help="Bootstrap replicates (default 10000 for full run, lower for smoke test).")
    args = parser.parse_args()

    n_boot = args.n_boot
    boot_seed = 42
    is_smoke_run = n_boot < 10_000

    # =========================================================================
    # INPUT PATHS (fixed, relative to PROJECT_ROOT)
    # =========================================================================
    results_dir = PROJECT_ROOT / "outputs" / "results"

    # Tabular OOF CSVs
    tab_oofs = {
        42: results_dir / "tabular_baseline" / "20260715T123221Z" / "seed42" / "oof_predictions.csv",
        123: results_dir / "tabular_baseline" / "20260715T123221Z" / "seed123" / "oof_predictions.csv",
        456: results_dir / "tabular_baseline" / "20260715T123221Z" / "seed456" / "oof_predictions.csv",
    }

    # CNN OOF CSVs
    cnn_oofs = {
        42: results_dir / "feature_ablation_cnn" / "20260713T232126Z" / "seed42" / "oof_predictions.csv",
        123: results_dir / "feature_ablation_cnn" / "20260714T223910Z" / "seed123" / "oof_predictions.csv",
        456: results_dir / "feature_ablation_cnn" / "20260715T223350Z" / "seed456" / "oof_predictions.csv",
    }

    # Published JSON results
    cnn_aggregate_json = results_dir / "feature_ablation_cnn" / "aggregate_20260716T120517Z.json"
    tabular_compare_json = results_dir / "tabular_baseline" / "compare_20260716T121803Z.json"

    print("=" * 80)
    print("POWER ANALYSIS: MDE CHARACTERIZATION")
    print("=" * 80)
    print(f"Bootstrap replicates: {n_boot}")
    print(f"Bootstrap seed: {boot_seed}")
    if is_smoke_run:
        print("MODE: SMOKE RUN (no fidelity gate)")
    else:
        print("MODE: FULL RUN (fidelity gate enabled)")
    print()

    # =========================================================================
    # LOAD PUBLISHED JSON RESULTS FOR FIDELITY GATING
    # =========================================================================
    if cnn_aggregate_json.exists():
        cnn_agg = json.loads(cnn_aggregate_json.read_text(encoding="utf-8"))
        print(f"Loaded CNN aggregate JSON: {rel_to_root(cnn_aggregate_json)}")
    else:
        cnn_agg = None
        print(f"WARNING: CNN aggregate JSON not found at {rel_to_root(cnn_aggregate_json)}")

    if tabular_compare_json.exists():
        tab_cmp = json.loads(tabular_compare_json.read_text(encoding="utf-8"))
        print(f"Loaded tabular compare JSON: {rel_to_root(tabular_compare_json)}")
    else:
        tab_cmp = None
        print(f"WARNING: Tabular compare JSON not found at {rel_to_root(tabular_compare_json)}")
    print()

    deltas_info = {}
    fidelity_ok = True

    # =========================================================================
    # H6: aggregate_across_seeds (CNN B-with-pressure vs A-current)
    # =========================================================================
    print("-" * 80)
    print("H6: Feature Ablation (B vs A, across 3 CNN seeds)")
    print("-" * 80)

    cnn_oof_paths = [cnn_oofs[seed] for seed in [42, 123, 456]]
    missing_cnn = [p for p in cnn_oof_paths if not p.exists()]
    if missing_cnn:
        print(f"ERROR: Missing CNN OOF CSVs: {missing_cnn}")
        sys.exit(1)

    h6_result = aggregate_across_seeds(cnn_oof_paths, n_boot=n_boot, boot_seed=boot_seed,
                                       return_draws=True)
    h6_draws = h6_result["bootstrap_draws"]
    h6_observed = h6_result["delta_pr_auc_mean_across_seeds"]

    print(f"  Observed delta (mean across seeds): {h6_observed:.6f}")
    print(f"  Bootstrap draws: {len(h6_draws)}")
    print(f"  CI: [{h6_result['delta_pr_auc_ci_low']:.6f}, {h6_result['delta_pr_auc_ci_high']:.6f}]")

    # Fidelity check (CI bounds AND point estimate vs the published record)
    pub_h6_ci = None
    if not is_smoke_run and cnn_agg is not None:
        pub_ci_low = cnn_agg.get("delta_pr_auc_ci_low")
        pub_ci_high = cnn_agg.get("delta_pr_auc_ci_high")
        pub_point = cnn_agg.get("delta_pr_auc_mean_across_seeds")
        if pub_ci_low is not None and pub_ci_high is not None:
            pub_h6_ci = [pub_ci_low, pub_ci_high]
            rep_ci_low = h6_result["delta_pr_auc_ci_low"]
            rep_ci_high = h6_result["delta_pr_auc_ci_high"]
            tol = 1e-12
            ci_low_match = abs(rep_ci_low - pub_ci_low) < tol if rep_ci_low is not None else False
            ci_high_match = abs(rep_ci_high - pub_ci_high) < tol if rep_ci_high is not None else False
            point_match = (abs(h6_observed - pub_point) < tol) if pub_point is not None else False
            fidelity_h6 = ci_low_match and ci_high_match and point_match
            print(f"  Fidelity check:")
            print(f"    Published CI: [{pub_ci_low:.6f}, {pub_ci_high:.6f}]")
            print(f"    Reproduced CI: [{rep_ci_low:.6f}, {rep_ci_high:.6f}]")
            print(f"    Point estimate match: {point_match}")
            print(f"    Match (tol={tol}): {fidelity_h6}")
            if not fidelity_h6:
                fidelity_ok = False
                print(f"    FIDELITY GATE FAILED FOR H6!")
        else:
            print(f"  WARNING: Cannot check fidelity (no CI in published JSON)")
    else:
        fidelity_h6 = None

    se_h6 = float(np.std(np.asarray(h6_draws), ddof=1))
    deltas_info["H6"] = {
        "name": "Feature Ablation (CNN B vs A)",
        "observed_delta": h6_observed,
        "replicate_se": se_h6,
        "published_ci": pub_h6_ci,
        "reproduced_ci": [h6_result["delta_pr_auc_ci_low"], h6_result["delta_pr_auc_ci_high"]],
        "n_boot_used": len(h6_draws),
        "n_boot_skipped": h6_result.get("n_boot_skipped_single_class", 0),
        "fidelity_exact": fidelity_h6 if not is_smoke_run else None,
    }
    print()

    # =========================================================================
    # H9/V1: CNN vs GBM_SF (primary tabular baseline)
    # =========================================================================
    print("-" * 80)
    print("H9/V1: CNN vs GBM(state_plus_fields)")
    print("-" * 80)

    # Load all OOF CSVs for both arms
    frames_v1 = []
    for seed in [42, 123, 456]:
        cnn_path = cnn_oofs[seed]
        tab_path = tab_oofs[seed]
        if not cnn_path.exists():
            print(f"ERROR: CNN OOF not found for seed {seed}: {cnn_path}")
            sys.exit(1)
        if not tab_path.exists():
            print(f"ERROR: Tabular OOF not found for seed {seed}: {tab_path}")
            sys.exit(1)

        cnn_df = read_oof_csv(cnn_path).sort_values("event_id").reset_index(drop=True)
        tab_df = read_oof_csv(tab_path).sort_values("event_id").reset_index(drop=True)

        if not (cnn_df["event_id"].to_numpy() == tab_df["event_id"].to_numpy()).all():
            print(f"ERROR: Seed {seed}: event_id mismatch between CNN and tabular OOF")
            sys.exit(1)
        if not np.array_equal(cnn_df["y"].to_numpy(), tab_df["y"].to_numpy()):
            print(f"ERROR: Seed {seed}: label mismatch between CNN and tabular OOF")
            sys.exit(1)

        y = cnn_df["y"].to_numpy()
        prob_cnn = cnn_df["prob_A"].to_numpy()
        prob_gbm_sf = tab_df["prob_gbm_state_plus_fields"].to_numpy()
        prob_gbm_f = tab_df["prob_gbm_fields_only"].to_numpy() if "prob_gbm_fields_only" in tab_df.columns else None
        sid = cnn_df["sid"].to_numpy()

        frames_v1.append({
            "seed": seed,
            "y": y,
            "prob_cnn": prob_cnn,
            "prob_gbm": prob_gbm_sf,
            "prob_f": prob_gbm_f,
            "sid": sid,
        })

    # Extract first frame to establish reference
    y_ref = frames_v1[0]["y"]
    sid_ref = frames_v1[0]["sid"]
    unique_groups = np.unique(sid_ref)
    group_to_idx = {g: np.where(sid_ref == g)[0] for g in unique_groups}

    # Call paired_cross_seed_bootstrap
    boot_result = paired_cross_seed_bootstrap(frames_v1, unique_groups, group_to_idx, y_ref,
                                              n_boot=n_boot, boot_seed=boot_seed)
    v1_draws = boot_result["draws"]
    v2_draws = boot_result["draws_v2"]
    v1_skipped = boot_result["n_skipped"]

    cnn_prs = [average_precision_score(y_ref, f["prob_cnn"]) for f in frames_v1]
    sf_prs = [average_precision_score(y_ref, f["prob_gbm"]) for f in frames_v1]
    v1_observed = float(np.mean([c - s for c, s in zip(cnn_prs, sf_prs)]))

    print(f"  Observed delta (mean across seeds): {v1_observed:.6f}")
    print(f"  Bootstrap draws: {len(v1_draws)}")
    print(f"  Skipped (single-class): {v1_skipped}")

    # Compute CI
    arr_v1 = np.asarray(v1_draws, dtype=np.float64)
    alpha = 0.025
    ci_v1_low = float(np.quantile(arr_v1, alpha)) if arr_v1.size else None
    ci_v1_high = float(np.quantile(arr_v1, 1.0 - alpha)) if arr_v1.size else None
    print(f"  CI: [{ci_v1_low:.6f}, {ci_v1_high:.6f}]")

    # Fidelity check (CI bounds AND point estimate vs the published record)
    pub_v1_ci = None
    if not is_smoke_run and tab_cmp is not None:
        pub_ci_low_v1 = tab_cmp.get("V1_cnn_vs_sf", {}).get("delta_pr_auc_ci_low")
        pub_ci_high_v1 = tab_cmp.get("V1_cnn_vs_sf", {}).get("delta_pr_auc_ci_high")
        pub_point_v1 = tab_cmp.get("V1_cnn_vs_sf", {}).get("delta_pr_auc_point")
        if pub_ci_low_v1 is not None and pub_ci_high_v1 is not None:
            pub_v1_ci = [pub_ci_low_v1, pub_ci_high_v1]
            tol = 1e-12
            ci_low_match = abs(ci_v1_low - pub_ci_low_v1) < tol if ci_v1_low is not None else False
            ci_high_match = abs(ci_v1_high - pub_ci_high_v1) < tol if ci_v1_high is not None else False
            point_match = (abs(v1_observed - pub_point_v1) < tol) if pub_point_v1 is not None else False
            fidelity_v1 = ci_low_match and ci_high_match and point_match
            print(f"  Fidelity check:")
            print(f"    Published CI: [{pub_ci_low_v1:.6f}, {pub_ci_high_v1:.6f}]")
            print(f"    Reproduced CI: [{ci_v1_low:.6f}, {ci_v1_high:.6f}]")
            print(f"    Point estimate match: {point_match}")
            print(f"    Match (tol={tol}): {fidelity_v1}")
            if not fidelity_v1:
                fidelity_ok = False
                print(f"    FIDELITY GATE FAILED FOR H9/V1!")
        else:
            print(f"  WARNING: Cannot check fidelity (no CI in published JSON)")
    else:
        fidelity_v1 = None

    se_v1 = float(np.std(np.asarray(v1_draws), ddof=1)) if v1_draws else 0.0
    deltas_info["H9_V1"] = {
        "name": "CNN vs GBM(state_plus_fields)",
        "observed_delta": v1_observed,
        "replicate_se": se_v1,
        "published_ci": pub_v1_ci,
        "reproduced_ci": [ci_v1_low, ci_v1_high],
        "n_boot_used": len(v1_draws),
        "n_boot_skipped": v1_skipped,
        "fidelity_exact": fidelity_v1 if not is_smoke_run else None,
    }
    print()

    # =========================================================================
    # H9/V2: CNN vs GBM_F (co-primary, same resampling as V1)
    # =========================================================================
    print("-" * 80)
    print("H9/V2: CNN vs GBM(fields_only)")
    print("-" * 80)

    v2_observed = float(np.mean([average_precision_score(y_ref, frames_v1[i]["prob_cnn"]) -
                                  average_precision_score(y_ref, frames_v1[i]["prob_f"])
                                  for i in range(len(frames_v1))
                                  if frames_v1[i]["prob_f"] is not None]))

    print(f"  Observed delta (mean across seeds): {v2_observed:.6f}")
    print(f"  Bootstrap draws: {len(v2_draws)}")
    print(f"  Skipped (single-class): {v1_skipped} (shared with V1)")

    # Compute CI
    arr_v2 = np.asarray(v2_draws, dtype=np.float64)
    ci_v2_low = float(np.quantile(arr_v2, alpha)) if arr_v2.size else None
    ci_v2_high = float(np.quantile(arr_v2, 1.0 - alpha)) if arr_v2.size else None
    print(f"  CI: [{ci_v2_low:.6f}, {ci_v2_high:.6f}]")

    # Fidelity check (CI bounds AND point estimate vs the published record)
    pub_v2_ci = None
    if not is_smoke_run and tab_cmp is not None:
        pub_ci_low_v2 = tab_cmp.get("V2_cnn_vs_f", {}).get("delta_pr_auc_ci_low")
        pub_ci_high_v2 = tab_cmp.get("V2_cnn_vs_f", {}).get("delta_pr_auc_ci_high")
        pub_point_v2 = tab_cmp.get("V2_cnn_vs_f", {}).get("delta_pr_auc_point")
        if pub_ci_low_v2 is not None and pub_ci_high_v2 is not None:
            pub_v2_ci = [pub_ci_low_v2, pub_ci_high_v2]
            tol = 1e-12
            ci_low_match = abs(ci_v2_low - pub_ci_low_v2) < tol if ci_v2_low is not None else False
            ci_high_match = abs(ci_v2_high - pub_ci_high_v2) < tol if ci_v2_high is not None else False
            point_match = (abs(v2_observed - pub_point_v2) < tol) if pub_point_v2 is not None else False
            fidelity_v2 = ci_low_match and ci_high_match and point_match
            print(f"  Fidelity check:")
            print(f"    Published CI: [{pub_ci_low_v2:.6f}, {pub_ci_high_v2:.6f}]")
            print(f"    Reproduced CI: [{ci_v2_low:.6f}, {ci_v2_high:.6f}]")
            print(f"    Point estimate match: {point_match}")
            print(f"    Match (tol={tol}): {fidelity_v2}")
            if not fidelity_v2:
                fidelity_ok = False
                print(f"    FIDELITY GATE FAILED FOR H9/V2!")
        else:
            print(f"  WARNING: Cannot check fidelity (no CI in published JSON)")
    else:
        fidelity_v2 = None

    se_v2 = float(np.std(np.asarray(v2_draws), ddof=1)) if v2_draws else 0.0
    deltas_info["H9_V2"] = {
        "name": "CNN vs GBM(fields_only)",
        "observed_delta": v2_observed,
        "replicate_se": se_v2,
        "published_ci": pub_v2_ci,
        "reproduced_ci": [ci_v2_low, ci_v2_high],
        "n_boot_used": len(v2_draws),
        "n_boot_skipped": v1_skipped,
        "fidelity_exact": fidelity_v2 if not is_smoke_run else None,
    }
    print()

    # =========================================================================
    # PROXY: Basin ablation (S vs SF on tabular arm only)
    # =========================================================================
    print("-" * 80)
    print("PROXY: Basin Ablation Design (S vs SF, tabular arm)")
    print("-" * 80)

    frames_basin = []
    for seed in [42, 123, 456]:
        tab_path = tab_oofs[seed]
        tab_df = read_oof_csv(tab_path).sort_values("event_id").reset_index(drop=True)
        y = tab_df["y"].to_numpy()
        prob_s = tab_df["prob_gbm_state_only"].to_numpy()
        prob_sf = tab_df["prob_gbm_state_plus_fields"].to_numpy()
        sid = tab_df["sid"].to_numpy()

        frames_basin.append({
            "seed": seed,
            "y": y,
            "prob_cnn": prob_s,
            "prob_gbm": prob_sf,
            "prob_f": None,  # Not used for proxy
            "sid": sid,
        })

    basin_result = paired_cross_seed_bootstrap(frames_basin, unique_groups, group_to_idx, y_ref,
                                               n_boot=n_boot, boot_seed=boot_seed)
    basin_draws = basin_result["draws"]
    basin_skipped = basin_result["n_skipped"]

    print(f"  Bootstrap draws: {len(basin_draws)}")
    print(f"  Skipped (single-class): {basin_skipped}")

    arr_basin = np.asarray(basin_draws, dtype=np.float64)
    ci_basin_low = float(np.quantile(arr_basin, alpha)) if arr_basin.size else None
    ci_basin_high = float(np.quantile(arr_basin, 1.0 - alpha)) if arr_basin.size else None
    print(f"  CI (S vs SF): [{ci_basin_low:.6f}, {ci_basin_high:.6f}]")
    print(f"  NOTE: conservative upper bound on SE of planned S-with-basin vs S-without-basin")

    se_basin = float(np.std(np.asarray(basin_draws), ddof=1)) if basin_draws else 0.0
    deltas_info["basin_ablation_design_proxy"] = {
        "name": "Tabular S vs SF (proxy for planned basin ablation)",
        "observed_delta": None,  # Design proxy, no observed experiment
        "replicate_se": se_basin,
        "published_ci": None,
        "reproduced_ci": [ci_basin_low, ci_basin_high],
        "n_boot_used": len(basin_draws),
        "n_boot_skipped": basin_skipped,
        "fidelity_exact": None,
    }
    print()

    # =========================================================================
    # COMPUTE MDE FOR ALL DELTAS
    # =========================================================================
    print("-" * 80)
    print("MDE COMPUTATION")
    print("-" * 80)

    Z975 = 1.959963984540054  # qnorm(0.975)
    Z80 = 0.8416212335729143   # qnorm(0.80)
    Z90 = 1.2815515655446004   # qnorm(0.90)

    sanity_check_results = {}
    for delta_name, delta_dict in deltas_info.items():
        se = delta_dict["replicate_se"]
        observed = delta_dict["observed_delta"]

        mde_80 = (Z975 + Z80) * se
        mde_90 = (Z975 + Z90) * se

        if observed is not None:
            below_mde_80 = abs(observed) < mde_80
        else:
            below_mde_80 = None

        delta_dict["mde_power80"] = mde_80
        delta_dict["mde_power90"] = mde_90
        delta_dict["observed_below_mde_80"] = below_mde_80

        print(f"{delta_name}:")
        print(f"  SE (replicate): {se:.6f}")
        print(f"  MDE@80%: {mde_80:.6f}")
        print(f"  MDE@90%: {mde_90:.6f}")
        if observed is not None:
            print(f"  Observed: {observed:.6f}")
            print(f"  Below MDE@80: {below_mde_80}")
        print()

        # Sanity check
        if delta_name in ["H6", "H9_V2"]:
            expected_range = (0.03, 0.05)
        elif delta_name == "H9_V1":
            expected_range = (0.04, 0.07)
        else:
            expected_range = None

        if expected_range:
            in_range = expected_range[0] <= mde_80 <= expected_range[1]
            sanity_check_results[delta_name] = {
                "mde_80": mde_80,
                "expected_range": expected_range,
                "in_expected_range": in_range,
            }

    # Sanity check summary
    print("-" * 80)
    print("SANITY CHECK: MDE@80 expected ranges")
    print("-" * 80)
    sanity_in_range = True
    for delta_name, check in sanity_check_results.items():
        status = "OK" if check["in_expected_range"] else "WARNING"
        print(f"  {delta_name}: {check['mde_80']:.6f} in {check['expected_range']} - {status}")
        if not check["in_expected_range"]:
            sanity_in_range = False
    print()

    # =========================================================================
    # FIDELITY GATE CHECK
    # =========================================================================
    if is_smoke_run:
        print("SMOKE RUN: skipping fidelity gate")
        fidelity_pass = None
    else:
        print("=" * 80)
        print("FIDELITY GATE RESULT")
        print("=" * 80)
        if fidelity_ok:
            print("PASS: All reproduced CIs match published values (tol=1e-12)")
            fidelity_pass = True
        else:
            print("FAIL: One or more reproduced CIs diverge from published values")
            fidelity_pass = False
            sys.exit(1)

    # =========================================================================
    # WRITE OUTPUT JSON
    # =========================================================================
    output_dir = PROJECT_ROOT / "outputs" / "results" / "power_analysis"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "note": ("Estimator-precision characterization only. No verdict is "
                 "re-read, re-interpreted, or revised. H6 NULL, H9/V1 NEGATIVE, "
                 "H9/V2 NULL all stand."),
        "method": "retrospective_mde_from_published_bootstrap_draws",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "boot_seed": boot_seed,
        "n_boot": n_boot,
        "is_smoke_run": is_smoke_run,
        "fidelity_gate_pass": fidelity_pass,
        "sanity_check_all_in_range": sanity_in_range,
        "input_paths": {
            "cnn_aggregate_json": rel_to_root(cnn_aggregate_json),
            "tabular_compare_json": rel_to_root(tabular_compare_json),
            "cnn_oof_csvs": [rel_to_root(cnn_oofs[seed]) for seed in [42, 123, 456]],
            "tabular_oof_csvs": [rel_to_root(tab_oofs[seed]) for seed in [42, 123, 456]],
        },
        "z_constants": {
            "z975": Z975,
            "z80": Z80,
            "z90": Z90,
        },
        "deltas": {},
    }

    for delta_name, delta_dict in deltas_info.items():
        report["deltas"][delta_name] = {
            "name": delta_dict["name"],
            "replicate_se": delta_dict["replicate_se"],
            "mde_power80": delta_dict["mde_power80"],
            "mde_power90": delta_dict["mde_power90"],
            "observed_delta": delta_dict["observed_delta"],
            "observed_below_mde_80": delta_dict["observed_below_mde_80"],
            "published_ci": delta_dict.get("published_ci"),
            "reproduced_ci": delta_dict.get("reproduced_ci"),
            "n_boot_used": delta_dict.get("n_boot_used"),
            "n_boot_skipped": delta_dict.get("n_boot_skipped"),
            "fidelity_exact": delta_dict["fidelity_exact"],
        }

    report_path = run_dir / "mde_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"Report written to: {rel_to_root(report_path)}")

    # Print summary table
    print()
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Delta':<30} {'SE':<12} {'MDE@80':<12} {'MDE@90':<12} {'Observed':<12}")
    print("-" * 80)
    for delta_name, delta_dict in deltas_info.items():
        obs_str = f"{delta_dict['observed_delta']:.6f}" if delta_dict["observed_delta"] is not None else "N/A"
        print(f"{delta_name:<30} {delta_dict['replicate_se']:<12.6f} "
              f"{delta_dict['mde_power80']:<12.6f} {delta_dict['mde_power90']:<12.6f} {obs_str:<12}")
    print("=" * 80)


if __name__ == "__main__":
    main()
