# analysis/basin_ablation_h10.py
"""
H10 basin ablation — paired comparison and single verdict read.

Pre-registration: docs/basin_ablation_preregistration.md (2026-07-24).
Computes the pre-registered paired delta

    delta_basin = PR-AUC(S with basin) - PR-AUC(S without basin)

where the with-basin arm is the PUBLISHED state_only OOF of run
20260715T123221Z (no retraining) and the without-basin arm is a new run of
the identical harness with --drop-basin. Applies the pre-registered decision
branches, including the two pre-declared degenerate outcomes (point-mass
distribution; proxy failure). The interval is read exactly once, by this
script, after all 3 seeds exist. No other verdict is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import average_precision_score  # noqa: E402

from analysis.tabular_baseline_kfold import paired_cross_seed_bootstrap  # noqa: E402
from src.utils.paths import rel_to_root  # noqa: E402

SEEDS = (42, 123, 456)
PUBLISHED_RUN = "outputs/results/tabular_baseline/20260715T123221Z"
DEFAULT_MDE_REPORT = "outputs/results/power_analysis/20260724T213423Z/mde_report.json"
Z975 = 1.959963984540054
Z80 = 0.8416212335729143
Z90 = 1.2815515655446004


def read_oof(path: Path) -> pd.DataFrame:
    """Project reader discipline: basin code "NA" is North Atlantic."""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    for col in df.columns:
        if col not in ("event_id", "sid"):
            df[col] = pd.to_numeric(df[col])
    return df.sort_values("event_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-basin-run", required=True, metavar="PATH",
                        help="Run dir of the --drop-basin execution (the new arm).")
    parser.add_argument("--mde-report", default=DEFAULT_MDE_REPORT, metavar="PATH",
                        help="mde_report.json carrying the pre-declared proxy SE.")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--boot-seed", type=int, default=42)
    args = parser.parse_args()

    mde_report = json.loads((PROJECT_ROOT / args.mde_report).read_text(encoding="utf-8"))
    proxy_se = float(mde_report["deltas"]["basin_ablation_design_proxy"]["replicate_se"])
    declared_mde80 = (Z975 + Z80) * proxy_se

    without_run = Path(args.without_basin_run)
    if not without_run.is_absolute():
        without_run = PROJECT_ROOT / without_run

    # Guard: the new arm must actually be a --drop-basin run.
    without_summary = json.loads((without_run / "summary.json").read_text(encoding="utf-8"))
    if without_summary.get("drop_basin") is not True:
        print("ABORT: the provided run does not declare drop_basin=true in summary.json.")
        sys.exit(1)
    for set_name, cols in without_summary["feature_sets"].items():
        leaked = [c for c in cols if c.startswith("basin_")]
        if leaked:
            print(f"ABORT: basin columns present in the without-basin arm ({set_name}): {leaked}")
            sys.exit(1)

    frames: List[Dict[str, Any]] = []
    identical_by_seed: Dict[str, bool] = {}
    per_seed_delta: List[float] = []
    for seed in SEEDS:
        with_df = read_oof(PROJECT_ROOT / PUBLISHED_RUN / f"seed{seed}" / "oof_predictions.csv")
        wo_df = read_oof(without_run / f"seed{seed}" / "oof_predictions.csv")

        if not (with_df["event_id"].to_numpy() == wo_df["event_id"].to_numpy()).all():
            raise ValueError(f"Seed {seed}: event sets differ between arms (folds not identical?).")
        if not np.array_equal(with_df["y"].to_numpy(), wo_df["y"].to_numpy()):
            raise ValueError(f"Seed {seed}: labels differ between arms.")

        y = with_df["y"].to_numpy(dtype=np.float64)
        prob_with = with_df["prob_gbm_state_only"].to_numpy(dtype=np.float64)
        prob_without = wo_df["prob_gbm_state_only"].to_numpy(dtype=np.float64)
        identical_by_seed[str(seed)] = bool(np.array_equal(prob_with, prob_without))
        per_seed_delta.append(float(average_precision_score(y, prob_with)
                                    - average_precision_score(y, prob_without)))
        frames.append({
            "seed": seed,
            "y": y,
            "prob_cnn": prob_with,      # slot A of the paired bootstrap = with-basin
            "prob_gbm": prob_without,   # slot B = without-basin; draws = A - B
            "prob_f": None,
            "sid": with_df["sid"].to_numpy(),
        })

    y_ref = frames[0]["y"]
    sid_ref = frames[0]["sid"]
    unique_groups = np.unique(sid_ref)
    group_to_idx = {g: np.where(sid_ref == g)[0] for g in unique_groups}

    boot = paired_cross_seed_bootstrap(frames, unique_groups, group_to_idx, y_ref,
                                       n_boot=args.n_boot, boot_seed=args.boot_seed)
    arr = np.asarray(boot["draws"], dtype=np.float64)
    point = float(np.mean(per_seed_delta))
    ci_low = float(np.quantile(arr, 0.025)) if arr.size else None
    ci_high = float(np.quantile(arr, 0.975)) if arr.size else None
    realized_se = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    realized_mde80 = (Z975 + Z80) * realized_se
    realized_mde90 = (Z975 + Z90) * realized_se

    all_identical = all(identical_by_seed.values())
    proxy_failed = realized_se > proxy_se
    near_degenerate = realized_se < 0.1 * proxy_se

    # Pre-registered branches, in the pre-registered order of precedence.
    if all_identical:
        branch = "degenerate_point_mass"
        verdict = ("ZERO measurable contribution by construction: the with-basin and "
                   "without-basin arms produced event-wise identical OOF predictions "
                   "(verified column-wise, all seeds). Exact equality, not lack of power.")
    elif ci_low is None:
        branch = "inconclusive"
        verdict = "INCONCLUSIVE: insufficient valid bootstrap draws."
    elif ci_low > 0:
        branch = "ci_low_gt_0"
        verdict = (f"Basin contributes measurably to the S rung "
                   f"(delta {point:+.4f}, CI [{ci_low:.4f}, {ci_high:.4f}] > 0). "
                   "Pre-registered consequence: the ladder reports the S rung BOTH ways, "
                   "and the paper states that 'storm state only' includes basin membership.")
    elif ci_high < 0:
        branch = "ci_high_lt_0"
        verdict = (f"UNEXPECTED direction (delta {point:+.4f}, CI [{ci_low:.4f}, "
                   f"{ci_high:.4f}] < 0): removing basin IMPROVED the S arm. "
                   "Reported as-is per the pre-registration; no rationalization.")
    else:
        mde_ref = realized_mde80 if proxy_failed else declared_mde80
        branch = "ci_contains_0"
        verdict = (f"Basin contribution NOT detectable at this design's resolution "
                   f"(delta {point:+.4f}, CI [{ci_low:.4f}, {ci_high:.4f}] contains 0; "
                   f"MDE@80 {mde_ref:.4f}). The S rung stands as published, with the MDE "
                   "stating the scope of this null.")

    result = {
        "hypothesis": "H10",
        "preregistration": "docs/basin_ablation_preregistration.md",
        "read_once": ("This file is the single authorized read of the H10 interval "
                      "(read-once rule). No re-runs, no re-thresholding."),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "with_basin_run": PUBLISHED_RUN,
        "without_basin_run": rel_to_root(without_run),
        "n_boot_requested": args.n_boot,
        "n_boot_used": int(arr.size),
        "n_boot_skipped": boot["n_skipped"],
        "boot_seed": args.boot_seed,
        "per_seed_delta_pr_auc": per_seed_delta,
        "delta_pr_auc_point": point,
        "delta_pr_auc_ci_low": ci_low,
        "delta_pr_auc_ci_high": ci_high,
        "predictions_identical_by_seed": identical_by_seed,
        "declared_proxy_se": proxy_se,
        "declared_design_mde80": declared_mde80,
        "realized_se": realized_se,
        "realized_mde80": realized_mde80,
        "realized_mde90": realized_mde90,
        "proxy_assumption_failed": proxy_failed,
        "near_degenerate_distribution": near_degenerate,
        "branch": branch,
        "verdict": verdict,
    }

    out_dir = (PROJECT_ROOT / "outputs" / "results" / "basin_ablation"
               / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "h10_result.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nVERDICT (H10, read once): {verdict}")
    print(f"result: {rel_to_root(out_path)}")


if __name__ == "__main__":
    main()
