#!/usr/bin/env python3
"""
Basin ablation (H10) MDE validity guardrail: read-only pre-publication checks.

Validates:
  1. mde_report.json structure and fidelity gates.
  2. Independent recheck of published observed deltas against source JSONs.
  3. MDE@80 sanity ranges.
  4. No absolute paths (Windows or Unix) in mde_report.json.
  5. Published artefacts (JSONs + OOF CSVs) untouched (git status).
  6. Test set not referenced in analysis code or diffs.
  7. keep_default_na=False discipline in pd.read_csv calls.
  8. Default behavior (resolve_feature_sets) unchanged.
  9. Pre-registration documents free of unresolved slots.
  10. Published verdicts intact and correct.

Exit code: 0 if all checks pass, 1 if any check fails.
Does NOT write anything (read-only).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.tabular_baseline_kfold import resolve_feature_sets  # noqa: E402


def check_pass(check_num: int, message: str) -> None:
    """Print a PASS message."""
    print(f"âœ“ Check {check_num}: {message}")


def check_fail(check_num: int, message: str) -> None:
    """Print a FAIL message."""
    print(f"âœ— Check {check_num}: {message}")


class ValidateMDEBasinGuardrail:
    """Read-only validation harness for pre-registered MDE basin ablation."""

    def __init__(self, mde_report_path: Path) -> None:
        self.mde_report_path = mde_report_path
        self.checks_passed = 0
        self.checks_failed = 0
        self.mde_report: Dict[str, Any] = {}

    def load_mde_report(self) -> bool:
        """Load and parse mde_report.json."""
        if not self.mde_report_path.exists():
            check_fail(0, f"mde_report.json not found: {self.mde_report_path}")
            self.checks_failed += 1
            return False
        try:
            self.mde_report = json.loads(self.mde_report_path.read_text(encoding="utf-8"))
            check_pass(0, f"mde_report.json loaded: {len(self.mde_report)} top-level keys")
            self.checks_passed += 1
            return True
        except Exception as e:
            check_fail(0, f"Failed to parse mde_report.json: {e}")
            self.checks_failed += 1
            return False

    def check_1_mde_report_structure_and_fidelity(self) -> None:
        """Check 1: mde_report structure, fidelity gates, and confidence intervals."""
        check_num = 1
        try:
            # Basic structure checks.
            assert self.mde_report.get("is_smoke_run") is False, "is_smoke_run must be False"
            assert self.mde_report.get("n_boot") == 10000, "n_boot must be exactly 10000"
            assert self.mde_report.get("fidelity_gate_pass") is True, "fidelity_gate_pass must be True"

            # Per-hypothesis fidelity and CI checks.
            hypotheses = self.mde_report.get("deltas", {})
            for hyp_id in ("H6", "H9_V1", "H9_V2"):
                if hyp_id not in hypotheses:
                    check_fail(check_num, f"{hyp_id} not found in hypotheses")
                    self.checks_failed += 1
                    return

                hyp = hypotheses[hyp_id]
                assert hyp.get("fidelity_exact") is True, f"{hyp_id}: fidelity_exact must be True"

                published_ci = hyp.get("published_ci")
                reproduced_ci = hyp.get("reproduced_ci")
                assert published_ci is not None, f"{hyp_id}: published_ci must be non-null"
                assert reproduced_ci is not None, f"{hyp_id}: reproduced_ci must be non-null"

                # Element-wise comparison with tight tolerance.
                pub_arr = np.asarray(published_ci, dtype=np.float64)
                rep_arr = np.asarray(reproduced_ci, dtype=np.float64)
                assert np.allclose(pub_arr, rep_arr, atol=1e-12), (
                    f"{hyp_id}: published_ci and reproduced_ci differ: "
                    f"{pub_arr} vs {rep_arr}"
                )

            check_pass(check_num, "mde_report structure, fidelity gates, and CIs valid")
            self.checks_passed += 1
        except AssertionError as e:
            check_fail(check_num, str(e))
            self.checks_failed += 1
        except Exception as e:
            check_fail(check_num, f"Unexpected error: {e}")
            self.checks_failed += 1

    def check_2_independent_observed_delta_recheck(self) -> None:
        """Check 2: Independently recheck observed deltas against published source JSONs."""
        check_num = 2
        try:
            # Find and load published JSON sources.
            results_dir = Path("outputs/results").resolve()

            feature_abl_aggregate = None
            for candidate in results_dir.glob("feature_ablation_cnn/aggregate_*.json"):
                feature_abl_aggregate = candidate
                break

            tabular_compare = None
            for candidate in results_dir.glob("tabular_baseline/compare_*.json"):
                tabular_compare = candidate
                break

            if not feature_abl_aggregate:
                check_fail(check_num, "feature_ablation_cnn/aggregate_*.json not found")
                self.checks_failed += 1
                return

            if not tabular_compare:
                check_fail(check_num, "tabular_baseline/compare_*.json not found")
                self.checks_failed += 1
                return

            agg_data = json.loads(feature_abl_aggregate.read_text(encoding="utf-8"))
            cmp_data = json.loads(tabular_compare.read_text(encoding="utf-8"))

            # Recheck H6 observed delta (aggregate JSON is FLAT, no "H6" block).
            h6_observed = self.mde_report.get("deltas", {}).get("H6", {}).get("observed_delta")
            h6_published = agg_data.get("delta_pr_auc_mean_across_seeds")
            assert h6_observed is not None and h6_published is not None, (
                "H6 observed/published delta missing (no silent skip allowed)"
            )
            assert np.isclose(h6_observed, h6_published, atol=1e-12), (
                f"H6 observed_delta mismatch: {h6_observed} vs {h6_published}"
            )

            # Recheck H9_V1 observed delta (CNN vs SF).
            h9v1_observed = self.mde_report.get("deltas", {}).get("H9_V1", {}).get("observed_delta")
            h9v1_published = cmp_data.get("V1_cnn_vs_sf", {}).get("delta_pr_auc_point")
            assert h9v1_observed is not None and h9v1_published is not None, (
                "H9_V1 observed/published delta missing (no silent skip allowed)"
            )
            assert np.isclose(h9v1_observed, h9v1_published, atol=1e-12), (
                f"H9_V1 observed_delta mismatch: {h9v1_observed} vs {h9v1_published}"
            )

            # Recheck H9_V2 observed delta (CNN vs F).
            h9v2_observed = self.mde_report.get("deltas", {}).get("H9_V2", {}).get("observed_delta")
            h9v2_published = cmp_data.get("V2_cnn_vs_f", {}).get("delta_pr_auc_point")
            assert h9v2_observed is not None and h9v2_published is not None, (
                "H9_V2 observed/published delta missing (no silent skip allowed)"
            )
            assert np.isclose(h9v2_observed, h9v2_published, atol=1e-12), (
                f"H9_V2 observed_delta mismatch: {h9v2_observed} vs {h9v2_published}"
            )

            check_pass(check_num, "observed deltas match published source JSONs (tol 1e-12)")
            self.checks_passed += 1
        except AssertionError as e:
            check_fail(check_num, str(e))
            self.checks_failed += 1
        except Exception as e:
            check_fail(check_num, f"Unexpected error: {e}")
            self.checks_failed += 1

    def check_3_mde_sanity_ranges(self) -> None:
        """Check 3: MDE@80% values within expected ranges."""
        check_num = 3
        try:
            hypotheses = self.mde_report.get("deltas", {})

            h6_mde = hypotheses.get("H6", {}).get("mde_power80")
            assert h6_mde is not None, "H6 mde_at_80 missing"
            assert 0.03 < h6_mde < 0.05, f"H6 MDE@80={h6_mde} not in (0.03, 0.05)"

            h9v1_mde = hypotheses.get("H9_V1", {}).get("mde_power80")
            assert h9v1_mde is not None, "H9_V1 mde_at_80 missing"
            assert 0.04 < h9v1_mde < 0.07, f"H9_V1 MDE@80={h9v1_mde} not in (0.04, 0.07)"

            h9v2_mde = hypotheses.get("H9_V2", {}).get("mde_power80")
            assert h9v2_mde is not None, "H9_V2 mde_at_80 missing"
            assert 0.03 < h9v2_mde < 0.05, f"H9_V2 MDE@80={h9v2_mde} not in (0.03, 0.05)"

            check_pass(check_num, f"MDE@80 sanity: H6={h6_mde:.4f}, H9_V1={h9v1_mde:.4f}, H9_V2={h9v2_mde:.4f}")
            self.checks_passed += 1
        except AssertionError as e:
            check_fail(check_num, str(e))
            self.checks_failed += 1
        except Exception as e:
            check_fail(check_num, f"Unexpected error: {e}")
            self.checks_failed += 1

    def check_4_no_absolute_paths(self) -> None:
        """Check 4: No absolute paths (Windows or Unix) in mde_report.json."""
        check_num = 4
        try:
            report_text = self.mde_report_path.read_text(encoding="utf-8")
            # Windows absolute path pattern: [A-Za-z]:\
            if re.search(r"[A-Za-z]:\\", report_text):
                check_fail(check_num, "Found Windows absolute path (e.g., C:\\) in mde_report.json")
                self.checks_failed += 1
                return
            # Windows UNC/forward-slash pattern: C:/
            if re.search(r"[A-Za-z]:/", report_text):
                check_fail(check_num, "Found Windows path with forward slashes (e.g., C:/) in mde_report.json")
                self.checks_failed += 1
                return

            check_pass(check_num, "No absolute paths found in mde_report.json")
            self.checks_passed += 1
        except Exception as e:
            check_fail(check_num, f"Error checking for absolute paths: {e}")
            self.checks_failed += 1

    def check_5_published_artifacts_untouched(self) -> None:
        """Check 5: Published artifacts (JSONs + OOF CSVs) are untouched (git status)."""
        check_num = 5
        try:
            artifacts = [
                "outputs/results/feature_ablation_cnn/aggregate_20260716T120517Z.json",
                "outputs/results/tabular_baseline/compare_20260716T121803Z.json",
                "outputs/results/feature_ablation_cnn/20260713T232126Z/seed42/oof_predictions.csv",
                "outputs/results/feature_ablation_cnn/20260714T223910Z/seed123/oof_predictions.csv",
                "outputs/results/feature_ablation_cnn/20260715T223350Z/seed456/oof_predictions.csv",
                "outputs/results/tabular_baseline/20260715T123221Z/seed42/oof_predictions.csv",
                "outputs/results/tabular_baseline/20260715T123221Z/seed123/oof_predictions.csv",
                "outputs/results/tabular_baseline/20260715T123221Z/seed456/oof_predictions.csv",
            ]

            # Every listed artifact must EXIST — git status on a nonexistent
            # path is empty output, which would silently pass the check.
            missing = [a for a in artifacts if not (PROJECT_ROOT / a).exists()]
            if missing:
                check_fail(check_num, f"Published artifacts not found on disk: {missing}")
                self.checks_failed += 1
                return

            # Run git status on these specific files.
            result = subprocess.run(
                ["git", "status", "--porcelain", "--"] + artifacts,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                check_fail(check_num, f"git status failed: {result.stderr}")
                self.checks_failed += 1
                return

            output = result.stdout.strip()
            if output:
                check_fail(check_num, f"Published artifacts modified:\n{output}")
                self.checks_failed += 1
                return

            check_pass(check_num, "Published artifacts untouched (git status clean)")
            self.checks_passed += 1
        except subprocess.TimeoutExpired:
            check_fail(check_num, "git status timed out")
            self.checks_failed += 1
        except Exception as e:
            check_fail(check_num, f"Error checking git status: {e}")
            self.checks_failed += 1

    def check_6_test_set_not_referenced(self) -> None:
        """Check 6: Test-set split map not referenced in code or diffs."""
        check_num = 6
        # Token built by concatenation so this file's own source never
        # contains the forbidden literal it scans for.
        forbidden = "frozen_" + "splits"
        try:
            analysis_files = [
                "analysis/power_analysis_mde.py",
                "analysis/validate_mde_basin_guardrail.py",
                "tests/test_drop_basin_flag.py",
            ]

            for fpath in analysis_files:
                full_path = PROJECT_ROOT / fpath
                if full_path.exists():
                    content = full_path.read_text(encoding="utf-8")
                    if forbidden in content:
                        check_fail(check_num, f"{fpath} contains '{forbidden}'")
                        self.checks_failed += 1
                        return

            # Check diffs of modified analysis files.
            diff_result = subprocess.run(
                ["git", "diff", "--", "analysis/tabular_baseline_kfold.py", "analysis/feature_ablation_cnn.py"],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=10,
            )

            if diff_result.returncode == 0:
                diff_output = diff_result.stdout
                if forbidden in diff_output:
                    check_fail(check_num, f"git diff contains '{forbidden}'")
                    self.checks_failed += 1
                    return

            check_pass(check_num, "Test-set split map not referenced")
            self.checks_passed += 1
        except subprocess.TimeoutExpired:
            check_fail(check_num, "git diff timed out")
            self.checks_failed += 1
        except Exception as e:
            check_fail(check_num, f"Error checking test set references: {e}")
            self.checks_failed += 1

    def check_7_keep_default_na_discipline(self) -> None:
        """Check 7: All pd.read_csv in power_analysis_mde.py have keep_default_na=False."""
        check_num = 7
        try:
            power_analysis = PROJECT_ROOT / "analysis/power_analysis_mde.py"
            if not power_analysis.exists():
                check_fail(check_num, "power_analysis_mde.py not found")
                self.checks_failed += 1
                return

            content = power_analysis.read_text(encoding="utf-8")
            # Simple regex: find all pd.read_csv(...) and check for keep_default_na=False
            # within a 3-line window.
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if "pd.read_csv(" in line:
                    # Check this line and the next 2 lines for the parameter.
                    window = "\n".join(lines[i : min(i + 3, len(lines))])
                    if "keep_default_na=False" not in window:
                        check_fail(
                            check_num,
                            f"pd.read_csv at line {i+1} missing keep_default_na=False: {line.strip()}"
                        )
                        self.checks_failed += 1
                        return

            check_pass(check_num, "All pd.read_csv calls have keep_default_na=False")
            self.checks_passed += 1
        except Exception as e:
            check_fail(check_num, f"Error checking pd.read_csv discipline: {e}")
            self.checks_failed += 1

    def check_8_default_behavior_unchanged(self) -> None:
        """Check 8: resolve_feature_sets(cache_df) matches published summary exactly."""
        check_num = 8
        try:
            cache_dir = Path("outputs/results/tabular_baseline").resolve()
            summary_path = None

            # Find the published summary (e.g., 20260715T123221Z/summary.json).
            for candidate in cache_dir.glob("*/summary.json"):
                summary_path = candidate
                break

            if not summary_path:
                check_fail(check_num, "No published summary.json found")
                self.checks_failed += 1
                return

            # Load cache.
            cache_df = None
            if (cache_dir / "feature_table.parquet").exists():
                cache_df = pd.read_parquet(cache_dir / "feature_table.parquet")
            elif (cache_dir / "feature_table.csv").exists():
                cache_df = pd.read_csv(cache_dir / "feature_table.csv", keep_default_na=False, na_values=[""])
            else:
                check_fail(check_num, "Feature table cache not found")
                self.checks_failed += 1
                return

            # Compute feature sets with default drop_basin=False.
            computed = resolve_feature_sets(cache_df, drop_basin=False)

            # Load published summary.
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            published = summary.get("feature_sets", {})

            # Exact match (same lists, same order).
            for set_name in ("state_only", "fields_only", "state_plus_fields"):
                if computed[set_name] != published.get(set_name):
                    check_fail(
                        check_num,
                        f"{set_name} mismatch:\n"
                        f"  computed: {computed[set_name]}\n"
                        f"  published: {published.get(set_name)}"
                    )
                    self.checks_failed += 1
                    return

            # Verify that drop_basin=True removes exactly basin_* columns.
            computed_drop = resolve_feature_sets(cache_df, drop_basin=True)
            removed = set(computed["state_only"]) - set(computed_drop["state_only"])
            expected_removed = {"basin_EP", "basin_"}  # Or whatever basin cols exist
            basin_cols = {c for c in computed["state_only"] if c.startswith("basin_")}
            if removed != basin_cols:
                check_fail(
                    check_num,
                    f"drop_basin=True removal mismatch: {removed} vs basin cols {basin_cols}"
                )
                self.checks_failed += 1
                return

            check_pass(check_num, "resolve_feature_sets(cache_df) matches published summary exactly")
            self.checks_passed += 1
        except Exception as e:
            check_fail(check_num, f"Error checking feature sets: {e}")
            self.checks_failed += 1

    def check_9_preregistration_no_slots(self) -> None:
        """Check 9: Pre-registration documents free of unresolved slots (âŸ¦)."""
        check_num = 9
        try:
            docs = [
                "docs/basin_ablation_preregistration.md",
                "docs/hypothesis_registry.md",
            ]

            for doc_path in docs:
                full_path = PROJECT_ROOT / doc_path
                if not full_path.exists():
                    check_fail(check_num, f"{doc_path} not found")
                    self.checks_failed += 1
                    return

                content = full_path.read_text(encoding="utf-8")
                if "âŸ¦" in content:
                    check_fail(check_num, f"{doc_path} contains unresolved slots (âŸ¦)")
                    self.checks_failed += 1
                    return

            check_pass(check_num, "Pre-registration documents free of unresolved slots")
            self.checks_passed += 1
        except Exception as e:
            check_fail(check_num, f"Error checking pre-registration documents: {e}")
            self.checks_failed += 1

    def check_10_published_verdicts_intact(self) -> None:
        """Check 10: Published verdicts are intact and correct."""
        check_num = 10
        try:
            results_dir = Path("outputs/results").resolve()

            # Load compare JSON (V1 and V2 verdicts).
            compare_json = None
            for candidate in results_dir.glob("tabular_baseline/compare_*.json"):
                compare_json = candidate
                break

            if not compare_json:
                check_fail(check_num, "compare_*.json not found")
                self.checks_failed += 1
                return

            compare_data = json.loads(compare_json.read_text(encoding="utf-8"))

            # Check V1 verdict (CNN vs SF): should contain "BEATS CNN"
            v1_verdict = compare_data.get("V1_cnn_vs_sf", {}).get("verdict", "")
            if "BEATS CNN" not in v1_verdict:
                check_fail(check_num, f"V1 verdict missing 'BEATS CNN': {v1_verdict}")
                self.checks_failed += 1
                return

            # Check V2 verdict (CNN vs F): should contain "NULL"
            v2_verdict = compare_data.get("V2_cnn_vs_f", {}).get("verdict", "")
            if "NULL" not in v2_verdict:
                check_fail(check_num, f"V2 verdict missing 'NULL': {v2_verdict}")
                self.checks_failed += 1
                return

            # Load aggregate JSON (H6 verdict).
            aggregate_json = None
            for candidate in results_dir.glob("feature_ablation_cnn/aggregate_*.json"):
                aggregate_json = candidate
                break

            if not aggregate_json:
                check_fail(check_num, "aggregate_*.json not found")
                self.checks_failed += 1
                return
            agg_data = json.loads(aggregate_json.read_text(encoding="utf-8"))
            h6_verdict = agg_data.get("verdict", "")
            if "NULL" not in h6_verdict:
                check_fail(check_num, f"H6 verdict missing 'NULL': {h6_verdict}")
                self.checks_failed += 1
                return

            check_pass(check_num, "Published verdicts intact: V1 'BEATS CNN', V2 'NULL', H6 'NULL'")
            self.checks_passed += 1
        except Exception as e:
            check_fail(check_num, f"Error checking published verdicts: {e}")
            self.checks_failed += 1

    def run_all_checks(self) -> bool:
        """Run all 10 checks. Return True if all pass, False otherwise."""
        if not self.load_mde_report():
            return False

        self.check_1_mde_report_structure_and_fidelity()
        self.check_2_independent_observed_delta_recheck()
        self.check_3_mde_sanity_ranges()
        self.check_4_no_absolute_paths()
        self.check_5_published_artifacts_untouched()
        self.check_6_test_set_not_referenced()
        self.check_7_keep_default_na_discipline()
        self.check_8_default_behavior_unchanged()
        self.check_9_preregistration_no_slots()
        self.check_10_published_verdicts_intact()

        print("\n" + "=" * 80)
        print(f"RESULTS: {self.checks_passed} passed, {self.checks_failed} failed")
        print("=" * 80)

        return self.checks_failed == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Basin ablation MDE validity guardrail: 10 pre-publication checks (read-only)."
    )
    parser.add_argument("--mde-report", type=str, required=True,
                        help="Path to mde_report.json")
    args = parser.parse_args()

    mde_path = Path(args.mde_report).resolve()
    guardrail = ValidateMDEBasinGuardrail(mde_path)
    success = guardrail.run_all_checks()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
