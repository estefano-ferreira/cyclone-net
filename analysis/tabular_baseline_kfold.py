# analysis/tabular_baseline_kfold.py
"""
SHIPS-like tabular baseline (H9): HistGradientBoosting + LogisticRegression on
scalar features, compared against CNN arm A predictions.

Question: do CNN spatial-temporal features add skill beyond scalar SHIPS-like
predictors? This script trains a tabular baseline (HistGradientBoostingClassifier
as primary, LogisticRegression as reference) on the same SID-grouped dev set as
the CNN feature ablation, then compares their held-out predictions.

Design (fairness and leakage controls):
  * Development set only: events in train+val splits; test never touched.
  * StratifiedGroupKFold grouped by SID: storms stay together (no storm leakage),
    stratified on RI label. Identical folds as CNN feature ablation.
  * FACTORIAL feature sets (decompose information content vs model class —
    an undecomposed CNN-vs-tabular comparison confounds the two):
    - S  (state_only):        wind_kt, pressure_mb, lat, basin, day-of-year,
                              persistence dv_past_12h/24h. No field info.
    - F  (fields_only):       mean/std/min/max of the 11 cube channels
                              (9 production + shear/rh). No state info —
                              the tabular counterpart of the CNN's diet.
    - SF (state_plus_fields): union; the PRIMARY baseline vs the CNN.
    Feature cache to parquet for reuse.
  * Models per seed per fold: HistGradientBoostingClassifier on each of
    S/F/SF (primary, no hyperparameter search) + LogisticRegression on SF
    (linear reference).
  * Per-seed OOF predictions (event held-out once per seed), then
    cluster (by-SID) bootstrap CI of GBM PR-AUC (or comparison delta vs CNN).
  * Gate: refuses to run (unless --no-require-gate) until
    outputs/results/pl_gate_census.json reports gate_pass=true.

Uncertainty quantification:
  * Pooled OOF: every dev event gets one prediction per model per seed.
  * Single-model CI: SID-cluster bootstrap on GBM PR-AUC across resampled storms.
  * Paired CI (--compare-cnn): delta PR-AUC(CNN) - PR-AUC(GBM) with cross-seed
    aggregation. CI crossing zero: NULL (no CNN advantage); CI > 0: CNN has skill;
    CI < 0: tabular baseline beats CNN (investigate).

Usage:
    # Safe by default — prints plan only.
    python analysis/tabular_baseline_kfold.py

    # Feature extraction only (allowed compute today).
    python analysis/tabular_baseline_kfold.py --build-features-only

    # Actually train (expensive; seeds x folds x 2 models).
    python analysis/tabular_baseline_kfold.py --folds 3 --seeds 42,123,456 --execute

    # Compare CNN arm A against an existing tabular run.
    python analysis/tabular_baseline_kfold.py --compare-cnn /path/to/feature_ablation_cnn/runid/seed42 --tabular /path/to/tabular_baseline/runid
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from analysis.feature_ablation_cnn import build_folds, load_pl_gated_dev_events  # noqa: E402
from analysis.feature_ablation_kfold import cluster_bootstrap_ci, enforce_gate  # noqa: E402
from src.utils.config import cfg_get, load_config  # noqa: E402
from src.utils.paths import rel_to_root  # noqa: E402

# Channels aggregated into the fields-only (F) tabular arm: the 9 production
# input channels plus the two pressure-level channels present in the cubes.
FIELD_CHANNELS = [
    "sst_K", "mslp_Pa", "u10_mps", "v10_mps", "wind_mps",
    "vort_1ps", "div_1ps", "grad_mslp_Pa_per_m", "sst_anom_K",
    "shear_850_200_mps", "rh_mid",
]
FIELD_STATS = ("mean", "std", "min", "max")

# Factorial feature sets: S (storm state, no field info), F (field aggregates,
# no state info), SF (union). SF is the primary baseline against the CNN; the
# S/F arms decompose information content vs model class (see pre-registration
# amendment of 2026-07-14).
FEATURE_SET_NAMES = ("state_only", "fields_only", "state_plus_fields")


def resolve_feature_sets(features_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Split feature columns into the factorial S / F / SF sets by prefix."""
    all_cols = [c for c in features_df.columns if c not in ("event_id", "sid", "ri_label")]
    field_cols = [c for c in all_cols if c.startswith("cube_")]
    state_cols = [c for c in all_cols if not c.startswith("cube_")]
    if not field_cols or not state_cols:
        raise ValueError("feature table lacks one of the factorial groups "
                         "(state/cube_) -- rebuild the cache (--rebuild-features)")
    return {
        "state_only": state_cols,
        "fields_only": field_cols,
        "state_plus_fields": state_cols + field_cols,
    }


def _single_model_cluster_ci(y: np.ndarray, prob: np.ndarray, groups: np.ndarray,
                             seed: int, n_boot: int, ci: float = 0.95) -> Dict[str, Any]:
    """SID-cluster percentile bootstrap CI of a single model's PR-AUC."""
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}
    draws: List[float] = []
    n_skipped = 0
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            n_skipped += 1
            continue
        draws.append(float(average_precision_score(yb, prob[idx])))
    arr = np.asarray(draws, dtype=np.float64)
    alpha = (1.0 - ci) / 2.0
    return {
        "pr_auc_ci_low": float(np.quantile(arr, alpha)) if arr.size else None,
        "pr_auc_ci_high": float(np.quantile(arr, 1.0 - alpha)) if arr.size else None,
        "n_boot_used": int(arr.size),
        "n_boot_skipped_single_class": n_skipped,
    }


def build_feature_table(cfg: Dict[str, Any], events_df: pd.DataFrame,
                        rebuild_features: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Extract scalar features from interim data with disk cache.

    Cache path: outputs/results/tabular_baseline/feature_table.parquet (fallback .csv).
    Features:
      - Scalar metadata: wind_kt, pressure_mb, center_lat, abs_lat, basin (one-hot).
      - Temporal: timestamp → day-of-year → sin/cos encoding.
      - Persistence: dv_past_12h, dv_past_24h (from same-sid time series).
      - Cube-mean scalars: spatial-temporal mean of wind_mps, shear_850_200_mps,
        rh_mid, sst_K, sst_anom_K. If missing: NaN→0 + indicator.
    """
    cache_dir = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve() / "tabular_baseline"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path_parquet = cache_dir / "feature_table.parquet"
    cache_path_csv = cache_dir / "feature_table.csv"

    # Try to load cache if it covers all events.
    if not rebuild_features and cache_path_parquet.exists():
        try:
            cached = pd.read_parquet(cache_path_parquet)
            if set(cached["event_id"]) == set(events_df["event_id"]):
                return cached, {"source": "parquet_cache", "n_rows": len(cached)}
        except Exception:
            pass
    if not rebuild_features and cache_path_csv.exists():
        try:
            cached = pd.read_csv(cache_path_csv)
            if set(cached["event_id"]) == set(events_df["event_id"]):
                return cached, {"source": "csv_cache", "n_rows": len(cached)}
        except Exception:
            pass

    interim = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    rows = []
    basin_map = {}  # Build basin one-hot mapping on the fly.

    # Step 1: Single-pass collect metadata and plan persistence joins.
    event_meta = {}
    for r in events_df.itertuples():
        meta_path = interim / f"{r.event_id}.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        event_meta[r.event_id] = {"sid": r.sid, "meta": meta, "ri_label": r.ri_label}
        if "basin" in meta:
            b = meta["basin"]
            if b not in basin_map:
                basin_map[b] = len(basin_map)

    # Build SID→events mapping for persistence feature construction.
    sid_to_events = {}
    for eid, info in event_meta.items():
        sid = info["sid"]
        if sid not in sid_to_events:
            sid_to_events[sid] = []
        sid_to_events[sid].append((eid, info))

    # Sort each SID's events by timestamp to enable past-relative lookups.
    for sid in sid_to_events:
        sid_to_events[sid].sort(
            key=lambda x: x[1]["meta"].get("timestamp", ""),
            reverse=False
        )

    # Step 2: Build feature rows.
    print(f"Building feature table from {len(event_meta)} interim artifacts...")
    for i, (event_id, info) in enumerate(event_meta.items()):
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1} events...")

        meta = info["meta"]
        sid = info["sid"]

        # --- Scalar metadata ---
        wind_kt_val = meta.get("wind_kt")
        if wind_kt_val is None:
            wind_kt_val = np.nan
        else:
            wind_kt_val = float(wind_kt_val)

        pressure_mb_val = meta.get("pressure_mb")
        if pressure_mb_val is None:
            pressure_mb_val = np.nan
        else:
            pressure_mb_val = float(pressure_mb_val)

        center_lat_val = meta.get("center_lat")
        if center_lat_val is None:
            center_lat_val = np.nan
        else:
            center_lat_val = float(center_lat_val)

        row = {
            "event_id": event_id,
            "sid": sid,
            "ri_label": info["ri_label"],
            "wind_kt": wind_kt_val,
            "pressure_mb": pressure_mb_val,
            "center_lat": center_lat_val,
            "abs_lat": abs(center_lat_val) if not np.isnan(center_lat_val) else np.nan,
        }

        # --- Basin one-hot ---
        for b_name in basin_map:
            row[f"basin_{b_name}"] = 1.0 if meta.get("basin") == b_name else 0.0

        # --- Timestamp: day-of-year sin/cos ---
        ts_str = meta.get("timestamp", "")
        doy = None
        if ts_str:
            try:
                dt = pd.Timestamp(ts_str)
                doy = dt.dayofyear
            except Exception:
                pass
        if doy is not None:
            angle = 2.0 * np.pi * doy / 365.0
            row["doy_sin"] = float(np.sin(angle))
            row["doy_cos"] = float(np.cos(angle))
        else:
            row["doy_sin"] = np.nan
            row["doy_cos"] = np.nan

        # --- Persistence features (dv_past_12h, dv_past_24h) ---
        # Find this event's position in its SID's sorted time series.
        sid_events = sid_to_events.get(sid, [])
        event_idx = None
        for idx, (eid, _) in enumerate(sid_events):
            if eid == event_id:
                event_idx = idx
                break

        row["dv_past_12h"] = np.nan
        row["dv_past_12h_missing"] = 0.0
        row["dv_past_24h"] = np.nan
        row["dv_past_24h_missing"] = 0.0

        if event_idx is not None and event_idx > 0:
            curr_wind = float(meta.get("wind_kt", np.nan))
            curr_ts = pd.Timestamp(meta.get("timestamp", ""))

            # Look for an event ~12h earlier.
            for prev_idx in range(event_idx - 1, -1, -1):
                prev_eid, prev_info = sid_events[prev_idx]
                prev_ts = pd.Timestamp(prev_info["meta"].get("timestamp", ""))
                hours_delta = (curr_ts - prev_ts).total_seconds() / 3600.0
                if 10 <= hours_delta <= 14:
                    prev_wind = float(prev_info["meta"].get("wind_kt", np.nan))
                    if not np.isnan(curr_wind) and not np.isnan(prev_wind):
                        row["dv_past_12h"] = curr_wind - prev_wind
                    else:
                        row["dv_past_12h"] = 0.0
                        row["dv_past_12h_missing"] = 1.0
                    break

            # Look for an event ~24h earlier.
            for prev_idx in range(event_idx - 1, -1, -1):
                prev_eid, prev_info = sid_events[prev_idx]
                prev_ts = pd.Timestamp(prev_info["meta"].get("timestamp", ""))
                hours_delta = (curr_ts - prev_ts).total_seconds() / 3600.0
                if 22 <= hours_delta <= 26:
                    prev_wind = float(prev_info["meta"].get("wind_kt", np.nan))
                    if not np.isnan(curr_wind) and not np.isnan(prev_wind):
                        row["dv_past_24h"] = curr_wind - prev_wind
                    else:
                        row["dv_past_24h"] = 0.0
                        row["dv_past_24h_missing"] = 1.0
                    break

        # Fill remaining NaN persistence with 0.
        if np.isnan(row["dv_past_12h"]):
            row["dv_past_12h"] = 0.0
            row["dv_past_12h_missing"] = 1.0
        if np.isnan(row["dv_past_24h"]):
            row["dv_past_24h"] = 0.0
            row["dv_past_24h_missing"] = 1.0

        # --- Cube field aggregates (fields reduced to tabular: the F arm) ---
        cube_path = interim / f"{event_id}.npy"
        if cube_path.exists():
            cube = np.load(cube_path)
            channels = list(meta.get("channels", []))
            for ch in FIELD_CHANNELS:
                if ch in channels:
                    arr = cube[..., channels.index(ch)]
                    stats = {
                        "mean": float(np.nanmean(arr)),
                        "std": float(np.nanstd(arr)),
                        "min": float(np.nanmin(arr)),
                        "max": float(np.nanmax(arr)),
                    }
                    missing = any(np.isnan(v) for v in stats.values())
                    for s in FIELD_STATS:
                        row[f"cube_{s}_{ch}"] = 0.0 if missing else stats[s]
                    row[f"cube_{ch}_missing"] = 1.0 if missing else 0.0
                else:
                    for s in FIELD_STATS:
                        row[f"cube_{s}_{ch}"] = 0.0
                    row[f"cube_{ch}_missing"] = 1.0
        else:
            for ch in FIELD_CHANNELS:
                for s in FIELD_STATS:
                    row[f"cube_{s}_{ch}"] = 0.0
                row[f"cube_{ch}_missing"] = 1.0

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Feature table built: {len(df)} rows, {len(df.columns)} columns.")

    # --- ANTI-LEAKAGE CHECK ---
    col_names = list(df.columns)
    bad_cols = [c for c in col_names if "dv12" in c.lower() or "dv24" in c.lower()]
    assert not bad_cols, f"LEAKAGE: features contain forbidden names (future targets): {bad_cols}"

    # Cache the result.
    try:
        df.to_parquet(cache_path_parquet, index=False)
    except ImportError:
        df.to_csv(cache_path_csv, index=False)

    audit = {
        "n_rows": len(df),
        "n_features": len(df.columns) - 3,  # Exclude event_id, sid, ri_label
        "cache_path": str(cache_path_parquet),
    }
    return df, audit


def build_run_plan(
    events_df: pd.DataFrame,
    features_df: pd.DataFrame,
    folds_by_seed: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
    seeds: Sequence[int],
    n_folds: int,
) -> Dict[str, Any]:
    """Build run plan (used by both --dry-run and --execute)."""
    y = events_df["ri_label"].to_numpy()
    rows = []
    for seed in seeds:
        for fold_k, (tr, te) in enumerate(folds_by_seed[seed]):
            rows.append({
                "seed": seed, "fold": fold_k,
                "n_train": int(len(tr)), "n_val": int(len(te)),
                "n_pos_val": int(y[te].sum()),
            })

    n_models = 4  # GBM x {S, F, SF} + LogReg on SF
    n_trainings = len(seeds) * n_folds * n_models

    return {
        "n_dev_events": int(len(events_df)),
        "n_positives": int(events_df["ri_label"].sum()) if len(events_df) else 0,
        "n_storms": int(events_df["sid"].nunique()) if len(events_df) else 0,
        "n_features": len(features_df.columns) - 3,  # Exclude event_id, sid, ri_label
        "folds": n_folds,
        "seeds": list(seeds),
        "models": ["GBM(state_only)", "GBM(fields_only)", "GBM(state_plus_fields)",
                   "LogReg(state_plus_fields)"],
        "n_trainings": n_trainings,
        "per_fold": rows,
    }


def format_run_plan_table(plan: Dict[str, Any]) -> str:
    """Format run plan as human-readable table."""
    lines = []
    lines.append("=" * 80)
    lines.append("TABULAR BASELINE (SHIPS-like) -- RUN PLAN (dry-run; pass --execute to train)")
    lines.append("=" * 80)
    lines.append(f"dev events: n={plan['n_dev_events']} positives={plan['n_positives']} "
                 f"storms={plan['n_storms']}")
    lines.append(f"features: {plan['n_features']}")
    lines.append(f"models: {', '.join(plan['models'])}")
    lines.append(f"folds={plan['folds']}  seeds={plan['seeds']}")
    lines.append(f"#trainings = folds x seeds x models = {plan['n_trainings']}")
    lines.append("-" * 80)
    lines.append(f"{'seed':>6} {'fold':>6} {'n_train':>9} {'n_val':>7} {'n_pos_val':>10}")
    for row in plan["per_fold"]:
        lines.append(f"{row['seed']:>6} {row['fold']:>6} {row['n_train']:>9} "
                     f"{row['n_val']:>7} {row['n_pos_val']:>10}")
    lines.append("=" * 80)
    return "\n".join(lines)


def run_execute(cfg: Dict[str, Any], events_df: pd.DataFrame, features_df: pd.DataFrame,
                folds_by_seed: Dict[int, List[Tuple[np.ndarray, np.ndarray]]],
                args: argparse.Namespace) -> Dict[str, Any]:
    """Execute training loop (called only with --execute)."""
    results_root = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = results_root / "tabular_baseline" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ALIGN the feature table to events_df row order: fold indices come from
    # build_folds(events_df, ...), so X rows MUST follow events_df exactly
    # (the cache is validated by event-id SET, not order — never trust order).
    features_df = (features_df.set_index("event_id")
                   .loc[events_df["event_id"]]
                   .reset_index())

    feature_sets = resolve_feature_sets(features_df)
    X_by_set = {name: features_df[cols].to_numpy(dtype=np.float64)
                for name, cols in feature_sets.items()}
    y_indexed = events_df["ri_label"].to_numpy(dtype=int)
    sid_by_event = events_df.set_index("event_id")["sid"].astype(str).to_dict()

    per_seed_results: Dict[str, Any] = {}
    for seed in args.seeds:
        folds = folds_by_seed[seed]
        oof_y: Dict[str, int] = {}
        oof_gbm: Dict[str, Dict[str, float]] = {name: {} for name in FEATURE_SET_NAMES}
        oof_logit: Dict[str, float] = {}

        for fold_k, (tr, te) in enumerate(folds):
            fold_dir = run_dir / f"seed{seed}" / f"fold{fold_k}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            fold_event_ids = features_df["event_id"].iloc[te].tolist()
            for eid, yval in zip(fold_event_ids, y_indexed[te]):
                oof_y[eid] = int(yval)

            # GBM per factorial feature set (S / F / SF).
            for set_name in FEATURE_SET_NAMES:
                Xs = X_by_set[set_name]
                gbm = HistGradientBoostingClassifier(random_state=seed)
                gbm.fit(Xs[tr], y_indexed[tr])
                probs = gbm.predict_proba(Xs[te])[:, 1]
                for eid, p in zip(fold_event_ids, probs):
                    oof_gbm[set_name][eid] = float(p)

            # LogReg reference on the full (SF) set only.
            X_sf = X_by_set["state_plus_fields"]
            # NaN is by-design in cube_* features (missing channels carry a
            # *_missing flag); GBM handles NaN natively, LogReg needs
            # imputation. Median fitted inside the pipeline -> train-fold
            # statistics only, no leakage.
            logit_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("logit", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
            ])
            logit_pipe.fit(X_sf[tr], y_indexed[tr])
            for eid, p in zip(fold_event_ids, logit_pipe.predict_proba(X_sf[te])[:, 1]):
                oof_logit[eid] = float(p)

        # Persist OOF predictions for this seed: one prob column per arm.
        common_ids = [eid for eid in oof_y
                      if all(eid in oof_gbm[name] for name in FEATURE_SET_NAMES)
                      and eid in oof_logit]
        y_pool = np.array([oof_y[e] for e in common_ids], dtype=int)
        groups_pool = np.array([sid_by_event.get(e, "unknown") for e in common_ids], dtype=str)
        pools = {name: np.array([oof_gbm[name][e] for e in common_ids], dtype=np.float64)
                 for name in FEATURE_SET_NAMES}
        logit_pool = np.array([oof_logit[e] for e in common_ids], dtype=np.float64)

        oof_csv = run_dir / f"seed{seed}" / "oof_predictions.csv"
        pd.DataFrame({
            "event_id": common_ids,
            "sid": groups_pool,
            "y": y_pool,
            "prob_gbm_state_only": pools["state_only"],
            "prob_gbm_fields_only": pools["fields_only"],
            "prob_gbm_state_plus_fields": pools["state_plus_fields"],
            "prob_logit_state_plus_fields": logit_pool,
        }).to_csv(oof_csv, index=False)

        has_both_classes = len(np.unique(y_pool)) > 1
        arms_out: Dict[str, Any] = {}
        for set_name in FEATURE_SET_NAMES:
            pool = pools[set_name]
            arms_out[f"gbm_{set_name}"] = {
                "n_features": len(feature_sets[set_name]),
                "pr_auc": float(average_precision_score(y_pool, pool)) if has_both_classes else None,
                "roc_auc": float(roc_auc_score(y_pool, pool)) if has_both_classes else None,
                "pr_auc_cluster_ci": _single_model_cluster_ci(
                    y_pool, pool, groups_pool, seed=seed, n_boot=args.n_boot),
            }
        arms_out["logit_state_plus_fields"] = {
            "pr_auc": float(average_precision_score(y_pool, logit_pool)) if has_both_classes else None,
            "roc_auc": float(roc_auc_score(y_pool, logit_pool)) if has_both_classes else None,
        }

        per_seed_results[str(seed)] = {
            "n_pooled_events": int(len(common_ids)),
            "arms": arms_out,
            "oof_predictions_csv": rel_to_root(oof_csv),
        }

    summary = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": rel_to_root(Path(args.config).resolve()),
        "n_events": int(len(events_df)),
        "n_positives": int(events_df["ri_label"].sum()),
        "n_storms": int(events_df["sid"].nunique()),
        "feature_sets": {name: cols for name, cols in feature_sets.items()},
        "folds": args.folds,
        "seeds": args.seeds,
        "n_boot": args.n_boot,
        "models": ["HistGradientBoostingClassifier (per feature set)",
                   "LogisticRegression (state_plus_fields only)"],
        "per_seed": per_seed_results,
        "run_dir": rel_to_root(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def compare_cnn(cnn_run_dirs: Sequence[Path], tabular_run_dir: Path,
                args: argparse.Namespace) -> Dict[str, Any]:
    """Compare tabular baseline GBM against CNN arm A across seeds.

    Computes pre-registered verdict: CI of PR-AUC(CNN_A) - PR-AUC(GBM).
    """
    # Load OOF CSVs from both sources.
    tabular_oofs: Dict[int, Path] = {}
    cnn_oofs: Dict[int, Path] = {}

    for cnn_dir in cnn_run_dirs:
        cnn_path = Path(cnn_dir)
        # Search for seed{s}/oof_predictions.csv under cnn_path.
        for oof_path in cnn_path.rglob("oof_predictions.csv"):
            match_str = oof_path.parent.name  # e.g., "seed42"
            if match_str.startswith("seed"):
                try:
                    seed = int(match_str[4:])
                    cnn_oofs[seed] = oof_path
                except ValueError:
                    pass

    tabular_path = Path(tabular_run_dir)
    for oof_path in tabular_path.rglob("oof_predictions.csv"):
        match_str = oof_path.parent.name
        if match_str.startswith("seed"):
            try:
                seed = int(match_str[4:])
                tabular_oofs[seed] = oof_path
            except ValueError:
                pass

    # Find common seeds.
    common_seeds = sorted(set(cnn_oofs.keys()) & set(tabular_oofs.keys()))
    if not common_seeds:
        print("No common seeds found in CNN and tabular run directories.")
        sys.exit(1)

    if len(common_seeds) < 3 and not args.allow_partial:
        print(f"Fewer than 3 common seeds ({len(common_seeds)}): "
              "pass --allow-partial for INTERMEDIATE results (no verdict).")
        sys.exit(1)

    print(f"Common seeds: {common_seeds}")

    # Aggregate across seeds: per-seed deltas for both co-primary endpoints.
    # The absolute per-arm values are recorded alongside — a report that
    # shows only deltas hides the arms themselves.
    deltas_pr_auc = []      # V1: CNN - GBM_SF
    deltas_pr_auc_v2 = []   # V2: CNN - GBM_F
    per_seed_absolute: Dict[str, Dict[str, Optional[float]]] = {}
    frames = []
    for seed in common_seeds:
        cnn_df = pd.read_csv(cnn_oofs[seed]).sort_values("event_id").reset_index(drop=True)
        tab_df = pd.read_csv(tabular_oofs[seed]).sort_values("event_id").reset_index(drop=True)

        if not (cnn_df["event_id"].to_numpy() == tab_df["event_id"].to_numpy()).all():
            raise ValueError(f"Seed {seed}: event sets do not align between CNN and tabular.")
        if not np.array_equal(cnn_df["y"].to_numpy(), tab_df["y"].to_numpy()):
            raise ValueError(f"Seed {seed}: labels do not align.")

        y = cnn_df["y"].to_numpy()
        prob_cnn = cnn_df["prob_A"].to_numpy()  # CNN arm A (full-resolution fields)
        # V1 arm: SF (state + field aggregates); legacy fallback for old runs.
        # V2 arm: F (field aggregates only) — co-primary per the amended
        # pre-registration (architecture justification at fixed diet).
        gbm_col = ("prob_gbm_state_plus_fields"
                   if "prob_gbm_state_plus_fields" in tab_df.columns else "prob_gbm")
        prob_gbm = tab_df[gbm_col].to_numpy()
        f_col = "prob_gbm_fields_only" if "prob_gbm_fields_only" in tab_df.columns else None
        prob_f = tab_df[f_col].to_numpy() if f_col else None

        if len(np.unique(y)) > 1:
            cnn_pr = float(average_precision_score(y, prob_cnn))
            gbm_pr = float(average_precision_score(y, prob_gbm))
            abs_row = {
                "cnn_pr_auc": cnn_pr,
                "cnn_roc_auc": float(roc_auc_score(y, prob_cnn)),
                f"{gbm_col.replace('prob_', '')}_pr_auc": gbm_pr,
            }
            # Decomposition arms (S / F / logit), when present in the tabular csv.
            for col in tab_df.columns:
                if col.startswith(("prob_gbm_", "prob_logit_")) and col != gbm_col:
                    abs_row[f"{col.replace('prob_', '')}_pr_auc"] = float(
                        average_precision_score(y, tab_df[col].to_numpy()))
            per_seed_absolute[str(seed)] = abs_row
            deltas_pr_auc.append(cnn_pr - gbm_pr)
            if prob_f is not None:
                deltas_pr_auc_v2.append(cnn_pr - float(average_precision_score(y, prob_f)))
        else:
            per_seed_absolute[str(seed)] = {"cnn_pr_auc": None}
            print(f"  Seed {seed}: no variance in labels, skipping per-seed delta.")

        # Keep frame for aggregated bootstrap.
        frames.append({
            "seed": seed,
            "y": y,
            "prob_cnn": prob_cnn,
            "prob_gbm": prob_gbm,
            "prob_f": prob_f,
            "sid": cnn_df["sid"].to_numpy(),
        })

    if not deltas_pr_auc:
        print("No valid seed deltas computed (all seeds had no label variance).")
        sys.exit(1)

    point_est = float(np.mean(deltas_pr_auc))

    # Aggregated cluster bootstrap: ONE shared SID resampling per replicate,
    # applied to every seed's paired predictions. Resampling is WITH
    # replacement, so a storm drawn k times must appear k times in the
    # replicate (np.isin would collapse duplicates and narrow the CI) —
    # hence the concatenated per-group index construction.
    sid_ref = frames[0]["sid"]
    y_ref = frames[0]["y"]
    for f in frames[1:]:
        if not np.array_equal(f["sid"], sid_ref) or not np.array_equal(f["y"], y_ref):
            raise ValueError("sid/label arrays differ across seeds -- all seeds must "
                             "cover the same dev events (cannot share the resampling).")
    unique_groups = np.unique(sid_ref)
    group_to_idx = {g: np.where(sid_ref == g)[0] for g in unique_groups}

    have_f = all(f["prob_f"] is not None for f in frames)

    rng = np.random.default_rng(args.bootstrap_seed)
    draws = []          # V1: CNN - GBM_SF
    draws_v2 = []       # V2: CNN - GBM_F (co-primary; same shared resampling)
    n_skipped = 0
    for _ in range(args.n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_idx[g] for g in sampled])
        yb = y_ref[idx]
        if len(np.unique(yb)) < 2:
            n_skipped += 1
            continue
        cnn_prs = [average_precision_score(yb, f["prob_cnn"][idx]) for f in frames]
        sf_prs = [average_precision_score(yb, f["prob_gbm"][idx]) for f in frames]
        draws.append(float(np.mean([c - s for c, s in zip(cnn_prs, sf_prs)])))
        if have_f:
            f_prs = [average_precision_score(yb, f["prob_f"][idx]) for f in frames]
            draws_v2.append(float(np.mean([c - fp for c, fp in zip(cnn_prs, f_prs)])))

    alpha = 0.025  # 95% CI
    arr = np.asarray(draws, dtype=np.float64)
    ci_low = float(np.quantile(arr, alpha)) if arr.size else None
    ci_high = float(np.quantile(arr, 1.0 - alpha)) if arr.size else None
    arr_v2 = np.asarray(draws_v2, dtype=np.float64)
    ci2_low = float(np.quantile(arr_v2, alpha)) if arr_v2.size else None
    ci2_high = float(np.quantile(arr_v2, 1.0 - alpha)) if arr_v2.size else None

    # V1 verdict — validity: CNN vs strongest tabular baseline (SF).
    if ci_low is None or ci_high is None:
        verdict = "INCONCLUSIVE: insufficient valid bootstrap draws."
    elif ci_low > 0:
        verdict = (f"CNN adds skill beyond the full tabular baseline "
                   f"(CI [{ci_low:.4f}, {ci_high:.4f}] > 0).")
    elif ci_high < 0:
        verdict = (f"tabular baseline BEATS CNN (CI [{ci_low:.4f}, {ci_high:.4f}] < 0) — "
                   "the CNN is not currently justified over a classical baseline.")
    else:
        verdict = (f"NULL — no detectable CNN advantage over the tabular baseline "
                   f"(CI [{ci_low:.4f}, {ci_high:.4f}] includes 0).")

    # V2 verdict — architecture justification: CNN vs field-aggregates arm (F),
    # co-primary per the amended pre-registration. Scope guard: a null/negative
    # V2 unjustifies THIS architecture; it does not establish that spatial
    # structure carries no information.
    if not have_f:
        verdict_v2 = ("NOT COMPUTED: tabular run lacks the fields_only arm "
                      "(legacy single-arm run) — rerun with the factorial harness.")
    elif ci2_low is None or ci2_high is None:
        verdict_v2 = "INCONCLUSIVE: insufficient valid bootstrap draws."
    elif ci2_low > 0:
        verdict_v2 = (f"CNN extracts spatial-structure signal beyond field aggregates "
                      f"(CI [{ci2_low:.4f}, {ci2_high:.4f}] > 0) — architecture justified "
                      "on the field diet.")
    elif ci2_high < 0:
        verdict_v2 = (f"field aggregates BEAT the CNN (CI [{ci2_low:.4f}, {ci2_high:.4f}] < 0) "
                      "— pre-registered consequence: architecture retired/redesigned.")
    else:
        verdict_v2 = (f"NULL — full grids give this CNN nothing detectable beyond "
                      f"aggregates (CI [{ci2_low:.4f}, {ci2_high:.4f}] includes 0) — pre-registered "
                      "consequence: architecture not justified in current form.")

    # Pre-registered discipline: fewer than 3 seeds has NO verdict value.
    if len(common_seeds) < 3:
        verdict = "INTERMEDIATE (fewer than 3 seeds; NO verdict value): " + verdict
        verdict_v2 = "INTERMEDIATE (fewer than 3 seeds; NO verdict value): " + verdict_v2

    # Cross-seed means of every absolute metric present.
    metric_keys = sorted({k for v in per_seed_absolute.values() for k in v})
    absolute_means = {}
    for k in metric_keys:
        vals = [v[k] for v in per_seed_absolute.values() if v.get(k) is not None]
        absolute_means[k] = float(np.mean(vals)) if vals else None

    result = {
        "method": "cross_seed_mean_delta_pr_auc__cluster_bootstrap_by_sid",
        "primary_comparison": {
            "cnn": "CNN arm A_current (full-resolution spatial fields; intensity-blind)",
            "tabular": gbm_col.replace("prob_", "") + " (GBM; primary pre-registered baseline)",
            "note": "decomposition arms (state_only / fields_only) are descriptive: "
                    "SF-S isolates field information, CNN-F isolates spatial structure "
                    "beyond aggregates.",
        },
        "n_seeds": len(common_seeds),
        "common_seeds": common_seeds,
        "per_seed_absolute": per_seed_absolute,
        "absolute_pr_auc_means": absolute_means,
        "ci_level": 0.95,
        "n_boot_requested": args.n_boot,
        "n_boot_skipped": n_skipped,
        "V1_cnn_vs_sf": {
            "per_seed_delta_pr_auc": deltas_pr_auc,
            "delta_pr_auc_point": point_est,
            "delta_pr_auc_ci_low": ci_low,
            "delta_pr_auc_ci_high": ci_high,
            "n_boot_used": int(arr.size),
            "verdict": verdict,
        },
        "V2_cnn_vs_f": {
            "per_seed_delta_pr_auc": deltas_pr_auc_v2,
            "delta_pr_auc_point": (float(np.mean(deltas_pr_auc_v2))
                                   if deltas_pr_auc_v2 else None),
            "delta_pr_auc_ci_low": ci2_low,
            "delta_pr_auc_ci_high": ci2_high,
            "n_boot_used": int(arr_v2.size),
            "verdict": verdict_v2,
        },
        # Legacy top-level fields kept for V1 (older readers).
        "per_seed_delta_pr_auc": deltas_pr_auc,
        "delta_pr_auc_point": point_est,
        "delta_pr_auc_ci_low": ci_low,
        "delta_pr_auc_ci_high": ci_high,
        "verdict": verdict,
        "verdict_v2": verdict_v2,
        "cnn_run_dirs": [str(d) for d in cnn_run_dirs],
        "tabular_run_dir": str(tabular_run_dir),
        "has_verdict": "INCONCLUSIVE" not in verdict,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml.")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="Comma-separated seeds.")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True,
                        help="Print plan and exit (default: on).")
    parser.add_argument("--execute", action="store_true", default=False,
                        help="Actually train. Overrides --dry-run.")
    parser.add_argument("--require-gate", action=argparse.BooleanOptionalAction, default=True,
                        help="Refuse to run unless gate passes (default: on).")
    parser.add_argument("--rebuild-features", action="store_true", default=False,
                        help="Force rebuild of feature cache.")
    parser.add_argument("--build-features-only", action="store_true", default=False,
                        help="Build/refresh feature cache and exit.")
    parser.add_argument("--compare-cnn", nargs="+", default=None, metavar="PATH",
                        help="Path(s) to CNN feature_ablation run dir(s) (searched recursively for oof_predictions.csv).")
    parser.add_argument("--tabular", type=str, default=None, metavar="PATH",
                        help="Path to a tabular run directory (for --compare-cnn mode).")
    parser.add_argument("--allow-partial", action="store_true", default=False,
                        help="Allow comparison with fewer than 3 seeds (prints INTERMEDIATE, no verdict).")
    args = parser.parse_args()
    args.seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    cfg = load_config(str(config_path))

    # --- COMPARE MODE ---
    if args.compare_cnn:
        if not args.tabular:
            print("--compare-cnn requires --tabular PATH.")
            sys.exit(1)
        result = compare_cnn([Path(p) for p in args.compare_cnn], Path(args.tabular), args)
        out_dir = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve() / "tabular_baseline"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"compare_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"\nVERDICT V1 (validity, CNN vs SF): {result['verdict']}")
        print(f"VERDICT V2 (architecture, CNN vs F): {result['verdict_v2']}")
        print(f"report: {out_path}")
        return

    # --- BUILD FEATURES ONLY ---
    if args.build_features_only:
        enforce_gate(cfg, args.require_gate)
        events_df, _ = load_pl_gated_dev_events(cfg)
        _, feat_audit = build_feature_table(cfg, events_df, rebuild_features=True)
        print("Feature table built and cached:")
        print(json.dumps(feat_audit, indent=2))
        return

    # --- NORMAL MODE (plan / execute) ---
    enforce_gate(cfg, args.require_gate)

    events_df, dev_audit = load_pl_gated_dev_events(cfg)
    print("DEV AUDIT (PL-gated):", json.dumps(dev_audit, indent=2))

    if len(events_df) == 0 or events_df["ri_label"].sum() < args.folds:
        print("Not enough events/positives for requested fold count — aborting.")
        sys.exit(1)

    # In dry-run mode, only load features from cache; don't extract.
    # Feature extraction only runs with --execute or --build-features-only.
    if args.execute:
        features_df, feat_audit = build_feature_table(cfg, events_df, rebuild_features=args.rebuild_features)
        print("FEATURE TABLE AUDIT:", json.dumps(feat_audit, indent=2))
    else:
        # Dry-run: check cache status without extraction.
        cache_dir = Path(cfg_get(cfg, "paths.results_dir", "./outputs/results")).resolve() / "tabular_baseline"
        cache_path_parquet = cache_dir / "feature_table.parquet"
        cache_path_csv = cache_dir / "feature_table.csv"

        features_df = None
        if cache_path_parquet.exists():
            try:
                features_df = pd.read_parquet(cache_path_parquet)
                print("FEATURE TABLE: cache (parquet) exists, n_rows =", len(features_df))
            except Exception:
                print("FEATURE TABLE: cache (parquet) exists but failed to load.")
        elif cache_path_csv.exists():
            try:
                features_df = pd.read_csv(cache_path_csv)
                print("FEATURE TABLE: cache (csv) exists, n_rows =", len(features_df))
            except Exception:
                print("FEATURE TABLE: cache (csv) exists but failed to load.")
        else:
            print("FEATURE TABLE: no cache found. Pass --build-features-only to build features, "
                  "or --execute to build and train in one run.")
            # For dry-run plan, estimate feature count from a minimal schema.
            features_df = pd.DataFrame({
                "event_id": events_df["event_id"],
                "wind_kt": np.nan, "pressure_mb": np.nan, "center_lat": np.nan, "abs_lat": np.nan,
                "doy_sin": np.nan, "doy_cos": np.nan,
                "dv_past_12h": np.nan, "dv_past_12h_missing": np.nan,
                "dv_past_24h": np.nan, "dv_past_24h_missing": np.nan,
            })
            # Add placeholder columns for basin and cube field aggregates.
            for basin in ["WP", "EP", "ATL", "IO"]:
                features_df[f"basin_{basin}"] = np.nan
            for ch in FIELD_CHANNELS:
                for s in FIELD_STATS:
                    features_df[f"cube_{s}_{ch}"] = np.nan
                features_df[f"cube_{ch}_missing"] = np.nan

    folds_by_seed = {s: build_folds(events_df, args.folds, s) for s in args.seeds}
    plan = build_run_plan(events_df, features_df, folds_by_seed, args.seeds, args.folds)

    print(format_run_plan_table(plan))

    if not args.execute:
        print("\nDRY RUN: no training was performed. Pass --execute to actually train.")
        return

    print("\nEXECUTING: training tabular models...")
    summary = run_execute(cfg, events_df, features_df, folds_by_seed, args)
    print(f"\nrun_dir: {summary['run_dir']}")


if __name__ == "__main__":
    main()
