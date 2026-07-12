#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CycloneNet — End-to-End Pipeline Runner

This script provides a command-line interface to all CycloneNet pipeline stages.
It automatically locates the project root (where config.yaml resides) so that
it can be invoked from any working directory. All scientific safeguards
(leakage-free splits, train-only normalization, physics‑guided losses) are
preserved by delegating to the dedicated modules.

Usage examples:
    python run.py prepare
    python run.py download-era5
    python run.py preprocess
    python run.py normalize
    python run.py train
    python run.py evaluate --split test
    python run.py dataqa --split test
    python run.py download-tchp
    python run.py preprocess-tchp
    python run.py baseline
    python run.py sweep --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "config.yaml").exists():
            return parent
    raise RuntimeError("config.yaml not found. Please run inside the CycloneNet project.")


PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, cfg_get, ensure_dirs
from src.utils.snapshot import save_run_snapshot
from src.utils.splits import SplitConfig, make_splits
from src.data.normalization import compute_norm_stats

logger = logging.getLogger("cyclonenet.run")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _normalized_dir(cfg: dict) -> Path:
    return Path(cfg_get(cfg, "paths.normalized_dir", "./data/normalized")).resolve()


def ensure_normalized_paths(cfg: dict) -> dict:
    normalized_dir = _normalized_dir(cfg)
    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("normalized_dir", str(normalized_dir))
    cfg["paths"].setdefault("splits_csv", str(normalized_dir / "splits.csv"))
    cfg["paths"].setdefault("normalization_stats", str(normalized_dir / "normalization_stats.json"))
    return cfg


def _generate_valid_manifest(cfg: dict) -> Path:
    interim_dir = Path(cfg_get(cfg, "paths.interim_data", "./data/interim")).resolve()
    out_dir = _normalized_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "valid_metadata.csv"

    rows = []
    for json_path in sorted(interim_dir.glob("era5_*.json")):
        event_id = json_path.stem
        cube_path = interim_dir / f"{event_id}.npy"
        lats_path = interim_dir / f"{event_id}_lats.npy"
        lons_path = interim_dir / f"{event_id}_lons.npy"
        if not cube_path.exists() or not lats_path.exists() or not lons_path.exists():
            continue
        with json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if not bool(meta.get("temporal_integrity_ok", False)):
            continue
        sid = str(meta.get("sid", "")).strip()
        if not sid:
            continue
        rows.append(
            {
                "event_id": event_id,
                "sid": sid,
                "storm_name": meta.get("storm_name", ""),
                "ri_label": meta.get("ri_label", 0),
                "dv12_kt": meta.get("dv12_kt", None),
                "dv24_kt": meta.get("dv24_kt", None),
            }
        )

    import pandas as pd

    df = pd.DataFrame(rows).drop_duplicates(subset=["event_id"])
    if df.empty:
        raise RuntimeError("No valid event artifacts were found in interim_data.")
    df.to_csv(manifest_path, index=False)
    logger.info("Valid manifest saved to %s | n_events=%d", manifest_path, len(df))
    return manifest_path


def _ensure_splits(cfg: dict, manifest_path: Path) -> None:
    split_cfg = SplitConfig.from_config(cfg)
    make_splits(manifest_path, split_cfg)
    logger.info("Splits generated at %s", split_cfg.path)


def cmd_prepare(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "prepare")
    from src.processors.ibtracs import run_prepare
    run_prepare(cfg, force=bool(args.force))
    logger.info("prepare: done")


def cmd_download_era5(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "download-era5")
    from src.downloaders.era5 import ERA5Downloader
    ERA5Downloader(cfg).download_required_batch()
    logger.info("download-era5: done")


def cmd_preprocess(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "preprocess")
    from src.processors.preprocess_scientific import run_preprocess
    run_preprocess(cfg)
    logger.info("preprocess: done")


def cmd_normalize(args):
    """
    Normalization pipeline:
      1. Audit preprocessed events for scientific training eligibility.
      2. Build the valid manifest from scientifically eligible events only.
      3. Create storm-based train/val/test splits from the valid manifest.
      4. Compute train-only normalization statistics using only valid train events.
    """
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "normalize")

    # Step 1-4 are handled inside the strict normalization module.
    # The module writes:
    # - valid_events.csv
    # - rejected_events.csv
    # - normalization_report.json
    # - normalization_stats.json
    from src.data.normalization import build_training_manifests, compute_norm_stats

    audit_info = build_training_manifests(cfg)
    manifest_path = Path(audit_info["valid_manifest_path"]).resolve()

    # Create storm-based splits from scientifically valid events only.
    _ensure_splits(cfg, manifest_path)

    # Compute train-only normalization statistics using only valid train events.
    compute_norm_stats(cfg)

    logger.info("normalize: done")


def cmd_train(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "train")
    stats_path = Path(cfg_get(cfg, "paths.normalization_stats", "./data/normalized/normalization_stats.json")).resolve()
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}. Run 'python run.py normalize' first.")
    from src.training.trainer import train
    train(cfg)
    logger.info("train: done")


def cmd_evaluate(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "evaluate")
    from src.evaluation.evaluate import run_evaluate
    run_evaluate(
        cfg,
        split=args.split,
        calibrate=args.calibrate,
        experimental_external_spatial=bool(getattr(args, "spatial", False)),
    )
    logger.info("evaluate: done")


def cmd_download_tchp(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "download-tchp")
    from src.downloaders.tchp import download_tchp
    download_tchp(cfg, force=bool(args.force))
    logger.info("download-tchp: done")


def cmd_download_ssh(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "download-ssh")
    from src.downloaders.ssh import run_download_ssh
    run_download_ssh(cfg, force=bool(args.force))
    logger.info("download-ssh: done")


def cmd_preprocess_adt(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "preprocess-adt")
    from src.processors.preprocess_adt import run_preprocess_adt
    run_preprocess_adt(cfg)
    logger.info("preprocess-adt: done")


def cmd_preprocess_tchp(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "preprocess-tchp")
    from src.processors.preprocess_tchp import run_preprocess_tchp
    run_preprocess_tchp(cfg)
    logger.info("preprocess-tchp: done")


def cmd_baseline(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "baseline")
    from src.baselines.tabular_lr import TabularBaseline

    baseline = TabularBaseline(cfg)
    baseline.train()
    metrics, pred_df = baseline.evaluate(split=args.split)
    out_dir = results_dir / "baseline"
    baseline.save_results(out_dir, metrics, pred_df)
    logger.info("baseline: done | metrics=%s", json.dumps(metrics))


def cmd_validate_sla(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "validate-sla")
    from src.evaluation.sla_validation import run_and_save

    out_path = results_dir / "sla" / f"sla_validation_{args.year}.json"
    report = run_and_save(cfg, year=int(args.year), window_deg=float(args.window_deg),
                          out_path=out_path, max_events=args.max_events)
    logger.info("validate-sla: done | status=%s", report.get("status"))


def cmd_causal(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "causal")
    from src.evaluation.causal_ablation import run_and_save

    out_path = results_dir / "causal" / f"causal_ablation_{args.split}.json"
    report = run_and_save(
        cfg,
        split=args.split,
        k=float(args.k),
        factor=float(args.factor),
        channels=list(args.channels),
        out_path=out_path,
        max_events=args.max_events,
    )
    logger.info("causal: done | evidence=%s", json.dumps(report.get("causal_evidence", {})))


def cmd_download_era5pl(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    if not bool(cfg_get(cfg, "download.pressure_levels.enabled", False)):
        raise SystemExit("download.pressure_levels.enabled is false in config.yaml — "
                         "enable it explicitly before downloading pressure-level data.")
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "download-era5pl")
    from src.downloaders.era5_pressure import ERA5PressureDownloader
    ERA5PressureDownloader(cfg).download_required_batch()
    logger.info("download-era5pl: done")


def cmd_backfill_pl_window(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    if not bool(cfg_get(cfg, "download.pressure_levels.enabled", False)):
        raise SystemExit("download.pressure_levels.enabled is false in config.yaml — "
                         "enable it explicitly before running the PL backfill.")
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "backfill-pl-window")
    from src.pipeline.pl_backfill import backfill_window
    y0, y1 = int(args.years[0]), int(args.years[1])
    manifest = backfill_window(cfg, y0, y1, dry_run=bool(args.dry_run))
    logger.info("backfill-pl-window: done | window=%d-%d | status=%s", y0, y1, manifest.get("status"))


def cmd_backfill_pl_windows(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    if not bool(cfg_get(cfg, "download.pressure_levels.enabled", False)):
        raise SystemExit("download.pressure_levels.enabled is false in config.yaml — "
                         "enable it explicitly before running the PL backfill.")
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "backfill-pl-windows")
    from src.pipeline.pl_backfill import make_windows, run_all_backfill_windows
    windows = make_windows(1980, 2019, window_years=2)
    manifests = run_all_backfill_windows(cfg, windows, dry_run=bool(args.dry_run))
    done = sum(1 for m in manifests if m.get("status") == "completed")
    logger.info("backfill-pl-windows: done | completed=%d/%d", done, len(manifests))


def cmd_process_window(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "process-window")
    from src.pipeline.windowed import process_window
    manifest = process_window(cfg, int(args.start_year), int(args.end_year),
                              delete_raw=not args.keep_raw)
    logger.info("process-window: done | status=%s", manifest.get("status"))


def cmd_process_windows(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "process-windows")
    from src.pipeline.windowed import run_all_windows
    manifests = run_all_windows(cfg, int(args.start_year), int(args.end_year),
                                window_years=int(args.window_years),
                                delete_raw=not args.keep_raw)
    done = sum(1 for m in manifests if m.get("status") == "completed")
    logger.info("process-windows: done | completed=%d/%d", done, len(manifests))


def cmd_dataqa(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "dataqa")
    from src.validation.dataqa import run_dataqa
    report = run_dataqa(cfg, split=args.split)
    out_dir = results_dir / "dataqa"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / f"dataqa_{args.split}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("DataQA report saved to %s", out_json)


def cmd_sweep(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"]) / "sweep"
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [0, 1, 2]
    save_run_snapshot(cfg, results_dir, "sweep")
    from src.utils.sweep import run_sweep
    run_sweep(cfg, seeds, results_dir)
    logger.info("sweep: done")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run.py", description="CycloneNet pipeline runner.")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare", help="Download and prepare IBTrACS event list")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("download-era5", help="Download monthly ERA5 NetCDF files")
    sp.set_defaults(func=cmd_download_era5)

    sp = sub.add_parser("preprocess", help="Extract event cubes and metadata")
    sp.set_defaults(func=cmd_preprocess)

    sp = sub.add_parser("normalize", help="Generate valid manifest, splits, and normalization stats")
    sp.set_defaults(func=cmd_normalize)

    sp = sub.add_parser("train", help="Train the model")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("evaluate", help="Evaluate trained model")
    sp.add_argument("--split", default="test", choices=["val", "test"])
    sp.add_argument("--calibrate", action="store_true")
    sp.add_argument(
        "--spatial",
        action="store_true",
        help="Run TCHP spatial validation (requires 'preprocess-tchp' to have enriched metadata)",
    )
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("download-tchp", help="Download TCHP validation data (NOAA/AOML/Copernicus)")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_download_tchp)

    sp = sub.add_parser("download-ssh", help="Download Sea Level Anomaly (SLA) ocean signal (minimal season)")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_download_ssh)

    sp = sub.add_parser("preprocess-adt", help="Sample ADT ocean channel into events + train-only ADT stats")
    sp.set_defaults(func=cmd_preprocess_adt)

    sp = sub.add_parser("preprocess-tchp", help="Enrich event metadata with audited TCHP peak locations")
    sp.set_defaults(func=cmd_preprocess_tchp)

    sp = sub.add_parser("baseline", help="Train/evaluate the tabular logistic-regression baseline")
    sp.add_argument("--split", default="test", choices=["val", "test"])
    sp.set_defaults(func=cmd_baseline)

    sp = sub.add_parser("validate-sla", help="Validate SLA co-location with TCHP and RI (observational)")
    sp.add_argument("--year", type=int, default=2023)
    sp.add_argument("--window-deg", type=float, default=5.0, dest="window_deg")
    sp.add_argument("--max-events", type=int, default=None, dest="max_events")
    sp.set_defaults(func=cmd_validate_sla)

    sp = sub.add_parser("causal", help="Counterfactual FuelMap ablation causal test (vs low-fuel control)")
    sp.add_argument("--split", default="test", choices=["val", "test"])
    sp.add_argument("--k", type=float, default=0.05, help="Top/bottom-k fraction for masks")
    sp.add_argument("--factor", type=float, default=0.5, help="Ablation strength in [0,1]")
    sp.add_argument("--channels", nargs="+", default=["sst_anom_K", "wind_mps"])
    sp.add_argument("--max-events", type=int, default=None, dest="max_events")
    sp.set_defaults(func=cmd_causal)

    sp = sub.add_parser("download-era5pl",
                        help="Download ERA5 pressure-level fields (shear + mid-level RH); "
                             "gated by download.pressure_levels.enabled")
    sp.set_defaults(func=cmd_download_era5pl)

    sp = sub.add_parser("backfill-pl-window",
                        help="Backfill 'shear_850_200_mps'/'rh_mid' channels for events in ONE year "
                             "window whose cube predates pressure-level extraction (1980-2019 archive); "
                             "gated by download.pressure_levels.enabled")
    sp.add_argument("--years", type=int, nargs=2, required=True, metavar=("START", "END"))
    sp.add_argument("--dry-run", action="store_true", help="Classify events only; no download/write/delete")
    sp.set_defaults(func=cmd_backfill_pl_window)

    sp = sub.add_parser("backfill-pl-windows",
                        help="Run backfill-pl-window for every 1980-2019 window (2-year steps), "
                             "resumable via outputs/provenance/pl_window_*.json manifests")
    sp.add_argument("--dry-run", action="store_true", help="Classify events only; no download/write/delete")
    sp.set_defaults(func=cmd_backfill_pl_windows)

    sp = sub.add_parser("process-window",
                        help="Windowed pipeline for ONE window: download ERA5, extract events, "
                             "verify artifacts, then discard raw data (only if verification passes)")
    sp.add_argument("--start-year", type=int, required=True)
    sp.add_argument("--end-year", type=int, required=True)
    sp.add_argument("--keep-raw", action="store_true", help="Skip the raw-discard step")
    sp.set_defaults(func=cmd_process_window)

    sp = sub.add_parser("process-windows",
                        help="Run all windows from start to end (resumable via provenance manifests)")
    sp.add_argument("--start-year", type=int, required=True)
    sp.add_argument("--end-year", type=int, required=True)
    sp.add_argument("--window-years", type=int, default=2)
    sp.add_argument("--keep-raw", action="store_true")
    sp.set_defaults(func=cmd_process_windows)

    sp = sub.add_parser("dataqa", help="Run artifact quality assurance")
    sp.add_argument("--split", default="test", choices=["train", "val", "test"])
    sp.set_defaults(func=cmd_dataqa)

    sp = sub.add_parser("sweep", help="Run multi-seed sweep")
    sp.add_argument("--seeds", default="0,1,2")
    sp.set_defaults(func=cmd_sweep)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)
    args.func(args)


if __name__ == "__main__":
    main()

