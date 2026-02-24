#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CycloneNet — end-to-end runner (reproducibility-first, cross-platform).

# CycloneNet V2
# Copyright (c) 2026 Estefano Senhor Ferreira
# Licensed under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
# https://creativecommons.org/licenses/by-nc/4.0/

Commands:
  python run.py prepare [--force]
  python run.py download-era5
  python run.py preprocess
  python run.py normalize
  python run.py train
  python run.py evaluate [--split test] [--calibrate]
  python run.py dataqa [--split test]
  python run.py download-tchp [--force]
  python run.py preprocess-tchp
  python run.py baseline [--split test]
  python run.py sweep --seeds 0 1 2 ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.config import load_config, cfg_get, ensure_dirs
from src.utils.snapshot import save_run_snapshot

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
    cfg.setdefault("paths", {})
    nd = _normalized_dir(cfg)
    cfg["paths"].setdefault("normalized_dir", str(nd.as_posix()))
    cfg["paths"].setdefault("valid_manifest", str((nd / "valid_events.csv").as_posix()))
    cfg["paths"].setdefault("splits_csv", str((nd / "splits.csv").as_posix()))
    cfg["paths"].setdefault("normalization_stats", str((nd / "normalization_stats.json").as_posix()))
    cfg["paths"].setdefault("sanity_report", str((nd / "sanity_report.json").as_posix()))
    cfg["paths"].setdefault("runtime_snapshot", str((nd / "runtime_config_snapshot.json").as_posix()))
    return cfg


def save_runtime_snapshot(cfg: dict) -> None:
    nd = _normalized_dir(cfg)
    nd.mkdir(parents=True, exist_ok=True)
    snap = Path(cfg_get(cfg, "paths.runtime_snapshot", nd / "runtime_config_snapshot.json")).resolve()
    snap.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------

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
    dl = ERA5Downloader(cfg)
    dl.download_required_batch()
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
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "normalize")
    from src.utils.normalization_helpers import ensure_valid_manifest, ensure_splits, sanity_gate_minimal
    manifest = ensure_valid_manifest(cfg)
    ensure_splits(cfg, manifest)
    sanity_gate_minimal(cfg, manifest)
    from src.data.normalization import compute_norm_stats
    compute_norm_stats(cfg)
    logger.info("normalize: done")


def cmd_train(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "train")
    stats_path = Path(cfg_get(cfg, "paths.normalization_stats", "./data/normalized/normalization_stats.json")).resolve()
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}. Run: python run.py normalize")
    from src.training.trainer import train
    train(cfg)
    logger.info("train: done")


def cmd_evaluate(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "evaluate")
    from src.evaluation.evaluate import run_evaluate
    run_evaluate(cfg, split=args.split, calibrate=args.calibrate)
    logger.info("evaluate: done")


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
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"DataQA report saved to {out_json}")


def cmd_download_tchp(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"])
    save_run_snapshot(cfg, results_dir, "download-tchp")
    from src.downloaders.tchp import download_tchp
    download_tchp(cfg, force=bool(args.force))
    logger.info("download-tchp: done")


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
    results_dir = Path(cfg["paths"]["results_dir"]) / "baseline"
    save_run_snapshot(cfg, results_dir, "baseline")
    from src.baselines.tabular_lr import TabularBaseline
    baseline = TabularBaseline(cfg)
    baseline.train()
    metrics, pred_df = baseline.evaluate(split=args.split)
    baseline.save_results(results_dir, metrics, pred_df)
    logger.info("baseline: done")


def cmd_sweep(args):
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    results_dir = Path(cfg["paths"]["results_dir"]) / "sweep"
    results_dir.mkdir(exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [0, 1, 2]
    all_metrics = []
    for seed in seeds:
        logger.info(f"Running sweep with seed {seed}")
        # Modificar config para usar esta seed
        cfg["splits"]["seed"] = seed
        cfg["training"]["seed"] = seed  # se existir
        seed_dir = results_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        from src.training.trainer import train
        train(cfg)  # isso salva em models/checkpoints/ e outputs/results/
        # Depois avaliar
        from src.evaluation.evaluate import run_evaluate
        run_evaluate(cfg, split="test", calibrate=False)
        # Copiar métricas para seed_dir
        test_metrics = Path(cfg["paths"]["results_dir"]) / "test_metrics.json"
        if test_metrics.exists():
            import shutil
            shutil.copy(test_metrics, seed_dir / "test_metrics.json")
            with open(seed_dir / "test_metrics.json") as f:
                metrics = json.load(f)
            all_metrics.append(metrics)
    # Agregar
    if all_metrics:
        summary = {}
        for key in all_metrics[0].keys():
            if isinstance(all_metrics[0][key], (int, float)):
                values = [m[key] for m in all_metrics]
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
        with open(results_dir / "sweep_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Sweep summary saved to {results_dir / 'sweep_summary.json'}")
    logger.info("sweep: done")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run.py", description="CycloneNet pipeline runner.")
    p.add_argument("--log-level", default="INFO", help="Logging level (INFO, DEBUG, ...)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare", help="Download/prepare IBTrACS and generate event list")
    sp.add_argument("--force", action="store_true", help="Rebuild outputs even if they exist")
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("download-era5", help="Download monthly ERA5 NetCDF files")
    sp.set_defaults(func=cmd_download_era5)

    sp = sub.add_parser("preprocess", help="Extract cubes/metadata into data/interim")
    sp.set_defaults(func=cmd_preprocess)

    sp = sub.add_parser("normalize", help="Gatekeeper: manifest+splits+minimal sanity+normalization stats")
    sp.set_defaults(func=cmd_normalize)

    sp = sub.add_parser("train", help="Train the model")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("evaluate", help="Evaluate on test split and write reports")
    sp.add_argument("--split", default="test", choices=["test", "val"], help="Split to evaluate")
    sp.add_argument("--calibrate", action="store_true", help="Apply Platt calibration using validation set")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("dataqa", help="Run data quality assurance checks")
    sp.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to check")
    sp.set_defaults(func=cmd_dataqa)

    sp = sub.add_parser("download-tchp", help="Download TCHP/OHC data from external sources")
    sp.add_argument("--force", action="store_true", help="Redownload even if exists")
    sp.set_defaults(func=cmd_download_tchp)

    sp = sub.add_parser("preprocess-tchp", help="Add TCHP maxima to event metadata")
    sp.set_defaults(func=cmd_preprocess_tchp)

    sp = sub.add_parser("baseline", help="Train and evaluate a tabular baseline (logistic regression)")
    sp.add_argument("--split", default="test", choices=["test", "val"], help="Split to evaluate")
    sp.set_defaults(func=cmd_baseline)

    sp = sub.add_parser("sweep", help="Run training with multiple seeds and aggregate results")
    sp.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated list of seeds")
    sp.set_defaults(func=cmd_sweep)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        args.func(args)
        return 0
    except Exception as e:
        logger.exception("Command failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())