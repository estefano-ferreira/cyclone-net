#!/usr/bin/env python3
"""
CycloneNet V2.x — end-to-end runner (scientific + reproducible).

Pipeline stages:
1) IBTrACS download + event list generation (RI labeling)
2) ERA5 selective download for required timestamps (monthly batches)
3) Optional split monthly -> daily files (disabled by default)
4) Preprocessing: build physics cubes + samples_metadata.csv
5) Normalization: compute training‑only stats (normalization_stats.json)
6) Training (single phase) using CycloneTrainer
7) Evaluation (metrics + optional calibration/interpretability)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import yaml
import torch


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger("run")


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dirs() -> None:
    from src.utils.config import CONFIG
    paths = CONFIG.get("paths", {})
    for k in ["raw_data", "interim_data", "processed_data", "checkpoints", "logs", "figures", "results"]:
        p = paths.get(k)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)


def cmd_prepare(args: argparse.Namespace) -> None:
    from src.downloaders.ibtracs import download_ibtracs
    from src.processors.ibtracs import generate_event_list
    ensure_dirs()
    csv_path = download_ibtracs(force_download=bool(args.force))
    if not csv_path:
        raise RuntimeError("IBTrACS download failed.")
    generate_event_list(csv_path)
    logger.info("prepare: done")


def cmd_download_era5(args: argparse.Namespace) -> None:
    from src.downloaders.era5 import ERA5Downloader
    ensure_dirs()
    ERA5Downloader().download_required_batch()
    logger.info("download-era5: done")


def cmd_split_era5(args: argparse.Namespace) -> None:
    logger.warning(
        "split-era5 is deprecated; monthly files are used directly.")
    # No action needed


def cmd_preprocess(args: argparse.Namespace) -> None:
    from src.utils.config import CONFIG
    from src.processors.preprocess_scientific import process_all_events
    ensure_dirs()
    process_all_events(str(CONFIG["paths"]["event_list"]))
    logger.info("preprocess: done")


def cmd_normalize(args: argparse.Namespace) -> None:
    from src.data.normalization import main as normalize_main
    ensure_dirs()
    normalize_main()
    logger.info("normalize: done")


def cmd_train(args: argparse.Namespace) -> None:
    from src.utils.config import CONFIG
    from src.training.trainer import CycloneTrainer
    from src.data.dataset import PhysicsDataset

    ensure_dirs()

    if args.lr is not None:
        CONFIG.setdefault("training", {})["learning_rate"] = float(args.lr)

    train_ds = PhysicsDataset(split="train", augment=True, balance_ri=False)
    val_ds = PhysicsDataset(split="val", augment=False, balance_ri=False)

    trainer = CycloneTrainer(
        CONFIG, train_dataset=train_ds, val_dataset=val_ds)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        trainer.model.load_state_dict(torch.load(
            ckpt_path, map_location=trainer.device))
        logger.info(f"Loaded checkpoint from {ckpt_path}")

    trainer.train(epochs=int(args.epochs))
    logger.info("train: done")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from src.utils.config import CONFIG
    ensure_dirs()
    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(
        CONFIG["paths"]["checkpoints"]) / "best_model_ri.pt"
    from src.evaluation.evaluate import main as evaluate_main
    evaluate_main(checkpoint=str(ckpt_path))
    logger.info("evaluate: done")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py", description="CycloneNet V2.x pipeline runner.")
    p.add_argument("--config", default="config.yaml",
                   help="Path to config.yaml (expected at project root).")
    p.add_argument("--log-level", default="INFO",
                   help="Logging level (INFO, DEBUG, ...)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("prepare")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("download-era5")
    sp.set_defaults(func=cmd_download_era5)

    sp = sub.add_parser("split-era5")
    sp.set_defaults(func=cmd_split_era5)

    sp = sub.add_parser("preprocess")
    sp.set_defaults(func=cmd_preprocess)

    sp = sub.add_parser("normalize")
    sp.set_defaults(func=cmd_normalize)

    sp = sub.add_parser("train")
    sp.add_argument("--epochs", type=int, default=200)
    sp.add_argument("--checkpoint", default=None)
    sp.add_argument("--lr", type=float, default=None)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("evaluate")
    sp.add_argument("--checkpoint", default=None)
    sp.set_defaults(func=cmd_evaluate)

    return p


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(args.log_level)

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        logger.error(f"Config not found: {cfg_path}")
        return 2

    try:
        args.func(args)
        return 0
    except Exception as e:
        logger.exception(f"Command failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
