#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CycloneNet — end-to-end runner (reproducibility-first, cross-platform).

Hard guarantees:
- NEVER renames/moves/rewrites existing ERA5 NetCDF (*.nc) files. Only reads them.
- Lazy imports per command to avoid import cascades.
- `normalize` is the pipeline gatekeeper:
    1) Builds a valid manifest from data/interim (*.json + *.npy pairs)
    2) Builds deterministic splits (train/val/test) using ONLY valid events
    3) Runs a minimal sanity gate (hard-fail, no plausibility heuristics):
        - shape/channel consistency
        - wind identity check if wind_mps exists
        - anti-leakage: total_heat_flux is loss-only, must not be model input
    4) Computes normalization stats ONLY for model.input_channels_names (train-only)

Commands:
  python run.py prepare
  python run.py download-era5
  python run.py preprocess
  python run.py normalize
  python run.py train
  python run.py evaluate
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd

from src.utils.config import load_config, cfg_get, ensure_dirs

logger = logging.getLogger("cyclonenet.run")


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _import_first(module_names: List[str]) -> Any:
    """Import the first available module from a list of module paths."""
    last_err: Exception | None = None
    for name in module_names:
        try:
            return importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise ImportError(
        f"Could not import any of: {module_names}. Last error: {last_err}")


def _resolve_callable(mod: Any, names: List[str]) -> Callable:
    """Return the first callable attribute found in `mod` by name."""
    for n in names:
        fn = getattr(mod, n, None)
        if callable(fn):
            return fn
    raise AttributeError(
        f"None of these callables exist in {mod.__name__}: {names}")


def _normalized_dir(cfg: dict) -> Path:
    return Path(cfg_get(cfg, "paths.normalized_dir", "./data/normalized")).resolve()


def ensure_normalized_paths(cfg: dict) -> dict:
    """
    Ensure runtime cfg has normalized output paths (in-memory only).
    We do NOT modify config.yaml on disk.
    """
    cfg.setdefault("paths", {})
    nd = _normalized_dir(cfg)

    cfg["paths"].setdefault("normalized_dir", str(nd.as_posix()))
    cfg["paths"].setdefault("valid_manifest", str(
        (nd / "valid_events.csv").as_posix()))
    cfg["paths"].setdefault("splits_csv", str((nd / "splits.csv").as_posix()))
    cfg["paths"].setdefault("normalization_stats", str(
        (nd / "normalization_stats.json").as_posix()))
    cfg["paths"].setdefault("sanity_report", str(
        (nd / "sanity_report.json").as_posix()))
    cfg["paths"].setdefault("runtime_snapshot", str(
        (nd / "runtime_config_snapshot.json").as_posix()))
    return cfg


def save_runtime_snapshot(cfg: dict) -> None:
    nd = _normalized_dir(cfg)
    nd.mkdir(parents=True, exist_ok=True)
    snap = Path(cfg_get(cfg, "paths.runtime_snapshot", nd /
                "runtime_config_snapshot.json")).resolve()
    snap.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# -----------------------------------------------------------------------------
# normalize helpers: manifest + splits + minimal sanity gate
# -----------------------------------------------------------------------------
def ensure_valid_manifest(cfg: dict) -> Path:
    """
    Build data/normalized/valid_events.csv by scanning for pairs:
      data/interim/{event_id}.json AND data/interim/{event_id}.npy
    """
    interim = Path(cfg_get(cfg, "paths.interim_data",
                   "./data/interim")).resolve()
    nd = _normalized_dir(cfg)
    nd.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(cfg_get(cfg, "paths.valid_manifest",
                         nd / "valid_events.csv")).resolve()

    json_files = sorted(glob.glob(str(interim / "era5_*.json")))
    rows = []
    for jp in json_files:
        eid = Path(jp).stem
        npy = interim / f"{eid}.npy"
        if npy.exists():
            rows.append({"event_id": eid})

    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv(manifest_path, index=False)
    logger.info("normalize: wrote manifest %s (n=%d)", manifest_path, len(df))
    return manifest_path


def ensure_splits(cfg: dict, manifest_path: Path) -> Path:
    """
    Build data/normalized/splits.csv from valid_events.csv only.
    Schema: event_id, split (train|val|test)
    """
    splits_path = Path(cfg_get(cfg, "paths.splits_csv",
                       "./data/normalized/splits.csv")).resolve()
    splits_path.parent.mkdir(parents=True, exist_ok=True)

    seed = int(cfg_get(cfg, "splits.seed", 1337))
    train_frac = float(cfg_get(cfg, "splits.train_frac", 0.70))
    val_frac = float(cfg_get(cfg, "splits.val_frac", 0.15))
    test_frac = float(cfg_get(cfg, "splits.test_frac", 0.15))
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("splits fractions must sum to 1.0")

    df = pd.read_csv(manifest_path)
    ids = df["event_id"].astype(str).values
    if len(ids) == 0:
        raise RuntimeError("No valid events found. Run preprocess first.")

    rng = np.random.RandomState(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_tr = int(train_frac * n)
    n_va = int(val_frac * n)

    split = np.array(["train"] * n, dtype=object)
    split[n_tr:n_tr + n_va] = "val"
    split[n_tr + n_va:] = "test"

    out = pd.DataFrame({"event_id": ids, "split": split})
    out.to_csv(splits_path, index=False)
    logger.info("normalize: wrote splits %s %s", splits_path,
                out["split"].value_counts().to_dict())
    return splits_path


def sanity_gate_minimal(cfg: dict, manifest_path: Path) -> None:
    """
    Minimal, scientifically safe sanity gate (hard-fail).
    Enforces:
      (1) structural consistency: cube C matches meta['channels'] length
      (2) mathematical identity: wind_mps == sqrt(u10_mps^2 + v10_mps^2) when wind exists
      (3) anti-leakage: total_heat_flux must not be a model input channel when configured loss-only
    """
    interim = Path(cfg_get(cfg, "paths.interim_data",
                   "./data/interim")).resolve()
    report_path = Path(cfg_get(cfg, "paths.sanity_report",
                       "./data/normalized/sanity_report.json")).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # (3) anti-leakage (global)
    input_names = list(cfg_get(cfg, "model.input_channels_names", []))
    thf_name = str(cfg_get(
        cfg, "physics_guided.losses.total_heat_flux_channel_name", "total_heat_flux_Wpm2"))
    exclude_thf = bool(
        cfg_get(cfg, "physics_guided.losses.exclude_total_heat_flux_from_input", True))
    if exclude_thf and thf_name in input_names:
        report = {
            "status": "FAILED",
            "reason": "leakage_detected",
            "details": {
                "message": "total_heat_flux is configured as physics-loss-only but is present in model.input_channels_names",
                "total_heat_flux_channel_name": thf_name,
                "model_input_channels_names": input_names,
            },
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise RuntimeError(f"Sanity gate FAILED (leakage). See: {report_path}")

    df = pd.read_csv(manifest_path)
    ids = df["event_id"].astype(str).tolist()
    if not ids:
        report = {"status": "FAILED", "reason": "empty_manifest"}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"Sanity gate FAILED (empty manifest). See: {report_path}")

    sample_n = int(cfg_get(cfg, "sanity.sample_n", 200))
    seed = int(cfg_get(cfg, "splits.seed", 1337))
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    sample = ids[: min(sample_n, len(ids))]

    wind_rel_err_max = float(cfg_get(cfg, "sanity.wind_rel_err_max", 0.05))

    failures: list[dict] = []
    checked = 0

    for eid in sample:
        jp = interim / f"{eid}.json"
        npy = interim / f"{eid}.npy"
        if not jp.exists() or not npy.exists():
            continue

        meta = _load_json(jp)
        chs = list(meta.get("channels", []))
        if not chs:
            failures.append({"event_id": eid, "error": "missing_channels"})
            continue

        cube = np.load(npy).astype(np.float32)
        checked += 1

        # (1) structural
        if cube.shape[-1] != len(chs):
            failures.append({
                "event_id": eid,
                "error": "shape_channel_mismatch",
                "cube_C": int(cube.shape[-1]),
                "channels_len": int(len(chs)),
            })
            continue

        # (2) wind identity
        if ("u10_mps" in chs) and ("v10_mps" in chs) and ("wind_mps" in chs):
            iu = chs.index("u10_mps")
            iv = chs.index("v10_mps")
            iw = chs.index("wind_mps")
            u = cube[:, :, :, iu]
            v = cube[:, :, :, iv]
            wind_true = np.sqrt(u * u + v * v)
            wind = cube[:, :, :, iw]
            denom = np.maximum(1e-6, float(np.nanmean(wind_true)))
            rel_err = float(np.nanmean(np.abs(wind - wind_true)) / denom)
            if (not np.isfinite(rel_err)) or (rel_err > wind_rel_err_max):
                failures.append({
                    "event_id": eid,
                    "error": "wind_identity_failed",
                    "rel_err": rel_err,
                    "threshold": wind_rel_err_max,
                })

    report = {
        "status": "PASSED" if not failures else "FAILED",
        "checked_events": checked,
        "sample_size": len(sample),
        "failure_count": len(failures),
        "failures": failures[:200],
        "notes": "Minimal gate: structural + wind identity + anti-leakage only.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(
            f"Sanity gate FAILED ({len(failures)} issues). See: {report_path}")

    logger.info(
        "normalize: sanity gate PASSED (checked=%d). Report: %s", checked, report_path)


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------
def cmd_prepare(args: argparse.Namespace) -> None:
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    save_runtime_snapshot(cfg)

    try:
        mod = _import_first(
            ["src.processors.ibtracs", "src.downloaders.ibtracs"])
        fn = _resolve_callable(
            mod, ["run_prepare", "prepare", "generate_event_list", "main"])
        try:
            fn(cfg, force=bool(args.force))  # type: ignore[misc]
        except TypeError:
            try:
                fn(force=bool(args.force))  # type: ignore[misc]
            except TypeError:
                fn()  # type: ignore[misc]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"prepare failed: {e}") from e

    logger.info("prepare: done")


def cmd_download_era5(args: argparse.Namespace) -> None:
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    save_runtime_snapshot(cfg)

    try:
        mod = _import_first(["src.processors.era5", "src.downloaders.era5"])
        if hasattr(mod, "ERA5Downloader"):
            dl = mod.ERA5Downloader()  # type: ignore[attr-defined]
            dl.download_required_batch()
        else:
            fn = _resolve_callable(
                mod, ["download_era5", "run_download_era5", "main"])
            try:
                fn(cfg)  # type: ignore[misc]
            except TypeError:
                fn()  # type: ignore[misc]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"download-era5 failed: {e}") from e

    logger.info("download-era5: done")


def cmd_preprocess(args: argparse.Namespace) -> None:
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    save_runtime_snapshot(cfg)

    mod = _import_first(["src.processors.preprocess_scientific"])
    fn = _resolve_callable(
        mod, ["run_preprocess", "run", "main", "process_all_events"])
    try:
        fn(cfg)  # type: ignore[misc]
    except TypeError:
        fn()  # type: ignore[misc]

    logger.info("preprocess: done")


def cmd_normalize(args: argparse.Namespace) -> None:
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    save_runtime_snapshot(cfg)

    manifest = ensure_valid_manifest(cfg)
    ensure_splits(cfg, manifest)
    sanity_gate_minimal(cfg, manifest)

    logger.info(
        "normalize: computing normalization stats (train-only, input channels only)")
    mod = _import_first(["src.data.normalization"])
    fn = _resolve_callable(mod, ["compute_norm_stats", "main"])
    try:
        fn()  # compute_norm_stats reads config.yaml
    except TypeError:
        fn(cfg)  # allow cfg-driven implementation
    logger.info("normalize: done")


def cmd_train(args: argparse.Namespace) -> None:
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    save_runtime_snapshot(cfg)

    stats_path = Path(cfg_get(cfg, "paths.normalization_stats", "./data/normalized/normalization_stats.json")).resolve()
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}. Run: python run.py normalize")

    mod = _import_first(["src.training.trainer"])
    fn = _resolve_callable(mod, ["train", "main"])

    # Training must be config-driven for reproducibility; do not fallback to fn() here.
    fn(cfg)  # type: ignore[misc]

    logger.info("train: done")

def cmd_evaluate(args: argparse.Namespace) -> None:
    cfg = ensure_normalized_paths(load_config("config.yaml"))
    ensure_dirs(cfg)
    save_runtime_snapshot(cfg)

    mod = _import_first(["src.evaluation.evaluate"])

    # Prefer config-driven wrappers; internal evaluate(...) requires many arguments.
    fn = _resolve_callable(mod, ["run_evaluate", "main"])

    # Evaluation must be config-driven for reproducibility; do not fallback to fn() here.
    fn(cfg)  # type: ignore[misc]

    logger.info("evaluate: done")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py", description="CycloneNet pipeline runner.")
    p.add_argument("--log-level", default="INFO",
                   help="Logging level (INFO, DEBUG, ...)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser(
        "prepare", help="Download/prepare IBTrACS and generate event list")
    sp.add_argument("--force", action="store_true",
                    help="Rebuild outputs even if they exist")
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser(
        "download-era5", help="Download monthly ERA5 NetCDF files")
    sp.set_defaults(func=cmd_download_era5)

    sp = sub.add_parser(
        "preprocess", help="Extract cubes/metadata into data/interim")
    sp.set_defaults(func=cmd_preprocess)

    sp = sub.add_parser(
        "normalize", help="Gatekeeper: manifest+splits+minimal sanity+normalization stats")
    sp.set_defaults(func=cmd_normalize)

    sp = sub.add_parser("train", help="Train the model")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser(
        "evaluate", help="Evaluate on test split and write reports")
    sp.set_defaults(func=cmd_evaluate)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        args.func(args)
        return 0
    except Exception as e:  # noqa: BLE001
        logger.exception("Command failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
