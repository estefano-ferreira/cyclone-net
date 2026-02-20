"""Storm-level data splitting (scientific, leakage-safe, config-driven)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import random
import pandas as pd

from src.utils.config import CONFIG, cfg_get


@dataclass(frozen=True)
class SplitConfig:
    train: float
    val: float
    test: float
    seed: int
    method: str
    persist: bool
    path: Path

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> "SplitConfig":
        method = str(cfg_get(cfg, "splits.method", "sid")).lower()
        seed = int(cfg_get(cfg, "splits.seed",
                   cfg_get(cfg, "training.seed", 42)))
        train = float(cfg_get(cfg, "splits.train", 0.70))
        val = float(cfg_get(cfg, "splits.val", 0.15))
        test = float(cfg_get(cfg, "splits.test", 0.15))
        persist = bool(cfg_get(cfg, "splits.persist", True))
        interim = Path(
            str(cfg_get(cfg, "paths.interim_data", "./data/interim"))).resolve()
        p = cfg_get(cfg, "paths.splits_file", None) or (
            interim / "splits.json")
        path = Path(str(p)).expanduser().resolve()
        return SplitConfig(train=train, val=val, test=test, seed=seed, method=method, persist=persist, path=path)


def _validate(cfg: SplitConfig) -> None:
    total = cfg.train + cfg.val + cfg.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")
    if cfg.method not in ("sid", "storm_name"):
        raise ValueError(f"Unknown splits.method: {cfg.method}")


def make_splits(metadata_csv: str | Path, cfg: SplitConfig) -> Dict[str, Any]:
    _validate(cfg)
    df = pd.read_csv(metadata_csv)
    group_col = "sid" if cfg.method == "sid" else "storm_name"
    if group_col not in df.columns:
        raise KeyError(f"Metadata missing column '{group_col}'")

    groups = sorted([str(x) for x in df[group_col].dropna().unique()])
    rng = random.Random(cfg.seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(n * cfg.train)
    n_val = int(n * cfg.val)

    obj = {
        "group_col": group_col,
        "seed": cfg.seed,
        "ratios": {"train": cfg.train, "val": cfg.val, "test": cfg.test},
        "groups": {
            "train": groups[:n_train],
            "val": groups[n_train:n_train + n_val],
            "test": groups[n_train + n_val:],
        },
    }

    if cfg.persist:
        cfg.path.parent.mkdir(parents=True, exist_ok=True)
        cfg.path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return obj


def load_splits(cfg: SplitConfig) -> Dict[str, Any]:
    _validate(cfg)
    if cfg.persist and cfg.path.exists():
        obj = json.loads(cfg.path.read_text(encoding="utf-8"))
        if "groups" in obj and "group_col" in obj:
            return obj
    raise FileNotFoundError(f"Splits file not found: {cfg.path}")
