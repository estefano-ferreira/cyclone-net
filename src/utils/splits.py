"""Storm-level data splitting (scientific, leakage-safe, config-driven)."""

from __future__ import annotations


import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.utils.config import cfg_get


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
        seed = int(cfg_get(cfg, "splits.seed", cfg_get(cfg, "training.seed", 42)))
        train = float(cfg_get(cfg, "splits.train", 0.70))
        val = float(cfg_get(cfg, "splits.val", 0.15))
        test = float(cfg_get(cfg, "splits.test", 0.15))
        persist = bool(cfg_get(cfg, "splits.persist", True))
        path = Path(str(cfg_get(cfg, "paths.splits_csv", "./data/normalized/splits.csv"))).resolve()
        return SplitConfig(train=train, val=val, test=test, seed=seed, method=method, persist=persist, path=path)


def _validate(cfg: SplitConfig) -> None:
    total = cfg.train + cfg.val + cfg.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")
    if cfg.method not in {"sid", "storm_name"}:
        raise ValueError(f"Unknown split method: {cfg.method}")


def make_splits(metadata_csv: str | Path, cfg: SplitConfig) -> Dict[str, Any]:
    _validate(cfg)
    df = pd.read_csv(metadata_csv)
    group_col = "sid" if cfg.method == "sid" else "storm_name"
    label_col = "ri_label" if "ri_label" in df.columns else None

    if group_col not in df.columns:
        raise KeyError(f"Metadata missing column '{group_col}'")
    groups_df = df[[group_col] + ([label_col] if label_col else [])].dropna(subset=[group_col]).copy()
    groups_df[group_col] = groups_df[group_col].astype(str)

    if label_col:
        group_pos = groups_df.groupby(group_col)[label_col].max().to_dict()
        positive_groups = [g for g, y in group_pos.items() if int(y) == 1]
        negative_groups = [g for g, y in group_pos.items() if int(y) == 0]
    else:
        unique_groups = sorted(groups_df[group_col].unique().tolist())
        positive_groups = []
        negative_groups = unique_groups

    rng = random.Random(cfg.seed)
    rng.shuffle(positive_groups)
    rng.shuffle(negative_groups)

    def assign(groups: List[str]) -> Dict[str, List[str]]:
        n = len(groups)
        n_train = int(round(n * cfg.train))
        n_val = int(round(n * cfg.val))
        n_train = min(n_train, n)
        n_val = min(n_val, max(0, n - n_train))
        return {
            "train": groups[:n_train],
            "val": groups[n_train:n_train + n_val],
            "test": groups[n_train + n_val:],
        }

    pos_assign = assign(positive_groups)
    neg_assign = assign(negative_groups)
    split_groups = {
        split: sorted(pos_assign[split] + neg_assign[split])
        for split in ["train", "val", "test"]
    }

    group_to_split = {g: split for split, groups in split_groups.items() for g in groups}
    out = df.copy()
    out["split"] = out[group_col].astype(str).map(group_to_split)
    out = out.dropna(subset=["split"]).copy()

    if "event_id" not in out.columns:
        raise KeyError("metadata_csv must contain 'event_id'")

    out_csv = out[["event_id", "split"]].copy()
    if cfg.persist:
        cfg.path.parent.mkdir(parents=True, exist_ok=True)
        out_csv.to_csv(cfg.path, index=False)

    summary = {
        "group_col": group_col,
        "seed": cfg.seed,
        "ratios": {"train": cfg.train, "val": cfg.val, "test": cfg.test},
        "counts": out_csv["split"].value_counts().to_dict(),
        "n_unique_groups": int(len(group_to_split)),
    }
    summary_path = cfg.path.with_suffix(".summary.json")
    if cfg.persist:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary