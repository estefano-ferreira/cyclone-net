"""Storm-level data splitting (scientific, leakage-safe, config-driven).

Split assignment is HASH-DETERMINISTIC per storm: each SID's split depends
only on sha256(SID), never on the composition of the dataset. Adding or
removing storms can never move a pre-existing storm to another split — the
property that keeps a frozen test benchmark valid as the archive grows.

A frozen-override map (JSON, SID -> split) takes priority over the hash so
historically assigned storms (in particular the frozen test benchmark) keep
their original split even where the hash disagrees.

There is NO label stratification (deliberate): stratification requires
knowing the full label set, which reintroduces composition dependence. Class
proportions per split are approximate and converge by the law of large
numbers as the dataset grows; on small datasets they can deviate — measure,
don't assume.
"""

from __future__ import annotations


import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.utils.config import cfg_get

# Resolution of the hash bucket in [0, 1). 1e8 buckets keep boundary-rounding
# far below any realistic fraction granularity.
_HASH_BUCKETS = 10**8


@dataclass(frozen=True)
class SplitConfig:
    train: float
    val: float
    test: float
    seed: int
    method: str
    persist: bool
    path: Path
    frozen_map_path: Optional[Path] = None

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> "SplitConfig":
        method = str(cfg_get(cfg, "splits.method", "sid")).lower()
        # NOTE: the seed no longer influences assignment (pure per-SID hash);
        # it is kept for config compatibility and recorded in the summary.
        seed = int(cfg_get(cfg, "splits.seed", cfg_get(cfg, "training.seed", 42)))
        # Accept both the documented `*_frac` keys (config.yaml) and the short
        # `train/val/test` aliases so configuration is actually honoured.
        train = float(cfg_get(cfg, "splits.train", cfg_get(cfg, "splits.train_frac", 0.70)))
        val = float(cfg_get(cfg, "splits.val", cfg_get(cfg, "splits.val_frac", 0.15)))
        test = float(cfg_get(cfg, "splits.test", cfg_get(cfg, "splits.test_frac", 0.15)))
        persist = bool(cfg_get(cfg, "splits.persist", True))
        path = Path(str(cfg_get(cfg, "paths.splits_csv", "./data/normalized/splits.csv"))).resolve()
        frozen = cfg_get(cfg, "paths.frozen_splits", "./data/normalized/frozen_splits.json")
        frozen_path = Path(str(frozen)).resolve() if frozen else None
        return SplitConfig(train=train, val=val, test=test, seed=seed, method=method,
                           persist=persist, path=path, frozen_map_path=frozen_path)


def hash_fraction(group_key: str) -> float:
    """Stable position of a group key in [0, 1).

    Depends ONLY on the key string (sha256), so it is invariant to dataset
    composition, input order, process, platform, and Python hash randomization.
    """
    digest = hashlib.sha256(group_key.encode("utf-8")).hexdigest()
    return (int(digest[:16], 16) % _HASH_BUCKETS) / _HASH_BUCKETS


def assign_split(group_key: str, cfg: SplitConfig,
                 frozen: Optional[Dict[str, str]] = None) -> str:
    """Assign one storm to a split: frozen override first, then pure hash."""
    if frozen:
        pinned = frozen.get(group_key)
        if pinned in ("train", "val", "test"):
            return pinned
    f = hash_fraction(group_key)
    if f < cfg.train:
        return "train"
    if f < cfg.train + cfg.val:
        return "val"
    return "test"


def load_frozen_map(path: Optional[Path]) -> Dict[str, str]:
    """Load the SID -> split override map (empty when absent)."""
    if path is None or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items()}


def _validate(cfg: SplitConfig) -> None:
    total = cfg.train + cfg.val + cfg.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")
    if cfg.method not in {"sid", "storm_name"}:
        raise ValueError(f"Unknown split method: {cfg.method}")


def make_splits(metadata_csv: str | Path, cfg: SplitConfig) -> Dict[str, Any]:
    """Assign every event to a split by its storm's hash (frozen map first).

    Guarantees:
      * one storm (SID) -> exactly one split (no storm-level leakage);
      * a storm's split never changes when other storms are added/removed;
      * frozen overrides (historical benchmark) win over the hash.
    """
    _validate(cfg)
    # keep_default_na: the event list carries literal "NA" (North Atlantic)
    # in text columns — pandas' default NA parsing must never run on it.
    df = pd.read_csv(metadata_csv, keep_default_na=False, na_values=[""])
    group_col = "sid" if cfg.method == "sid" else "storm_name"

    if group_col not in df.columns:
        raise KeyError(f"Metadata missing column '{group_col}'")
    if "event_id" not in df.columns:
        raise KeyError("metadata_csv must contain 'event_id'")

    # Split assignment is an inviolable path: an event that cannot be
    # assigned must FAIL LOUDLY, never be silently dropped.
    missing_group = df[group_col].isna() | (df[group_col].astype(str).str.strip() == "")
    if missing_group.any():
        sample = df.loc[missing_group, "event_id"].astype(str).head(5).tolist()
        raise ValueError(
            f"{int(missing_group.sum())} event(s) have a missing '{group_col}' and "
            f"cannot be assigned to a split (e.g. {sample}). Refusing to continue — "
            f"silent exclusion from splits is forbidden."
        )

    frozen = load_frozen_map(cfg.frozen_map_path)

    groups = sorted(df[group_col].astype(str).unique().tolist())
    group_to_split = {g: assign_split(g, cfg, frozen) for g in groups}
    n_frozen_used = sum(1 for g in groups if g in frozen)

    out = df.copy()
    out["split"] = out[group_col].astype(str).map(group_to_split)
    unmapped = out["split"].isna()
    if unmapped.any():
        sample = out.loc[unmapped, "event_id"].astype(str).head(5).tolist()
        raise ValueError(
            f"{int(unmapped.sum())} event(s) received no split assignment "
            f"(e.g. {sample}). Refusing to continue — silent exclusion from "
            f"splits is forbidden."
        )

    out_csv = out[["event_id", "split"]].copy()
    if cfg.persist:
        cfg.path.parent.mkdir(parents=True, exist_ok=True)
        out_csv.to_csv(cfg.path, index=False)

    summary = {
        "group_col": group_col,
        "method": "sid_hash",
        "seed": cfg.seed,  # informational only — assignment ignores it
        "ratios": {"train": cfg.train, "val": cfg.val, "test": cfg.test},
        "counts": out_csv["split"].value_counts().to_dict(),
        "n_unique_groups": int(len(group_to_split)),
        "n_frozen_overrides_applied": int(n_frozen_used),
        "frozen_map": str(cfg.frozen_map_path) if cfg.frozen_map_path else None,
    }
    summary_path = cfg.path.with_suffix(".summary.json")
    if cfg.persist:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
