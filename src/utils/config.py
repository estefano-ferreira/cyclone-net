from __future__ import annotations
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    for k in ["raw_data", "interim_data", "processed_data", "results_dir", "logs_dir"]:
        p = Path(cfg["paths"][k])
        p.mkdir(parents=True, exist_ok=True)


def cfg_get(cfg: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# ---------------------------------------------------------------------
# Backward-compatible global config
# Many modules import CONFIG; keep it available to avoid import cascades.
# ---------------------------------------------------------------------
try:
    CONFIG = load_config("config.yaml")
except Exception:
    # Do not crash on import; CLI commands may call load_config later.
    CONFIG = {}
