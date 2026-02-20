"""Configuration loader (single source of truth)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


def cfg_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def load_config(config_name: str = "config.yaml") -> Dict[str, Any]:
    here = Path(__file__).resolve()
    project_root = None
    config_path = None
    for parent in [here.parent, *here.parents]:
        candidate = parent / config_name
        if candidate.exists():
            project_root = parent
            config_path = candidate
            break
    if project_root is None:
        project_root = Path.cwd().resolve()
        config_path = project_root / config_name
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    load_dotenv()
    cfg.setdefault("download", {}).setdefault("cds_api", {})
    if not cfg["download"]["cds_api"].get("key"):
        cfg["download"]["cds_api"]["key"] = os.getenv("CDSAPI_KEY", "")
    if not cfg["download"]["cds_api"].get("url"):
        cfg["download"]["cds_api"]["url"] = os.getenv(
            "CDSAPI_URL", "https://cds.climate.copernicus.eu/api"
        )

    cfg.setdefault("paths", {})
    for k, v in list(cfg["paths"].items()):
        if v is None:
            continue
        p = Path(str(v))
        if not p.is_absolute():
            p = project_root / p
        cfg["paths"][k] = p.resolve()

    cfg["project_root"] = project_root
    cfg["config_path"] = config_path.resolve()
    return cfg


CONFIG = load_config()
