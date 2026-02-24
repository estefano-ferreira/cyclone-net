# src/utils/snapshot.py
"""
Capture a snapshot of the runtime environment and configuration.
This ensures full provenance for every experiment.
"""

import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import xarray as xr

from .git import get_git_revision_short_hash


def get_versions() -> Dict[str, str]:
    """Return versions of key libraries and Python environment."""
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "xarray": xr.__version__,
        "platform": platform.platform(),
    }


def save_run_snapshot(cfg: Dict[str, Any], run_dir: Path, command: str) -> None:
    """
    Save a JSON snapshot containing the configuration, versions, Git commit,
    and timestamp to run_dir/run_snapshot.json.
    """
    snapshot = {
        "command": command,
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_revision_short_hash(),
        "config": cfg,
        "versions": get_versions(),
        "cwd": str(Path.cwd()),
    }
    snapshot_path = run_dir / "run_snapshot.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)