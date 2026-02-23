# src/processors/preprocess_tchp.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

from src.utils.config import cfg_get
from src.utils.tchp_utils import load_tchp_file, find_tchp_max, get_tchp_file_path

logger = logging.getLogger(__name__)


def add_tchp_to_metadata(cfg: Dict[str, Any]) -> None:
    """
    For each event in data/interim, load corresponding TCHP file,
    find the maximum in the vicinity, and add tchp_max_lat/lon to the JSON metadata.
    """
    interim_dir = Path(cfg_get(cfg, "paths.interim_data",
                       "./data/interim")).resolve()
    tchp_dir = Path(cfg_get(cfg, "paths.tchp_dir",
                    "./data/external/tchp")).resolve()
    if not tchp_dir.exists():
        logger.error(f"TCHP directory not found: {tchp_dir}")
        return

    # Find all JSON files
    json_files = sorted(interim_dir.glob("era5_*.json"))
    if not json_files:
        logger.warning("No event JSON files found in interim directory.")
        return

    updated = 0
    skipped = 0

    for json_path in tqdm(json_files, desc="Adding TCHP to metadata"):
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Skip if already has TCHP fields
        if "tchp_max_lat" in meta and meta["tchp_max_lat"] is not None:
            skipped += 1
            continue

        # Extract timestamp and center coordinates
        timestamp = pd.to_datetime(meta.get("timestamp"))
        lat = meta.get("center_lat")
        lon = meta.get("center_lon")
        if timestamp is None or lat is None or lon is None:
            logger.debug(
                f"Skipping {json_path.name}: missing timestamp/center")
            continue

        year = timestamp.year
        # Determine source automatically based on year
        if year >= 2022:
            src = 'noaa'
        elif year >= 1993:
            src = 'aoml'
        else:
            src = 'copernicus'
        tchp_file = get_tchp_file_path(tchp_dir, year, src)
        if not tchp_file.exists():
            logger.debug(f"TCHP file not found: {tchp_file}")
            continue

        tchp_data = load_tchp_file(
            tchp_file, timestamp, lat, lon, window_deg=5)
        if tchp_data is None:
            continue
        tchp, lats_tchp, lons_tchp = tchp_data
        tchp_max_lat, tchp_max_lon = find_tchp_max(
            tchp, lats_tchp, lons_tchp, window_px=3)

        # Update metadata
        meta["tchp_max_lat"] = tchp_max_lat
        meta["tchp_max_lon"] = tchp_max_lon

        # Save back
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        updated += 1

    logger.info(
        f"TCHP metadata added to {updated} events ({skipped} already had).")


def run_preprocess_tchp(cfg: Dict[str, Any]) -> None:
    add_tchp_to_metadata(cfg)
