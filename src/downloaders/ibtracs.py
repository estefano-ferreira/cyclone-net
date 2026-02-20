"""IBTrACS download and basic parsing."""

import logging
import requests
from pathlib import Path
import pandas as pd

from src.utils.config import CONFIG

logger = logging.getLogger(__name__)

IBTRACS_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.ALL.list.v04r00.csv"


def download_ibtracs(force_download: bool = False) -> Path:
    """Download official IBTrACS CSV if not present."""
    output_path = CONFIG["paths"]["raw_data"] / "ibtracs.ALL.list.v04r00.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force_download:
        if output_path.stat().st_size > 10_000_000:
            logger.info(f"IBTrACS already exists ({output_path})")
            return output_path
        else:
            logger.warning("Existing file too small, re-downloading.")

    logger.info(f"Downloading IBTrACS from {IBTRACS_URL}...")
    response = requests.get(IBTRACS_URL, stream=True, timeout=120)
    response.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"Download complete: {output_path}")
    return output_path
