"""
CycloneNet V2.1 – IBTrACS downloader.

This module handles downloading the official IBTrACS dataset from NOAA.
The downloaded file is stored in the raw data directory and remains unmodified.

Author: Estefano Senhor Ferreira
License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)
"""
import logging
import requests
from pathlib import Path
from tqdm import tqdm 

from src.utils.config import cfg_get

logger = logging.getLogger(__name__)

IBTRACS_URL = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.ALL.list.v04r00.csv"

def download_ibtracs(cfg: dict | None = None, force_download: bool = False) -> Path:
    """Download official IBTrACS CSV if not present, with progress bar."""
    if cfg is None:
        from src.utils.config import CONFIG
        cfg = CONFIG
    raw_dir = Path(cfg_get(cfg, "paths.raw_data", "./data/raw")).resolve()
    output_path = raw_dir / "ibtracs.ALL.list.v04r00.csv"
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
    total_size = int(response.headers.get('content-length', 0))
    with open(output_path, 'wb') as f, tqdm(
        desc="IBTrACS",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    logger.info(f"Download complete: {output_path}")
    return output_path