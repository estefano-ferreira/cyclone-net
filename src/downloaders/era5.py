#!/usr/bin/env python3
"""
CycloneNet: ERA5 downloader – scientific version.
Downloads monthly NetCDF files from Copernicus CDS and preserves them untouched.
No splitting, no renaming, no deletion.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import cdsapi
import pandas as pd
from tqdm import tqdm

from src.utils.config import CONFIG

logger = logging.getLogger(__name__)


def find_existing_monthly_file(raw_dir: Path, year: int, month: int) -> Optional[Path]:
    """Find an existing monthly file for given year/month (any suffix)."""
    pattern = f"era5_{year}_{month:02d}*.nc"
    matches = sorted(raw_dir.glob(pattern))
    return matches[0] if matches else None


class ERA5Downloader:
    """Downloads missing monthly ERA5 batches. Original files are never altered."""

    def __init__(self):
        self.c = cdsapi.Client(
            url=CONFIG["download"]["cds_api"]["url"],
            key=CONFIG["download"]["cds_api"]["key"]
        )
        self.raw_dir = Path(CONFIG["paths"]["raw_data"]).resolve()
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.area = CONFIG["download"].get(
            "spatial_subset", [35, -100, 5, -20])
        self.max_workers = CONFIG["download"].get("max_workers", 2)
        self.max_retries = 3
        self.retry_delay = 5
        self.required_vars = CONFIG["download"]["variables"]

    def _download_month(self, year: int, month: int, days: List[int], hours: List[int]) -> Optional[Path]:
        """Download one monthly batch if missing."""
        existing = find_existing_monthly_file(self.raw_dir, year, month)
        if existing:
            logger.info(f"File {existing.name} already exists. Skipping.")
            return None

        filename = f"era5_{year}_{month:02d}.nc"
        filepath = self.raw_dir / filename

        request = {
            "product_type": "reanalysis",
            "variable": self.required_vars,
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in days],
            "time": [f"{h:02d}:00" for h in hours],
            "data_format": "netcdf",
            "grid": CONFIG["download"]["grid"],
            "area": self.area,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Downloading {year}-{month:02d} (attempt {attempt})")
                self.c.retrieve(
                    CONFIG["download"]["dataset"],
                    request,
                    filepath.as_posix()
                )
                if filepath.exists():
                    logger.info(f"Downloaded {filename}")
                    return filepath
                else:
                    raise RuntimeError("File missing after download.")
            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {e}")
                if filepath.exists():
                    filepath.unlink()  # remove partial file
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** (attempt - 1)))

        logger.error(
            f"Failed to download {year}-{month:02d} after {self.max_retries} attempts.")
        return None

    def download_required_batch(self) -> None:
        """Download all months listed in required_timestamps.csv."""
        ts_path = self.raw_dir / "required_timestamps.csv"
        if not ts_path.exists():
            logger.error(
                "required_timestamps.csv not found. Run --prepare first.")
            return

        df_ts = pd.read_csv(ts_path)
        groups = df_ts.groupby(["year", "month"])

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for (year, month), group in groups:
                days = sorted(group["day"].unique())
                hours = sorted(group["hour"].unique())
                futures.append(executor.submit(
                    self._download_month, year, month, days, hours))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Monthly batches"):
                try:
                    result = future.result()
                    if result:
                        logger.info(f"Finished: {result.name}")
                except Exception as e:
                    logger.error(f"Error downloading a batch: {e}")
