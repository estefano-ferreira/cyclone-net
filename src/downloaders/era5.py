"""
CycloneNet: ERA5 downloader – scientific version.
Downloads monthly NetCDF files from Copernicus CDS and preserves them untouched.
No splitting, no renaming, no deletion.

Author: Estefano Senhor Ferreira
License: CC BY-NC 4.0
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import cdsapi
import requests
import urllib3
from tqdm import tqdm

from src.utils.config import cfg_get

# Disable insecure request warnings if SSL verification is off
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


def generate_required_timestamps(
    event_list_csv: Path,
    out_csv: Path,
    year_range: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Create required_timestamps.csv from event list, optionally filtering by year range.
    """
    df = pd.read_csv(event_list_csv)
    if "timestamp" in df.columns:
        df["dt"] = pd.to_datetime(df["timestamp"])
    elif "datetime" in df.columns:
        df["dt"] = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M")
    else:
        raise ValueError("Event list must have 'timestamp' or 'datetime' column")

    # Apply year range filter if provided
    if year_range is not None:
        start_year, end_year = year_range
        df = df[
            (df["dt"].dt.year >= start_year) & (df["dt"].dt.year <= end_year)
        ].copy()

    df["year"] = df["dt"].dt.year
    df["month"] = df["dt"].dt.month
    df["day"] = df["dt"].dt.day
    df["hour"] = df["dt"].dt.hour
    required = (
        df[["year", "month", "day", "hour"]]
        .drop_duplicates()
        .sort_values(["year", "month", "day", "hour"])
    )
    required.to_csv(out_csv, index=False)
    logger.info(
        f"Required timestamps saved to {out_csv} ({len(required)} rows)"
    )


class ERA5Downloader:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        # SSL verification setting
        verify_ssl = cfg_get(cfg, "download.cds_api.verify_ssl", True)
        session = requests.Session()
        if not verify_ssl:
            session.verify = False
            logger.warning(
                "SSL verification disabled for CDS API. This is insecure and should only be used for testing."
            )

        self.c = cdsapi.Client(
            url=cfg["download"]["cds_api"]["url"],
            key=cfg["download"]["cds_api"]["key"],
            session=session,
        )

        self.raw_dir = Path(cfg["paths"]["raw_data"]).resolve()
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Use spatial subset from config, fallback to a safe Atlantic default
        self.area = cfg["download"].get(
            "spatial_subset", [60, -140, 0, -20]   # N, W, S, E
        )
        self.max_workers = cfg["download"].get("max_workers", 2)
        self.max_retries = 3
        self.retry_delay = 5
        self.required_vars = cfg["download"]["variables"]

        # Get year range from config
        self.year_range = None
        years_cfg = cfg_get(cfg, "download.years", None)
        if years_cfg is not None and len(years_cfg) == 2:
            self.year_range = (int(years_cfg[0]), int(years_cfg[1]))

    def find_existing_monthly_file(self, year: int, month: int) -> Optional[Path]:
        pattern = f"era5_{year}_{month:02d}*.nc"
        matches = sorted(self.raw_dir.glob(pattern))
        return matches[0] if matches else None

    def _download_month(
        self, year: int, month: int, days: List[int], hours: List[int]
    ) -> Optional[Path]:
        # Skip if year outside config range (safety check)
        if self.year_range is not None:
            start, end = self.year_range
            if year < start or year > end:
                logger.warning(
                    f"Skipping year {year} (outside config range {start}-{end})"
                )
                return None

        existing = self.find_existing_monthly_file(year, month)
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
            "grid": self.cfg["download"]["grid"],
            "area": self.area,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Downloading {year}-{month:02d} (attempt {attempt})"
                )
                self.c.retrieve(
                    self.cfg["download"]["dataset"],
                    request,
                    filepath.as_posix(),
                )
                if filepath.exists():
                    logger.info(f"Downloaded {filename}")
                    return filepath
                else:
                    raise RuntimeError("File missing after download.")
            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {e}")
                if filepath.exists():
                    filepath.unlink()
                if attempt < self.max_retries:
                    sleep_time = self.retry_delay * (2 ** (attempt - 1))
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
        logger.error(
            f"Failed to download {year}-{month:02d} after {self.max_retries} attempts."
        )
        return None

    def download_required_batch(self) -> None:
        # Ensure required_timestamps.csv exists
        ts_path = self.raw_dir / "required_timestamps.csv"
        event_list = Path(
            cfg_get(
                self.cfg,
                "paths.event_list",
                "./data/event_list_augmented.csv",
            )
        )
        if not event_list.exists():
            raise FileNotFoundError(
                f"Event list not found: {event_list}. Run prepare first."
            )
        if not ts_path.exists():
            generate_required_timestamps(
                event_list, ts_path, year_range=self.year_range
            )

        df_ts = pd.read_csv(ts_path)
        groups = df_ts.groupby(["year", "month"])

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for (year, month), group in groups:
                days = sorted(group["day"].unique())
                hours = sorted(group["hour"].unique())
                futures.append(
                    executor.submit(self._download_month, year, month, days, hours)
                )

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Monthly batches"
            ):
                try:
                    result = future.result()
                    if result:
                        logger.info(f"Finished: {result.name}")
                except Exception as e:
                    logger.error(f"Error downloading a batch: {e}")