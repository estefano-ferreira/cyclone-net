"""
CycloneNet: ERA5 pressure-levels downloader.

Downloads the minimal pressure-level fields required by the SHIPS-style
environmental channels (see src/processors/pressure_channels.py):

  * u/v_component_of_wind at [850, 200] hPa  -> era5pl_wind_YYYY_MM.nc
  * relative_humidity at [700, 600, 500] hPa -> era5pl_rh_YYYY_MM.nc

Two requests per month keep the transfer minimal (7 field-levels instead of
15 if all variables were requested at all levels). Grid, area, and required
timestamps mirror the single-level downloader so the two archives stay
co-registered. Files are monthly, skip-if-exists, and never modified.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cdsapi
import pandas as pd
import requests
import urllib3
from tqdm import tqdm

from src.utils.config import cfg_get

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

PL_DATASET_DEFAULT = "reanalysis-era5-pressure-levels"


class ERA5PressureDownloader:
    """Monthly pressure-level downloads driven by required_timestamps.csv."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

        verify_ssl = cfg_get(cfg, "download.cds_api.verify_ssl", True)
        session = requests.Session()
        if not verify_ssl:
            session.verify = False
            logger.warning("SSL verification disabled for CDS API (testing only).")

        # Credentials come EXCLUSIVELY from ~/.cdsapirc — see era5.py.
        self.c = cdsapi.Client(session=session)

        self.raw_dir = Path(cfg["paths"]["raw_data"]).resolve()
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = cfg_get(cfg, "download.pressure_levels.dataset", PL_DATASET_DEFAULT)
        self.area = cfg["download"].get("spatial_subset", [60, -140, 0, -20])
        self.grid = cfg["download"]["grid"]
        self.max_workers = cfg["download"].get("max_workers", 2)
        self.max_retries = 3
        self.retry_delay = 5

        self.wind_levels = [str(v) for v in cfg_get(cfg, "download.pressure_levels.wind_levels", [850, 200])]
        self.rh_levels = [str(v) for v in cfg_get(cfg, "download.pressure_levels.rh_levels", [700, 600, 500])]

        self.year_range: Optional[Tuple[int, int]] = None
        years_cfg = cfg_get(cfg, "download.years", None)
        if years_cfg is not None and len(years_cfg) == 2:
            self.year_range = (int(years_cfg[0]), int(years_cfg[1]))

    # Two file kinds per month: wind (shear levels) and RH (mid-level layer).
    def _jobs_for_month(self, year: int, month: int) -> List[Tuple[str, dict]]:
        return [
            (
                f"era5pl_wind_{year}_{month:02d}.nc",
                {"variable": ["u_component_of_wind", "v_component_of_wind"],
                 "pressure_level": self.wind_levels},
            ),
            (
                f"era5pl_rh_{year}_{month:02d}.nc",
                {"variable": ["relative_humidity"],
                 "pressure_level": self.rh_levels},
            ),
        ]

    def _download_month(self, year: int, month: int, days: List[int], hours: List[int]) -> List[Path]:
        if self.year_range is not None:
            start, end = self.year_range
            if year < start or year > end:
                logger.warning("Skipping year %s (outside config range %s-%s)", year, start, end)
                return []

        written: List[Path] = []
        for filename, job in self._jobs_for_month(year, month):
            filepath = self.raw_dir / filename
            if filepath.exists():
                logger.info("File %s already exists. Skipping.", filename)
                continue

            request = {
                "product_type": "reanalysis",
                "year": str(year),
                "month": f"{month:02d}",
                "day": [f"{d:02d}" for d in days],
                "time": [f"{h:02d}:00" for h in hours],
                "data_format": "netcdf",
                "grid": self.grid,
                "area": self.area,
                **job,
            }

            for attempt in range(1, self.max_retries + 1):
                try:
                    logger.info("Downloading %s (attempt %d)", filename, attempt)
                    self.c.retrieve(self.dataset, request, filepath.as_posix())
                    if not filepath.exists():
                        raise RuntimeError("File missing after download.")
                    written.append(filepath)
                    break
                except Exception as exc:
                    logger.error("Attempt %d failed for %s: %s", attempt, filename, exc)
                    if filepath.exists():
                        filepath.unlink()
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (2 ** (attempt - 1)))
            else:
                logger.error("Failed to download %s after %d attempts.", filename, self.max_retries)
        return written

    def download_required_batch(self) -> None:
        """Download PL files for every (year, month) in required_timestamps.csv.

        Reuses the same required-timestamps source as the single-level
        downloader so both archives cover exactly the same event months.
        """
        ts_path = self.raw_dir / "required_timestamps.csv"
        if not ts_path.exists():
            from src.downloaders.era5 import generate_required_timestamps

            event_list = Path(cfg_get(self.cfg, "paths.event_list", "./data/event_list_augmented.csv"))
            if not event_list.exists():
                raise FileNotFoundError(f"Event list not found: {event_list}. Run prepare first.")
            generate_required_timestamps(event_list, ts_path, year_range=self.year_range)

        df_ts = pd.read_csv(ts_path)
        groups = df_ts.groupby(["year", "month"])

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._download_month, int(year), int(month),
                                sorted(group["day"].unique()), sorted(group["hour"].unique()))
                for (year, month), group in groups
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="PL monthly batches"):
                try:
                    for path in future.result():
                        logger.info("Finished: %s", path.name)
                except Exception as exc:
                    logger.error("Error downloading a PL batch: %s", exc)

        # Sequential mop-up pass — same CDS concurrent-job-quota failure mode
        # as the single-level downloader; stragglers recover when retried
        # one at a time.
        missing = [(int(y), int(m), g) for (y, m), g in groups
                   if any(not (self.raw_dir / fn).exists()
                          for fn, _ in self._jobs_for_month(int(y), int(m)))]
        if missing:
            logger.warning("Sequential PL mop-up for %d month(s).", len(missing))
            for year, month, group in missing:
                self._download_month(year, month,
                                     sorted(group["day"].unique()),
                                     sorted(group["hour"].unique()))
