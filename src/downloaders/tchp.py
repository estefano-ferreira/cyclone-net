# src/downloaders/tchp.py
from __future__ import annotations

import logging
import ftplib
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import xarray as xr
from tqdm import tqdm

from src.utils.config import cfg_get
from src.utils.tchp_utils import get_tchp_file_path

logger = logging.getLogger(__name__)


class TCHPDownloader:
    """
    Downloader for Tropical Cyclone Heat Potential (TCHP) data.
    Uses NOAA/AOML ERDDAP server for recent years (>=2022) and falls back to FTP for historical data.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.tchp_dir = Path(
            cfg_get(cfg, "paths.tchp_dir", "./data/external/tchp")
        ).resolve()
        self.tchp_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = bool(cfg_get(cfg, "download.tchp.enabled", False))
        self.source = str(cfg_get(cfg, "download.tchp.source", "auto"))
        self.years = self._resolve_years()
        self.max_workers = int(cfg_get(cfg, "download.max_workers", 4))

        # ERDDAP base URL for the aomlTCHP dataset
        self.erddap_url = "https://cwcgom.aoml.noaa.gov/erddap/griddap/aomlTCHP"
        # Spatial bounds for the Atlantic basin (adjustable via config)
        self.lat_min = float(cfg_get(cfg, "download.tchp.lat_min", 0.0))
        self.lat_max = float(cfg_get(cfg, "download.tchp.lat_max", 60.0))
        self.lon_min = float(cfg_get(cfg, "download.tchp.lon_min", -100.0))
        self.lon_max = float(cfg_get(cfg, "download.tchp.lon_max", -10.0))

    def _resolve_years(self) -> List[int]:
        """Get list of years from config or from event list."""
        years_cfg = cfg_get(self.cfg, "download.tchp.years", None)
        if years_cfg is not None and len(years_cfg) == 2:
            start, end = years_cfg
            return list(range(start, end + 1))
        # Fallback: extract years from event list
        event_list_path = Path(
            cfg_get(self.cfg, "paths.event_list", "./data/event_list_augmented.csv")
        )
        if event_list_path.exists():
            df = pd.read_csv(event_list_path)
            if "timestamp" in df.columns:
                years = pd.to_datetime(df["timestamp"]).dt.year.unique()
            elif "datetime" in df.columns:
                years = pd.to_datetime(
                    df["datetime"], format="%Y%m%d %H%M"
                ).dt.year.unique()
            else:
                raise ValueError("Cannot determine years from event list")
            # Keep only years >= 1993 (AOML coverage) to avoid Copernicus dependency
            years = [y for y in years if y >= 1993]
            return sorted(years)
        raise ValueError("No years specified and event list not found")

    def download_year(self, year: int, force: bool = False) -> Optional[Path]:
        """
        Download TCHP data for a specific year.
        Uses ERDDAP for years >= 2022, otherwise falls back to AOML FTP.
        """
        if self.source == "auto":
            if year >= 2022:
                src = "noaa"  # ERDDAP
            elif year >= 1993:
                src = "aoml"  # FTP
            else:
                logger.warning(f"Year {year} < 1993, no TCHP data available")
                return None
        else:
            src = self.source

        out_path = get_tchp_file_path(self.tchp_dir, year, src)
        if out_path.exists() and not force:
            logger.info(f"File {out_path.name} already exists. Skipping.")
            return out_path

        if src == "noaa":
            return self._download_erddap(year, out_path)
        elif src == "aoml":
            return self._download_aoml_ftp(year, out_path)
        else:
            raise ValueError(f"Unknown source: {src}")

    def _download_erddap(self, year: int, out_path: Path) -> Optional[Path]:
        """
        Download TCHP for a given year from NOAA ERDDAP using xarray.
        The data is saved as a NetCDF file.
        """
        logger.info(f"Downloading TCHP for {year} from NOAA ERDDAP")
        try:
            # Build time range
            start = f"{year}-01-01T00:00:00Z"
            end = f"{year}-12-31T23:59:59Z"

            # Open dataset via OpenDAP (lazy loading)
            ds = xr.open_dataset(self.erddap_url, engine="netcdf4")

            # Select time slice and spatial subset
            # Note: ERDDAP longitude is 0-360. Convert our bounds if needed.
            lon_min_erddap = self.lon_min if self.lon_min >= 0 else self.lon_min + 360
            lon_max_erddap = self.lon_max if self.lon_max >= 0 else self.lon_max + 360

            subset = ds.sel(
                time=slice(start, end),
                latitude=slice(self.lat_min, self.lat_max),
                longitude=slice(lon_min_erddap, lon_max_erddap),
            )

            # Load the data into memory (this triggers the actual download)
            tchp = subset["Tropical_Cyclone_Heat_Potential"].load()

            # Save to NetCDF
            tchp.to_netcdf(out_path)
            logger.info(f"Saved TCHP {year} to {out_path}")
            return out_path
        except Exception as e:
            logger.error(f"Failed to download TCHP for {year} via ERDDAP: {e}")
            return None

    def _download_aoml_ftp(self, year: int, out_path: Path) -> Optional[Path]:
        """Download from AOML FTP historical archive with progress bar."""
        ftp_host = "ftp.aoml.noaa.gov"
        remote_path = f"/pub/phod/pub/tcp data/TCHP_historical/tchp_{year}.nc"
        logger.info(
            f"Downloading AOML TCHP for {year} from ftp://{ftp_host}{remote_path}"
        )
        try:
            ftp = ftplib.FTP(ftp_host)
            ftp.login()
            ftp.voidcmd("TYPE I")
            size = ftp.size(remote_path)
            with open(out_path, "wb") as f, tqdm(
                desc=f"TCHP {year}",
                total=size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                def callback(chunk):
                    f.write(chunk)
                    pbar.update(len(chunk))
                ftp.retrbinary(f"RETR {remote_path}", callback)
            ftp.quit()
            logger.info(f"Saved to {out_path}")
            return out_path
        except Exception as e:
            logger.error(f"Failed to download AOML TCHP for {year}: {e}")
            return None

    def download_all(self, force: bool = False) -> List[Path]:
        """Download all required years in parallel."""
        if not self.enabled:
            logger.info(
                "TCHP download is disabled. Set download.tchp.enabled=true in config."
            )
            return []
        downloaded = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.download_year, year, force): year
                for year in self.years
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Downloading TCHP years",
            ):
                try:
                    result = future.result()
                    if result:
                        downloaded.append(result)
                except Exception as e:
                    logger.error(f"Error downloading year: {e}")
        return downloaded


def download_tchp(cfg: dict, force: bool = False) -> None:
    """Entrypoint for run.py."""
    downloader = TCHPDownloader(cfg)
    downloader.download_all(force=force)