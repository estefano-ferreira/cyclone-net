# src/downloaders/tchp.py
from __future__ import annotations

import logging
import urllib.request
import ftplib
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import cfg_get
from src.utils.tchp_utils import get_tchp_file_path

logger = logging.getLogger(__name__)


class TCHPDownloader:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.tchp_dir = Path(
            cfg_get(cfg, "paths.tchp_dir", "./data/external/tchp")).resolve()
        self.tchp_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = bool(cfg_get(cfg, "download.tchp.enabled", False))
        self.source = str(cfg_get(cfg, "download.tchp.source", "auto"))
        self.years = self._resolve_years()
        self.max_workers = int(cfg_get(cfg, "download.max_workers", 4))

    def _resolve_years(self) -> List[int]:
        """Get list of years from config or from event list."""
        years_cfg = cfg_get(self.cfg, "download.tchp.years", None)
        if years_cfg is not None and len(years_cfg) == 2:
            start, end = years_cfg
            return list(range(start, end + 1))
        # Fallback: extract years from event list
        event_list_path = Path(
            cfg_get(self.cfg, "paths.event_list", "./data/event_list_augmented.csv"))
        if event_list_path.exists():
            df = pd.read_csv(event_list_path)
            if 'timestamp' in df.columns:
                years = pd.to_datetime(df['timestamp']).dt.year.unique()
            elif 'datetime' in df.columns:
                years = pd.to_datetime(
                    df['datetime'], format='%Y%m%d %H%M').dt.year.unique()
            else:
                raise ValueError("Cannot determine years from event list")
            return sorted(years)
        raise ValueError("No years specified and event list not found")

    def download_year(self, year: int) -> Optional[Path]:
        """Download TCHP for a specific year using the appropriate source."""
        # Determine source
        if self.source == 'auto':
            if year >= 2022:
                src = 'noaa'
            elif year >= 1993:
                src = 'aoml'
            else:
                src = 'copernicus'
        else:
            src = self.source

        out_path = get_tchp_file_path(self.tchp_dir, year, src)
        if out_path.exists():
            logger.info(f"File {out_path.name} already exists. Skipping.")
            return out_path

        if src == 'noaa':
            return self._download_noaa(year, out_path)
        elif src == 'aoml':
            return self._download_aoml(year, out_path)
        elif src == 'copernicus':
            return self._download_copernicus(year, out_path)
        else:
            raise ValueError(f"Unknown source: {src}")

    def _download_noaa(self, year: int, out_path: Path) -> Optional[Path]:
        """Download from NOAA ERDDAP."""
        url = f"https://cwcgom.aoml.noaa.gov/erddap/griddap/aomlTCHP.nc?Tropical_Cyclone_Heat_Potential[({year}-01-01T12:00:00Z):1:({year}-12-31T12:00:00Z)][(0.0):1:(60.0)][(-100.0):1:(-10.0)],D26[({year}-01-01T12:00:00Z):1:({year}-12-31T12:00:00Z)][(0.0):1:(60.0)][(-100.0):1:(-10.0)]"
        logger.info(f"Downloading NOAA TCHP for {year} from {url}")
        try:
            urllib.request.urlretrieve(url, out_path)
            logger.info(f"Saved to {out_path}")
            return out_path
        except Exception as e:
            logger.error(f"Failed to download NOAA TCHP for {year}: {e}")
            return None

    def _download_aoml(self, year: int, out_path: Path) -> Optional[Path]:
        """Download from AOML FTP historical archive."""
        ftp_host = 'ftp.aoml.noaa.gov'
        remote_path = f'/pub/phod/pub/tcp data/TCHP_historical/tchp_{year}.nc'
        logger.info(
            f"Downloading AOML TCHP for {year} from ftp://{ftp_host}{remote_path}")
        try:
            ftp = ftplib.FTP(ftp_host)
            ftp.login()
            with open(out_path, 'wb') as f:
                ftp.retrbinary(f'RETR {remote_path}', f.write)
            ftp.quit()
            logger.info(f"Saved to {out_path}")
            return out_path
        except Exception as e:
            logger.error(f"Failed to download AOML TCHP for {year}: {e}")
            return None

    def _download_copernicus(self, year: int, out_path: Path) -> Optional[Path]:
        """Download from Copernicus Marine Service (requires registration)."""
        # Requer instalação do copernicusmarine e credenciais
        logger.warning(
            "Copernicus download requires registration and manual setup.")
        logger.info(
            "Please follow instructions at https://data.marine.copernicus.eu/register")
        # Exemplo de implementação (se credenciais estiverem configuradas)
        try:
            import copernicusmarine
            # Obter credenciais da config (recomenda-se usar variáveis de ambiente)
            username = cfg_get(
                self.cfg, "download.tchp.copernicus.username", None)
            password = cfg_get(
                self.cfg, "download.tchp.copernicus.password", None)
            if username and password:
                copernicusmarine.login(username, password)
            # Subset para o ano
            copernicusmarine.subset(
                dataset_id=cfg_get(
                    self.cfg, "download.tchp.copernicus.dataset_id", "cmems_mod_glo_phy_my_0.25deg_P1D-m"),
                variables=["thetao"],
                minimum_longitude=-100,
                maximum_longitude=-10,
                minimum_latitude=0,
                maximum_latitude=60,
                start_datetime=f"{year}-01-01T00:00:00",
                end_datetime=f"{year}-12-31T23:59:59",
                output_filename=str(out_path),
            )
            logger.info(f"Saved to {out_path}")
            return out_path
        except ImportError:
            logger.error(
                "copernicusmarine package not installed. Run: pip install copernicusmarine")
            return None
        except Exception as e:
            logger.error(f"Failed to download Copernicus TCHP for {year}: {e}")
            return None

    def download_all(self) -> List[Path]:
        """Download all required years in parallel."""
        if not self.enabled:
            logger.info(
                "TCHP download is disabled. Set download.tchp.enabled=true in config.")
            return []
        downloaded = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(
                self.download_year, year): year for year in self.years}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading TCHP years"):
                year = futures[future]
                try:
                    result = future.result()
                    if result:
                        downloaded.append(result)
                except Exception as e:
                    logger.error(f"Error downloading year {year}: {e}")
        return downloaded


def download_tchp(cfg: dict) -> None:
    """Entrypoint for run.py."""
    downloader = TCHPDownloader(cfg)
    downloader.download_all()
