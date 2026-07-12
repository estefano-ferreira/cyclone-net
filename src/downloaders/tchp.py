# src/downloaders/tchp.py
from __future__ import annotations

import ftplib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from socket import timeout as socket_timeout
from typing import List, Optional

import pandas as pd
import xarray as xr
from tqdm import tqdm

from src.utils.config import cfg_get
from src.utils.tchp_utils import get_tchp_file_path

logger = logging.getLogger(__name__)

# Optional dependency (do not hard-fail import at module import time)
try:
    import copernicusmarine  # type: ignore
except Exception:  # pragma: no cover
    copernicusmarine = None


def _ensure_dir(path: Path) -> None:
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def tchp_time_steps(path: Path) -> int:
    """
    Number of time steps in a TCHP NetCDF, or -1 if unreadable.

    A valid yearly TCHP file must contain time steps; empty stubs (a previous bug
    wrote 0-length files when the ERDDAP slice fell outside the dataset's coverage)
    return 0 and must be treated as invalid, not as a successful download.
    """
    try:
        import xarray as xr
        for eng in ("scipy", "h5netcdf", "netcdf4"):
            try:
                with xr.open_dataset(path, engine=eng) as ds:
                    return int(ds.sizes.get("time", 0))
            except Exception:
                continue
        return -1
    except Exception:
        return -1


def _safe_replace(src_tmp: Path, dst: Path) -> None:
    """
    Replace destination file atomically (best-effort on Windows).

    Writing to a temp file first avoids partial/corrupted outputs on failures.
    """
    try:
        if dst.exists():
            dst.unlink()
        src_tmp.replace(dst)
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied when replacing output file: {dst}. "
            "The file may be locked by another process (Explorer preview, antivirus, etc.) "
            "or the directory may be read-only."
        ) from e


def _write_dataarray_netcdf_safe(da: xr.DataArray, out_path: Path) -> None:
    """
    Write a DataArray to NetCDF in a Windows-safe way:
    - ensure directory exists
    - write to a temporary file
    - replace the final target
    - prefer a non-HDF5 engine when available
    """
    out_path = out_path.resolve()
    _ensure_dir(out_path.parent)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    # Prefer scipy engine to avoid HDF5-related path/lock/unicode issues on Windows.
    # If scipy is not installed, fall back to default behavior.
    wrote = False
    last_err: Optional[Exception] = None

    for engine in ("scipy", None):
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            if engine is None:
                da.to_netcdf(tmp_path)
            else:
                da.to_netcdf(tmp_path, engine=engine)
            wrote = True
            break
        except Exception as e:
            last_err = e
            # Try next engine option
            continue

    if not wrote:
        raise RuntimeError(f"Failed to write NetCDF for {out_path}. Last error: {last_err}")

    _safe_replace(tmp_path, out_path)


class TCHPDownloader:
    """
    Downloader for Tropical Cyclone Heat Potential (TCHP) data.

    Supported sources:
      - NOAA ERDDAP (AOML ERDDAP dataset aomlTCHP) for gridded TCHP
      - AOML FTP historical archive (legacy; may change over time)
      - Copernicus Marine Service (optional fallback)

    Notes:
      - For reproducibility, the output is stored year-by-year as NetCDF.
      - This dataset is intended for *external validation*, not as a model input feature.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.tchp_dir = Path(cfg_get(cfg, "paths.tchp_dir", "./data/external/tchp")).resolve()
        _ensure_dir(self.tchp_dir)

        self.enabled = bool(cfg_get(cfg, "download.tchp.enabled", False))
        self.source = str(cfg_get(cfg, "download.tchp.source", "auto")).lower().strip()

        self.years = self._resolve_years()
        self.max_workers = int(cfg_get(cfg, "download.max_workers", 4))

        # ERDDAP endpoint – can be overridden in config, otherwise use default
        self.erddap_url = str(
            cfg_get(
                cfg,
                "download.tchp.erddap_url",
                "https://cwcgom.aoml.noaa.gov/erddap/griddap/aomlTCHP"
            )
        )

        # Spatial bounds (Atlantic basin by default; configurable)
        self.lat_min = float(cfg_get(cfg, "download.tchp.lat_min", 0.0))
        self.lat_max = float(cfg_get(cfg, "download.tchp.lat_max", 60.0))
        self.lon_min = float(cfg_get(cfg, "download.tchp.lon_min", -100.0))
        self.lon_max = float(cfg_get(cfg, "download.tchp.lon_max", -10.0))

        # Common variable names for TCHP (order of preference)
        self.var_candidates = [
            "Tropical_Cyclone_Heat_Potential",
            "tchp",
            "TCHP",
            "tchp_ssh",
        ]

    def _resolve_years(self) -> List[int]:
        """Get list of years from config or from event list."""
        years_cfg = cfg_get(self.cfg, "download.tchp.years", None)
        if years_cfg is not None and len(years_cfg) == 2:
            start, end = int(years_cfg[0]), int(years_cfg[1])
            return list(range(start, end + 1))

        # Fallback: extract years from event list
        event_list_path = Path(cfg_get(self.cfg, "paths.event_list", "./data/event_list_augmented.csv"))
        if event_list_path.exists():
            df = pd.read_csv(event_list_path)
            if "timestamp" in df.columns:
                years = pd.to_datetime(df["timestamp"]).dt.year.unique()
            elif "datetime" in df.columns:
                years = pd.to_datetime(df["datetime"], format="%Y%m%d %H%M").dt.year.unique()
            else:
                raise ValueError("Cannot determine years from event list: missing 'timestamp'/'datetime' column.")

            # Keep only years where TCHP is expected to exist
            years = [int(y) for y in years if int(y) >= 1993]
            return sorted(years)

        raise ValueError("No years specified and event list not found.")

    def download_year(self, year: int, force: bool = False) -> Optional[Path]:
        """
        Download TCHP data for a specific year.

        Source selection:
          - If source == 'auto': prefer ERDDAP for all years >= 1993.
            If ERDDAP fails for 1993–2021, fall back to FTP.
            If still fails, try Copernicus (if available).
          - If source is explicitly set: use that source only.
        """
        if year < 1993:
            logger.warning(f"Year {year} < 1993, no TCHP data available.")
            return None

        if self.source not in {"auto", "noaa", "aoml", "copernicus"}:
            raise ValueError(f"Unknown source: {self.source}")

        # Determine output name based on the *selected* source label
        if self.source == "auto":
            preferred = "noaa"
        else:
            preferred = self.source

        out_path = get_tchp_file_path(self.tchp_dir, year, preferred)
        if out_path.exists():
            steps = tchp_time_steps(out_path)
            if steps > 0 and not force:
                logger.info(f"File {out_path.name} already exists ({steps} time steps). Skipping.")
                return out_path
            if steps <= 0:
                # Remove a previously-written empty/invalid stub so it cannot masquerade
                # as a successful download and can be cleanly re-attempted.
                logger.warning(f"Removing invalid TCHP file {out_path.name} (time steps={steps}).")
                try:
                    out_path.unlink()
                except OSError:
                    pass

        if preferred == "noaa":
            result = self._download_erddap(year, out_path)
            if result is not None:
                return result

            # Auto fallback chain (only when source=='auto')
            if self.source == "auto" and 1993 <= year <= 2021:
                ftp_out = get_tchp_file_path(self.tchp_dir, year, "aoml")
                result = self._download_aoml_ftp(year, ftp_out)
                if result is not None:
                    return result

            if self.source == "auto":
                cop_out = get_tchp_file_path(self.tchp_dir, year, "copernicus")
                result = self._download_copernicus(year, cop_out)
                return result

            return None

        if preferred == "aoml":
            return self._download_aoml_ftp(year, out_path)

        if preferred == "copernicus":
            return self._download_copernicus(year, out_path)

        return None

    def _download_erddap(self, year: int, out_path: Path) -> Optional[Path]:
        """
        Download TCHP for a given year from NOAA ERDDAP using xarray/OPeNDAP.

        Key robustness decisions:
          - Use tz-naive timestamps (no 'Z') to avoid pandas timezone index errors
          - Try multiple variable names
          - Avoid HDF5-based writing when possible (engine='scipy')
          - Write via tmp -> replace to avoid partial file issues on Windows
        """
        logger.info(f"Downloading TCHP for {year} from NOAA ERDDAP ({self.erddap_url})")

        try:
            # tz-naive range (avoid 'Z' suffix to prevent timezone-aware indexing errors)
            start = pd.Timestamp(year=year, month=1, day=1)
            end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59)

            # Open remote dataset
            ds = xr.open_dataset(self.erddap_url)

            # ERDDAP aomlTCHP commonly uses lon in [-180, 180] degrees_east.
            # Your config is already in that convention (e.g., -100..-10 for Atlantic).
            subset = ds.sel(
                time=slice(start, end),
                latitude=slice(self.lat_min, self.lat_max),
                longitude=slice(self.lon_min, self.lon_max),
            )

            # Critical: the ERDDAP aomlTCHP dataset only covers 2022-present. For earlier
            # years the time slice is empty. Never write a 0-length file and report success
            # (the previous behaviour left misleading 4 KB stubs); return None so the caller
            # can try a fallback or clearly mark the year unavailable.
            if int(subset.sizes.get("time", 0)) == 0:
                logger.warning(
                    f"ERDDAP has no TCHP data for {year} "
                    f"(dataset coverage starts 2022). No file written."
                )
                ds.close()
                return None

            # Find the first available variable name
            var_name = None
            for candidate in self.var_candidates:
                if candidate in subset.data_vars:
                    var_name = candidate
                    break
            if var_name is None:
                raise KeyError(
                    f"No TCHP variable found in dataset. Tried: {self.var_candidates}. "
                    f"Available: {list(subset.data_vars)}"
                )

            da = subset[var_name].load()
            logger.info(f"Using variable '{var_name}' from ERDDAP")

            # Save safely
            _write_dataarray_netcdf_safe(da, out_path)

            logger.info(f"Saved TCHP {year} to {out_path}")
            return out_path

        except PermissionError as e:
            logger.error(
                f"Failed to write TCHP output for {year}: {e}. "
                "If this persists, set paths.tchp_dir to an ASCII-only path (e.g., C:\\data\\cyclonenet\\tchp) "
                "or close any process locking the target folder/file."
            )
            return None
        except Exception as e:
            logger.error(f"Failed to download TCHP for {year} via ERDDAP: {e}")
            return None

    def _download_aoml_ftp(self, year: int, out_path: Path) -> Optional[Path]:
        """
        Download TCHP from AOML FTP historical archive with retries and progress bar.

        WARNING:
          FTP paths and archives can change over time. If you consistently get "550 No such file",
          prefer ERDDAP or update the remote_path to the current archive structure.
        """
        ftp_host = "ftp.aoml.noaa.gov"
        remote_path = f"/pub/phod/pub/tcp data/TCHP_historical/tchp_{year}.nc"

        max_retries = 3
        timeout = 300  # seconds
        _ensure_dir(out_path.parent)

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    f"Downloading AOML TCHP for {year} (attempt {attempt}/{max_retries}) "
                    f"from ftp://{ftp_host}{remote_path}"
                )

                ftp = ftplib.FTP(ftp_host, timeout=timeout)
                # Authentication policy: credentials NEVER live in code or in
                # the project config. If ~/.netrc has an entry for this host,
                # use it; otherwise fall back to anonymous login (the AOML
                # archive is public open data).
                netrc_auth = None
                try:
                    import netrc as _netrc

                    netrc_auth = _netrc.netrc().authenticators(ftp_host)
                except (FileNotFoundError, _netrc.NetrcParseError):
                    pass
                if netrc_auth:
                    login_name, _, login_password = netrc_auth
                    ftp.login(login_name or "anonymous", login_password or "")
                else:
                    ftp.login()  # anonymous (public archive)
                ftp.voidcmd("TYPE I")

                size = ftp.size(remote_path)

                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                if tmp_path.exists():
                    tmp_path.unlink()

                with open(tmp_path, "wb") as f, tqdm(
                    desc=f"TCHP {year}",
                    total=size if size is not None else 0,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:

                    def callback(chunk: bytes) -> None:
                        f.write(chunk)
                        pbar.update(len(chunk))

                    ftp.retrbinary(f"RETR {remote_path}", callback)

                ftp.quit()
                _safe_replace(tmp_path, out_path)

                logger.info(f"Saved TCHP {year} to {out_path}")
                return out_path

            except (socket_timeout, ftplib.error_temp, Exception) as e:
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    sleep_time = 5 * (2 ** (attempt - 1))
                    logger.info(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(
                        f"Failed to download TCHP for {year} after {max_retries} attempts."
                    )
                    return None

        return None

    def _download_copernicus(self, year: int, out_path: Path) -> Optional[Path]:
        """
        Download TCHP from Copernicus Marine Service using its Python client.

        Requirements:
          - copernicusmarine installed and authenticated via `copernicusmarine login`
            or COPERNICUSMARINE_SERVICE_* environment variables (credentials are
            NEVER read from the project config)
        """
        if copernicusmarine is None:
            logger.error(
                "Copernicus download requested but 'copernicusmarine' is not available. "
                "Install it or disable this source."
            )
            return None

        logger.info(f"Downloading TCHP for {year} from Copernicus Marine Service")
        try:
            _ensure_dir(out_path.parent)

            dataset_id = str(cfg_get(self.cfg, "download.tchp.copernicus.dataset_id", ""))

            if not dataset_id:
                raise ValueError("Missing download.tchp.copernicus.dataset_id in config.")

            # Credentials come EXCLUSIVELY from copernicusmarine's own store
            # (`copernicusmarine login`) or COPERNICUSMARINE_SERVICE_* env vars —
            # never from the project config, which is serialized into run
            # snapshots and was the source of a credential leak.
            kwargs = {}

            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=["tchp"],  # may vary; adjust if needed
                minimum_longitude=self.lon_min,
                maximum_longitude=self.lon_max,
                minimum_latitude=self.lat_min,
                maximum_latitude=self.lat_max,
                start_datetime=f"{year}-01-01",
                end_datetime=f"{year}-12-31",
                output_filename=out_path.name,
                output_directory=str(out_path.parent),
                overwrite=True,  # copernicusmarine v2 removed 'force_download'
                disable_progress_bar=True,
                **kwargs,
            )

            if out_path.exists():
                logger.info(f"Saved TCHP {year} to {out_path}")
                return out_path

            logger.error(f"Copernicus download finished but file not found: {out_path}")
            return None

        except Exception as e:
            logger.error(f"Copernicus download failed for {year}: {e}")
            return None

    def download_all(self, force: bool = False) -> List[Path]:
        """Download all required years in parallel."""
        if not self.enabled:
            logger.info("TCHP download is disabled. Set download.tchp.enabled=true in config.")
            return []

        downloaded: List[Path] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.download_year, year, force): year for year in self.years}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading TCHP years"):
                try:
                    result = future.result()
                    if result is not None:
                        downloaded.append(result)
                except Exception as e:
                    year = futures.get(future, "unknown")
                    logger.error(f"Error downloading year {year}: {e}")

        return downloaded


def download_tchp(cfg: dict, force: bool = False) -> None:
    """Entrypoint for run.py."""
    downloader = TCHPDownloader(cfg)
    downloader.download_all(force=force)