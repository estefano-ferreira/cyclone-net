from __future__ import annotations

"""
CycloneNet — Sea Surface Height Anomaly (SLA) downloader.

Why this dataset
----------------
A tropical cyclone's intensification is sustained (or cut off) by the SUBSURFACE
ocean heat reservoir — exactly the quantity the current surface-only model cannot
see. Sea Level Anomaly (SLA) from satellite altimetry is the established SURFACE
signature of that reservoir: warm-core eddies and the Loop Current appear as
positive SLA highs that co-locate with high ocean heat content (TCHP).

SLA is a single 2D field (tiny vs a full 3D ocean reanalysis), so it adds real
physical information at minimal data mass. It serves two purposes:
  - an independent ground-truth for the energy-source location, and
  - a candidate NEW model input that finally gives the network subsurface context.

Source: Copernicus Marine DUACS reprocessed L4 product
        cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D  (variable: sla)
"""

import logging
from pathlib import Path
from typing import List, Optional

from src.utils.config import cfg_get

logger = logging.getLogger(__name__)

try:
    import copernicusmarine  # type: ignore
except Exception:  # pragma: no cover
    copernicusmarine = None

DEFAULT_DATASET_ID = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"


def _month_window(year: int, months: List[int]) -> tuple[str, str]:
    """Start/end datetimes spanning the first to last requested season month."""
    m0, m1 = min(months), max(months)
    start = f"{year}-{m0:02d}-01T00:00:00"
    # last day of m1: step to first of next month minus a second is overkill; use 31 and
    # let the server clamp to available data (DUACS is daily, server-side clamped).
    end = f"{year}-{m1:02d}-28T23:59:59" if m1 == 2 else f"{year}-{m1:02d}-30T23:59:59"
    if m1 in (1, 3, 5, 7, 8, 10, 12):
        end = f"{year}-{m1:02d}-31T23:59:59"
    return start, end


def ssh_file_path(ocean_dir: Path, year: int) -> Path:
    return ocean_dir / f"ssh_sla_{year}.nc"


def download_ssh(cfg: dict, force: bool = False) -> List[Path]:
    """
    Download SLA for the configured years/season/box. Defaults stay deliberately
    minimal (one peak season, storm-prone box) to avoid excessive data mass.
    """
    if copernicusmarine is None:
        logger.error("copernicusmarine is not installed; cannot download SLA.")
        return []

    enabled = bool(cfg_get(cfg, "download.ssh.enabled", True))
    if not enabled:
        logger.info("SSH download disabled (download.ssh.enabled=false).")
        return []

    dataset_id = str(cfg_get(cfg, "download.ssh.dataset_id", DEFAULT_DATASET_ID))
    variables = list(cfg_get(cfg, "download.ssh.variables", ["sla"]))
    years_cfg = list(cfg_get(cfg, "download.ssh.years", [2020, 2020]))
    years = list(range(int(years_cfg[0]), int(years_cfg[-1]) + 1))
    months = list(cfg_get(cfg, "download.ssh.season_months", [8, 9, 10]))

    lat_min = float(cfg_get(cfg, "download.ssh.box.lat_min", 5.0))
    lat_max = float(cfg_get(cfg, "download.ssh.box.lat_max", 40.0))
    lon_min = float(cfg_get(cfg, "download.ssh.box.lon_min", -100.0))
    lon_max = float(cfg_get(cfg, "download.ssh.box.lon_max", -15.0))

    # Reuse the Copernicus Marine credentials already configured for TCHP.
    username = cfg_get(cfg, "download.tchp.copernicus.username", None)
    password = cfg_get(cfg, "download.tchp.copernicus.password", None)

    ocean_dir = Path(cfg_get(cfg, "paths.ocean_dir", "./data/external/ocean")).resolve()
    ocean_dir.mkdir(parents=True, exist_ok=True)

    creds = {}
    if username and password:
        creds = {"username": str(username), "password": str(password)}

    out_files: List[Path] = []
    for year in years:
        out_path = ssh_file_path(ocean_dir, year)
        if out_path.exists() and not force:
            logger.info("SLA %d already present: %s", year, out_path)
            out_files.append(out_path)
            continue

        start, end = _month_window(year, months)
        logger.info("Downloading SLA %d [%s..%s] box lat[%.0f,%.0f] lon[%.0f,%.0f]",
                    year, start, end, lat_min, lat_max, lon_min, lon_max)
        try:
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=variables,
                minimum_longitude=lon_min, maximum_longitude=lon_max,
                minimum_latitude=lat_min, maximum_latitude=lat_max,
                start_datetime=start, end_datetime=end,
                output_filename=out_path.name,
                output_directory=str(ocean_dir),
                overwrite=True,
                disable_progress_bar=True,
                **creds,
            )
            if out_path.exists():
                size_mb = out_path.stat().st_size / 1e6
                logger.info("Saved SLA %d -> %s (%.1f MB)", year, out_path, size_mb)
                out_files.append(out_path)
            else:
                logger.error("SLA download finished but file missing: %s", out_path)
        except Exception as exc:
            logger.error("SLA download failed for %d: %s", year, exc)

    return out_files


def run_download_ssh(cfg: dict, force: bool = False) -> None:
    """Entrypoint for run.py."""
    download_ssh(cfg, force=force)
