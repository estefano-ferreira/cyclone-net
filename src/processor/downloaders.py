"""
CycloneNet: Physics-Guided Framework for Targeted RI Detection.
---------------------------------------------------------------
Software Engineer: Estefano Senhor Ferreira
License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)

This work is licensed under CC BY-NC 4.0. 
Commercial use is strictly prohibited without prior authorization.
Copyright (c) 2026 Estefano Senhor Ferreira
"""

import logging
import requests
from cdsapi import Client
from pathlib import Path
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH)

logger = logging.getLogger(__name__)


def download_hurdat2(output_path: Path, force_download: bool = False) -> None:
    url = os.getenv(
        "HURDAT2_URL", "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2024-040425.txt")
    if output_path.exists() and not force_download:
        logger.info(f"HURDAT2 already exists in {output_path}")
        return

    try:
        logger.info(f"Downloading HURDAT2 from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        logger.info(f"HURDAT2 saved in {output_path}")
    except requests.RequestException as e:
        logger.error(f"Error when downloading HURDAT2: {e}")
        raise


def setup_cds_client():
    url = os.getenv('CDS_URL', 'https://cds.climate.copernicus.eu/api/v2')
    key = os.getenv('CDS_KEY')
    if not url or not key:
        logger.error("Variables CDS_URL/KEY not configured")
        return None
    try:
        client = Client(url=url, key=key)
        logger.info("CDS client configured")
        return client
    except Exception as e:
        logger.error(f"Failed to configure CDS client.: {e}")
        return None


def download_era5_for_event(event: dict, output_dir: Path) -> list:
    """
    Robust ERA5 download for an IR event.
    Guarantees:
    - creates folder if it does not exist
    - uses new API correctly
    - avoids empty files
    - never crashes the pipeline
    """
    client = setup_cds_client()
    if not client:
        logger.error("CDS client not initialized")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    event_time = datetime.strptime(event["datetime"], "%Y%m%d %H%M")
    lat, lon = event["lat"], event["lon"]

    # area: N, W, S, E
    area = [lat + 10, lon - 10, lat - 10, lon + 10]

    variables = [
        "sea_surface_temperature",
        "mean_sea_level_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ]

    output_files = []

    for hour_offset in [0, -6, -12, -18, -24]:
        target_time = event_time + timedelta(hours=hour_offset)

        date_str = target_time.strftime("%Y-%m-%d")
        time_str = target_time.strftime("%H:00")

        output_file = output_dir / \
            f"era5_{date_str}_{time_str.replace(':','')}.nc"

        # already exists and is valid
        if output_file.exists() and output_file.stat().st_size > 10_000:
            logger.info(f"✓ ERA5 already exists: {output_file.name}")
            output_files.append(output_file)
            continue

        try:
            logger.info(f"⬇️ Downloading ERA5: {date_str} {time_str} UTC")

            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable": variables,
                    "year": [str(target_time.year)],
                    "month": [f"{target_time.month:02d}"],
                    "day": [f"{target_time.day:02d}"],
                    "time": [time_str],
                    "area": area,
                    "data_format": "netcdf",
                },
                str(output_file),
            )

            if output_file.exists() and output_file.stat().st_size > 10_000:
                logger.info(f"✓ ERA5 saved: {output_file.name}")
                output_files.append(output_file)
            else:
                logger.error(f"❌ Invalid ERA5 file: {output_file}")

        except Exception as e:
            logger.error(f"❌ Error downloading ERA5: {e}")

            # retry to rate-limit
            if "rate limit" in str(e).lower():
                logger.warning("⏳ Waiting 60s for rate-limit...")
                time.sleep(60)

    return output_files


def test_cds_connection():
    logger.info("Testing connection with CDS...")
    client = setup_cds_client()
    if not client:
        return False
    try:
        test_file = Path("test_connection.nc")
        client.retrieve('reanalysis-era5-single-levels', {
            'variable': ['sea_surface_temperature'],
            'year': '2019',
            'month': '08',
            'day': '31',
            'time': '00:00',
            'area': [35, -81, 15, -61],
            'format': 'netcdf'
        }, str(test_file))
        if test_file.exists():
            logger.info("Test successful!")
            test_file.unlink()
            return True
        else:
            logger.error("Test file not created.")
            return False
    except Exception as e:
        logger.error(f"Test failed.: {e}")
        return False
