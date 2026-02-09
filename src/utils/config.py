"""
CycloneNet: Physics-Guided Framework for Targeted RI Detection.
---------------------------------------------------------------
Software Engineer: Estefano Senhor Ferreira
License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)

This work is licensed under CC BY-NC 4.0. 
Commercial use is strictly prohibited without prior authorization.
Copyright (c) 2026 Estefano Senhor Ferreira
"""

import sys
from pathlib import Path
import logging
import os
from dotenv import load_dotenv

load_dotenv()


def find_project_root():
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "src").exists() and (parent / "data").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"

PATHS = {
    'project_root': PROJECT_ROOT,
    'raw_hurdat': DATA_DIR / 'raw' / 'hurdat2' / 'hurdat2.txt',
    'raw_era5_dir': DATA_DIR / 'raw' / 'era5',
    'interim_data': DATA_DIR / 'interim',
    'processed_data': DATA_DIR / 'processed',
    'processed_cubes': DATA_DIR / 'processed' / 'storm_cubes.npz',
    'output_figures': PROJECT_ROOT / 'outputs' / 'figures',
    'output_predictions': PROJECT_ROOT / 'outputs' / 'predictions',
    'logs': PROJECT_ROOT / 'outputs' / 'logs',
}

PARAMS = {
    'ri_threshold_knots': 30,
    'ri_window_hours': 24,
    'storm_name': 'ANDREW',
    'year': '1992',
    'lead_time_hours': 6,
    'T': 5, 'H': 40, 'W': 40, 'C': 4,
}


def validate_config():
    """Ensures the physical existence of the output folders."""
    for key in ['output_figures', 'output_predictions', 'logs', 'interim_data', 'processed_data']:
        PATHS[key].mkdir(parents=True, exist_ok=True)
    return True


def setup_logging():
    """Configure the logging system for scientific monitoring."""
    log_file = PATHS['logs'] / 'pipeline.log'

    # Create the log folder if it doesn't exist
    PATHS['logs'].mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Anomaly monitoring system initiated.")
