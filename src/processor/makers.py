"""
CycloneNet: Physics-Guided Framework for Targeted RI Detection.
---------------------------------------------------------------
Software Engineer: Estefano Senhor Ferreira
License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)

This work is licensed under CC BY-NC 4.0. 
Commercial use is strictly prohibited without prior authorization.
Copyright (c) 2026 Estefano Senhor Ferreira
"""

import numpy as np
from pathlib import Path
from typing import List, Optional
import logging
logger = logging.getLogger(__name__)


def create_cube_series(event: dict, era5_files: List[Path],
                       cube_shape: tuple = (40, 40, 4)) -> Optional[List[np.ndarray]]:
    """
    Creates a time series of cubes for an IR event.

    CORRECTION: Each cube must have shape (H, W, C) and series (T, H, W, C)

    Args:
        event: Dictionary with event information
        era5_files: List of ERA5 files (sorted chronologically)
        cube_shape: Target shape of the cubes (H, W, C)

    Returns:
        List of numpy cubes (each cube: (H, W, C)) or None in case of error
    """
    logger.info(
        f"Creating a series of cubes for an event in {event['datetime']}")

    if not era5_files:
        logger.error("The ERA5 file list is empty.")
        return None

    cube_series = []

    # Check each file before attempting to extract it.
    valid_files = []
    for era5_file in era5_files:
        if isinstance(era5_file, str):
            era5_file = Path(era5_file)

        if not era5_file.exists():
            logger.error(f"File not found: {era5_file}")
            logger.error(f"  Current directory: {Path.cwd()}")
            logger.error(f"  Directory exists? {era5_file.parent.exists()}")
            continue

        file_size = era5_file.stat().st_size
        if file_size < 1000:
            logger.warning(
                f"Suspicious file (very small): {era5_file} ({file_size} bytes)")
            continue

        valid_files.append(era5_file)
        logger.info(
            f"✓ Valid file: {era5_file.name} ({file_size/1024:.1f} KB)")

    if not valid_files:
        logger.error("No valid files found.")
        return None

    logger.info(f"Processing {len(valid_files)} valid files")

    valid_files.sort(key=lambda x: x.name)

    for i, era5_file in enumerate(valid_files):
        logger.info(
            f"Extracting cube from {era5_file.name} ({i+1}/{len(valid_files)})")

        from .processors import extract_storm_cube

        try:
            cube = extract_storm_cube(
                era5_file=era5_file,
                storm_center_lat=event['lat'],
                storm_center_lon=event['lon'],
                window_size_deg=10
            )
        except Exception as e:
            logger.error(f"Error in the extract_storm_cube function.: {e}")
            cube = None

        if cube is None:
            logger.warning(
                f"It was not possible to extract the cube from {era5_file.name}")

            H, W, C = cube_shape
            logger.info(f"Creating a simulated cube shape ({H}, {W}, {C})")
            # Use the fallback function that already returns (H, W, C)
            cube = create_fallback_cube(event['lat'], event['lon'], 10)

            cube_series.append(cube)
            continue

        # Resize to consistent shape (H, W, C)
        if cube.shape != cube_shape:
            logger.info(
                f"Adjusting shape of {cube.shape} to {cube_shape} via Center Crop")

            # 1. If the cube is larger than the target, cut off the center.
            if cube.shape[0] >= cube_shape[0] and cube.shape[1] >= cube_shape[1]:
                start_h = (cube.shape[0] - cube_shape[0]) // 2
                start_w = (cube.shape[1] - cube_shape[1]) // 2
                cube = cube[start_h:start_h + cube_shape[0],
                            start_w:start_w + cube_shape[1],
                            :cube_shape[2]]

            # 2. If it's smaller (which is rare coming from an 81x81), zoom in only if necessary.
            else:
                try:
                    from scipy import ndimage
                    scale_factors = [cube_shape[0]/cube.shape[0],
                                     cube_shape[1]/cube.shape[1], 1]
                    cube = ndimage.zoom(cube, scale_factors, order=1)
                except ImportError:
                    # Emergency fallback (zeros)
                    new_cube = np.zeros(cube_shape, dtype=cube.dtype)
                    h_min, w_min = min(cube.shape[0], cube_shape[0]), min(
                        cube.shape[1], cube_shape[1])
                    new_cube[:h_min, :w_min, :] = cube[:h_min,
                                                       :w_min, :min(cube.shape[2], cube_shape[2])]
                    cube = new_cube

        cube_series.append(cube)
        logger.info(f"✓ Cube {i+1} extracted: shape {cube.shape}")

    if not cube_series:
        logger.error("No valid cube was created.")
        return None

    logger.info(
        f"Series created with {len(cube_series)} cubes (each: {cube_series[0].shape})")
    return cube_series


def create_fallback_cube(lat: float, lon: float, window_size: float) -> np.ndarray:
    """
    Auxiliary function to create a fallback cube.
    (This function is a wrapper for the function in processors.py)
    """
    from .processors import create_fallback_cube as processors_fallback
    return processors_fallback(lat, lon, window_size)
