"""
CycloneNet: Physics-Guided Framework for Targeted RI Detection.
---------------------------------------------------------------
Software Engineer: Estefano Senhor Ferreira
License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)

This work is licensed under CC BY-NC 4.0. 
Commercial use is strictly prohibited without prior authorization.
Copyright (c) 2026 Estefano Senhor Ferreira
"""

from src.models.physics_model import predict_numpy, hotspot_numpy
import numpy as np
import logging

logger = logging.getLogger(__name__)


def predict_intensity(x_batch: np.ndarray) -> np.ndarray:
    logger.info("🧠 Using Physics-Guided CycloneNet")
    return predict_numpy(x_batch)


def compute_sensitivity(x_batch):
    """
    Scientific placeholder for intensity gradient.
    It will be replaced by physics_model later.
    """
    import numpy as np

    if x_batch.ndim == 4:
        x_batch = np.expand_dims(x_batch, axis=0)

    B, T, H, W, C = x_batch.shape

    # controlled fictitious sensitivity
    sensitivity = np.ones((B, T, H, W, C)) * 0.01
    return sensitivity


def compute_hotspot(x_batch: np.ndarray) -> np.ndarray:
    logger.info("🔥 Calculating physical hotspot")
    return hotspot_numpy(x_batch)


def get_critical_coordinates(hotspot_map: np.ndarray, event_data: dict) -> list:
    import numpy as np
    from src.utils.config import PARAMS  # Imports centralized lead time

    lead_time = PARAMS.get('lead_time_hours', 0)
    center_lat = event_data['lat']
    center_lon = event_data['lon']
    H, W = hotspot_map.shape

    # 1. CREATE FOCUS MASK (Gaussian Blur)
    x, y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    d = np.sqrt(x*x + y*y)
    sigma, mu = 0.5, 0.0
    gauss = np.exp(-((d-mu)**2 / (2.0 * sigma**2)))

    # 2. APPLY THE MASK
    refined_hotspot = hotspot_map * gauss

    # 3. VECTOR EXTRACTION (OPTIONAL/ADVANCED)
    # Here we could read u10 and v10 to adjust the offset
    # For now, keep the precision extraction based on activation

    flat_indices = np.argsort(refined_hotspot.ravel())[-5:][::-1]

    targets = []
    for idx in flat_indices:
        h_idx, w_idx = np.unravel_index(idx, (H, W))

        # Spatial Offset Calculation (0.25° per pixel in the ERA5 grid)
        # O (20, 20) is the center of our 40x40 window
        lat_offset = (20 - h_idx) * 0.25
        lon_offset = (w_idx - 20) * 0.25

        # If lead_time > 0, the hotspot identified by the model
        # already represents the ‘intention’ of the storm's physics to move.
        targets.append({
            'lat': round(float(center_lat + lat_offset), 2),
            'lon': round(float(center_lon + lon_offset), 2),
            'intensity_weight': round(float(refined_hotspot[h_idx, w_idx]), 4),
            'lead_time': lead_time
        })

    return targets
