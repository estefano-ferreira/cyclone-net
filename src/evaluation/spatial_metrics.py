# src/evaluation/spatial_metrics.py
"""
Spatial metrics for validating FuelMap against external proxies (e.g., TCHP).
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from scipy.stats import spearmanr


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers between two points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def peak_distance(
    pred_lat: float,
    pred_lon: float,
    true_lat: float,
    true_lon: float
) -> float:
    """Distance between predicted and true peak locations."""
    return haversine_distance(pred_lat, pred_lon, true_lat, true_lon)


def top_k_overlap(
    fuelmap: np.ndarray,
    proxy_map: np.ndarray,
    k: int = 10
) -> float:
    """
    Fraction of top-k fuelmap pixels that are also in top-k proxy pixels.
    Both fuelmap and proxy_map are 2D arrays.
    """
    fuel_flat = fuelmap.flatten()
    proxy_flat = proxy_map.flatten()

    fuel_top = np.argsort(fuel_flat)[-k:]
    proxy_top = np.argsort(proxy_flat)[-k:]

    overlap = len(set(fuel_top).intersection(set(proxy_top)))
    return overlap / k


def rank_correlation(
    fuelmap: np.ndarray,
    proxy_map: np.ndarray
) -> float:
    """
    Spearman rank correlation between fuelmap and proxy map values.
    """
    fuel_flat = fuelmap.flatten()
    proxy_flat = proxy_map.flatten()
    # Remove NaN or invalid values
    valid = np.isfinite(fuel_flat) & np.isfinite(proxy_flat)
    if valid.sum() < 2:
        return float('nan')
    corr, _ = spearmanr(fuel_flat[valid], proxy_flat[valid])
    return float(corr)


def compute_spatial_metrics(
    pred_lat: float,
    pred_lon: float,
    true_lat: float,
    true_lon: float,
    fuelmap: np.ndarray,
    proxy_map: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all spatial metrics for a single event.
    """
    metrics = {
        "peak_distance_km": peak_distance(pred_lat, pred_lon, true_lat, true_lon),
        "top10_overlap": top_k_overlap(fuelmap, proxy_map, k=10),
        "rank_correlation": rank_correlation(fuelmap, proxy_map),
    }
    return metrics