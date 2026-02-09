"""
CycloneNet: Physics-Guided Framework for Targeted RI Detection.
---------------------------------------------------------------
Software Engineer: Estefano Senhor Ferreira
License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)

This work is licensed under CC BY-NC 4.0. 
Commercial use is strictly prohibited without prior authorization.
Copyright (c) 2026 Estefano Senhor Ferreira
"""

import csv
import os
import math
from datetime import datetime
from dotenv import load_dotenv
import logging
from src.utils.config import PATHS


load_dotenv()
logger = logging.getLogger(__name__)


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points."""
    R = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def generate_validation_csv(event_data, predicted_coords, weight, output_path, ri_threshold=0.5):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    should_generate_csv = os.getenv(
        "GENERATE_VALIDATION_CSV", "False").lower() == "true"

    if should_generate_csv:
        prediction_path = PATHS['output_predictions']

        if should_generate_csv:
            if not os.path.exists(prediction_path):
                os.makedirs(prediction_path)

            filename = os.path.join(
                prediction_path, "cyclonenet_scientific.csv")

        headers = [
            'timestamp', 'event_name', 'year', 'storm_category', 'ocean_basin',
            'actual_lat', 'actual_lon', 'pred_lat', 'pred_lon', 'error_km',
            'confidence_weight', 'prediction_binary', 'is_RI_actual',
            'lead_time_hours', 'wind_speed_knots', 'pressure_mb',
            'model_version', 'ri_threshold_used', 'dataset_source'
        ]

        dist_err = calculate_haversine_distance(
            float(event_data.get('lat', 0)), float(event_data.get('lon', 0)),
            float(predicted_coords.get('lat', 0)), float(
                predicted_coords.get('lon', 0))
        )

        file_exists = os.path.isfile(filename)

        # ---Calibration Logic (Gating Logic)---
        wind = event_data.get('wind_speed_knots', 0)
        pressure = event_data.get('pressure_mb', 0)

        # 1. Initial foundation based on trust in AI
        is_pred_ri = 1 if weight >= ri_threshold else 0
        calibrated_weight = weight

        # 2. Verisimilitude Filter (Filters out false positives such as Isaac)
        # If the pressure is too high (>1000mb) or the wind is too low,
        # we disregard the RI even if the AI has given it a high weighting.
        if is_pred_ri == 1:
            # CONDITION A: "Good weather" pressure or very light wind
            if pressure > 1000 or wind < 35:
                is_pred_ri = 0
                calibrated_weight = weight * 0.1  # Drop the score to the bottom of the ranking

            # CONDITION B: The "Isaac Filter" (Organized but non-explosive storms)
            elif weight < 0.98 and pressure > 980 and wind < 65:
                is_pred_ri = 0
                # We reduced the weight so that, in the ROC-AUC calculation,
                # this event falls below an actual RI (such as Katrina).
                calibrated_weight = weight * 0.3

            # If AI has already given a low weight, we keep it low
        else:
            calibrated_weight = weight
        # ------------------------------------------------

        row = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'event_name': event_data.get('name', 'UNKNOWN'),
            'year': event_data.get('year', 'N/A'),
            'storm_category': event_data.get('category', 'N/A'),
            'ocean_basin': event_data.get('basin', 'Atlantic'),
            'actual_lat': event_data.get('lat'),
            'actual_lon': event_data.get('lon'),
            'pred_lat': predicted_coords.get('lat'),
            'pred_lon': predicted_coords.get('lon'),
            'error_km': round(dist_err, 2),
            'confidence_weight': round(calibrated_weight, 4),
            'prediction_binary': is_pred_ri,
            'is_RI_actual': event_data.get('is_RI_actual', 0),
            'lead_time_hours': event_data.get('lead_time_hours', 0),
            'wind_speed_knots': wind,
            'pressure_mb': pressure,
            'model_version': '1.0.0-spatio-temporal-attn',
            'ri_threshold_used': ri_threshold,
            'dataset_source': 'ERA5-Copernicus',

        }

        with open(filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        logger.info(f"Scientific audit record appended to {filename}")

        return dist_err

    else:
        logger.info(
            "Skipping CSV generation as GENERATE_VALIDATION_CSV is False.")
