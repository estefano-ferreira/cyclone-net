"""
CycloneNet: Physics-Guided Framework for Targeted RI Detection.
---------------------------------------------------------------
Software Engineer: Estefano Senhor Ferreira
License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)

This work is licensed under CC BY-NC 4.0. 
Commercial use is strictly prohibited without prior authorization.
Copyright (c) 2026 Estefano Senhor Ferreira
"""

import os
from src.utils.config import PATHS, PARAMS, validate_config, setup_logging
from src.processor.downloaders import download_hurdat2, download_era5_for_event
from src.processor.processors import parse_hurdat2, find_ri_events
from src.processor.makers import create_cube_series
from src.models.core import predict_intensity, compute_hotspot
from src.processor.metrics_handler import generate_validation_csv
from src.visualization.plotters import plot_hotspot_map, plot_storm_track
import logging
import numpy as np
import sys
from pathlib import Path
import pandas as pd
from src.models.core import get_critical_coordinates
from datetime import timedelta

# Forces Python to see the src folder
sys.path.append(str(Path(__file__).resolve().parent))


# Initialize logs and folders
setup_logging()
validate_config()
logger = logging.getLogger(__name__)


def main() -> int:
    import pandas as pd
    setup_logging()
    validate_config()

    logger.info(
        f"CURRENT EXPERIMENT: Lead Time = {PARAMS['lead_time_hours']}h | Storm = {PARAMS['storm_name']}")
    logger.info("Initiating CycloneNet Scientific Framework")

    # 1. Data Acquisition
    download_hurdat2(PATHS["raw_hurdat"])
    hurdat_df = parse_hurdat2(PATHS["raw_hurdat"])
    ri_events = find_ri_events(hurdat_df, PARAMS["storm_name"], PARAMS["year"])

    if not ri_events:
        logger.warning(
            f"No RI events for {PARAMS['storm_name']}. Falling back to standard tracking...")

        # Filter by storm name (regardless of the year column for now)
        temp_df = hurdat_df[hurdat_df['name'] == PARAMS["storm_name"]]

        # Converts to a dictionary so we can filter the year manually if needed.
        all_records = temp_df.to_dict('records')

        # Filter the records that match the PARAMS year.
        ri_events = [e for e in all_records if str(
            e.get('datetime'))[:4] == str(PARAMS["year"])]

        if not ri_events:
            logger.error(
                f"Storm {PARAMS['storm_name']} {PARAMS['year']} not found in database.")
            return 1

        # Set is_RI_actual to 0 for all events, since find_ri_events failed.
        is_ri_case = 0
    else:
        is_ri_case = 1

    # 2. Predictive Processing Loop
    for idx, event in enumerate(ri_events):
        try:
            # We populated the event dictionary with the data from the current experiment.
            event['name'] = PARAMS.get('storm_name', 'UNKNOWN')
            event['year'] = PARAMS.get('year', 'N/A')
            event['lead_time_hours'] = PARAMS.get('lead_time_hours', 0)
            event['is_RI_actual'] = is_ri_case

            # Maps wind and pressure if your parser uses different keys.
            event['wind_speed_knots'] = event.get('wind', 0)
            event['pressure_mb'] = event.get('pressure', 0)

            # --- FORECAST LOGIC ---
            target_dt = pd.to_datetime(event['datetime'])
            input_dt = target_dt - timedelta(hours=event['lead_time_hours'])

            input_event_payload = event.copy()
            input_event_payload['datetime'] = input_dt.strftime('%Y%m%d %H%M')

            ts_target = target_dt.strftime('%Y%m%d_%H%M')
            ts_input = input_dt.strftime('%Y%m%d_%H%M')

            logger.info(f"--- Processing Event {idx+1} ---")
            logger.info(f"GROUND TRUTH (Target): {ts_target}")
            logger.info(
                f"PREDICTOR DATA (Input): {ts_input} (T-{event['lead_time_hours']}h)")

            # Download and Build Cubes
            era5_files = download_era5_for_event(
                input_event_payload, PATHS["raw_era5_dir"])
            cube_series = create_cube_series(input_event_payload, era5_files)

            if not cube_series:
                logger.warning(
                    f"Data gap detected for input {ts_input}. Skipping event...")
                continue

            # 3. Neural Inference
            X = np.stack(cube_series, axis=0)
            X = np.nan_to_num(X, nan=0.0)
            X_input = np.expand_dims(X, axis=0)

            intensity = predict_intensity(X_input)
            hotspot = compute_hotspot(X_input)

            # 4. Spatial Validation & Metric Archiving
            out_dir = PATHS["output_figures"]
            plot_storm_track(hurdat_df, event,
                             save_path=out_dir / f"track_{ts_target}.png")
            plot_hotspot_map(
                hotspot[0], event, save_path=out_dir / f"hotspot_{ts_target}.png")

            critical_points = get_critical_coordinates(hotspot[0], event)

            if critical_points:
                best_guess = critical_points[0]
                output_root = PATHS["output_figures"].parent

                env_threshold = float(os.getenv("RI_THRESHOLD", "0.6"))

                error_km = generate_validation_csv(
                    event_data=event,
                    predicted_coords=best_guess,
                    weight=best_guess['intensity_weight'],
                    output_path=output_root,
                    ri_threshold=env_threshold
                )

                logger.info(
                    f"Scientific metrics archived. Prediction Error: {error_km:.2f} km")

                for pt in critical_points:
                    logger.info(
                        f"   -> Forecasted Lat: {pt['lat']}, Lon: {pt['lon']} (Weight: {pt['intensity_weight']})")

        except Exception as e:
            logger.error(f"Critical failure in event {idx}: {str(e)}")
            continue

    logger.info("Pipeline Execution Completed Successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
