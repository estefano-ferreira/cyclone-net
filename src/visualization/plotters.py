"""
CycloneNet: Physics-Guided Framework for Targeted RI Detection.
---------------------------------------------------------------
Software Engineer: Estefano Senhor Ferreira
License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)

This work is licensed under CC BY-NC 4.0. 
Commercial use is strictly prohibited without prior authorization.
Copyright (c) 2026 Estefano Senhor Ferreira
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import logging
import cartopy.crs as ccrs
import cartopy.feature as cfeature

logger = logging.getLogger(__name__)


def plot_storm_track(hurdat_df: pd.DataFrame, event: dict,
                     save_path: Optional[Path] = None):
    """Plot the trajectory and save it on disc.."""
    try:
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)

        storm_data = hurdat_df[hurdat_df['storm_id'] == event['storm_id']]
        plt.plot(storm_data['lon'], storm_data['lat'],
                 'b-', transform=ccrs.PlateCarree())
        plt.plot(event['lon'], event['lat'], 'ro',
                 transform=ccrs.PlateCarree())

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Saved image: {save_path}")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error plot_storm_track: {e}")


def plot_hotspot_map(hotspot_data: np.ndarray, event: dict, save_path: Optional[Path] = None):
    """
    Plot the sensitivity map (anomalies that fuel the cyclone).
    """
    try:
        # 1. Ensure that the data is 2D (H, W)
        # The hotspot usually comes as (T, H, W, C) or (H, W). Let's take the temporal and channel average.
        data = np.array(hotspot_data)

        if data.ndim == 4:  # (T, H, W, C)
            data = np.mean(data, axis=(0, -1))
        elif data.ndim == 3:  # (H, W, C) ou (T, H, W)
            data = np.mean(data, axis=0 if data.shape[0] < 10 else -1)

        # 2. Configure the figure
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # 3. Define the extension (5-degree window around the eye)
        extent = [event['lon']-5, event['lon'] +
                  5, event['lat']-5, event['lat']+5]

        # 4. Plot the heat map
        im = ax.imshow(data, extent=extent, cmap='magma',
                       origin='lower', transform=ccrs.PlateCarree())

        plt.colorbar(im, ax=ax, label='Energy Sensitivity Index')
        plt.title(f"Sensitivity Map - Event RI\n{event['datetime']}")

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Saved Hotspot Map: {save_path.name}")

        plt.close(fig)

    except Exception as e:
        logger.error(f"Error plot_hotspot_map: {e}")


def plot_cube_slice(cube, channel_idx=0, save_path=None):
    # Simple implementation to avoid import errors in __init__
    plt.figure()
    plt.imshow(cube[..., channel_idx].mean(axis=0))
    if save_path:
        plt.savefig(save_path)
    plt.close()
