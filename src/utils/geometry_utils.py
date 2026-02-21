"""
CycloneNet: Geometry utilities for coordinate conversion and soft-argmax.
Now includes function to convert normalized coordinates to geographic using precomputed lat/lon grids.
"""

import torch
import torch.nn.functional as F
import numpy as np


def soft_argmax(heatmap, temperature=1.0):
    """
    Compute expected coordinates in normalized [-1, 1] from a heatmap.

    Args:
        heatmap: (B, 1, H, W) tensor (after sigmoid)
        temperature: float, softmax temperature (higher = more diffuse)

    Returns:
        Tuple of (y_coords, x_coords) each of shape (B,)
    """
    B, _, H, W = heatmap.shape
    heatmap_flat = heatmap.view(B, -1)
    weights = F.softmax(heatmap_flat * temperature, dim=1)

    y_grid = torch.linspace(-1, 1, H, device=heatmap.device).view(1,
                                                                  H, 1).expand(1, H, W).reshape(1, -1)
    x_grid = torch.linspace(-1, 1, W, device=heatmap.device).view(1,
                                                                  1, W).expand(1, H, W).reshape(1, -1)

    y_pred = (weights * y_grid).sum(dim=1)  # (B,)
    x_pred = (weights * x_grid).sum(dim=1)  # (B,)
    return y_pred, x_pred


def normalized_to_geographic(y_norm, x_norm, lats_2d, lons_2d):
    """
    Convert normalized coordinates (in [-1, 1]) to geographic coordinates using bilinear interpolation
    on the precomputed latitude/longitude grids.

    Args:
        y_norm: float, normalized y coordinate (-1 = top, 1 = bottom)
        x_norm: float, normalized x coordinate (-1 = left, 1 = right)
        lats_2d: np.ndarray of shape (H, W) with latitude values
        lons_2d: np.ndarray of shape (H, W) with longitude values

    Returns:
        (lat, lon): floats, geographic coordinates
    """
    H, W = lats_2d.shape

    # Convert normalized to pixel indices (floating point)
    y = (y_norm + 1) / 2 * (H - 1)
    x = (x_norm + 1) / 2 * (W - 1)

    # Bilinear interpolation
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, H - 1)
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, W - 1)

    y0 = max(0, y0)
    y1 = max(0, y1)
    x0 = max(0, x0)
    x1 = max(0, x1)

    wy = y - y0
    wx = x - x0

    lat = (1 - wy) * (1 - wx) * lats_2d[y0, x0] + \
          (1 - wy) * wx * lats_2d[y0, x1] + \
        wy * (1 - wx) * lats_2d[y1, x0] + \
        wy * wx * lats_2d[y1, x1]

    lon = (1 - wy) * (1 - wx) * lons_2d[y0, x0] + \
          (1 - wy) * wx * lons_2d[y0, x1] + \
        wy * (1 - wx) * lons_2d[y1, x0] + \
        wy * wx * lons_2d[y1, x1]

    return float(lat), float(lon)
