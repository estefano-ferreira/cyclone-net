"""Geometry utilities for coordinate conversion and soft-argmax."""

import torch
import torch.nn.functional as F
import numpy as np


def soft_argmax(heatmap, temperature=1.0):
    """Compute expected coordinates in normalized [-1, 1] from a heatmap."""
    B, _, H, W = heatmap.shape
    heatmap_flat = heatmap.view(B, -1)
    weights = F.softmax(heatmap_flat * temperature, dim=1)

    y_grid = torch.linspace(-1, 1, H, device=heatmap.device).view(1,
                                                                  H, 1).expand(1, H, W).reshape(1, -1)
    x_grid = torch.linspace(-1, 1, W, device=heatmap.device).view(1,
                                                                  1, W).expand(1, H, W).reshape(1, -1)

    y_pred = (weights * y_grid).sum(dim=1)
    x_pred = (weights * x_grid).sum(dim=1)
    return torch.stack([y_pred, x_pred], dim=1)


def normalized_to_geographic(y_norm, x_norm, lats_2d, lons_2d):
    """Convert normalized coordinates to geographic using bilinear interpolation."""
    H, W = lats_2d.shape

    y = (y_norm + 1) / 2 * (H - 1)
    x = (x_norm + 1) / 2 * (W - 1)

    y0 = int(np.floor(y))
    y1 = min(y0 + 1, H - 1)
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, W - 1)

    y0, y1 = max(0, y0), max(0, y1)
    x0, x1 = max(0, x0), max(0, x1)

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
