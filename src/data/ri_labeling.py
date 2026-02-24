
"""
CycloneNet V2.1 – RI labeling strategies aligned with forensic diagnostics.

Supported modes:
- t0_is_positive_if_future_delta_meets_threshold:
    Label at time t0 is 1 if V(t0+horizon_hours)-V(t0) >= delta_v_kt.
- future_window:
    Backward-compatible alias of the above (forecast-like framing).
- event_window:
    Label *states within the intensification interval* as positive.
    Recommended for retrospective forensic diagnostic mapping.

IMPORTANT:
- Default arguments below are *fallbacks only*.
  In the pipeline, always pass values from config.yaml (single source of truth).
  
Author: Estefano Senhor Ferreira
License: CC BY-NC 4.0  
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd


RIMode = Literal[
    "future_window",
    "event_window",
    "t0_is_positive_if_future_delta_meets_threshold",
]


def label_ri(
    df: pd.DataFrame,
    mode: RIMode,
    delta_v_kt: float = 30.0,
    window_hours: int = 24,
    horizon_hours: Optional[int] = None,
    time_step_hours: int = 6,
    sid_col: str = "sid",
    wind_col: str = "wind_knots",
) -> pd.Series:
    """Return a binary RI label per row of df (same index).

    Assumptions:
      - df contains track points at a fixed cadence within each SID group.
      - wind is in knots.

    Args:
        df: Track dataframe (must contain sid_col and wind_col).
        mode: RI labeling mode (see module docstring).
        delta_v_kt: Threshold in knots over the window/horizon.
        window_hours: Window size for event_window, and default horizon if horizon_hours is None.
        horizon_hours: Forward horizon for the t0 labeling mode. If None, uses window_hours.
        time_step_hours: Cadence between track points (typically 6).
        sid_col: Storm identifier column.
        wind_col: Wind column (knots).

    Returns:
        pd.Series of int {0,1}.
    """
    if sid_col not in df.columns or wind_col not in df.columns:
        raise KeyError(f"DataFrame must contain '{sid_col}' and '{wind_col}'")

    effective_hours = int(
        horizon_hours) if horizon_hours is not None else int(window_hours)
    steps = int(round(effective_hours / float(time_step_hours)))
    if steps <= 0:
        raise ValueError("window_hours/time_step_hours must be positive.")

    if mode in ("future_window", "t0_is_positive_if_future_delta_meets_threshold"):
        future = df.groupby(sid_col, sort=False)[wind_col].shift(-steps)
        delta = future - df[wind_col]
        out = (delta >= float(delta_v_kt)).fillna(False).astype(int)
        out.name = "ri_label"
        return out

    if mode == "event_window":
        future = df.groupby(sid_col, sort=False)[wind_col].shift(-steps)
        delta = future - df[wind_col]
        starts = (delta >= float(delta_v_kt)).fillna(False)

        labels = np.zeros(len(df), dtype=np.int32)
        for _, g in df.groupby(sid_col, sort=False):
            idx = g.index.to_numpy()
            start_mask = starts.loc[idx].to_numpy().astype(bool)
            start_positions = np.where(start_mask)[0]
            for pos in start_positions:
                end = min(pos + steps, len(idx) - 1)
                labels[idx[pos: end + 1]] = 1

        return pd.Series(labels, index=df.index, name="ri_label")

    raise ValueError(f"Unknown RI labeling mode: {mode}")
