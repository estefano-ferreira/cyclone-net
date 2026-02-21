"""
CycloneNet V2.1 – Rapid Intensification (RI) labeling strategies.

This module provides functions to add RI labels to storm track data,
supporting both future‑window (start‑only) and event‑window (full interval) modes.

Author: Estefano Senhor Ferreira
License: CC BY-NC 4.0
"""

from typing import Literal, Optional
import numpy as np
import pandas as pd

RIMode = Literal["future_window", "event_window",
                 "t0_is_positive_if_future_delta_meets_threshold"]


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
    """Return binary RI label per row."""
    if sid_col not in df or wind_col not in df:
        raise KeyError(f"DataFrame must contain '{sid_col}' and '{wind_col}'")

    effective_hours = int(
        horizon_hours) if horizon_hours is not None else int(window_hours)
    steps = int(round(effective_hours / float(time_step_hours)))
    if steps <= 0:
        raise ValueError("window_hours/time_step_hours must be positive.")

    if mode in ("future_window", "t0_is_positive_if_future_delta_meets_threshold"):
        future = df.groupby(sid_col, sort=False)[wind_col].shift(-steps)
        delta = future - df[wind_col]
        return (delta >= delta_v_kt).fillna(False).astype(int)

    if mode == "event_window":
        future = df.groupby(sid_col, sort=False)[wind_col].shift(-steps)
        delta = future - df[wind_col]
        starts = (delta >= delta_v_kt).fillna(False)

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
