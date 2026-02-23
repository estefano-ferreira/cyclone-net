from __future__ import annotations
import pandas as pd

def label_ri(df: pd.DataFrame, ri_threshold_kt_24h: float = 30.0) -> pd.DataFrame:
    """Label Rapid Intensification using best-track wind in knots.

    RI (classic) is defined as ΔV >= 30 kt over 24h.
    Assumes df is sorted by time within each storm id (sid) and has wind_kt.
    """
    df = df.copy()
    df["wind_kt"] = pd.to_numeric(df["wind_kt"], errors="coerce")
    df["wind_kt_shift_24"] = df.groupby("sid")["wind_kt"].shift(-4)  # 4*6h = 24h
    df["dv24_kt"] = df["wind_kt_shift_24"] - df["wind_kt"]
    df["ri_label"] = (df["dv24_kt"] >= float(ri_threshold_kt_24h)).astype(int)
    return df

def add_wind_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Add continuous targets dv12 and dv24 (knots)."""
    df = df.copy()
    df["wind_kt"] = pd.to_numeric(df["wind_kt"], errors="coerce")
    df["wind_kt_shift_12"] = df.groupby("sid")["wind_kt"].shift(-2)  # 2*6h = 12h
    df["wind_kt_shift_24"] = df.groupby("sid")["wind_kt"].shift(-4)
    df["dv12_kt"] = df["wind_kt_shift_12"] - df["wind_kt"]
    df["dv24_kt"] = df["wind_kt_shift_24"] - df["wind_kt"]
    return df
