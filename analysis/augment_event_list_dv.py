"""Augment IBTrACS-derived event list with continuous targets and consistent RI labels.

This script makes CycloneNet training artifacts scientifically consistent:
- Computes continuous wind deltas (knots), assuming 6-hour cadence:
    dv12_kt = V(t+12h) - V(t)
    dv24_kt = V(t+24h) - V(t)

- Recomputes ri_label explicitly from dv24_kt using the classic definition:
    RI = 1 if dv24_kt >= 30 kt, else 0

Safety:
- DOES NOT touch any NetCDF (.nc) files.
- Reads/writes only CSV files.

Input:
  data/event_list.csv
Required columns:
  sid, timestamp, wind_knots
Optional columns preserved:
  lat, lon, pressure_mb, storm_name, basin, nc_filename, etc.

Output:
  data/event_list_augmented.csv
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

IN_PATH = Path("data/event_list.csv")
OUT_PATH = Path("data/event_list_augmented.csv")

RI_THRESHOLD_24H_KT = 30.0  # Classic RI definition
CADENCE_HOURS = 6
DV12_STEPS = int(12 / CADENCE_HOURS)  # 2 steps
DV24_STEPS = int(24 / CADENCE_HOURS)  # 4 steps


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {IN_PATH.resolve()}")

    df = pd.read_csv(IN_PATH)

    required = {"sid", "timestamp", "wind_knots"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"event_list.csv missing required columns: {sorted(missing)}")

    df["sid"] = df["sid"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["wind_knots"] = pd.to_numeric(df["wind_knots"], errors="coerce")

    # Keep the original label (if present) for audit, then recompute a consistent one.
    if "ri_label" in df.columns:
        df["ri_label_original"] = df["ri_label"]

    df = df.dropna(subset=["sid", "timestamp", "wind_knots"]).copy()
    df = df.sort_values(["sid", "timestamp"]).reset_index(drop=True)

    df["wind_shift_12"] = df.groupby("sid")["wind_knots"].shift(-DV12_STEPS)
    df["wind_shift_24"] = df.groupby("sid")["wind_knots"].shift(-DV24_STEPS)

    df["dv12_kt"] = df["wind_shift_12"] - df["wind_knots"]
    df["dv24_kt"] = df["wind_shift_24"] - df["wind_knots"]

    df["ri_label"] = (df["dv24_kt"] >= RI_THRESHOLD_24H_KT).astype(int)

    df = df.drop(columns=["wind_shift_12", "wind_shift_24"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    nonnull12 = int(df["dv12_kt"].notna().sum())
    nonnull24 = int(df["dv24_kt"].notna().sum())
    inconsistencies = int((((df["dv24_kt"] >= RI_THRESHOLD_24H_KT) != (df["ri_label"] == 1))).sum())

    print(f"Wrote: {OUT_PATH}")
    print("Columns:", df.columns.tolist())
    print(f"Non-null dv12_kt: {nonnull12} / {len(df)}")
    print(f"Non-null dv24_kt: {nonnull24} / {len(df)}")
    print(f"Inconsistencies (should be 0): {inconsistencies}")


if __name__ == "__main__":
    main()
