from __future__ import annotations
import pandas as pd


def add_wind_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Add continuous targets dv12 and dv24 (knots) using strict-temporal semantics.

    Strict-temporal semantics:
    - Partner = exact temporal match at t0+12h / t0+24h, same SID
    - dv12_kt / dv24_kt = NULL when no exact temporal partner or wind missing
    - No tolerance window; no positional shifts.

    Requires:
    - df sorted by (sid, timestamp) before calling
    - df has 'timestamp' column (datetime64 or parseable)
    - df has 'wind_kt' column (numeric, or coercible)

    Returns a copy with dv12_kt and dv24_kt added (nullable Int64 or float with NaN).
    """
    if "timestamp" not in df.columns:
        raise ValueError("add_wind_deltas requires 'timestamp' column (datetime64 or parseable)")

    df = df.copy()
    df["wind_kt"] = pd.to_numeric(df["wind_kt"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Build a deduplicated (sid, timestamp) -> wind lookup for exact partners.
    partners = df[["sid", "timestamp", "wind_kt"]].drop_duplicates(["sid", "timestamp"])

    for hours, col_name in ((12, "dv12_kt"), (24, "dv24_kt")):
        # Create a shifted timestamp: the PREVIOUS time from which we want wind.
        # For dv24, we need wind at t0+24h, so we shift the partners back by 24h.
        p = partners.assign(t_partner=partners["timestamp"] - pd.Timedelta(hours=hours))
        p = p.rename(columns={"wind_kt": f"wind_partner_{col_name}"})[
            ["sid", "t_partner", f"wind_partner_{col_name}"]
        ]

        # Left merge: find partner at exact t0+hours.
        # If no partner exists (outer rows), the merge will produce NaN.
        df = df.merge(
            p,
            left_on=["sid", "timestamp"],
            right_on=["sid", "t_partner"],
            how="left",
        ).drop(columns=["t_partner"])

        # Compute delta: partner_wind - current_wind.
        # NULL if either wind is missing.
        df[col_name] = df[f"wind_partner_{col_name}"] - df["wind_kt"]
        df = df.drop(columns=[f"wind_partner_{col_name}"])

    return df


def label_ri(df: pd.DataFrame, ri_threshold_kt_24h: float = 30.0) -> pd.DataFrame:
    """Label Rapid Intensification using best-track wind with strict-temporal semantics.

    RI (classic) is defined as dv24 >= 30 kt over 24h.
    Uses strict-temporal matching: only events with dv24 defined.
    NULL (pd.NA) when dv24 is undefined; never silently 0.

    Requires:
    - df has 'dv24_kt' column (from add_wind_deltas; may contain NaN/NA)

    Returns a copy with ri_label added (nullable Int64).
    """
    df = df.copy()

    # Ensure dv24_kt exists and is numeric.
    if "dv24_kt" not in df.columns:
        raise ValueError("label_ri requires 'dv24_kt' column (from add_wind_deltas)")

    df["dv24_kt"] = pd.to_numeric(df["dv24_kt"], errors="coerce")

    # Create nullable Int64 label.
    # 1 if dv24_kt >= threshold
    # 0 if dv24_kt < threshold
    # pd.NA if dv24_kt is NaN/None
    ri_threshold = float(ri_threshold_kt_24h)
    df["ri_label"] = pd.NA
    mask_defined = df["dv24_kt"].notna()
    df.loc[mask_defined & (df["dv24_kt"] >= ri_threshold), "ri_label"] = 1
    df.loc[mask_defined & (df["dv24_kt"] < ri_threshold), "ri_label"] = 0
    df["ri_label"] = df["ri_label"].astype("Int64")

    return df
