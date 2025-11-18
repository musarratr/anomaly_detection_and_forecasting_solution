# src/data/io.py

import os
import pandas as pd


def load_raw_oxygen(path: str,
                    time_col: str = "time",
                    sensor_id_col: str = "sensor_id",
                    value_col: str = "Oxygen[%sat]") -> pd.DataFrame:
    """Load the raw oxygen dataset from CSV."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data not found at {path}")
    df = pd.read_csv(path)

    # Generic sensor id: we just use tags as opaque identifiers.
    # This will still generalize to new customers & tag values.

    df["sensor_id"] = (
        df["System"].astype(str)
        + "|"
        + df["EquipmentUnit"].astype(str)
        + "|"
        + df["SubUnit"].astype(str)
    )

    df.rename(columns={"Oxygen[%sat]": "oxygen"}, inplace=True)

    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found in columns: {df.columns}")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    if sensor_id_col not in df.columns:
        df[sensor_id_col] = "global"  # fallback if not provided
    if value_col not in df.columns:
        raise ValueError(f"value_col '{value_col}' not found in columns: {df.columns}")
    return df


def save_dataframe(df: pd.DataFrame, path: str, index: bool = False) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)
