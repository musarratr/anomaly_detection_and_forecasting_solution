# src/data/preprocessing.py

import pandas as pd


def basic_cleaning(
    df: pd.DataFrame,
    time_col: str,
    sensor_id_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Drop null timestamps/values and sort by sensor_id, time."""
    df_clean = df.copy()
    df_clean = df_clean[df_clean[time_col].notna() & df_clean[value_col].notna()]
    df_clean = df_clean.sort_values([sensor_id_col, time_col]).reset_index(drop=True)
    return df_clean
