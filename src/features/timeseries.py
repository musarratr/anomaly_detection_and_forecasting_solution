# src/features/timeseries.py

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Add minute-of-day and day-of-week + cyclical encodings."""
    df = df.copy()
    dt = df[time_col].dt
    df["minute_of_day"] = dt.hour * 60 + dt.minute
    df["dayofweek"] = dt.dayofweek

    df["sin_time"] = np.sin(2 * np.pi * df["minute_of_day"] / 1440.0)
    df["cos_time"] = np.cos(2 * np.pi * df["minute_of_day"] / 1440.0)
    df["sin_dow"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    return df


def add_lag_features(
    df: pd.DataFrame,
    sensor_id_col: str,
    value_col: str,
    lag_minutes: list,
) -> pd.DataFrame:
    """Add simple lag features (in minutes) per sensor."""
    df = df.copy()
    for lag in lag_minutes:
        df[f"lag_{lag}"] = df.groupby(sensor_id_col)[value_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    sensor_id_col: str,
    value_col: str,
    rolling_window_minutes: int,
) -> pd.DataFrame:
    """Add rolling mean (baseline) per sensor."""
    df = df.copy()
    df["roll_mean_60"] = (
        df.groupby(sensor_id_col)[value_col]
        .rolling(window=rolling_window_minutes, min_periods=rolling_window_minutes // 2)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df
