# src/inference/horizon_forecast.py

"""
Given processed & scored history for a sensor, produce a 1-week
minute-level forecast using the trained forecaster (iterative 1-step).
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def forecast_one_week(
    config_path: str,
    processed_path: str | None,
    sensor_id: str,
    output_path: str,
):
    # ------------------------------------------------------------------ #
    # 1. Load config
    # ------------------------------------------------------------------ #
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    forecast_cfg = cfg["forecast"]
    out_cfg = cfg["output"]

    time_col = data_cfg["time_col"]
    sensor_id_col = data_cfg["sensor_id_col"]
    value_col = data_cfg["value_col"]

    # If processed_path is not given, use the one from config
    if processed_path is None:
        processed_path = data_cfg["processed_path"]

    # Resolve path & sanity check
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"Processed dataset not found at '{processed_path}'. "
            "Make sure you run the training pipeline first, or adjust "
            "`data.processed_path` in configs/pipeline_config.yaml or "
            "pass --processed with the correct path."
        )

    # ------------------------------------------------------------------ #
    # 2. Load processed data (scored + features) and trained forecaster
    # ------------------------------------------------------------------ #
    df = pd.read_csv(processed_path)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    forecaster = joblib.load(out_cfg["forecaster_path"])

    # ------------------------------------------------------------------ #
    # 3. Filter to the requested sensor and sort by time
    # ------------------------------------------------------------------ #
    df_sensor = df[df[sensor_id_col] == sensor_id].copy()
    if df_sensor.empty:
        raise ValueError(
            f"No data found for sensor_id='{sensor_id}' in '{processed_path}'. "
            "Double-check the sensor_id string (including pipes) and make sure "
            "it matches exactly what is in the processed file."
        )

    df_sensor = df_sensor.sort_values(time_col).reset_index(drop=True)

    # Keep only time + value for the iterative history
    hist = df_sensor[[time_col, value_col]].copy().reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 4. Build 1-week iterative forecast
    # ------------------------------------------------------------------ #
    horizon_minutes = 7 * 24 * 60  # one week
    feature_lags = forecast_cfg["lag_minutes"]
    max_lag = max(feature_lags)
    roll_window = forecast_cfg["rolling_window_minutes"]

    feature_cols = (
        [f"lag_{lag}" for lag in feature_lags]
        + [
            "roll_mean_60",
            "minute_of_day",
            "dayofweek",
            "sin_time",
            "cos_time",
            "sin_dow",
            "cos_dow",
        ]
    )

    future_rows = []
    current_time = hist[time_col].iloc[-1]

    for _ in range(horizon_minutes):
        current_time = current_time + pd.Timedelta(minutes=1)

        # Time-based features
        minute_of_day = current_time.hour * 60 + current_time.minute
        dayofweek = current_time.dayofweek

        feat = {
            "minute_of_day": minute_of_day,
            "dayofweek": dayofweek,
            "sin_time": np.sin(2 * np.pi * minute_of_day / 1440.0),
            "cos_time": np.cos(2 * np.pi * minute_of_day / 1440.0),
            "sin_dow": np.sin(2 * np.pi * dayofweek / 7.0),
            "cos_dow": np.cos(2 * np.pi * dayofweek / 7.0),
        }

        # Need enough history for all lags
        if len(hist) <= max_lag:
            raise RuntimeError(
                "Not enough history to create lag features for forecasting. "
                f"Need more than {max_lag} points, have {len(hist)}."
            )

        # Lag features from history
        for lag in feature_lags:
            feat[f"lag_{lag}"] = hist[value_col].iloc[-lag]

        # Rolling mean over last `roll_window` minutes
        last_window = hist[value_col].iloc[-roll_window:]
        feat["roll_mean_60"] = last_window.mean()

        # Build DataFrame in correct column order
        X_future = pd.DataFrame([feat])[feature_cols]

        # Predict one step
        y_future = forecaster.predict(X_future)[0]

        # Store forecast
        future_rows.append(
            {time_col: current_time, sensor_id_col: sensor_id, "forecast": y_future}
        )

        # Append forecast to history for next iteration
        hist = pd.concat(
            [hist, pd.DataFrame({time_col: [current_time], value_col: [y_future]})],
            ignore_index=True,
        )

    df_future = pd.DataFrame(future_rows)

    # ------------------------------------------------------------------ #
    # 5. Save forecast
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_future.to_csv(output_path, index=False)
    print(
        f"Saved 1-week forecast for sensor '{sensor_id}' "
        f"to '{output_path}'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="1-week horizon forecast for a single sensor."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--processed",
        type=str,
        default=None,
        help=(
            "Path to processed (scored + features) CSV. "
            "If omitted, uses data.processed_path from config."
        ),
    )
    parser.add_argument(
        "--sensor_id",
        type=str,
        required=True,
        help="Exact sensor_id string, e.g. "
             "'System_10|EquipmentUnit_10|SubUnit_07'. "
             "IMPORTANT: must be quoted in the shell because of pipes.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/forecast_1week_single_sensor.csv",
        help="Where to save the 1-week forecast CSV.",
    )
    args = parser.parse_args()

    forecast_one_week(args.config, args.processed, args.sensor_id, args.output)
