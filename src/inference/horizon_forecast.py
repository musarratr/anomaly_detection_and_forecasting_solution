# src/inference/horizon_forecast.py

"""
Given processed & scored history for a sensor, produce a 1-week
minute-level forecast using the trained forecaster (iterative 1-step).
"""

import argparse
import yaml
import joblib
import numpy as np
import pandas as pd

from src.features.timeseries import add_time_features


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def forecast_one_week(
    config_path: str,
    processed_path: str,
    sensor_id: str,
    output_path: str,
):
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    forecast_cfg = cfg["forecast"]
    out_cfg = cfg["output"]

    time_col = data_cfg["time_col"]
    sensor_id_col = data_cfg["sensor_id_col"]
    value_col = data_cfg["value_col"]

    df = pd.read_csv(processed_path)
    df[time_col] = pd.to_datetime(df[time_col])

    forecaster = joblib.load(out_cfg["forecaster_path"])

    # Filter for sensor
    df_sensor = df[df[sensor_id_col] == sensor_id].copy()
    if df_sensor.empty:
        raise ValueError(f"No data found for sensor_id={sensor_id}")

    df_sensor = df_sensor.sort_values(time_col).reset_index(drop=True)

    # Keep only time + value for iterative loop
    hist = df_sensor[[time_col, value_col]].copy().reset_index(drop=True)

    horizon_minutes = 7 * 24 * 60  # one week
    feature_lags = forecast_cfg["lag_minutes"]
    max_lag = max(feature_lags)
    feature_cols = (
        [f"lag_{lag}" for lag in feature_lags]
        + ["roll_mean_60", "sin_time", "cos_time", "sin_dow", "cos_dow"]
    )

    future_rows = []
    current_time = hist[time_col].iloc[-1]

    for _ in range(horizon_minutes):
        current_time = current_time + pd.Timedelta(minutes=1)

        # Build feature row for current_time from history
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

        if len(hist) <= max_lag:
            raise RuntimeError("Not enough history to create lag features for forecasting.")

        for lag in feature_lags:
            feat[f"lag_{lag}"] = hist[value_col].iloc[-lag]

        last_60 = hist[value_col].iloc[-forecast_cfg["rolling_window_minutes"] :]
        feat["roll_mean_60"] = last_60.mean()

        X_future = pd.DataFrame([feat])[feature_cols]
        y_future = forecaster.predict(X_future)[0]

        future_rows.append(
            {time_col: current_time, sensor_id_col: sensor_id, "forecast": y_future}
        )

        hist = pd.concat(
            [hist, pd.DataFrame({time_col: [current_time], value_col: [y_future]})],
            ignore_index=True,
        )

    df_future = pd.DataFrame(future_rows)
    df_future.to_csv(output_path, index=False)
    print(f"Saved 1-week forecast for sensor {sensor_id} to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1-week horizon forecast for a sensor.")
    parser.add_argument("--config", type=str, default="configs/pipeline_config.yaml")
    parser.add_argument("--processed", type=str, default="data/oxygen_processed_full.csv")
    parser.add_argument("--sensor_id", type=str, required=True)
    parser.add_argument(
        "--output", type=str, default="data/forecast_1week_single_sensor.csv"
    )
    args = parser.parse_args()

    forecast_one_week(args.config, args.processed, args.sensor_id, args.output)
