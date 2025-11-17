# src/inference/batch_inference.py

"""
Batch inference script:

- Load trained models + config.
- Load a NEW raw CSV of oxygen data.
- Run anomaly scoring.
- Build features and generate 1-step-ahead predictions for each row.
- Save scored + predicted dataset.
"""

import argparse
import json
import os

import joblib
import yaml
import pandas as pd

from src.data.io import load_raw_oxygen, save_dataframe
from src.data.preprocessing import basic_cleaning
from src.features.timeseries import add_time_features, add_lag_features, add_rolling_features
from src.models.anomaly import AnomalyConfig, OxygenAnomalyDetector


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_batch_inference(config_path: str, input_path: str, output_path: str):
    cfg = load_config(config_path)

    data_cfg = cfg["data"]
    anomaly_cfg = cfg["anomaly"]
    forecast_cfg = cfg["forecast"]
    out_cfg = cfg["output"]

    time_col = data_cfg["time_col"]
    sensor_id_col = data_cfg["sensor_id_col"]
    value_col = data_cfg["value_col"]

    # 1. Load models
    a_cfg = AnomalyConfig(**anomaly_cfg)
    detector = OxygenAnomalyDetector(a_cfg)
    # For rule-based, we need context baseline; load training processed data
    df_train_processed = pd.read_csv(data_cfg["processed_path"])
    df_train_processed[time_col] = pd.to_datetime(df_train_processed[time_col])
    detector.fit(df_train_processed, time_col=time_col, sensor_id_col=sensor_id_col, value_col=value_col)

    forecaster = joblib.load(out_cfg["forecaster_path"])

    # 2. Load & clean NEW raw data
    df_raw = load_raw_oxygen(
        input_path,
        time_col=time_col,
        sensor_id_col=sensor_id_col,
        value_col=value_col,
    )
    df_clean = basic_cleaning(df_raw, time_col, sensor_id_col, value_col)

    # 3. Anomaly scoring
    df_scored = detector.score(
        df_clean, time_col=time_col, sensor_id_col=sensor_id_col, value_col=value_col
    )

    # 4. Feature building
    df_feat = add_time_features(df_scored, time_col=time_col)
    df_feat = add_lag_features(df_feat, sensor_id_col, value_col, forecast_cfg["lag_minutes"])
    df_feat = add_rolling_features(
        df_feat, sensor_id_col, value_col, forecast_cfg["rolling_window_minutes"]
    )

    feature_cols = (
        [f"lag_{lag}" for lag in forecast_cfg["lag_minutes"]]
        + ["roll_mean_60", "sin_time", "cos_time", "sin_dow", "cos_dow"]
    )
    df_infer = df_feat.dropna(subset=feature_cols).copy()

    # 5. Predict
    df_infer["y_pred"] = forecaster.predict(df_infer[feature_cols])

    # 6. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_dataframe(df_infer, output_path)

    print(f"Inference complete. Saved scored & predicted data to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inference: anomaly + forecast.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to NEW raw oxygen CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output_scored_forecast.csv",
        help="Where to save scored + forecasted CSV.",
    )
    args = parser.parse_args()
    run_batch_inference(args.config, args.input, args.output)
