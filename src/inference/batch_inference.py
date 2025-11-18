# src/inference/batch_inference.py

"""
Batch inference script:

- Load trained forecaster + anomaly config.
- Load the processed training data to rebuild the anomaly context baseline.
- Load NEW raw oxygen data.
- Run anomaly scoring on the new data.
- Build the SAME feature set used at training time.
- Generate 1-step-ahead predictions for each row.
- Save scored + predicted dataset.
"""

import argparse
import json
import os

import joblib
import pandas as pd
import yaml

from src.data.io import load_raw_oxygen, save_dataframe
from src.data.preprocessing import basic_cleaning
from src.features.timeseries import (
    add_time_features,
    add_lag_features,
    add_rolling_features,
)
from src.models.anomaly import AnomalyConfig, OxygenAnomalyDetector


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_batch_inference(config_path: str, input_path: str | None, output_path: str):
    cfg = load_config(config_path)

    data_cfg = cfg["data"]
    anomaly_cfg = cfg["anomaly"]
    forecast_cfg = cfg["forecast"]
    out_cfg = cfg["output"]

    time_col = data_cfg["time_col"]
    sensor_id_col = data_cfg["sensor_id_col"]
    value_col = data_cfg["value_col"]

    # If no explicit input path is provided, default to the raw_path from config
    if input_path is None:
        input_path = data_cfg["raw_path"]

    # ------------------------------------------------------------------ #
    # 1. Load trained forecaster
    # ------------------------------------------------------------------ #
    forecaster = joblib.load(out_cfg["forecaster_path"])

    # ------------------------------------------------------------------ #
    # 2. Rebuild anomaly context baseline from processed training data
    #    (this file was written by train_pipeline.py)
    # ------------------------------------------------------------------ #
    df_train_processed = pd.read_csv(
        data_cfg["processed_path"],
        low_memory=False,  # avoid DtypeWarning about mixed types
    )
    # Ensure time column is datetime
    df_train_processed[time_col] = pd.to_datetime(
        df_train_processed[time_col], errors="coerce"
    )

    # Fit anomaly detector on processed training data
    a_cfg = AnomalyConfig(**anomaly_cfg)
    detector = OxygenAnomalyDetector(a_cfg)
    detector.fit(
        df_train_processed,
        time_col=time_col,
        sensor_id_col=sensor_id_col,
        value_col=value_col,
    )

    # ------------------------------------------------------------------ #
    # 3. Load & clean NEW raw data
    # ------------------------------------------------------------------ #
    df_raw = load_raw_oxygen(
        input_path,
        time_col=time_col,
        sensor_id_col=sensor_id_col,
        value_col=value_col,
    )
    df_clean = basic_cleaning(df_raw, time_col, sensor_id_col, value_col)

    # ------------------------------------------------------------------ #
    # 4. Anomaly scoring on NEW data
    # ------------------------------------------------------------------ #
    df_scored = detector.score(
        df_clean,
        time_col=time_col,
        sensor_id_col=sensor_id_col,
        value_col=value_col,
    )

    # ------------------------------------------------------------------ #
    # 5. Build features for NEW data (must match training exactly)
    # ------------------------------------------------------------------ #
    df_feat = add_time_features(df_scored, time_col=time_col)
    df_feat = add_lag_features(
        df_feat,
        sensor_id_col=sensor_id_col,
        value_col=value_col,
        lag_minutes=forecast_cfg["lag_minutes"],
    )
    df_feat = add_rolling_features(
        df_feat,
        sensor_id_col=sensor_id_col,
        value_col=value_col,
        rolling_window_minutes=forecast_cfg["rolling_window_minutes"],
    )

    # IMPORTANT: feature_cols MUST MATCH train_pipeline.py
    feature_cols = (
        [f"lag_{lag}" for lag in forecast_cfg["lag_minutes"]]
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

    # Drop rows without full feature set
    df_infer = df_feat.dropna(subset=feature_cols).copy()

    # (Optional safety) ensure all features are numeric
    # df_infer[feature_cols] = df_infer[feature_cols].apply(
    #     pd.to_numeric, errors="coerce"
    # )

    # ------------------------------------------------------------------ #
    # 6. Predict using trained forecaster
    # ------------------------------------------------------------------ #
    df_infer["y_pred"] = forecaster.predict(df_infer[feature_cols])

    # ------------------------------------------------------------------ #
    # 7. Save scored + predicted data
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_dataframe(df_infer, output_path)

    print(f"Inference complete. Saved scored & forecasted data to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch inference: anomaly detection + 1-step forecast."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to NEW raw oxygen CSV. "
             "If omitted, uses data.raw_path from config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/new_scored_forecast.csv",
        help="Where to save scored + forecasted CSV.",
    )
    args = parser.parse_args()

    run_batch_inference(args.config, args.input, args.output)
