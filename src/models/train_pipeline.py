# src/models/train_pipeline.py

"""
End-to-end TRAINING / RETRAINING pipeline:

1. Load raw oxygen data.
2. Clean data (drop nulls, sort).
3. Fit anomaly detector & score data.
4. Build forecasting features from cleaned + scored data.
5. Train global forecaster with TVT split.
6. Save models and baseline stats for monitoring.
"""

import argparse
import json
import os

import yaml
import joblib
import numpy as np
import pandas as pd

from src.data.io import load_raw_oxygen, save_dataframe
from src.data.preprocessing import basic_cleaning
from src.features.timeseries import add_time_features, add_lag_features, add_rolling_features
from src.models.anomaly import AnomalyConfig, OxygenAnomalyDetector
from src.models.forecaster import ForecastConfig, train_global_forecaster


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_training(config_path: str):
    cfg = load_config(config_path)

    data_cfg = cfg["data"]
    anomaly_cfg = cfg["anomaly"]
    forecast_cfg = cfg["forecast"]
    out_cfg = cfg["output"]

    time_col = data_cfg["time_col"]
    sensor_id_col = data_cfg["sensor_id_col"]
    value_col = data_cfg["value_col"]

    # 1. Load & clean raw data
    df_raw = load_raw_oxygen(
        data_cfg["raw_path"],
        time_col=time_col,
        sensor_id_col=sensor_id_col,
        value_col=value_col,
    )
    df_clean = basic_cleaning(df_raw, time_col, sensor_id_col, value_col)

    # 2. Fit anomaly detector & score
    a_cfg = AnomalyConfig(**anomaly_cfg)
    detector = OxygenAnomalyDetector(a_cfg)
    detector.fit(df_clean, time_col=time_col, sensor_id_col=sensor_id_col, value_col=value_col)

    df_scored = detector.score(
        df_clean, time_col=time_col, sensor_id_col=sensor_id_col, value_col=value_col
    )

    # 3. Build forecasting features (time, lags, rolling)
    df_feat = add_time_features(df_scored, time_col=time_col)
    df_feat = add_lag_features(
        df_feat, sensor_id_col=sensor_id_col, value_col=value_col,
        lag_minutes=forecast_cfg["lag_minutes"],
    )
    df_feat = add_rolling_features(
        df_feat, sensor_id_col=sensor_id_col, value_col=value_col,
        rolling_window_minutes=forecast_cfg["rolling_window_minutes"],
    )

    # Drop rows with NaNs in features (due to initial lags/rolls)
    feature_cols = (
        [f"lag_{lag}" for lag in forecast_cfg["lag_minutes"]]
        + ["roll_mean_60", "sin_time", "cos_time", "sin_dow", "cos_dow"]
    )
    df_model = df_feat.dropna(subset=feature_cols + [value_col]).copy()

    # 4. Filter out top severity quantile for training (anomaly-aware training)
    q_cut = anomaly_cfg["severity_quantile_for_training_cutoff"]
    sev_cut = df_model["severity"].quantile(q_cut)
    df_model_trainable = df_model[df_model["severity"] < sev_cut].copy()

    # 5. Train global forecaster
    f_cfg = ForecastConfig(
        valid_days=forecast_cfg["valid_days"],
        test_days=forecast_cfg["test_days"],
        lag_minutes=forecast_cfg["lag_minutes"],
        rolling_window_minutes=forecast_cfg["rolling_window_minutes"],
        learning_rates=forecast_cfg["learning_rates"],
        max_depths=forecast_cfg["max_depths"],
        max_iter=forecast_cfg["max_iter"],
    )

    forecaster, metrics, split_info = train_global_forecaster(
        df_model_trainable,
        f_cfg,
        time_col=time_col,
        value_col=value_col,
        feature_cols=feature_cols,
    )

    print("Training complete.")
    print("Best parameters:", metrics["params"])
    print("Train MAE / RMSE:", metrics["train"])
    print("Valid MAE / RMSE:", metrics["valid"])
    print("Test  MAE / RMSE:", metrics["test"])

    # 6. Save processed dataset sample and full processed path
    save_dataframe(df_model, data_cfg["processed_path"])

    # 7. Save anomaly config
    os.makedirs(out_cfg["model_dir"], exist_ok=True)
    with open(out_cfg["anomaly_config_path"], "w", encoding="utf-8") as f:
        json.dump(a_cfg.__dict__, f, indent=2)

    # 8. Save forecaster
    joblib.dump(forecaster, out_cfg["forecaster_path"])

    # 9. Save baseline stats for monitoring
    # Use test metrics + severity distribution
    sev_median = float(df_model["severity"].median())
    baseline_stats = {
        "severity_median": sev_median,
        "test_mae": float(metrics["test"][0]),
        "test_rmse": float(metrics["test"][1]),
        "valid_start": split_info["valid_start"],
        "test_start": split_info["test_start"],
        "params": metrics["params"],
    }
    with open(out_cfg["baseline_stats_path"], "w", encoding="utf-8") as f:
        json.dump(baseline_stats, f, indent=2)

    print("Models and baseline stats saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train anomaly detector + forecaster.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()
    run_training(args.config)
