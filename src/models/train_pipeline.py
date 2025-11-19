# src/models/train_pipeline.py

"""
End-to-end TRAINING / RETRAINING pipeline:

1. Load raw oxygen data.
2. Clean data (drop nulls, sort).
3. Fit anomaly detector & score data.
4. Remove top-quantile severe anomalies (reduces training noise).
5. Build forecasting features from cleaned + scored data.
6. Train global forecaster with TVT split (train / valid / test).
7. Save models and baseline stats for monitoring.
"""

import argparse
import json
import os

import joblib
import numpy as np
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
from src.models.forecaster import ForecastConfig, train_global_forecaster


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_training_from_config(cfg: dict):
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
    detector.fit(
        df_clean,
        time_col=time_col,
        sensor_id_col=sensor_id_col,
        value_col=value_col,
    )

    df_scored = detector.score(
        df_clean,
        time_col=time_col,
        sensor_id_col=sensor_id_col,
        value_col=value_col,
    )

# 3. Remove the most severe anomalies (top quantile) before feature engineering
    q_cut = a_cfg.severity_quantile_for_training_cutoff
    sev_cut = df_scored["severity"].quantile(q_cut)
    df_scored = df_scored[df_scored["severity"] < sev_cut].copy()

# 4. Build forecasting features (time encodings, lags, rolling mean)
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

# 5. Drop rows with NaNs in features (due to initial lags/rolls)
    feature_cols = (
        [f"lag_{lag}" for lag in forecast_cfg["lag_minutes"]]
        + [
            "roll_mean_60",
            "sin_time",
            "cos_time",
            "sin_dow",
            "cos_dow",
        ]
    )
    df_model = df_feat.dropna(subset=feature_cols + [value_col]).copy()

# 6. Train global forecaster (time-based train/valid/test split)
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
        df_model,
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

# 7. Save processed dataset (scored + engineered features)
    save_dataframe(df_model, data_cfg["processed_path"])

# 8. Save anomaly config
    os.makedirs(out_cfg["model_dir"], exist_ok=True)
    with open(out_cfg["anomaly_config_path"], "w", encoding="utf-8") as f:
        json.dump(a_cfg.__dict__, f, indent=2)

# 9. Save forecaster model
    joblib.dump(forecaster, out_cfg["forecaster_path"])

# 10. Save baseline stats for monitoring
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

    return {
        "forecaster": forecaster,
        "metrics": metrics,
        "baseline_stats": baseline_stats,
        "output_paths": {
            "anomaly_config": out_cfg["anomaly_config_path"],
            "forecaster": out_cfg["forecaster_path"],
            "baseline_stats": out_cfg["baseline_stats_path"],
            "processed_data": data_cfg["processed_path"],
        },
    }


def run_training(config_path: str):
    cfg = load_config(config_path)
    return run_training_from_config(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train anomaly detector + forecaster."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()
    run_training(args.config)
