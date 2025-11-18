# src/models/forecaster.py

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class ForecastConfig:
    valid_days: int = 7
    test_days: int = 7
    lag_minutes: List[int] | None = None
    rolling_window_minutes: int = 60
    learning_rates: List[float] | None = None
    max_depths: List[int] | None = None
    max_iter: int = 200

    def __post_init__(self):
        if self.lag_minutes is None:
            self.lag_minutes = [1, 5, 60]
        if self.learning_rates is None:
            self.learning_rates = [0.03, 0.05, 0.1]
        if self.max_depths is None:
            self.max_depths = [None, 8]


def time_based_splits(df: pd.DataFrame, cfg: ForecastConfig, time_col: str):
    """Compute time-based train/valid/test masks."""
    max_time = df[time_col].max()
    test_start = max_time - pd.Timedelta(days=cfg.test_days)
    valid_start = test_start - pd.Timedelta(days=cfg.valid_days)

    train_mask = df[time_col] < valid_start
    valid_mask = (df[time_col] >= valid_start) & (df[time_col] < test_start)
    test_mask = df[time_col] >= test_start

    return train_mask, valid_mask, test_mask, valid_start, test_start


def train_global_forecaster(
    df_model: pd.DataFrame,
    cfg: ForecastConfig,
    time_col: str,
    value_col: str,
    feature_cols: List[str],
):
    """Train HistGradientBoostingRegressor with time-based TVT split."""
    train_mask, valid_mask, test_mask, valid_start, test_start = time_based_splits(
        df_model, cfg, time_col
    )
    train = df_model.loc[train_mask].copy()
    valid = df_model.loc[valid_mask].copy()
    test = df_model.loc[test_mask].copy()

    X_train, y_train = train[feature_cols], train[value_col]
    X_valid, y_valid = valid[feature_cols], valid[value_col]
    X_test, y_test = test[feature_cols], test[value_col]

    best_mae = np.inf
    best_model: HistGradientBoostingRegressor | None = None
    best_params: Dict = {}

    # Simple grid search over learning_rate x max_depth
    for lr in cfg.learning_rates:
        for depth in cfg.max_depths:
            model = HistGradientBoostingRegressor(
                learning_rate=lr,
                max_depth=depth,
                max_iter=cfg.max_iter,
                random_state=42,
            )
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_valid)
            mae_val = mean_absolute_error(y_valid, y_val_pred)

            if mae_val < best_mae:
                best_mae = mae_val
                best_model = model
                best_params = {"learning_rate": lr, "max_depth": depth}

    # Final metrics with best model
    forecaster = best_model
    y_pred_train = forecaster.predict(X_train)
    y_pred_valid = forecaster.predict(X_valid)
    y_pred_test = forecaster.predict(X_test)

    def mae_rmse(y_true, y_pred) -> Tuple[float, float]:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return mae, rmse

    metrics = {
        "params": best_params,
        "train": mae_rmse(y_train, y_pred_train),
        "valid": mae_rmse(y_valid, y_pred_valid),
        "test": mae_rmse(y_test, y_pred_test),
    }

    split_info = {
        "valid_start": valid_start.isoformat(),
        "test_start": test_start.isoformat(),
    }

    return forecaster, metrics, split_info
