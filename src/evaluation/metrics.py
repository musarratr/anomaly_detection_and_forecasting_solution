# src/evaluation/metrics.py

from typing import Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae_rmse(y_true, y_pred) -> Tuple[float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, rmse
